# Qwen3-VL e2e: Quasar gaps and other failures

Running log of everything the Qwen3-VL e2e bring-up hit. Section 1 is the main output: ttnn ops/paths that work on
WH/BH but not on Quasar. Every workaround the harness or model copy uses to get past a gap is listed with it, so no
gap is hidden by a workaround.

Status values: `open` (not fixed in ttnn), `worked-around` (harness/model avoids it; still open in ttnn),
`fixed` (fix landed; workaround removed), `unverified` (suspected from precedent, not yet reproduced on Quasar).

## 1. Ops supported on WH/BH but not on Quasar

| # | Op / path | Quasar failure | How found | Repro | Workaround in this branch | Status |
|---|---|---|---|---|---|---|
| Q1 | `ttnn.scatter` (dim 0, INT32 index, bf16 src) — used by `merge_vision_tokens_single_user_ttnn` to place image embeddings into the text embeddings | `TT_FATAL @ tt_metal/impl/metal2_host_api/program_spec.cpp:2024: !(is_gen2_arch(hal) && self_loop_kernel->is_data_movement_kernel())` — scatter's program uses a DM self-loop DFB, disallowed on gen2 | craq-sim 8x4, 2026-10-05 | `TT_METAL_SIMULATOR=/localdev/$USER/sim/libttsim.so pytest models/experimental/ops/quasar/tests/qwen3_vl_ops/test_scatter.py` (case `00_2766x2560_bf16_int-dram`) | `QuasarModelArgs.device_scatter = False` → merge copies rows on host (`tt/common.py::_merge_on_host`); pure data movement, values unchanged | worked-around |
| Q2 | Matmul / `ttnn.linear` with `fp32_dest_acc_en=True` | Not yet reproduced on Quasar. The llama Quasar e2e forces `fp32_dest_acc_en=False` on every compute config; on ttsim WH the same configs are undefined (see S1) | precedent (llama32_1b_quasar `test_llama_e2e.py:248-270`) | — | Quasar config uses `HIFI4_FP16` and strips fp32 dest acc from every `compute_kernel_config_*` (`tt/quasar_config.py::strip_fp32_dest_acc`) | unverified |
| Q3 | `ttnn.tilize` of FLOAT32 (also hit by `ttnn.from_torch`/`as_tensor` fp32 → bf16 TILE, which tilizes in fp32 before the typecast) | Quasar's `llk_unpack_tilize_*` ignore `UnpackToDestEn` and unpack Float32→Float32 to SrcA: `UNPACKER_0 (ILLEGAL_FORMAT_CONVERSION)` on emu-quasar-2x3, hang on craq-sim | tracked: [#57780](https://github.com/tenstorrent/tt-metal/issues/57780) | #57780 repro (fp32 `ttnn.tilize`) | harness workaround `host_cast_fp32_upload` casts the torch source to bf16 on host. Candidate fix: interim f7048f62da7 (branch `gchoudhary/quasar-tilize-default-factory-unpacker-illegal-format-conversion`, not in HEAD) + DEST-bank sync stack ending in #59290 (open); per #57780 comments, tilize still needs its own UNPACK handshake and the single-core factory still creates legacy DM kernels | worked-around (open in ttnn, #57780) |
| Q4 | DRAM-sharded weights + `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (captured layout for decode/lm_head linears) | not attempted: the 2-core emulator grid has fewer cores than DRAM banks on WH (12), and Quasar DRAM-sharded matmul is pending [#58912](https://github.com/tenstorrent/tt-metal/pull/58912) | design decision | — | Quasar config keeps weights DRAM interleaved and lets `ttnn.linear` pick the program (`create_dram_sharded_mem_config` / `dram_matmul_config` overrides) | worked-around (revisit when #58912 lands) |

## 2. Simulator (ttsim WH/BH) issues — not Quasar gaps, but they block the WH/BH baseline

| # | Issue | Error | Repro | Workaround | Status |
|---|---|---|---|---|---|
| S1 | `ttnn.linear` with `fp32_dest_acc_en=True` (any `packer_l1_acc`) on ttsim WH | `UndefinedBehavior: tensix_unpacr: unpack_to_dst=0 in_data_format=0 out_data_format=0` (fp32 unpacked to srcA) | `.superpowers/sdd/.../repro_linear.py 1 0` on ttsim WH | Quasar config disables fp32 dest acc (Q2) | open (ttsim strictness or real WH UB in matmul fp32 reload — needs a WH owner's call) |
| S2 | `ttnn.from_torch(fp32 → bf16, TILE, mesh_mapper=Replicate)` on ttsim WH | same `UndefinedBehavior` as S1: ttsim WH sees the fp32 tilize unpack to SrcA (`unpack_to_dst=0`), although #57780 says WH/BH route this case to DEST — worth a WH owner checking which tilize factory `from_torch` picks here | `repro_astensor.py mapper` | `host_cast_fp32_upload` | open (likely same path as Q3/#57780) |
| S3 | `ttnn.scatter` (INT32 index) on ttsim WH | `UndefinedBehavior: tensix_unpacr: unpack_to_dst=0 in_data_format=8` | `qwen3_vl_ops/test_scatter.py` on ttsim WH | Q1's host merge (Quasar config only) | open |

## 3. Bugs found in the Quasar model copy (fixed here)

| # | Bug | Symptom | Fix |
|---|---|---|---|
| M1 | `tt/vision_mlp.py` cached weight and bias under the same file name; `as_tensor` keys only on name+dtype+layout, so with bf16 weights the bias load returned the weight | `binary_ng ... Invalid subtile broadcast type` in fc1 (bias was `[1024, 4096]`) | `.weight` / `.bias` cache suffixes (as `patch_merger.py` already does) |
| M2 | Vision patches were always padded to the next multiple of 2048, plus a whole extra 2048 when already aligned (e.g. 2048 → 4096). Only partly over-padding: up to 1024 patches a multiple of 128 suffices, but above 1024 the MLP/WO reshape into 1024-row chunks and above 2048 the QKV matmul into 2048-row chunks, so 2048-multiples are required there | 256 tiny patches padded to 2048 (8x vision work) | `vision_padded_seq_len`: multiple of 128 up to 1024, of 2048 above (demo 11008 → 12288 unchanged) |
| M3 | `DropInVisionTransformer.forward[_single_user]` slices `[:, 0:1, :, :out_hidden]` + reshapes the merger/deepstack outputs, then deallocates the sources. On a 1-device mesh the full-range slice and reshape are views of the same buffer (verified: identical `buffer_address`), so the returned tensors read freed memory (same family as the existing `_mesh_partition_and_free` fix) | deepstack embeddings garbage (PCC 0.27, 10x too small) once the next allocation reused the buffer → `text.layer1` PCC 0.83; image embeddings survived only by read order | `_free_unless_aliased`: free the source only if the view does not share its buffer |
| M4 | `models/tt_transformers` `ModelArgs.find_prefill_grid` returns `(rows, cols)` but every consumer (`matmul_config`, `MinimalMatmulConfig`, `VISION_WO_PREFILL_PROGCFG`) reads it as `(x, y)`; invisible on the square 8x8 grid (shared code, not fixed there) | on a 2x1 grid: `compute_with_storage_grid_size (1, 2) must fit within device grid (2, 1)` in vision WO | `_QuasarArgsMixin.find_prefill_grid` returns `(x, y)` |

## 4. Hardware vs simulator differences

| # | Observation | Evidence | Next step | Status |
|---|---|---|---|---|
| H1 | Same Quasar config (bf16, `HIFI4_FP16`, fp32 dest acc stripped), tiny, V=2 T=2 K=1, full grid: ttsim WH passes every stage (>= 0.9993); real WH fails from `vision.block0` (0.971) to `text.logits.prefill` (0.175) | user run `wh_bh_hw_native/20261005T213004Z` vs `dev/eleventh` | on WH hardware: `run_wh_bh.sh --grid native --debug fast --fp32-dest-acc -- --qwen-dump-stages` (fp32 dest acc kept) and `run_wh_bh.sh --grid native --debug fast --config native` (bf8 control) | open |
