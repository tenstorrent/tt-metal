# DeepSeek-V4.1-Flash: ttnn prefill for disaggregated serving

This package is the prefill half of DeepSeek-V4.1-Flash disaggregated serving on Tenstorrent Blackhole. The decode half is
the tt-blaze 48-stage decode ring. DeepSeek-V4.1-Flash-1017 (new weights, same architecture) runs the same code with its own
weight cache. Every measurement and defect behind this code is in the tt-blaze V4.1 postmortem
(`docs/deepseek_v4_1_flash/postmortem.md`), cited by entry id (`DS41F-00NN`). It builds on the DeepSeek-V4-Flash prefill
(`tt/v4/`, whose README is on tt-metal `sdawle/dsv4-flash-prefill-ttnn`).

## What it is

- **Where:** one Blackhole galaxy, SP 8 x TP 4 (host 41 on the DC16-2 pod), one runner rank.
- **What it computes (default):** the encoder, layers 0-19 with Engram before layers 1 and 14, in full, and layer 20 KV-only.
  The ring replays the prompt's last R >= 2541 tokens itself, which also yields the first token, so there is no LM head here
  (DS41F-0037). `V41_PREFILL_DECODER=1` builds all 40 layers and exports every layer's window; the ring then replays fewer
  than 128 tokens (DS41F-0073).
- **Host work:** the embedding lookup and the Engram hash / table gather run on the host, through tt-blaze's ttnn-free
  `EngramHost` (`blaze.models.deepseek_v4_1_flash_4_GLX.engram_host`). The device runs Engram's `wkv` and gate (`engram.py`).
- **Chunks:** eager by default. `PREFILL_CHUNK_SIZE` must be a multiple of 512 (64 tokens per chip x SP 8,
  `V41Prefill.CHUNK_ALIGN`), and `PREFILL_MAX_SEQ_LEN` a multiple of the chunk.
- **Trace islands:** `V41_PREFILL_ISLANDS=1` captures each block's mHC / MoE islands once the runner's service buffers exist
  (`on_services_ready`). `PREFILL_V41_PATH_A=1` moves the CSA sources to the V4 sparse path A, and `PREFILL_V41_A3=1` runs
  their attention on islands too (DS41F-0067, DS41F-0068).
- **KV export:** after every chunk, `kv_export.py` writes the new rows into export caches in the ring's contract layout
  (3 configs: `swa_window`, `csa_unified`, `index_k`; 256-row window rings) and builds the `KvChunkAddressTable`. Consumer
  layers hold no rows: the table points them at their source's rows, and the decoder layers at layer 20's. The common
  engine's migration driver moves the rows to the ring, as for V4.
- **Vision:** `vision.py` is the DeepSeek-ViT + aligner in ttnn. `V41_PREFILL_VIT=1` builds it inside the runner,
  replicated on the runner's mesh, so a request can name an image (DS41F-0045, DS41F-0049).

## Code map (files this branch adds or changes on top of the V4 prefill)

Paths are under `models/demos/deepseek_v3_d_p/` unless they start with `common/` (= `models/demos/common/prefill/`).

| File | Role |
|---|---|
| `tt/runners/adapters/deepseek_v4_1_flash.py`, `common/adapter.py` | the adapter for `PREFILL_MODEL=deepseek_v4_1_flash` (config, weight cache, export allocation, runtime) and its registration |
| `tt/v41/runtime.py` | `V41PrefillRuntime`: compile, `prefill_chunk` (chunk, export, acks), islands at `on_services_ready`, warm-all, ViT, sidecars |
| `tt/v41/prefill.py` | `V41Prefill`: the chunk loop over the blocks, embedding, Engram, image-span rows, `cache_snapshot` |
| `tt/v41/block.py` | `V41PrefillBlock`: single-pass mHC (`V41HyperConnection`), attention, MoE, and their islands |
| `tt/v41/attention.py` | V4.1 attention on the V4 modules: `V41SWA`, `V41CSA` (sources), `V41CSAConsumer`, `V41CSAReindex`; path A / A3; fp4 index quant |
| `tt/v41/moe.py`, `tt/v41/engram.py` | the V4.1 MoE on `TtMoe` (384 experts, lazy FP4 dequant); the device half of Engram |
| `tt/v41/config.py`, `tt/v41/weights.py` | `V41Config` from the checkpoint's `inference/config.json` and the per-layer CSA roles; the checkpoint reader (no ttnn) |
| `tt/v41/kv_export.py` | export caches in the ring's contract layout, the per-chunk export, the KV chunk table |
| `tt/v41/vision.py`, `scripts/v41_vit_rows.py` | ViT + aligner in ttnn; a one-chip front end that writes an image's aligner rows |
| `common/runners/producer_manifests/producer_manifest_deepseek_v4_1_flash_e2e.yaml`, `..._e2e_relay.yaml` | driver manifests: done / handoff on NFS, or host-local and relayed by the launcher |
| `tt/v4/attention/csa.py`, `tt/mla/heavily_compressed_attention.py` | V4.1 changes to shared V4 files: rate-generic cuts (ratio 2), column-block indexer scorer (DS41F-0066), mask concat (DS41F-0064) |
| `tests/pcc/test_v41_*.py` | device tests, below |

The rest of the diff against main (`tt/v4/`, the V4 adapter and manifests, the common runner, `tt/moe/`, the C++ changes) is
the V4 branch's, described in its `tt/v4/README.md`. `git diff b1860c1403a HEAD` (the V4 merge commit) lists only the files
in this table.

## How to run

The measured runs use the tt-blaze launchers. The recipe is tt-blaze branch `sdawle/dsv4_1_flash_blaze`,
`tests/blaze/models/deepseek_v4_1_flash/launchers/disagg/README.md` (`e2e_v41.sh`). The 1017 branch `sdawle/dsv41_1017`
carries the newer pod driver (`launchers/disagg/e2e_v41.sh`, `disagg_sweep.sh`) and the 1017 run order
(`launchers/pod_1017/README.md`).

1. **Runner** (prefill galaxy): `python -m models.demos.common.prefill.runners.prefill_runner` with
   `PREFILL_MODEL=deepseek_v4_1_flash`, `PREFILL_ENABLE_MIGRATION=1` and the knobs below. `PYTHONPATH` must also hold a
   ttnn-free `blaze` package for `EngramHost`: the real `blaze/__init__.py` imports ttnn. The pod uses a stub, a `blaze/`
   with an import-free `__init__.py` and `blaze/models` linked to a tt-blaze tree's.
2. **Migration driver:** `python -m models.demos.common.prefill.runners.migration_driver --manifest
   models/demos/common/prefill/runners/producer_manifests/producer_manifest_deepseek_v4_1_flash_e2e_relay.yaml
   --verify-migration off`. The trace dir holds `ids[:S']`; the ring gets the full prompt for its replay.
3. **1017:** source tt-blaze `tests/blaze/models/deepseek_v4_1_flash/launchers/v41_1017_env.sh` (branch `sdawle/dsv41_1017`).
   It sets `PREFILL_HF_MODEL`, `PREFILL_TTNN_CACHE`, `DEEPSEEK_V4_V41_HF_MODEL` and `DSV41_ENGRAM_TABLE_DIR` to 1017's. The
   tt-metal 1017 branch adds only a `.tt_model_id` guard in the adapter's `weight_cache_path`: a cache built from the other
   checkpoint is refused (DS41F-0077).

**Knobs a user sets.** Defaults are the code's.

| Knob | Default | Effect |
|---|---|---|
| `PREFILL_CHUNK_SIZE`, `PREFILL_MAX_SEQ_LEN` | 5120, 11 x chunk | chunk (a multiple of 512) and per-slot capacity (a multiple of the chunk; `e2e_v41.sh` refuses otherwise) |
| `PREFILL_NUM_LAYERS` | 40 | the contract's layer count: the driver migrates layers [0, 40) and drains 40 acks per chunk |
| `PREFILL_NUM_USERS`, `PREFILL_V41_ALIAS_SLOTS` | 2, 0 | slots; `ALIAS_SLOTS=1` keeps one physical slot behind a table of `PREFILL_NUM_USERS` slots (one prefill fanned out to every ring slot) |
| `V41_PREFILL_ISLANDS` | 0 | 1 = mHC / MoE trace islands, captured at `on_services_ready` plus warm replays (implies warm-all) |
| `V41_PREFILL_WARM_ALL` | 0 | 1 = dummy chunks at every position up to capacity at compile, so no live-width program compiles inside a request |
| `PREFILL_V41_PATH_A` | 0 | 1 = CSA sources on the sparse path A once the context holds a window + top-k of entries; consumers use the source's top-k (DS41F-0067) |
| `PREFILL_V41_A3` | 0 | 1 = the CSA attention of every path-A chunk on two islands per layer and slot (DS41F-0068). Needs path A and islands |
| `PREFILL_V41_A3_RAGGED` | 1 | the ragged final chunk on the attention islands; 0 = eager |
| `PREFILL_V41_A3_CHUNK0` | 0 | chunk 0 on its own A2 island. Do not set: it hangs the all-40-layer prefill and corrupts the migrated KV (DS41F-0076, open) |
| `PREFILL_V41_A3_SWA` | 0 | layers 0-1 on islands; no gain (DS41F-0068) |
| `V41_PREFILL_DECODER` | 0 | 1 = all 40 layers in the prefill and every window exported (DS41F-0073). `V41_PREFILL_DECODER_CACHE` = their weight cache (unset: converted every start); `V41_PREFILL_DECODER_EXPERT_DTYPE` = `bfp4` or `bfp8` |
| `V41_PREFILL_EXPERT_DTYPE` | `bfp8` | `bfp4` for the routed experts |
| `PREFILL_CSA_CAP_BUCKETS`, `PREFILL_CSA_BUCKET_STEP` | 1, 32768 | shared V4 knob: scorer-width buckets for the CSA sources' A2 islands |
| `PREFILL_INDEXER_SCORE_COLS`, `PREFILL_INDEXER_SCORE_BLOCK_ABOVE` | 65536, 400000 | the indexer scorer runs in column blocks only above 400,000 live keys (DS41F-0066) |
| `PREFILL_MIGRATION_ROW_BASE`, `PREFILL_MIGRATION_ROW_DIVISOR` | unset | driver: rows = base + ceil(S / div); V4.1 uses 256 / 1 |
| `PREFILL_MIGRATION_STREAM` | 0 | driver: migrate finished rows while the prefill runs (DS41F-0071). `PREFILL_MIGRATION_STREAM_SPLIT` (20), `PREFILL_MIGRATION_STREAM_DIV` (2), `PREFILL_MIGRATION_STREAM_CHUNKS` (4) follow V4.1's layer roles |
| `PREFILL_HF_MODEL`, `PREFILL_TTNN_CACHE` | pod NFS paths | checkpoint; weight cache at `<cache>/<sp>x<tp>` |
| `DEEPSEEK_V4_V41_HF_MODEL`, `DSV41_ENGRAM_TABLE_DIR` | pod NFS path, unset | `config.py`'s default checkpoint (tests, `v41_vit_rows.py`; the runner uses `PREFILL_HF_MODEL`); the Engram tables for `EngramHost` |
| `V41_PREFILL_VIT`, `PREFILL_REQUEST_SIDECAR_DIR`, `V41_PREFILL_IMAGE_ROWS` | 0, unset, unset | ViT in the runner; per-request sidecars (`slot_<s>.json`, written by the producer) naming an image or rows; launch-wide image rows |

- Others: `V41_PREFILL_GATE_BIAS_CENTER` (`mid`, DS41F-0040), `V41_PREFILL_VL_BIAS` (1), `V41_MHC_STACKED` (1),
  `V41_MHC_HOST_SINKHORN` (0), `V41_PREFILL_INDEX_FP4` (0, DS41F-0048), `V41_PREFILL_EMBED_PARTITION` (1),
  `V41_PREFILL_ENGRAM_EARLY` (1), `PREFILL_MASK_CONCAT` (1). Measurement only: `V41_PREFILL_PROFILE`.
- `PREFILL_USE_TRACE` only reserves a trace region here. The V4.1 runtime reports `use_trace=False`, and its islands record
  with the runner's default trace region of 0, i.e. tt-metal's dynamic DRAM trace allocation (`mesh_trace.cpp`).

## Tests

All need a device; all but `test_v41_fp4_quant.py` read the real checkpoint. Run them as `PYTHONPATH=<this tree>:<ttnn-free
blaze stub> pytest ...` (several import tt-blaze code). The goldens are tt-blaze per-layer prefill traces
(`golden/prefill_trace.py`) at `V41_PREFILL_TRACE`. Mesh 8x4 unless noted.

| Test | Checks |
|---|---|
| `test_v41_prefill.py` | the encoder (+ layer 20 KV-only) vs `model.py`: every block's streams, the next `pre`, the end-of-prompt cache snapshot |
| `test_v41_attention.py`, `test_v41_attention_chunked.py` | V4.1 attention vs `model.py` on real inputs; chunked CSA (layers 2 and 3) on path B or path A |
| `test_v41_path_a.py` | path A vs path B chunk by chunk on the same weights and inputs |
| `test_v41_kv_export.py` | the runtime over a real prompt, then every chunk the ring asks for read back over UMD through the table vs `model.py`'s state |
| `test_v41_fp4_quant.py` | the device fp4 fake quant of the indexer q / k vs `model.py`'s formula (1x1) |
| `test_v41_vision.py`, `test_v41_vision_submesh.py` | ViT + aligner on one chip vs fp32 `vision.py`; the runner's in-process ViT (8x4) vs the one-chip front end |

## Performance

Not copied here. The measured TTFT tables, disaggregated vs the ring's own prefill-by-decode per ISL, with the path, chunk
and replay length of each row, are in tt-blaze branch `sdawle/dsv4_1_flash_blaze`, `docs/deepseek_v4_1_flash/README.md`
("Combined ISL x BS table" and "TTFT per ISL"). Each number cites its postmortem entry. The 1017 disagg check is in
DS41F-0077.

## Branches

| Model | tt-metal (this prefill) | tt-blaze (decode ring, launchers) |
|---|---|---|
| DeepSeek-V4.1-Flash | `sdawle/dsv41-prefill-ttnn` | `sdawle/dsv4_1_flash_blaze` |
| DeepSeek-V4.1-Flash-1017 | `sdawle/dsv41p_1017` (= V4.1 + the cache guard) | `sdawle/dsv41_1017` |

- Main merge: tt-metal main `20778bdbcda` (tt-blaze main's tt-metal pin), merge commit `35ba4e1de42`. The decode package
  import followed tt-blaze's rename to `blaze.models.deepseek_v4_1_flash_4_GLX` in `537783ee1e1` (DS41F-0078).
- Device validation of the merged trees is in DS41F-0078 / DS41F-0079. As of those entries the merged prefill had not run
  on a device; the tt-blaze tables are pre-merge.
