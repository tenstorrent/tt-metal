# GLM-5.3-Flash functional decoder (stage 1, single Blackhole chip)

Design of record: `~/ashai/GLM_5.3_DESIGN.md` (Phase 1 = this stage). Code: `models/demos/glm53_flash/`.
Host: `sjc3-glx-a04u07`, one chip of the 32-chip Blackhole Galaxy (`TT_VISIBLE_DEVICES=0`, `MeshShape(1,1)`),
tt-metal main `8cfcaf754d3` + the commits on `arg/glm53_flash`.

## Contract

`tt/functional_decoder.py::FunctionalDecoder` is one decoder layer of any kind with device-owned state.

| Call | Input | Output | State effect |
|---|---|---|---|
| `from_state_dict(state_dict, hf_config=cfg_or_config_json, layer_idx, mesh_device, max_batch, max_seq_len)` | raw checkpoint tensors of `model.language_model.layers.{i}.` (prefix stripped, fp8 groups dequantized by `reference/checkpoint.load_layer_raw`) | decoder | allocates the KDA user slots / the DSA paged latent + pooled-key index caches |
| `prefill_forward(streams, user, seq_len)` | packed mHC streams `[1,1,T_pad,4*4096]` fp32, `T_pad % 32 == 0`, rows past `seq_len` padding | `[1,1,T_pad,4*4096]` fp32 | KDA: writes slot `user`; DSA: fills the user's cache pages `[0, T_pad)`, pooled keys, tail ring |
| `decode_forward(streams, positions)` | `[1,1,B,4*4096]` fp32 (B == max_batch user slots); `positions` = `DecodePositions` from `make_decode_positions(host_positions)` / `update_decode_positions` (persistent `[B]` int32: position, position // 4, position % 4, + host values) | `[1,1,B,4*4096]` | KDA state updated in place; DSA latent appended and the pooled-key index maintained; dense flash-MLA for every slot (HF-exact through position 2050 = index_topk + index_kpool - 2), then the rows of the users whose host position is > 2050 are overwritten by the per-user indexer + `sparse_sdpa` result (eager, host chunk geometry) |
| `capture_decode_trace(streams, positions)` / `replay_decode(trace_id)` | persistent device inputs | `(trace_id, output tensor, warm-up output)` | the dense decode graph is trace-replayable (state and pooled-key index updated inside the captured graph, HF-exact through position 2050); refused for DSA decoders with max_seq_len > 2051 unless `allow_sparse_range` (the sparse rows are never captured) |

Residual convention: HF `[B, S, 4, 4096]` <-> packed `[1, 1, B*S, 16384]` (`host_streams_to_device` /
`device_streams_to_host`, test boundary only).

## Modules

| File | Content | Reuses |
|---|---|---|
| `tt/config.py` | frozen config from `config.json` | |
| `tt/mhc.py` | mHC hyper-connection (fused Sinkhorn parametrization, fp32) | `deepseek_v3_d_p/tt/mhc/TtMHCWrap` |
| `tt/kda.py` | KDA: prefill via `ttKDA` (chunked kernels), decode = composite recurrence on `[B*H,128,128]` fp32 with in-place state writeback | `deepseek_v3_d_p/tt/kda` |
| `tt/dsa.py` | MLA (NoPE): dense causal SDPA for prefill rows [0, 2048), paged latent cache, absorbed `paged_flash_multi_latent_attention_decode` for every decode slot; prefill rows >= 2048 and decode users at position > 2050 via the indexer + `sparse_sdpa` | paged cache ops, flash MLA decode, sparse SDPA |
| `tt/indexer.py` | GLM k-pool indexer: pooled-key cache, `indexer_score_dsa(key_compression_ratio=4)`, `topk_large_indices` over 4x-expanded pool scores, tail ids | |
| `tt/moe.py` | router (exact fp32 composite top-8) + `sparse_matmul` experts per 32-token block + shared expert | gpt_oss sparse-matmul pattern |
| `tt/mlp.py` | clamped SwiGLU (dense layers, shared expert) | |
| `tt/decoder_layer.py` | mHC(attention) -> mHC(FFN) composition | |
| `reference/` | checkpoint access + fp8 dequant, HF layer reference runner (dumps), KDA reference equivalence check | transformers 5.20 `glm5_next` |

## Evidence (real weights, PCC vs HF)

| Test | What | Result |
|---|---|---|
| `test_kda.py` | layer 0 KDA, T=96 / 1000 + 3 decode | prefill 0.99994 / 0.99995, state 0.9999, decode 0.9999 |
| `test_moe.py` | layer 4 MoE, T=32 / 256 | router sets identical 32/32, 255/256; output 0.99993 / 0.99995 |
| `test_dsa.py` | layer 3 MLA, T=1, 2, 3, 31, 33, 96, 1000 + 3 decode | prefill 0.99989 (T=1) .. 0.99961 (T=1000), latent cache 0.99997, decode 0.9991-0.9999 |
| `test_decoder_layer.py` | full layers 0 / 3 / 4 vs HF dump, T=256 + 4 decode; T=200 non-aligned live HF | 0.99984 / 0.99992 / 0.99997 prefill, 0.9999 decode; 0.99980 / 0.99993 |
| `test_decode_trace.py` | traced decode replay vs the HF dump (layers 0/3/4); layer 3: the pooled-key index written inside the captured graph (65 complete pools + the 4 ring rows) vs a host recomputation over the 256 prefill + 4 replayed tokens; two users (100 + 37 tokens, slots) | replay PCC 0.9999 (same as the eager run of the same steps); index after the replays 0.999997 (worst pool row 0.99999), ring rows k 0.999994 / gate 0.999995; per-user 0.9998-0.99998 |
| `test_dsa_sparse.py` | layer 3 at 4097 tokens (repeated prompt): dense rows [0,2048) + sparse rows, decode 4097..4100 | prefill 0.99992 (rows<2048 0.99994, rows>=2048 0.9999), decode 0.9999 |
| `test_dsa_long.py` | random-token prompt: 4097 + 12 sparse decode steps with TT/HF selected-id sets (contiguous-prefix contract asserted) and the pooled-key cache vs a host recomputation; 2040-token dense prefill (slot 1, permuted page table) then decode 2040..2063 with the short user 0 checked every step; 16385-token prefill + 2 decode; prompts 2042/2043 (tails 2, 3) + 12 decode steps with the pool-cache check | prefill 0.99994 (rows<2048 0.99988, >=2048 0.99986), decode 0.9997+, 98-99 % ids common, pool cache 0.99999; transition >= 0.9996 both users; 16K 0.99983 (0.99988 / 0.99980); tails 2/3: prefill 0.99987 / 0.99990, decode >= 0.99978, pool cache 0.99999 |
| `test_batch8.py` | B = 8 and B = 32 user slots (prompt lengths 1..200 mixed, permuted page table), layers 3 and 4, 3 / 2 batched decode steps; every per-user sample held to 0.995, assertion deferred to the end (`_logs/fix3_batch8_B8.log`, `fix3_batch8_B32-3.log`, `fix3_batch8_B32-4.log`) | B=8: prefill >= 0.9968 (1-token user), decode worst per user 0.999377 (layer 3) / 0.999363 (layer 4). B=32 (flash-MLA decode on 120 cores = 3 cores/user): layer 4 PASSED, prefill worst 0.999937, decode worst per user 0.999698 / 0.999944 (steps 0 / 1); layer 3: all 64 decode samples > 0.995 (worst 0.997380 / 0.999489), but the case FAILS the per-user prefill gate on user 18 (S=5, offset 1620): 0.994797, one of its five rows at 0.98025 (see Known limits) |
| `test_determinism.py` | layers 0, 3: repeated prefill and traced-decode replay with `throw_exception_on_fallback`; sparse prefill + sparse decode repeated | bit-identical outputs |
| watcher | `TT_METAL_WATCHER=10` on the layer-3 dump test (`~/ashai/glm53/_logs/watcher_layer3`) | PASSED, 10 dumps, no watcher exceptions |
| `test_perf_layer.py` | warmed prefill S=1024 (wall), traced decode B=1 (one replay, tt-perf-report `Device Time`) | prefill: layer 3 325.7 ms wall (re-measured 2026-10-20 at 5c5122ccf3a + the review-3 fixes, `GLM53_PERF_PHASE=prefill`, no tracy, `_logs/fix3_perf_layer3_prefill.log`; host load average 60-110 on 64 cores from other sessions' CPU jobs during the run; the earlier 314 ms predates eca757b2ffc and 4cd5f8fe829, which added the prefill index maintenance), layer 4 297.8 ms wall from the tt-perf-report run that predates the cached-scalars fix 4cd5f8fe829 (its device-time table, 293.4 ms over 1002 ops, is unaffected: the fix removed host-side uploads, not device ops); decode: layer 3 10.87 ms/step device (435 ops, wall 11.31), layer 4 10.76 ms/step device (471 ops, wall 11.2) |

Context contract: `doc/context_contract.json`. Full log: `doc/functional_decoder/work_log.md`.
Profiler note: Tracy device zones of the SDPA prefill kernels (dense and sparse share `compute_streaming.hpp`: `:632` vs
`:1769`, hash 0x6835) collide on main 8cfcaf754d3, so prefill tables exist for layer 4 only; decode tables for both layers
come from one-replay runs (`GLM53_PERF_DECODE_STEPS=1`, layer 3 with `GLM53_PERF_SEED_ONLY=1`), tables under `doc/functional_decoder/tracy/`.

## Running

```bash
source ~/ashai/glm53/env.sh
TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests -s           # all device tests (one chip)
python models/demos/glm53_flash/reference/hf_layer_reference.py --layers 0,1,2,3,4 --seq-len 256 --num-decode 4 \
    --out bringup/references/glm53_flash/T256                            # regenerate the HF dumps (CPU)
```

## Known limits (single chip, v1)

- Prefill is single-shot (one chunk from position 0 per user): the design's chunked prefill (``actual_start``
  carries exist in ttKDA and the layer plumbs them, the DSA cache is filled from 0) is not wired through the public
  API yet.
- Single-chip capacity (31.83 GiB allocatable DRAM = 8 banks x 3.979 GiB; 7.35 GiB resident after the layer-3 build,
  `_logs/fix3_perf_layer3_prefill.log`), two different answers:
  (i) decode is not capacity-bound at the advertised 1 M positions for one user: 11 DSA layers x 1,048,576 x 512 x 2 B
  = 11 GiB of latent pages + pooled keys (one [128] bf16 row per 4 tokens = 64 B/token: 0.69 GiB) + 2 x 32 x 128 x 2 B
  ring rows per user and layer; the single-chip decode bound is validation time of the v1 composite paths.
  (ii) single-shot prefill IS capacity-bound: for a prompt of T tokens (S = T - 2048 sparse query rows,
  W = 32 * ceil(max_seq_len / 128) pooled-key columns, about T/4) the peak is inside `TtDSAIndexer.select` at the 4x
  `repeat_interleave` of the scores (ttnn implements it along the last dim as transpose, 4x concat, transpose back):
  the TILE scores `[S, W]` bf16, their ROW_MAJOR copy, the transposed copy, the repeated `[4W, S]` and the result
  `[S, 4W]` are live together = 22 * S * W bytes (about 5.5 * T^2), plus the tensors held across the call (fp32 streams
  64 KiB/row, pre-norm + normalized input 16 KiB/row, q 32 KiB/row, q_resid 3 KiB/row, the sparse-row slices and
  indexer queries 19 KiB/row, the dense-row output 64 MiB, the caches 1 KiB + 64 B per token). The next-highest point,
  `sparse_sdpa` (q_rm + o_rm `[64, S, 512]` bf16 = 128 KiB/row, kv_rm, indices `[S, 2176]` u32), is linear in T.
  Evaluated: T = 16385 -> 1.21 GiB quadratic, 3.35 GiB peak (the validated point); 32768 -> 9.4 GiB; 49152 ->
  18.3 GiB; the largest single-shot prompt that fits the 24.48 GiB free next to the layer-3 weights is about
  57 K tokens (58368, peak 24.4 GiB; computed, not validated). Chunked prefill (in progress on another
  branch) scores `[T_chunk, W]` per chunk and removes this bound.
- Runtime audit method: `grep -nE 'from_torch|to_torch|torch\.'` over `tt/*.py` forward paths (hits are
  construction-time or first-use-cached constants only) plus `throw_exception_on_fallback=True` runs
  (`test_determinism.py`); the ring tail rows and the KDA scalars use tensors created once per geometry/value.

- Performance is the design's Phase 2/3 work: the KDA decode recurrence and the per-block MoE are composites,
  the sparse decode untilizes the user's latent prefix per step, and the decode SDPA runs on 4 cores per user.
- The sparse DSA decode path takes host positions (chunk geometry) and is therefore not captured in the trace;
  the pooled-key index is nevertheless maintained inside the traced dense graph (checked against a host
  recomputation in `test_decode_trace.py`), so a user can cross 2050 and continue on the eager sparse rows.
- Indexer scores are ranked at bf16 resolution (`indexer_score_dsa` + `topk_large_indices`): 98-99 % of the
  HF-selected ids per step on a random prompt, layer PCC >= 0.9997.
- `moe_grouped_topk` is not used (bf16 ranking swaps near-tied experts); the exact fp32 composite costs ~8x5 ops.
- Per-row PCC tail of the MoE layers = near-tie routed-expert swaps: when the 8th and 9th biased router scores are
  within ~3e-3 (18 % of tokens are < 1e-3 apart, 39 % < 3e-3), the exact fp32 router on the device's bf16
  post-attention stream can pick HF's 9th expert; such rows score 0.98-0.999 (layer 3, 32 prompts of the B=32 case:
  21/2542 rows < 0.997, min 0.98025; every low row checked in `_logs/fix3_router_flip_user18.log` is such a swap,
  rows with identical expert sets are at 0.9999). A 1..5-row prompt is a handful of draws of this distribution,
  which is why `test_batch8.py` B=32 layer 3 fails the 0.995 per-user gate on its S=5 user 18 (0.994797) while all
  64 decode samples pass; the bar is kept and the case is red on this draw. Phase 2/3 option: fp32 router input.
