# WS2 · Glue-op cut at batch 1 (B, 16 Nov 2026)

Branch `ws2-glue`, gate commit `bb5b1041fd` (= `integration` 7360b71a63 + WS2). Precision `bfp8_release_selected`.
Status: **done — both WS2 proofs met (batch-1 step 11.48 ms ≤ 11.5; 26 programs ≤ 30) and every common gate line passes**.

## Result

| | WS0 (integration) | WS2 | Target |
|---|---|---|---|
| Batch-1 native step, 128 prompt (`decode_timing.json`) | 18.6 ms | **11.48 ms** (87.1 tok/s) | ≤ 11.5 ms |
| Batch-1 native step, 4K prompt | 19.0 ms | 11.96 ms | |
| Batch 32 step | 501.0 ms | 372.3 ms (B8 97.1, B16 190.7) | no regression (≤ 3%): faster |
| Programs per sliding decode layer, WS2 scope (router and experts excluded) | ~99 | **26** | ≤ 30 |
| Programs per sliding decode layer, total (`ops_profile.txt`) | 132 | 59 (26 WS2 + 25 router + 8 experts) | (≤ 50 after WS1) |
| Traced decode layer, sliding / full (2-layer bench) | 604.5 / 643.7 µs | 354.3 / 370.7 µs | |
| quick_check `B1_layers_ms` | 18.74 | 11.81 (B32 371.3) | |

## What changed

Batch-1 decode unless noted; prefill is unchanged. Two changes also reach the batched (8-32 row) decode path: the
2-link all-reduces and the one-program softcap (B8/B16/B32 got 25-26 % faster). The decoder changes have
environment switches (`GEMMA4_26B_WS2_{HEADS,NORMS,GEGLU,CCL_LINKS,CCL_WORKERS,FUSED_CACHE,FOLD_SHARED,
GATHER_SHARDED,PACK_MOE,PACK_ATTN}=0`) that restore the previous path; most have their own commit.

| Change | Programs (sliding) | Sliding layer µs |
|---|---|---|
| Head split, q/k/v RMSNorm and q/k RoPE as one program (gemma4 `QkvHeadsNormRope`); RoPE rows gathered once per step and layer type in `model.py` | 132 → ~85 | 604.5 → 483.9 (per-layer gather still in the bench) |
| Residual stream bf16 and width-sharded over 11x4 cores between norms and between layers; gamma inside the sharded RMSNorm; post-feedforward pair through the final norm's residual input; residual add + layer scalar as one sharded add | → 68 | 470.0 → 430.9 (bench gathering RoPE once) |
| SDPA writes L1; O-projection emits the collective dtype; no BFP8→BF16 typecast after the MoE collective | | 430.9 → 423.3 |
| gemma4 `ConcatHeadsDecode` and `GegluDecode` (shared MLP) | | 423.3 → 420.4 |
| All decode all-reduces on 2 fabric links (was 1), one worker per link | | 420.4 → 386.1 (full 428.8 → 407.9) |
| Expert input written in its decode dtype (BFP8); experts skip the redundant typecast | | |
| One fused KV-cache update (v head on its own core: `ws2_ops.qkv_heads_norm_rope_b1`); shared pre-feedforward gamma folded into the shared MLP decode weights | 33 → 31 in scope | 386.1 → 380.3 |
| Attention all-gather writes the sharded norm layout (sliding); MoE partials in the collective dtype | 31 → 29 | 380.3 → 376.8 |
| **Packed batch-1 all-reduce**: row 0 of the MoE shared+routed partials packed into 8 dense bf16 tiles (was 176 padded tiles), unpacked straight into the norm layout (`tt/kernels/pack_rows.cpp`, `unpack_rows.cpp`); same for the full-layer attention all-reduce (4 tiles) | 29 → 26 | 376.8 → 354.3 (full 403.8 → 370.7) |
| Model: sharded final norm at batch 1; logit softcap as one `unary_chain` (all batch sizes) | | full-model step 11.47 → 11.37 ms |

Other measured and rejected: gemma4 `FusedNormDecode` for the norm family (16-44 µs per program, +15 µs per layer);
lower norm fidelity (6.3 → 6.0 µs per norm); packing the sliding attention all-reduce (no faster, +2 programs);
a batch-1 greedy sampler from `ttnn.max` / `ttnn.argmax` (W-reductions over the 65536-wide vocab shard run
single-core: 260 µs / 43 ms).

## Accuracy

`tests/test_device_numerics.py`: 4 passed (`results/WS2/numerics.txt`). PCC vs HF, at least 0.995 and at most
0.0005 below integration:

| Layer | Prefill PCC (WS0) | Traced decode PCC (WS0) |
|---|---|---|
| 0 sliding | 0.99882 (0.99882) | **0.99984** (0.99964) |
| 5 full | 0.99906 (0.99906) | **0.99991** (0.99988) |

`perf/tools/quick_check.py` against `results/WS0/quick_baseline.json`: PASS; batched vs one-at-a-time decode at 8 users
min per-user PCC 0.99987 / 0.99998, KV caches equal (`results/WS2/quick_check.json`).

Batch-1 decode now keeps the residual in bf16 between attention and MoE (was FP32), feeds bf16 to the QKV matmul
(was FP32) and runs both batch-1 all-reduces in bf16 (were BFP8 for the sliding MoE pair and the full-layer
attention). The decode PCC went up on balance (layer 0 0.99964 → 0.99984). The layer precision summary records these
batch-1 dtypes in an informational `b1_decode` section; the policy fields in `config/precision.json` still govern
prefill and the batched path and are unchanged.

Agreement (`results/WS2/agreement.json`): greedy 128 tokens on `integration` 7360b71a63 as the reference; `ws2-glue`
teacher-forced on those tokens agrees on the top-1 token at **255 / 256 = 99.6 %** (128/128 and 127/128). Free-running
greedy is token-identical on the first prompt and diverges at token 27 of the second (`fibonacci_iterative` vs
`fibonacci`, a near tie).

## Serving and GPQA

`perf/tools/serve_bench.sh qb2port <out> perf,gpqa` (vLLM, local):

| | Result |
|---|---|
| GPQA-10 | **9 / 10** (≥ 8) |
| Serving 128/128, 1 user | 2/2 requests, TTFT 87 ms, decode 86.6 tok/s |
| Serving 128/128, 16 users | 32/32 requests, no failures; decode 5.2 tok/s/user, 71.9 tok/s total |

## Changes in code owned by WS1 (each its own commit)

- `_decode_shared_batch`: each slot's input norm uses `_ws2_input_norm`, so a slot writes the same K/V decoded alone
  or in a batch (quick_check `kv_equal`).
- `OptimizedExperts._chunk`: no activation typecast when the dtype already matches; the mix matmul output dtype is
  `decode_output_dtype` (default bf16, unchanged).
- The batched path passes the plain normalized row to the shared MLP when its gamma is folded into the decode weights.

## Known limits

- Batch 1 only. The batched decode path (8-32 rows) still uses the FP32 residual chain; WS1 replaces it.
- Router (25 programs) and TP experts (8) are WS1's and dominate the remaining program count.
- Outside the layers each step spends ~0.52 ms in the LM head matmul (BFP8, 196 MB per chip, ~375 GB/s) and
  ~0.49 ms in the common sampler (top-k over 32 padded rows x 65536 per chip, 279 µs in `TopkLargeIndices`).
  Both are the largest remaining batch-1 items outside WS4/WS5.
- The two pack/unpack kernels are batch-1 data-movement programs (one valid row); larger batches use the old path.

## Proof files (`perf/results/WS2/`)

`decode_timing.json`, `numerics.txt`, `quick_check.json`, `agreement.json`, `ops_profile.txt`, `gpqa/summary.json`,
`serve_perf_c1.json`, `serve_perf_c16.json`.
