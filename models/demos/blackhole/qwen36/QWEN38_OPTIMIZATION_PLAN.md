# Porting qwen38_27b_qb2 optimizations into qwen36 — tiered plan

Branch: `atupe/qwen38-optimizations` (local, uncommitted). Source of the optimizations: `models/demos/qwen38_27b_qb2` ("B"). Target: `models/demos/blackhole/qwen36` ("A"), Qwen3.6/3.8-27B, TP=4 on QB2 (4× Blackhole, P300x2, mesh 1×4).
Qwen3.6-27B and Qwen3.8-27B are architecturally identical (configs differ only in `transformers_version`), so all measurements below use the local Qwen3.6-27B weights for both implementations.

## Baselines (this machine, ISL 128, batch 1)
| | decode t/s/u | ms/token | TTFT | accuracy_512 top-1 / top-5 |
|---|---|---|---|---|
| A (qwen36, main) | 27.7 | 36.1 (35.6 device + 0.5 host) | 120 ms | 98.44 / 100.00 |
| B (qwen38_27b_qb2 demo) | 36.4 | 27.4 | 54 ms | 97.66 / 99.80 (measured with a teacher-forcing harness on the same reference) |
B README quotes 39.4 t/s/u / 67.9 ms TTFT via vLLM (different harness, Qwen3.8 weights); B's earlier PR #57893 quoted 35.8 t/s/u via vLLM.

## Accuracy gate used for every change
`pytest models/demos/blackhole/qwen36/demo/text_demo.py -k accuracy_512`: teacher-forced, 512 positions after a 512-token prefill of *A Tale of Two Cities*, compared with a CPU HuggingFace reference (`models/tt_transformers/tests/reference_outputs/Qwen3.6-27B.refpt`). top-1 = TT argmax equals reference top-1; top-5 = TT argmax within reference top-5. One token = 0.195 pt. Targets (`models/model_targets.yaml`): top-1 ≥ 97, top-5 ≥ 99 (the pytest asserts 0.5 below that). Plus: generated text for the perf prompt must stay coherent and answer the prompt.

## Tier 1 — no accuracy change expected (same math, different layout/orchestration)
| # | Change (from B) | Evidence | Expected gain |
|---|---|---|---|
| 1.1 | Replicated residual (L1, 40 cores) + one `all_reduce_async` per row-parallel output with a persistent buffer: 2 CCLs/layer instead of 4 (AG + RS ×2) | **Measured** in prototype: 27.7 → 28.8 t/s/u, accuracy unchanged | −1.4 ms/token |
| 1.2 | GDN decode on B's fused kernels: `kda.qkv_causal_conv1d_silu` conv, `chunk_gated_delta_rule` recurrence, `kda.sigmoid_gated_rms_norm` gate-norm (replaces ~69 small ops/layer: shift-register conv, repeat_interleave, typecasts, 3 matmuls + elementwise chain) | Profile: A GDN-layer internals ~260 µs vs B ~207 µs | ~−2.5 ms/token (×48 layers) |
| 1.3 | Attention decode cleanup: fused K+V paged cache update, sigmoid fused into the gate multiply, partial RoPE without transposes, fewer reshards (A ~45 device ops vs B ~29) | Profile: ~45 µs/attention layer | ~−0.7 ms/token (×16 layers) |
| 1.4 | Traced prefill for short prompts (<2048 tokens) + first token sampled on device (A runs ~3.3k ops eagerly and reads back full-vocab logits) | Profile + code reading | TTFT ~120 → ~60–70 ms |
| 1.5 | DRAM-sharded multi-reader matmul for attention QKV (only QKV benefits at BFP8) | Matmul microbenchmark: 70 → 55 µs | ~−0.25 ms/token |
| 1.6 | Device-resident decode loop: sampled token fed back on device, on-device position/RoPE increment, deferred readback | Measured host overhead 0.5 ms/token | ~−0.5 ms/token |
| | **Tier 1 total** | | **≈ 32–33 t/s/u (+17–19%), TTFT ≈ halved** |

## Tier 2 — selective BFP4 (each group measured alone with no measurable loss; combination untested)
| Group → BFP4 + LoFi | Measured alone (on top of 1.1, base 28.8) | accuracy_512 |
|---|---|---|
| MLP down | 29.8 t/s/u | 98.44 / 100 |
| GDN in-proj | 29.5 | 98.63 / 100 |
| Attention QKV + wo | 29.3 | 98.24 / 100 |
| + B's DRAM-sharded multi-reader configs for those groups (only pay off at BFP4) | microbenchmark, ~−0.9 ms | neutral |
Tier 1 + Tier 2 ≈ 35–36.5 t/s/u (≈ B). Combined accuracy must be measured; a full 198-question GPQA run is recommended before adopting.

## Tier 3 — not recommended without full task evals
GDN out-proj and LM head at BFP4 (+0.27 / +0.22 t/s/u each). All-BFP4 (B's recipe) measured 32.0 t/s/u at 96.88 / 99.41 (below the 97 top-1 target); with LM head kept BFP8: 31.7 at 97.07 / 99.41. A T3K port of the same BFP4 policy scored 81–85% on full GPQA vs the published 89.2.
Not worth porting: `FABRIC_1D_RING` (measured no effect). BFP8 KV cache only matters at long context (already available in A via `QWEN_SDPA_BF8=1`).

## Measurement protocol per change
Environment: `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` (workaround for tt-metal#55957 warm-cache hang), `HF_MODEL=/home/runara/models/Qwen3.6-27B`, `MESH_DEVICE=P150x4`.
1. `-k "traced_128 and not traced_128k"` ×2 (TTFT, decode t/s/u, generated text).
2. `-k accuracy_512` (gate).
3. Regression smoke: `-k traced_4k`, `-k batched_128_b8`.
Each change is behind its own env flag (default on in this branch; `=0` restores the previous path). Results are recorded in `QWEN38_OPTIMIZATION_LOG.md`.
