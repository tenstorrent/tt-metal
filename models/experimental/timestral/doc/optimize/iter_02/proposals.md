# Round 3 proposals (from `iter_02/analysis.md`, mapped to the optimize skill)

Current best (round 2, paper / accuracy): **679.6 ms** steady wall and **530.9 ms** device (round 0: 1445.3 / 1307.2; round 1: 836.4 / 702.0).
- Matmuls take 290.9 ms (1.89x their 154 ms roofline). Non-matmul ops take 44.9 %.
- The time-attention core is the largest remaining block: create heads, two RoPE calls and SDPA cost 6.85 ms per layer, 164 ms over the model.
- The performance precision is now 19 ms *slower* on device than accuracy, because bf8 weights disable the fused residual add.

Savings overlap and are not additive. Estimates are full-model device ms unless marked "wall". The gates are unchanged:
- the PCC suite (15 tests);
- paper normalized PCC ≥ 0.999;
- a paper median improvement of ≥ 2 % (device time is the primary signal; host wall is noisy on this shared machine);
- no regression > 3 % on `long_context`, `multivariate` or the `performance()` precision.

| # | Proposal | Finding | Est. savings | Confidence | Risk / complexity |
|---|---|---|---|---|---|
| R3-1 | Fused head split + RoPE: `minimal_matmul_split` → `wan_fused_rmsnorm_post_allgather` (identity stats, RoPE fused) → SDPA on an (H, R, S, Dh) view → `nlp_concat_heads` | F1 | 45–55 ms | low-medium | high (4 ops reused outside their usual contract; micro-benchmark first) |
| R3-2 | `rotary_embedding_llama` kernel: keep batch-broadcast cos / sin resident across the batch loop (alternative to R3-1) | F2 | 35–42 ms | medium | medium (shared ttnn kernel outside the timestral tree; needs the op's unit tests) |
| R3-3 | Prune the last `time_variate` layer to the decoder rows after its time-attention SDPA | F5 | 12–16 ms | medium-high | low |
| R3-4 | Performance precision: keep wo / w_down in bf16 so the fused residual add runs | F4 | ≈ 33 ms (performance only) | high | low |
| R3-5 | Host wall: bf16 cast before `from_torch`, per-patch scaler statistics, median host split | F3 | 30–45 ms wall | medium-high | low-medium |
| R3-6 | Pack wg into the split QKV matmul ([wg \| wv] on the V-path), sigmoid in the gate MUL (needs R3-1) | F6 | 12–18 ms | low-medium | medium |
| R3-7 | Unpadded residual stream outside the attention core (pack / unpack around it) | F7 | 15–40 ms net | low | high (gate on a pack / unpack micro-benchmark ≤ 1.2 ms) |
| R3-8 | Matmul gaps: SwiGLU `[16,12,2]` and SFPU approx mode for silu, QKV re-sweep | F8 | 10–20 ms | low | low |
| R3-9 | RMSNorm program config; embed ADD folded into its matmul, MUL keep + ADD query as one `addcmul` | F10 | 5–10 ms | low-medium | low |
| R3-10 | Multivariate only: variate attention in row-major order with a same-slot mask (no permutes, no 4x padding); profile first | F11 | 200–300 ms on multivariate (guard shape) | low | high |

Dropped or deferred:
- F9 (fuse the gate MUL): ≈ 3 ms model-only; the larger version needs kernel changes in SDPA or minimal_matmul.
- L1-resident M-chunks: very low confidence; R3-1 / R3-2 attack the same traffic.
- The context / decoder split stream: a design study, not a candidate yet.
- Trace capture: host issue is hidden behind device time (≤ 8 ms wall).

## R3-1. Fused head split + RoPE

- **Hypothesis:** create heads (2157 µs) plus two RoPE calls (2 x 1625 µs) move ≈ 1.8 GB per time layer. `minimal_matmul_split(chunks=3)` writes q, k and v flat. `wan_fused_rmsnorm_post_allgather` with constant identity stats (norm scale exactly 1) then writes `[1, 12, M, 64]` per tensor, applying RoPE to q and k from the existing pair-order cos / sin and trans_mat. That is a free view of (12, R, S, Dh), so causal SDPA runs per (head, row) as today, and `nlp_concat_heads` restores `[1, 1, M, 768]`.
- **Expected:** 4.3–4.8 ms per layer instead of 6.85 ms, i.e. −2.1 to −2.6 ms x 24.
- **Optimize-skill rules:**
  - "Optimize the operation topology before fine-tuning individual op configs";
  - "Use SDPA ... when the target model fits their contracts";
  - "Avoid data movement before tuning math".
- **Risks:**
  - The post op is tuned for the Wan DiT (batch 1, S ≈ 10–75k), so its speed at S = 98304 and its numerics with identity stats are unknown.
  - Full-length cos / sin tables add 12.6 MB each per forward.
  - `output_concat_heads` cannot be used on the (H, R, ...) view.
- **Experiment:** a micro-benchmark of the 3-op split + post-op chain against create heads + 2 RoPE at the paper shape (time and PCC), then integrate in `_time_context` and run the PCC suite.

## R3-2. RoPE kernel: cos / sin resident across batches

- **Hypothesis:** both `rotary_embedding_llama` paths re-read the shared cos / sin tiles for every batch (reload: every row). That is 604 MB per call instead of 302 MB, which is why the call takes 1.63 ms. Hoisting the cos / sin reserve / read in the reader, and the wait / pop in compute, out of the batch loop when cos has batch 1 should give 0.75–0.9 ms per call.
- **Expected:** −0.75 to −0.88 ms x 48.
- **Risks:**
  - This changes a shared ttnn kernel used by the llama-family models.
  - It must pass `tests/ttnn/unit_tests/operations/transformers/test_rotary_embedding_llama*.py`.
  - It lives outside `models/experimental/timestral`.
- **Experiment:** the change plus the op's unit tests, the R2-2 micro-benchmark (`(R·H, 1, S, Dh)` view), then the PCC suite and benchmark.

## R3-3. Prune the last layer to the decoder rows

- **Hypothesis:** the head reads only tokens `ll .. ll+h`. In the final `time_variate` layer, nothing after the time-attention SDPA mixes tokens across time, so the gate MUL, wo, the variate sublayer, the MLP and their norms can run on R·h = 2048 rows instead of 98304.
- **Expected:** removes ≈ 15.6 ms and adds ≈ 1.8 ms of slices and small ops, ≈ −14 ms net.
- **Optimize-skill rule:** "Count material matmuls ... in the measured path".
- **Correctness:** multivariate variate attention mixes variates within one slot only, so the decoder slots of all variates are enough.
- **Experiment:** add a `last_layer` mode to `TtTransformerBlock` that slices after SDPA, adjust `TtTimestral.head`, then run the PCC suite (block + forecaster tests) and benchmark.

## R3-4. Performance precision: bf16 wo / w_down

- **Hypothesis:** `dit_minimal_matmul_addcmul_fused` needs residual dtype == weight dtype. With bf8 weights the 54 residual adds come back (+33.5 ms). wo / w_down are DRAM-bound on activations, so bf8 weights save nothing there; LoFi still applies.
- **Expected:** performance precision 549.9 → ≈ 516 ms. No effect on accuracy precision.
- **Optimize-skill rule:** "OPT-013: prove dtype policy reached the measured ops"; precision one tensor group at a time.
- **Experiment:** a per-role weight dtype in `TtTimestralPrecision`, then the PCC suite and benchmark (performance precision PCC must stay ≥ 0.999 normalized).

## R3-5. Host wall

- **Hypothesis:**
  - `upload_inputs` converts fp32 tokens to bf16 inside `ttnn.from_torch` (21.6 ms). A torch `.to(torch.bfloat16)` first costs 3.7 ms in total, and the result is bit-identical.
  - The float64 cumsum scaler statistics over 2048 steps per row cost 24 ms. Per-patch sums of x and x² followed by a 64-step cumsum give the same patch-boundary statistics.
- **Expected:** −30 to −45 ms wall, no device change.
- **Correctness:** the golden scaler tests (`test_causal_patch_scaler_matches_naive`) and the PCC suite. The float64 change is not bit-identical, but differs far below bf16.
- **Experiment:** CPU timing before and after, then the benchmark with the split taken as a median of the steady runs.

## R3-6 to R3-10

See `analysis.md` §7 (F6, F7, F8, F10, F11) for evidence and mechanisms. Each would follow the same flow:
1. a micro-benchmark or sweep at the exact shape;
2. integration;
3. the PCC suite and benchmark;
4. a record in `candidates.md`.
