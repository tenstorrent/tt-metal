# Round 1 proposals (from `analysis.md`, mapped to the optimize skill)

Current best (round 1, paper / accuracy): **836.4 ms** steady wall and **702.0 ms** device. Matmuls take 310.8 ms (44 %). Non-matmul ops take 55.6 %, almost all of them DRAM-bound elementwise / attention-plumbing ops whose per-call time did not change in round 1.

Savings overlap and are not additive. The estimates are full-model device ms unless marked "wall". The gates are the same as round 0:
- the PCC suite (15 tests);
- paper normalized PCC ≥ 0.999;
- a paper median improvement of ≥ 2 %;
- no regression > 3 % on `long_context`, `multivariate` or the `performance()` precision.

| # | Proposal | Finding | Est. savings | Confidence | Risk / complexity |
|---|---|---|---|---|---|
| R2-1 | Fused SwiGLU: `minimal_matmul(fuse_swiglu=True)` replaces up + gate + MUL | F1 (round 0 P2) | 70–100 ms | medium | medium |
| R2-2 | RoPE via `rotary_embedding_llama`, with Wq/Wk rows permuted at load | F3 (round 0 P3) | 45–60 ms | medium | medium |
| R2-3 | Residual ADD fused into wo / w_down (`dit_minimal_matmul_addcmul_fused`) | F4 | 25–35 ms net | medium | medium |
| R2-4 | SDPA `output_concat_heads=True` (drops 24 concat-heads ops) | F5 | 15–17 ms | high | low |
| R2-5 | Gate sigmoid moved into the consumer MUL (`input_tensor_b_activations`), wg on `minimal_matmul` | F6 | ≈ 8 ms | medium-high | low |
| R2-6 | Host prepare / upload / postprocess (cProfile first) | round 0 P4 | 30–60 ms wall | low | low |
| R2-7 | Drop per-row tile padding outside attention | F2 (round 0 P6) | 0–50 ms after R2-1/3/5 | low | high |

Dropped: round 0 P5 (packing wg into wqkv). Since P1, in0 is read once, so a merge saves one 151 MB read but adds a split and loses the per-role fused activation (analysis §7).

Deferred: multivariate variate attention (+357 ms over paper, F9). It is a guard shape, not the target, and has not been profiled. Precision per tensor group (LoFi pays only once R2-1 makes the MLP compute-bound). L1-resident M-chunks (after R2-1/3/4).

Exit criteria note: the "main matmuls > 65 % FLOPs and non-matmul < 15 %" criterion cannot be reached with stock ops. The 768x768 matmuls top out at 64.4 % at their DRAM roofline, and the irreducible non-matmul work (≈ 180 ms) exceeds the matmul floor (≈ 149 ms). The loop will therefore end on the "< 3 % for two rounds" or "6 rounds" criteria.

## R2-1. Fused SwiGLU

- **Hypothesis:** the MLP front half is 7.78 ms per layer (up + SILU 2.79, gate 2.28, MUL 2.71), 186.6 ms over 24 layers (26.7 % of device time). It moves 2.32 GB per layer. One `minimal_matmul(fuse_swiglu=True)` on a tile-pair-interleaved [up | gate] weight (768 x 4096 → 2048) reads xn once and writes the product once (560 MB). It is compute-bound at about 3.4–4.6 ms.
- **Optimize-skill rules:** "Fused or packed repeated same-input projections ... paired gate/up projections in 3-matmul MLPs"; it must beat the tuned split path (round 1) after counting all elementwise work.
- **Correctness:** `minimal_matmul` computes `silu(col tile 2p) * col tile 2p+1`, so the even column tiles take `w_up` and the odd ones `w_gate` (`models/tt_dit/utils/tensor.py::prepare_for_fused_swiglu`). The RMSNorm gain stays folded into both halves. N_block and subblock_w must be even.
- **Experiment:** a standalone sweep of the fused op (M/K/N blocks) at M = 98304 against the round 1 split path (5.07 ms of matmul + 2.71 ms MUL), then integrate in `tt/mlp.py`.

## R2-2. RoPE via `rotary_embedding_llama`

- **Hypothesis:** `rotary_embedding` is 48 x 2.0 ms = 95.8 ms (13.7 %) at 29.5 % of DRAM peak, from 4 serialized single-tile reads per tile. The llama prefill op reads each tile once and cos/sin once per batch; expect 0.7–1.0 ms per call.
- **The change:** permute the Q and K output rows of `wqkv` at load time so rotate-half becomes the interleaved-pair form (q·k is invariant to the same head_dim permutation on both). Build llama-style cos/sin tables (identity on partial-RoPE dims) and a trans_mat. Only the shared-positions path changes.
- **Optimize-skill rule:** "Use SDPA and other optimized composite ttnn ops ... where the target model fits their contracts".
- **Experiment:** a rotary microbenchmark (old op vs llama op on (1024, 12, 96, 64)), a CPU check of the permutation, then integrate and run the PCC suite.

## R2-3. Residual ADD fused into wo / w_down

- **Hypothesis:** all 54 residual ADDs (1.0 ms each, 54 ms) directly follow a `minimal_matmul` (30 wo, 24 down). `dit_minimal_matmul_addcmul_fused` computes `residual + 1.0 * (A @ W) * ones[1, N]` in the matmul's epilogue, at the cost of reading one extra 151 MB tensor (+0.3–0.55 ms per call).
- **The change:** the attention and MLP sublayers take the residual input and return `x + out` directly; `TtTransformerBlock` drops `add_residual` for those calls.
- **Experiment:** a standalone timing of the fused op vs minimal_matmul + add at the wo / down shapes, then integrate.

## R2-4. SDPA `output_concat_heads=True`

- **Hypothesis:** concat-heads is 24 x 720 µs = 17.3 ms. Plain (non-chunked) SDPA can write `[b, 1, s, H * Dh]` directly.
- **Experiment:** flip the flag in `TtAttention._sdpa` for the time path (and the blocked variate path if it supports it), run the PCC suite and benchmark.

## R2-5. Gate activation into the consumer MUL

- **Hypothesis:** the fused SFPU sigmoid costs ≈ 300 µs per call inside the matmul (1.04 vs 1.36 ms in the round 1 sweep); the gate MUL is DRAM-bound with cycles to spare. `ttnn.mul(merged, gate, input_tensor_b_activations=[SIGMOID])` absorbs it, and wg then runs on the 110-core `minimal_matmul` path like wo.
- **Experiment:** integrate in `tt/attention.py` (30 calls) and benchmark. The SILU half is subsumed by R2-1.

## R2-6. Host-side wall time

- **Hypothesis:** prepare 52 ms, upload 36 ms and postprocess 18 ms total 105.5 ms per call, now 12.6 % of the headline wall time.
- **Experiment:** cProfile `prepare_inputs` / `upload_inputs` / `postprocess` at the paper shape. Then remove redundant work: build tokens at their final padded size, skip fp32 copies, cache per-shape RoPE uploads, vectorize the postprocess. Gate on bit-identical host outputs.

## R2-7. Drop per-row padding outside attention

- **Hypothesis:** 31 % of the flat-op work is on pad rows (157 ms gross today). The cost is one unpack/pack pair per attention sublayer at ≈ 1–2 ms each (48–96 ms), so the net is 0–50 ms once R2-1/3/5 shrink the flat work.
- **Experiment:** gate on a tiled-reshape micro-benchmark at the real shape; proceed only if a pair costs ≤ 1.5 ms. Do it last.
