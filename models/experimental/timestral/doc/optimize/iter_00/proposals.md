# Round 0 proposals (from `analysis.md`, mapped to the optimize skill)

Baseline (paper / accuracy): **1445 ms** steady wall and **1307 ms** device. Matmuls take 819 ms (62.5 %), all SLOW at 13–20 % of HiFi2 FLOPs.

Savings overlap and are not additive. The estimates are full-model device ms unless marked "wall". Every candidate is gated on:
- the PCC suite in `tests/pcc/test_modules.py` (all 15 tests);
- paper-shape normalized PCC ≥ 0.999;
- a paper median improvement of ≥ 2 %;
- no regression > 3 % on `long_context`, `multivariate` or the `performance()` precision.

| # | Proposal | Findings | Est. savings | Confidence | Risk / complexity |
|---|---|---|---|---|---|
| P1 | Tuned matmul engine per linear role, with activations fused in-kernel | F1, F3 | 400–600 ms matmul + 90–100 ms unary | high | medium |
| P2 | Fused SwiGLU: one `minimal_matmul(fuse_swiglu=True)` for up/gate/silu/mul | F1, F3, F6 | 200–280 ms (overlaps P1) | medium | medium |
| P3 | Rotary via `rotary_embedding_llama`, with the head_dim permuted at load | F4 | 40–65 ms | medium | medium |
| P4 | Host prepare/upload/postprocess | headline | 30–60 ms wall | low | low |
| P5 | Pack shared-input projections (`wg` into `wqkv`; variate `wg` + `wv`) | F7 | 15–30 ms (after P1) | low | low |
| P6 | Drop per-row tile padding outside attention (66 vs 96 tokens) | F2 | 150–290 ms net (scales with what is left) | low | high |

## P1. Tuned matmul engine per linear role, activations fused

- **Hypothesis:** the auto-selected `MatmulMultiCoreReuseMultiCast1DProgramConfig` (`mcast_in0=0`, `per_core_M=28`, `in0_block_w=2`, `out_block_w < per_core_N`) re-reads in0 from DRAM once per N block, 3–6x (rows 65–119). It runs at about 180 GB/s effective. A 2D multicast config sized so each core holds its whole N slice, or `minimal_matmul`, should approach the DRAM/compute roofline (LB 0.6–1.2 ms per call against 3.2–8.0 ms today).
- **Fused activations:** `ttnn.linear(activation=)` only fuses with a user core grid (`matmul.cpp:323-330`). Passing `fused_activation` in the program config, or using `minimal_matmul`, removes the 56 separate SILU/SIGMOID ops (101.7 ms).
- **Optimize-skill rules:**
  - "Start with a tuned 2D matmul for large prefill projections";
  - "consider `ttnn.experimental.minimal_matmul` as a candidate";
  - "Explicitly configure `program_config`";
  - the `in0_block_w` floor and the sweep upward;
  - "Leave a table for every dominant matmul search".
- **Prior art:** `chronos_forecast/tt/program_configs.py` (`fused_batch_matmul_config` and `in1_mcast_1d_config`), measured on the same P150 grid (11x10).
- **Experiment:**
  1. Write `tests/perf/sweep_matmul.py`, a standalone device sweep at the exact roles: M = 98304 with K x N = 768x2304, 768x768 (+sigmoid), 2048x768, 64x3072 (+bias, silu), 3072x768 (+bias) and 64x768 (+bias).
  2. Run each role on both engines: 2D grid with `in0_block_w` / out_block / subblock candidates, and `minimal_matmul` with M/K/N block and subblock candidates. Record device µs from wall time with sync and PCC against torch.
  3. Integrate the per-role winners in `tt/common.py::linear` (role-aware config), one role group at a time.

## P2. Fused SwiGLU MLP

- **Hypothesis:** the MLP front half (rows 87–90 / 115–118) is 4 ops and 16.0 ms per layer (384 ms over 24 layers):
  - `up` linear 5.3 ms;
  - SILU 2.8 ms;
  - `gate` linear 5.3 ms;
  - MUL 2.7 ms.
- **The change:** `minimal_matmul(fuse_swiglu=True)` on a tile-pair-interleaved [up | gate] weight (768x4096, output 2048 wide) computes `silu(up) * gate` in one op that reads `xn` once and writes the 2048-wide product once.
- **Optimize-skill rules:** OPT-010, "compare packed and split 3-matmul MLP gate/up". The packed form must beat the best split form from P1, counting all elementwise work.
- **Correctness:**
  - Reference: `w3(gate * silu(up))`, where `minimal_matmul` computes `silu(col 2p) * col 2p+1`. So even column tiles take `w_up` and odd column tiles take `w_gate` (`prepare_for_fused_swiglu` in `models/tt_dit/utils/tensor.py`).
  - The RMSNorm gain stays folded into both halves.
- **Experiment:** add it to the P1 sweep as an extra role, then integrate behind the MLP and compare whole-MLP time against the split P1 path.

## P3. Rotary embedding

- **Hypothesis:** `rotary_embedding` runs at 29.5 % of DRAM peak. It does 4 serialized single-tile reads per tile and re-reads the cos/sin tables per (row, head), at 2.0 ms per call x 48 = 95.9 ms.
- **The change:**
  - Q·K is invariant to permuting `head_dim` identically in Q and K. So permuting the Q/K output rows of `wqkv` at load time turns rotate-half RoPE into the interleaved-pair form that `ttnn.experimental.rotary_embedding_llama` (prefill, trans_mat) expects.
  - V and the attention output are untouched.
  - Partial RoPE keeps cos = 1 and sin = 0 on the identity dims.
  - Only the shared-positions path changes; per-row positions keep the manual path.
- **Optimize-skill rule:** "Use SDPA and other optimized composite ttnn ops ... where the target model fits their contracts".
- **Experiment:**
  1. A standalone rotary microbenchmark: old op vs llama op on (1024, 12, 96, 64) with real-shape cos/sin.
  2. Check the `head_dim` permutation on the CPU golden dataflow (exact equivalence).
  3. Integrate and run the PCC suite.

## P4. Host-side wall time (not device)

- **Hypothesis:** prepare 52 ms, upload 36 ms and postprocess 18 ms total 107 ms per call, 7.4 % of the headline wall time.
- **Experiment:**
  1. Run cProfile over `prepare_inputs` / `upload_inputs` / `postprocess` at the paper shape.
  2. Remove redundant work: build the tokens directly at their final padded size, skip fp32 copies before `from_torch`, cache per-shape RoPE/cos/sin uploads, and vectorize the postprocess.
  3. Keep the float64 causal scaling (reference numerics).
- **Gate:** bit-identical host outputs (tokens, loc, scale) against the current implementation.

## P5. Pack shared-input projections

- **Hypothesis:** `wg` and `wqkv` both read `xn`, and so do variate `wg` and `wv`. That is 54 extra reads of 151 MB and 54 extra ops. Today merging N doesn't cut traffic because re-reads scale with N blocks, so this only pays after P1.
- **Optimize-skill rules:** OPT-001 and "When several matmuls consume the same input activation ... packed/fused projection". It must beat the tuned separate path after counting the slice/split ops.
- **Experiment:** do it after P1. Pack at load time ([q, k, v, g] rows), run one matmul, then either split heads on a 2304-wide slice plus a slice for g, or split heads with a fused offset if supported.

## P6. Remove per-row padding outside attention

- **Hypothesis:** 1024 x 66 = 2112 tiles is already tile-aligned. Only create-heads / rotary / SDPA / concat-heads need the per-row (rows, 96) view; linears, norms and elementwise ops could run on the packed (1, 1, 67584, D) stream. That cuts their work by 31 %.
- **Cost:** one padded↔packed re-layout pair per attention sublayer. Row-granular moves estimated at ≥ 30–55 ms total.
- **Experiment:**
  1. Prototype the re-layout pair alone (`ttnn.reshape`/`pad`/`slice` on TILE vs ROW_MAJOR round trip) at the real shape.
  2. Proceed only if the pair costs ≤ 1.5 ms. The gain must be measured after P1–P3, since it scales with whatever per-token cost remains.

## Deferred

- **Precision per tensor group (bf8 / LoFi):** the analysis §8 shows math is only about 1/7 of matmul time today, so LoFi saved 56 ms. Retry after P1/P2, once the matmuls are closer to compute-bound.
- **L1-resident M-chunks:** this would cut the 172 GB of non-matmul DRAM traffic, the structural floor. It belongs in a later round.
- **Tracing:** no device gain while the forward is device-bound (analysis §7, host side).
