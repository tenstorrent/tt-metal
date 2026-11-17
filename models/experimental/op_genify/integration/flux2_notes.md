# FLUX.2-dev (tt_dit) × genify_matmul — running log (#65150)

Checkout `/localdev/avelickov/tt-metal`, branch `avelickov/65150-matmul-integration` (main 8e28a65 + the op),
QuietBox 2 (Blackhole, 4 chips), FLUX.2 on a (2, 2) mesh: SP2 on axis 0 × TP2 on axis 1, Linear, FSDP.

## Changes

- `models/tt_dit/layers/linear.py`: `TT_GENIFY_MATMUL=1` (adapter `enabled()`) routes
  - `_linear` (both the ≤128-row `ttnn.linear` branch and the `minimal_matmul` branch; `fuse_swiglu` →
    `activation="swiglu"`; output placement as stock: DRAM for the `ttnn.linear` branch, the input's
    memory config for `minimal_matmul`),
  - `ColParallelLinear` split path (`minimal_matmul_split`): one genify over the full N + `ttnn.chunk`
    (one `slice` per chunk),
  - `ColParallelLinear` fused addcmul path (`dit_minimal_matmul_addcmul_fused`): genify residual/gate
    when scalar == 1 and no bias,
  - `RowParallelLinear` list input (`minimal_matmul([x0, x1])`, concat over K): `ttnn.concat` + genify.
  - Fallback to stock, logged once per (site, reason): fused GELU, SwiGLU with bias, addcmul with bias
    or scalar ≠ 1, sharded / row-major operands, non tile-aligned chunk or concat segments, anything
    genify's support gate raises (`NotImplementedError`).
  - Untouched: Ring-topology CCL-fused paths (`_forward_fabric_agmm`, `all_gather_minimal_matmul_async`,
    `forward_fused_addcmul`).

## Log

- 20:30 `test_linear.py -k "2x2 and test_col_parallel_linear and not mochi_spatial"`, genify: 72 passed (5:11),
  op counter: 72 genify_matmul calls, 0 stock matmuls; min PCC 99.9992 %.
- 20:36 `test_linear.py -k "2x2 and test_row_parallel_linear"`, genify: 56 passed (4:05), 56 genify_matmul calls,
  0 stock; min PCC 99.9991 %.
- 20:40 genify: `test_linear.py` mochi_spatial col (4) + `test_feedforward.py -k "2x2 and mochi_prompt_ff"` (4) +
  `tests/encoders/test_transformer.py -k "test_prefill and 2x2sp"` (6): 14 passed (1:45), 78 genify_matmul calls
  (FF: ff1 SwiGLU + ff2; encoder: 11 per test), 0 stock.
- 20:52 stock baselines of the same 142 tests: 142 passed (op counter: minimal_matmul 88, ttnn.linear 118, genify 0).
  Per test worst PCC genify − stock: −0.0001 … +0.0002 percentage points (median 0); min PCC 99.9811 % in both
  modes (encoder logits), max RMSE/σ 2.0 % in both.
- 21:00 `flux2_test_transformer_pcc.py` (random weights, 1 double + 2 single, 1024², 512 prompt tokens, vs diffusers;
  targets PCC ≥ 0.996, RMSE/σ ≤ 0.09), all pass:

  | case | stock PCC / RMSE | genify PCC / RMSE | genify vs stock PCC / RMSE |
  |---|---|---|---|
  | prompt_replicated (bh_2x2_linear row) | 0.997518 / 0.0705 | 0.997519 / 0.0705 | 0.999993 / 0.0037 |
  | prompt_sharded (QB2 pipeline setting) | 0.997517 / 0.0706 | 0.997519 / 0.0705 | 0.999993 / 0.0037 |

  Op counter: stock 22 matmuls (minimal_matmul 11, minimal_matmul_split 5, dit_minimal_matmul_addcmul_fused 2,
  ttnn.linear 4); genify 22 genify_matmul, 0 stock, no fallback logged.
- 21:01–21:13 profiles (`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000 scripts/run_safe_pytest.sh --profile …`), two runs
  per mode, all PASS. CSVs under `generated/profiler/reports/`:
  DiT stock `2026_11_17_21_01_23`, `2026_11_17_21_09_40`; DiT genify `2026_11_17_21_03_54`, `2026_11_17_21_11_21`;
  TE stock `2026_11_17_21_05_56`, `2026_11_17_21_12_15`; TE genify `2026_11_17_21_07_12`, `2026_11_17_21_13_05`.
  Table: `flux2_perf_table.md` / `.csv` / `_glue.csv`, built by `flux2_perf_table.py` (pairs matmuls by order,
  aligns the op sequences with difflib to isolate glue, mean of 4 chips, averaged over the two runs).
  - Per denoising step (215 matmuls): stock 318.06 ms → genify 272.38 ms (0.856); glue +25.65 ms (QKV slices
    12.5 ms, single-block concat-K 13.1 ms, time_mod slices 0.03 ms) → 298.02 ms (0.937, −20.0 ms).
  - Whole step, scaled: 936.4 → 899.9 ms, but CCL ops vary ±4 % run to run (stock runs 917 / 956 ms); without
    all-gather / reduce-scatter / ring SDPA: 370.4 → 350.6 ms (−19.9 ms, matches matmul + glue).
  - Per image (prompt encoding, 201 matmuls): 69.57 → 70.44 ms (1.013): context_embedder 0.86×, TE qkv 0.83×,
    o_proj 1.10×, gate/up 1.02×, down 1.05× (genify uses 80–88 of 110 cores at M = 512).
  - Net per site incl. glue: single-block QKV 1247.5 → 1116.6 + 218.9 = 1335.5 µs (+7 %, ×47/step) and single-block
    out 1689.9 → 1430.7 + 274.4 = 1705.1 µs (+1 %, ×47) are slower than stock; everything else is faster except the
    M = 1 MLPs (linear_2 +8–9 %, guidance linear_1 14.4 vs 10.7 µs on 24 cores).
- 21:15 fallback checks (genify on): `test_feedforward.py -k "2x2 and sd35_text_ff"` 4 passed, ff1 (fused GELU) ran
  stock minimal_matmul with one log line, ff2 genify; `test_feedforward_swiglu_fused` (1×1) 3 passed, the bias
  variant fell back ("SwiGLU with bias"), the two bias-free ones ran genify SwiGLU (PCC 99.9976 %).
