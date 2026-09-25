# MiniMax-H3 on Wormhole Galaxy: ff1 SwiGLU LUT silu — handoff (paused 2026-09-21)

State of the ff1 SwiGLU epilogue work at the moment it was paused, 2026-09-21 ~23:20 on `UF-EV-B12-GWH02`, branch
`minimax_h3_wh_optimizations` at `c9487436276` plus the working tree described below (committed as-is on 2026-09-23, still behind the opt-in define). Context and the
experiments that led here: [ff1.md](ff1.md) §3.2 (change B). Block-level context: [README.md](README.md).

## 1. What the tree does right now

**The model's ff1 path is unchanged.** Both SwiGLU kernels still run the committed epilogue: `copy` gate and up
into DST, `silu_tile<false>` (the bf16-grade silu landed 2026-09-19, commit `a5454bba743`), `mul_binary_tile`,
pack. Verified after the last edit with the model's exact config (HiFi2, fp32 dest, `math_approx_mode=True`,
(8,7,10) 2x2, 4-device ring, 10 back-to-back calls):

| | host ms per call | worst PCC | worst rel-RMSE |
|---|---|---|---|
| committed kernel, measured earlier in the session | 15.97 | 0.9999694 | 0.00828 |
| working tree as left, default build | 16.08 | 0.9999694 | 0.00828 |

Identical numerics to the last digit; the host time is within the run-to-run band (15.97-16.08 ms across four runs).
`test_linear_swiglu` 4/4 at PCC 0.99998.

**The LUT silu is in the tree but compiled out by default.** It is selected by a compile define; since 2026-09-24 the three
fused-SwiGLU program factories set it when `TT_MM_SWIGLU_LUT_SILU=1` is in the environment (`compute_throttle_utils.cpp`,
`add_swiglu_lut_silu_define_if_needed`; the define is part of the kernel hash, so LUT and exact builds coexist in the kernel
cache). Block profile and a 50-step 15 s video with it on: [README.md](README.md) Part 2, *Roofline with every optimization on*
(ff1 15.70 -> 12.11 ms with fp32 dest off as well; CLIP 35.90). What follows describes the tree as paused on 2026-09-21:

- `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/swiglu_lut.hpp` (new, untracked): `swiglu_lut_tile_init()`
  programs the 6-segment fp16 sigmoid table into LReg0-2 / LReg4-6; `swiglu_lut_tile(gate_idst)` computes
  `DST[gate] = gate * sigmoid_lut(gate) * DST[gate + 1]` in one SFPU pass (`sfpi::lut<LutMode::Fp16x6_HWM4>`,
  sign retained, + 0.5). `kSwigluLutSilu` (`swiglu_lut.hpp:40-42`) is `true` only under `-DSWIGLU_LUT_SILU`.
- `all_gather_minimal_matmul_async/device/kernels/compute.cpp:61` and `minimal_matmul/device/kernels/compute.cpp:95`:
  `if constexpr (kSwigluLutSilu)` around the epilogue's SFPU calls; the `else` branch is the committed code, byte for
  byte. Both files include the header; a `static_assert(UP_DST == GATE_DST + 1)` guards the fused pass's DST layout.
- `models/tt_dit/tests/models/minimax_h3/tools/fit_sigmoid_lut.py` (new, untracked): fits and scores the tables; its
  HWM4 row reproduces the coefficients in the header exactly.

Nothing is committed. `git status` shows the two modified `compute.cpp`, the new header and the new tool.

## 2. Measured: before / after on the whole ff1 op

All three variants on the same bench (`transformer_op_mesh_bench.py --op ff1 --iters 10`: one TP ring, fused
SwiGLU, `bias=None`, HiFi2, fp32 dest, (8,7,10) 2x2, ping-pong semaphores like `CCLManager`), same host, same
hour. Host time is wall clock over 10 back-to-back calls; device time is Tracy `DEVICE KERNEL DURATION` of the
AGMM op, mean over 4 devices x calls 1-10 (call 0 excluded), from a second run of the same command under
`python -m tracy -r -p`. PCC / rel-RMSE against fp32 torch on the first 2048 rows of every device, worst device.

| variant | how it was built | host ms/call | device us | worst PCC | worst rel-RMSE |
|---|---|---|---|---|---|
| A. 2026-09-18 baseline: fp32-accurate `silu_tile` | one-token revert of commit `a5454bba743`, then restored | 16.56 | 15,773 | 0.9999716 | 0.00805 |
| B. committed: `silu_tile<false>` + `mul_binary_tile` | the tree as left (default build) | 15.97 | 15,173 | 0.9999694 | 0.00828 |
| C. fused 6-segment LUT silu | `-DSWIGLU_LUT_SILU` (then gated on `math_approx_mode`, see §4) | **14.62** | **13,554** | 0.9999734 | 0.00851 |

C vs A: **-11.7% host, -14.1% device**. C vs B: -8.5% host, -10.7% device. PCC moves by 4e-6 in C's favour;
rel-RMSE by +0.0005 against it (bar: PCC > 0.9995, rel-RMSE < 0.02). Per-device device-time medians: A 15,533-15,893
us, B 15,055-15,302, C 13,324-13,708. Tracy reports: `generated/profiler/reports/2026_09_21_23_12_29` (C),
`2026_09_21_23_13_01` (B), `2026_09_21_23_13_26` (A); `tools/op_time_from_profiler_csv.py <report>/ops_perf_results_*.csv AllGatherMinimalMatmul`
prints the per-call numbers.

Epilogue accounting: B -> C removes 1,619 us of the ~2,191 us SwiGLU zone measured in ff1.md exp 13, leaving
~0.6 ms (copy, one LUT pass, pack per gate/up pair). The K loop (13.0 ms, 8.0 of it the HiFi2 roofline) is untouched.

`test_linear_swiglu` with the LUT compiled in (K=256 fp32 randn inputs, gate std ~16, so most |x| > 4 where the
table saturates at 1): PCC 0.9999828-0.9999838 on 4/4 cases, equal to or above the exact path's 0.9999823-0.9999829.

Fitter output (`fit_sigmoid_lut.py`, bench gate distribution: std 0.50, 4.6% above |x| = 1, 0.007% above 2, none
above 4), rel-RMSE of the bf16 SwiGLU output vs an exact-sigmoid fp32 golden:

| sigmoid | rel-RMSE | max \|sigmoid error\| on [-8, 8] |
|---|---|---|
| exact (bf16 output rounding only) | 0.00166 | 0 |
| shipped 8-bit 3-segment table (`sigmoid_appx`) | 0.02505 (fails the bar) | 0.119 |
| fitted fp16 3-segment (1, 2) | 0.00858 | 0.119 |
| **fitted fp16 6-segment HWM4 (0.5, 1, 1.5, 2, 4)** — the one in the header | **0.00230** | **0.018** |

## 3. Reproduce

```bash
# default build (variant B), host ms + PCC / rel-RMSE on all 4 devices; ~40 s including mesh open
timeout 600 python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff1 --iters 10
# device kernel time for the same run
timeout 900 python -m tracy -r -p models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff1 --iters 10
R=$(ls -td generated/profiler/reports/*/ | head -1)
python models/tt_dit/tests/models/minimax_h3/tools/op_time_from_profiler_csv.py $R/ops_perf_results*.csv AllGatherMinimalMatmul
# single-device PCC of both SwiGLU kernels' epilogue (5 s)
python -m pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py -k test_linear_swiglu -q
```

Variant C: add `defines["SWIGLU_LUT_SILU"] = "1";` next to `defines["FUSE_SWIGLU"] = "1";` in
`all_gather_minimal_matmul_async_program_factory.cpp:663` (and the equivalent line of the `minimal_matmul` factory for
the single-device kernel), rerun the above; kernels are JIT-built, no `build_metal.sh` needed. Variant A: change
`silu_tile<false>(GATE_DST)` to `silu_tile(GATE_DST)` in the AGMM `compute.cpp` `else` branch, measure, restore.

Recovery if a ring run hangs (none did in this session): ff1.md §6.

## 4. Decisions still open, and what was tried for the switch

1. **How the LUT is enabled.** *Resolved 2026-09-24 as an env-var opt-in, `TT_MM_SWIGLU_LUT_SILU=1`, read by the program factories
   next to the other `TT_MM_*` matmul switches; a host-side op argument remains the cleaner long-term route if it is adopted.*
   The session first tied it to the compute config's `math_approx_mode` (`APPROX`), which
   the model sets to `True` (`transformer_block_minimax_h3.py:149`) — semantically right (silu ignored the flag before)
   and zero plumbing, but it would have flipped the model's epilogue on the spot, so it was replaced by the inert
   define before pausing. If the approx-mode route is taken later: `APPROX` is emitted by the JIT descriptors for the
   MATH and PACK builds only (`tt_metal/jit_build/genfiles.cpp`, `emit_math_scalar_descriptors`), so the constant must be
   guarded `#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)` with a `false` fallback for UNPACK, or the unpack
   TRISC fails to compile with `'APPROX' was not declared`. The other route is a host-side op argument on both ops.
   Note the sweep harness's `ff1_swiglu` use case runs `math_approx_mode=False` (the model runs True); the mesh bench
   runs True.
2. **Model-level numerics.** The op-level bar passes with margin, and PCC is unchanged, but the sigmoid error is 0.018 at
   |x| = 4 and the table saturates to 1 above it; the bench's gates have std 0.5, the model's may not. Run CLIP /
   VBench (README Part 4) with the LUT enabled before adopting it for the model.
3. **Blackhole.** The header uses `sfpi::lut<Fp16x6_HWM4>` and `vLut16ss/vLut16ii`, the same API the Blackhole gelu
   kernel uses (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h:211-233`); compiled and run on
   Wormhole only. `minimal_matmul` SwiGLU is used by Blackhole models, which is another reason the define is inert.
4. **Still to do when un-paused:** decide 1, run 2, then fold the results into ff1.md §3.2 / §4 / §5 and the README
   per-op and experiments tables, commit (fetch-and-guard first; the remote branch moves), and optionally re-profile the
   block (`tools/block_profile_stats.py`) to update the README's 3-AGMM row (31.85 ms; -1.6 ms projected).
5. Unrelated to the LUT but found alongside it (ff1.md §3.2, exp 15): `swiglu_block` processes the padded
   `N_block_tiles` rather than the core's clipped width; at (8,7,10) that is 30 vs 28 tiles per core (7% of the
   epilogue). Not measured on its own.
