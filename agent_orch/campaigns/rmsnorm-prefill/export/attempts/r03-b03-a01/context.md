## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: the rules, the metric and allowed_paths.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp: the PRE (mm_row_stat), x*gamma prescale and packed-AG combine
  branch (2 ELWADD -> transpose_dest<fp32> -> mul_unary 1/H -> add_unary eps -> rsqrt, all VectorMode::RC), then the
  single POST pass.
- tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h, binop_with_scalar.h: both are hard-wired to VectorMode::RC,
  8 iterations per face.
- tt_metal/hw/inc/api/compute/experimental/add_rsqrt.h + blackhole llk_api/experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h:
  fused rsqrt(x*INPUT_SCALE + addend). It takes vec_mode and ITERATIONS template args and uses the same
  `_calculate_sqrt_body_` as rsqrt_tile. BH only.
- blackhole llk_api/llk_sfpu/ckernel_sfpu_sqrt.h: the non-approx rsqrt body is ~25 SFPU instructions per iteration.
- tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h: VectorMode::R = faces 0,1, ITERATIONS each.
- tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_triangle_solve.h (comment): one SFPLOAD = 4 rows x 8 cols, even
  cols at addr, odd cols at addr+2. So ITERATIONS=2 covers rows 0-3 fully, and 1 iteration would miss the odd
  columns.
- tt-llk/tt_llk_blackhole/llk_lib/llk_math_transpose_dest.h: the 32-bit transpose path STALLWAITs on SFPU and math
  before its MOVs, so SFPU work before it is safe.
- tests/tt_metal/.../norm_fidelity_mul_reduce.cpp, models/demos/deepseek_v3_b1/unified_kernels/rmsnorm.hpp: existing
  add_rsqrt_tile users with a non-RC vector mode and a reduced ITERATIONS.
## Nodes consulted
- All 23 nodes' reflections (r01-*, r02-*).
- r02-b02-a04: the key one. Its C_COMB/C_POST zones measured a 1.27-1.28 µs fixed gap between the combine unpack and
  POST start. Its comb.py is reused for the measurement here (same zone names).
- r02-b02-a03 (parent): PRE tail findings. The S round trip there, not the stage count, is what costs.
- r02-b02-a01: the drain is throughput-bound from its start, so starting it earlier moves its end earlier.
## Docs / external references
- none beyond the in-tree LLK headers above.
