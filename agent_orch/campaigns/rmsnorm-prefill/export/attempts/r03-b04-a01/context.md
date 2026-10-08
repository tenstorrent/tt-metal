## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml: job definition, allowed paths, gate.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp: PRE (mm_row_stat), x*gamma prepass, packed-AG combine chain
  (add_tiles x2 -> transpose_dest<fp32> -> mul/add/rsqrt full tile -> pack), single POST pass.
- tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h, binop_with_scalar.h: the tile APIs are SFPU_UNARY_CALL with
  ITERATIONS=8 and VectorMode::RC.
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h, ckernel_sfpu_rsqrt.h: the fp32 rsqrt body
  is ~30+ SFPU instructions per iteration (expensive on 32 iterations).
- ckernel_sfpu_binop_with_unary.h: calculate_binop_with_scalar<APPROX, MODE, ITERATIONS, fp32>(param).
- tt-llk llk_math_eltwise_sfpu_common.h: VectorMode::R = faces 0,1 with the functor's full ITERATIONS.
- tt-llk ckernel_sfpu_triangle_solve.h / ckernel_sfpu_reshuffle_rows.h: one SFPLOAD = 4 rows x 8 cols of a face, odd
  columns at address +2, so ITERATIONS=2 covers rows 0-3.
- tt-llk llk_math_transpose_dest.h: 32-bit transpose_dest is a replay/mop sequence (cheap compared to SFPU rsqrt).
- llk_math_eltwise_unary_sfpu_macros.h: SFPU_UNARY_CALL macro.
## Nodes consulted
- All 23 nodes (proposal, reflection, summary). Key ones:
- r02-b02-a04: C_COMB/C_POST zones; unpack idles 1.27-1.28 µs on the combine math+pack; reflection #1 suggests
  partial-face SFPU. comb.py reused.
- r02-b02-a03 (root): current best, the compute code this edits.
- r02-b02-a02/a03, r01-b04-a04: PRE tail findings (DST->L1->SrcA handoffs are the expensive part, not op count).
- r01-b02-a04, r01-b03-a04, r02-b01-a01, r02-b03-a01, r02-b02-a01: drain is NoC-throughput bound, so the drain end
  tracks when the first output tile appears.
## Docs / external references
- none beyond the in-tree LLK sources above.
