## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — rules, shapes, gate.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE (mm_row_stat), x*gamma pre-pass, P_NRED packed-AG
  combine (add_tiles x2, transpose_dest<true>, mul/add/rsqrt on full tile), single POST pass.
- tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h, binop_with_scalar.h — the APIs hard-code VectorMode::RC and
  ITERATIONS=8; SFPU_UNARY_CALL lets me pass both.
- tt-llk blackhole llk_math_eltwise_sfpu_common.h — VectorMode::R = faces 0,1 calling the full sfpu_func (ITERATIONS
  is the template arg); sfpu_start stalls on MATH.
- tt-llk blackhole llk_math_transpose_dest.h — 32-bit path; STALLWAIT on WAIT_SFPU before the moves.
- ckernel_sfpu_rsqrt.h / ckernel_sfpu_sqrt.h / ckernel_sfpu_binop_with_unary.h — loops over ITERATIONS with dst_reg++.
- ckernel_sfpu_generic_moe_gate_topk_top8.h — SFPLOAD at offset 0/2 = 4 rows even/odd columns: 2 iterations cover
  rows 0-3 of a face.
## Nodes consulted
- All 23 nodes' reflections (r01-*, r02-*).
- r02-b02-a04 — measured the 1.27-1.28 µs post-AG combine gap on TRISC_0 (C_COMB -> C_POST); suggested restricting
  the SFPU to the stat faces.
- r02-b02-a03 (parent/round root) — current code; PRE side already squeezed.
- r01-b04-a04, r02-b02-a02 — PRE tail findings (not repeated here).
## Docs / external references
- none
