## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job definition, gate, allowed paths
- $HISTORY and every node's reflection.md (all 23 nodes, r01 + r02) — see Nodes consulted
- kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE (DST-accumulated x*x + ones*S^T matmul), x*gamma prescale under the
  AG, and the post-AG combine (2 ELWADD + transpose_dest<fp32> + mul_unary + add_unary + rsqrt on the full tile)
- tt_metal/hw/inc/api/compute/transpose_dest.h, tt-llk blackhole llk_math_transpose_dest.h — 32-bit transpose cost is a
  few dozen MOV instructions + cfg RMWs; stalls on WAIT_SFPU, so SFPU may run before it
- tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h, ckernel_sfpu_rsqrt.h, ckernel_sfpu_sqrt.h — rsqrt_tile is
  VectorMode::RC, 8 iterations/face, ~25-instruction body (non-approx: poly + 1 NR step + edge v_ifs)
- tt_metal/hw/inc/api/compute/experimental/add_rsqrt.h + experimental ckernel_sfpu_add_rsqrt.h — BH-only fused
  rsqrt(x*INPUT_SCALE + addend) with template VectorMode and ITERATIONS
- tt-llk blackhole llk_math_eltwise_sfpu_common.h (apply_vector_mode: R = faces 0,1; face step via SETRWC CR_D, i.e.
  relative to the face base) and cmath_common.h inc_dst_addr
- tt-llk blackhole ckernel_sfpu_triangle_solve.h / experimental ckernel_sfpu_rope.h comments — SFPU lane map on BH:
  one SFPLOAD = 4 rows x 8 cols of one column parity; addr+2 = odd columns of the same rows. So row 0 of a face needs
  iterations 0 and 1.
- /opt/tenstorrent/sfpi/include/sfpi_classes.h — dst_reg++ is TTINCRWC by SFP_DESTREG_STRIDE=2
## Nodes consulted
- r02-b02-a04 — TRISC_0 zones C_COMB/C_POST: 1.27-1.28 µs fixed gap between the combine's unpack and POST start on all
  shapes; its next-step #1 is this mechanism. I reuse its zone placement to measure.
- r02-b02-a03 (round root) — current code; PRE-side tail lives in the DST->L1->SrcA handoff, not in single-tile FPU ops
- r02-b02-a02 — BRISC gather costs; shows sub-µs fixed costs on the AG path are measurable
- r01-b04-a04 — PRE tail is fixed per-row overhead, not per-tile; the same logic applies to the post-AG combine
- r02-b02-a01, r02-b01/b03/b04 — drain/NoC work: drain is throughput-bound from its start, so starting POST earlier
  moves the drain end earlier
- r01-b0x — column split, trid read, gamma on BRISC, x*gamma under AG: all already in this lineage or orthogonal
## Docs / external references
- none beyond the in-tree LLK/SFPI headers above
