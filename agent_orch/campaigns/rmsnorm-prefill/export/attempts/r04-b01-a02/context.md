## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — rules, gate (pcc 0.99999, max_abs 0.05).
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE resident path (DST-accumulated mul_tiles x*x, S pack,
  mm_row_stat matmul ones*S^T), prescale x*gamma, POST. All fidelity comes from MATH_FIDELITY (HiFi4, set in
  device_operation.cpp init_device_compute_kernel_config default).
- tt_metal/hw/inc/api/compute/eltwise_binary.h — mul_init = state_configure + math binary init<MATH_FIDELITY> +
  unpack_AB_init; mul_tiles = unpack_AB + math binary<..., MATH_FIDELITY>.
- tt_metal/hw/inc/api/compute/matmul.h — matmul_init/matmul_tiles use MATH_FIDELITY, MM_THROTTLE=0.
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h — in0 -> SrcB, in1 -> SrcA (so the ones
  tile is in SrcB: HiFi2 is exact for the row-sum matmul).
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h — explicit-fidelity LLK templates.
- tt_metal/hw/inc/api/compute/sentinel/compute_kernel_sentinel.h — state_configure needs call_line.
## Nodes consulted
- All 39 nodes via history.md; full proposal/reflection for r04-b01-a01 (parent), r04-b02-a01, r04-b03-a01,
  r04-b04-a01; fidelity-related reflection sections of r01-b01-a03, r01-b01-a04, r01-b02-a02, r01-b04-a03,
  r01-b04-a04, r02-b02-a01/a02, r03-b01-a03, r03-b02-a01, r03-b03-a02, r03-b04-a02.
- r03-b01-a03 fid.py — BH fidelity mask emulation: HiFi2 drops SrcB's last bf16 mantissa bit, max_abs ~0.0226 emu.
- r01-b04-a04 #4 argued per-tile PRE throughput isn't on the critical path (at that time, whole-row gamma on BRISC was
  later); r03-b04-a02 / r03-b03-a02 later found PRE compute bound / a ~0.7 µs PRE tail.
## Docs / external references
- tt-metal#58723 (as quoted in r03-b01-a03): BH ELWMUL 82.6 (HiFi4) vs 34.6 (HiFi2) cycles/tile.
