## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths, gate.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — resident PRE (ELWMUL DST-accumulated S, pack, ones*S^T matmul, pack), post-AG combine, end-of-kernel pops; transformation_mat_cb only used with fused RoPE.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp + dit_rmsnorm_scalar_setup.hpp — writer builds the reduce scalars at start; num_stats==1 for RMS; Noc object available for async_write_zeros.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — c_11 (trans_mat) is always created (bf16, 1 tile) on the worker cores; worker writer used iff use_mux == packed_ag_enabled; default compute config HiFi4, approx=true, fp32 dest.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h, llk_math_eltwise_binary.h — HiFi4 matmul = 64 MVMULs/tile in one MOP; HiFi4 ELWMUL = 32 ELWMULs in 4 MOP runs; dest-reuse semantics.
- tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_reduce.h — REDUCE_COL SUM: 4 passes of 8 loads + tree add + SFPTRANSP + store to row 0 (~100 cycles).
- tt_metal/hw/inc/api/compute/{compute_kernel_api.h, eltwise_binary_sfpu.h, tile_move_copy.h, matmul.h}, api/dataflow/noc.h — sfpu_reduce, mul_binary_tile, copy_init/copy_tile, MM_THROTTLE default 0, async_write_zeros on a CircularBuffer.

## Nodes consulted
- All 47 nodes (proposal/reflection) via /tmp dump; key ones:
- r02-b02-a02 — x*x^T diagonal matmul PRE: stat ~0.35-0.5 µs earlier at the slowest core vs the ones*S^T path, lost to a 0.36 µs BRISC diagonal gather.
- r02-b02-a03 — today's ones*S^T path; S round trip is the PRE tail.
- r03-b03-a02 — transpose_dest + SFPU col-sum on S: neutral (fp32 transpose_dest cost); sfpu_reduce<SUM,Float32,REDUCE_COL> works in this kernel.
- r04-b01-a02/a03 — HiFi2 PRE (now forbidden) showed the PRE tail maps 1:1 to kernel end; pre.py for measuring it.
- r04-b04-a02 — parent (round root), writer stack; r04-b04-a03 — launch-skew caveat for per-shape comparisons.

## Docs / external references
- none beyond the tree.
