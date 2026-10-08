## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job, allowed paths, gate.
- kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE (DST-accumulated ELWMUL S, reduce, transpose), x*gamma, POST.
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — push_stick (two 64 B face-row writes; parent's diag gather).
- kernels/dataflow/dit_rmsnorm_scalar_setup.hpp + ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl — the SUM
  reduce scalar tile (c_4) is fp32, zero-filled, 1.0 in row 0 of each face: usable as the "ones" row for A * S^T.
- tt_metal/hw/inc/api/compute/matmul.h — matmul_tiles is C = A*B with transpose applying to B (in1); state_configure
  maps srcA=in1, srcB=in0.
- tt_metal/hw/inc/internal/tt-2xx/risc_common.h — invalidate_l1_cache on BH is just a fence (so the parent's gather
  cost is the 64 serialized L1 accesses, not the fences).
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — CB formats (pre_intermediate, stats_transposed_local
  fp32), fp32_dest_acc required.
## Nodes consulted
- All 21 nodes (proposal/score/reflection). Key ones:
- r02-b02-a02 (parent) — diag-matmul stat works (stat ready -0.35..0.6 µs) but BRISC gather +0.36 µs; reflection #1 is this.
- r02-b02-a01 (grandparent, best 1.2385) — dual-NoC drain; code base reverted to for writer/factory.
- r01-b04-a04 / r01-b01-a04 — PRE tail is the fixed reduce+transpose chain, not per-tile cost.
- r02-b01/b03/b04, r01-b0x dual-NoC/placement nodes — drain/placement variants; not repeated here.
## Docs / external references
- none beyond the in-tree headers above.
