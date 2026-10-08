## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — rules, allowed paths, metric.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — post-go stick read: 8 x 64 B reads of the
  ring_size gathered sticks from the stats scratch via a TensorAccessor (generic over DRAM/L1 interleaved).
- dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp — fabric fused write + atomic inc into
  page (my_device, forwarder, round) of the scratch, address via addrgen_detail::get_noc_address(accessor) (generic).
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — compute_sizing (4 pages x 4352 B for these shapes),
  make_stats_tensor_spec (the single source of the scratch's memory config), L1 margin (110 KB).
- device/dit_fused_distributed_rmsnorm_device_operation.cpp — validate compares the caller buffer's buffer type
  with the spec; compute_output_specs/create_output_tensors pass the caller's tensor through.
- dit_fused_distributed_rmsnorm.cpp — create_stats_buffer builds the tensor from make_stats_tensor_spec.
- tests/.../test_fused_rms_norm_prefill.py — allocates two stats buffers via create_stats_buffer (ping-pong).
- tt_metal/fabric/hw/inc/linear/addrgen_api.h — get_noc_address is accessor-generic (BH: no coord flip).
## Nodes consulted
- All 27 nodes (proposal + reflection). Key ones:
- r02-b02-a04 — measured go -> W_DRAIN start = 0.65-0.72 µs (stick DRAM read); multicast go + address pre-staging
  neutral; reflection #3 names an L1 landing buffer as the way to remove it.
- r03-b01-a01 (parent), r03-b02/b03/b04-a01 — the round's convergence on the combine SFPU cut; drain end follows the
  POST start 1:1, so fixed post-AG savings reach the kernel end.
- r02-b02-a01 — cross-call coupling: shorter / tighter calls also shrink the next call's start skew.
## Docs / external references
- none beyond the code.
