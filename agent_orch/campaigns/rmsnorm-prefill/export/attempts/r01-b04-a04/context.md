## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job, metric, allowed paths
- kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE: per-tile mul_tiles + pack_tile<true> with L1 accumulation
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — BRISC gamma read (parent) is serial before W_PUSH
- tt_metal/hw/inc/api/compute/eltwise_binary.h, tt-llk blackhole llk_math_eltwise_binary.h — ELWMUL MOP hardcodes
  acc_to_dest=0, HiFi phases rely on Dst accumulation, no ZEROACC in the execute path
- tt_metal/hw/inc/api/dataflow/circular_buffer.h — pages_available_at_front() for non-blocking stick poll
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — fp32_dest_acc_en is required; pre_intermediate fp32
## Nodes consulted
- all 12 nodes (proposal + reflection); key: r01-b04-a03 (parent, zone timeline: PRE tail 1.3-1.9 µs, gamma ends
  close to PRE), r01-b02-a02/a03 (PRE ~125 ns/tile gates the AG start), r01-b01-a03 (same suggestion)
## Docs / external references
- tt-isa-documentation (DeepWiki): ELWMUL always accumulates onto Dst (Dst += SrcA*SrcB), no AddDst bit
