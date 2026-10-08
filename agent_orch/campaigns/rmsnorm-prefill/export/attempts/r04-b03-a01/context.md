## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: the rules, metric, allowed paths.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: push_stick does write barrier + atomic barrier per push; the gamma loop may call it early.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp (PRE section): considered a lower-fidelity PRE x*x, but deferred.
- ccl/dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp: F_COLLECT waits on fwd_arrival_sem for all present workers, then the fabric send. It reads the packet buffer only after the count, so payload-before-inc ordering is all it needs.
- tt_metal/hw/inc/api/semaphore.h: `up(noc,x,y,val,vc=NOC_UNICAST_WRITE_VC)`.
- tt_metal/hw/inc/api/dataflow/noc.h: async_write uses NOC_UNICAST_WRITE_VC by default, and async_writes_flushed works in the DM_DYNAMIC_NOC mode.
- tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp: the fused unicast write + atomic inc does flush, then inc on the same noc/vc. That is the ordering precedent.
## Nodes consulted
- all r01-r03 reflections, through history.md and git show. Most relevant:
- r03-b01-a03: W_PUSH = 0.63 µs even with the stat ready, and its reflection #1 proposes this change.
- r03-b02-a02 (round root): the go -> stick read is not memory-latency bound, and the AG fabric section is unchanged.
- r03-b04-a03: gamma streaming, the likely obvious port for a sibling; avoided to keep the branches diverse.
- r03-b02-a03, r03-b03-a03: the drain cmd-buf/VC tricks failed; the drain is per-core bound.
