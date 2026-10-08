## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml: process, allowed paths, gate.
- device/kernels/dataflow/dit_rmsnorm_forwarder.cpp (parent's fork): gather_mcast path. F_FABRIC = fabric send +
  own-page mcast + out_ready wait + barriers. F_MCAST = 3 peer-page mcasts + go set_multicast + barrier, all after
  the last arrival.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp (grep): gather_mcast gating (BH, 1 forwarder,
  max_rounds == 1, ring <= 8), out_ready = caller GlobalSemaphore. No host change is needed.
- ttnn/.../ccl/common/kernels/minimal_ccl_common.hpp: fused_write_atomic_and_advance_local_read_address_for_fabric_write
  takes a uint32 val.
- tt_metal/fabric/fabric_edm_packet_header.hpp: NocUnicastAtomicIncFusedCommandHeader::val is uint32. The
  NOC_MULTICAST_WRITE fabric type is marked "mcast has bug", so fabric multicast into the worker CBs is ruled out.
- tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp: the EDM fused path does write, then
  flush_write_to_noc_pipeline, then noc_semaphore_inc on the same noc/vc. So the payload lands before the field bit.
- tt_metal/hw/inc/api/dataflow/dataflow_api.h: multicast loopback API; noc_semaphore_wait_min uses
  invalidate_l1_cache in its poll, which I copy.
## Nodes consulted
- All 39 nodes: proposal, score and reflection (dumped via git show of every tag).
- r04-b02-a01 (parent): F_MCAST 1.21 µs, constant; go 1.03/1.15 µs after F_FABRIC end; worker side go->comb 0.065 µs.
  Reflection #1 is this mechanism.
- r03-b02-a02 (grandparent): go 0.17/0.52 + 0.60 µs stick read; F_FABRIC median 2.4-2.7 µs vs min ~0.8-1.3 µs
  (cross-chip skew), which is why overlap should pay.
- r03-b04-a02: per-field values in out_ready via the 32-bit fused inc work on HW (16-bit per-wave fields).
- r02-b02-a04: one small set_multicast costs ~0.1 µs more than a unicast inc.
- r04-b03-a01 / r04-b04-a01 / r04-b01-a01: sibling wins (push handshake, posted drain, gamma streaming) live on other
  branches. Not combined here, to keep one mechanism.
## Docs / external references
- none
