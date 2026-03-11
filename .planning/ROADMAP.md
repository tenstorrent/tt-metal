# Roadmap: Fabric Auto-Packetization

## Phase 1: fabric-auto-packetization

**Goal:** Enable all fabric write APIs in `tt_metal/fabric/hw/inc/linear/api.h` and `tt_metal/fabric/hw/inc/mesh/api.h` to transparently handle payloads larger than `FABRIC_MAX_PACKET_SIZE` by auto-packetizing under the hood. Existing single-packet APIs renamed `_single_packet`. New auto-packetizing wrappers keep original names.

**Requirements:** AP-01, AP-02, AP-03, AP-04, AP-05, AP-06

**Plans:** 7 plans

Plans:
- [ ] 01-01-PLAN.md — Wave 0: test infrastructure scaffolding (kernels + host runner + common types)
- [ ] 01-02-PLAN.md — linear/api.h: unicast + multicast unicast + sparse multicast renames and chunking wrappers
- [ ] 01-03-PLAN.md — linear/api.h: scatter + fused-scatter renames and chunking wrappers
- [ ] 01-04-PLAN.md — mesh/api.h: unicast + multicast unicast renames and chunking wrappers
- [ ] 01-05-PLAN.md — mesh/api.h: scatter + fused-scatter renames and chunking wrappers
- [ ] 01-06-PLAN.md — mesh/api.h: new addrgen overloads for multicast_fused_scatter_write_atomic_inc
- [ ] 01-07-PLAN.md — Integration test execution and hardware validation
