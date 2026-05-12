### Description
Remove unused loopback multicast helper APIs from `ttml` dataflow utilities that were introduced during earlier fused SwiGLU work but are no longer referenced by current kernels.

### Changes Made
- [x] **Refactor:** Removed dead loopback mcast helpers from `sources/ttml/metal/common/dataflow_utils.hpp`.
- [x] Kept `mcast_sender_signal_receivers_loopback` because it is used by `moe_group`.

### Testing
- [x] Codebase reference check (`rg`) confirms removed symbols have no call sites.
- [ ] Full build / runtime validation not run in this cleanup-only change.

### Additional Context
- Follow-up cleanup after:
  - `[tt-train] Semi-fused SwiGLU with elemwise-bw kernel, optimizations, and benchmark` (`#39969`, merged)
  - `[tt-train] Gate-up fused SwiGLU forward (2-step) + path choice` (`#38767`, closed)
  - `[tt-train] Fused SwiGLU forward + single-sender multicast, packer L1 accumulation, batched tile multicast` (`#34172`, merged)
