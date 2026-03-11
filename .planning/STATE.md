# Project State

## Current Position
- **Phase:** 01-fabric-auto-packetization
- **Current Plan:** 08 (complete)
- **Status:** Phase complete (all 8 plans done)

## Progress
- Plan 01: COMPLETE (test infrastructure scaffolding)
- Plan 02: COMPLETE (linear/api.h unicast + multicast renames)
- Plan 03: COMPLETE (linear/api.h scatter + fused-scatter renames)
- Plan 04: COMPLETE (mesh/api.h unicast + multicast renames)
- Plan 05: COMPLETE (mesh/api.h scatter + fused-scatter renames)
- Plan 06: COMPLETE (mesh/api.h new addrgen overloads)
- Plan 07: COMPLETE (integration test runners + device kernels)
- Plan 08: COMPLETE (gap closure - compile-probe kernels for 7 missing wrapper families)

## Decisions
- Used (c) 2025 Tenstorrent AI ULC copyright for new test files
- Kernel stubs include both mesh/api.h and linear/api.h for comprehensive compile validation
- CompileOnlyKernels test assigns unicast to RISCV_0, multicast to RISCV_1
- Sparse multicast wrapper is passthrough (no chunking) due to pre-computed hop bitmask
- Fused op wrappers: intermediate chunks as regular writes, final chunk as fused with atomic_inc
- Connection manager wrappers use breadth-first: for_each_header inside chunk loop
- Mesh addrgen overload references updated to _single_packet (Rule 3 auto-fix for compilation)
- Mesh conn mgr wrappers use Pattern 5: route-setup for_each_header pass before chunk loop
- Scatter wrappers are passthrough (no chunking) because pre-computed NOC scatter addresses cannot be independently chunked
- Addrgen internal calls updated to _single_packet to avoid wrapper indirection
- Large-page scatter fallback delegates to existing unicast_write and fused_unicast_with_atomic_inc addrgen overloads
- _with_state scatter large-page path sets noc_send_type to NOC_UNICAST_WRITE before fallback
- Raw-size kernels pass NOC address components to device, compute 64-bit addr on-device via safe_get_noc_addr
- Reused rx_addrgen.cpp receiver kernel for completion signaling (protocol-agnostic semaphore wait)
- Used second Program objects in compile-only tests to work around 2-kernel-per-program RISCV limit
- Compile-probe kernels use dummy args for template instantiation (not runtime-correct)

## Blockers
None

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 01    | 01   | 2min     | 2     | 4     |
| 01    | 02   | 5min     | 2     | 1     |
| 01    | 04   | 6min     | 2     | 1     |
| 01    | 03   | 4min     | 2     | 1     |
| 01    | 05   | 6min     | 2     | 1     |
| 01    | 06   | 5min     | 2     | 1     |
| 01    | 07   | 7min     | 1     | 5     |
| 01    | 08   | 3min     | 2     | 4     |

## Last Session
- **Timestamp:** 2026-03-11T03:39:05Z
- **Stopped At:** Completed 01-08-PLAN.md (all phase 01 plans complete)
