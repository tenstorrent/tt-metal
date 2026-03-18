---
phase: 06-vc2-sender-integration-end-to-end-verification
plan: 01
subsystem: firmware, testing
tags: [vc2, receiver-channel, trid-tracker, connection-dispatch, yaml-config]

# Dependency graph
requires:
  - phase: 04-builder-wiring-flow-control
    provides: VC2 sender dispatch block in firmware, CT arg emission
  - phase: 05-connection-api-testing
    provides: VC2 connection API (fabric_vc2_connection.hpp), use_vc2 field on ConnectionKey
provides:
  - VC2 receiver channel step in firmware main loop (packets sent on VC2 are now received)
  - vc_id field on sender configs parsed from YAML
  - VC2 connection dispatch in test infrastructure (append_fabric_vc2_connection_rt_args)
affects: [06-02, end-to-end-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "VC2 receiver follows VC1 pattern: channel pointers + trid tracker + conditional compilation"
    - "vc_id=2 implies use_vc2=true (single source of truth for VC selection)"

key-files:
  created: []
  modified:
    - tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
    - tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_common_types.hpp
    - tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_config.cpp
    - tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_device_setup.cpp

key-decisions:
  - "VC2 receiver uses VC0 downstream array as dummy (forwarding always disabled via DISABLE_RX_CH2_FORWARDING)"
  - "VC2 trid tracker added to teardown() for proper transaction ack drain on shutdown"
  - "vc_id=2 automatically sets use_vc2=true in resolve step (no ambiguity between fields)"

patterns-established:
  - "Receiver channel addition pattern: channel pointers init + trid tracker decl + template param + main loop step + teardown"

requirements-completed: [TEST-01, TEST-02]

# Metrics
duration: 5min
completed: 2026-03-18
---

# Phase 6 Plan 1: VC2 Sender Integration Summary

**VC2 receiver channel step in firmware main loop plus vc_id-based connection dispatch in test infrastructure**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-18T21:12:33Z
- **Completed:** 2026-03-18T21:17:35Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Firmware main loop now has a complete VC2 receiver channel step (packets sent on VC2 are actually received)
- Test sender configs carry vc_id field, parsed from YAML, propagated through all config paths
- VC2 senders with vc_id=2 now connect via the private VC2 connection API instead of hardcoded VC0

## Task Commits

Each task was committed atomically:

1. **Task 1: Add VC2 receiver step to firmware main loop** - `1a4fb143123` (feat)
2. **Task 2: Add vc_id to sender config and dispatch VC2 connection API** - `249cb33380f` (feat)

## Files Created/Modified
- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` - VC2 receiver channel step with trid tracker, channel pointers, teardown integration
- `tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_common_types.hpp` - vc_id field on ParsedSenderConfig and SenderConfig
- `tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_config.cpp` - vc_id YAML parsing, resolve propagation, split propagation, YAML output
- `tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_device_setup.cpp` - VC2 connection dispatch replacing TODO/hardcoded VC0

## Decisions Made
- VC2 receiver uses VC0 downstream array as dummy parameter since forwarding is always disabled (DISABLE_RX_CH2_FORWARDING=1)
- Added VC2 trid tracker to teardown() to ensure proper transaction ack drain on shutdown
- vc_id=2 automatically implies use_vc2=true in the resolve step, preventing ambiguity between the two fields

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added VC2 trid tracker to teardown function**
- **Found during:** Task 1 (VC2 receiver step)
- **Issue:** Plan did not mention updating teardown() but without it, pending VC2 receiver NOC transactions would not be acked before counter reset
- **Fix:** Added VC2 trid tracker parameter to teardown signature and body, guarded by FABRIC_2D_VC2_SERVICED
- **Files modified:** tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
- **Verification:** Build succeeds, pattern matches VC1 teardown handling
- **Committed in:** 1a4fb143123 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Essential for correctness -- without teardown integration, NOC counters could reset before VC2 transactions complete. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Firmware and test infrastructure are now fully wired for VC2 end-to-end
- Ready for Plan 06-02: end-to-end verification tests with VC2 YAML configs

---
*Phase: 06-vc2-sender-integration-end-to-end-verification*
*Completed: 2026-03-18*
