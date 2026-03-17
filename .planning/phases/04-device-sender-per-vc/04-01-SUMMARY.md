---
phase: 04-device-sender-per-vc
plan: 01
subsystem: fabric
tags: [fabric, edm, kernel, sender-channel, constexpr, naming]

# Dependency graph
requires:
  - phase: 03-host-sender-per-vc
    provides: host-side per-VC sender channel naming and sizing (is_sender_channel_serviced_, num_used_sender_channels_per_vc)
provides:
  - VC0_SENDER_CHANNEL_START = 0 constant in fabric_erisc_router_ct_args.hpp
  - All five is_sender_channel_serviced[VC0_SENDER_CHANNEL_START] guard expressions in fabric_erisc_router.cpp
affects: [05-channel-allocator, 06-stream-reg-assignment]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "VC0_SENDER_CHANNEL_START / VC1_SENDER_CHANNEL_START mirrors VC0_RECEIVER_CHANNEL / VC1_RECEIVER_CHANNEL pattern — all VC boundary indices are named constants"

key-files:
  created: []
  modified:
    - tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp
    - tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp

key-decisions:
  - "VC0_SENDER_CHANNEL_START placed adjacent to VC0/VC1_RECEIVER_CHANNEL block (after line 127) to group all VC start index constants together"
  - "RT arg parsing loops (for i < MAX_NUM_SENDER_CHANNELS) left unchanged — flat wire format, not per-VC guards"
  - "Only is_sender_channel_serviced[0] guard uses replaced; run_sender_channel_step template call literals (absolute flat indices 0-4) left as-is per plan spec"

patterns-established:
  - "Per-VC sender index naming: VC0_SENDER_CHANNEL_START = 0, VC1_SENDER_CHANNEL_START = MAX_NUM_SENDER_CHANNELS_VC0"

requirements-completed: [DS-01, DS-02]

# Metrics
duration: 6min
completed: 2026-03-13
---

# Phase 4 Plan 01: Device Sender Per-VC Summary

**VC0_SENDER_CHANNEL_START = 0 constant added to fabric_erisc_router_ct_args.hpp; all five is_sender_channel_serviced[0] opaque literals in fabric_erisc_router.cpp replaced with named constant**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-03-13T23:33:00Z
- **Completed:** 2026-03-13T23:38:52Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `VC0_SENDER_CHANNEL_START = 0` to `fabric_erisc_router_ct_args.hpp` adjacent to `VC0_RECEIVER_CHANNEL` / `VC1_RECEIVER_CHANNEL` constants, completing sender/receiver naming symmetry
- Replaced all 5 `is_sender_channel_serviced[0]` literal guard expressions in `fabric_erisc_router.cpp` with `is_sender_channel_serviced[VC0_SENDER_CHANNEL_START]`
- Build passed cleanly (zero new errors); sanity test passed all 12 latency golden comparisons with no hangs

## Task Commits

Each task was committed atomically:

1. **Task 1: Add VC0_SENDER_CHANNEL_START constant and replace is_sender_channel_serviced[0] literals** - `935addea397` (feat)
2. **Task 2: Build and sanity test** - no commit (build/test verification only, no file changes)

## Files Created/Modified
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` - Added `VC0_SENDER_CHANNEL_START = 0` constant after `VC1_RECEIVER_CHANNEL`
- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` - Replaced 5 `is_sender_channel_serviced[0]` with `is_sender_channel_serviced[VC0_SENDER_CHANNEL_START]`

## Decisions Made
- `VC0_SENDER_CHANNEL_START` placed immediately after `VC1_RECEIVER_CHANNEL` (line 127 insertion point) to group all VC start index constants together — mirrors how `VC1_SENDER_CHANNEL_START` sits near `MAX_NUM_SENDER_CHANNELS_VC0`
- `run_sender_channel_step` template call arguments (absolute flat indices 0, 1, 2, 3, 4) were NOT changed — they are flat kernel slot indices, not the same concept as the VC0-start guard checked by `is_sender_channel_serviced`
- RT arg parsing loops `for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++)` left unchanged — flat wire format, correct as-is per DS-02

## Additional [0] sites found vs. plan
Pre-identified 5 sites matched exactly: lines 2236, 2244, 2259, 2521, 2870. No additional occurrences discovered. Grep confirmed zero residual `is_sender_channel_serviced[0]` literals post-edit.

## Build Result
Build: PASSED. Kernel files installed cleanly to build artifacts. Zero new errors.

## Sanity Test Result
PASSED. All 12 latency tests passed golden comparison. No hangs. Board required one `tt-smi -r` reset prior to the test run due to a stale TLB lock from a prior unrelated process (error -12 on first attempt); second attempt succeeded immediately after reset.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
- First test run hit `tt_tlb_alloc failed with error code -12` — stale TLB lock held by dead process (TID 1343213). Resolved with `tt-smi -r 0,1,2,3` board reset. Not related to code changes.

## Next Phase Readiness
- Phase 4 complete (1/1 plans done)
- Phase 5 (Channel Allocator) can proceed: both sender and receiver VC naming constants are now in place
- No blockers

---
*Phase: 04-device-sender-per-vc*
*Completed: 2026-03-13*
