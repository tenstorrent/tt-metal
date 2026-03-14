---
phase: 03-host-sender-per-vc
plan: 01
subsystem: fabric-builder
tags: [fabric, erisc, builder, per-vc, cleanup, rename]

# Dependency graph
requires:
  - phase: 02-device-receiver-per-vc
    provides: "Device-side receiver channel per-VC indexing"
provides:
  - "Dead AllocatorConstructionParams struct removed from fabric_builder_config.hpp"
  - "MAX_RISC_CORES_PER_ETH_CHAN constant added to builder_config namespace"
  - "is_sender_channel_serviced_ and is_receiver_channel_serviced_ use explicit RISC core dimension"
  - "num_used_sender_channels documented as derived from num_used_sender_channels_per_vc"
affects: [04-host-sender-per-vc-logic, future-per-vc-phases]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MAX_RISC_CORES_PER_ETH_CHAN vs MAX_NUM_VCS: distinct constants for distinct semantic roles (even if equal by coincidence on current HW)"
    - "Arrays indexed by risc_id use MAX_RISC_CORES_PER_ETH_CHAN; arrays indexed by vc use MAX_NUM_VCS"

key-files:
  created: []
  modified:
    - tt_metal/fabric/builder/fabric_builder_config.hpp
    - tt_metal/fabric/erisc_datamover_builder.hpp

key-decisions:
  - "Add MAX_RISC_CORES_PER_ETH_CHAN = 2 constant alongside MAX_NUM_VCS = 2 with comment documenting their coincidental equality"
  - "Remove AllocatorConstructionParams without replacement — it was unused dead code with flat (non-per-VC) channel counts"

patterns-established:
  - "is_sender_channel_serviced_[risc_id][channel_id]: outer dim is MAX_RISC_CORES_PER_ETH_CHAN, not MAX_NUM_VCS"

requirements-completed: [HS-01, HS-02]

# Metrics
duration: 45min
completed: 2026-03-13
---

# Phase 3 Plan 01: Host Sender Per-VC Cleanup Summary

**Dead AllocatorConstructionParams struct removed and is_sender/receiver_channel_serviced_ outer dimension renamed from MAX_NUM_VCS to MAX_RISC_CORES_PER_ETH_CHAN to eliminate misleading coincidental aliasing**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-03-13T17:30:00Z
- **Completed:** 2026-03-13T18:15:45Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Removed 29-line dead `AllocatorConstructionParams` struct (flat sender/receiver channel counts, never instantiated outside its definition)
- Added `MAX_RISC_CORES_PER_ETH_CHAN = 2` constant with comment explaining it equals `MAX_NUM_VCS` only by coincidence on current BH/WH hardware
- Changed `is_sender_channel_serviced_` and `is_receiver_channel_serviced_` outer dimension type from `MAX_NUM_VCS` to `MAX_RISC_CORES_PER_ETH_CHAN` with clarifying comment
- Updated `num_used_sender_channels` inline comment to document it as a derived total, directing per-VC logic to `num_used_sender_channels_per_vc`
- Build verified clean: 213/213 targets, zero errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove dead AllocatorConstructionParams and fix is_sender/receiver_channel_serviced_ sizing** - `052a7a884e8` (refactor)
2. **Task 2: Build verification** - no files changed, build passed cleanly

**Plan metadata:** (docs commit to follow)

## Files Created/Modified
- `tt_metal/fabric/builder/fabric_builder_config.hpp` - Removed AllocatorConstructionParams struct; added MAX_RISC_CORES_PER_ETH_CHAN constant to builder_config namespace
- `tt_metal/fabric/erisc_datamover_builder.hpp` - Changed is_sender/receiver_channel_serviced_ outer dim to MAX_RISC_CORES_PER_ETH_CHAN; updated num_used_sender_channels comment

## Decisions Made
- Added `MAX_RISC_CORES_PER_ETH_CHAN = 2` as a separate constant (not an alias) because the two constants represent distinct concepts even though they happen to be equal on current hardware. The comment explicitly documents the coincidental equality to prevent future confusion.
- `AllocatorConstructionParams` was verified unused via grep before removal — only its definition appeared in the codebase.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None — clang-format reformatted the long array declaration onto two lines (standard pre-commit behavior), re-staged and committed cleanly.

## Self-Check

- `tt_metal/fabric/builder/fabric_builder_config.hpp` exists: FOUND
- `tt_metal/fabric/erisc_datamover_builder.hpp` exists: FOUND
- Task 1 commit `052a7a884e8`: FOUND (current HEAD)
- Build passed: 213/213 targets, zero errors

## Self-Check: PASSED

## Next Phase Readiness
- Phase 3 Plan 01 cleanup complete
- `is_sender_channel_serviced_` now uses the semantically correct `MAX_RISC_CORES_PER_ETH_CHAN` outer dimension
- `num_used_sender_channels_per_vc` is clearly documented as the canonical source for per-VC sender logic
- Ready for Phase 3 Plan 02: host-side sender per-VC logic changes

---
*Phase: 03-host-sender-per-vc*
*Completed: 2026-03-13*
