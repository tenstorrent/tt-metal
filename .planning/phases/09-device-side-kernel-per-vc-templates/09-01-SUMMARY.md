---
phase: 09-device-side-kernel-per-vc-templates
plan: "01"
subsystem: fabric-router-kernel
tags: [constexpr, per-vc, template-metaprogramming, fabric, risc-v-kernel]

# Dependency graph
requires:
  - phase: 08-host-side-per-vc-consolidation
    provides: Host-side per-VC named_args emission (ACTUAL_VC0_SENDER_CHANNELS, ENABLE_FIRST_LEVEL_ACK_VC0/VC1, etc.)
provides:
  - MAX_NUM_VCS constexpr constant in device kernel header
  - MAX_NUM_SENDER_CHANNELS_PER_VC, vc_sender_channel_start_per_vc, ACTUAL_SENDER_CHANNELS_PER_VC, ENABLE_FIRST_LEVEL_ACK_PER_VC foundation arrays
  - extract_vc_sender_channels<T,VC,N,SRC_N>() compile-time slice helper template
  - Per-VC sender channel arrays: is_sender_channel_serviced_vc0/vc1, sender_ch_live_check_skip_vc0/vc1, sender_channel_is_traffic_injection_channel_vc0/vc1, sender_channel_ack_noc_ids_vc0/vc1, sender_channel_ack_cmd_buf_ids_vc0/vc1
  - is_sender_channel_serviced_vc<VC,CH>() function template accessor
  - SENDER_NUM_BUFFERS_ARRAY_VC0/VC1 inside tt::tt_fabric namespace
affects:
  - 09-02 (per-VC runtime array splitting and loop templating in fabric_erisc_router.cpp)
  - 09-03 (flat array removal after all consumers migrated)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "extract_vc_sender_channels<T,VC,N,SRC_N>() pattern for compile-time VC slicing from flat arrays"
    - "is_sender_channel_serviced_vc<VC,CH>() double-template accessor enabling dead code elimination per channel"
    - "MAX-sized per-VC arrays (not ACTUAL-sized) to allow uniform per-VC channel indexing"

key-files:
  created: []
  modified:
    - tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp

key-decisions:
  - "MAX_NUM_VCS defined locally in device kernel header as constexpr size_t = 2 (no host header available on device side)"
  - "Per-VC arrays sized to MAX_NUM_SENDER_CHANNELS_VC0/VC1 (not ACTUAL counts) so they can be indexed uniformly by per-VC channel index without ACTUAL as a template param"
  - "extract_vc_sender_channels helper placed before namespace tt::tt_fabric so it is usable both at file scope and inside the namespace"
  - "Flat arrays completely preserved — additive only per plan spec; removal deferred to Plan 03"
  - "sender_ch_live_check_skip_vc0/vc1 sliced from the MAX-sized all_ intermediary, not the NUM_SENDER_CHANNELS-sized final array, for consistent sizing"

patterns-established:
  - "Per-VC CT-arg arrays: define helper extract_vc_sender_channels, then define _vc0/_vc1 variants immediately after each flat array"
  - "Function template accessors with if constexpr (VC == 0) dispatch allow compiler to constant-fold both VC and channel index"

requirements-completed: ["Phase 9 goal"]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 9 Plan 01: Device-Side Per-VC Foundation Summary

**Per-VC constexpr arrays and extract_vc_sender_channels helper template added to fabric_erisc_router_ct_args.hpp, enabling dead-code-eliminating per-VC channel accessors for Plan 02 loop templating**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-14T22:03:58Z
- **Completed:** 2026-03-14T22:06:38Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `MAX_NUM_VCS = 2` device-side constexpr and four per-VC foundation arrays (`MAX_NUM_SENDER_CHANNELS_PER_VC`, `vc_sender_channel_start_per_vc`, `ACTUAL_SENDER_CHANNELS_PER_VC`, `ENABLE_FIRST_LEVEL_ACK_PER_VC`)
- Added `extract_vc_sender_channels<T,VC,N,SRC_N>()` compile-time VC-slice helper and ten per-VC sender channel config arrays covering all five flat sender channel config arrays
- Added `is_sender_channel_serviced_vc<VC,CH>()` function template accessor and `SENDER_NUM_BUFFERS_ARRAY_VC0/VC1` inside `tt::tt_fabric` namespace; build passes with zero new errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Add per-VC constexpr foundation constants** - `50693dbc768` (feat)
2. **Task 2: Add per-VC sender channel constexpr arrays built from flat arrays** - `59f467fad46` (feat)

**Plan metadata:** (docs commit — created with state update below)

## Files Created/Modified

- `/home/snijjar/tt-metal/tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` — Added 94 lines of per-VC constexpr infrastructure (MAX_NUM_VCS, foundation arrays, slice helper, per-VC arrays, template accessor)

## Decisions Made

- `MAX_NUM_VCS` defined locally as `constexpr size_t MAX_NUM_VCS = 2` because device kernel headers cannot include host-side `fabric_builder_config.hpp` where the host-side `builder_config::MAX_NUM_VCS` lives.
- Per-VC arrays sized to `MAX_NUM_SENDER_CHANNELS_VC0/VC1` (max capacity) rather than `ACTUAL_VCn_SENDER_CHANNELS` so the arrays can be indexed uniformly by per-VC channel index in templated functions without needing the actual counts as template parameters.
- `sender_ch_live_check_skip_vc0/vc1` sliced from the `MAX_NUM_SENDER_CHANNELS`-sized `sender_ch_live_check_skip_all_` intermediary (not the `NUM_SENDER_CHANNELS`-sized final array) to maintain `MAX`-sized per-VC arrays consistent with other per-VC arrays.

## Deviations from Plan

None — plan executed exactly as written. The only minor implementation choice was using `sender_ch_live_check_skip_all_` (MAX-sized) rather than `sender_ch_live_check_skip` (NUM-sized) as the source for per-VC slicing, which aligns with the plan's note to "Check each array's source size carefully."

## Issues Encountered

- Clang-format reformatted one long line in the `SENDER_NUM_BUFFERS_ARRAY_VC0/VC1` definitions on the first commit attempt. Re-staged and committed successfully after the hook applied its changes.

## Next Phase Readiness

- Per-VC CT-arg infrastructure is complete and buildable. Plan 02 can now use `is_sender_channel_serviced_vc<VC,CH>()`, `sender_ch_live_check_skip_vc0/vc1`, and the other per-VC arrays to template `fabric_erisc_router.cpp` sender channel loops on VC.
- No blockers. All flat arrays preserved for backward compatibility during migration.

---
*Phase: 09-device-side-kernel-per-vc-templates*
*Completed: 2026-03-14*
