---
phase: 09-device-side-kernel-per-vc-templates
plan: "02"
subsystem: fabric-router-kernel
tags: [constexpr, per-vc, template-metaprogramming, fabric, risc-v-kernel, std-tuple]

# Dependency graph
requires:
  - phase: 09-device-side-kernel-per-vc-templates
    plan: "01"
    provides: Per-VC constexpr arrays (SENDER_NUM_BUFFERS_ARRAY_VC0/VC1, is_sender_channel_serviced_vc0/vc1, vc_sender_channel_start_per_vc)
provides:
  - SenderFreeSlotsTuple, SenderConnectionEstablishedTuple, SenderFromReceiverCreditsTuple type aliases in fabric_erisc_router.cpp
  - any_sender_channels_active_vc<VC> per-VC template function
  - any_sender_channels_active(SenderFreeSlotsTuple) fold-expression dispatch wrapper
  - update_telemetry accepting SenderFreeSlotsTuple instead of flat array
  - run_sender_channel_step templated on per-VC sized arrays with VC-local sender_channel_index
  - populate_local_sender_channel_free_slots_stream_id_ordered_map working on SenderFreeSlotsTuple
  - wait_for_static_connection_to_ready accepting SenderFreeSlotsTuple
  - Per-VC tuple variables in run_fabric_edm_main_loop replacing flat arrays
affects:
  - 09-03 (will update remaining execute_main_loop call sites to pass std::get<VC>() per-VC arrays)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SenderFreeSlotsTuple = std::tuple<std::array<uint32_t, VC0_N>, std::array<uint32_t, VC1_N>> for per-VC runtime state"
    - "Fold expression over index_sequence for per-VC dispatch: (... || [&]<size_t VC>() { ... }.template operator()<VCs>())"
    - "Non-type template parameter const std::array<T, N>& for compile-time array binding in any_sender_channels_active_vc"
    - "VC-local sender_channel_index in run_sender_channel_step; global = vc_sender_channel_start_per_vc[VC] + local"

key-files:
  created: []
  modified:
    - tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp

key-decisions:
  - "any_sender_channels_active dispatches to per-VC template via fold expression over MAX_NUM_VCS index_sequence; no old flat-array overload kept since all call sites now pass SenderFreeSlotsTuple"
  - "run_sender_channel_step uses VC_RECEIVER_CHANNEL as VC index to compute global_sender_channel_index; call sites will migrate from flat to per-VC-local indices in Plan 03"
  - "SenderFromReceiverCreditsTuple VC1 elements initialized with global offset indices via placement new loop; VC0 uses existing init function"
  - "NUM_SENDER_CHANNELS template parameter removed from run_fabric_edm_main_loop (no longer deducible after flat-array param replaced with SenderFreeSlotsTuple)"
  - "wait_for_static_connection_to_ready updated with get_free_slots_stream_id lambda to translate global index to per-VC tuple access"
  - "Build will have compile errors in execute_main_loop call sites (channel_connection_established[i], local_sender_channel_free_slots_stream_ids[i]) - expected per plan; Plan 03 fixes remaining call sites"

patterns-established:
  - "Per-VC tuple pattern: use std::get<VC>(tuple)[vc_local_index] to access per-VC arrays; compute global index as vc_sender_channel_start_per_vc[VC] + vc_local_index for channel objects"
  - "SenderFreeSlotsTuple used as the canonical type for free-slots stream ID maps throughout the main loop and its helper functions"

requirements-completed: ["Phase 9 goal"]

# Metrics
duration: 15min
completed: 2026-03-14
---

# Phase 9 Plan 02: Per-VC Helper Function Templating and Runtime Array Splitting Summary

**Per-VC tuple type aliases and per-VC templated helper functions in fabric_erisc_router.cpp: any_sender_channels_active_vc<VC>, split flat sender channel runtime arrays into SenderFreeSlotsTuple/SenderConnectionEstablishedTuple/SenderFromReceiverCreditsTuple**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-14T22:07:52Z
- **Completed:** 2026-03-14T22:17:28Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `SenderFreeSlotsTuple`, `SenderConnectionEstablishedTuple`, and `SenderFromReceiverCreditsTuple` type aliases that split flat `NUM_SENDER_CHANNELS`-sized arrays into per-VC `std::tuple<std::array<T, VC0_N>, std::array<T, VC1_N>>` form
- Replaced `any_sender_channels_active` (flat array) with per-VC `any_sender_channels_active_vc<VC>` template plus a fold-expression dispatch wrapper accepting `SenderFreeSlotsTuple`; updated `update_telemetry` to accept the new type
- Split flat runtime array declarations in `run_fabric_edm_main_loop` and `kernel_main`; updated `run_sender_channel_step` to accept per-VC sized arrays with VC-local indexing and global channel index derivation via `vc_sender_channel_start_per_vc`; updated `populate_local_sender_channel_free_slots_stream_id_ordered_map` and `wait_for_static_connection_to_ready` to use per-VC tuple

## Task Commits

Both tasks committed together (Task 1 and 2 interleaved in same file edit):

1. **Task 1 + Task 2: Template helpers and split runtime arrays** - `cc9c7ff05da` (feat)

**Plan metadata:** (docs commit — created with state update below)

## Files Created/Modified

- `/home/snijjar/tt-metal/tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` — Added 101 lines net (type aliases, templated functions, per-VC declarations, helper function updates)

## Decisions Made

- `any_sender_channels_active` dispatcher uses a C++20 template lambda fold expression `(... || [&]<size_t VC>()` over `std::make_index_sequence<MAX_NUM_VCS>{}` — consistent with patterns already used in this file (lines 2072, 2736, 3118 etc.)
- `run_sender_channel_step`'s `sender_channel_index` parameter is now a VC-local index; global channel index computed as `vc_sender_channel_start_per_vc[VC_RECEIVER_CHANNEL] + sender_channel_index`. This is a breaking API change for call sites which Plan 03 will fix.
- `NUM_SENDER_CHANNELS` removed from `run_fabric_edm_main_loop` template parameters since its only use (flat array parameter type) was replaced with `SenderFreeSlotsTuple`.
- VC1 `sender_channel_from_receiver_credits` elements initialized with global offset `vc_sender_channel_start_per_vc[1] + i` via placement new, since both `SenderChannelFromReceiverCredits` types use global sender channel index to select stream registers.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Updated wait_for_static_connection_to_ready for SenderFreeSlotsTuple**
- **Found during:** Task 2 (runtime array splitting)
- **Issue:** `wait_for_static_connection_to_ready` also accepted `std::array<uint32_t, NUM_SENDER_CHANNELS>&` and accessed `local_sender_channel_free_slots_stream_ids[sender_channel_idx]` with a global index — would not compile after changing kernel_main's declaration to `SenderFreeSlotsTuple`
- **Fix:** Updated function signature to accept `SenderFreeSlotsTuple&`; added `get_free_slots_stream_id(global_idx)` lambda that dispatches to `std::get<0>()` or `std::get<1>()` based on whether `global_idx < MAX_NUM_SENDER_CHANNELS_VC0`
- **Files modified:** `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`
- **Committed in:** `cc9c7ff05da` (Task 1+2 commit)

---

**Total deviations:** 1 auto-fixed (missing helper function update)
**Impact on plan:** Necessary for correctness — `wait_for_static_connection_to_ready` is called with `local_sender_channel_free_slots_stream_ids` from `kernel_main`, which is now `SenderFreeSlotsTuple`.

## Issues Encountered

- Tasks 1 and 2 are committed together in a single commit (`cc9c7ff05da`) because Task 1's type aliases are immediately used by Task 2's declarations in the same file. The commit message reflects Task 1; Task 2 description is in this SUMMARY.
- As expected by the plan, `execute_main_loop` lambda call sites (e.g., `channel_connection_established[0]`, `local_sender_channel_free_slots_stream_ids[0]`, `sender_channel_from_receiver_credits[0]`) still use flat-array subscript syntax on what are now tuple variables. These will produce compile errors. Plan 03 migrates these call sites.

## Next Phase Readiness

- Per-VC tuple type aliases and updated helper functions are in place. Plan 03 can now update the `execute_main_loop` call sites to use `std::get<VC>(tuple)[local_idx]` access patterns.
- The structural migration is complete: all major functions accept per-VC typed parameters. Only the internal `execute_main_loop` usages remain to be updated.
- No blockers.

---
*Phase: 09-device-side-kernel-per-vc-templates*
*Completed: 2026-03-14*
