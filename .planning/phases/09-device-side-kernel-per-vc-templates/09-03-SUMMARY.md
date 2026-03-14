---
phase: 09-device-side-kernel-per-vc-templates
plan: "03"
subsystem: fabric-router-kernel
tags: [constexpr, per-vc, template-metaprogramming, fabric, risc-v-kernel, std-tuple, std-get]

# Dependency graph
requires:
  - phase: 09-device-side-kernel-per-vc-templates
    plan: "02"
    provides: SenderFreeSlotsTuple/SenderConnectionEstablishedTuple/SenderFromReceiverCreditsTuple type aliases; run_sender_channel_step accepting per-VC sized arrays with VC-local indexing

provides:
  - execute_main_loop using std::get<VC>(tuple) per-VC array access for all run_sender_channel_step call sites
  - initialize_state_for_txq1_active_mode_sender_side converted to per-VC constexpr dispatch
  - edm_read_counter init using fold-expression over index_sequence instead of flat loop
  - Fixed SENDER_NUM_BUFFERS_ARRAY_VC0/VC1 constexpr out-of-bounds issue in RISC-V GCC 15.1.0

affects:
  - Phase 10 (downstream kernel changes, if any)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "std::get<VC>(SenderConnectionEstablishedTuple)[local_idx] for per-VC element access at call sites"
    - "VC1 run_sender_channel_step uses VC-local indices 0,1,2,3 (not global ACTUAL_VC0_SENDER_CHANNELS+N)"
    - "Padded MAX_NUM_SENDER_CHANNELS intermediary via constexpr function to safely slice per-VC arrays from actual-count arrays"

key-files:
  created: []
  modified:
    - tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
    - tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp

key-decisions:
  - "VC1 call sites change sender_channel_index from global (ACTUAL_VC0_SENDER_CHANNELS+N) to VC-local (0,1,2,3) to match run_sender_channel_step's VC-local parameter contract from Plan 02"
  - "initialize_state_for_txq1_active_mode_sender_side converted to per-VC constexpr dispatch with start+i flat memory indexing preserved for correct L1 layout"
  - "Sequential MAX_NUM_SENDER_CHANNELS runtime arg read loops (lines 3010, 3015, 3085) kept flat — plan specified keeping sequential RT arg reads flat since output arrays remain flat"
  - "SENDER_NUM_BUFFERS_ARRAY_VC0/VC1: use MAX_NUM_SENDER_CHANNELS-padded SENDER_NUM_BUFFERS_ARRAY_ALL as extract source to avoid constexpr OOB in RISC-V GCC 15.1.0 when NUM_SENDER_CHANNELS < MAX_NUM_SENDER_CHANNELS_VC0"

patterns-established:
  - "Per-VC call site pattern: pass std::get<0>(tuple) for VC0 calls, std::get<1>(tuple) for VC1 calls"
  - "Padded all_ intermediary pattern for constexpr-safe slicing when source array is actual-count-sized but target requires max-count"

requirements-completed: ["Phase 9 goal"]

# Metrics
duration: 30min
completed: 2026-03-14
---

# Phase 9 Plan 03: Execute Main Loop Per-VC Call Site Migration Summary

**Completed per-VC migration by converting all execute_main_loop run_sender_channel_step call sites from flat SenderFreeSlotsTuple[N] to std::get<VC>(tuple)[local_idx], and fixed a constexpr OOB in SENDER_NUM_BUFFERS_ARRAY_VC0/VC1 slicing that caused RISC-V kernel compile failure**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-03-14T22:20:00Z
- **Completed:** 2026-03-14T22:33:32Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Migrated all `run_sender_channel_step` VC0 call sites in `execute_main_loop` to pass `std::get<0>(channel_connection_established)`, `std::get<0>(local_sender_channel_free_slots_stream_ids)`, `std::get<0>(sender_channel_from_receiver_credits)` per-VC arrays
- Migrated all `run_sender_channel_step` VC1 call sites to use `std::get<1>(...)` and VC-local indices 0,1,2,3 instead of global `ACTUAL_VC0_SENDER_CHANNELS+N` indices
- Converted super_speedy_mode direct flat accesses `channel_connection_established[0]` etc. to `std::get<0>(...)[0]`
- Fixed pre-existing constexpr out-of-bounds issue in `SENDER_NUM_BUFFERS_ARRAY_VC0/VC1` slicing — previously used `NUM_SENDER_CHANNELS` as source array size causing OOB when `NUM_SENDER_CHANNELS < MAX_NUM_SENDER_CHANNELS_VC0`
- Sanity test: all 12 golden latency comparisons pass, no hangs

## Task Commits

1. **Task 1: Migrate execute_main_loop call sites to per-VC tuple dispatch** - `845583806a1` (feat)
2. **Task 2: Fix SENDER_NUM_BUFFERS_ARRAY_VC0/VC1 constexpr OOB bug** - `0daaa509e6b` (fix)

**Plan metadata:** (docs commit — created with state update below)

## Files Created/Modified

- `/home/snijjar/tt-metal/tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` — Migrated 8 run_sender_channel_step call sites + 3 super_speedy_mode direct accesses + 2 init loops to per-VC dispatch
- `/home/snijjar/tt-metal/tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` — Fixed SENDER_NUM_BUFFERS_ARRAY_VC0/VC1 slicing via MAX_NUM_SENDER_CHANNELS-padded intermediary

## Decisions Made

- VC1 call sites change `sender_channel_index` from global (`ACTUAL_VC0_SENDER_CHANNELS+N`) to VC-local (0,1,2,3) to match `run_sender_channel_step`'s VC-local parameter contract established in Plan 02.
- Sequential `MAX_NUM_SENDER_CHANNELS` runtime arg read loops kept flat (plan specified judgment: keep flat when output arrays remain flat).
- `SENDER_NUM_BUFFERS_ARRAY_VC0/VC1` fix: use `SENDER_NUM_BUFFERS_ARRAY_ALL` (MAX-padded) as slice source, consistent with how all other per-VC arrays use `MAX_NUM_SENDER_CHANNELS`-sized sources.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed SENDER_NUM_BUFFERS_ARRAY_VC0/VC1 constexpr out-of-bounds in RISC-V GCC 15.1.0**
- **Found during:** Task 2 (sanity test run)
- **Issue:** `extract_vc_sender_channels<size_t, 0, MAX_NUM_SENDER_CHANNELS_VC0, NUM_SENDER_CHANNELS>(SENDER_NUM_BUFFERS_ARRAY)` accessed indices 0..MAX_NUM_SENDER_CHANNELS_VC0-1 on an array of size NUM_SENDER_CHANNELS. In 1D fabric configs (NUM_SENDER_CHANNELS=1), this is out-of-bounds in constexpr. RISC-V GCC 15.1.0's `std::array::operator[]` calls non-constexpr `__glibcxx_assert_fail` on OOB access, causing kernel compile failure.
- **Fix:** Added `build_sender_num_buffers_all_()` constexpr function that pads `SENDER_NUM_BUFFERS_ARRAY` into a `MAX_NUM_SENDER_CHANNELS`-sized array. Per-VC arrays now slice from the padded `SENDER_NUM_BUFFERS_ARRAY_ALL`, using `MAX_NUM_SENDER_CHANNELS` as `SRC_N` (same pattern as all other per-VC arrays).
- **Files modified:** `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp`
- **Verification:** Kernel compiles, all 12 golden latency comparisons pass
- **Committed in:** `0daaa509e6b` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug)
**Impact on plan:** Necessary for correctness — RISC-V kernel failed to compile before fix. The bug was introduced in Plan 01 but only surfaced when the kernel was first exercised in Plan 03's sanity test.

## Issues Encountered

- Pre-existing Plan 01 bug in `SENDER_NUM_BUFFERS_ARRAY_VC0/VC1` slicing — was invisible during Plan 01/02 since no kernel compilation was triggered (plans left intentional compile errors). Caught and fixed in Task 2.

## Next Phase Readiness

- Full per-VC migration of sender channel iteration in `execute_main_loop` is complete. No flat cross-VC sender channel business logic loops remain.
- All flat arrays preserved (still have usages elsewhere). Removal candidates deferred to future cleanup phase.
- Build: zero errors. Sanity test: all 12 golden comparisons pass, no hangs.
- No blockers for Phase 10.

---
*Phase: 09-device-side-kernel-per-vc-templates*
*Completed: 2026-03-14*
