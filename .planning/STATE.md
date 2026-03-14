---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Completed 06-stream-reg-assignment-01-PLAN.md
last_updated: "2026-03-14T00:45:00Z"
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 5
  completed_plans: 5
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Each PR is self-contained, correct, and independently reviewable
**Current focus:** Phase 6 — Stream Register Assignment

## Phase Validation Procedure

Every phase must pass before moving to the next:

**1. Build:**
```bash
./build_metal.sh -c -e --build-tests
```

**2. Sanity test (must NOT hang):**
```bash
build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
  --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml
```

**Emergency reset if test hangs:**
```bash
tt-smi -r
```

Note: Neither the build nor the sanity test should hang. Any hang is a regression introduced by this phase.

---

## Current Phase

**Phase 6: Stream Register Assignment — Plan 01 Complete (2026-03-14)**

Plan 01 completed:
- Added five per-VC grouping array members to `StreamRegAssignments` in `erisc_datamover_builder.hpp`
- Replaced all 33 positional `stream_ids[N]` accesses in the CT-arg emission block with named `StreamRegAssignments::*_per_vc` array references
- All existing named scalar constexpr members preserved; `get_all_stream_ids()` preserved
- Wire format preserved: CT arg name strings and numerical stream ID values unchanged
- Build: PASSED (zero new errors)
- Sanity test: PASSED (all 12 latency golden comparisons passed, no hangs)

Key decisions:
- Inner array dimension set to 4 (2D mesh max channels per VC) — z-router VC0 has 5 but uses named constants directly
- VC1 pkts-acked row is all-zero placeholders since VC1 has no first-level acks
- `get_all_stream_ids()` retained unchanged to preserve backward compatibility

**Phase 5: Channel Allocator — Plan 01 Complete (2026-03-14)**

Plan 01 completed:
- Updated `FabricStaticSizedChannelsAllocator::emit_channel_allocations_ct_args` to accept per-VC arrays instead of flat scalars
- Updated `FabricRemoteChannelsAllocator::emit_channel_allocations_ct_args` to accept per-VC bool array instead of flat count
- Updated single call site in `erisc_datamover_builder.cpp` to pass per-VC arrays
- Build: PASSED (zero new errors)
- Sanity test: PASSED (all 12 latency tests passed golden comparison, no hangs)

Key decisions:
- Derive local flat scalars at top of function body to keep existing downstream logic unchanged — minimizes diff surface
- Remote allocator iterates per-VC bool array for sequential entry index emission — semantically identical, now correctly scoped
- Retain `actual_sender_channels_vc0/vc1` and `num_receiver_channels` locals in builder (still used by NOC/cmd-buf loops above the emit calls)

## Session

**Last session:** 2026-03-14
**Stopped at:** Completed 06-stream-reg-assignment-01-PLAN.md

## Next Plan

**Phase 6 complete — all 1 plans done**
All planned per-VC refactor phases complete. StreamRegAssignments now expresses per-VC structure at the type level.
