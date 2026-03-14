---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Completed 05-channel-allocator-01-PLAN.md
last_updated: "2026-03-14T00:07:53.488Z"
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 4
  completed_plans: 4
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Each PR is self-contained, correct, and independently reviewable
**Current focus:** Phase 5 — Channel Allocator

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
**Stopped at:** Completed 05-channel-allocator-01-PLAN.md

## Next Plan

**Phase 5 complete — all 1 plans done**
All 6 phases complete. Allocator API is fully per-VC across constructor + emit methods.
