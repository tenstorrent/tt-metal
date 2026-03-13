---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-03-13T23:40:00Z"
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

**Phase 4: Device Sender Per-VC — Plan 01 Complete (2026-03-13)**

Plan 01 completed:
- Added `VC0_SENDER_CHANNEL_START = 0` to `fabric_erisc_router_ct_args.hpp` after `VC1_RECEIVER_CHANNEL`, completing sender/receiver naming symmetry
- Replaced all 5 `is_sender_channel_serviced[0]` literal guard expressions in `fabric_erisc_router.cpp` with `is_sender_channel_serviced[VC0_SENDER_CHANNEL_START]`
- Build: PASSED (zero new errors)
- Sanity test: PASSED (all 12 latency tests passed golden comparison, no hangs)

Key decisions:
- `VC0_SENDER_CHANNEL_START` placed adjacent to `VC0/VC1_RECEIVER_CHANNEL` block in `ct_args.hpp` to group all VC start index constants together
- `run_sender_channel_step` template call flat indices (0-4) left unchanged — they are absolute kernel slot indices, not VC-start guards
- RT arg parsing loops (`for i < MAX_NUM_SENDER_CHANNELS`) left unchanged — flat wire format intact (DS-02)

## Next Plan

**Phase 4 complete — all 1 plans done**
Ready for Phase 5 (Channel Allocator).
