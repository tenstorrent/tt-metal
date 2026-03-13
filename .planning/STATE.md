# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Each PR is self-contained, correct, and independently reviewable
**Current focus:** Phase 3 — Host Sender Per-VC (Plan 01 complete)

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

**Phase 3: Host Sender Per-VC — Plan 02 Complete (2026-03-13)**

Plan 01 completed:
- Removed dead `AllocatorConstructionParams` struct from `fabric_builder_config.hpp` (flat-only, never instantiated)
- Added `MAX_RISC_CORES_PER_ETH_CHAN = 2` constant to `builder_config` namespace
- Changed `is_sender_channel_serviced_` and `is_receiver_channel_serviced_` outer dim to `MAX_RISC_CORES_PER_ETH_CHAN`
- Updated `num_used_sender_channels` comment to document it as derived from `num_used_sender_channels_per_vc`
- Build: PASSED (213/213 targets, zero errors)

Plan 02 completed:
- `compute_mesh_router_builder.cpp`: replaced flat `edm_config.num_used_sender_channels` with explicit per-VC sum `num_used_sender_channels_per_vc[0] + num_used_sender_channels_per_vc[1]`
- Build: PASSED (127/127 targets, zero errors)
- Sanity test: PASSED (all 12 latency tests passed golden comparison, no hangs)

Key decisions:
- `MAX_RISC_CORES_PER_ETH_CHAN` added as separate constant (not alias) from `MAX_NUM_VCS` to distinguish RISC core indexing from VC indexing
- `AllocatorConstructionParams` removal confirmed safe via grep — only definition existed, no users
- CT arg emission loops in `erisc_datamover_builder.cpp` left with flat guard — semantically equivalent, plan deferred

## Next Plan

**Phase 3 complete — all 2 plans done**
Ready for Phase 4 (host-sender-per-vc-logic) or next PR preparation.
