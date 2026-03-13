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

**Phase 3: Host Sender Per-VC — Plan 01 Complete (2026-03-13)**

Plan 01 completed:
- Removed dead `AllocatorConstructionParams` struct from `fabric_builder_config.hpp` (flat-only, never instantiated)
- Added `MAX_RISC_CORES_PER_ETH_CHAN = 2` constant to `builder_config` namespace
- Changed `is_sender_channel_serviced_` and `is_receiver_channel_serviced_` outer dim to `MAX_RISC_CORES_PER_ETH_CHAN`
- Updated `num_used_sender_channels` comment to document it as derived from `num_used_sender_channels_per_vc`
- Build: PASSED (213/213 targets, zero errors)

Key decisions:
- `MAX_RISC_CORES_PER_ETH_CHAN` added as separate constant (not alias) from `MAX_NUM_VCS` to distinguish RISC core indexing from VC indexing
- `AllocatorConstructionParams` removal confirmed safe via grep — only definition existed, no users

## Next Plan

**Phase 3 Plan 02: Host Sender Per-VC Logic**
Migrate host-side sender channel arrays to per-VC indexing.
Key files: `erisc_datamover_builder.hpp/cpp`, `fabric_builder_context.hpp/cpp`
