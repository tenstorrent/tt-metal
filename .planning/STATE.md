# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Each PR is self-contained, correct, and independently reviewable
**Current focus:** Phase 1 — Host Receiver Per-VC (In Progress)

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

Note: The sanity test does NOT hang on `main`. Any hang is a regression introduced by this phase.

---

## Current Phase

**Phase 1: Host Receiver Per-VC**

Partially complete. All allocator and builder host-side changes done:
- `FabricStaticSizedChannelsAllocator`, `FabricRemoteChannelsAllocator` — receiver arrays collapsed to per-VC scalars
- `FabricEriscDatamoverConfig` — `is_receiver_channel_active_per_vc` (bool)
- `FabricBuilderContext` — `max_receiver_channels_per_vc_` (bool)
- Dead builder fields removed

Remaining for Phase 1:
- Verify compilation
- Run tests to confirm no regressions

## Next Phase

**Phase 2: Device Receiver Per-VC**
Migrate device-side kernel code to per-VC receiver channel indexing.
Key files: `fabric_erisc_router_ct_args.hpp`, `fabric_erisc_datamover_channels.hpp`, `fabric_erisc_router.cpp`
