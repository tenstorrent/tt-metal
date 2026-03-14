---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Completed 09-01-PLAN.md
last_updated: "2026-03-14T22:07:52.866Z"
progress:
  total_phases: 10
  completed_phases: 6
  total_plans: 11
  completed_plans: 9
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Each PR is self-contained, correct, and independently reviewable
**Current focus:** Phase 9 — Device-side kernel per-VC templates

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

**Phase 9: Device-Side Kernel Per-VC Templates — Plan 01 Complete (2026-03-14)**

Plan 01 completed:
- Added `MAX_NUM_VCS = 2` device-side constexpr and four per-VC foundation arrays to `fabric_erisc_router_ct_args.hpp`
- Added `extract_vc_sender_channels<T,VC,N,SRC_N>()` compile-time VC-slice helper template
- Added ten per-VC sender channel config arrays (_vc0/_vc1 variants for all five flat sender arrays)
- Added `is_sender_channel_serviced_vc<VC,CH>()` function template accessor for dead-code-eliminating per-channel access
- Added `SENDER_NUM_BUFFERS_ARRAY_VC0/VC1` inside `tt::tt_fabric` namespace
- Build: PASSED (zero new errors)

Key decisions:
- MAX_NUM_VCS defined locally on device side (no host header access from kernel)
- Per-VC arrays sized to MAX (not ACTUAL) for uniform per-VC channel indexing
- extract_vc_sender_channels placed at file scope (usable both inside and outside tt::tt_fabric)
- Flat arrays fully preserved (additive only); removal deferred to Plan 03

**Phase 8: Host-Side Per-VC Consolidation — Complete (2026-03-14)**

Plan 01 completed:
- Merged symmetric _vc0/_vc1 constants into per_vc arrays in fabric_builder_config
- Unified get_vc0/get_vc1_downstream_edm_count into get_downstream_edm_count_for_vc(vc, is_2D)
- Merged initialize_vc0_mappings/initialize_vc1_mappings into initialize_vc_mappings(vc)
- Updated all call sites including test files

Plan 02 completed:
- Converted _vc0/_vc1 locals in erisc_datamover_builder to per_vc arrays
- Replaced individual named_args assignments with fmt::format loop
- Added enable_first_level_ack_per_vc = {enable_first_level_ack, 0}
- Build: PASSED (zero new errors)
- Sanity test: PASSED (all 12 latency golden comparisons passed, no hangs)

Key decisions:
- Asymmetric constants kept standalone (no symmetric counterpart to merge)
- initialize_vc_mappings uses if/else branching for VC-specific logic
- fmt::format loop for per-VC named_args produces identical key strings for device compatibility

**Phase 7: Reorganize Buffer Slot Configs by VC — Plan 01 Complete (2026-03-14)**

Plan 01 completed:
- Replaced `PerVcBufferSlots` struct with `VcSlotConfig` array-of-struct indexed by VC
- All 3 static slot option tables converted to `std::array<VcSlotConfig, MAX_NUM_VCS>` format
- `get_optimal_num_slots_per_vc` returns `std::array<VcSlotConfig, MAX_NUM_VCS>` instead of writing 8 output refs
- `configure_buffer_slots_helper` returns `BufferSlotAllocation` struct instead of taking 4 output arrays
- Build: PASSED (zero new errors)
- Sanity test: PASSED (all 12 latency golden comparisons passed, no hangs)

Key decisions:
- VcSlotConfig uses `sender_slots`/`receiver_slots` field names for brevity
- BufferSlotAllocation returned by value; constructor uses reference aliases for minimal downstream diff
- Used VcSlotConfigArray type alias inside helper for table type readability

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

**Last session:** 2026-03-14T22:07:52.862Z
**Stopped at:** Completed 09-01-PLAN.md

## Next Plan

**Phase 9 Plan 01 complete — per-VC CT-arg infrastructure ready**
Per-VC constexpr arrays and template accessor in `fabric_erisc_router_ct_args.hpp`. Ready for Phase 9 Plan 02 (runtime array splitting and loop templating in `fabric_erisc_router.cpp`).
