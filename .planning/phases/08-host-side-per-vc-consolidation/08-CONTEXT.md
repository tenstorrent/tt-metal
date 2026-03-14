# Phase 8: Host-side per-VC consolidation - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Merge all remaining `_vc0`/`_vc1` named constants and split functions into `_per_vc` arrays and unified functions across host builder code. This covers `fabric_builder_config`, `erisc_datamover_builder`, `fabric_router_channel_mapping`, `fabric_tensix_builder`, and `router_connection_mapping`. CT args wire format must remain unchanged.

</domain>

<decisions>
## Implementation Decisions

### Downstream EDM constants
- Merge symmetric pair: `num_downstream_edms_2d_vc0`/`_vc1` → `num_downstream_edms_2d_per_vc = {3, 3}`
- Merge z-router sender counts: `num_sender_channels_z_router_vc0`/`_vc1` → `num_sender_channels_z_router_per_vc = {5, 4}`
- Keep asymmetric constants standalone: `num_downstream_edms_vc0 = 1` (no VC1 1D counterpart) and `num_downstream_edms_2d_vc1_with_z = 4` (no VC0 counterpart)
- Update derived constants (`num_sender_channels_z_router`, `num_downstream_edms_2d`) to sum from the arrays

### Downstream EDM count function
- Replace `get_vc0_downstream_edm_count()` and `get_vc1_downstream_edm_count()` with single `get_downstream_edm_count_for_vc(uint32_t vc, bool is_2D_routing)`
- Implementation uses switch on vc — centralizes the VC1-non-2D TT_FATAL guard
- The irregular vc/is_2D mapping makes a pure array lookup insufficient (VC1 has no 1D path)

### Channel mapping merge
- Merge `initialize_vc0_mappings()` and `initialize_vc1_mappings()` into single `initialize_vc_mappings(uint32_t vc)` called in a loop
- Use if/switch branching inside the function for VC-specific logic (VC0 is simpler, VC1 has intermesh/z-routing conditionals)
- Match PR approach for VC1 base channel offset computation

### Datamover builder locals and named_args
- Convert all `_vc0`/`_vc1` local variables to per-VC arrays (including asymmetric ones like `enable_first_level_ack_per_vc = {true, false}`)
- Use `fmt::format` loops for ALL named_args assignments — loop everything possible, even asymmetric values
- Use nested per-VC loops (outer: VC, inner: channels-per-VC) for stream ID named_args with offset computation, not flat 0-N loops

### Claude's Discretion
- Exact variable naming for per-VC locals in erisc_datamover_builder.cpp
- Whether `router_connection_mapping.cpp` hardcoded VC literals (0/1) need constants or stay as numeric (they're semantic routing decisions, not arbitrary)
- How `fabric_tensix_builder.cpp` call site adapts (likely just `get_downstream_edm_count_for_vc(0, is_2D)`)

</decisions>

<specifics>
## Specific Ideas

- Reference PR #39538 for target state — match its approach where decisions above don't override
- The `get_downstream_edm_count` (total, non-per-VC) function on line 97 likely stays as-is since it sums across VCs

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `builder_config::MAX_NUM_VCS` constant (value 2) — used for all per-VC array sizing throughout phases 1-7
- `std::array<T, builder_config::MAX_NUM_VCS>` — established pattern for per-VC arrays

### Established Patterns
- Per-VC arrays use `std::array<T, builder_config::MAX_NUM_VCS>` consistently
- Phase 6 already converted stream reg assignment arrays to per-VC grouping (`StreamRegAssignments::*_per_vc`)
- Phase 6 already uses per-VC loops for CT-arg emission in `get_compile_time_args`

### Integration Points
- `get_vc0_downstream_edm_count` / `get_vc1_downstream_edm_count` called from `erisc_datamover_builder.cpp` (~line 975) and `fabric_tensix_builder.cpp` (line 485)
- `initialize_vc0_mappings` / `initialize_vc1_mappings` called only from `initialize_mappings()` in the same file
- `num_sender_channels_z_router_vc0`/`_vc1` referenced in `fabric_router_channel_mapping.cpp` and `erisc_datamover_builder.cpp`
- Named args keys like `NUM_DOWNSTREAM_SENDERS_VC0` are read by device-side kernel — string names must not change

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-host-side-per-vc-consolidation*
*Context gathered: 2026-03-14*
