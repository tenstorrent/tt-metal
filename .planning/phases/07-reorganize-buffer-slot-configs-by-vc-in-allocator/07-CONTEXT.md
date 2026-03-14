# Phase 7: Reorganize buffer slot configs by VC in allocator - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Restructure the buffer slot configuration internals in `FabricStaticSizedChannelsAllocator` to use per-VC indexing, replacing hardcoded vc0/vc1 named fields with array-based per-VC types. This completes the per-VC refactor started in phases 1-6 by bringing the allocator's internal slot configuration logic in line with the per-VC array pattern used everywhere else. CT args wire format must remain unchanged.

</domain>

<decisions>
## Implementation Decisions

### Helper function return type
- `get_optimal_num_slots_per_vc` should return a struct instead of writing to 8 output reference scalars
- Create a new `VcSlotConfig` struct with `{sender_slots, receiver_slots}` per VC
- Return type: `std::array<VcSlotConfig, MAX_NUM_VCS>`
- Keep `PerVcBufferSlots` naming separate — `VcSlotConfig` is the new per-VC type

### Static lookup table format
- Convert all 3 static tables (`mesh_buffer_slot_options`, `other_buffer_slot_options`, `default_with_tensix_buffer_slot_options`) to use `std::array<VcSlotConfig, MAX_NUM_VCS>` format
- Remove `PerVcBufferSlots` struct entirely — it is replaced by the array-of-VcSlotConfig representation
- Full consistency: tables and return type use the same per-VC array pattern

### configure_buffer_slots_helper signature
- Also restructure the helper's own signature — collapse the 4 separate 2D output arrays into a single per-VC struct return or fewer params
- This is a bigger refactor but creates a cleaner boundary between slot selection and slot application

### Claude's Discretion
- Exact struct field naming for VcSlotConfig (e.g., `sender_slots`/`receiver_slots` vs `num_sender_buffer_slots`/`num_receiver_buffer_slots`)
- Whether configure_buffer_slots_helper returns a value or takes a single output struct
- How the .fill() application logic at the end of the function maps from the new return type to the existing per-VC member arrays

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches as long as the per-VC array pattern matches phases 1-6.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `builder_config::MAX_NUM_VCS` constant (value 2) — used for all per-VC array sizing
- Existing per-VC array members in `FabricStaticSizedChannelsAllocator` (`sender_channels_num_buffers[vc][ch]`, `receiver_channel_num_buffers[vc]`, etc.) — already correct pattern

### Established Patterns
- Per-VC arrays use `std::array<T, builder_config::MAX_NUM_VCS>` throughout phases 1-6
- Receiver channels are per-VC scalars (one receiver per VC), senders are per-VC × per-channel 2D arrays

### Integration Points
- `configure_buffer_slots_helper` is called once from the constructor — single call site
- `get_optimal_num_slots_per_vc` lambda has 3 call sites (MUX mode, default mode, and implicitly through get_num_buffer_slots)
- Member variables populated by the helper are consumed by `emit_ct_args` and `emit_channel_allocations_ct_args` — wire format must not change

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-reorganize-buffer-slot-configs-by-vc-in-allocator*
*Context gathered: 2026-03-14*
