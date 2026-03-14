# Phase 5: Channel Allocator - Research

**Researched:** 2026-03-13
**Domain:** C++ fabric channel allocator API — `FabricStaticSizedChannelsAllocator` and callers
**Confidence:** HIGH (all findings from direct source inspection)

---

## Summary

The `FabricStaticSizedChannelsAllocator` constructor and internal state are already fully per-VC.
The constructor takes `std::array<size_t, MAX_NUM_VCS> num_used_sender_channels_per_vc` and
`std::array<bool, MAX_NUM_VCS> is_receiver_channel_active_per_vc`, and all internal data
structures (`sender_channels_base_address`, `receiver_channel_base_address`, etc.) are indexed
`[vc][channel]` or `[vc]`. The internals were already migrated in prior phases.

The remaining Phase 5 work lives in one method: `emit_channel_allocations_ct_args`. Its signature
takes three flat scalars — `num_used_vc0_sender_channels`, `num_used_vc1_sender_channels`, and
`num_used_receiver_channels` — rather than per-VC arrays. These three scalars are derived from
flat counts (`config.num_used_sender_channels_per_vc[0/1]` and `config.num_used_receiver_channels`)
at the call site in `erisc_datamover_builder.cpp`. Replacing them with per-VC arrays would make the
API consistent throughout.

The `FabricRemoteChannelsAllocator::emit_channel_allocations_ct_args` has a parallel signature
issue: it takes `num_used_receiver_channels` as a flat scalar. This should also be replaced with
`is_receiver_channel_active_per_vc` (the bool array) so the remote allocator is consistent.

**Primary recommendation:** Change both `emit_channel_allocations_ct_args` signatures to accept
per-VC arrays instead of flat scalar counts. Update the single call site in
`erisc_datamover_builder.cpp` accordingly.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CA-01 | Channel allocator uses both per-VC receiver and sender channel data | Both sender and receiver params are already per-VC in constructor; the gap is in `emit_channel_allocations_ct_args` which still takes flat scalars |
| CA-02 | Allocator API is consistent — no mixed flat/per-VC indexing | Legacy flat-total getter methods exist on the class; `emit_channel_allocations_ct_args` param list mixes vc0/vc1 scalars with a flat receiver count |
</phase_requirements>

---

## Standard Stack

Not applicable — pure C++ refactor within existing codebase, no new libraries.

---

## Architecture Patterns

### Current API (what exists today)

**`FabricStaticSizedChannelsAllocator` constructor — already per-VC:**
```cpp
// Source: tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp:31
FabricStaticSizedChannelsAllocator(
    tt::tt_fabric::Topology topology,
    const FabricEriscDatamoverOptions& options,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& num_used_sender_channels_per_vc,
    const std::array<bool, builder_config::MAX_NUM_VCS>& is_receiver_channel_active_per_vc,
    size_t channel_buffer_size_bytes,
    size_t available_channel_buffering_space,
    const std::vector<MemoryRegion>& memory_regions);
```

**`emit_channel_allocations_ct_args` — currently flat/mixed:**
```cpp
// Source: tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp:46
void emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    size_t num_used_vc0_sender_channels,   // flat: VC0 count
    size_t num_used_vc1_sender_channels,   // flat: VC1 count (separately!)
    size_t num_used_receiver_channels) const;  // flat: total receiver count
```

**Remote allocator — also flat:**
```cpp
// Source: tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp:59
void emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    size_t num_used_receiver_channels) const;  // flat scalar
```

**The call site in `erisc_datamover_builder.cpp:1258` (the only caller):**
```cpp
// Source: tt_metal/fabric/erisc_datamover_builder.cpp:1256
auto* static_alloc_ptr = dynamic_cast<FabricStaticSizedChannelsAllocator*>(config.channel_allocator.get());
TT_FATAL(static_alloc_ptr != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator");
static_alloc_ptr->emit_channel_allocations_ct_args(
    ct_args, actual_sender_channels_vc0, actual_sender_channels_vc1, num_receiver_channels);
// ...
config.remote_channels_allocator->emit_channel_allocations_ct_args(ct_args, num_receiver_channels);
```

Where `num_receiver_channels = config.num_used_receiver_channels` (flat total, line 905).

### Target API (per-VC)

**`emit_channel_allocations_ct_args` — after Phase 5:**
```cpp
void emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& num_used_sender_channels_per_vc,
    const std::array<bool, builder_config::MAX_NUM_VCS>& is_receiver_channel_active_per_vc) const;
```

**Remote allocator — after Phase 5:**
```cpp
void emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    const std::array<bool, builder_config::MAX_NUM_VCS>& is_receiver_channel_active_per_vc) const;
```

**Updated call site:**
```cpp
static_alloc_ptr->emit_channel_allocations_ct_args(
    ct_args,
    config.num_used_sender_channels_per_vc,
    config.is_receiver_channel_active_per_vc);

config.remote_channels_allocator->emit_channel_allocations_ct_args(
    ct_args,
    config.is_receiver_channel_active_per_vc);
```

### Internal logic change inside `emit_channel_allocations_ct_args`

Currently the method body uses the flat scalars to compute:
- `num_used_sender_channels = num_used_vc0_sender_channels + num_used_vc1_sender_channels`
- `num_unused_channels = total_sender_channels - num_used_sender_channels`
- The receiver index mapping iterates `for (size_t i = 0; i < num_used_receiver_channels; ++i)`

After change: derive those same values from the arrays:
```cpp
size_t num_used_vc0 = num_used_sender_channels_per_vc[0];
size_t num_used_vc1 = num_used_sender_channels_per_vc[1];
size_t num_used_sender_channels = num_used_vc0 + num_used_vc1;

size_t num_used_receiver_channels =
    static_cast<size_t>(is_receiver_channel_active_per_vc[0]) +
    static_cast<size_t>(is_receiver_channel_active_per_vc[1]);
```
The rest of the existing logic remains unchanged.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Summing booleans to get active receiver count | Custom bool-sum loop | Already established pattern: `static_cast<size_t>(arr[0]) + static_cast<size_t>(arr[1])` used in `get_num_receiver_channels()` and `FabricEriscDatamoverConfig` constructor |

---

## Common Pitfalls

### Pitfall 1: Legacy flat getter methods on the allocator

**What goes wrong:** `get_num_sender_channels()` (no vc arg) and `get_num_receiver_channels()` return flat totals (sum across VCs). They are marked "legacy" in comments but still callable. They are used indirectly inside `emit_channel_allocations_ct_args` at line 552-554 today.

**How to avoid:** After parameter change, ensure the internal body of `emit_channel_allocations_ct_args` derives totals from the new per-VC params (not from these legacy getters). The legacy getters can remain for backward compatibility with other callers (see below).

**Warning signs:** If `total_sender_channels` (used for `num_unused_channels` calculation) is still obtained from `get_num_sender_channels()` that calls the member array — that is fine and correct, since it represents the allocator's allocated total (which may exceed `num_used_sender_channels`).

### Pitfall 2: `num_used_receiver_channels` vs total allocated receiver channels

**What goes wrong:** There are two distinct counts: (a) the number of receiver channels the allocator _allocated memory for_ (derived from `is_receiver_channel_active_per_vc` at construction time), and (b) the number the router actually uses at CT arg emission time (passed in as parameter). In the existing code these are always equal. After the change they remain equal — the per-VC bool array already captures the active state.

**How to avoid:** Pass `config.is_receiver_channel_active_per_vc` directly from the caller — it is the canonical source for whether each VC's receiver channel is active.

### Pitfall 3: Remote allocator signature

**What goes wrong:** The remote allocator's `emit_channel_allocations_ct_args` also takes `num_used_receiver_channels` as a flat scalar. If only the static allocator's signature is updated and the remote allocator is left with a flat param, the API remains partially mixed.

**How to avoid:** Update both signatures in the same PR (same plan).

### Pitfall 4: Callers outside `erisc_datamover_builder.cpp`

**What goes wrong:** `emit_channel_allocations_ct_args` is called at exactly two locations (both in `erisc_datamover_builder.cpp`, lines 1258 and 1264). There are no other callers. However, the `FabricStaticSizedChannelsAllocator` class is `dynamic_cast` to in several other files.

**Files that `dynamic_cast` to `FabricStaticSizedChannelsAllocator`** (none call `emit_channel_allocations_ct_args`):
- `tt_metal/fabric/control_plane.cpp:2333`
- `tt_metal/fabric/fabric.cpp:223`
- `tt_metal/fabric/fabric_tensix_builder_impl.cpp:468`
- `tt_metal/fabric/fabric_mux_config.cpp:224,244`
- `tt_metal/fabric/erisc_datamover_builder.cpp:784,959,1241,1256,1502,1585`

These callers use getter methods (`get_receiver_channel_base_address`, `get_sender_channel_base_address`, `get_receiver_channel_number_of_slots`, `get_sender_channel_number_of_slots`, `get_num_sender_channels`) — none use `emit_channel_allocations_ct_args`. They are unaffected by the signature change.

---

## Code Examples

### What changes in `emit_channel_allocations_ct_args` (static allocator)

Old signature:
```cpp
// Source: tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp:46
void emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    size_t num_used_vc0_sender_channels,
    size_t num_used_vc1_sender_channels,
    size_t num_used_receiver_channels) const;
```

New signature:
```cpp
void emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& num_used_sender_channels_per_vc,
    const std::array<bool, builder_config::MAX_NUM_VCS>& is_receiver_channel_active_per_vc) const;
```

In the `.cpp` body, replace:
```cpp
size_t num_used_sender_channels = num_used_vc0_sender_channels + num_used_vc1_sender_channels;
```
with:
```cpp
size_t num_used_vc0_sender_channels = num_used_sender_channels_per_vc[0];
size_t num_used_vc1_sender_channels = num_used_sender_channels_per_vc[1];
size_t num_used_sender_channels = num_used_vc0_sender_channels + num_used_vc1_sender_channels;
size_t num_used_receiver_channels =
    static_cast<size_t>(is_receiver_channel_active_per_vc[0]) +
    static_cast<size_t>(is_receiver_channel_active_per_vc[1]);
```
All downstream logic in the method body is unchanged.

### What changes in `emit_channel_allocations_ct_args` (remote allocator)

Old:
```cpp
// Source: tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp:59
void emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    size_t num_used_receiver_channels) const;
```

New:
```cpp
void emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    const std::array<bool, builder_config::MAX_NUM_VCS>& is_receiver_channel_active_per_vc) const;
```

In the `.cpp` body, replace:
```cpp
for (size_t i = 0; i < num_used_receiver_channels; ++i) {
    ct_args.push_back(static_cast<uint32_t>(i));
}
```
with:
```cpp
size_t entry_index = 0;
for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
    if (is_receiver_channel_active_per_vc[vc]) {
        ct_args.push_back(static_cast<uint32_t>(entry_index));
        ++entry_index;
    }
}
```

### Updated call site (erisc_datamover_builder.cpp)

Replace lines 1258-1264:
```cpp
// OLD:
static_alloc_ptr->emit_channel_allocations_ct_args(
    ct_args, actual_sender_channels_vc0, actual_sender_channels_vc1, num_receiver_channels);
// ...
config.remote_channels_allocator->emit_channel_allocations_ct_args(ct_args, num_receiver_channels);

// NEW:
const std::array<size_t, builder_config::MAX_NUM_VCS> actual_sender_channels_per_vc = {
    actual_sender_channels_vc0, actual_sender_channels_vc1};
static_alloc_ptr->emit_channel_allocations_ct_args(
    ct_args, actual_sender_channels_per_vc, config.is_receiver_channel_active_per_vc);
// ...
config.remote_channels_allocator->emit_channel_allocations_ct_args(
    ct_args, config.is_receiver_channel_active_per_vc);
```

Note: `actual_sender_channels_vc0` and `actual_sender_channels_vc1` are still derived from
`actual_sender_channels_per_vc_` override (or `config.num_used_sender_channels_per_vc`) at lines
1016-1021. These local variables already exist and can be packed into the array directly.

---

## Open Questions

1. **Legacy `get_num_sender_channels()` and `get_num_receiver_channels()` on static allocator**
   - What we know: These return flat totals and are marked as "legacy" in comments. They are used inside `emit_channel_allocations_ct_args` (for `total_sender_channels` and `total_receiver_channels`).
   - What's unclear: Whether Phase 5 should also remove or deprecate these legacy getters, or leave them as-is for Phase 6 and other callers.
   - Recommendation: Leave them for now. Their usage inside `emit_channel_allocations_ct_args` is computing the _allocated total_ (not the used count), which is distinct from the passed-in used counts — keeping this separation is correct. Phase 6 can address stream register assignment which may need per-VC getters.

2. **`num_receiver_channels` local variable in `get_compile_time_args`**
   - What we know: Line 905 derives `num_receiver_channels = config.num_used_receiver_channels` (flat total), used at lines 1177-1195 for receiver NOC/cmd-buf named arg arrays, and previously passed to both `emit_channel_allocations_ct_args` calls.
   - What's unclear: After Phase 5 removes the receiver count param from `emit_channel_allocations_ct_args`, this variable is still used by the NOC/cmd-buf named arg loops. Those loops are flat-indexed by a flat channel ID `i` and guarded by `(i < num_receiver_channels)`.
   - Recommendation: The Phase 5 change does NOT touch the NOC/cmd-buf loops — `num_receiver_channels` stays for those guards. Only the `emit_channel_allocations_ct_args` call no longer uses it.

---

## Scope Summary

Phase 5 is a **narrow, 2-3 file change**:

| File | Change |
|------|--------|
| `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp` | Update `emit_channel_allocations_ct_args` declaration |
| `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp` | Update `emit_channel_allocations_ct_args` definition body |
| `tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp` | Update `emit_channel_allocations_ct_args` declaration |
| `tt_metal/fabric/builder/fabric_remote_channels_allocator.cpp` | Update `emit_channel_allocations_ct_args` definition body |
| `tt_metal/fabric/erisc_datamover_builder.cpp` | Update 2 call sites (lines 1258, 1264) |

No kernel changes. No CT args wire format changes. No changes to constructor signatures (already per-VC).

---

## Sources

### Primary (HIGH confidence)
- Direct source inspection of `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp` — constructor signature, method declarations, legacy getters
- Direct source inspection of `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp` — `emit_channel_allocations_ct_args` body
- Direct source inspection of `tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp` and `.cpp` — remote allocator API
- Direct source inspection of `tt_metal/fabric/erisc_datamover_builder.cpp` — call sites at lines 1256-1264, `num_receiver_channels` derivation at line 905
- `tt_metal/fabric/erisc_datamover_builder.hpp` — `FabricEriscDatamoverConfig` fields confirming `is_receiver_channel_active_per_vc` and `num_used_sender_channels_per_vc` are already per-VC

## Metadata

**Confidence breakdown:**
- Current API state: HIGH — read directly from source
- Callers / call sites: HIGH — grep confirmed single call site pair
- Proposed target API: HIGH — mechanical substitution of scalars with arrays following established patterns
- Scope of change: HIGH — confined to 5 files, 2 method signatures

**Research date:** 2026-03-13
**Valid until:** This research reflects exact source state on branch `snijjar/convert-flat-to-per-vc-indexing-receiver-channels` as of 2026-03-13. Valid until any of the 5 files listed above are modified.
