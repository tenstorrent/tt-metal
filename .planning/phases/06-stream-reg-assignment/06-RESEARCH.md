# Phase 6: Stream Reg Assignment - Research

**Researched:** 2026-03-14
**Domain:** Host-side stream register assignment for fabric router channels (VC-aware indexing)
**Confidence:** HIGH

## Summary

The fabric router assigns hardware stream registers to channels. On the host side, `StreamRegAssignments`
(in `erisc_datamover_builder.hpp`) holds a flat struct of `static constexpr uint32_t` values for every
stream ID in the system. The builder's `get_compile_time_args()` emits these IDs to the kernel as named
compile-time args using `StreamRegAssignments::get_all_stream_ids()`, which returns a flat `std::array`
indexed by sequential integer (0–32).

The problem: the comments inside `StreamRegAssignments` already label each constant with its VC ("VC0",
"VC1") but there is no type-level grouping by VC. The `to_sender_packets_acked_streams` array (device-side,
built in `fabric_erisc_router_ct_args.hpp`) is flat over `MAX_NUM_SENDER_CHANNELS` — VC0 occupies indices
[0–3], VC1 is padded with zeros at [4–7], with no per-VC sub-structure. Similarly,
`sender_channel_free_slots_stream_ids` (device-side, `fabric_erisc_router.cpp`) is a flat array over all
sender channels across both VCs. These are fed by the host emitting numbered named args
(`SENDER_CHANNEL_0_FREE_SLOTS_STREAM_ID` through `SENDER_CHANNEL_7_FREE_SLOTS_STREAM_ID`) from the flat
`get_all_stream_ids()` array.

The refactor goal for this phase is for the host-side `StreamRegAssignments` struct and its CT-arg
emission logic to express per-VC grouping so that stream register assignments for sender and receiver
channels are organized by VC rather than by flat sequential index.

**Primary recommendation:** Add per-VC accessor arrays or sub-structs to `StreamRegAssignments`; update the
CT-arg emission loop in `get_compile_time_args()` to reference them by VC, not by positional offset in a
flat all-streams array. Do NOT change the wire format (CT arg names and values must stay the same on
the device side).

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SR-01 | Host stream register assignment table uses per-VC indexing for both sender and receiver | `StreamRegAssignments` struct in `erisc_datamover_builder.hpp` needs per-VC grouping; CT-arg emission in `erisc_datamover_builder.cpp` consumes it |
| SR-02 | Stream register map correctly assigns registers per-VC | `to_sender_packets_acked_streams` and `sender_channel_free_slots_stream_ids` are flat; host-side emission must drive per-VC accessor paths |
</phase_requirements>

## Standard Stack

### Core

| Component | Location | Purpose | Current State |
|-----------|----------|---------|---------------|
| `StreamRegAssignments` struct | `tt_metal/fabric/erisc_datamover_builder.hpp` lines 109–205 | Flat table of all stream IDs | `static constexpr uint32_t` fields; `get_all_stream_ids()` returns flat `std::array<uint32_t, 33>` |
| CT-arg emission | `tt_metal/fabric/erisc_datamover_builder.cpp` lines 1027–1060 | Fills `named_args` map | Accesses flat array by positional offset (0–32) |
| Device CT-arg receivers | `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` lines 32–72 | One `constexpr uint32_t` per named arg | Unchanged by this phase — wire format stays fixed |

### Supporting

| Component | Location | Purpose | Note |
|-----------|----------|---------|------|
| `to_sender_packets_acked_streams` | `fabric_erisc_router_ct_args.hpp` lines 530–542 | Device-side flat array, VC0 in [0–3], VC1 zeros in [4–7] | Device side; do not change in this phase |
| `sender_channel_free_slots_stream_ids` | `fabric_erisc_router.cpp` lines 308–325 | Flat array over all sender channels | Device side; do not change |
| `vc_0_free_slots_stream_ids` / `vc_1_free_slots_stream_ids` | `fabric_erisc_router.cpp` lines 489–499 | Per-VC arrays (already VC-aware) | Already correct on device side |
| `fabric_tensix_builder_impl.cpp` | lines 971–978 | References specific `StreamRegAssignments` constants by name | Uses named constants directly; unaffected by refactor |

## Architecture Patterns

### Current Structure (Flat)

```
StreamRegAssignments {
    static constexpr uint32_t to_receiver_0_pkts_sent_id = 0;   // VC0
    static constexpr uint32_t to_receiver_1_pkts_sent_id = 1;   // VC1
    static constexpr uint32_t to_sender_0_pkts_acked_id  = 2;   // VC0 sender ch 0
    static constexpr uint32_t to_sender_1_pkts_acked_id  = 3;   // VC0 sender ch 1
    static constexpr uint32_t to_sender_2_pkts_acked_id  = 4;   // VC0 sender ch 2
    static constexpr uint32_t to_sender_3_pkts_acked_id  = 5;   // VC0 sender ch 3
    // [6–13]: to_sender_N_pkts_completed_id (VC0: 6–9, VC1: 10–13)
    // [14–17]: vc_0_free_slots_from_downstream_edge_N (VC0 receiver)
    // [18–21]: vc_1_free_slots_from_downstream_edge_N (VC1 receiver)
    // [22–29]: sender_channel_N_free_slots_stream_id (flat, VC0: 22–25, VC1: 26–29)
    ...

    static const auto& get_all_stream_ids();  // returns flat array[33]
}
```

CT-arg emission (builder.cpp, lines 1027–1060):
```cpp
const auto& stream_ids = StreamRegAssignments::get_all_stream_ids();
named_args["TO_RECEIVER_0_PKTS_SENT_ID"]          = stream_ids[0];   // VC0
named_args["TO_RECEIVER_1_PKTS_SENT_ID"]          = stream_ids[1];   // VC1
named_args["TO_SENDER_0_PKTS_ACKED_ID"]           = stream_ids[2];   // VC0 ch0
// ...all indexed by positional integer offset
named_args["SENDER_CHANNEL_0_FREE_SLOTS_STREAM_ID"] = stream_ids[22];
named_args["SENDER_CHANNEL_4_FREE_SLOTS_STREAM_ID"] = stream_ids[26]; // VC1 ch0
```

### Target Structure (Per-VC)

The goal is to express the per-VC grouping structurally. Two approaches are viable:

**Option A: Per-VC sub-arrays inside `StreamRegAssignments`**

Add arrays grouped by VC:
```cpp
struct StreamRegAssignments {
    // ... existing named constants unchanged ...

    // Per-VC receiver stream IDs: indexed by VC (0 or 1)
    static constexpr std::array<uint32_t, MAX_NUM_VCS> to_receiver_pkts_sent_ids = {
        to_receiver_0_pkts_sent_id,
        to_receiver_1_pkts_sent_id};

    // Per-VC sender acked stream IDs: indexed by [vc][vc_relative_channel]
    // VC0: channels 0-3, VC1: (first level acks not used, zeros)
    static constexpr std::array<std::array<uint32_t, MAX_SENDER_CHANNELS_PER_VC>, MAX_NUM_VCS>
        to_sender_pkts_acked_ids_per_vc = {{
            {to_sender_0_pkts_acked_id, to_sender_1_pkts_acked_id,
             to_sender_2_pkts_acked_id, to_sender_3_pkts_acked_id},  // VC0
            {0, 0, 0, 0}  // VC1 (not used)
        }};

    // Per-VC sender completed stream IDs: indexed by [vc][vc_relative_channel]
    static constexpr std::array<std::array<uint32_t, MAX_SENDER_CHANNELS_PER_VC>, MAX_NUM_VCS>
        to_sender_pkts_completed_ids_per_vc = {{
            {to_sender_0_pkts_completed_id, ..., to_sender_3_pkts_completed_id},  // VC0
            {to_sender_4_pkts_completed_id, ..., to_sender_7_pkts_completed_id}   // VC1
        }};

    // Per-VC sender free slots stream IDs: indexed by [vc][vc_relative_channel]
    static constexpr std::array<std::array<uint32_t, MAX_SENDER_CHANNELS_PER_VC>, MAX_NUM_VCS>
        sender_channel_free_slots_stream_ids_per_vc = {{
            {sender_channel_0_free_slots_stream_id, ..., sender_channel_3_free_slots_stream_id},  // VC0
            {sender_channel_4_free_slots_stream_id, ..., sender_channel_7_free_slots_stream_id}   // VC1
        }};
};
```

Then the CT-arg emission loop accesses by VC + channel-within-VC.

**Option B: Emit by VC loop in `get_compile_time_args()`**

Keep `StreamRegAssignments` struct largely unchanged but restructure the emission code to iterate over VCs
and channels-per-VC using the existing per-VC count fields (`num_used_sender_channels_per_vc`,
`is_receiver_channel_active_per_vc`). Still index the flat constants by name (no positional offset).

**Recommendation: Option A** — it makes the structural intent explicit at the struct level, which is the
spirit of SR-01/SR-02. The CT-arg names (wire format) do not change; only how they are populated changes.

### What Must NOT Change

The device-side `NAMED_CT_ARG("...")` lookups in `fabric_erisc_router_ct_args.hpp` are the wire format.
These string names must not be renamed. The numerical stream ID values (22–29 for sender free slots,
0–21 for receiver/acked/completed) must not change.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Mapping flat index → VC | Custom index arithmetic | Per-VC sub-arrays in `StreamRegAssignments` |
| Validation that per-VC counts match stream ID count | Runtime assert | Compile-time `static_assert` on array sizes |

## Common Pitfalls

### Pitfall 1: Changing CT arg names or values
**What goes wrong:** Renaming `"SENDER_CHANNEL_0_FREE_SLOTS_STREAM_ID"` or changing the value `22`
would break the wire format between host and device.
**How to avoid:** Only change how the host-side struct *groups* the constants and how emission code
*accesses* them. All `named_args["..."] = value` statements keep the same key string and same value.

### Pitfall 2: Off-by-one when splitting flat VC0/VC1 arrays
**What goes wrong:** VC0 sender channels are [0–3] (4 channels), VC1 are [4–7] in the flat array.
The VC1 acked streams are all zero (VC1 does not use first-level acks, see line 539 of ct_args.hpp).
Getting this split wrong produces wrong stream IDs for VC1 channels.
**How to avoid:** Cross-check with the existing comments in `StreamRegAssignments` which already label
each constant with its VC. The split: `VC0 sender completed_id` = `to_sender_{0,1,2,3}_pkts_completed_id`,
`VC1` = `to_sender_{4,5,6,7}_pkts_completed_id`.

### Pitfall 3: Touching device-side arrays
**What goes wrong:** `to_sender_packets_acked_streams` and `sender_channel_free_slots_stream_ids` in
`fabric_erisc_router_ct_args.hpp` and `fabric_erisc_router.cpp` are device-side kernel code. Changing
them is out of scope for this phase and would require re-testing all router paths.
**How to avoid:** Limit all changes to `erisc_datamover_builder.hpp` and `erisc_datamover_builder.cpp`.

### Pitfall 4: `get_all_stream_ids()` is still used by other code
**What goes wrong:** Removing `get_all_stream_ids()` breaks `erisc_datamover_builder.cpp` emission
and possibly other callers.
**How to avoid:** Keep `get_all_stream_ids()` in place (or keep the underlying flat constants).
The refactor adds per-VC grouping on top — it does not remove the existing flat interface until all
callers are updated.

### Pitfall 5: `fabric_tensix_builder_impl.cpp` references specific named constants
**What goes wrong:** `fabric_tensix_builder_impl.cpp` references
`StreamRegAssignments::sender_channel_1_free_slots_stream_id` directly (lines 971–978). These are
`static constexpr` fields that must remain accessible by name.
**How to avoid:** Do not remove any existing named `static constexpr` members from the struct.
Only add new per-VC groupings.

## Code Examples

### Current flat emission (from `erisc_datamover_builder.cpp` lines 1027–1060)

```cpp
// Source: tt_metal/fabric/erisc_datamover_builder.cpp
const auto& stream_ids = StreamRegAssignments::get_all_stream_ids();
named_args["TO_RECEIVER_0_PKTS_SENT_ID"] = stream_ids[0];
named_args["TO_RECEIVER_1_PKTS_SENT_ID"] = stream_ids[1];
named_args["TO_SENDER_0_PKTS_ACKED_ID"]  = stream_ids[2];
// ... 30 more positional accesses ...
named_args["SENDER_CHANNEL_4_FREE_SLOTS_STREAM_ID"] = stream_ids[26];  // VC1 ch0
```

### Target per-VC emission (illustrative — exact naming is planner's choice)

```cpp
// Receiver stream IDs: one per VC
for (size_t vc = 0; vc < MAX_NUM_VCS; vc++) {
    named_args[fmt::format("TO_RECEIVER_{}_PKTS_SENT_ID", vc)] =
        StreamRegAssignments::to_receiver_pkts_sent_ids[vc];
}

// Sender acked stream IDs: per VC, per channel within VC
// (VC0 only; VC1 uses 0 placeholders)
for (size_t ch = 0; ch < num_used_sender_channels_per_vc[0]; ch++) {
    named_args[fmt::format("TO_SENDER_{}_PKTS_ACKED_ID", ch)] =
        StreamRegAssignments::to_sender_pkts_acked_ids_per_vc[0][ch];
}

// Sender free slots: per VC, flat sender channel index (VC0 offset 0, VC1 offset 4)
size_t flat_idx = 0;
for (size_t vc = 0; vc < MAX_NUM_VCS; vc++) {
    for (size_t ch = 0; ch < num_used_sender_channels_per_vc[vc]; ch++) {
        named_args[fmt::format("SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID", flat_idx)] =
            StreamRegAssignments::sender_channel_free_slots_stream_ids_per_vc[vc][ch];
        flat_idx++;
    }
}
```

Note: The arg name strings use the flat sender channel index (0–7), not the VC-relative index. This
preserves the wire format while the indexing logic becomes VC-aware.

### Existing per-VC constants already in `StreamRegAssignments` (from `erisc_datamover_builder.hpp`)

```cpp
// Source: tt_metal/fabric/erisc_datamover_builder.hpp, lines 126-141
// VC0 receiver free slots (4 downstream edges)
static constexpr uint32_t vc_0_free_slots_from_downstream_edge_1 = 14;
static constexpr uint32_t vc_0_free_slots_from_downstream_edge_2 = 15;
static constexpr uint32_t vc_0_free_slots_from_downstream_edge_3 = 16;
static constexpr uint32_t vc_0_free_slots_from_downstream_edge_4 = 17;
// VC1 receiver free slots (4 downstream edges)
static constexpr uint32_t vc_1_free_slots_from_downstream_edge_1 = 18;
// ...
```

These are already VC-labeled by name but not grouped as arrays. The refactor would add arrays:
```cpp
static constexpr std::array<uint32_t, 4> vc_0_free_slots_stream_ids = {14, 15, 16, 17};
static constexpr std::array<uint32_t, 4> vc_1_free_slots_stream_ids = {18, 19, 20, 21};
```

## Key File Inventory

| File | Role | Change Needed |
|------|------|---------------|
| `tt_metal/fabric/erisc_datamover_builder.hpp` | `StreamRegAssignments` struct definition | Add per-VC grouping arrays |
| `tt_metal/fabric/erisc_datamover_builder.cpp` | CT-arg emission (`get_compile_time_args`) lines 1027–1060 | Update to use per-VC accessors |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` | Device CT-arg consumers | NO CHANGE (wire format) |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | Device kernel | NO CHANGE |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp` | Device flow control | NO CHANGE |
| `tt_metal/fabric/fabric_tensix_builder_impl.cpp` | Uses specific named constants | NO CHANGE (references named fields, not new arrays) |

## Scope Boundary

This phase covers exactly:

1. `StreamRegAssignments` struct in `erisc_datamover_builder.hpp` — add per-VC arrays
2. CT-arg emission in `erisc_datamover_builder.cpp` — use per-VC accessors

The device-side `to_sender_packets_acked_streams` (flat array with VC1 zeros) in
`fabric_erisc_router_ct_args.hpp` is a separate per-VC device refactor and is NOT in scope for
this host-only phase.

## Sources

### Primary (HIGH confidence)
- Direct source read: `tt_metal/fabric/erisc_datamover_builder.hpp` — full `StreamRegAssignments` struct
- Direct source read: `tt_metal/fabric/erisc_datamover_builder.cpp` lines 1027–1060 — CT-arg emission
- Direct source read: `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` lines 1–100, 500–560 — device-side CT-arg consumers and per-VC arrays
- Direct source read: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` lines 308–413, 489–499 — device-side stream ID arrays
- Direct source read: `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp` lines 56–75 — stream-register-based credit sender

## Metadata

**Confidence breakdown:**
- Struct anatomy (what fields exist, what values they hold): HIGH — read directly from source
- Emission code location and pattern: HIGH — read directly from source
- Wire format constraints (CT arg names/values): HIGH — read directly from device-side consumers
- Scope boundary (what is/isn't in this phase): HIGH — cross-checked against phase description and requirements

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable codebase; any change to `erisc_datamover_builder.hpp` stream ID values invalidates)
