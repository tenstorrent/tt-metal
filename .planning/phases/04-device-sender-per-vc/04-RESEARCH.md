# Phase 4: Device Sender Per-VC - Research

**Researched:** 2026-03-13
**Domain:** Device-side RISC-V kernel — fabric erisc router sender channel indexing
**Confidence:** HIGH

## Summary

Phase 4 mirrors Phase 2 (Device Receiver Per-VC) but for the sender side. The device kernel
`fabric_erisc_router.cpp` already uses named VC constants for receiver channels
(`VC0_RECEIVER_CHANNEL = 0`, `VC1_RECEIVER_CHANNEL = 1`). The sender side has analogous
constants already defined in `fabric_erisc_router_ct_args.hpp` — specifically
`ACTUAL_VC0_SENDER_CHANNELS`, `ACTUAL_VC1_SENDER_CHANNELS`, `VC1_SENDER_CHANNEL_START`,
`MAX_NUM_SENDER_CHANNELS_VC0`, and `MAX_NUM_SENDER_CHANNELS_VC1` — but the kernel loop
already uses these where it matters for VC1 routing.

The remaining flat-index usages in the kernel are at the stream register initialization
section (lines ~2885–2891, ~2905–2906) and two `is_sender_channel_serviced[0]` guarded
code blocks that use channel index `0` as a literal rather than a named constant. The VC1
sender channel loops in `run_sender_channel_step` already use `ACTUAL_VC0_SENDER_CHANNELS`
as the VC1 start offset, which is correct. The `run_sender_channel_step` template itself
uses `is_sender_channel_serviced[sender_channel_index]` — a generic indexed access, not
a hardcoded constant — so it is already per-VC correct.

The CT args wire format does not need to change. All constants are already emitted correctly
by the host side. This phase is purely a cosmetic/semantic clarity refactor: replacing the
few remaining literal `[0]` indices used in sender-channel contexts with a named constant
(e.g. `VC0_SENDER_CHANNEL_START = 0`) to match the receiver pattern.

**Primary recommendation:** Define `VC0_SENDER_CHANNEL_START = 0` in `fabric_erisc_router_ct_args.hpp`
alongside `VC0_RECEIVER_CHANNEL`, then replace the small number of hardcoded `[0]` sender channel
literal indices in `fabric_erisc_router.cpp` with this constant.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DS-01 | Device-side sender channel arrays use per-VC indexing | Identified all flat-index sites; `ACTUAL_VC0_SENDER_CHANNELS`, `VC1_SENDER_CHANNEL_START` already in ct_args; VC1 loop calls already correct |
| DS-02 | CT args parsing for sender channels uses per-VC accessors | RT arg loops already use `MAX_NUM_SENDER_CHANNELS` flat; already semantically equivalent — no structural change needed; cosmetic rename targets identified |
</phase_requirements>

## Standard Stack

### Core
| File | Role | Notes |
|------|------|-------|
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` | Defines all CT args and channel constants | Where `VC0_SENDER_CHANNEL_START` constant should be added |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | Main device kernel loop | Contains all sender channel usages to update |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp` | 1-sender-channel fast path | Uses `RX_CH_TRID_STARTS[0]` — receiver index, not sender; no sender changes needed here |

No new libraries, headers, or dependencies are required.

## Architecture Patterns

### Existing Receiver Pattern (Template for This Phase)
The receiver channel pattern in the kernel uses named constants everywhere:

```cpp
// In fabric_erisc_router_ct_args.hpp (already exists):
constexpr size_t VC0_RECEIVER_CHANNEL = 0;
constexpr size_t VC1_RECEIVER_CHANNEL = 1;

// Usage in fabric_erisc_router.cpp:
if constexpr (is_receiver_channel_serviced[VC0_RECEIVER_CHANNEL]) { ... }
init_ptr_val<to_receiver_packets_sent_streams[VC0_RECEIVER_CHANNEL]>(0);
```

### Sender Constants Already in ct_args
The following are already defined (lines 82–84 of `fabric_erisc_router_ct_args.hpp`):

```cpp
constexpr size_t MAX_NUM_SENDER_CHANNELS_VC0 = (MAX_NUM_SENDER_CHANNELS >= 9) ? 5 : 4;
constexpr size_t MAX_NUM_SENDER_CHANNELS_VC1 = MAX_NUM_SENDER_CHANNELS - MAX_NUM_SENDER_CHANNELS_VC0;
constexpr size_t VC1_SENDER_CHANNEL_START = MAX_NUM_SENDER_CHANNELS_VC0;
```

And named CT args (lines 148–149):
```cpp
constexpr size_t ACTUAL_VC0_SENDER_CHANNELS = NAMED_CT_ARG("ACTUAL_VC0_SENDER_CHANNELS");
constexpr size_t ACTUAL_VC1_SENDER_CHANNELS = NAMED_CT_ARG("ACTUAL_VC1_SENDER_CHANNELS");
```

### Missing Constant to Add
A `VC0_SENDER_CHANNEL_START = 0` constant is absent. The receiver pattern has
`VC0_RECEIVER_CHANNEL = 0`. The parallel sender constant should be added near
`VC0_RECEIVER_CHANNEL` and `VC1_RECEIVER_CHANNEL` in `fabric_erisc_router_ct_args.hpp`.

### Recommended Project Structure
No structural change. One constant added to `fabric_erisc_router_ct_args.hpp`,
targeted substitutions in `fabric_erisc_router.cpp`.

### Anti-Patterns to Avoid
- **Do not change RT arg loop bounds:** The loops at lines ~2944–2951 and ~3019–3022
  use `MAX_NUM_SENDER_CHANNELS` as the flat bound. This is correct — the RT args wire
  format is flat and must not change. These loops are not the target of this phase.
- **Do not touch `super_speedy_mode` path sender logic:** The speedy path enforces
  `NUM_SENDER_CHANNELS == 1` via `static_assert`. The `RX_CH_TRID_STARTS[0]` usages
  in `fabric_erisc_router_speedy_path.hpp` are receiver-side, not sender-side; leave them.
- **Do not rename the VC1 loop offsets:** `ACTUAL_VC0_SENDER_CHANNELS` is already
  correct as the VC1 start offset in `run_sender_channel_step` calls; these are not flat indices.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| VC0 sender start index | Don't use literal `0` | `VC0_SENDER_CHANNEL_START` (add to ct_args) | Mirrors `VC0_RECEIVER_CHANNEL` pattern; makes VC membership explicit |
| VC1 start offset | Don't use `MAX_NUM_SENDER_CHANNELS_VC0` literal math | `VC1_SENDER_CHANNEL_START` (already defined) | Already exists in ct_args |

## Common Pitfalls

### Pitfall 1: Confusing "which calls need changing" vs "which are already correct"
**What goes wrong:** Planner tries to change the VC1 `run_sender_channel_step` calls
that already use `ACTUAL_VC0_SENDER_CHANNELS` as the channel index. These are already
per-VC correct — `ACTUAL_VC0_SENDER_CHANNELS` is the runtime VC1 channel start offset.
**Why it happens:** The template parameter is an absolute channel index, not a VC-relative
index, so seeing `ACTUAL_VC0_SENDER_CHANNELS` inside a VC1-labeled call looks odd.
**How to avoid:** The template `sender_channel_index` is the flat global index. VC0 calls
use 0–4, VC1 calls use `ACTUAL_VC0_SENDER_CHANNELS` through `ACTUAL_VC0_SENDER_CHANNELS+3`.
This is correct and must not change.

### Pitfall 2: Treating RT arg loops as targets
**What goes wrong:** Planner changes the flat `for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++)`
RT arg parsing loops to use per-VC loops or named constants.
**Why it happens:** The loops look "flat."
**How to avoid:** The RT arg wire format is flat (host emits flat arrays). DS-02 says
"CT args parsing uses per-VC accessors" — but the actual parsing loops already correctly
use `MAX_NUM_SENDER_CHANNELS` as the flat bound because the wire format has that many
entries. The per-VC aspect is in how the parsed values are subsequently *used*, not in
how they are *read*.

### Pitfall 3: Over-scoping to `to_sender_packets_acked_streams[0..3]`
**What goes wrong:** Changing `to_sender_packets_acked_streams[0]`, `[1]`, `[2]`, `[3]`
in the stream init section (lines 2885–2886, 2905–2906) to use per-VC constants.
**Why it happens:** These look like hardcoded sender channel indices.
**How to avoid:** These indices are sender channel flat indices (0=VC0 ch0, 1=VC0 ch1,
2=VC0 ch2, 3=VC0 ch3 — the first-level ack streams are only for VC0 channels, see
`to_sender_packets_acked_streams` comment: "VC1 does not use first level acks"). A
per-VC renaming of `[0]` and `[1]` would require loop unrolling that adds complexity
without clarity. This is borderline; the planner should assess whether adding
`VC0_SENDER_CHANNEL_START` as the first index makes semantic intent clearer. It is
LOW priority and does not block DS-01 or DS-02.

### Pitfall 4: CT args wire format breakage
**What goes wrong:** Adding or reordering CT args to try to group them by VC.
**Why it happens:** Misunderstanding DS-02 ("CT args parsing uses per-VC accessors").
**How to avoid:** DS-02 means the constants used to *interpret* the already-parsed args
should be VC-named constants. The wire encoding (positional arg order) must not change.

## Code Examples

### Flat `is_sender_channel_serviced[0]` uses to replace (HIGH confidence)
Source: `fabric_erisc_router.cpp`, verified by inspection

```cpp
// CURRENT (lines ~2870, ~2236, ~2244, ~2259, ~2521, ~2596, ~3323):
if constexpr (is_sender_channel_serviced[0]) { ... }

// PROPOSED — after adding VC0_SENDER_CHANNEL_START = 0:
if constexpr (is_sender_channel_serviced[VC0_SENDER_CHANNEL_START]) { ... }
```

Note: `is_sender_channel_serviced[0]` appears 6 times in the kernel.

### Stream init literal sender indices to replace (MEDIUM priority)
Source: `fabric_erisc_router.cpp` lines 2890–2891, verified by inspection

```cpp
// CURRENT:
init_ptr_val<sender_channel_free_slots_stream_ids[0]>(SENDER_NUM_BUFFERS_ARRAY[0]);  // LOCAL WORKER
init_ptr_val<sender_channel_free_slots_stream_ids[1]>(SENDER_NUM_BUFFERS_ARRAY[1]);  // Compact index 0

// PROPOSED (if VC0_SENDER_CHANNEL_START = 0 is defined):
init_ptr_val<sender_channel_free_slots_stream_ids[VC0_SENDER_CHANNEL_START]>(SENDER_NUM_BUFFERS_ARRAY[VC0_SENDER_CHANNEL_START]);
init_ptr_val<sender_channel_free_slots_stream_ids[VC0_SENDER_CHANNEL_START + 1]>(SENDER_NUM_BUFFERS_ARRAY[VC0_SENDER_CHANNEL_START + 1]);
```

However, `[0]` is clearly labeled "LOCAL WORKER" and `[1]` is "Compact index 0" — these are
semantic roles within VC0. The planner may decide to leave these with their explanatory comments.

### New constant to add in `fabric_erisc_router_ct_args.hpp`
```cpp
// Near VC0_RECEIVER_CHANNEL and VC1_RECEIVER_CHANNEL (around line 126):
constexpr size_t VC0_SENDER_CHANNEL_START = 0;
// Note: VC1_SENDER_CHANNEL_START is already defined above as MAX_NUM_SENDER_CHANNELS_VC0
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Flat `[0]` / `[1]` receiver channel indices | `VC0_RECEIVER_CHANNEL`, `VC1_RECEIVER_CHANNEL` constants | Phase 2 (already done) | Receiver pattern is the template for this phase |
| Flat `[0]` sender channel index in `is_sender_channel_serviced` | `VC0_SENDER_CHANNEL_START` (to be added) | Phase 4 (this phase) | Aligns sender pattern with receiver pattern |
| `ACTUAL_VC0_SENDER_CHANNELS` as VC1 start offset | Already correct — no change | Pre-existing | VC1 sender calls already use per-VC math |

**Already correct (do not change):**
- `run_sender_channel_step<VC1_RECEIVER_CHANNEL, ACTUAL_VC0_SENDER_CHANNELS, ...>` — uses VC1 receiver constant + correct VC1 channel offset
- `run_sender_channel_step<VC0_RECEIVER_CHANNEL, 0..4, ...>` — literal indices 0–4 are absolute, correct; VC0 context is set by the first template parameter
- RT arg parsing loops `for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++)` — flat, correct for wire format

## Open Questions

1. **Should `is_sender_channel_serviced[0]` uses all be updated, or only the subset in kernel_main?**
   - What we know: 6 call sites exist. Several are in the main loop (telemetry open/close guards,
     speedy path check). One is in the multi-txq setup block. One is in `kernel_main` initialization.
   - What's unclear: Whether the planner should batch all 6 or just the kernel_main / initialization uses.
   - Recommendation: Update all 6 for consistency. They are all the same mechanical change.

2. **Do the `to_sender_packets_acked_streams[0..3]` init lines need per-VC names?**
   - What we know: These are VC0-only (VC1 has no first-level acks), hardcoded 0/1 for 1D, 0/1/2/3 for 2D.
   - What's unclear: Whether adding `VC0_SENDER_CHANNEL_START` here adds clarity or noise.
   - Recommendation: LOW priority; if the planner adds the constant anyway, consider using it
     for index `[0]` only (the local-worker channel). Indices 1–3 are downstream-router channels
     whose identity is already clear from context and comments.

## Sources

### Primary (HIGH confidence)
- Direct inspection of `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` — all CT arg constants, existing VC sender/receiver constants, `is_sender_channel_serviced` array
- Direct inspection of `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` — all `is_sender_channel_serviced[0]` call sites (6 total), RT arg parsing loops, stream init section, `run_sender_channel_step` template calls (VC0 and VC1)
- Direct inspection of `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp` — confirmed no sender channel flat indices; `RX_CH_TRID_STARTS[0]` is receiver-side

### Secondary (MEDIUM confidence)
- Phase 3 SUMMARY files (03-01, 03-02) — confirmed that host-side sender changes are complete and what "per-VC" means in this project's convention
- `.planning/REQUIREMENTS.md` — DS-01, DS-02 scope confirmed

## Metadata

**Confidence breakdown:**
- Scope of changes: HIGH — all call sites directly verified by grep and line-by-line reading
- No CT args wire format risk: HIGH — RT arg loops use `MAX_NUM_SENDER_CHANNELS`, not affected
- VC1 sender calls already correct: HIGH — `ACTUAL_VC0_SENDER_CHANNELS` is the right offset
- New constant name: HIGH — `VC0_SENDER_CHANNEL_START = 0` mirrors `VC0_RECEIVER_CHANNEL` pattern exactly

**Research date:** 2026-03-13
**Valid until:** 2026-06-13 (stable — device kernel structure changes infrequently)
