# X280 Profiler → Host Packet Pipeline — Design Decisions

**Status:** agreed design, pre-implementation (2026-07-03). Branch `mo/x280_tests`.
**Scope of this doc:** how profiler data travels from the Tensix cores, through the X280
"presenter," across the D2H socket, and into typed host-side callbacks. Captures the
architecture and the decisions behind it so future packet types (events, payload, sync,
program-zone) slot in without re-litigating the shape.

---

## Motivation

The X280 drains the per-RISC kernel-profiler SPSC rings and gets that data to the host.
The first cut wired the host receiver **directly** to the Tracy handler
(`tracy_handler_->PushDeviceMarker(...)`), bypassing the real-time profiler's existing
publish/subscribe path (`InvokeProgramRealtimeProfilerCallbacks` → registered callbacks →
`HandleRecord`). That's a bespoke shortcut: it couples the receiver to one consumer, and
any future consumer (CSV/perf-report sink, event handlers) would have to be bolted on
separately.

**Principle:** the X280 is only a *transport + presenter*. Once data is on the host, it
rides the same publish/subscribe path the RT profiler uses. Consumers are subscribers.

---

## Key facts this design rests on

- **64B is the universal flit / entry / page size.** `RT_PROFILER_ENTRY_SIZE = 64`
  (`realtime_profiler_ring_buffer.hpp`), commented "matches D2H socket page size"; the
  X280's `PAGE = 64`. One flit = one 64B page = one profiler packet, end to end.
- **A packet-type taxonomy already exists** (`hostdevcommon/.../profiler_common.h`):
  `enum PacketTypes { ZONE_START, ZONE_END, ZONE_TOTAL, TS_DATA, TS_EVENT, TS_DATA_16B }`.
  The core already tags markers by type (inside `timer_id`: `(id>>16)&0x7`).
- **The RT entry is already a tagged 64B record**: general entries carry raw kernel_profiler
  marker data; the sync entry uses a sentinel id (`l1_data[3] = REALTIME_PROFILER_SYNC_MARKER_ID`)
  as a discriminator.

---

## Data flow

```
Tensix core (kernel_profiler.hpp)
  └─ writes 2-word markers into per-RISC L1 SPSC ring
     (no DRAM push; on full ring: STALL, terminate-aware, never drop)
        │
        ▼
X280 PRESENTER (profzone.c)
  └─ drains rings, AUGMENTS each worker-core marker with data the core lacks
     (core_x/core_y, risc; later: back-pressure/stall time, …)
  └─ emits ONE 64B WIRE PACKET per marker: header{type} + raw fields
        │  (D2H socket)
        ▼
HOST: decode → enrich → dispatch   (NOT a relay)
  └─ read wire header.type
  └─ per-type ENRICHMENT, done ONCE, centrally:
        • decipher hash → zone name
        • translate virtual → NOC0 (keep BOTH)
  └─ build the ENRICHED packet, hand to registry[type]
        │
        ▼
SUBSCRIBERS (typed callbacks)
  └─ Tracy handler (today): pushes ZONE_START/END on the correct lane
  └─ CSV / perf-report, event handlers, … (later) — just more subscribers
```

---

## Decisions

### D1 — Two representations: wire packet vs enriched packet
- **Wire packet** — what the X280 emits and what comes off the socket. POD, packed,
  `≤64B`, `PacketHeader{ uint16_t type }` at offset 0. Directly castable for decode. The
  64B / POD / packed constraints apply **only here**.
- **Enriched packet** — what callbacks receive. A host-built struct, **not** size- or
  POD-constrained (never transported). Carries fully-resolved, human-meaningful data.
- **Zero-copy is intentionally relaxed for enriched types.** For zones the callback gets a
  host-constructed object (name is a string, NOC0 is computed), not a raw cast. Zero-copy
  still applies to the wire decode and to any future type that needs no enrichment.

### D2 — Host side is decode → enrich → dispatch, not a relay
Enrichment is centralized and happens **once** per packet, before any callback runs.
Rationale (the hard constraint): **callbacks must never receive ciphered data** (raw
name hashes, virtual-only coords) and **no two callbacks may redo the same translation.**
The host layer therefore legitimately contains per-packet-type logic — that is expected,
not a smell.

Concretely for a zone:
- **Name deciphered** host-side (hash → name), resolved to a **stable reference**
  (`string_view`/pointer into the session-lived hash→name table — no per-marker string
  allocation).
- **Both coordinates presented**: `core_virtual{x,y}` (what the X280 relays) **and**
  `core_noc0{x,y}` (translated, matches the standard DeviceProfiler / DRAM view).

### D3 — Presenter lives in the X280
For now it only augments worker-core packets (attach coord/risc). Later it may do more.
**Non-core packet sources (sync, program-zone, …) are out of scope now** — multiple
different sources may generate them; the presenter is simply where they will slot in.

### D4 — Option B: one record *type* per kind (not one tagged record)
Distinct POD/enriched structs per kind; type-safe subscribers; a new kind is a new struct
(+ its enrichment), touching nothing existing.
- **`ZONE_START` and `ZONE_END` are ONE type** (`WorkerZonePacket`) with start/end as an
  inner field (reuse `timer_id`'s packet bits) — same layout, same handler.
- `TS_DATA` / `TS_EVENT` / sync / program-zone become their own types later.

### D5 — Single `Register`, auto-detecting the type
```cpp
Register([](const WorkerZonePacket& z) { ... });   // filed under WorkerZonePacket::kType
```
- Deduce the **enriched** type `T` from the callback's argument (function-traits on
  `operator()`), require each packet to expose `static constexpr PacketType kType`, store
  in `registry[T::kType]` behind a `void(const void*)` shim that casts.
- **Escape hatch:** `Register<T>(cb)` explicit-template overload for ambiguous cases
  (e.g. generic lambdas).
- **Avoid** the fully-dynamic "type as a runtime 2nd arg" form (callback takes
  `const void*`, loses type safety) unless a dynamic subscriber is ever needed.

### D6 — Common header is minimal: `{ uint16_t type }`
At offset 0 of every wire packet, so the host reads the tag before it knows the type.
Coord/timestamp stay per-packet (not promoted into the shared header) for now.

### D7 — Core side stays ~unchanged
No DRAM push; on a full L1 ring **stall, terminate-aware, never drop**. **[Later]** stamp
back-pressure/stall time into the flit the X280 reads, so profiler-induced stalls are
measurable rather than invisible. This is a **separate change**, sequenced after the host
packet framework.

---

## Now-scope vs deferred

**Now:** the generic framework (wire/enriched split, single `Register` + typed dispatch,
minimal header) + exactly **one** packet type — `WorkerZonePacket` — with the Tracy handler
as its subscriber (virtual→NOC0 + name resolution moved into host enrichment; the direct
`PushDeviceMarker` is deleted). Behavior already verified on primed bh-17 (correct nesting,
NOC0 coords, real names) is preserved; only the plumbing changes.

**Deferred:** non-core packet types (sync, program-zone); event/payload types
(`TS_DATA`/`TS_EVENT`); CSV / perf-report subscriber (needs `run_host_id` per marker, which
the wire stream does not yet carry); core-side back-pressure stamping (D7); packing
multiple markers per 64B flit (today: one marker per page — wasteful but simple).

---

## Packet catalog

| Type | Status | Wire fields (raw) | Enrichment (host) | Enriched fields (callback) |
|---|---|---|---|---|
| `WorkerZonePacket` | **now** | type, core_x/y (virtual), risc, timer_id, timestamp | hash→name, virtual→NOC0 | type, chip, core_virtual{x,y}, core_noc0{x,y}, risc, timer_id, name, timestamp, is_start |
| Event (`TS_EVENT`) | later | — | — | — |
| Payload (`TS_DATA/16B`) | later | — | — | — |
| Sync | later (non-core) | — | — | — |
| Program-zone | later (non-core) | — | — | — |

---

## Where things live (planned)

- **Wire + enriched packet structs, `PacketHeader`, `Register`/dispatch:** new header
  `tt-metalium/experimental/realtime_profiler_packets.hpp`, alongside the existing
  `experimental/realtime_profiler.hpp`.
- **X280 presenter (wire emit):** `tools/x280_bm/src/profzone.c`.
- **Host decode/enrich/dispatch:** `tt_metal/distributed/realtime_profiler_manager.cpp`
  (`process_x280_page` becomes decode→enrich→`Invoke`).
- **Tracy subscriber:** `tt_metal/impl/dispatch/realtime_profiler_tracy_handler.cpp`
  registers for `WorkerZonePacket`; `PushDeviceMarker` removed.

See also `FINDINGS.md` (§16 throughput, drain/pair history) and `PRIMER.md`.
