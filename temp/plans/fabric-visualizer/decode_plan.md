# Task 4 — Decode (Python, offline)

Parent: `visualizer_plan.md` §"4. Decode". Design reference: `fabric_debug_infrastructure_design.md` §3.7
("Decode and reports", "Router liveness").

Goal: turn a directory of manifests + snapshots + `.bin` sidecars into one `decoded.json` that the
viewer (task 5) reads. Decode owns every piece of fabric knowledge: region slicing, struct layouts,
enum tables, packet-header layouts, credit polarity, liveness classification, stall score. It needs
no device, no ttexalens, no MPI. Capture and manifest formats do not change in this task.

---

## 1. What we learned reading the inputs and the kernel (drives every decision below)

### 1.1 What is actually on disk today (T3K, `FABRIC_2D`, 40 routers)

Manifest (`generated/fabric/fabric_debug_manifest_rank_1_of_1.json`, 214 KB):

- `run` {arch, fabric_config, fabric_type, reliability_mode, tensix_config, udm_mode, host_rank, mpi_rank,
  world_size, local_mesh_ids, written_at}
- `hal` {unreserved, go_msg, launch, fabric_telemetry, routing_table, router_state, router_command,
  eth_fw_mailbox} — `router_state` / `router_command` are 4-byte windows *inside* `routing_table`
  (`RouterStateManager` at offset 0 / 16).
- `heartbeat` {address 8064, magic `0xDCBA0000`, magic_mask `0xFFFF0000`, period_iters 64}
- `fabric_context` {topology, is_2d_routing, packet_header_size_bytes 96, max_payload_size_bytes,
  channel_buffer_size_bytes 4448, `routing_2d_route_buffer_size` 36 **or** `routing_1d_extension_words`,
  tensix_enabled, bubble_flow_control}
- `enums` {EDMStatus, TerminationSignal, RouterCommand, RunMsg}. **`RouterState` is not exported.**
- `layouts`: 5 layouts × 94 regions. Region `backing` values seen: `group | unreserved_l1 | fixed_l1 |
  stream_reg`. **`group` exists** (parents like `lifecycle`, `sender.0`) even though the region-tree plan
  listed three backings; `manifest.py` tolerates it silently. Region `schema` values seen:
  `u32, EDMStatus, TerminationSignal, handshake_info_t, heartbeat_word, perf_telemetry, code_profiling,
  channel_trimming, u32_counter_array, EDMChannelWorkerLocationInfo, SenderChannelProducerCursor,
  packet_ring, stream_remote_dest_buf_space_available, fabric_telemetry, routing_l1_info_t, raw`, plus
  none on groups.
- `meshes[].chips[]` {fabric_chip_id, mesh_coord, physical_chip_id, asic_id, is_local, master_router_chan,
  routers[] {eth_chan, direction, routing_plane, link_class, logical_core, virtual_core, layout_id, instance}}
- `links[]` {src, dst endpoints, direction, routing_plane, link_class, wrap, cross_host} — 40 directed edges.

Snapshot (`slice_e_live.json`, 225 KB + 6.3 MB `.bin`): per router `status`, `health` {reset, reset_bits,
wall_clock}, `lifecycle` {edm_status, termination_signal, go_signal}, `liveness[3]` {t, heartbeat,
wall_clock}, `streams` {pre, post, torn} keyed by stringified id, `blobs` {unreserved, fabric_telemetry,
routing_table, go_msg, launch} each {address, size, offset, sha256, status, error}, `identity`.

### 1.2 The heartbeat word is shared with the Ethernet base firmware

`slice_e_live` / `slice_e_killed` liveness samples contain **two formats on the same router**:
`0xDCBAxxxx` (fabric loop, `FABRIC_KERNEL_HEARTBEAT_MAGIC`) and `0xABCDxxxx`
(`BASE_FW_HEARTBEAT_SIGNATURE = 0xABCD` in `umd/device/api/umd/device/firmware/erisc_firmware.hpp`).
Router 2's three samples were `dcba, abcd, abcd`. The base FW writes the same word when the router
context-switches into it (`enable_context_switch` / `IDLE_CONTEXT_SWITCHING` in the router loop).

Consequences:

- Classification must be **per sample**: `fabric | base_fw | other`. Only fabric-format samples say
  anything about the fabric loop. Base-FW samples are themselves evidence the ERISC is alive and idle
  enough to context switch; they are *not* "no progress".
- `fabric_heartbeat_counter` is `uint16_t`, written every 64 iterations. An idle loop runs millions of
  iterations per second, so the low 16 bits wrap many times per second. Two fabric samples 1 s apart are
  effectively unordered; **"advancing" is "any two fabric samples differ", "static" is "all fabric samples
  identical"**. There is no monotonic delta to compute. (Coincidental equality of two 64-aligned values is
  ~1/1024; three, ~1e-6. Acceptable.)
- This explains the slice E result "heartbeat still moves after `SIGKILL`" — the fabric loop keeps running
  without a host. The README slice E paragraph gets the corrected wording in slice E of this task.

### 1.3 Ring slot positions are not in L1

- Sender ring: the write cursor lives in the *producer* (worker or upstream router); the router's read
  counter is a kernel local. `SenderChannelProducerCursor` (`sender.s.control.buffer_index_sem`) is
  written **only on connection close** (`edm_fabric_worker_adapters.hpp:580`). `conn_info.edm_read_counter`
  is copied to L1 only when no worker is connected or at close (`fabric_erisc_router.cpp:1146-1153`).
- Receiver ring: `ackptr / wr_sent_ptr / wr_flush_ptr / completion_ptr` are all kernel locals.
- What *is* observable: counts, via stream registers (§1.4).

So decode reports **occupancy count** with its source, and decodes the header in **every** slot as the
"last header written to this slot" with `slot_state: unknown`. It does not paint slots occupied/free.
That is the honest version of the parent plan's "mark each slot occupied | free | torn | unknown".

### 1.4 Stream register semantics (from `fabric_erisc_router.cpp` and `fabric_stream_regs.hpp`)

All values are `STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE` (what capture stores as `buf_space_available`),
initialised with `init_ptr_val` before the handshake.

| Region id | Init | Who increments | Who decrements | Value means | Healthy |
|---|---|---|---|---|---|
| `sender.s.free_slots` | ring `count` | router after completion (`increment_local_update_ptr_val(+1)`) | producer per packet written (worker via `worker_free_slots_stream_id`, or upstream router) | free slots in **this** sender ring | high; `== count` idle |
| `sender.s.credits.acked` | 0 | remote receiver per packet seen | local router as it consumes | first-level acks not yet consumed | low; 0 idle |
| `sender.s.credits.completed` | 0 | remote receiver per packet completed | local router as it consumes | completions not yet consumed | low; 0 idle |
| `credits.receiver.r.pkts_sent` | 0 | remote sender per packet sent | local receiver as it processes | packets in **this** receiver ring not yet processed | low; 0 idle; `== receiver ring count` means receiver is full |
| `credits.downstream.vcV.edgeK.free_slots` | downstream sender ring count | downstream router when it frees a slot | local router when it forwards | free slots in the **downstream router's** sender ring | high |
| `credits.vc2_receiver.free_slots`, `credits.tensix_relay.free_slots` | ring count | | | as `sender.s.free_slots` | high |

Live T3K idle values agree: streams 0..3 = 14, 4, 4, 4 (== sender ring counts 14, 4, 4, 4);
stream 4 = 0; streams 5..12 = 0; `edge3` (enabled) = 4; disabled edges 13, 14, 16 hold garbage
(11987, 99520, 57650) — **disabled regions must never be scored**.

Counter-backed credits (`VCx_USES_COUNTER_CREDITS`, BH-leaning): `credits.to_sender_*` /
`credits.receiver_*` are monotonic `u32` counters, not pending counts. Decode reports them as
`kind: counter` with no polarity; a pending count needs both ends and is a later derived field.

### 1.5 Struct layouts decode has to hardcode (all little-endian)

- `handshake_info_t` (`edm_handshake.hpp`): `local_value u32 @0`, `neighbor_mesh_id u16 @4`,
  `neighbor_device_id u8 @6`, pad, `padding[2] @8`, `scratch[4] @16..32`. **Region is 16 B, struct is
  32 B**: `scratch` overlaps `credits.to_sender_ack` in the real layout. Decode reads the 16 B region and
  decodes the first three fields only; notes `struct_size 32 > region_size 16` once per layout.
- `EDMChannelWorkerLocationInfo` (`fabric_edm_types.hpp`): `worker_semaphore_address @0`,
  `worker_teardown_semaphore_address @16`, `worker_xy @32` (`x u16`, `y u16`), `edm_read_counter @48`.
- `SenderChannelProducerCursor`: `write_counter @0`, `write_index @4`.
- `sender.s.control.connection_sem` (`u32`): `0 unused | 1 open | 2 close_request`
  (`fabric_connection_interface.hpp`). Decode carries this table; it is not in `manifest.enums`.
- `FabricTelemetry` (`fabric_telemetry_msgs.h`, 160 B): `StaticInfo @0` {version u32, mesh_id u16,
  neighbor_mesh_id u16, device_id u8, neighbor_device_id u8, direction u8, supported_stats u8,
  fabric_config u32} = 16 B; `DynamicInfo @16` {tx_bandwidth 32 B @16, rx_bandwidth 32 B @48,
  erisc[2] @80, each 24 B {router_state u32, pad 4, tx_heartbeat u64, rx_heartbeat u64}};
  `postcode u32 @128`; `scratch[7] @132`. Verify `16 + 112 + 4 + 28 == 160` in a unit test.
- `routing_l1_info_t` (`fabric_common.h:867`, 2704 B): `state u32 @0` (`RouterState`), `command u32 @16`
  (`RouterCommand`), `my_mesh_id u16 @32`, `my_device_id u16 @34`, `intra_mesh_direction_table 96 B @36`,
  `inter_mesh_direction_table 384 B @132`, 1D/2D route union 1160 B @516, `exit_node_table 1024 B @1676`,
  `my_mesh_coord_y @2700`, `my_mesh_coord_x @2701`, `mesh_y_size @2702`, `mesh_x_size @2703`.
  Direction tables are 3-bit packed **through a compress/decompress mapping**
  (`compressed_direction_table.cpp` host side, `fabric_direction_table_interface.h` device side). Port
  `get_direction` + `decompress_value` and unit-test against bytes packed by the same rule. Route union
  and exit-node table stay raw (`raw_ref`) in v1.
- Packet headers (`fabric_edm_packet_header.hpp`). Base (44 B): `command_fields 40 B @0`,
  `payload_size_bytes u16 @40`, `noc_send_type u8 @42`, `src_ch_id u8 @43`. Then:
  - 1D: `LowLatencyPacketHeaderT<E>`: `routing.value u32 @44`, `route_buffer[E] u32 @48`, padded to 48
    (`E=0`) or 64. Production 1D is always `ROUTING_MODE_LOW_LATENCY` (`fabric_context.cpp:395`), so the
    dynamic `PacketHeader` (also 64 B) is not a live ambiguity; `E = fabric_context.routing_1d_extension_words`.
  - 2D: `HybridMeshPacketHeaderT<R>` (packed): `routing.value u32 @44`, `route_buffer[R] @48`,
    `dst_start_chip_id u16 @48+R`, `dst_start_mesh_id u16 @50+R`, `mcast_params[4] u16 @52+R`; padded to
    16. `R = fabric_context.routing_2d_route_buffer_size` (36 → 96 B, matches
    `packet_header_size_bytes`).
  - UDM (`run.udm_mode == ENABLED`): 2D header + `UDMControlFields 16 B`.
  - Header type is fully determined by the manifest; **no manifest change needed**. Decode asserts
    `sizeof(chosen) == fabric_context.packet_header_size_bytes` and marks the ring `unsupported` if not.
  - `NocCommandFields` by `noc_send_type`: `unicast_write / unicast_read` {noc_address u64};
    `unicast_inline_write` {noc_address u64, value u32}; `unicast_seminc` {noc_address u64, val u32,
    flush u8}; `unicast_seminc_fused` {noc_address u64, semaphore_noc_address u64, val u32, flush u8};
    `mcast_write` {address u32, x_start, y_start, size_x, size_y u8}; `mcast_seminc` {address u32,
    val u32, x_start, y_start, size_x, size_y}; `unicast_scatter_write` {noc_address[4] u64,
    chunk_size[3] u16, chunk_count u8, chunk_encoding u8}; `sparse_mcast_write` {noc_address[4] u64,
    counts[4] u8, num_dests, num_chips, write_idx, chip_idx}. Unknown `noc_send_type` → raw only.
  - NOC address split (WH and BH `noc_parameters.h`): `local = addr & ((1<<36)-1)`, `x = (addr>>36)&0x3F`,
    `y = (addr>>42)&0x3F`. Report as `{x, y, local_addr}` next to the raw u64.

### 1.6 Where each region's bytes come from

| Region backing / id | Source |
|---|---|
| `unreserved_l1`, allocated, size > 0 | `blobs.unreserved` bytes at `offset + (address - blob.address)` |
| `fixed_l1` `hal.telemetry` | `blobs.fabric_telemetry` (address match) |
| `fixed_l1` `hal.routing_table` | `blobs.routing_table` |
| `fixed_l1` `lifecycle.heartbeat` (8064) | **not in any blob** → from `liveness[]` samples |
| `stream_reg` | `streams.post[str(id)]`; `pre` for torn; absent id → `unknown` |
| `group` | container only, no value |
| anything with `allocated: false` or `size == 0` | `status: unallocated`, no bytes |

General rule: a region's bytes come from the blob whose `[address, address+size)` covers it; if no
blob covers it, `status: not_captured`. Blob `status != ok` → region inherits that status.

### 1.7 Existing code to reuse

- `capture/manifest.py::load_manifest` → validation + `sha256` + `data`. Decode walks `data["meshes"]`
  directly for non-local chips (the `router_targets` list is local-only by design).
- `capture/tests/test_manifest.py::required_blocks` and `test_peek.py::fixture_manifest` build a minimal
  valid manifest; decode fixtures extend them rather than inventing a second minimal manifest.
- Test import style: `from tt_metal.fabric.debug.visualizer.capture.manifest import ...`, run as
  `python3 -m pytest tt_metal/fabric/debug/visualizer/decode/tests -q --noconftest`. `decode/` gets an
  `__init__.py` like `capture/`.
- Nothing from `fabric_erisc_dumper.py` / `fabric_erisc_constants.py`: its stream map is wrong for this
  fabric (region-tree plan §7) and its polarity note is by id range, not by role.

---

## 2. Package layout

```
tt_metal/fabric/debug/visualizer/
  decode/
    __init__.py
    cli.py          # argparse; discover inputs; write decoded.json (tmp + rename)
    inputs.py       # find/pair manifest, snapshots, .bin; sha/size verification; DecodeInput records
    merge.py        # cross-rank union of topology, one owner per router, coverage
    regions.py      # slice a router's layout against its blobs/streams → RegionValue list
    structs.py      # schema-key → decoder; enum lookups; direction-table unpack
    headers.py      # header type selection from fabric_context; slot header decode
    credits.py      # per-channel normalized credits, occupancy counts, stall score
    liveness.py     # heartbeat sample classification; lifecycle/exit-state words
    output.py       # assemble decoded.json object; schema constants
    tests/
      __init__.py
      fixtures.py   # build manifest + synthetic .bin + snapshot JSON in a tmp dir
      test_inputs.py test_merge.py test_regions.py test_structs.py test_headers.py
      test_credits.py test_liveness.py test_decoded_schema.py
  schema/
    fabric_debug_decoded_schema.json   # new, decoded_version 1
```

No new dependencies: `json`, `struct`, `hashlib`, `pathlib`; `jsonschema` only in tests (already used).

---

## 3. Input discovery and pairing (`inputs.py`)

CLI: `python3 tt_metal/fabric/debug/visualizer/decode/cli.py <dir | files...> -o decoded.json
[--allow-manifest-mismatch] [--skip-bin-hash] [--slots none|headers] [--expert-raw]`.

1. Collect `*.json`; classify by `kind` (`fabric_debug_manifest` / `fabric_debug_snapshot`); ignore others.
2. For each snapshot, find its manifest: the manifest file in the same set whose bytes hash to
   `snapshot.manifest.sha256`. `snapshot.manifest.path` is never used to open a file (schema says so).
   No hash match → error, or with `--allow-manifest-mismatch` fall back to the only manifest with the same
   `run` identity and record `manifest_sha_verified: false` on every router from that snapshot.
3. `.bin`: `snapshot.raw.file` resolved next to the snapshot JSON. Verify size, and sha256 unless
   `--skip-bin-hash` (Galaxy-sized files). Missing `.bin` → the snapshot still decodes; every L1 region is
   `not_captured`.
4. Every manifest in the set is loaded via `load_manifest` (so validation is the same as capture's).
5. Refuse to mix inputs whose `run.arch` or `run.fabric_config` differ (`DecodeError`).

One snapshot + one manifest is the only hardware-tested path; multi-file behaviour is fixture-tested.

---

## 4. Merge and coverage (`merge.py`)

- Topology: union of `meshes[]` / `links[]` across manifests keyed by `mesh_id` / `(src, dst)`; a chip that
  appears in several manifests must agree on `mesh_coord`, `physical_chip_id`, `asic_id` (`asic_id` may be
  `null` on the non-local side; local wins). Disagreement → `DecodeError`.
- Router ownership: a router is decoded from the snapshot whose manifest marks its chip `is_local`. Two
  snapshots both local for one endpoint → `DecodeError` (two runs in one directory).
- Coverage per router: `ok | unreadable | reset | torn | unknown | unsupported` copied from capture, plus
  decode-only `not_captured` (in topology, no local snapshot). Summary counts at the top level. Identity
  mismatch (`routing_l1_info_t.my_mesh_id/my_device_id` ≠ endpoint, recomputed here from the blob, not
  taken from `identity.matches_manifest`) is a per-router flag and a summary count.

---

## 5. Region slicing and struct decode (`regions.py`, `structs.py`)

For every region of the router's layout, emit:

```json
{ "id": "sender.0.control.conn_info", "parent": "sender.0", "backing": "unreserved_l1",
  "schema": "EDMChannelWorkerLocationInfo", "allocated": true, "enabled": true,
  "status": "ok", "address": 98640, "size": 64,
  "raw_ref": { "file": "slice_e_live.bin", "offset": 336, "size": 64 },
  "value": { "worker_semaphore_address": 0, "worker_teardown_semaphore_address": 0,
             "worker_xy": {"x": 0, "y": 0}, "edm_read_counter": 0 } }
```

- `raw_ref` always points into the `.bin`; decoded.json never embeds L1 bytes except `raw_hex` for regions
  ≤ 64 B when `--expert-raw` is given.
- Decoders by `schema` (§1.5 offsets): `u32` → `{word: v, words: [...]}`; `EDMStatus`, `TerminationSignal`
  → `{raw, name}` via `manifest.enums`, `name: null` when not in the table; `handshake_info_t`;
  `EDMChannelWorkerLocationInfo`; `SenderChannelProducerCursor`; `u32_counter_array` → `count` words;
  `fabric_telemetry`; `routing_l1_info_t`; `heartbeat_word` → from liveness (§7); `packet_ring` → §6;
  `perf_telemetry | code_profiling | channel_trimming | raw` → `raw_ref` only; `stream_remote_dest_buf_space_available`
  → `{pre, post, torn}`. `RouterState` comes from a built-in table `{0 INITIALIZING, 1 RUNNING, 2 PAUSED,
  3 DRAINING, 4 RETRAINING}` (`fabric_telemetry_msgs.h`) unless `manifest.enums.RouterState` exists.
- Unknown `schema` string → `value: null`, `status: unsupported`, never an exception. A new region in the
  manifest must appear in the panel without a decode change (parent plan §5).
- Region `status` precedence: blob `unreadable` > router `reset` > blob other non-ok > `torn` (streams only)
  > `not_captured` > `ok`.

---

## 6. Packet rings and headers (`headers.py`)

- Header type from `fabric_context` + `run.udm_mode` (§1.5). Ring slot `i` at `address + i*stride`; header
  is the first `packet_header_size_bytes` of the slot; payload is never read.
- Per slot: `{index, raw_ref, header: {...}, plausible: bool, slot_state: "unknown"}`.
  `plausible` = `noc_send_type <= NOC_SEND_TYPE_LAST (8)` ∧ `payload_size_bytes <= max_payload_size_bytes`
  ∧ (2D) `dst_start_mesh_id` is a mesh in the topology. It is a hint for the viewer's dimming, not a claim.
- Ring summary: `{depth: count, occupied_count, occupancy_source}` where `occupied_count` comes from
  `credits.py` (§7) — `depth - free_slots` for sender rings, `pkts_sent` for receiver rings — or `null`
  with `occupancy_source: null` when the stream is disabled/unknown/torn. `occupied_count > depth` or
  `< 0` → `occupancy_status: inconsistent`.
- `--slots none` skips header decode (Galaxy size control); ring summary still emitted.

---

## 7. Credits, liveness, stall score (`credits.py`, `liveness.py`)

### 7.1 Channels (normalized, backing-independent)

Per router, using only `enabled` regions:

```json
"channels": {
  "senders": [ { "index": 0, "vc": 0, "role": "worker|upstream", "depth": 14, "free_slots": 14,
                 "occupied": 0, "acked_pending": 0, "completed_pending": 0,
                 "credit_backing": "stream_reg|counter", "connection": {"raw": 0, "name": "unused"},
                 "torn": false, "status": "ok" } ],
  "receivers": [ { "index": 0, "vc": 0, "depth": 8, "pkts_pending": 0, "torn": false, "status": "ok" } ],
  "downstream": [ { "vc": 0, "edge": 3, "free_slots": 4, "depth": null, "status": "ok" } ]
}
```

- `role`: `worker` when `index == instance.worker_sender_channel`, else `upstream`.
- `vc` from `instance.sender_channels_per_vc` prefix sums (fabric-scoped flat index, region-tree plan §9).
- `depth` for `downstream` is `null` in v1: which neighbour router's ring it mirrors is not in the manifest
  (open question §10). Value and polarity are still reported.
- Counter-backed VCs: `acked_pending / completed_pending` are `null`, `counters: {to_sender_ack, ...}` raw.

### 7.2 Liveness

Per router:

```json
"liveness": { "classification": "advancing|static|reset|unknown|insufficient",
              "fabric_samples": 2, "base_fw_samples": 1, "other_samples": 0,
              "samples": [ {"t": "...", "raw": 3703230720, "format": "fabric|base_fw|other"} ] }
```

Rules, in order: router `status == reset` or `health.reset_bits.erisc0` → `reset`; `status == unreadable`
→ `unknown`; fewer than 2 fabric-format samples → `insufficient` (a router that context-switched every
time is alive but unproven); all fabric samples equal → `static`; else `advancing`. Constants:
`magic/mask` from `manifest.heartbeat`; `0xABCD0000` base-FW signature is a decode constant with a comment
pointing at UMD's `erisc_firmware.hpp`.

Lifecycle words → `{edm_status: {raw, name}, termination: {raw, name}, go_signal: {raw, name}}` and one
derived `exit_state` per design doc §3.7: `READY + KEEP_RUNNING` → `running_or_host_gone`;
`TERMINATED + terminate set` → `orderly_exit`; `READY + terminate set` → `teardown_stuck`; init postcode →
`initializing`; `0` / non-enum → `wiped_or_never_ran`; anything else → `unknown`.

### 7.3 Stall score (viewer colouring only)

Per router, `null` if `status != ok` or all relevant channels are `torn`/`unknown`; else the max of:
`occupied / depth` over enabled senders; `pkts_pending / depth` over enabled receivers; `1.0` for any
enabled downstream edge with `free_slots == 0`. Attached to the router and copied onto the outgoing
`links[]` entry whose `src` is that router (`link.stall_score`, `link.status`). Documented as a heuristic;
idle-healthy and hung-backpressure look alike in one snapshot (design doc §3.7) — the number ranks, it does
not diagnose. Observations list (design doc "reports") is task 6; v1 emits only the score and the inputs
to it.

---

## 8. Output (`output.py`, `schema/fabric_debug_decoded_schema.json`)

```json
{ "decoded_version": 1, "kind": "fabric_debug_decoded", "generated_at": "...",
  "inputs": [ { "manifest": {"path", "sha256", "mpi_rank", "host_rank"},
                "snapshot": {"path", "sha256", "captured_at", "provenance"},
                "raw": {"file", "size", "sha256", "verified": true} } ],
  "run": { "arch", "fabric_config", "topology", "is_2d_routing", "udm_mode", "world_size" },
  "fabric_context": { ...copied..., "packet_header_type": "HybridMeshPacketHeaderT<36>" },
  "enums": { ...manifest.enums + RouterState + ConnectionState... },
  "coverage": { "routers_total", "captured", "ok", "unreadable", "reset", "torn", "unknown",
                "unsupported", "not_captured", "identity_mismatch", "manifest_sha_unverified" },
  "topology": { "meshes": [...], "links": [ {...manifest link..., "stall_score", "status"} ] },
  "routers": [ { "id", "layout_id", "instance", "direction", "link_class", "routing_plane",
                 "capture": {"status", "error", "snapshot_index", "torn", "owner_alive"},
                 "identity": {"my_mesh_id", "my_device_id", "matches"},
                 "lifecycle": {...}, "liveness": {...}, "channels": {...}, "stall_score",
                 "rings": [ {...} ], "regions": [ {...} ], "warnings": [ "..." ] } ] }
```

- `regions[]` keeps manifest order so the viewer's panel groups by `parent` without sorting.
- Size estimate: 40 routers × (94 regions ≈ 25 KB + 60 slot headers ≈ 15 KB) ≈ 1.6 MB on T3K; Galaxy ≈ 8×.
  Acceptable for v1; per-router lazy files remain the parent plan's open question.
- Written `tmp + rename` like the manifest and `.bin`.

---

## 9. Tests (no hardware)

`fixtures.py` builds, in a temp dir: a manifest (extending `required_blocks()` with one layout of ~12
regions: lifecycle words, one sender with control + 4-slot ring + 3 streams, one receiver ring + pkts_sent,
one downstream edge, `hal.telemetry`, `hal.routing_table`, `padding`), a `.bin` with `struct.pack`ed
known bytes at every region offset, and a snapshot JSON referencing them (streams pre/post, liveness,
blobs with correct offsets and sha256). A second fixture manifest marks the same chip non-local for merge
tests.

1. `test_inputs`: pairing by sha; mismatch refused; `--allow-manifest-mismatch` flags; missing `.bin` →
   `not_captured`; `.bin` size/sha mismatch refused; arch mismatch refused.
2. `test_merge`: two ranks, local/non-local → one owner; both local → error; coverage counts;
   `not_captured` for a chip with no snapshot.
3. `test_regions`: offsets into `.bin` for `unreserved_l1` and `fixed_l1`; heartbeat from liveness;
   `stream_reg` pre/post/torn; unallocated / size-0 / `group`; blob `unreadable` propagates; unknown schema
   → `unsupported` not exception.
4. `test_structs`: every decoder against hand-packed bytes, including `FabricTelemetry` offsets summing to
   160, `routing_l1_info_t` tail bytes at 2700..2703, direction-table unpack vs a packing helper written from
   `fabric_direction_table_interface.h`, enum name lookups with unknown raw.
5. `test_headers`: type selection for 1D `E=0/1`, 2D `R=36`, UDM; size assertion mismatch → `unsupported`;
   each `noc_send_type` command decode; NOC address split; `plausible` rules; `--slots none`.
6. `test_credits`: idle (`free == depth`) → occupied 0, score 0; backpressure (`free 0`, `pkts_pending ==
   depth`) → score 1.0; disabled stream ignored even when garbage; torn → `null` occupancy; counter-backed
   VC → `counters` present, pendings `null`; `free > depth` → `inconsistent`.
7. `test_liveness`: fabric×3 differing → `advancing`; identical → `static`; `dcba, abcd, abcd` →
   `insufficient` with counts; reset wins; `exit_state` table.
8. `test_decoded_schema`: fixture output validates against `fabric_debug_decoded_schema.json`; every
   region id in the manifest appears exactly once per router.

Manual, this host (recorded in README, not committed as fixtures — the `.bin` is 6 MB): run decode on
`generated/fabric/{slice_e_live, slice_e_killed}.json`. Expected: 40 routers `ok`; all `exit_state
running_or_host_gone`; identity matches 40/40; senders `occupied 0`, receivers `pkts_pending 0`, every
enabled downstream `free_slots 4`; stall score 0 everywhere; liveness `advancing` or `insufficient`, none
`static`; `routing_l1_info_t.my_mesh_coord` equals `chip.mesh_coord`; `mesh_y_size/x_size == 2/4`;
telemetry `static_info.mesh_id/device_id` equal the endpoint; the killed decode differs from the live one
only in liveness sample values and `owner_alive`.

Optional if a T3K is free: capture once under a real all-gather to see non-zero `occupied` and a
non-`unused` `connection_sem`; record the numbers. Not a gate.

---

## 10. Slices (each builds, each reviewable on its own)

| # | Slice | Files | Done when |
|---|---|---|---|
| A | Inputs, merge, skeleton, schema | `inputs.py`, `merge.py`, `output.py`, `cli.py`, schema, `test_inputs`, `test_merge`, `test_decoded_schema` | done |
| B | Regions + structs + liveness | `regions.py`, `structs.py`, `liveness.py`, tests 3, 4, 7 | done |
| C | Rings + headers | `headers.py`, test 5 | done |
| D | Credits + stall score + link annotation | `credits.py`, test 6 | done |
| E | Validation + README | README, slice E paragraph correction (heartbeat shared with base FW) | done |

Stop for review after **A**, after **C**, and after **E**.

---

## 11. Risks and how the plan handles them

- **Direction-table compress mapping.** If porting `decompress_value` is fiddly, ship the tables as
  `raw_ref` in B and add the unpack in a follow-up; identity/coords/state do not depend on it.
- **Header ambiguity on future 1D dynamic-routing builds.** Size check + `unsupported` ring rather than a
  wrong decode; manifest could later carry `routing_mode` (one line in `make_fabric_context_block`).
- **Garbage in disabled streams.** Every scoring/occupancy path filters on `enabled`; test 6 covers it.
- **`.bin` hash cost on Galaxy.** `--skip-bin-hash`; size is always checked.
- **Decoded size.** `--slots none`; per-router files later without a schema change (`rings[].slots` is
  already optional).
- **Two runs in one directory.** Sha pairing + one-owner rule make it an error, not a silent merge.
- **Heartbeat false "static".** Equality over 64-aligned 16-bit values; three samples make coincidence
  negligible; classification also reports sample counts so the viewer can show why.
- **Blackhole.** Two `reset_bits`; `erisc1` `null` on WH is preserved; counter-backed credits are
  expressed via `kind: counter`; header/NOC split constants are the same on BH. Untested here.

---

## 12. Open questions (not blocking)

- Mapping `credits.downstream.vcV.edgeK` to the neighbour router (and thus its ring depth). The kernel's
  `get_vc0_downstream_sender_channel_free_slots_stream_id(compact_index)` order plus `direction` should
  determine it; needs a read of the adapter setup around `fabric_erisc_router.cpp:2953-3100`. Cheap to add
  as `instance.downstream_edges[]` in the manifest later.
- Export `RouterState` (and the connection-state table) from `make_enums_block` so decode's built-in tables
  become fallbacks only. One-line C++ change; not required for this task.
- `lifecycle.handshake` region size 16 vs `sizeof(handshake_info_t)` 32 — the kernel really does let
  `scratch` overlap the counter block. Report to the builder owners; decode reads 16 B.
- Whether `decoded.json` should carry the manifest's full `layouts` so the viewer can render a router panel
  with no manifest on disk. Leaning yes (it is ~50 KB); decide when the viewer's loader is written.
