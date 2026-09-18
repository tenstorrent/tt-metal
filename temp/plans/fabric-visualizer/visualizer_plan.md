# Fabric hang visualizer — plan

Working plan from the debug-tool design discussion. This is the document to iterate on as we implement.

Task-specific plans live beside this file in `temp/plans/fabric-visualizer/`:

- `topology_manifest_plan.md` — task 1 topology portion (done)
- `launch_state_plan.md` — task 1 (done)
- `region_tree_plan.md` — task 2 (done)
- `capture_plan.md` — task 3 first cut (done: health + streams 0–29)
- `capture_extensions_plan.md` — task 3 rest (done: raw image, liveness, identity; T3K validated)
- `decode_plan.md` — task 4 (done: offline decode; idle T3K occupancy 0 / stall 0)
- `viewer_plan.md` — task 5 (next: static HTML/JS map + panel)

Reviewed against `temp/fabric_debug_infrastructure_design.md` (another engineer's analyzer design). Decisions
taken from that review are in "Decisions" below; the task list has been rewritten around them. Nothing
here has shipped, so tasks extend the existing manifest / capture code and schemas in place — there is no
"v2" of either.

Related existing code:

- `tt_metal/fabric/debug/` — ERISC dumper, constants, binary analyzer
- `tt_metal/fabric/control_plane.cpp` — topology, port map, routing tables
- `tt_metal/fabric/erisc_datamover_builder.cpp` — per-router named CT args
- `tt_metal/fabric/fabric_host_utils.cpp` — existing YAML serializers (mesh coords, asic mapping, intermesh)
- `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` — L1 structs (`routing_l1_info_t`, …)
- `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp` — `FabricConfig` (1D / ring / 2D / torus X/Y/XY)

---



## Purpose

Build a hang-debug tool that:

1. **Captures** live fabric state from a cluster (ttexalens peeks of ERISC registers and L1).
2. **Interprets** that state using a host-written description of *this* fabric instance (topology + per-router ABI).
3. **Shows** it as a zoomable map: galaxies / meshes → chips → eth routers → channels / VCs → buffers and named registers.

The goal is to replace log archaeology for flow-control hangs with a graph: which link is stalled, in which direction, with which packets sitting in which buffers.

v1 is **static**: manifest + snapshot files, opened in a page. No live websocket. Capture may still poll on the Python side and write multiple samples; the UI can later scrub `samples[]` without changing the schema much.

---



## Lay of the land: what you need in a debug session, and where it comes from

A useful session answers: *which hop is stuck, why, and what is in flight?* That data is split across four places. ttexalens only sees the last one.

### 1. Topology (who is connected to whom)

Needed to **draw** the cluster and to know which eth core is a fabric hop.


| Question                                   | Source                                                                                  | When known                                    |
| ------------------------------------------ | --------------------------------------------------------------------------------------- | --------------------------------------------- |
| 1D vs 2D vs torus (wrap X/Y/XY)            | `FabricConfig` / ControlPlane                                                           | Host runtime, fabric init (`SetFabricConfig`) |
| Mesh ids, chip coordinates, host ranks     | ControlPlane / TopologyMapper / mesh graph                                              | Same                                          |
| Galaxy / multi-mesh / intermesh            | Mesh graph + ControlPlane exit / intermesh maps                                         | Same                                          |
| Physical chip id ↔ fabric node             | TopologyMapper                                                                          | Same                                          |
| **Which eth channels are fabric routers**  | `Cluster::get_fabric_ethernet_channels`: link up **and** `EthRouterMode::FABRIC_ROUTER` | Same                                          |
| Direction E/W/N/S/Z and neighbor chip/chan | `router_port_directions_to_physical_eth_chan_map_`                                      | Same                                          |


This is **not** “all eth tiles on the chip” and **not** ttexalens active vs idle.

Nested sets (do not confuse them):

1. Every ethernet tile — `get_block_locations("eth")`.
2. Link-up eth — UMD active channels (local *and* `ethernet_connections_to_remote_devices`).
3. **Fabric routers** — subset of (2) assigned as fabric EDMs. This is the visualizer vertex set.

ControlPlane already serializes some of this (`serialize_mesh_coordinates_to_file`, asic mapping, intermesh ports). We should emit **one** sidecar that includes `fabric_config`, mesh shapes, chips, and an **edge list** (including torus wrap and intermesh). Layout engines key off that data; they should not hardcode “torus means draw wraparound” independently of the edge list.

### 2. Per-router ABI (how to *name* and *find* state on one ERISC)

Needed to label registers, know credit transport, and parse SRAM.

**Not in ControlPlane.** Produced by the **EDM builder** when it compiles each `fabric_erisc_router` (`named_args` in `erisc_datamover_builder.cpp`, consumed as CT args in `fabric_erisc_router_ct_args.hpp`).

Can **differ per ERISC on the same chip** (interior vs intermesh-edge vs tensix mux).


| Question                                                                                               | Source                                                                                       |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| Stream IDs (`to_receiver_0_pkts_sent_id`, sender free-slots, ack/completion streams, …)                | Builder named CT args / `StreamAssignment`                                                   |
| VC / sender / receiver counts (`ACTUAL_VC*`, serviced channels)                                        | Same                                                                                         |
| Credits in overlay streams vs L1 counters (`VC*_USES_COUNTER_CREDITS`; implied by multi-txq / express) | Same                                                                                         |
| First-level ack vs fused completion                                                                    | `ENABLE_FIRST_LEVEL_ACK_VC*`                                                                 |
| 2D / my direction / edge flags                                                                         | `IS_2D_FABRIC`, `MY_DIRECTION`, `IS_INTERMESH_ROUTER_ON_EDGE`, `IS_INTRAMESH_ROUTER_ON_EDGE` |
| Buffer geometry                                                                                        | `CHANNEL_BUFFER_SIZE`, buffer counts / bases                                                 |
| L1 addresses for counters, ptrs, conn info, status                                                     | Named CT args (`TERMINATION_SIGNAL_ADDR`, ack counter bases, …)                              |


UI implication: do **not** special-case every CT flag. Group by **observable effect** (credit transport, ack vs fused completion, channel topology, routing decode, buffer geometry). Show “credits” whether they live in a stream register or an L1 counter.

The existing dumper hardcodes streams 14–29 / 0–13. That will rot (5-wide VC0, extra senders, counter-credit VCs). The sidecar ABI replaces that map.

### 3. Typed L1 contents (what is *in* SRAM)

Addresses often come from HAL + ABI; **values** come from a peek.


| Object                                   | Typical home                                 | Notes                                                                                                       |
| ---------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `routing_l1_info_t`                      | `ROUTING_TABLE_BASE` (HAL; AERISC vs IERISC) | `my_mesh_id` / `my_device_id`, intra direction table (per dest chip), inter direction table (per dest mesh) |
| Compressed 2D paths                      | `ROUTING_PATH_BASE_2D`                       | NS/EW recipe; not the same as the direction table                                                           |
| Exit-node LUT                            | `EXIT_NODE_TABLE_BASE`                       | Dest mesh → exit chip on this mesh                                                                          |
| Channel ptrs, occupancy, `ack_counter[]` | Builder L1 map                               | Depends on FLA / counter-credit plan                                                                        |
| Circular buffers                         | Channel buffer bases + slot size             | Occupied slots decoded as packet headers (dest, `hop_index`, hop cmd E/W/N/S bits)                          |


ControlPlane **writes** routing tables at init; it does not keep a live copy for hang debug. After a hang, read L1.

Snapshot consistency: read **ptrs first, then buffers**; stamp the sample. A hung-but-not-frozen EDM can mutate mid-read.

### 4. Live overlay / L1 numbers (is it stalled *right now*)

ttexalens peeks. Architecture-specific overlay indices already live in `fabric_erisc_constants.py` (WH vs BH `BUF_SPACE_AVAILABLE`, etc.).

Polarity (already documented in the dumper README):

- **Free-slot** streams: high = good (space to send).
- **Ack/completion** remote space: low/zero = good (remote consuming). Rising = remote not draining.

Normalize to a **stall score** for coloring so one heatmap does not mix polarities.

Wall clock frozen / in reset / unreadable = core dead, not a fabric stall. Grey out eth that is not in the fabric port map.

---



## ttexalens (brief)

ttexalens is the **out-of-process debugger** used by `fabric_erisc_dumper.py`. It does not go through the (possibly wedged) tt-metal host process.

What it is good for:

- `init_ttexalens()` → `Context` with `context.devices`
- List chips; `device.get_block_locations("eth")` → `OnChipCoordinate`
- `read_from_device(coord, address, device.id, nbytes, context)` — registers and L1
- Arch: `device._arch` or `WormholeDevice` / `BlackholeDevice` (overlay indices differ)

What it is **not**:

- Meshes, galaxies, E/W/N/S/Z, routing planes
- Fabric vs tunnel vs unused eth
- Stream ID meaning, VC counts, CT flags

Idle vs active in ttexalens (`idle_eth_blocks`) is **“appears in local** `get_ethernet_connections()`**”**. Multi-host links live in `ethernet_connections_to_remote_devices`. Those cores are often live fabric/galaxy hops but look **idle** to ttexalens. That is why the dumper README says use `--include-idle` on multi-host. The visualizer must **ignore** that flag and use the fabric port map from the sidecar.

Hang debug: host may be stuck. Topology + ABI must already be on disk from init. ttexalens only **fills numbers** into that map. Viewing a snapshot later needs no devices.

Reuse from the dumper: init, read, arch detect, 32-bit unpack. Do **not** grow the dumper into this tool (hardcoded streams, wrong idle filter, CLI-for-humans).

---



## Architecture

```
Fabric init (C++)                    Hang (Python, per host)            Offline (Python)         View
─────────────────                    ───────────────────────            ────────────────         ────
delete stale manifest at init        ttexalens:                         join manifests +
ControlPlane topology +                liveness sequence                snapshots, slice
FabricContext constants +              pre-streams                      L1 by region tree,
builder template + per-router          UNRESERVED L1 image              merge hosts
region tree (task 2)                   post-streams
        │                                    │                                │
        ▼                                    ▼                                ▼
 manifest_rank_N.json  ─────────►  snapshot_rank_N.json + .bin  ───►  decoded.json  ───►  viewer.html
 (one write, after every            (raw + identity + status +        (named fields,       (map + generic
  router is READY_FOR_TRAFFIC)       manifest sha256)                  coverage)            region panel)
```

Versioned JSON **schema** is the contract. Frontend, capture, decode, and tests all speak it.

Three stages, three responsibilities:

- **Capture** writes raw bytes plus identity plus status. It never interprets. A capture taken before the
ABI task lands is re-decodable after it.
- **Decode** is offline, needs no device, no ttexalens, no MPI. It owns all fabric knowledge (region tree,
enum tables, credit polarity, stall score). Multi-host merge happens here.
- **View** renders `decoded.json`. It knows how to lay out a graph and render a region tree; it does not
know stream ids or CT flags.

Optional **fixture manifests** (same schema) for UI work without hardware: 1D line, 2D mesh, torus XY, two-mesh.

Compile-time flags: persist as schema, not as a forest of UI `if`s. Skip flags that do not change what is displayed (ctx-switch interval, noc flush, …).

---



## Decisions (from the design-doc review)

Kept from our plan:

- **ttexalens is the capture backend.** It tunnels to non-MMIO chips (T3K / Galaxy WH), it already
works, and it never takes UMD `CHIP_IN_USE`. The design doc's C++ `TTDevice` collector is BH-first,
MMIO-only, and duplicates ttexalens. Not building it.
- **Manifest is always written**, no env gate. A hang on a job that forgot a flag must still be debuggable.
The doc's reasons for gating (partial commit, concurrent clobber) are handled by atomic `tmp + rename`
and by deleting the stale file at the start of init.
- **JSON snapshot keyed by** `(mesh_id, chip_id, eth_chan)`, one file per host, viewer opens a folder.
Not the doc's YAML index + per-read checksums bundle.
- **Static viewer with a topology map** first. No websockets.
- **SSH /** `tt-run` **fan-out stays future work.**

Taken from the design doc:

- **Kill-first is policy, not mechanism.** With ttexalens nothing stops a live peek. Default guidance is
"kill workload, then capture" so nobody races a Metal process on a shared box, but capture must also
work on a live fabric (that is the only way to get hang-time state) and records `owner_alive` as
provenance. Capture-before-kill then capture-after-kill is also the L1-retention check the doc asks for.
- **Dump the whole** `UNRESERVED` **L1 interval per router**, once, raw. Everything in "typed L1 decode"
becomes offline slicing of bytes we already have. `.bin` sidecar per host, indexed from the JSON.
- **Liveness is a sequence, not a wall clock.** RISC reset → `EDMStatus` magic → termination word →
`go_msg` → fabric heartbeat word (`0xDCBA0000 | counter`, written every 64 loop iterations) sampled
N times, T apart. Addresses come from the manifest.
- **Pre/post stream reads bracket the L1 image.** Difference = torn; the viewer says so instead of
coloring a link from a half-moved pointer.
- **Join devices by ASIC id**, not `physical_chip_id`. Metal and ttexalens each run their own discovery.
- **Status taxonomy**: `ok | unreadable | reset | torn | unknown | unsupported`. Unknown is not failed,
and a torn value never matches a comparison.
- **Per-router ABI is a region tree from the builder's finalized allocation**, not a flat dump of
`named_args`. Regions have `parent`, `backing`, `address/size`, optional `count/stride`, `enabled`,
`schema`. Identical trees intern under a `layout_id`. Fabric-wide constants (`channel_buffer_size`,
packet header size, routing mode) live once under `fabric_context`.
- **One manifest write, after `wait_for_fabric_router_sync`.** Runtime-init failures are out of scope, so a
topology-only file has no consumer. The previous manifest is deleted at the start of fabric init, giving
the invariant: file present ⇔ the last init in this cwd reached `READY_FOR_TRAFFIC`. No staged in-memory
object in task 1 (everything is derivable at the write point). Task 2 adds a host-side container on
`FabricContext` (a `std::vector` indexed by chip id) only because each router's finalized
`FabricEriscDatamoverConfig` is built and discarded on a parallel compile worker, so the builder has to
leave a copy behind for the serializer. Nothing is ever written to device for this tool.
- **Enum tables exported into the manifest** (`EDMStatus`, `RouterState`, connection state). Viewer maps
raw → name; unknown raw stays raw.
- **Payload is never decoded by default.** Headers on demand, payload only in an expert hex view.
- **Telemetry region is dumped as a sibling provider**, never extended for debug.
- **No `launch_id`.** It would pair snapshot *file* to manifest *file*, not manifest to *device*. File
pairing: the snapshot records the manifest's `sha256`. Device pairing: capture reads
`routing_l1_info_t.my_mesh_id / my_device_id` from each router and decode compares to the manifest.
Re-init between hang and capture reprograms the routers anyway, so there is no hung state left to
mis-pair.

Explicitly not doing now (doc's post-M1 list, and ours): debug-snapshot counters / tiers, scratchpad,
connection-event ring, packet trace, rule pack, compare sessions, live poll. The region tree must be able
to express those as more named regions when they arrive; that is the only accommodation.

---



## Task breakdown

Status legend: **done** = on this branch and validated on T3K; **next** = the immediate work; the rest
is ordered but not scheduled. Code root is `tt_metal/fabric/debug/visualizer/`:

```
tt_metal/fabric/debug/           # pre-existing ERISC dumper / constants / analyzer (untouched)
  visualizer/
    README.md
    schema/    fabric_debug_manifest_schema.json, fabric_debug_snapshot_schema.json, fabric_debug_decoded_schema.json
    capture/   manifest.py, peek.py, snapshot.py, cli.py, tests/       # done
    decode/    (task 4, done)
    viewer/    (task 5, next)
    fixtures/  (task 5)
```

Suggested order: 1 → 2 → 3 → 4 → 5 → 6 → 7. Tasks 1–3 are all "make the manifest and the raw
capture complete enough that nothing later needs a re-capture." Task 5 (viewer) can start against
fixtures as soon as the decoded schema in task 4 is drafted.

### 1. Complete manifest, written once at READY_FOR_TRAFFIC (C++) — **done** (`launch_state_plan.md`)

Today: `serialize_fabric_debug_manifest_to_file(const ControlPlane&)` writes topology at the end of
`configure_routing_tables_for_fabric_ethernet_channels()`. Done and validated (`topology_manifest_plan.md`).

Change the lifecycle and extend the serializer in place:

- **Delete** the previous manifest at the start of fabric init (`FabricFirmwareInitializer::init`, under
`INIT_FABRIC`). **Write** once in `FabricFirmwareInitializer::configure()` after
`wait_for_fabric_router_sync` succeeds. Remove the control-plane hook. Same path as today.
- Always on. No env var. `tmp` + `rename` so a kill during the write leaves no half-file.
- No new in-memory object: at the write point `ControlPlane`, `FabricContext` and `FabricBuilderContext`
are alive and final, and the builder context already persists per-chip master router channels.
- The serializer has one caller (the initializer, after sync) and `TT_FATAL`s without a builder context.
No partial manifests, no `complete` flag.
- Manifest additions (schema stays version 1):
  - `run.written_at` (human provenance only)
  - `hal`: `unreserved`, `go_msg`, `launch`, `fabric_telemetry`, `routing_table`, `router_state`,
  `router_command`, `eth_fw_mailbox` `{base, size}` for `ACTIVE_ETH`
  - `heartbeat`: address / magic / mask / period, from constants lifted out of the router kernel into
  `fabric_common.h`
  - `fabric_context`: topology, 2D flag, header / payload / channel-buffer sizes, route-buffer size
  - `router_template`: `edm_status_address`, termination, local sync, handshake, ack addr, diagnostics
  map, `addresses_to_clear` (fabric-wide, identical on every router)
  - `stream_assignment` per mesh from `StreamAssignment::named_args()` — named streams now, ahead of
  the region tree
  - `enums`: `EDMStatus`, `TerminationSignal`, `RouterCommand`, `RunMsg`
  - `chip.master_router_chan`; `layouts: {}` placeholder for task 2
- Snapshot gains `manifest.sha256`; `manifest.py` parses the new blocks.
- Test: delete the standalone `ControlPlaneFixture` test (its control plane never compiles routers).
Replace with `test_fabric_debug_manifest.cpp` on `Fabric1DFixture` / `Fabric2DFixture`, which run the
real `MetalContext` → `FabricFirmwareInitializer` path, then compare the file on disk field-by-field
against the live `ControlPlane` / `FabricContext` / `FabricBuilderContext` in the same process.



### 2. Per-router region tree (C++, EDM builder) — **done** (`region_tree_plan.md`)

Replaces the earlier "dump `named_args`" idea. Same producer, structured output. Task 1 is done.

Findings from reading the builder that the task plan works around: router builders are destroyed
on the per-device compile worker, so the tree is published from `create_kernels` into a per-chip
slot on `FabricBuilderContext` and interned at serialize time; the kernel's `HANDSHAKE_ADDR` is
`round_up(UNRESERVED base, 16)`, not `config.handshake_addr` (task 1 exported the latter — fixed in
task 2), and on BH it overlaps `perf_telemetry`; `edm_channel_ack_addr` is never read by the kernel.

- After `FabricEriscDatamoverConfig` is finalized for a router, walk its allocation (bump allocator +
channel allocator + HAL-fixed siblings + overlay streams) and emit named regions:
`id, parent, backing (unreserved_l1 | fixed_l1 | stream_reg), address, size, [count, stride], allocated, enabled, writer, schema`.
- Regions cover: lifecycle/status words, termination, per-sender / per-receiver channel control and
connection info, ack/completion counters (WH stream-backed vs BH packed L1, expressed as different
`backing`), channel buffer rings (`count`, `stride = channel_buffer_size`; header/payload split is
a slice using `fabric_context`), diagnostics (perf / profiling / trimming as rows, not the tree),
telemetry (HAL sibling), `unused` / `padding`.
- Do **not** grow `FabricRouterDiagnosticBufferMap`; separate type, same source data.
- Intern identical trees under `layout_id`; router rows bind `(mesh_id, chip_id, eth_chan)` → `layout_id`
plus instance fields (direction, peer, erisc ownership). Galaxy manifests stay small.
- Group by observable effect (credit transport, ack vs fused completion, channel set, routing decode,
buffer geometry), not one region per CT `if constexpr`.
- Validation result: builder addresses are `TT_FATAL`-checked against the CT args the kernel receives,
and `DebugManifestMatchesLiveFabric` (1D + 2D) is green on WH T3K and on BH p300 / loudbox. The
dumper's `FABRIC_STREAM_GROUPS` map turned out to be **wrong** for this fabric, not a reference:
`StreamAssignment` packs from the worker free-slots pin at id 0 and allocates only what the topology
needs (WH 1D: 0–1 sender free slots, 2 receiver pkts-sent, 3–4 acked, 5–6 completed, 7–10 VC0
downstream edges, nothing above 10). Capture must drive stream selection from `layouts`, not from
that table.



### 3. Capture: raw image, liveness, identity (Python + ttexalens) — **done** (`capture_extensions_plan.md`)

Done (`capture_plan.md`): manifest load, router enumeration from `is_local` chips, coordinate
verification, reset + wall clock, streams 0–29, snapshot schema, CLI, tests, 40/40 routers on T3K.

Add, in this order:

- **Device join by ASIC id.** `peek.py` currently indexes `context.devices[physical_chip_id]`. Build a
`unique_id → device` map from ttexalens's cluster descriptor and join on the manifest's `asic_id`;
keep `physical_chip_id` as a cross-check and record disagreement as a warning.
- **Liveness sequence per router**, addresses from the manifest `hal` block:
  1. RISC reset state → if asserted, status `reset`, still dump bytes.
  2. `EDMStatus` word → enum or `unknown`.
  3. termination-signal word.
  4. `go_msg` (only meaningful when reset clear and `EDMStatus` is a real enum).
  5. fabric heartbeat word sampled `--liveness-samples N` (default 3) at `--liveness-interval` (default
    1 s). One round peeks every router once, then waits; do not serialize a full cadence per router.
     Record raw samples; decode classifies `advancing` / `static` / `not_fabric_format`.
- **Per-router read recipe:** pre-stream registers → full `[unreserved_base, +size)` read → post-stream
registers. Pre ≠ post is recorded, not hidden.
- **Raw sidecar.** L1 images go to `fabric_debug_snapshot_rank_<r>_of_<n>.bin`; each router's snapshot
entry gets `l1_image: { file, offset, size, sha256 }`. JSON stays small; Galaxy fits.
- **Also read** the fixed siblings named in the manifest: telemetry region, `routing_l1_info_t`
(and 2D path / exit-node tables where present).
- **Status enum** on every router and every provider read: `ok | unreadable | reset | torn | unknown | unsupported`, plus a free-text `error`. Replace the current `ok: bool`.
- **Provenance:** `owner_alive` (best effort: `/dev/tenstorrent/`* open by another pid; warn, do not
refuse), manifest `sha256`, ttexalens version, capture wall time. Read
`routing_l1_info_t.my_mesh_id / my_device_id` per router so decode can confirm the manifest describes
this device.
- **Multi-sample** stays out; `samples[]` remains length 1. Liveness samples are their own small array,
not `samples[]`.
- Validation on T3K: 40 routers; every image is `unreserved_size` bytes; `EDMStatus` decodes to
`READY_FOR_TRAFFIC` on an idle fabric; heartbeat advances on idle routers; kill the owning process
without `close_device`, capture again, and confirm the image is neither zeroed nor reset (this is the
post-kill retention check — write the result into the README).



### 4. Decode (Python, offline) — **done** (`decode_plan.md`)

Findings from reading the captures and the kernel that the task plan works around: the WH heartbeat
word at `0x1F80` is shared with the ETH base firmware (`0xABCD….` samples interleave with `0xDCBA….`),
so liveness is classified per sample and "static" is equality over fabric-format samples only; ring
slot positions are not in L1 (cursors are kernel locals / persisted only on close), so occupancy is a
count from the stream registers and slots carry "last header written" with `slot_state: unknown`.

New package `visualizer/decode/`. Input: a directory of manifests + snapshots (+ `.bin`). Output:
`decoded.json`, the only thing the viewer reads.

- Merge across ranks by `(mesh_id, chip_id, eth_chan)`; take chip/router detail from the file with
`is_local: true`; refuse to mix files whose `run.arch` / `fabric_config` disagree; verify each
snapshot's `manifest.sha256` against the manifest it sits next to; flag routers whose
`routing_l1_info_t` identity disagrees with the manifest; report missing ranks as coverage holes.
- Slice each L1 image by the router's `layout_id` tree; attach the typed value of every leaf region
(u32 words, enums via the manifest tables, pointers, counters). Unknown raw stays raw.
- Ring occupancy: use pointers / free-slot state to mark each slot `occupied | free | torn | unknown`;
decode packet **headers** of occupied slots (dest, hop index, command bits) using `fabric_context`;
never payload.
- Credits: one normalized `free_slots` / `credits` per channel regardless of backing (stream reg on WH,
packed L1 on BH), polarity applied here. This is where the stall score is computed; the viewer just
colors.
- Liveness classification from raw samples; torn flag from pre/post stream diff.
- Fixture-driven tests: fixture manifest + synthetic `.bin` → known `decoded.json`. No hardware.
- Output schema `fabric_debug_decoded_schema.json`, versioned like the others.



### 5. Viewer (static HTML + JS) — **next** (`viewer_plan.md`)

Reads **only** `decoded.json` (file picker; optional loopback HTTP for committed fixtures). No
manifest/snapshot/`.bin` in the browser. Vanilla HTML + JS, no bundler.

Two halves plus the decode summary the map needs:

- **Map:** SVG; chip grid from `mesh_coord`; wrap arcs from `links[].wrap` (never inferred); drill
  mesh → chip → port; edge colour from `stall_score`; grey / hatch / outline from `capture.status`.
- **Selection summary:** `channels`, occupancy counts, lifecycle, liveness. Knows decoded schema keys,
  not stream ids.
- **Generic region tree:** group by `parent`; new regions appear with no UI change. Expert hex only
  when the file was decoded with `--expert-raw`. Ring slots stay `unknown`; occupancy is a count bar.

Committed small fixtures (1D, 2×2, torus wrap, two-mesh, stalled link, coverage holes). T3K 4 MB decode
is opened by picker, not checked in. Task 6 (walk / diff / observations) stays later.



### 6. Hang UX

After the map and panel work on real captures:

- Backpressure walk: from a stalled edge, follow the edge list upstream and highlight the chain of
routers whose downstream credits are exhausted.
- Two-capture diff (before-kill vs after-kill, or two captures a minute apart): show which words moved.
- Ranked observations list ("credits exhausted on X→Y, receiver occupied 8/8, heartbeat static") as
observations, never as root cause claims. Data-driven rule pack is later.
- Optional recording: capture CLI `--samples N --interval T` filling `samples[]`; viewer slider.



### 7. Later / blocked

- Cluster gather over SSH / `tt-run` fan-out. Per-host identity is `run.mpi_rank` + `host_rank` inside the
  files, so this is unblocked but still not scheduled.
- Fallback ABI from the compiled ELF in the metal cache when a capture arrives with a topology-only
manifest.
- Debug counters / scratchpad / connection ring / packet trace: additional named regions in the tree;
nothing else in this tool changes.
- Live poll into the viewer.
- Blackhole validation of every step (untested here; 2-ERISC ownership and packed L1 credits are the
known differences and are expressed via `backing`).

---



## Language stack


| Piece        | Language               | Why                                                                              |
| ------------ | ---------------------- | -------------------------------------------------------------------------------- |
| Launch state | C++ on `FabricContext` | ControlPlane + builder already live here; nlohmann json already linked           |
| Schemas      | JSON Schema in-repo    | Shared ABI: manifest, snapshot, decoded                                          |
| Capture      | Python 3 + ttexalens   | Out-of-process, tunnels to non-MMIO chips, already proven by the dumper          |
| Decode       | Python 3, no device    | Offline; owns all fabric knowledge; fixture-testable                             |
| Frontend     | Static HTML + JS       | Map + generic region panel; view `decoded.json` with no cards / no Metal process |


Not a C++GUI. Not a C++ collector. Not a browser connected to ttexalens.

---



## Non-goals (this tool, current scope)

- Live streaming from cluster to browser
- Decoding or displaying packet payload by default
- Using ttexalens idle/active as the router set
- Encoding torus wrap only in the UI without dumping those edges
- One CT-flag checkbox per kernel `if constexpr`; one region per `if constexpr`
- Extending `FabricTelemetry` or `FabricRouterDiagnosticBufferMap` for debug
- Env-gating the manifest
- Refusing to capture because another process holds a chip (warn + record, do not abort)
- Turning a torn or incomplete read into a root-cause claim
- **SSH/SCP gatherer**: multi-host *viewing* is in scope (N per-host files into one decode), automating the
copy is not.

---



## Future work

- **Cluster gather** (`--hosts h0,h1` or `tt-run` fan-out): SSH to each host, run capture, `scp` into one
directory, run decode. Pure orchestration; ttexalens still runs on the box that owns the ASICs.
Identity is `mpi_rank` + `host_rank` inside the files, so this is unblocked.
- New router instrumentation (debug counters, scratchpad, connection ring, packet trace) as additional
named regions.
- Live poll into the same viewer with the same color function.

---



## Open questions

- ~~Task 2's host-side per-chip container for region trees~~ — resolved in `region_tree_plan.md`:
`FabricBuilderContext::router_debug_instances_`, per-chip slot written by the compile worker, interned at join.
- ~~How the builder exposes its finalized allocation walk without duplicating the address math~~ — resolved:
the producer reads addresses/stream ids from the named CT-arg maps `create_kernel` already builds and
`TT_FATAL`s them against the config fields; rings come from the channel allocator getters.
- Two unrelated processes with the same cwd overwrite each other's manifest. `sha256` in the snapshot
makes it detectable at decode; is that enough, or does the filename need a pid?
- `owner_alive` detection: `/dev/tenstorrent` fd scan vs nothing. Heuristic is acceptable.
- Blackhole: UNRESERVED end (`MEM_ERISC_MAX_SIZE`), 2-ERISC ownership fields on the region tree, heartbeat
address alias with `eth_status_t.heartbeat[0]`. Untested here.
- `decoded.json` size on Galaxy once every slot header is decoded; may need per-router lazy files.

