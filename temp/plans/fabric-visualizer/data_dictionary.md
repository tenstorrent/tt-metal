# Fabric visualizer data dictionary (working draft)

Every field the visualizer pipeline produces, grouped by what it is *for* rather than by which file it lives in.

Artifacts:

- **M** = manifest (`fabric_manifest_rank_*.json`, written by fabric init)
- **S** = snapshot (`fabric_snapshot_*.json` + `.bin`, written by capture)
- **D** = decoded (`fabric_snapshot_decoded.json`, written by decode, read by viewer)

Status column: **ok** = definition is clear and the field earns its place. **?** = definition or purpose is unclear. **dup** = same fact exists elsewhere. **unused** = captured/emitted but nothing consumes it yet. **out of scope** = captured and defined, but the visualizer is not designed around it.

**Out of scope for the current design: Tensix extension modes (`MUX`, `UDM`).** When `run.tensix_config` is not `DISABLED`, Tensix helper cores (a mux on BRISC, plus a relay on NCRISC in `UDM`) sit in the data path between workers and routers and between routers of different directions. Capture does not read those cores, and receiver downstream edges may point into a mux instead of a sibling router. We keep capturing and defining the related fields (`run.tensix_config`, `run.udm_mode`, `instance.has_tensix_extension`, `instance.udm_mode`, `lifecycle.local_tensix_sync`, `credits.tensix_relay.free_slots`, `relay.`*) so the data is there later, but decode and the viewer do not interpret them, and their output for these runs should not be treated as a complete picture. The viewer should say so when a run uses either mode.

**Out of scope for the current design: host-process liveness (`owner_alive`).** The goal was to tell whether the host process that brought up the fabric is still running, which would split `exit_state: running_or_host_gone` into "workload still running" and "host crashed, fabric left up". No available signal answers that reliably today:

- The current check (`capture/snapshot.py`, `owner_alive()`) scans `/proc/<pid>/fd` on the capture host for any other process holding a `/dev/tenstorrent/*` fd. It is not tied to this run (any process using any Tenstorrent device counts), it usually returns `null` without root because other users' `fd` directories are unreadable, it sees only the container's processes inside a container, and it runs once, on the capture host only.
- Recording the manifest writer's pid and start time and checking that pid is reliable in normal runs, where one process builds fabric and runs the workload. In split fabric-manager runs (`run_fabric_manager` with `INIT_FABRIC` only), the writer exits after bring-up and workloads run in other processes, so the check reports the wrong process.
- UMD's `CHIP_IN_USE` lock, taken in `LocalChip::start_device()` and held for the owner's lifetime, is the right per-chip signal, and `LockManager::probe_mutex` reports its owner. But `tt_umd` has no Python binding for it. Even in C++ it names the owner only through the `/dev/shm` half; when only the KMD half is held (owner in another container) the owner is unknown (`{0, 0}`). Probing also briefly takes a free lock and recovers a dead owner's lock, so it is not read-only.

Until a UMD binding exists, the field stays as captured, is defined as "some other process on this host has a Tenstorrent device open", and decode and the viewer do not use it to classify routers. Possible later work: a `probe_mutex` binding in UMD for the current chip owner, plus the writer pid, start time, and fabric-manager mode in the manifest for who built fabric.

---

## Group 1 — File identity and pairing

Purpose: tie a snapshot to the manifest it was captured against, and tell multi-host files apart after they have been copied and renamed.


| Field                                                            | In               | Current meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Status / TODO                                                                                                                                                                                           |
| ---------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `manifest_version` / `snapshot_version` / `decoded_version`      | M/S/D            | Breaking-change counter, all `1`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | ok                                                                                                                                                                                                      |
| `kind`                                                           | M/S/D            | Literal file-type tag                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | ok                                                                                                                                                                                                      |
| `run.arch`                                                       | M, copied to S/D | `tt::ARCH` string; capture uses it to pick overlay register indices                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | ok                                                                                                                                                                                                      |
| `run.fabric_config`                                              | M/S/D            | `FabricConfig` string; viewer picks layout engine from it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | ok                                                                                                                                                                                                      |
| `run.fabric_type`                                                | M                | Requested wrap connectivity: which mesh axes the fabric expects to have torus links (`MESH`, `TORUS_X`, `TORUS_Y`, `TORUS_XY`). Derived from `fabric_config` by `get_fabric_type()`; `FABRIC_1D_RING` maps to `TORUS_XY` on a UBB Galaxy and `MESH` elsewhere. Input to `meshes[].torus` and `links[].wrap`; the viewer reads those instead of this field                                                                                                                                                                                                                                                         | ok                                                                                                                                                                                                      |
| `run.reliability_mode`                                           | M                | `FabricReliabilityMode`: how fabric init handled cabling that doesn't match the MGD. `STRICT_SYSTEM_HEALTH_SETUP_MODE` (default) makes a missing chip pair or short link count fatal, so every expected link is present. `RELAXED_SYSTEM_HEALTH_SETUP_MODE` skips missing pairs and accepts short counts. because routing planes need to be uniform, the minimum amount of links that can be present across all planes is used. This can make the router set smaller than what the MGD describes. `DYNAMIC_RECONFIGURATION_SETUP_MODE` is an unsupported placeholder. No visualizer consumer reads this field yet | add MGD expected link/plane counts so the viewer can tell intentional degradation from missing links. this is a build-time link measurement, does not cover links that go down during runtime of fabric |
| `run.tensix_config`                                              | M                | `FabricTensixConfig`: whether Tensix worker cores act as helpers for each ethernet router. `DISABLED` (default): the router is only the ERISC kernel and workers connect to it directly. `MUX`: one Tensix core per eth channel runs a mux (BRISC) that takes worker and cross-direction traffic in front of the router. `UDM`: the mux plus a relay (NCRISC) that receives traffic from the router. Both modes turn off VC2; dispatch links never get a helper. This is the requested mode; per-router `instance.has_tensix_extension` / `instance.udm_mode` show what each router was built with                | out of scope (see top)                                                                                                                                                                                  |
| `run.udm_mode`                                                   | M                | `FabricUDMMode` (`DISABLED` / `ENABLED`), a separate setting from `tensix_config == UDM`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | out of scope; ? how it interacts with `tensix_config`                                                                                                                                                   |
| `run.host_rank` vs `run.mpi_rank`                                | M/S              | Mesh-graph host rank vs MPI process rank                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | ok, but worth one sentence on when they differ                                                                                                                                                          |
| `run.world_size`, `run.local_mesh_ids`                           | M                | MPI world size; meshes owned by this host                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | ok                                                                                                                                                                                                      |
| `run.written_at`                                                 | M                | When this rank wrote the manifest                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | ok                                                                                                                                                                                                      |
| `captured_at`                                                    | S                | When the snapshot file started                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | ok                                                                                                                                                                                                      |
| `manifest.{path, sha256, run}`                                   | S                | Pairing key is `sha256`; path is informational                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | ok                                                                                                                                                                                                      |
| `provenance.{ttexalens_version, tt_umd_version, hostname, argv}` | S                | Capture tool and host info                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | ok                                                                                                                                                                                                      |
| `provenance.owner_alive`                                         | S                | Best-effort check, at snapshot time and on the capture host only, for any other process holding a `/dev/tenstorrent/*` fd. `true` = one was found; `false` = none, and every process was readable; `null` = none found but some processes were unreadable (usual without root). Not tied to the fabric owner, so it means "devices in use", not "owner alive"                                                                                                                                                                                                                                                     | out of scope (see top)                                                                                                                                                                                  |
| `raw.{file, size, sha256}`                                       | S                | `.bin` sidecar holding the raw L1 blobs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ok                                                                                                                                                                                                      |
| `inputs[]`, `raw_files[]`                                        | D                | Which manifest/snapshot/raw files were merged, and whether their hashes verified                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | ok                                                                                                                                                                                                      |
| `generated_at`                                                   | D                | When decode ran                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | ok                                                                                                                                                                                                      |


---

## Group 2 — Topology (what is connected to what)

Purpose: draw the chip grid and the cables. The manifest is the only source of truth; the viewer must not re-derive any of it.


| Field                                              | In  | Current meaning                                                                                         | Status                                                   |
| -------------------------------------------------- | --- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `meshes[].mesh_id`, `shape`                        | M/D | Mesh id; row-major dims (axis 0 = N/S, axis 1 = E/W)                                                    | ok                                                       |
| `meshes[].torus.{x,y}`                             | M   | Realized wrap per axis: `run.fabric_type` requests the axis AND its extent is > 2. What the viewer uses | ok                                                       |
| `chips[].fabric_chip_id`, `mesh_coord`             | M/D | Chip id within mesh; grid position                                                                      | ok                                                       |
| `chips[].physical_chip_id`                         | M/D | UMD chip id = ttexalens device id; null for remote chips                                                | ok                                                       |
| `chips[].asic_id`                                  | M/D | ASIC unique id (hex string); what capture actually joins on                                             | ok                                                       |
| `chips[].is_local`                                 | M   | This host owns the chip and can peek it                                                                 | ok                                                       |
| `chips[].master_router_chan`                       | M   | Eth channel of the chip's master router                                                                 | ? what "master router" means for debug; is it displayed? |
| `routers[].eth_chan`                               | M   | Ethernet channel; primary key within a chip                                                             | ok                                                       |
| `routers[].direction`                              | M/D | Port facing: `E W N S Z C NONE`                                                                         | ? what `C` and `NONE` mean                               |
| `routers[].routing_plane`                          | M/D | Routing plane index                                                                                     | ? not defined anywhere                                   |
| `routers[].link_class`                             | M/D | `intramesh` / `intermesh` / `unknown`                                                                   | dup with `instance.is_inter_mesh`                        |
| `routers[].logical_core`, `virtual_core`           | M   | Eth core coords; `virtual` = ttexalens "translated"                                                     | ok                                                       |
| `links[].src`, `dst`                               | M/D | One entry per *direction* of a cable (each cable appears twice)                                         | ok                                                       |
| `links[].direction`, `routing_plane`, `link_class` | M   | Copied from the source router                                                                           | dup                                                      |
| `links[].wrap`                                     | M   | Torus wrap-around edge; drawn as an arc                                                                 | ok                                                       |
| `links[].cross_host`                               | M   | Cable crosses a host boundary                                                                           | ok                                                       |
| `topology.links[].status`                          | D   | Capture status of the source router, projected onto the edge                                            | ? edge status = src status only; dst ignored?            |


---

## Group 3 — Per-router build configuration (`routers[].instance`)

Purpose: static facts about how *this* router was compiled, so decode knows which channels exist and how credits flow.


| Field                                                | Current meaning                                                                                                               | Status                                          |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `peer`                                               | Endpoint on the other end of the cable                                                                                        | dup with `links[].dst`                          |
| `is_inter_mesh`                                      | Faces another mesh                                                                                                            | dup with `link_class`                           |
| `is_dispatch_link`                                   | Link carries dispatch traffic                                                                                                 | ? what the viewer should do with it             |
| `num_active_eriscs`                                  | 1 on WH; 1 or 2 on BH                                                                                                         | ok                                              |
| `sender_channels_per_vc`, `receiver_channels_per_vc` | Enabled channel count per VC (after trimming)                                                                                 | ok                                              |
| `worker_sender_channel`                              | Flat sender index the local worker connects to                                                                                | ok                                              |
| `sender_producers[]`                                 | Per flat sender channel: `"worker"`, or the direction whose traffic feeds it                                                  | ok                                              |
| `credit_plan.vc{0,1,2}_uses_counters`                | Credits for that VC live in L1 counters instead of stream registers                                                           | ok, but "counter vs stream" needs one paragraph |
| `first_level_ack_vc0`                                | VC0 uses first-level acks (so `acked` credits are meaningful)                                                                 | ? what first-level ack is                       |
| `downstream_edm_mask_vc0/vc1`                        | Bitmask of which of the 4 downstream edges are wired                                                                          | dup with `downstream_edges_*`                   |
| `downstream_edges_vc0/vc1[]`                         | For each edge 1–4: sibling router direction and sender channel it lands in                                                    | ok                                              |
| `has_tensix_extension`, `udm_mode`                   | This router was built with a Tensix mux / UDM relay (false on dispatch links even when `run.tensix_config` is not `DISABLED`) | out of scope                                    |


---

## Group 4 — Fabric-wide constants and address maps

Purpose: numbers capture and decode need to find and interpret memory. None of these are "state".


| Field                                                                                                             | In  | Current meaning                                                         | Status                                            |
| ----------------------------------------------------------------------------------------------------------------- | --- | ----------------------------------------------------------------------- | ------------------------------------------------- |
| `fabric_context.topology`, `is_2d_routing`                                                                        | M/D | Topology string; 2D vs 1D routing                                       | dup with `run.fabric_config`                      |
| `fabric_context.packet_header_size_bytes`, `max_payload_size_bytes`, `channel_buffer_size_bytes`                  | M   | Slot geometry for packet rings                                          | ok                                                |
| `fabric_context.routing_2d_route_buffer_size`, `routing_1d_extension_words`                                       | M   | Header route-field sizing                                               | ok (used by header decode)                        |
| `fabric_context.tensix_enabled`, `bubble_flow_control`                                                            | M   | Feature flags                                                           | ? does anything consume `bubble_flow_control`     |
| `fabric_context.packet_header_type`                                                                               | D   | e.g. `HybridMeshPacketHeaderT<36>`                                      | ok                                                |
| `hal.{unreserved, go_msg, launch, fabric_telemetry, routing_table}`                                               | M   | HAL L1 regions that capture bulk-reads                                  | ok                                                |
| `hal.{router_state, router_command, eth_fw_mailbox}`                                                              | M   | HAL L1 regions                                                          | unused — listed but capture never reads them      |
| `heartbeat.{address, magic, magic_mask, period_iters}`                                                            | M   | Where the heartbeat word is and how to recognise a fabric-written value | ok                                                |
| `router_template.edm_status_address`, `termination_signal_address`, `edm_local_sync_address`, `handshake_address` | M   | Addresses identical across routers                                      | dup with `lifecycle.*` regions in the layout      |
| `router_template.unused_config_handshake_address`, `edm_channel_ack_addr`                                         | M   | Addresses of dead fields                                                | dup with `unused.*` regions                       |
| `router_template.diagnostics.{perf_telemetry, code_profiling, trimming}`                                          | M   | Diagnostic buffers                                                      | dup with `diagnostics.*` regions                  |
| `router_template.addresses_to_clear`, `router_buffer_clear_size_words`                                            | M   | What fabric init zeroes                                                 | ? debug purpose unclear                           |
| `stream_assignment[mesh_id]`                                                                                      | M   | Named stream-register ids per mesh                                      | ? overlaps with `stream_id` on each stream region |
| `enums.{EDMStatus, TerminationSignal, RouterCommand, RunMsg}`                                                     | M/D | Name tables so the viewer never guesses                                 | ok (`RouterCommand` unused?)                      |


---

## Group 5 — The region tree (`layouts[id].regions`)

Purpose: a named map of every piece of router memory we know about. This is the biggest group and probably the one that most needs definitions.

Each region carries:


| Attribute                            | Meaning                                                        | Status                                                   |
| ------------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------- |
| `id`, `parent`                       | Dotted name; tree parent                                       | ok                                                       |
| `backing`                            | `group` (folder), `unreserved_l1`, `fixed_l1`, `stream_reg`    | ok                                                       |
| `address`, `size`, `count`, `stride` | L1 placement; `count`/`stride` for arrays and rings            | ok                                                       |
| `stream_id`                          | Overlay stream index for `stream_reg` backing                  | ok                                                       |
| `allocated`                          | Memory exists in this build                                    | ok                                                       |
| `enabled`                            | This build actually uses it                                    | ? how `allocated && !enabled` should be displayed        |
| `writer`                             | Who writes it: `none erisc0 erisc1 any_erisc host peer worker` | ok, but "peer" vs "worker" needs a definition per credit |
| `schema`                             | How to decode the bytes                                        | ok                                                       |
| `overlaps`                           | Other region ids sharing the same bytes                        | ok                                                       |


Regions emitted today, by folder:

**5a.** `lifecycle` — router bring-up/teardown words


| Region                         | Schema              | Question                                    |
| ------------------------------ | ------------------- | ------------------------------------------- |
| `lifecycle.handshake`          | `handshake_info_t`  | What does a hung-state handshake look like? |
| `lifecycle.edm_status`         | `EDMStatus`         | ok                                          |
| `lifecycle.termination_signal` | `TerminationSignal` | ok                                          |
| `lifecycle.local_sync`         | u32                 | ? what syncs on it                          |
| `lifecycle.local_tensix_sync`  | u32                 | out of scope (enabled only with MUX/UDM)    |
| `lifecycle.heartbeat`          | `heartbeat_word`    | Shared with base firmware on WH             |


**5b.** `diagnostics` — optional instrumentation buffers: `perf_telemetry`, `code_profiling`, `trimming`. Only enabled when the matching compile flag is set. ? Are these ever decoded, or only carried as raw bytes?

**5c.** `credits` — flat flow-control storage


| Region                                            | Backing                              | Question                                       |
| ------------------------------------------------- | ------------------------------------ | ---------------------------------------------- |
| `credits.to_sender_ack` / `to_sender_completion`  | L1 counter array, writer `peer`      | Overlaps per-sender `sender.N.credits.*`       |
| `credits.receiver_ack` / `receiver_completion`    | L1 counter array, writer `any_erisc` | ? what these mean and whether decode uses them |
| `credits.downstream.vc{0,1}.edge{1-4}.free_slots` | stream                               | Free slots I have in the next-hop sender       |
| `credits.vc2_receiver.free_slots`                 | stream                               | ? what VC2 is                                  |
| `credits.tensix_relay.free_slots`                 | stream                               | out of scope (UDM only)                        |


**5d.** `sender.N` — one folder per flat sender channel


| Region                                 | Schema                         | Question                            |
| -------------------------------------- | ------------------------------ | ----------------------------------- |
| `sender.N.ring`                        | `packet_ring`                  | ok                                  |
| `sender.N.free_slots`                  | stream                         | ok                                  |
| `sender.N.credits.acked` / `completed` | stream or L1 counter           | ? what "pending" means once decoded |
| `sender.N.control.buffer_index`        | u32                            | ?                                   |
| `sender.N.control.buffer_index_sem`    | `SenderChannelProducerCursor`  | ? vs `buffer_index`                 |
| `sender.N.control.conn_info`           | `EDMChannelWorkerLocationInfo` | Who is connected to this channel    |
| `sender.N.control.flow_semaphore`      | u32                            | ?                                   |
| `sender.N.control.connection`          | u32 → `CONNECTION_STATE`       | ok (`open` / `unused` …)            |
| `sender.N.control.termination_status`  | u32                            | ?                                   |


**5e.** `receiver.N` — `receiver.N.ring` (packet ring), `receiver.N.pkts_sent` (stream), plus `receiver.downstream_teardown_sem.{0..}` (u32, ? purpose).

**5f. Other folders** — `relay.connection_buffer_index` (UDM, out of scope), `misc.notify_worker_src` (BH only, ? purpose), `unused.config_handshake`, `unused.edm_channel_ack`, `hal.telemetry`, `hal.routing_table`, `padding.N` (gaps in UNRESERVED that no one owns).

---

## Group 6 — Capture health (can we trust this router's bytes?)


| Field                                                                | In                                     | Current meaning                                                                       | Status                                                                |
| -------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `status`                                                             | S/D (router, blob, region, ring, link) | `ok unreadable reset torn unknown unsupported` (+ `not_captured`, `unallocated` in D) | ? `unknown` vs `unsupported` not defined; one enum reused at 5 levels |
| `error`                                                              | S/D                                    | Human text when not ok                                                                | ok                                                                    |
| `asic_id_matches_physical`                                           | S                                      | ASIC-id lookup and physical-id lookup found the same device                           | ok                                                                    |
| `identity.{my_mesh_id, my_device_id, matches_manifest}`              | S/D                                    | What the router's own routing table says it is, vs manifest                           | ok                                                                    |
| `health.reset`, `reset_bits.{erisc0, erisc1}`                        | S                                      | ETH_RISC_RESET register and decoded bits                                              | dup-ish with `status: reset`                                          |
| `health.wall_clock`                                                  | S                                      | Chip wall clock at the health read                                                    | dup with `liveness[].wall_clock`                                      |
| `streams.{pre, post, torn}`                                          | S                                      | Stream values before/after the L1 image; `torn` if any moved                          | ok                                                                    |
| `blobs[name].{address, size, offset, sha256, status}`                | S                                      | Where each raw L1 read lives in `.bin`                                                | ok                                                                    |
| `capture.{snapshot_index, torn, owner_alive, manifest_sha_verified}` | D                                      | Per-router roll-up of the above; `owner_alive` is copied from `provenance` unchanged  | ok; `owner_alive` out of scope (see top)                              |
| `coverage.*`                                                         | D                                      | Counts of routers by status, plus `identity_mismatch`, `manifest_sha_unverified`      | ok                                                                    |


---

## Group 7 — Lifecycle state (where is the router in its life?)


| Field                          | In                  | Current meaning                                              | Status                               |
| ------------------------------ | ------------------- | ------------------------------------------------------------ | ------------------------------------ |
| `lifecycle.edm_status`         | S → D `{raw, name}` | `EDMStatus` word (e.g. `READY_FOR_TRAFFIC` = `0xA3B3C3D3`)   | ok                                   |
| `lifecycle.termination_signal` | S → D               | `TerminationSignal` word                                     | ok                                   |
| `lifecycle.go_signal`          | S → D               | High byte of `go_msg_t` word 0, named from `RunMsg`          | ? what it tells us on a fabric ERISC |
| `lifecycle.exit_state`         | D                   | Derived from `edm_status` + `termination_signal` (see below) | ? names need agreement               |


Current `exit_state` rules in `decode/liveness.py`:

- `wiped_or_never_ran` — `edm_status` is 0 or not a known name
- `orderly_exit` — `TERMINATED` and a nonzero termination signal
- `teardown_stuck` — `READY_FOR_TRAFFIC` and a nonzero termination signal
- `running_or_host_gone` — `READY_FOR_TRAFFIC` and no termination signal
- `initializing` — any of the bring-up statuses (`STARTED` … `INITIALIZATION_COMPLETE`)
- `unknown` — anything else

---

## Group 8 — Liveness (is the router loop still running?)


| Field                                     | In  | Current meaning                                                                             | Status                                                                                                           |
| ----------------------------------------- | --- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `liveness[].{t, heartbeat, wall_clock}`   | S   | Several heartbeat reads spaced in time                                                      | ok                                                                                                               |
| `liveness.samples[].format`               | D   | `fabric` (matches `heartbeat.magic`), `base_fw` (`0xABCD....`), `other`                     | ok                                                                                                               |
| `liveness.{fabric,base_fw,other}_samples` | D   | Counts by format                                                                            | ok                                                                                                               |
| `liveness.classification`                 | D   | `reset`, `unknown` (unreadable), `insufficient` (< 2 fabric samples), `static`, `advancing` | ? on WH, 20/40 T3K routers were `insufficient` because base FW owns the word; is the heartbeat meaningful there? |


The router has two kinds of heartbeat, and they answer different questions.

**Loop heartbeat** (`lifecycle.heartbeat`, the word the rows above sample). `fabric_erisc_router.cpp` writes `FABRIC_KERNEL_HEARTBEAT_MAGIC | counter` every `FABRIC_KERNEL_HEARTBEAT_PERIOD_ITERS` main-loop iterations, whether or not the iteration did any work. It answers "is the router kernel still looping?" It keeps advancing through a deadlock or a dead link, so it cannot detect a stall.

On Wormhole the word (`0x1F80`) is shared with ethernet base firmware, which writes `0xABCDxxxx` there when the router context-switches to it. This has three consequences:

- A changing word does not prove the fabric kernel is alive. Base firmware keeps writing it after the fabric kernel exits or hangs, which is why the heartbeat still moved after `SIGKILL`.
- Only samples that land on a fabric value (`0xDCBAxxxx`) count. The fabric writes once per 64 iterations and base firmware overwrites in between, so routers that context-switch often (likely idle ones, not verified) collect too few fabric samples and classify as `insufficient`.
- The value is `0xDCBA0000 | counter`: the top 16 bits are always `0xDCBA` (`FABRIC_KERNEL_HEARTBEAT_MAGIC`, mask `0xFFFF0000`), which is how decode tells a fabric write from a base firmware one. The bottom 16 bits are a `uint16_t` loop counter written only on multiples of 64, so there are 1024 distinct values and the counter wraps every 65536 iterations. The value is not monotonic because of that wrap, which likely happens many times between samples a second apart (loop rate not measured), and because base firmware overwrites it in between. Decode only checks whether fabric values changed; a wrap can land on the same value, a roughly 1-in-1024 false `static` per sample pair.

Blackhole uses a separate address (`0x7CC70`) that base firmware does not write. But both fabric ERISCs write that one word with their own counters, so samples interleave two sequences, and one live ERISC keeps the word moving even if the other has hung. The loop heartbeat and `liveness.classification` are per core, not per ERISC. Per-ERISC coverage today is only the reset bits (`reset_bits.erisc0/erisc1`) and the telemetry entries (`dynamic_info.erisc[0..1]`), and the telemetry is read once.

If the router loop itself freezes (see the spin-wait note below), the loop heartbeat stops too. A stopped loop heartbeat with a valid fabric value is a strong signal.

**Progress heartbeats** (`hal.telemetry` → `dynamic_info.erisc[i].{tx_heartbeat, rx_heartbeat}`, one pair per ERISC). `update_telemetry()` increments each one when that side made progress or had nothing to do.


| Field                | In               | Current meaning                                                                                                                                                                                                                            | Status                                                                    |
| -------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `tx_heartbeat`       | S (raw blob) → D | Advances when a packet was sent over ethernet (across any sender channel), or no packets are there to send (no work). Stops when a sender has work but cannot send (waiting for credit returns from remote receiver, or ethernet TX stuck) | ok, but read only once per capture, so movement is invisible              |
| `rx_heartbeat`       | S (raw blob) → D | Advances when a receiver forwarded a packet to NoC/downstream, or no packets exist to send / forward. Stops when packets are waiting but cannot be forwarded                                                                               | ok, but read only once per capture                                        |
| `router_state`       | S (raw blob) → D | Mirror of `RouterStateManager.state`: `INITIALIZING RUNNING PAUSED DRAINING RETRAINING`                                                                                                                                                    | ok                                                                        |
| telemetry stats mask | —                | Compile-time `FABRIC_TELEMETRY_STATS_MASK` decides whether the fields above are updated at all                                                                                                                                             | missing: not in manifest, so "not moving" and "not enabled" look the same |


What counts as progress (`fabric_erisc_router.cpp`):

- **TX progress** (`run_sender_channel_step_impl`): a packet was handed to ethernet. All three must hold: the remote receiver has a free slot (`has_space_for_packet()`, replenished by completions from the peer), the sender channel has an unsent packet (`free_slots != num_buffers`), and the ethernet TX queue is not busy. Processing returned acks or completions is not tx progress.
- **RX progress** (`run_receiver_channel_step_impl`): a packet was forwarded out of the receiver buffer, meaning every downstream target had space and the NoC transaction ids had flushed. Sending a first-level ack back to the sender is not rx progress.

"Ethernet TX stuck" means `eth_txq_is_busy()` keeps returning true: the ERISC's single hardware transmit queue is still pushing a previous transfer (packet or credit/ack message) onto the wire. On Wormhole the check is `ETH_TXQ_CMD != 0`. A healthy queue clears in microseconds; a link that is not carrying traffic may never clear it (plausible, not verified on hardware). What happens next depends on a compile-time flag:

- `ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA` off: the send is skipped for that iteration. The TX heartbeat stops; the loop heartbeat keeps going.
- `ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA` on: the router spins inside the send until the queue frees. A queue that never frees freezes the whole loop: loop heartbeat, both progress heartbeats, and termination checks all stop. `ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK` does the same for acks and completions sent by the receiver. Neither flag is in the manifest yet.

Granularity caveats. The counters are one pair per ERISC, updated once per outer loop pass after many inner iterations, with progress OR'd across every channel:

- **Whole side, not per channel.** Any sender channel sending anything during the pass advances TX, even if another channel is stuck the whole time. The same holds for RX. Which channel is stuck comes from Group 9 (`free_slots`, `occupied`, downstream `free_slots`).
- **RX idle checks only receiver channel 0.** `receiver_idle` reads `to_receiver_packets_sent_streams[0]`. On a router with a second VC, a stuck VC1 receiver next to an empty VC0 receiver counts as idle, so RX keeps advancing. TX does not have this gap: `any_sender_channels_active` checks every sender channel.
- **Two-ERISC Blackhole.** The ERISCs typically split the work, one servicing senders and the other receivers. On the ERISC that does not service senders, `tx_progress` is always 0, so its TX counter only advances when all senders are empty; it tracks occupancy, not progress. The same applies to RX on the ERISC that does not service receivers. Only the TX counter of the sender ERISC and the RX counter of the receiver ERISC measure progress. Decode needs the per-ERISC channel assignment to tell them apart; the region `writer` field (`erisc0` / `erisc1`) could supply it.

Both progress heartbeats count idle as healthy, so a stopped one means work is pending and not moving. That gives these signatures:


| Case                                     | TX heartbeat              | RX heartbeat                                     | Downstream `free_slots` |
| ---------------------------------------- | ------------------------- | ------------------------------------------------ | ----------------------- |
| Healthy or idle                          | advancing                 | advancing                                        | > 0                     |
| Deadlock (routers on the cycle)          | stopped                   | stopped                                          | 0                       |
| Dead link (the two routers on the cable) | stopped if it had traffic | advancing (nothing arrives, so receiver is idle) | fine on its own edges   |
| Upstream of a dead link                  | stopped                   | stopped                                          | 0 toward the dead link  |


A dead link is where the backpressure trail ends: a router whose TX stopped but whose RX is still idle-advancing. A deadlock trail loops back on itself with no such router. The router loop has no link-status check, so this indirect signature is the only runtime evidence of a dead link.

Proposed changes:

- Sample `tx_heartbeat` and `rx_heartbeat` for every ERISC in each liveness round, alongside the loop heartbeat (two 64-bit reads per ERISC per round).
- In decode, classify each side as `advancing` or `stalled`. A value that did not change implies pending work, because idle also advances. Whether an advancing router is busy or idle comes from channel occupancy (Group 9), not from the heartbeat.
- Emit the telemetry stats mask in the manifest so decode can report `disabled` instead of `stalled`.
- Emit `ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA` and `ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK` in the router instance, so decode knows whether a stuck TX queue freezes the whole loop.
- On multi-ERISC routers, only classify the TX counter of the ERISC that services senders and the RX counter of the ERISC that services receivers, using the per-ERISC channel assignment.

---

## Group 9 — Credits and flow control (`channels`)

Purpose: the backpressure picture. This is what the viewer uses to show "stuck".

**Sender channel** (`channels.senders[]`)


| Field                                            | Current meaning                                                  | Status                                                        |
| ------------------------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------- |
| `index`, `vc`                                    | Flat sender index; VC it belongs to                              | ok                                                            |
| `role`                                           | `worker` or `upstream`                                           | dup-ish with `producer`                                       |
| `producer`                                       | `"worker"` or direction feeding this channel                     | ok                                                            |
| `depth`                                          | Ring slot count                                                  | ok                                                            |
| `free_slots`                                     | Raw `sender.N.free_slots` stream value                           | ok                                                            |
| `occupied`                                       | `depth - free_slots`                                             | ok                                                            |
| `acked_pending`, `completed_pending`             | Raw `acked`/`completed` stream values (null when counter-backed) | ? "pending" may be the wrong word — it's the raw stream count |
| `counters.{to_sender_ack, to_sender_completion}` | Counter values when credit plan uses L1 counters                 | ? how to compare with stream-backed case                      |
| `credit_backing`                                 | `stream_reg` or `counter`                                        | ok                                                            |
| `connection`                                     | `{raw, name}` of `control.connection`                            | ok                                                            |
| `torn`, `status`                                 | Stream moved during capture; `ok torn unknown inconsistent`      | ok                                                            |


**Receiver channel** (`channels.receivers[]`): `index`, `vc`, `depth`, `pkts_pending` (= raw `pkts_sent` stream), `torn`, `status`. ? `pkts_sent` → `pkts_pending` rename needs a stated reason.

**Downstream edge** (`channels.downstream[]`): `vc`, `edge`, `direction`, `dest_sender_channel`, `free_slots`, `depth` (always null today), `torn`, `status`. `free_slots == 0` is what the viewer calls "starved".

---

## Group 10 — Packet rings (`rings[]`)


| Field                                     | Current meaning                                                 | Status                                               |
| ----------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------- |
| `id`                                      | `sender.N.ring` / `receiver.N.ring`                             | ok                                                   |
| `depth`, `stride`                         | Slot count; bytes per slot                                      | ok                                                   |
| `occupied_count`                          | Copied from channel `occupied` / `pkts_pending`                 | dup                                                  |
| `occupancy_source`                        | `"stream"` or null                                              | ? only one real value                                |
| `occupancy_status`                        | `ok inconsistent unknown`                                       | ok                                                   |
| `slots[].index`, `raw_ref`, `payload_ref` | Slot byte ranges in `.bin`                                      | ok                                                   |
| `slots[].slot_state`                      | Always `unknown` (read/write indices live in RISC registers)    | ? field with one value                               |
| `slots[].header`                          | Decoded packet header if structurally plausible (`H` in viewer) | ? residual memory vs live packet not distinguishable |
| `slots[].error`                           | Header decode error                                             | ok                                                   |


---

## Group 11 — Region values (`routers[].regions[]` in D)

Same attributes as Group 5, plus: `status` (adds `unallocated`), `error`, `raw_ref` (pointer into `.bin`), `raw_hex` (only with `--expert-raw`, ≤ 64 B), `value` (schema-decoded payload, shape depends on `schema`). `warnings[]` on the router collects decode complaints. ? `value` has no schema per `schema` type yet.