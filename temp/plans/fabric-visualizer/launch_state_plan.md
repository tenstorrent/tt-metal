# Task plan — complete manifest, written once at READY_FOR_TRAFFIC

Scope: move the manifest write from "end of routing-table configuration" to "after every router on this
rank reports `READY_FOR_TRAFFIC`", delete any previous manifest at the start of fabric init, and extend
the serializer with everything capture and decode need that is *not* per-router layout: HAL fixed
addresses, fabric-wide packet/buffer constants, router-template fixed addresses, stream-register
assignment, enum tables, master/dispatch flags.

Parent plan: `visualizer_plan.md`, task 1. Depends on nothing. Task 2 (region tree) and task 3 (capture
extensions) consume what this adds.

**Decisions that shaped this revision** (superseding the earlier staged-object draft):

- Runtime init failures are out of scope, so a topology-only manifest has no consumer. **One write.**
- The previous manifest is **deleted at the start of fabric init**. Invariant: file present ⇒ the most recent
  init in this cwd reached `READY_FOR_TRAFFIC`; file absent ⇒ it did not.
- **No `launch_id`.** It would only pair snapshot file to manifest file, not manifest to device. File pairing
  is done with `manifest.sha256` in the snapshot; device pairing is done by capture reading
  `routing_l1_info_t.my_mesh_id / my_device_id` and comparing to the manifest (task 3/4).
  `run.written_at` stays in the manifest as human provenance only; it is not copied into the snapshot.
- **No in-memory launch-state object in this task.** At the write point `ControlPlane`, `FabricContext` and
  `FabricBuilderContext` are all alive and final; everything here is derivable on the spot. (Task 2 will
  need a **host-side** container — a `std::vector` on `FabricContext` indexed by chip id, same as
  `FabricBuilderContext::master_router_chans_` — because each router's `FabricEriscDatamoverConfig` is
  built and discarded inside `compile_fabric()` on a worker thread. Nothing is ever written to device for
  this tool.)
- **The serializer has one caller: the firmware initializer, after router sync.** The standalone
  `ControlPlaneFixture` test is removed; a production-path device test replaces it (§6). Consequently there
  is no "no builder context" branch and no `run.complete` flag: the serializer `TT_FATAL`s if
  `has_builder_context()` is false, because that means it was called from the wrong place.

**Explicitly deferred** (task 2): per-router region trees, `layout_id` interning, `layouts` contents,
per-router `is_dispatch_link`.

---

## 1. What the code looks like today (verified)

| Thing | Where | Consequence |
|---|---|---|
| Serializer | `fabric_host_utils.cpp:537` `serialize_fabric_debug_manifest_to_file(const ControlPlane&, path)` | Extended in place. Signature unchanged |
| Current hook | `control_plane.cpp:1267` `export_fabric_debug_manifest(*this, rtoptions_)` at the end of `configure_routing_tables_for_fabric_ethernet_channels()` | Becomes the **delete** site. The write moves |
| Router sync | `fabric_firmware_initializer.cpp:343` `configure()` → `wait_for_fabric_router_sync(...)` then `initialized_.test_and_set()` | **Write goes between those two lines.** Mock/emule return earlier and never write (correct: no router ran) |
| `FabricContext` lifetime | Built in the `ControlPlane` ctor (`control_plane.cpp:758`), only for tt-fabric configs; destroyed on teardown or when the control plane is rebuilt on config change | Alive at the write point |
| `FabricBuilderContext` | Lazy on first `get_builder_context()`; ctor reads rtoptions and trimming profiles (`fabric_builder_context.cpp:159`) | Always constructed by the time routers have compiled, which is the only time the serializer runs. `TT_FATAL(has_builder_context())` at the top of the serializer documents that |
| Router fixed addresses | `builder_context.get_fabric_router_config()` → `FabricEriscDatamoverConfig`: `edm_status_address`, `termination_signal_address`, `edm_local_sync_address`, `handshake_address`, `edm_channel_ack_addr`, `get_telemetry_and_metadata_buffer_map()`, `router_buffer_clear_size_words`; `get_fabric_router_addresses_to_clear()` | Fabric-wide ("layout is identical across all router cores"). One `router_template` block |
| Master router | `builder_context.get_fabric_master_router_chan(chip_id)` / `get_num_fabric_initialized_routers(chip_id)` | Already persisted from the compile workers; read after join. No new recording needed |
| Dispatch links | `FabricBuilder::dispatch_links_` — local to the builder, not persisted | Not available at the write point without new plumbing. **Defer to task 2's host-side per-chip container.** Not needed to debug a hang |
| Stream ids | `builder_context.get_stream_assignment(mesh_id).named_args()` → ordered `(name, id)`, fabric-wide per mesh | Emit per local mesh. Gives named streams now, before the region tree |
| HAL | `hal.get_dev_addr/get_dev_size(ACTIVE_ETH, HalL1MemAddrType::{UNRESERVED, GO_MSG, LAUNCH, FABRIC_TELEMETRY, ROUTING_TABLE, ROUTER_STATE, ROUTER_COMMAND, ETH_FW_MAILBOX})` | From `ControlPlane::hal_`; serializer already has `MetalContext::instance()`; use `MetalContext::instance().hal()` to avoid changing the signature |
| Heartbeat | `fabric_erisc_router.cpp:1892` literal `0x7CC70` (BH) / `0x1F80` (WH), `0xDCBA0000 \| counter`, every 64 iterations. Not in any host header | Lift into `fabric_common.h` (§2.4) |
| Enums | `EDMStatus`, `TerminationSignal` (`fabric_edm_packet_header.hpp`), `RouterCommand` (`fabric_common.h:809`), `RUN_MSG_GO/DONE` (`dev_msgs.h`). `enchantum` already in use in the serializer | `enum_table<E>()` helper via `enchantum::entries` |
| Existing test | `ControlPlaneFixture.TestT3kFabricDebugManifest`: standalone control plane, direct call, invariants | **Removed.** Its control plane never compiles routers, so it can only ever see a partial file. Replaced by §6 |
| Device fixtures | `tests/tt_metal/tt_fabric/common/fabric_fixture.hpp` `BaseFabricFixture::DoSetUpTestSuite`: `SetFabricConfig` → `MeshDevice::create_unit_meshes` → full `MetalContext` / `FabricFirmwareInitializer` path. `Fabric1DFixture`, `Fabric2DFixture` wrap it | The production path, for free. Test body runs after `configure()`, so the manifest is already on disk and the live `ControlPlane` / builder context are reachable through `MetalContext::instance()` for comparison |
| Schema | `manifest_version: 1`, no `additionalProperties: false` | Grow in place, stay at version 1 |

---

## 2. Design

### 2.1 Lifecycle

```text
FabricFirmwareInitializer::init()                          [fabric_firmware_initializer.cpp:307, INIT_FABRIC branch]
  ├─ remove(manifest_path)                                 # NEW: drop the previous generation's file
  ├─ write_routing_tables_to_all_chips()
  └─ compile_and_configure_fabric()                        # parallel compile, then configure
FabricFirmwareInitializer::configure()                     [fabric_firmware_initializer.cpp:343]
  ├─ wait_for_fabric_router_sync()                         # every master router READY_FOR_TRAFFIC
  ├─ serialize_fabric_debug_manifest_to_file(cp, path)     # NEW: the one write
  └─ initialized_.test_and_set()
```

The `control_plane.cpp:1267` hook is removed; the control plane no longer knows the manifest exists.

Both sites use one `fabric_debug_manifest_path(rtoptions)` helper (existing filename logic lifted out of the
anonymous namespace in `control_plane.cpp` into `fabric_host_utils.hpp`):
`<logs_dir>/generated/fabric/fabric_debug_manifest_rank_<r+1>_of_<n>.json`.

Both sites are `try`/`catch` → `log_warning`. A debug artifact never fails init.

Every fabric init (first, or a re-init after `set_fabric_config`) passes through `init()` then `configure()`,
so every generation deletes then writes. Nothing stale survives.

### 2.2 Write

Plain write via `tmp` + `rename`. Not for staging (there is none) but because the write point is also the
moment the workload is about to start; if the host is killed during the write we want no half-file.
Same directory ⇒ same filesystem ⇒ atomic replace. Five lines; keep it.

`written_at` is an ISO-8601 UTC string. It exists so a human looking at a directory of manifests and
snapshots can see when the fabric came up relative to the capture. It has no role in pairing.

### 2.3 Serializer structure

`serialize_fabric_debug_manifest_to_file` grows from one function into a handful of static helpers in
`fabric_host_utils.cpp`, each producing one top-level block:

```cpp
TT_FATAL(fc.has_builder_context(), "fabric debug manifest must be serialized after routers are compiled");
json run_block(cp, cluster);                    // existing + written_at
json hal_block(hal);                            // ACTIVE_ETH fixed regions
json heartbeat_block(arch);
json fabric_context_block(fc);
json router_template_block(builder_ctx);
json stream_assignment_block(cp, builder_ctx);
json enums_block();
json meshes_and_links(cp, cluster, builder_ctx);// existing, + master_router_chan per chip
```

No conditionals on "is the builder there". Every block is always present.

### 2.4 Heartbeat constant

In `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h`:

```cpp
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR_WORMHOLE  = 0x1F80;
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR_BLACKHOLE = 0x7CC70;
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_MAGIC          = 0xDCBA0000;
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_MAGIC_MASK     = 0xFFFF0000;
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_PERIOD_ITERS   = 64;
```

Kernel keeps its `#if defined(ARCH_BLACKHOLE)` selector and uses the named constants; host selects on
`cluster.arch()`. One definition, two selectors. Confirm the ERISC binary is unchanged after the edit.
While there, note next to the constant whether the BH address aliases `eth_status_t.heartbeat[0]` (the
design doc claims it does); do not move the address.

---

## 3. Manifest additions (schema stays version 1, all additive)

```jsonc
{
  "manifest_version": 1,
  "kind": "fabric_debug_manifest",
  "run": {
    // existing: arch, fabric_config, fabric_type, reliability_mode, tensix_config, udm_mode,
    //           host_rank, mpi_rank, world_size, local_mesh_ids
    "written_at": "2026-09-15T07:00:00Z"      // human provenance only
  },
  "hal": {                                    // ACTIVE_ETH
    "unreserved":       { "base": 0, "size": 0 },
    "go_msg":           { "base": 0, "size": 0 },
    "launch":           { "base": 0, "size": 0 },
    "fabric_telemetry": { "base": 0, "size": 0 },
    "routing_table":    { "base": 0, "size": 0 },
    "router_state":     { "base": 0, "size": 0 },
    "router_command":   { "base": 0, "size": 0 },
    "eth_fw_mailbox":   { "base": 0, "size": 0 }
  },
  "heartbeat": { "address": 8064, "magic": 3703504896, "magic_mask": 4293918720, "period_iters": 64 },
  "fabric_context": {
    "topology": "Mesh",
    "is_2d_routing": true,
    "packet_header_size_bytes": 96,
    "max_payload_size_bytes": 4352,
    "channel_buffer_size_bytes": 4448,
    "routing_2d_route_buffer_size": 36,       // xor routing_1d_extension_words
    "tensix_enabled": false,
    "bubble_flow_control": false
  },
  "router_template": {                        // identical on every router
    "edm_status_address": 0,
    "termination_signal_address": 0,
    "edm_local_sync_address": 0,
    "handshake_address": 0,
    "edm_channel_ack_addr": 0,
    "diagnostics": { "perf_telemetry": {"base":0,"size":0}, "code_profiling": {...}, "trimming": {...} },
    "addresses_to_clear": [ ],
    "router_buffer_clear_size_words": 0
  },
  "stream_assignment": { "0": { "sender_0_free_slots_id": 22, "...": 0 } },   // key = mesh_id
  "enums": {
    "EDMStatus": { "STARTED": 2695905488, "READY_FOR_TRAFFIC": 2746598355, "TERMINATED": 2763375828, "...": 0 },
    "TerminationSignal": { "KEEP_RUNNING": 0, "GRACEFULLY_TERMINATE": 1, "IMMEDIATELY_TERMINATE": 2 },
    "RouterCommand": { "RUN": 0, "PAUSE": 1, "DRAIN": 3, "...": 0 },
    "RunMsg": { "RUN_MSG_GO": 128, "RUN_MSG_DONE": 0 }
  },
  "layouts": {},                              // task 2
  "meshes": [ { "...": 0, "chips": [ { "...": 0, "master_router_chan": 5, "routers": [ ] } ] } ],
  "links": [ ]
}
```

Integers as JSON numbers (`asic_id` stays a hex string; it exceeds 2^53).

Schema: add `$defs` `l1_region`, `hal`, `heartbeat`, `fabric_context`, `router_template`,
`stream_assignment`, `enums`; all of them plus `run.written_at` and `chip.master_router_chan` **required**.
There is no partial manifest any more, so nothing is optional.

---

## 4. Snapshot side (small, belongs here because pairing needs both halves)

- `snapshot.manifest` gains `sha256` of the manifest file bytes as read. That is the file-to-file pairing.
  `written_at` is **not** copied; it adds nothing the hash does not.
- `manifest.py`: parse and expose `hal`, `heartbeat`, `fabric_context`, `router_template`,
  `stream_assignment`, `enums`. Capture does not use them yet (task 3); loading them now means the schema
  and loader move together.
- Snapshot schema updated; one test.

Device-to-manifest pairing (`routing_l1_info_t.my_mesh_id/my_device_id` vs manifest) is task 3, where L1
is read.

---

## 5. Edits

| File | Change |
|---|---|
| `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | heartbeat constants |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:1892` | use them |
| `tt_metal/fabric/fabric_host_utils.hpp` | declare `fabric_debug_manifest_path(const RunTimeOptions&)`; serializer doc comment: "call only after router sync; requires builder context" |
| `tt_metal/fabric/fabric_host_utils.cpp` | `TT_FATAL(has_builder_context())`; §2.3 block helpers; `written_at`; `tmp`+`rename` |
| `tt_metal/fabric/control_plane.cpp:120–136, 1267` | **remove** `export_fabric_debug_manifest` and its call. The control plane no longer knows about the manifest |
| `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp` | `init()`, inside the `INIT_FABRIC` branch before `write_routing_tables_to_all_chips()`: `std::filesystem::remove(path)` in try/catch. `configure()`, after `wait_for_fabric_router_sync`: `serialize_fabric_debug_manifest_to_file(control_plane_, path)` in try/catch → `log_warning`. Both use `fabric_debug_manifest_path(rtoptions_)` |
| `tt_metal/fabric/debug/visualizer/schema/fabric_debug_manifest_schema.json` | §3 |
| `tt_metal/fabric/debug/visualizer/schema/fabric_debug_snapshot_schema.json` | `manifest.sha256` |
| `tt_metal/fabric/debug/visualizer/capture/manifest.py`, `snapshot.py`, `cli.py`, tests | §4 |
| `tt_metal/fabric/debug/visualizer/README.md` | "file present ⇔ last init in this cwd reached READY_FOR_TRAFFIC" |
| `tests/tt_metal/tt_fabric/fabric_router/test_routing_tables.cpp` | **delete** `TestT3kFabricDebugManifest` |
| `tests/tt_metal/tt_fabric/fabric_data_movement/test_fabric_debug_manifest.cpp` | **new**, §6. Add to `tests/tt_metal/tt_fabric/sources.cmake` next to the other `fabric_data_movement` sources |

---

## 6. Validation — production-path test

New file `tests/tt_metal/tt_fabric/fabric_data_movement/test_fabric_debug_manifest.cpp`, built into the
existing fabric unit-test binary. Uses `Fabric1DFixture` and `Fabric2DFixture` unchanged: their
`SetUpTestSuite` is `SetFabricConfig` → `MeshDevice::create_unit_meshes`, which runs the real
`FabricFirmwareInitializer::init()` (delete) and `configure()` (write). By the time the test body runs the
file on disk **is** the production artifact, and the live objects it was derived from are still reachable.

```cpp
TEST_F(Fabric1DFixture, DebugManifestMatchesLiveFabric) { check_manifest_against_live(FabricConfig::FABRIC_1D); }
TEST_F(Fabric2DFixture, DebugManifestMatchesLiveFabric) { check_manifest_against_live(FabricConfig::FABRIC_2D); }
```

`check_manifest_against_live(expected_config)`:

1. **Existence and freshness.** `fabric_debug_manifest_path(rtoptions)` exists; `written_at` is after the
   test process start time; no `*.tmp.*` sibling in the directory.
2. **Schema.** Parses; every top-level block listed in §3 is present (`run`, `hal`, `heartbeat`,
   `fabric_context`, `router_template`, `stream_assignment`, `enums`, `layouts`, `meshes`, `links`).
3. **Run.** `run.fabric_config == enum_name(expected_config)`; `run.arch` matches
   `MetalContext::instance().get_cluster().arch()`; `mpi_rank == 0`, `world_size == 1` on this box.
4. **Live comparison (this is the point of the test).** With
   `const auto& cp = MetalContext::instance().get_control_plane();`
   `const auto& fc = cp.get_fabric_context(); const auto& bc = fc.get_builder_context();`
   - `router_template.edm_status_address == bc.get_fabric_router_sync_address_and_status().first`
   - `router_template.termination_signal_address == bc.get_fabric_router_termination_address_and_signal().first`
   - `router_template.addresses_to_clear == bc.get_fabric_router_addresses_to_clear()`
   - `fabric_context.channel_buffer_size_bytes == fc.get_fabric_channel_buffer_size_bytes()`, same for header
     and max payload; `is_2d_routing == fc.is_2D_routing_enabled()`
   - `hal.unreserved.{base,size} == hal.get_dev_addr/get_dev_size(ACTIVE_ETH, UNRESERVED)`; same for the
     other HAL regions
   - `heartbeat.address == FABRIC_KERNEL_HEARTBEAT_ADDR_WORMHOLE` on WH (BH constant otherwise)
   - for every local chip: `master_router_chan == bc.get_fabric_master_router_chan(physical_chip_id)` and
     `routers.size() == bc.get_num_fabric_initialized_routers(physical_chip_id)`
   - for every local mesh: `stream_assignment[mesh] == bc.get_stream_assignment(mesh).named_args()` as a
     map
   - `enums.EDMStatus.READY_FOR_TRAFFIC == 0xA3B3C3D3` etc., straight from the C++ enumerators
5. **Topology invariants** carried over from the deleted test: every chip local with a physical id;
   `(chip, eth_chan)` unique; every link has a reverse link with opposite direction and same routing plane;
   direction agrees with coordinate delta; no wrap on a MESH topology.
6. **Router set vs live control plane.** For each local chip, the manifest's `routers[].eth_chan` set equals
   `cp.get_active_fabric_eth_channels(node)` keys.

Running both fixtures in one binary also covers "second init overwrites": `Fabric2DFixture` runs after
`Fabric1DFixture` tears down (`SetFabricConfig(DISABLED)`), so the 2D test proves the file was replaced,
not appended to, and its `fabric_config` flipped.

Manual on T3K, once:

- Kill a fabric test with `SIGKILL` right after "Fabric Initialized"; the file still parses (rename is atomic).
- `stream_assignment["0"]` free-slot ids fall in 14–29 and ack/pkts-sent ids in 0–13, matching what
  `fabric_erisc_dumper.py` hardcodes. Any difference is drift the manifest is meant to expose; note it.
- Read the heartbeat word at `heartbeat.address` on one router with the dumper's `read_from_device`; upper
  half is `0xDCBA`.

Python: `manifest.py` tests get a fixture regenerated from a real T3K manifest (checked in, small);
loader rejects a file missing any required block; snapshot test asserts `manifest.sha256` equals a hash
computed independently in the test.

---

## 7. Risks

- **`configure()` on a partially-failed multi-device init.** `wait_for_fabric_router_sync` throws on
  timeout, so the write is never reached and the file stays absent. That is the intended invariant.
- **`TERMINATE_FABRIC`-only or "Fabric Manager" processes.** Both the delete and the write live in
  `FabricFirmwareInitializer` under `INIT_FABRIC`, so a process that did not init the fabric neither
  deletes nor writes the initializer's manifest. This is why the control-plane hook is removed rather than
  turned into the delete site.
- **Serializer called too early by a future caller.** `TT_FATAL(has_builder_context())` makes that loud
  instead of producing a partial file.
- **Two processes, same cwd.** Files clobber. Detectable by `sha256` in the snapshot not matching the file
  on disk at decode time. Accepted.
- **Kernel constant relocation.** Verify the ERISC binary hash is unchanged.
- **Test fixture skip conditions.** `BaseFabricFixture::SetUp` skips below 2 devices; fine here. The new
  test file must not introduce a fixture of its own; reuse `Fabric1DFixture` / `Fabric2DFixture` so it
  shares device bring-up with the rest of the binary.
- **Blackhole.** Untested here; `hal` uses `ACTIVE_ETH` which is where BH routers run too; heartbeat
  selector on `cluster.arch()`; the test compares against the arch-selected constant so it stays valid.

---

## 8. Step order

1. Heartbeat constants into `fabric_common.h`; kernel uses them; build; confirm binary unchanged.
2. `fabric_debug_manifest_path()` helper; delete in `init()` + write in `configure()` under `INIT_FABRIC`;
   remove the control-plane hook and the `ControlPlaneFixture` test. Build; run `Fabric1DFixture` tests;
   file appears with today's content at the new time.
3. New `test_fabric_debug_manifest.cpp` with steps 1–3, 5, 6 of §6 (what today's content supports). Passes.
4. Serializer: `TT_FATAL(has_builder_context())`; split into block helpers; add `run.written_at`, `hal`,
   `heartbeat`, `fabric_context`, `router_template`, `stream_assignment`, `enums`, `master_router_chan`,
   `layouts: {}`; `tmp`+`rename`. Extend schema. Extend the test with §6 step 4. Passes on 1D and 2D.
5. Python: `manifest.py` fields, snapshot `sha256`, schemas, tests, README, regenerated fixture.
6. Manual T3K checks from §6.
