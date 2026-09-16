# TT Fabric Debug Visualizer

A tool for visually analyzing the state of fabric at a static point in time.

The intended hang-debug workflow is: *fabric init* writes a topology **manifest**, the *capture tool* peeks the live fabric-router cores named in that file, and a *later viewer* renders the combined artifacts. Capture does not start fabric and does not need a live host process, rather just the topology manifest from the target-device's host, and a connection to the target-device (via tt-exalens) so that we can actually extract device-state.

This README covers the schema contract, the capture tool, the offline decoder, and the static viewer.

## Components

```
visualizer/
  schema/     JSON schemas for the manifest, snapshot, and decoded state
  capture/    ttexalens peek driven by the manifest
  decode/     offline interpreter of those artifacts
  viewer/     static HTML/JS page that reads one decoded.json
```



### Dependencies

- **ttexalens.** Capture attaches to the target-device and reads ERISC health registers plus strean registers on the cores the manifest names. It does not use ttexalens to invent topology or to decide which ethernet tiles are fabric routers.
- **Standalone fabric debug helpers** in `tt_metal/fabric/debug/` (not this folder): `fabric_erisc_constants.py` and `fabric_erisc_utils.py`. Capture reuses those for overlay register addresses and architecture names so it does not copy that math. It does **not** subclass or invoke `fabric_erisc_dumper.py`.

Python 3.10+ and `jsonschema` are enough for the offline tests. Live capture additionally needs a working ttexalens install and powered devices that are not exclusively locked.

### Artifact dependencies

Capture never invents topology. It requires:

1. **A fabric debug manifest** written by each MPI rank to disk after fabric routers reach `READY_FOR_TRAFFIC`. Each MPI rank writes:
  `<logs_dir>/generated/fabric/fabric_debug_manifest_rank_<rank+1>_of_<world_size>.json`
   `logs_dir` is the process CWD unless `TT_METAL_LOGS_PATH` is set. The file lists meshes, chips, directed links, HAL/builder addresses, and the ethernet cores that are actually fabric routers (`is_local`, `physical_chip_id`, `asic_id`, `eth_chan`, logical/translated coordinates). Remote chips appear for topology but are not peek targets on this host. File present means the last fabric init in this cwd completed router sync.
2. **The JSON schemas** in `schema/`. These are the stability boundary for the tool — consumers depend on `manifest_version` / `snapshot_version` and the documented shapes, not on ControlPlane C++ signatures.
  - `schema/fabric_debug_manifest_schema.json` — topology artifact from fabric init
  - `schema/fabric_debug_snapshot_schema.json` — live peek artifact from capture
  - `schema/fabric_debug_decoded_schema.json` — typed offline state from decode

A snapshot copies the manifest's `run` identity (`arch`, `fabric_config`, `host_rank`, `mpi_rank`, `world_size`) and records `manifest.sha256` of the file bytes as read. That hash is how a snapshot pairs to a manifest after files have been copied and renamed. `run.written_at` stays on the manifest only. `host_rank` is the mesh-graph slice; `mpi_rank` is the MPI process. They are not always the same.

On a multi-host cluster, run capture once per host against that host's manifest. The viewer will merge per-host snapshots by `(mesh_id, chip_id, eth_chan)`. Automated SSH gather is future work.

### Capture tool

`capture/cli.py` is the entry point: one process, one host, one `--manifest` in, one `--output` snapshot out.

```bash
# from the tt-metal repo root
python3 tt_metal/fabric/debug/visualizer/capture/cli.py \
  --manifest generated/fabric/fabric_debug_manifest_rank_1_of_1.json \
  -o generated/fabric/fabric_debug_snapshot_rank_1_of_1.json
```

`--manifest` and `--output` are both required. Capture does not search `generated/fabric/` for a default file.

Internally the CLI is a thin wrap around four modules:

- `manifest.py` loads the JSON, checks that it is a v1 `fabric_debug_manifest`, and enumerates this host's peek targets: chips with `is_local: true`, each with a `physical_chip_id`, one target per `(mesh_id, chip_id, eth_chan)`. Duplicate endpoints are an error. Each local router must name a `layout_id` that exists in `manifest.layouts`. Remote chips stay in the file for later drawing; they are not read here. `router_layout` / `stream_regs_for_router` look up that interned region tree (allocated overlay stream ids by default; `enabled_only=True` restricts to enabled).
- `peek.py` uses ttexalens on those targets only. It joins chips by the manifest's ASIC unique id (`physical_chip_id` is only a cross-check), checks that `eth_chan` resolves to the expected logical ethernet core, then reads reset bits, lifecycle words, overlay streams, and chunked HAL-region blobs (`unreserved`, `fabric_telemetry`, `routing_table`, `go_msg`, `launch`). Streams are the layout's allocated ids (`--streams all` reads 0–31). They are read before and after the HAL image; a moved value sets `streams.torn` and router `status` `torn` unless a higher-priority status applies. Before the main pass it samples every router's heartbeat and wall clock in fleet-wide rounds (`--liveness-samples`, `--liveness-interval`). Overlay register indices come from `run.arch`; using the wrong architecture would read a real but unrelated register.
- `rawfile.py` concatenates those blobs into a sibling `.bin` (`tmp` + rename). The snapshot JSON names the file, its size, and its SHA-256; each router blob entry stores `offset` / `sha256` into that sidecar. JSON is written last so a snapshot on disk always has its bytes.
- `snapshot.py` packages peek samples into the snapshot object: file `captured_at`, manifest identity plus the manifest byte `sha256`, capture-tool/host provenance, the `.bin` sidecar reference, and one `samples[]` entry. Router `status` is one of `ok`, `unreadable`, `reset`, `torn`, `unknown`, or `unsupported`; raw values remain present when only part of a router was readable.
- `cli.py` initializes ttexalens, wires `read_from_device` into peek, writes the sidecar, then atomically writes the snapshot JSON. `--raw` overrides the sidecar path (default: output with a `.bin` suffix). `--read-chunk` sets the bulk L1 piece size (default 65536). `--no-l1-image` skips the UNRESERVED image but still captures the smaller HAL siblings. `--streams layout|all` selects allocated layout ids (default) or overlay streams 0–31.

A typical hang is assumed to leave fabric mostly stuck, so snapshots taken a bit apart on different hosts still describe the same jam. Wall clocks will move; router ids should not.

### Decode tool

`decode/cli.py` is offline. It never talks to a device. Pair snapshots to manifests by `manifest.sha256` (not by filename), merge one owner per router, and write one `fabric_debug_decoded` JSON (`tmp` + rename).

```bash
python3 tt_metal/fabric/debug/visualizer/decode/cli.py \
  generated/fabric \
  -o generated/fabric/fabric_debug_decoded.json
```

Pass directories or individual JSON files. `--slots none` skips packet-header decode (Galaxy size control); ring summaries still include occupancy counts from streams. `--expert-raw` embeds hex for captured regions up to 64 bytes. `--allow-manifest-mismatch` pairs by run identity when the exact hash is unavailable.

Decode does not invent occupancy from packet headers. Sender `occupied` is `depth - free_slots`; receiver `pkts_pending` is the `pkts_sent` stream; every ring slot stays `slot_state: unknown` because the kernel read/write indices live in RISC locals. `stall_score` is a heuristic in `[0, 1]` for later colouring, not a diagnosis: idle-healthy and hung-backpressure can look the same in one snapshot.

#### Offline tests

```bash
python3 -m pytest tt_metal/fabric/debug/visualizer/capture/tests tt_metal/fabric/debug/visualizer/decode/tests tt_metal/fabric/debug/visualizer/viewer/tests -q --noconftest
```

`--noconftest` skips the repo-root `conftest.py`, which otherwise imports `ttnn`. These tests do not need devices.

#### Live T3K checklist (slice E)

Expected on an idle Wormhole T3K after `FABRIC_2D` is up (`8` chips, `40` routers):

1. All `40` routers `status: ok`; each `blobs.unreserved.size` is `154624`; sidecar ≈ `6.1 MB`.
2. `lifecycle.edm_status` is `READY_FOR_TRAFFIC` (`2746467283` / `0xA3B3C3D3`) on every router.
3. `identity.my_mesh_id` / `my_device_id` match the manifest row for all `40`.
4. Heartbeat increases across the three liveness rounds while the owner is alive.
5. **Kill-then-capture:** `SIGKILL` the fabric owner (no `close_device`), capture again. Image must not be all-zero / reset. Heartbeat may still move: on Wormhole the fabric kernel and ethernet base firmware share the same L1 word (`0x1F80`), so a changing word is not proof the fabric heartbeat is alive. Record pass or fail.
6. Wall-clock the image pass. If it is minutes, make `--no-l1-image` the documented first look.

Hold fabric without teardown, then freeze or kill the owner:

```bash
export TT_METAL_HOME=/path/to/tt-metal
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools
python3 - <<'PY'
import time, ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))  # T3K
print("READY", flush=True)
while True:
    time.sleep(30)
PY
# after generated/fabric/fabric_debug_manifest_rank_1_of_1.json appears:
python3 tt_metal/fabric/debug/visualizer/capture/cli.py \
  --manifest generated/fabric/fabric_debug_manifest_rank_1_of_1.json \
  -o generated/fabric/fabric_debug_snapshot_rank_1_of_1.json
kill -9 <owner-pid>
# capture again to the same flags, different -o
```

**2026-09-15 (`wh-lb-83-…`, idle `FABRIC_2D` 2×4, 40 routers):**

1. Live capture: `40 ok`; every `unreserved` blob `154624` B; sidecar `6306720` B (~6.02 MiB). Wall time **3.8 s** with default 3×1 s liveness (so the image pass is ~2 s). Leave `--no-l1-image` as an opt-in.
2. `lifecycle.edm_status` is `READY_FOR_TRAFFIC` (`0xA3B3C3D3`) on all 40, live and after `SIGKILL`.
3. `identity.matches_manifest` is true on all 40.
4. Heartbeat **changes** across the three live rounds on all 40 (not a monotonic integer — the low bits move). After `SIGKILL` it still moved on 39/40 over 3 s: the ERISC kernel keeps running without a host process.
5. Kill-then-capture **retains L1**. No router in reset; no `unreserved` image is all-zero; all 40 `unreserved` SHA-256 values match the live capture. `owner_alive` was `true` while the ttnn owner ran and `null` after `SIGKILL` (the `/proc` fd scan is best-effort).
6. Image pass is seconds, not minutes.

Decode of those same artifacts (`slice_e_live.json` / `slice_e_killed.json`, 2026-09-15):

1. `40/40` routers `ok`; `exit_state` `running_or_host_gone` on all 40; routing-table identity matches the manifest row on all 40; `routing_l1_info_t.my_mesh_coord` equals `chip.mesh_coord`; `mesh_shape` is `{y: 2, x: 4}`; telemetry `static_info.mesh_id/device_id` match the endpoint.
2. Credits on the idle fabric: 160 enabled senders `occupied 0` (worker depth 14, three upstream depth 4); 40 receivers `pkts_pending 0`; 64 enabled downstream edges `free_slots 4`. All 200 packet rings `occupancy_status ok` with `occupied_count 0`. `stall_score` is `0` on every router and every outgoing link.
3. `connection_sem` is **not** all `unused` while idle: 72 `open`, 88 `unused`. Worker (and some upstream) connections stay open after fabric init; occupancy still reads empty.
4. Liveness is `advancing` on 20 routers and `insufficient` on 20 (those 20 sampled only `0xABCD…` base-FW words). None classified `static`. Live vs killed decode differs in `owner_alive` (`true` → `null`) and the heartbeat sample values; occupancy and stall scores do not change.
5. Heartbeat after `SIGKILL` still moves because the ERISC keeps running without a host process, and because that word is shared with ethernet base firmware. Kill-then-capture is still the hang-debug premise for **L1 retention**, not for a frozen heartbeat.

`BaseFabricFixture` now reads `MetalContext::instance().get_cluster().arch()` instead of `get_umd_arch_name()`, which was a second UMD topology discovery.

### Viewer

The viewer is a static page. It never talks to a device, ttexalens, or MPI, and it does not re-derive topology, credit polarity, or occupancy. Open one `kind: fabric_debug_decoded` file (`decoded_version: 1`) via the file picker, or serve the folder and choose a committed fixture.

```bash
# from the tt-metal repo root
python3 tt_metal/fabric/debug/visualizer/viewer/serve.py
# http://127.0.0.1:8765  (Cache-Control: no-store)
# fallback: python3 -m http.server --bind 127.0.0.1 8765
#   from tt_metal/fabric/debug/visualizer/viewer
```

`file://` works for the picker but cannot `fetch` `fixtures/index.json`. Committed fixtures (1D line, 2×2 mesh, torus wrap, two meshes, stalled link, coverage holes) are the UI's hardware stand-in. Do not commit a 4 MB T3K decode.

Regenerate fixtures after decode-output changes:

```bash
python3 tt_metal/fabric/debug/visualizer/viewer/tests/make_fixtures.py
python3 -m pytest tt_metal/fabric/debug/visualizer/viewer/tests -q --noconftest
```

Click a chip to open its cardinal N/S/E/W router view, select a router, and inspect its sender/receiver
buffer rows. A slot marked `H` contains a structurally plausible decoded packet header (NOC operation,
payload size, source channel, destination and route fields). It is still residual ring memory:
occupancy is `occupied/depth` from streams, while every physical `slot_state` remains `unknown`
because the live read/write indices are not captured. Raw packet payload bytes are not embedded.

`stall_score` is a colour heuristic: idle-healthy and hung-backpressure can look the same at occupancy
0. The stalled-link fixture is the hot-edge check; idle T3K is uniformly cool.

#### Open a T3K decode

```bash
python3 tt_metal/fabric/debug/visualizer/decode/cli.py \
  generated/fabric/slice_e_live.json generated/fabric/fabric_debug_manifest_rank_1_of_1.json \
  -o /tmp/decoded_slice_e_live.json
python3 tt_metal/fabric/debug/visualizer/viewer/serve.py
# pick /tmp/decoded_slice_e_live.json
```

Expected on idle Wormhole T3K (`FABRIC_2D` 2×4):

1. Coverage `40/40 ok`, header `WORMHOLE_B0` / `FABRIC_2D` / `HybridMeshPacketHeaderT<36>`.
2. One 2×4 chip grid; 40 ports; 40 edges all cool (`stall_score` 0).
3. Click a port: senders occupied 0 (worker depth 14, upstream 4), receivers pending 0, some `connection` `open`. Region tree shows named credits without the UI knowing stream ids.
4. Identity matches; `exit_state` `running_or_host_gone`; liveness `advancing` or `insufficient` — insufficient is not reset.
5. Killed decode: same map; `owner_alive` null in capture provenance.

**2026-09-15 picker pass** (`/tmp/decoded_slice_e_live.json`, same artifacts as decode slice E): coverage `ok: 40`, `stall_score` 0 on every router and link, 40 cool ports on a 2×4 grid. The stalled-link fixture is required to see a hot edge.

#### Producing a manifest for a live capture

Any process that brings up fabric writes the manifest. Capture itself does not. On a T3K the smallest way to force a write is:

```bash
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal   # required if CWD is not the repo
export TT_METAL_SLOW_DISPATCH_MODE=1
./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter='Fabric1DFixture.DebugManifestMatchesLiveFabric'
```

