# TT Fabric Debug Visualizer

A tool for visually analyzing the state of fabric at a static point in time.

The intended hang-debug workflow is: *fabric init* writes a topology **manifest**, the *capture tool* peeks the live fabric-router cores named in that file, and a *later viewer* renders the combined artifacts. Capture does not start fabric and does not need a live host process, rather just the topology manifest from the target-device's host, and a connection to the target-device (via tt-exalens) so that we can actually extract device-state.

This README will grow with the viewer. What exists today is the schema contract and the capture tool.

## Components

```
visualizer/
  schema/     JSON schemas for the manifest and snapshot
  capture/    ttexalens peek driven by the manifest
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

#### Offline tests

```bash
python3 -m pytest tt_metal/fabric/debug/visualizer/capture/tests/ -q --noconftest
```

`--noconftest` skips the repo-root `conftest.py`, which otherwise imports `ttnn`. These tests do not need devices.

#### Live T3K checklist (slice E)

Expected on an idle Wormhole T3K after `FABRIC_2D` is up (`8` chips, `40` routers):

1. All `40` routers `status: ok`; each `blobs.unreserved.size` is `154624`; sidecar ≈ `6.1 MB`.
2. `lifecycle.edm_status` is `READY_FOR_TRAFFIC` (`2746467283` / `0xA3B3C3D3`) on every router.
3. `identity.my_mesh_id` / `my_device_id` match the manifest row for all `40`.
4. Heartbeat increases across the three liveness rounds while the owner is alive.
5. **Kill-then-capture:** `SIGKILL` the fabric owner (no `close_device`), capture again. Image must not be all-zero / reset; heartbeat must now be static. That is the hang-debug premise; record pass or fail.
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

`BaseFabricFixture` now reads `MetalContext::instance().get_cluster().arch()` instead of `get_umd_arch_name()`, which was a second UMD topology discovery.

#### Producing a manifest for a live capture

Any process that brings up fabric writes the manifest. Capture itself does not. On a T3K the smallest way to force a write is:

```bash
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal   # required if CWD is not the repo
export TT_METAL_SLOW_DISPATCH_MODE=1
./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter='Fabric1DFixture.DebugManifestMatchesLiveFabric'
```

