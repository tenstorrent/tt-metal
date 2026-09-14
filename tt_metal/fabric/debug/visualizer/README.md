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

1. **A fabric debug manifest** written by ControlPlane at the end of `configure_routing_tables_for_fabric_ethernet_channels()`. Each MPI rank writes:
  `<logs_dir>/generated/fabric/fabric_debug_manifest_rank_<rank+1>_of_<world_size>.json`
   `logs_dir` is the process CWD unless `TT_METAL_LOGS_PATH` is set. The file lists meshes, chips, directed links, and the ethernet cores that are actually fabric routers (`is_local`, `physical_chip_id`, `eth_chan`, logical/translated coordinates). Remote chips appear for topology but are not peek targets on this host.
2. **The JSON schemas** in `schema/`. These are the stability boundary for the tool — consumers depend on `manifest_version` / `snapshot_version` and the documented shapes, not on ControlPlane C++ signatures.
  - `schema/fabric_debug_manifest_schema.json` — topology artifact from fabric init
  - `schema/fabric_debug_snapshot_schema.json` — live peek artifact from capture

A snapshot copies the manifest's `run` identity (`arch`, `fabric_config`, `host_rank`, `mpi_rank`, `world_size`) so files can still be matched after they have been copied and renamed. `host_rank` is the mesh-graph slice; `mpi_rank` is the MPI process. They are not always the same.

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

- `manifest.py` loads the JSON, checks that it is a v1 `fabric_debug_manifest`, and enumerates this host's peek targets: chips with `is_local: true`, each with a `physical_chip_id`, one target per `(mesh_id, chip_id, eth_chan)`. Duplicate endpoints are an error. Remote chips stay in the file for later drawing; they are not read here.
- `peek.py` uses ttexalens on those targets only. For each router it checks that `eth_chan` indexes an ethernet block whose logical coordinates match the manifest, then reads `ETH_RISC_RESET`, the 64-bit wall clock, and `BUF_SPACE_AVAILABLE` on streams 0–29. Overlay register indices come from `run.arch` in the manifest (wrong arch would read a real but unrelated register). Per-router failures are recorded on the sample (`ok: false` plus `error`) instead of aborting the whole capture. `capture_time` is stamped when the reads begin, not when the JSON is later assembled.
- `snapshot.py` packages peek samples into the snapshot object: file `captured_at`, a copy of the manifest run identity, and a `samples[]` array (one sample in v1). Stream ids stay numeric and opaque until a per-router ABI exists.
- `cli.py` initializes ttexalens, wires `read_from_device` into peek, and atomically writes the snapshot JSON.

A typical hang is assumed to leave fabric mostly stuck, so snapshots taken a bit apart on different hosts still describe the same jam. Wall clocks will move; router ids should not.

#### Offline tests

```bash
python3 -m pytest tt_metal/fabric/debug/visualizer/capture/tests/ -q --noconftest
```

`--noconftest` skips the repo-root `conftest.py`, which otherwise imports `ttnn`. These tests do not need devices.

#### Producing a manifest for a live capture

Any process that brings up fabric writes the manifest. Capture itself does not, and does not need `TT_METAL_SLOW_DISPATCH_MODE`. On a T3K the smallest way to force a write is:

```bash
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal   # required if CWD is not the repo
export TT_METAL_SLOW_DISPATCH_MODE=1             # required by ControlPlaneFixture only
./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter='ControlPlaneFixture.TestT3kFabricDebugManifest'
```

