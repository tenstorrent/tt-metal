# Fabric manifest generation

The fabric manifest is the frozen description of one fabric run: which ethernet cores are routers, where their capturable regions sit, and how the meshes and links are wired. Capture peeks the cores this file names. It does not invent topology.

The JSON schema in [`schema/fabric_manifest_schema.json`](schema/fabric_manifest_schema.json) is the contract. Consumers depend on `manifest_version` and the shapes there. This page describes how the host builds that file. The capture, decode, and viewer workflow is in [README.md](README.md).

Each MPI rank writes one file:

`<logs_dir>/generated/fabric/fabric_manifest_rank_<rank+1>_of_<world_size>.json`

`logs_dir` is the process CWD unless `TT_METAL_LOGS_PATH` is set. `fabric_manifest_path()` in `tt_metal/fabric/fabric_manifest.cpp` builds that path. `kind` is `fabric_manifest`.

## When the file is written

The manifest is collected in two steps. The router builders are destroyed when compilation returns, so the per-router snapshot has to be taken while `FabricBuilder` still owns them. The file itself is written later, after the routers have reached traffic-ready.

1. **Snapshot, during compile.** `create_and_compile_tt_fabric_program()` in `tt_metal/fabric/fabric_init.cpp` runs the existing build, then asks each router for a manifest instance before the program is compiled:

   `discover_channels` → `create_routers` → `connect_routers` → `compile_ancillary_kernels` → `create_kernels` → `build_and_publish_manifest_router_instances` → `Program::compile`

2. **Write, after router sync.** `FabricFirmwareInitializer::init()` deletes a stale manifest at that path, then compiles and configures fabric. `configure()` calls `wait_for_fabric_router_sync()`, then `serialize_fabric_manifest_to_file()`. The write is `tmp` plus rename. A file on disk means this `INIT_FABRIC` configure finished router sync.

Mock and emule skip configure, so they never write the file. `TERMINATE_FABRIC` compiles routers to set up the fabric context and does not take the write path. If serialization throws, configure logs a warning and fabric init continues.

## What the builder records

Most of the fabric build is unchanged. Three places feed the manifest.

**Connect records downstream edges.** While `connect_routers()` wires receivers to downstream senders, `StaticSizedChannelConnectionWriterAdapter::add_downstream_connection()` stores the downstream direction in a compact slot. Compact slots are 0–3 (`EDGE_1`..`EDGE_4`). The walk later stays on those four slots.

**Kernel creation finalizes the compile-time arguments.** `ComputeMeshRouterBuilder::create_kernel()` is the last builder mutation the snapshot reads. The instance is built from the named compile-time arguments already stored on the erisc datamover builder.

**The snapshot copies that state off the builders.** `FabricBuilder::build_and_publish_manifest_router_instances()` calls `make_manifest_router_instance()` on each router and publishes the vector on `FabricBuilderContext`, keyed by physical chip id. `ComputeMeshRouterBuilder::make_manifest_router_instance()` gathers the named compile-time arguments for each RISC and calls `build_manifest_router_instance()` in `tt_metal/fabric/builder/fabric_manifest_router_instance.cpp`. That function fills one `ManifestRouterInstance`: channel counts, credit plan, the region tree, and `downstream_edges_vc0` / `downstream_edges_vc1`.

Downstream edges come from `FabricEriscDatamoverBuilder::get_manifest_downstream_edges()`, which reads the static-sized connection adapter. Each `ManifestDownstreamEdge` carries a 1-based `edge` (`compact + 1`), an `eth_chan_directions` value, and a sender channel. A 1D connection that never recorded a sender channel is emitted as sender channel 1. Direction stays an enum until the JSON write, which calls `direction_to_str()`.

Publish is once per chip. A second publish for the same chip is fatal, and the instance count must match the number of routers that chip built.

## How the file is assembled

`serialize_fabric_manifest_to_file()` joins three inputs:

- **Builder instances** on `FabricBuilderContext`, one vector per local chip.
- **ControlPlane** topology: meshes, chip coordinates, active ethernet channels, peers, routing planes.
- **HAL and runtime**: architecture, heartbeat address, fabric context, the shared router template.

Serialization requires a builder context, which exists only after routers have been compiled. An active local router with no published instance is fatal. A published instance whose peer disagrees with ControlPlane is fatal.

Top-level keys, and the function that fills each one:

| Key | Source |
| --- | --- |
| `run` | `make_run_json` — arch, fabric config and type, host rank, MPI rank, world size, `written_at` |
| `hal` | `make_hal_json` — HAL region bases and sizes capture uses for bulk reads |
| `heartbeat` | `make_heartbeat_json` — architecture-specific L1 address and magic |
| `fabric_context` | `make_fabric_context_json` — topology, packet header and channel buffer sizes, tensix and bubble-flow-control flags |
| `router_template` | `make_router_template_json` — status, handshake, and diagnostic addresses shared by every router |
| `stream_assignment` | `make_stream_assignment_json` — per-mesh overlay stream ids |
| `enums` | `make_enums_json` — numeric values capture compares against, including `EDMStatus` |
| `layouts` | `build_manifest_layout_index` — interned region trees |
| `meshes` | ControlPlane mesh graph, plus one router object per active local channel |
| `links` | One directed edge per active local channel, for the viewer |

## Layouts and regions

Each `ManifestRouterInstance` carries its own `ManifestRouterRegionLayout`: a flat list of `ManifestRouterRegion` nodes with parent ids. `build_manifest_router_instance()` builds that tree with `make_group_region`, `make_l1_region`, and `make_stream_region`.

`ManifestRegionBacking` says what a node is:

- `GROUP` is a folder. `allocated` is false and the node has no address of its own. The folders are `lifecycle`, `diagnostics`, `credits`, `sender`, `receiver`, `relay`, `unused`, `hal`, and `padding`. Blackhole also adds `misc`. Sender and receiver channels are nested groups under those two roots.
- `UNRESERVED_L1`, `FIXED_L1`, and `STREAM_REG` are real storage. Capture reads those.

Identical region trees are interned when the file is written, in `build_manifest_layout_index()`. The canonical JSON of a layout is hashed with FNV-1a and stored as `L` plus 16 hex digits. Each router object holds that `layout_id` instead of a copy of the tree. `layouts[id].router_count` is how many routers share it. A hash collision on two different trees is fatal.

## Routers and links

`meshes[].chips[]` is the whole mesh graph, including chips this rank does not own. A chip is local when ControlPlane can map it to a physical device. Remote chips are still listed, with `is_local: false`, null `physical_chip_id` / `asic_id` / `master_router_chan`, and an empty `routers` array, so a later viewer can draw the host boundary.

Local `routers[]` is `ControlPlane::get_active_fabric_eth_channels()`: link-up, assigned `EthRouterMode::FABRIC_ROUTER`, and still present after routing-plane trimming. Each entry has `eth_chan`, facing `direction`, `routing_plane`, `link_class` (`intramesh` or `intermesh`), logical and virtual core coordinates, `layout_id`, and an `instance` object (channel counts, credit plan, tensix extension, and the downstream edges above).

`links[]` is emitted once per directed source channel, so a physical cable appears twice, once from each end. Each link records `src`, `dst`, `direction`, `routing_plane`, `link_class`, `cross_host`, and `wrap`. `wrap` is set when the axis is a realized torus and the coordinate delta spans the mesh.

A snapshot pairs back to this file by `manifest.sha256` of the file bytes, which capture records. Filenames are not the pairing key. See [README.md](README.md).
