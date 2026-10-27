# Describe

`fabric_manifest_describe.cpp` describes the blocks a manifest states once, rather than per router: the run, the
fabric settings and the architecture. It also describes each mesh's credit transport for the topology pass. Each
`describe_*` is a short read of something process-wide at write time, with no joining to do, so they share one file.

| Function | Fills | Reads |
| --- | --- | --- |
| `describe_run(control_plane)` | `run` | ControlPlane, the MPI context, the clock, the build's version header |
| `describe_fabric(control_plane)` | `fabric` | ControlPlane, FabricContext, the Metal environment |
| `describe_arch(hal)` | `arch` | The HAL, UMD, and fabric's heartbeat constants |
| `describe_credit_transport(builder_context, mesh_id)` | A mesh's `credit_transport` | The builder context's stream assignment for the mesh |

## Run

`run` describes this process and the build it runs:

| Key | From |
| --- | --- |
| `host_rank` | ControlPlane's host rank binding for this process: which part of a mesh this host owns |
| `mpi_rank`, `world_size` | The MPI context: this process and how many there are |
| `written_at` | The time the manifest was written, in UTC, as ISO 8601 |
| `build.git_sha`, `build.git_dirty` | The commit tt-metal was built from, and whether tracked files had uncommitted changes |

**Build identity.** `git_sha` and `git_dirty` come from the version header CMake refreshes on every build
(`cmake/version.cmake`). With uncommitted changes, the SHA doesn't say exactly what code ran. `git_sha` is empty when the source tree is neither a git checkout nor a `git archive` export. The one file that reads the header,
`fabric_manifest_build_identity.cpp`, is kept out of the unity build (`SKIP_UNITY_BUILD_INCLUSION` in
`tt_metal/fabric/CMakeLists.txt`), so a new commit recompiles only that file.

## Fabric

`fabric.config` is what the run asked for: the settings fabric was configured with, `SetFabricConfig`'s arguments.

| Key | From |
| --- | --- |
| `fabric_config`, `reliability_mode`, `tensix_config`, `udm_mode`, `fabric_manager` | ControlPlane |
| `router_config.max_packet_payload_size_bytes` | ControlPlane's router config. Null when the run didn't override the default |
| `num_routing_planes` | The Metal environment (below) |

**Routing planes.** `num_routing_planes` is the most routing planes fabric opens per link, or null when it opens every
available one. The environment keeps the largest count any `SetFabricConfig` call asked for, and stores "no limit" as
the `uint8_t` maximum, which the manifest writes as null. It's the limit that was asked for: a link with fewer
Ethernet cores than the limit opens only as many as it has. The routers in `meshes` show what each link got.

The rest of `fabric` is what FabricContext derived from the settings:

| Key | From |
| --- | --- |
| `topology` | The fabric topology |
| `is_2d_routing` | Whether 2D routing is on |
| `packet_header_size_bytes`, `max_payload_size_bytes`, `channel_buffer_size_bytes` | The packet header, payload and channel buffer sizes |
| `routing_2d_route_buffer_size` | 2D routing only: the packet header's route buffer size |
| `routing_1d_extension_words` | 1D routing only: the packet header's extension words |
| `multi_txq` | Whether senders and receivers transmit on separate Ethernet TX queues (`FabricEriscDatamoverConfig::multi_txq_enabled`) |

## Arch

`arch` describes what the architecture fixes on every router core, stated once since it's the same on all of them.

**L1 areas** (`arch.l1_areas`), each an address and a schema:

| Area | From |
| --- | --- |
| `heartbeat` | The word the router writes `magic \| iteration count` to every `period_iters` main-loop iterations, from `fabric_common.h`, with its `magic`, `magic_mask` and `period_iters`. On Blackhole it's base firmware's own heartbeat word, which the router writes over while it runs |
| `fabric_telemetry` | The HAL's `FABRIC_TELEMETRY` area, as a `FabricTelemetry` |
| `routing_table` | The HAL's `ROUTING_TABLE` area, as a `routing_l1_info_t` |
| `eth_fw_mailbox` | The HAL's `ETH_FW_MAILBOX` area, as `u32`s. Null where the HAL has no Ethernet firmware mailbox |
| `base_fw_port_status`, `base_fw_train_status`, `base_fw_retrain_count`, `base_fw_rx_link_up`, `base_fw_postcode` | The words base firmware keeps about its Ethernet link, each a `u32` at the address the HAL gives among its Ethernet firmware mailbox values. Null where the HAL gives none, as on Wormhole |

A HAL area must hold a whole number of its type, and becomes an array when it holds more than one.

**Stream registers** (`arch.stream_registers`): where each Ethernet stream's registers are, from the HAL: the first
stream's `base`, the `stride` between streams, and the `count` of streams fabric uses. For each register the manifest
reads (`buf_space_available` and `remote_src`), its offset within a stream's block and how many low bits hold its value.
`remote_src` is fabric's scratch register: four 6-bit fields on Wormhole and Blackhole (`fabric_stream_regs.hpp`).

**Core registers** (`arch.core_registers`): `soft_reset`, `wall_clock_low` and `wall_clock_high`, each a `u32` at its
HAL address. The array keeps the read order: reading the low word latches the high word. `soft_reset.reset_masks` is
UMD's reset bit for each of the HAL's physical ERISCs on the Ethernet core, one on Wormhole and two on Blackhole,
whether or not the ERISC mode runs a processor on it. A router's ERISCs say which physical ERISC each runs on
([ROUTER_PASS.md](ROUTER_PASS.md)).

## Credit transport

Each mesh on this host gets a `credit_transport`: for each VC some router in the fabric has senders on, how the
senders' credits travel. The topology pass calls `describe_credit_transport` for the mesh
([TOPOLOGY_PASS.md](TOPOLOGY_PASS.md)). It reads the builder context's stream assignment for the mesh:

- `backing`: `l1_counter` when the VC's credits travel in L1 counters, and `stream_register` otherwise.
- `reasons`: why the builder put the VC on counters: `multi_txq` (VC0 and VC1, when senders and receivers use
separate TX queues), `express` (VC1, when the mesh has express routing), or `no_completion_register` (VC2, whose
sender sits past the stream register table). Empty for a stream register VC.

A sender channel's credits point into the counter arrays or at a stream register to match ([ROUTER_PASS.md](ROUTER_PASS.md)).

## Adding a fact stated once

1. Add it to the block's model type (`RunInfo`, `FabricInfo`, `FabricSettings` or `Arch` in

   `fabric_manifest_model.hpp`).
2. Read it in the block's `describe_*` here.
3. Write it in the writer (`fabric_manifest.cpp`), and add it to `fabric_manifest_schema.json` and

   [FORMAT.md](FORMAT.md).
4. Describe it in this doc.
