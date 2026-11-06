# Describe

`fabric_manifest_describe.cpp` describes the blocks a manifest states once, rather than per router: the run, the
fabric settings and the architecture. It also describes each mesh's credit transport for the topology pass. Each
`describe_*` is a short read of something process-wide at write time, with no joining to do, so they share one file.

| Function | Fills | Reads |
| --- | --- | --- |
| `describe_run(control_plane, run_id)` | `run` | The run id the writer agreed on, the MPI context, the physical system descriptor, the clock, the build's version header |
| `describe_fabric(control_plane)` | `fabric` | ControlPlane, FabricContext, the Metal environment |
| `describe_arch(hal)` | `arch` | The HAL, UMD, and fabric's heartbeat constants |
| `describe_credit_transport(builder_context, mesh_id)` | A mesh's `credit_transport` | The builder context's credit transport plan for the mesh |

Like the topology pass, describe reads each fact from the object that already knows it, and works out nothing fabric
has already decided.

## Run

`run` describes this process and the build it runs:

| Key | From |
| --- | --- |
| `run_id` | `write_fabric_manifest`: rank 0 makes it and broadcasts it ([WRITER.md](WRITER.md)) |
| `mpi_rank`, `world_size` | `DistributedContext::get_world_context()`: this process and how many there are in the whole job, not just its sub-context |
| `subcontext` | `get_current_world()` when tt-run split the job (`subcontext_id()` has a value): the sub-context's id, and this process's rank in it and its size |
| `host_name` | `PhysicalSystemDescriptor::my_host_name()`: the name discovery gave this host, the cluster descriptor's cluster id. On a mock cluster without one, it's the mock descriptor's file name |
| `written_at` | The time the manifest was written, in UTC, as ISO 8601 |
| `build.git_sha`, `build.git_sha_short`, `build.git_dirty` | The commit tt-metal was built from, and whether tracked files had uncommitted changes |
| `build.version`, `build.build_type` | The version string and the CMake build type, such as `Release` |

**Build identity.** Every `build` value comes from the version header CMake refreshes on every build
(`cmake/version.cmake`). With uncommitted changes, the SHA doesn't say exactly what code ran. `git_sha` is empty when
the source tree is neither a git checkout nor a `git archive` export. The one file that reads the header,
`common/fabric_visualizer_build_identity.cpp`, is its own library (`fabric_visualizer_build_identity`, which capture
links too), so a new commit recompiles only that file.

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
| `packet_header` | The registered name ([TYPE_REGISTRY.md](TYPE_REGISTRY.md)) of the header type `visit_packet_header_type` (`packet_header_types.hpp`) picks for FabricContext's `get_packet_header_config()` (the routing mode and its 1D extension words or 2D route buffer size), such as `HybridMeshPacketHeaderT<36>` |
| `packet_header_size_bytes` | The bytes a packet's header takes (`get_fabric_packet_header_size_bytes()`): the type's size, plus the `UDMControlFields` that follow it in UDM mode |
| `max_payload_size_bytes`, `channel_buffer_size_bytes` | The payload and channel buffer sizes |
| `routing_2d_route_buffer_size` | 2D routing only: the packet header's route buffer size |
| `routing_1d_extension_words` | 1D routing only: the packet header's extension words |
| `multi_txq` | Whether senders and receivers transmit on separate Ethernet TX queues (`FabricEriscDatamoverConfig::multi_txq_enabled`) |
| `host_clears` | What fabric init zeroes in every router's L1 before launch (`configure_fabric_cores`): each address on `get_fabric_router_addresses_to_clear()`, with the bytes of the zero buffer it writes there, `router_buffer_clear_size_words` words |
| `intermesh_vc` | `FabricBuilderContext::get_intermesh_vc_config()`: the inter-mesh VC mode (`disabled`, `edge_only`, `full_mesh` or `full_mesh_with_pass_through`), whether routers need VC1 and VC2, and whether `TT_METAL_FABRIC_DISABLE_INTERMESH_VC` put inter-mesh traffic on VC0 |

## Arch

`arch` describes what the architecture fixes on every router core, stated once since it's the same on all of them.

**L1** (`arch.l1`): the HAL's `BASE` area on an active Ethernet core, its address and size. On Wormhole that stops 32
bytes short of the core's 256 KB, which UMD keeps for its L1 barrier.

**L1 areas** (`arch.l1_areas`). Each is an L1 region, registered with the type it holds and its location from the HAL
or fabric (TYPE_REGISTRY.md's "Where regions are registered"):

| Area | From |
| --- | --- |
| `heartbeat` | The word the router writes `magic \| iteration count` to every `period_iters` main-loop iterations, from `fabric_common.h`, with its `magic`, `magic_mask` and `period_iters`. On Blackhole it's base firmware's own heartbeat word, which the router writes over while it runs. The address is picked by arch, copying the kernel's `#if ARCH_BLACKHOLE` (`fabric_erisc_router.cpp`), since no input carries it |
| `fabric_telemetry` | The HAL's `FABRIC_TELEMETRY` area, as a `FabricTelemetry`. `arch.fabric_telemetry_version` is `FABRIC_TELEMETRY_VERSION` (`fabric_telemetry_msgs.h`), which layout of the struct it holds |
| `routing_table` | The HAL's `ROUTING_TABLE` area, as a `routing_l1_info_t` |
| `eth_fw_mailbox` | The HAL's `ETH_FW_MAILBOX` area, as `uint32_t`s. Null where the HAL has no Ethernet firmware mailbox |
| `base_fw_port_status`, `base_fw_train_status`, `base_fw_retrain_count`, `base_fw_rx_link_up`, `base_fw_postcode` | The words base firmware keeps about its Ethernet link, each a `uint32_t` at the address the HAL gives among its Ethernet firmware mailbox values. Null where the HAL gives none, as on Wormhole |

A HAL area (`describe_hal_l1_area<T>`) must hold a whole number of its type, and that number is its `num_elements`.

`arch` doesn't describe stream registers. Each stream register the manifest names carries its own address and value
width, worked out from the HAL where it's read ([ROUTER_PASS.md](ROUTER_PASS.md)).

**Core registers** (`arch.core_registers`): `soft_reset`, `wall_clock_low` and `wall_clock_high`, each a `uint32_t` at
its HAL address, written with its name, address and type. The array keeps the read order: reading the low word latches
the high word. `soft_reset.reset_masks` is UMD's reset bit for each of the HAL's physical ERISCs on the Ethernet core,
one on Wormhole and two on Blackhole, whether or not the ERISC mode runs a processor on it. A router's ERISCs say which
physical ERISC each runs on ([ROUTER_PASS.md](ROUTER_PASS.md)).

## Credit transport

Each mesh on this host gets a `credit_transport`: for each VC some router in the fabric has senders on, how the
senders' credits travel. The topology pass calls `describe_credit_transport` for the mesh
([TOPOLOGY_PASS.md](TOPOLOGY_PASS.md)).

`backing` is `l1_counter` when the VC's credits travel in L1 counters, and `stream_register` otherwise. It's the
builder's own decision: the credit transport plan the builder context made for the mesh (`CreditTransportPlan`), which
has one flag per VC (`vc0_uses_counters`, `vc1_uses_counters`, `vc2_uses_counters`).

A sender channel's credits point into the counter arrays or at a stream register to match
([ROUTER_PASS.md](ROUTER_PASS.md)).

## Adding a fact stated once

1. Add it to the block's model type (`RunInfo`, `FabricInfo`, `FabricSettings` or `Arch` in
   `fabric_manifest_model.hpp`).
2. Read it in the block's `describe_*` here, from the object that already knows it. If it's an L1 area, register it
   with its type ([TYPE_REGISTRY.md](TYPE_REGISTRY.md)).
3. Write it in the writer (`fabric_manifest.cpp`), and add it to `fabric_manifest_schema.json` and
   [FORMAT.md](FORMAT.md).
4. Describe it in this doc.
