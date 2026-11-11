# Manifest format

This is the reference for the manifest's JSON: every block and key, the shapes that repeat throughout, and the terms
the rest of the docs use. [`fabric_manifest_schema.json`](../fabric_manifest_schema.json) is the formal JSON Schema,
and readers depend on it and on `manifest_version`. The other docs say how each value is worked out.

## Glossary

| Term | Meaning |
| --- | --- |
| Mesh | A logical grid of chips, defined by the mesh graph descriptor. A fabric can have several, joined by inter-mesh links |
| Chip | One device in a mesh, named by its mesh and its chip id in that mesh. Keyed `C<chip id>` |
| Host rank | Which part of a mesh a host owns. A mesh can be split across hosts, each owning a block of it |
| MPI rank | The process. There's normally one per host rank, and each writes its own manifest |
| Local chip | A chip this process opened. Only local chips list their routers. Every other chip is remote |
| Router | The fabric kernel on one Ethernet core of a chip, moving packets over that core's link |
| Direction | Which way a router's link faces: `E`, `W`, `N`, `S`, or `Z` (out of the mesh's plane) |
| Routing plane | One of the parallel sets of routers a chip has per direction. A link's routers are on the same plane |
| Key | A router's name in its chip: its direction and routing plane, such as `E0` |
| Path | A router's name in the fabric: `M<mesh>/C<chip>/<key>`, such as `M0/C7/E0` |
| Peer | The router on the other end of a router's Ethernet link, usually on another chip |
| Sibling | Another router on the same chip and routing plane, facing a different direction |
| ERISC | One of the Ethernet core's RISC processors. A router's kernel runs on one or two of them |
| Physical ERISC | Which ERISC on the core a processor actually runs on, which can differ from the processor's index |
| VC | Virtual channel: an independent class of traffic through a router, each with its own channels |
| Sender channel | Where a producer (a worker, the tensix mux, or a sibling router) hands the router packets to send over the link |
| Receiver channel | Where packets arriving over the link land, before the router forwards them to a sibling or delivers them |
| Worker channel | The sender channel of a VC that a local worker or the tensix mux feeds, rather than a sibling |
| Downstream edge | A router's persistent connection to one of its siblings' sender channels, which its receiver forwards packets on |
| Live direction | A direction a 2D router forwards packets to on a VC. It ignores a packet's request for any other |
| Ring buffer | A channel's packet slots in L1, each one channel buffer in size |
| Credits | How a sender channel learns its packets were received (`acked`) or done with (`completed`) |
| Credit counter | An L1 word counting one sender channel's credits, when credits travel in L1 rather than stream registers |
| Stream register | One of the Ethernet core's hardware stream registers, which fabric uses as counters and scratch |
| Region | Router state the manifest describes: an L1 region, or a stream register. It says where the bytes are and the type they hold |
| Multi-sampled | A region capture reads in each of its rounds, as well as once in its full read. Its role says what its change across the rounds means |
| Active | Whether anything writes an L1 region on this run. An inactive region is allocated, but its bytes are whatever the L1 held |
| Type name | A registered C++ type's name, such as `EDMStatus`, which decode maps back to the type ([TYPE_REGISTRY.md](TYPE_REGISTRY.md)) |
| Field | One value a router's kernel was fed, keyed by name ([FIELD_TABLES.md](FIELD_TABLES.md)) |
| Field set | One distinct fields block, which every router, channel or edge with those same fields names by its id, such as `F0` |
| Tensix mux | A tensix core that carries VC0 traffic for a router's non-worker sender channels, in mux mode |
| Channel trimming | A profile that turns off channels a workload doesn't use |
| Bubble flow control | Deadlock avoidance on VC0, which adds first-level acks |
| Express routing | Extra links between non-adjacent chips of a mesh |
| Wrap link | A link that closes a torus axis, joining the chips at its two ends |

## Overview

```
{
  "manifest_version": 1,
  "file_kind": "fabric_manifest",
  "run":        { run_id, mpi_rank, world_size, subcontext?: { id, rank, size }, host_name, written_at,
                  build: { git_sha, git_sha_short, version, build_type, git_dirty } },
  "fabric":     { config: { ... }, topology, is_2d_routing, packet_header, ... },
  "arch":       { name, l1: { address, size_bytes }, l1_areas: { ... }, heartbeat_protocol: { ... }, core_registers: [ ... ] },
  "vocabulary": { <list name>: [ <name>, ... ] },
  "field_sets": { "F<n>": { erisc<r>: { ... } } },
  "meshes": {
    "M<mesh>": {
      shape, torus, express_routing, express_axis, credit_transport,
      "chips": {
        "C<chip>": {
          mesh_coord, host_rank, physical_chip_id, asic_id, is_local, z_port_role, local_sync,
          "routers": {
            "<key>": {
              identity, link, shape, credit_counters,
              channels: { senders: { vc<v>: { ch<c>: { ..., fields: "F<n>" } } }, receivers: { ... } },
              intra_chip_downstream_edges: { vc<v>: { edge<n>: { ..., fields: "F<n>" } } },
              live_downstreams: { vc0: [ <direction>, ... ], vc1: [ ... ] } | null,
              fields: "F<n>", eriscs: { erisc<r>: { ... } }, leftover_l1
            }
          }
        }
      }
    }
  }
}
```

`manifest_version` is `1`, and changes when readers can't take the shape as it is. `file_kind` is always
`"fabric_manifest"`, naming which of the visualizer's files this is. The file is written compactly, on one line.

## Shapes that repeat

**L1 region**: `{ address, num_elements, element_size, type, multi_sampled, active }`. `num_elements` elements of
`type`, `element_size` bytes apart from `address`. Element `i` holds a `type` in its first `sizeof(type)` bytes, at
`address + i * element_size`; the rest of the element, if `element_size` is larger, isn't typed. Whether fabric init
zeroes part of a region before launch is `fabric.host_clears`.

```json
{ "address": 88960, "num_elements": 12, "element_size": 4, "type": "unsigned int", "multi_sampled": null,
  "active": true }
```

**Stream register**: `{ stream_id, address, value_bits, type, multi_sampled }`. One word at `address`, in stream
`stream_id`'s registers. Its value is the word's low `value_bits` bits, read as `type`; the bits above can hold other
state.

```json
{ "stream_id": 7, "address": 4290016420, "value_bits": 17, "type": "unsigned int", "multi_sampled": null }
```

**Multi-sampled**: a region's or stream register's `multi_sampled`. Capture reads every region once, in its full read.
`null` means that's all; otherwise capture also reads the region in each of its rounds, before the full read, and the
value is the region's role (`multi_sample_role`), which says what its change across the rounds means:

| Role | The region | What its change says |
| --- | --- | --- |
| `loop_heartbeat` | Written every pass of the router's loop, stuck or not | Moving: the loop runs. Stopped: it doesn't |
| `fabric_telemetry` | The fabric telemetry area: each ERISC's TX and RX heartbeats and router state, and the bandwidth counts | A heartbeat moves while its side makes progress or has nothing to do, and stops when it's stuck. Which an ERISC writes is its `telemetry_enabled` and `telemetry_stats_mask` fields |

A multi-sampled region is always active.

How many rounds capture reads, and how far apart, is capture's choice, not the manifest's. Only the arch areas
`heartbeat` and `fabric_telemetry` are multi-sampled now; stream registers carry the key so one can be marked with no
change to the format.

**Active**: an L1 region's `active`. A region can go unwritten on a run in three ways, each said where the writer's
configuration lives:

| Case | Example | What the manifest says |
| --- | --- | --- |
| Not allocated | Code profiling results with profiling off (address 0) | The field is `null` |
| Allocated, never written | Blackhole's perf telemetry buffer with bandwidth telemetry off; leftover L1 | `"active": false` |
| Written, some members skipped | The telemetry area, where each ERISC writes only the statistics its telemetry options name | Nothing: the router's fields say what's written, and decode reads them |

An inactive region's bytes are still captured, so something writing it unexpectedly still shows. An arch area is always
active (`arch`, below). A stream register has no `active`.

**Type name**: the `type` of a region or stream register, or `fabric.packet_header`. It's a registered C++ type's name
without its namespaces, as the build that wrote the manifest spells it: `EDMStatus`, `HybridMeshPacketHeaderT<36>`,
`unsigned int` for `uint32_t`, `byte` for `std::byte`. Decode, built from the same commit, maps the name back to the
type and reads the bytes into it. The manifest doesn't describe a type's members ([TYPE_REGISTRY.md](TYPE_REGISTRY.md)).

**Fields**: a fields block is keyed by ERISC (`erisc<r>`), and each ERISC's entry holds that ERISC's fields, as the
builder gave its compile-time arguments. ERISCs can differ, and nothing in the manifest checks that they agree. Most
routers, channels and edges have the same fields as many others, so each distinct block is written once, in
`field_sets`, and a router's, channel's or edge's `fields` is the id of its block there.

**Field**: `{ category, kind, ... }`, keyed by the field's name within an ERISC's fields. `category` is `lifecycle`,
`kernel_params`, `flow_control`, `control_info` or `diagnostics`.

| `kind` | Keys besides `category` and `kind` |
| --- | --- |
| `l1` | An L1 region's keys |
| `stream_reg` | A stream register's keys |
| `number` | `value`, an integer |
| `flag` | `value`, a boolean |
| `enum` | `value`, an integer, and `value_name`, its enumerator's name in lower case |

A field is `null` when the builder didn't allocate what it describes (it fed the kernel address 0, or the unused stream
id). A field the builder doesn't emit in this configuration, such as a 2D-only field on 1D, is left out.
[FIELDS.md](FIELDS.md) lists every field.

**Paths.** Wherever the manifest refers to a router, or to something in one, it writes a path:

- `link.peer`, `local_sync.master` and a sibling `producer` are router paths, such as `M0/C1/W1`.
- A downstream edge's `downstream_channel` is a channel's path, such as `M0/C0/S1/channels/senders/vc0/ch1`.
- A counter-backed credit names its array by its path in the router, such as `credit_counters/to_sender_ack`.

## run

| Key | Value |
| --- | --- |
| `run_id` | The fabric run: the UTC time and 8 hex digits, such as `20261105T221500Z-1a2b3c4d`. Every rank of the fabric has the same one, and it names the folder the manifest is in |
| `mpi_rank`, `world_size` | This process's MPI rank in the whole job, and the job's number of processes |
| `subcontext` | Only when tt-run splits the job into sub-contexts (`-M`, or a rank bindings mapping), each its own fabric: `id`, and this process's `rank` within the sub-context and its `size`. Mesh and chip names are per sub-context, so files are joined within one sub-context |
| `host_name` | The host this process runs on: the cluster id UMD stamps on the cluster descriptor, which is `TT_METAL_CLUSTER_ID` when set and otherwise the OS hostname |
| `written_at` | When the manifest was written, in UTC, as ISO 8601 |
| `build.git_sha` | The commit tt-metal was built from. Empty outside a git checkout or `git archive` export |
| `build.git_sha_short` | The short form of `git_sha` |
| `build.version`, `build.build_type` | The tt-metal version string, and the CMake build type, such as `Release` |
| `build.git_dirty` | Whether tracked files had uncommitted changes, in which case the SHA doesn't say exactly what ran |

## fabric

| Key | Value |
| --- | --- |
| `config.fabric_config`, `config.reliability_mode`, `config.tensix_config`, `config.udm_mode`, `config.fabric_manager` | The settings fabric was configured with |
| `config.num_routing_planes` | The most routing planes fabric opens per link, or null for every available one |
| `config.router_config.max_packet_payload_size_bytes` | The payload size override, or null for the default |
| `topology` | The fabric topology, such as `mesh` |
| `is_2d_routing` | Whether 2D routing is on |
| `packet_header` | The type name of the packet header the run's routers use, such as `HybridMeshPacketHeaderT<36>`. Each ring buffer slot starts with one |
| `packet_header_size_bytes` | The bytes the header takes in a slot: the type's size, plus `UDMControlFields` in UDM mode, so the payload starts this far into the slot |
| `max_payload_size_bytes`, `channel_buffer_size_bytes` | Payload and channel buffer sizes |
| `routing_2d_route_buffer_size` | 2D routing only: the packet header's route buffer size |
| `routing_1d_extension_words` | 1D routing only: the packet header's extension words |
| `multi_txq` | Whether senders and receivers transmit on separate Ethernet TX queues |
| `host_clears` | What fabric init zeroes in every router's L1 before launch: `[{ address, size_bytes }]`, each the bytes it writes at that address |
| `intermesh_vc` | `{ mode, requires_vc1, requires_vc2, intermesh_vc_disabled }`: the inter-mesh VC mode (`intermesh_vc_mode`), which decides whether routers have VC1 and VC2 senders |

[DESCRIBE.md](DESCRIBE.md) says where each comes from.

## arch

| Key | Value |
| --- | --- |
| `name` | The architecture, such as `wormhole_b0` |
| `l1` | The whole L1 of the Ethernet core a router runs on: `size_bytes` bytes from `address`. Every L1 region the manifest names lies inside it |
| `l1_areas` | L1 regions the architecture fixes on every router core, one per `ArchArea` keyed by its name in lower case, in the enum's order: `heartbeat` (multi-sampled as `loop_heartbeat`), `fabric_telemetry` (multi-sampled as `fabric_telemetry`), `routing_table`, `eth_fw_mailbox`, and base firmware's link status words `base_fw_port_status`, `base_fw_train_status`, `base_fw_retrain_count`, `base_fw_rx_link_up` and `base_fw_postcode`. An area the architecture doesn't have is null. Every router core writes each area, so each is `active` |
| `heartbeat_protocol` | How the router kernel writes its heartbeat word (`l1_areas.heartbeat`): `{ magic, magic_mask, period_iters }`, the word being `magic \| iteration count`, written every `period_iters` main-loop iterations |
| `core_registers` | An array, in read order: `soft_reset` (with `reset_masks`, each physical ERISC's reset bit), `wall_clock_low` and `wall_clock_high`, each with its `name`, `address` and `type`. Reading the low wall clock word latches the high word. The model's `k_core_registers` holds this order, which the writer and the reader follow |
| `fabric_telemetry_version` | `FABRIC_TELEMETRY_VERSION`: which layout of `FabricTelemetry` the `fabric_telemetry` area holds |

## vocabulary

Every name each label can hold, so a reader knows all of a label's values without the C++ enums. Each list is named for
what the label is:

| List | Holds the names of |
| --- | --- |
| `arch` | `arch.name` |
| `fabric_config`, `reliability_mode`, `tensix_config`, `udm_mode`, `fabric_manager_mode` | `fabric.config`'s settings |
| `fabric_topology` | `fabric.topology` |
| `credit_backing` | A mesh's `credit_transport` |
| `z_port_role` | A chip's `z_port_role` |
| `edge_capability` | A link's `edge_capability` |
| `credit_counter_indexed_by` | A credit counter array's `indexed_by` |
| `field_category`, `field_kind` | A field's `category` and `kind` |
| `channel_status` | A channel's `status` |
| `erisc_processor` | An ERISC's `processor` |
| `kernel_opt_level` | An ERISC's `opt_level` |
| `intermesh_vc_mode` | `fabric.intermesh_vc.mode` |
| `multi_sample_role` | A region's or stream register's `multi_sampled` |

An `enum` field isn't a label: it writes its enumerator's name next to its value (`value_name`), so it has no list here.

## field_sets

Each distinct fields block, keyed `F<n>` in the order the writer first wrote it. Every `fields` in `meshes` is one of
these ids, and every block here is named by at least one. To read a router's, channel's or edge's fields, look its
`fields` id up here.

```json
"field_sets": {
  "F0": { "erisc0": { "conn_info": { "category": "control_info", "kind": "l1", ... }, ... }, "erisc1": { ... } },
  "F1": { "erisc0": { "free_slots": { "category": "flow_control", "kind": "stream_reg", ... }, ... } },
  ...
}
```

## meshes

### Mesh

`meshes.M<mesh>`, by mesh id.

| Key | Value |
| --- | --- |
| `shape` | The mesh's extent in each dimension |
| `torus` | 2D meshes only: `{ y, x }`, whether each axis wraps |
| `express_routing` | Whether the mesh has express routing |
| `express_axis` | The axis the express links run along, `"y"` or `"x"`: the router's `FABRIC_EXPRESS_AXIS`. `null` when `express_routing` is false |
| `credit_transport` | Meshes on this host only: for each VC some router has senders on, `vc<v>: { backing }`. `backing` is `stream_register` or `l1_counter` |
| `chips` | Every chip in the mesh, by chip id |

### Chip

`chips.C<chip>`. Every chip in the mesh graph is listed, local or not.

| Key | Value |
| --- | --- |
| `mesh_coord` | The chip's coordinates in its mesh |
| `host_rank` | Which host rank of its mesh owns the chip, local or remote. A mesh can be split across hosts (its `host_topology`), each host rank owning part of its chips. Null when the mesh graph doesn't say |
| `physical_chip_id` | This process's id for the chip. Null for a remote chip |
| `asic_id` | The chip's hardware id, as a hex string since it exceeds what JSON numbers represent exactly. Null for a remote chip |
| `is_local` | Whether this process opened the chip |
| `z_port_role` | Local chips only: the chip's role on the Z axis, or `none` |
| `local_sync` | Local chips only: `master` (the router that leads the chip's startup sync), `num_routers`, and `router_channels_mask` (bit N set for the router on Ethernet channel N). Null when the chip has no routers |
| `routers` | Local chips only: the chip's routers, by key |

[TOPOLOGY_PASS.md](TOPOLOGY_PASS.md) covers how chips and their routers are resolved.

### Router

`routers.<key>`.

| Key | Value |
| --- | --- |
| `identity` | `eth_chan`, `logical_core` and `virtual_core`. The router's mesh and chip are in its path, and its direction and routing plane in its key |
| `link` | `edge_capability`, `peer` (a router path, or null when the link reaches no active router), `cross_host`, `wrap` and `dispatch_link` |
| `shape` | `num_vcs`, `senders_per_vc`, `receivers_per_vc`, `num_active_eriscs`, `channel_trimming_overrides_applied` and `has_tensix_extension` (the router's tensix mux carries its VC0 senders other than the worker channel) |
| `credit_counters` | The four L1 credit counter arrays, `to_sender_ack`, `to_sender_completion`, `receiver_ack` and `receiver_completion`, each an L1 region plus `indexed_by`: `own_sender_channel` for the `to_sender` arrays, `peer_sender_channel` for the `receiver` arrays. Each element is one sender channel's, by its compact index |
| `channels` | `senders` and `receivers`, each keyed `vc<v>` then `ch<c>`. VCs with no channels of that kind are left out |
| `intra_chip_downstream_edges` | Keyed `vc<v>` then `edge<n>`. VCs with no edges are left out |
| `live_downstreams` | Null in 1D. In 2D, `vc0` and `vc1`: the router's live directions on each, such as `["west"]`, in enum order (`east`, `west`, `north`, `south`, `z`). A VC whose downstream array is empty (its `vc<v>_downstream_edm_size` field is 0) has `[]`. "Live downstreams" below says how they relate to the edges |
| `fields` | The id of the router's fields in `field_sets` |
| `eriscs` | Keyed `erisc<r>`, in RISC order: each with its `processor`, `physical_erisc` and `opt_level` (the level its kernel was compiled at, `o3` or `os`) |
| `leftover_l1` | The L1 from the end of what the router places to the end of the L1 the router may load into, as `byte`, or null when there is none |

### Channels

Both kinds of channel have:

| Key | Value |
| --- | --- |
| `status` | `active`, or why no ERISC runs it: `vc_not_serviced`, `mux` or `trimmed` |
| `serviced_by` | The ERISCs that run it, as `erisc<r>` |
| `ring_buffer` | Its packet slots, one element per slot (`element_size` is the slot size), typed as the run's packet header: each slot starts with one, and the payload after it is untyped. Null when it has none |
| `fields` | The id of its fields in `field_sets` |

A sender channel also has:

| Key | Value |
| --- | --- |
| `producer` | What feeds it: `"worker"`, `"tensix_mux"`, a sibling router's path, or null |
| `credits` | Keyed by ERISC: `completed`, and on VC0, on an ERISC whose `vc0_bubble_flow_control` is set, `acked`. Each is `{ array, index }` into the router's credit counters, where the router takes every VC's credits |

A receiver channel also has:

| Key | Value |
| --- | --- |
| `forwards_on` | Keyed by ERISC: the VC whose downstream edges that ERISC gives the step, as `vc<v>`, or null when that ERISC doesn't run the step or forwards to no sibling |

[ROUTER_PASS.md](ROUTER_PASS.md) covers how each is worked out.

### Downstream edges

| Key | Value |
| --- | --- |
| `downstream_channel` | The path of the sibling's sender channel the edge lands on |
| `through_tensix_mux` | Whether the edge lands at the sibling's tensix mux rather than the sibling itself |
| `fields` | The id of the edge's fields in `field_sets`: its free-slots stream and teardown semaphore |

`edge<n>` is the kernel's `EDGE_<n>`.

### Live downstreams

A 2D router forwards a packet only to the directions that both the packet's action and the router's
`live_downstreams` name, and ignores the action's other bits. The router computes its live directions at compile time
from its fields, and the manifest records what the same function (`Routing2DCodec::live_downstreams`) gives for them:

- Every entry of the VC's downstream array is live: entry `n - 1` is the direction `edge<n>` names. The array has
  `vc<v>_downstream_edm_size` entries, which the builder sets to one past the highest entry it connected, and at
  least 1.
- Except that an E/W-facing router never forwards onto N/S unless it can turn: its `intermesh_router` field is set (a
  landing decodes Y first) or its `intramesh_edge` field is (an exit may leave in any direction). `live_downstreams` takes
  both fields and decides this itself. Inside a mesh a
  packet travels Y first, so it never turns from X onto Y.

So the live directions and the edges differ both ways. On the 2D T3K mock, `M0/C1/E0` has edges to W and S, and only
`west` is live: the S edge is never used. `M0/C0/E0`, on a corner chip, has an edge to S only, so its VC0 array has
three entries (W, N, S); N and S are trimmed, and `west` is live with no edge behind it. A correct route never asks
for it.

## Excerpt

One router from a Blackhole 2D mock manifest, trimmed and indented:

```json
"E1": {
  "identity": { "eth_chan": 5, "logical_core": [0, 5], "virtual_core": [25, 25] },
  "link": { "edge_capability": "intramesh_cardinal", "peer": "M0/C1/W1", "cross_host": false, "wrap": false,
            "dispatch_link": false },
  "shape": { "num_vcs": 1, "senders_per_vc": [4, 0, 0], "receivers_per_vc": [1, 0, 0], "num_active_eriscs": 2,
             "channel_trimming_overrides_applied": false, "has_tensix_extension": false },
  "credit_counters": {
    "to_sender_ack": { "address": 88960, "num_elements": 12, "element_size": 4, "type": "unsigned int",
                       "indexed_by": "own_sender_channel" },
    ...
  },
  "channels": {
    "senders": { "vc0": { "ch0": {
      "status": "active", "serviced_by": ["erisc0"], "producer": "worker",
      "ring_buffer": { "address": 90912, "num_elements": 17, "element_size": 4448, "type": "HybridMeshPacketHeaderT<36>" },
      "credits": {
        "erisc0": { "completed": { "array": "credit_counters/to_sender_completion", "index": 0 } },
        "erisc1": { "completed": { "array": "credit_counters/to_sender_completion", "index": 0 } }
      },
      "fields": "F0"
    }, ... } },
    "receivers": { "vc0": { "ch0": {
      "status": "active", "serviced_by": ["erisc1"], "forwards_on": { "erisc0": null, "erisc1": "vc0" }, ...
    } } }
  },
  "intra_chip_downstream_edges": { "vc0": { "edge3": {
    "downstream_channel": "M0/C0/S1/channels/senders/vc0/ch1", "through_tensix_mux": false, "fields": "F2"
  }, ... } },
  "live_downstreams": { "vc0": ["west"], "vc1": [] },
  "fields": "F3",
  "eriscs": { "erisc0": { "processor": "riscv_0", "physical_erisc": 0, "opt_level": "o3" },
              "erisc1": { "processor": "riscv_1", "physical_erisc": 1, "opt_level": "o3" } },
  "leftover_l1": { "address": 451200, "num_elements": 4064, "element_size": 1, "type": "byte" },
  ...
}
```

And two of the blocks it names, from the same manifest's `field_sets`:

```json
"F2": {
  "erisc0": { "free_slots": { "category": "flow_control", "kind": "stream_reg", "stream_id": 7,
                              "address": 4290016420, "value_bits": 17, "type": "unsigned int" }, ... },
  "erisc1": { ... }
},
"F3": {
  "erisc0": { "local_handshake_master": { "category": "lifecycle", "kind": "flag", "value": false },
              "vc0_bubble_flow_control": { "category": "kernel_params", "kind": "flag", "value": false }, ... },
  "erisc1": { ... }
}
```
