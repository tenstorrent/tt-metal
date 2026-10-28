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
| Ring buffer | A channel's packet slots in L1, each one channel buffer in size |
| Credits | How a sender channel learns its packets were received (`acked`) or done with (`completed`) |
| Credit counter | An L1 word counting one sender channel's credits, when credits travel in L1 rather than stream registers |
| Stream register | One of the Ethernet core's hardware stream registers, which fabric uses as counters and scratch |
| Region | An L1 range the manifest describes: an address, a size, and a schema |
| Schema | How to read a value: an integer width, `bytes`, `pad`, or a named type in `types` |
| Field | One value a router's kernel was fed, keyed by name ([FIELD_TABLES.md](FIELD_TABLES.md)) |
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
  "run":        { host_rank, mpi_rank, world_size, written_at, build: { git_sha, git_dirty } },
  "fabric":     { config: { ... }, topology, is_2d_routing, packet_header: { size, schema }, ... },
  "arch":       { name, l1_areas: { ... }, stream_registers: { ... }, core_registers: [ ... ] },
  "types":      { structs: { <Name>: { size, members: [ ... ] } }, enums: { <Name>: { values: { ... } } },
                  tagged_unions: { <Name>: { size, tag: { schema }, cases: [ ... ] } } },
  "vocabulary": { <list name>: [ <name>, ... ] },
  "meshes": {
    "M<mesh>": {
      shape, torus, express_routing, credit_transport,
      "chips": {
        "C<chip>": {
          mesh_coord, physical_chip_id, asic_id, is_local, z_port_role, local_sync,
          "routers": {
            "<key>": {
              identity, link, shape, credit_counters,
              channels: { senders: { vc<v>: { ch<c>: { ... } } }, receivers: { ... } },
              intra_chip_downstream_edges: { vc<v>: { edge<n>: { ... } } },
              fields: { ... }, eriscs: { erisc<n>: { ... } }, leftover_l1
            }
          }
        }
      }
    }
  }
}
```

`manifest_version` is `1`, and changes when readers can't take the shape as it is. `file_kind` is always
`"fabric_manifest"`, naming which of the visualizer's files this is.

## Shapes that repeat

**L1 region**: `{ address, size, schema, cleared_by_host }`. An array also has `num_elements` and `size_per_element`.
`cleared_by_host` means fabric init zeroes the region before launching the routers.

```json
{ "address": 98336, "size": 48, "num_elements": 12, "size_per_element": 4, "schema": "u32", "cleared_by_host": false }
```

**Stream register**: `{ stream_id, register, schema }`. `register` is `buf_space_available` (an increment-on-write
count) or `remote_src` (a plain read-write register). Its address is
`base + stream_id * stride + registers[register].offset`, from `arch.stream_registers`, and its value is the word's low `registers[register].bits` bits. The bits above `buf_space_available`'s count can hold other state.

```json
{ "stream_id": 9, "register": "buf_space_available", "schema": "u32" }
```

**Schema**: how to read a value.

| Schema | Read as |
| --- | --- |
| `u8` to `u64`, `i8` to `i64` | An unsigned or signed integer, as wide as one element |
| `enum:<Name>` | An enum, described in `types.enums` |
| `struct:<Name>` | A struct, described in `types.structs`. A class template instance's name has its arguments as C++ spells them, such as `HybridMeshPacketHeaderT<36>` |
| `union:<Name>` | A tagged union, described in `types.tagged_unions`: one of several types, picked by the value of the enclosing struct's member named by `tag_member` |
| `bytes` | Raw bytes, for code that reads them by its own rules, such as a table packed below byte granularity |
| `pad` | Bytes that mean nothing |

Every `struct:`, `enum:` and `union:` schema names an entry in `types` ([WRITER.md](WRITER.md)).

**Field**: `{ category, kind, ... }`, keyed by the field's name. `category` is `lifecycle`, `kernel_params`,
`flow_control`, `control_info` or `diagnostics`.

| `kind` | Keys besides `category` and `kind` |
| --- | --- |
| `l1` | An L1 region's keys |
| `stream` | A stream register's keys |
| `number` | `value`, an integer |
| `flag` | `value`, a boolean |
| `enum` | `value`, an integer, and `schema`, the `enum:<Name>` it's a value of |

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
| `host_rank` | This process's host rank |
| `mpi_rank`, `world_size` | This process's MPI rank, and the number of processes |
| `written_at` | When the manifest was written, in UTC, as ISO 8601 |
| `build.git_sha` | The commit tt-metal was built from. Empty outside a git checkout or `git archive` export |
| `build.git_dirty` | Whether tracked files had uncommitted changes, in which case the SHA doesn't say exactly what ran |

## fabric

| Key | Value |
| --- | --- |
| `config.fabric_config`, `config.reliability_mode`, `config.tensix_config`, `config.udm_mode`, `config.fabric_manager` | The settings fabric was configured with |
| `config.num_routing_planes` | The most routing planes fabric opens per link, or null for every available one |
| `config.router_config.max_packet_payload_size_bytes` | The payload size override, or null for the default |
| `topology` | The fabric topology, such as `mesh` |
| `is_2d_routing` | Whether 2D routing is on |
| `packet_header` | The packet header type the run's routers use: its `size`, and a `struct:` schema naming its layout, such as `struct:LowLatencyPacketHeaderT<0>` |
| `max_payload_size_bytes`, `channel_buffer_size_bytes` | Payload and channel buffer sizes |
| `routing_2d_route_buffer_size` | 2D routing only: the packet header's route buffer size |
| `routing_1d_extension_words` | 1D routing only: the packet header's extension words |
| `multi_txq` | Whether senders and receivers transmit on separate Ethernet TX queues |

[DESCRIBE.md](DESCRIBE.md) says where each comes from.

## arch

| Key | Value |
| --- | --- |
| `name` | The architecture, such as `wormhole_b0` |
| `l1_areas` | L1 regions the architecture fixes on every router core: `heartbeat` (with its `magic`, `magic_mask` and `period_iters`), `fabric_telemetry`, `routing_table`, `eth_fw_mailbox`, and base firmware's link status words `base_fw_port_status`, `base_fw_train_status`, `base_fw_retrain_count`, `base_fw_rx_link_up` and `base_fw_postcode`. An area the architecture doesn't have is null |
| `stream_registers` | `base`, `stride` and `count` of the Ethernet stream registers, and for each register (`buf_space_available`, `remote_src`) its `offset` within a stream and its value's `bits` |
| `core_registers` | An array, in read order: `soft_reset` (with `reset_masks`, each physical ERISC's reset bit), `wall_clock_low` and `wall_clock_high`, each with its `name`, `address`, `size` and `schema`. Reading the low wall clock word latches the high word |

## types

| Key | Value |
| --- | --- |
| `structs.<Name>` | `size`, and `members` in order, each with its `name`, `offset`, `size`, `schema`, for an array `num_elements` and `size_per_element`, and for a tagged union `tag_member`. Members cover every byte of the struct |
| `enums.<Name>` | `values`: each enumerator's lower-case name, and its integer value |
| `tagged_unions.<Name>` | `size`, `tag.schema` (the `enum:` its tag holds), and `cases`, each a member as in a struct, at offset 0, with the `tag_value` that picks it |

To read a tagged union, read the enclosing struct's `tag_member`, then the case whose `tag_value` names that value.

`types` holds every struct, enum and tagged union a schema names, and every one those name in turn, and nothing else. A reader decodes bytes from these entries without the C++ headers ([STRUCT_LAYOUTS.md](STRUCT_LAYOUTS.md)).

## vocabulary

Every name each label can hold, so a reader knows all of a label's values without the C++ enums. Each list is named for
what the label is:

| List | Holds the names of |
| --- | --- |
| `arch` | `arch.name` |
| `fabric_config`, `reliability_mode`, `tensix_config`, `udm_mode`, `fabric_manager_mode` | `fabric.config`'s settings |
| `fabric_topology` | `fabric.topology` |
| `credit_backing`, `l1_credit_counter_reason` | A mesh's `credit_transport` |
| `z_port_role` | A chip's `z_port_role` |
| `edge_capability` | A link's `edge_capability` |
| `credit_counter_indexed_by` | A credit counter array's `indexed_by` |
| `stream_register` | A stream register's `register` |
| `field_category`, `field_kind` | A field's `category` and `kind` |
| `channel_status` | A channel's `status` |
| `erisc_processor` | An ERISC's `processor` |

These are names the manifest writes as strings, so nothing appears both here and in `types.enums`.

## meshes

### Mesh

`meshes.M<mesh>`, by mesh id.

| Key | Value |
| --- | --- |
| `shape` | The mesh's extent in each dimension |
| `torus` | 2D meshes only: `{ y, x }`, whether each axis wraps |
| `express_routing` | Whether the mesh has express routing |
| `credit_transport` | Meshes on this host only: for each VC some router has senders on, `vc<v>: { backing, reasons }`. `backing` is `stream_register` or `l1_counter`, and `reasons` says why counters |
| `chips` | Every chip in the mesh, by chip id |

### Chip

`chips.C<chip>`. Every chip in the mesh graph is listed, local or not.

| Key | Value |
| --- | --- |
| `mesh_coord` | The chip's coordinates in its mesh |
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
| `shape` | `num_vcs`, `senders_per_vc`, `receivers_per_vc`, `num_active_eriscs`, `channel_trimming_overrides_applied` and `vc0_bubble_flow_control` |
| `credit_counters` | The four L1 credit counter arrays, `to_sender_ack`, `to_sender_completion`, `receiver_ack` and `receiver_completion`, each an L1 region plus `indexed_by`: `own_sender_channel` for the `to_sender` arrays, `peer_sender_channel` for the `receiver` arrays. Each element is one sender channel's, by its compact index |
| `channels` | `senders` and `receivers`, each keyed `vc<v>` then `ch<c>`. VCs with no channels of that kind are left out |
| `intra_chip_downstream_edges` | Keyed `vc<v>` then `edge<n>`. VCs with no edges are left out |
| `fields` | The router's fields |
| `eriscs` | Keyed `erisc<n>`, in RISC order: each with its `processor`, `physical_erisc` and `fields` |
| `leftover_l1` | The L1 from the end of what the router uses to the end of loadable L1, as `bytes`, or null when there is none |

### Channels

Both kinds of channel have:

| Key | Value |
| --- | --- |
| `status` | `active`, or why no ERISC runs it: `vc_not_serviced`, `mux` or `trimmed` |
| `serviced_by` | The ERISCs that run it, as `erisc<n>` |
| `ring_buffer` | Its packet slots, as `bytes` with one element per slot, or null when it has none |
| `fields` | Its fields |

A sender channel also has:

| Key | Value |
| --- | --- |
| `producer` | What feeds it: `"worker"`, `"tensix_mux"`, a sibling router's path, or null |
| `credits` | `completed`, and on VC0 with bubble flow control `acked`. Each is a stream register, null when unallocated, or `{ array, index }` into the router's credit counters |

A receiver channel also has:

| Key | Value |
| --- | --- |
| `forwards_on` | The VC whose downstream edges it forwards on, as `vc<v>`, or null when it forwards to no sibling |

[ROUTER_PASS.md](ROUTER_PASS.md) covers how each is worked out.

### Downstream edges

| Key | Value |
| --- | --- |
| `downstream_channel` | The path of the sibling's sender channel the edge lands on |
| `through_tensix_mux` | Whether the edge lands at the sibling's tensix mux rather than the sibling itself |
| `fields` | The edge's fields: its free-slots stream and teardown semaphore |

`edge<n>` is the kernel's `EDGE_<n>`.

## Excerpt

One router from a 2D mock manifest, trimmed:

```json
"E1": {
  "identity": { "eth_chan": 0, "logical_core": [0, 0], "virtual_core": [25, 16] },
  "link": { "edge_capability": "intramesh_cardinal", "peer": "M0/C1/W1", "cross_host": false, "wrap": false,
            "dispatch_link": true },
  "shape": { "num_vcs": 1, "senders_per_vc": [4, 0, 0], "receivers_per_vc": [1, 0, 0], "num_active_eriscs": 1,
             "channel_trimming_overrides_applied": false, "vc0_bubble_flow_control": false },
  "channels": {
    "senders": { "vc0": { "ch0": {
      "status": "active", "serviced_by": ["erisc0"], "producer": "worker",
      "ring_buffer": { "address": 100256, "size": 62272, "num_elements": 14, "size_per_element": 4448,
                       "schema": "bytes", "cleared_by_host": false },
      "credits": { "completed": { "stream_id": 9, "register": "buf_space_available", "schema": "u32" } },
      "fields": { ... }
    }, ... } },
    "receivers": { "vc0": { "ch0": { "status": "active", "serviced_by": ["erisc0"], "forwards_on": "vc0", ... } } }
  },
  "intra_chip_downstream_edges": { "vc0": { "edge3": {
    "downstream_channel": "M0/C0/S1/channels/senders/vc0/ch1", "through_tensix_mux": false,
    "fields": { "free_slots": { "category": "flow_control", "kind": "stream", "stream_id": 15,
                                "register": "buf_space_available", "schema": "u32" }, ... }
  }, ... } },
  ...
}
```
