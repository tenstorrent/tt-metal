# Fabric manifest

The fabric manifest is a JSON description of one fabric run, written at fabric init time. It names every fabric router, what the router kernel was fed, where and what state each router keeps (ie. L1, stream registers), and how the routers connect. Capture, decode and the viewer use it as the source of truth for a fabric's configuration in a run.

[`fabric_manifest_schema.json`](fabric_manifest_schema.json) is the JSON Schema of the file. Readers depend on
`manifest_version` and the shapes there.

## Enabling manifest generation

Set `TT_METAL_FABRIC_GENERATE_MANIFEST=1`. Each MPI rank writes one file:

`<logs_dir>/generated/fabric/fabric_manifest_rank_<rank + 1>_of_<world_size>.json`

`logs_dir` is `TT_METAL_LOGS_PATH`, or the working directory when that is not set.

### Notes:

- Fabric init deletes a stale manifest at that path, whether or not the option is set.
- The file is written after the routers are compiled and launched, and before the host waits for router sync, so a
  run that hangs during bring-up still leaves a manifest. A file on disk does not mean the routers reached sync.
- Mock clusters (`TT_METAL_MOCK_CLUSTER_DESC_PATH`) produce manifests without hardware.
- The file is written to a temporary name and renamed into place, so a reader never sees a partial file.
- When the manifest code cannot describe what the builder built, fabric init fails with `Failed to write fabric
  manifest ...` and a message starting with `Fabric manifest:`.

## How it is built

The router builders are destroyed once the fabric program is compiled, so the manifest is built in three steps:

1. **Collect, per router, while the builders exist** (`fabric_manifest_collector.cpp`). It reads the builder objects
   and the inputs (ie. ct args, defines) each ERISC was fed, and fills the model
   (`fabric_manifest_model.hpp`). Plain facts are read through the field tables (`fabric_manifest_fields.hpp`), one
   entry each; facts that need logic are computed in the collector.
2. **Join, per chip** (`fabric_manifest_chip_pass.cpp`). It adds what only ControlPlane, the cluster and the chip's
   other routers know: router keys, peers, cores, cross-host and wrap links, and sibling producers.
3. **Write** (`fabric_manifest.cpp`). The writer is the one place that decides the JSON's key names and shape.

The types at L1 addresses come from the struct layouts in `fabric_struct_layouts.hpp` and from the HAL. The run's
packet header layout comes from `fabric_packet_header_layouts.cpp`.

## Shape

| Key | Contents |
| --- | --- |
| `manifest_version`, `kind` | `1` and `"fabric_manifest"` |
| `run` | Architecture, fabric, reliability, tensix and UDM configs, host rank, MPI rank and world size, `written_at`, and the commit the build came from (`git_sha`, `git_dirty`) |
| `fabric_context` | Topology, the packet header type (`packet_header`, as `{size, schema}`), payload and channel buffer sizes, the route buffer size, `multi_txq` |
| `vocabulary` | Every field `category` and `kind`, and every chip `z_port_role` |
| `archs.<arch>.areas` | L1 areas the architecture fixes: heartbeat, telemetry, routing table, go and launch messages, Ethernet firmware mailbox, base firmware link status |
| `archs.<arch>.stream_registers` | Where the stream registers are and how wide their values are: `base`, `stride`, `count`, and each register's `offset` and `bits`, from the HAL |
| `archs.<arch>.core_registers` | The soft reset and wall clock registers, in the order they are read, with each physical ERISC's reset bit |
| `archs.<arch>.types` | Every struct and tagged union a schema names, including the run's packet header and the routing table as the run's routing mode lays it out. A struct has its size and members in order, and a union its size, its tag's enum and its members |
| `enums` | Every enum a schema names, with its values |
| `routing_encodings` | The values routes are encoded with: 1D hop values, 2D action bits and the codes that widen to them, the multicast tree edge layout, and the scatter chunk encoding's width |
| `meshes` | Each mesh, its chips, and each local chip's routers |

### Paths

A router's path is `M<mesh>/C<chip>/<key>`, for example `M0/C7/E0`. Its key is the direction it faces (`E`, `W`,
`N`, `S` or `Z`) and its routing plane. Other parts of the manifest refer to a router, or to something in it, by
path:

- `link.peer` and `local_sync.master` are router paths.
- A sender channel's `producer` is a sibling router's path, `"worker"` or `"tensix_mux"`.
- A downstream edge's `downstream_channel` is a channel's path, for example `M0/C7/W0/channels/senders/vc0/ch1`.
- A counter-backed credit names its array by its path in the router, for example `credit_counters/to_sender_ack`.

### Meshes, chips and routers

- **Mesh** (`meshes.M<n>`): `shape`, `torus` (2D meshes only), `express_routing`, `credit_transport` (meshes on this
  host only) and `chips`.
- **Chip** (`chips.C<n>`): `mesh_coord`, `physical_chip_id`, `asic_id` (a hex string) and `is_local`. Every chip in
  the mesh graph is listed, so the viewer can draw the host boundary. A chip this rank cannot map to a device has
  only those keys, with nulls. A local chip also has `z_port_role`, `local_sync` and `routers`.
- **Router** (`routers.<key>`): `identity`, `link`, `shape`, `credit_counters`, `channels`,
  `intra_chip_downstream_edges`, `fields`, `eriscs` and `leftover_l1`.
  - `channels.senders` and `channels.receivers` are keyed `vc<n>`, then `ch<m>`. VCs without channels are left out.
  - `intra_chip_downstream_edges` is keyed `vc<n>`, then `edge<n>`.
  - `eriscs` is keyed `erisc<n>`, by the ERISC's index; each has its `processor`, `physical_erisc` and `fields`.

### Regions, stream registers and schemas

- An **L1 region** is `{address, size, schema, cleared_by_host}`. An array also has `num_elements` and
  `size_per_element`, and a packed table `num_elements` and `bits_per_entry`. `cleared_by_host` means the host zeroes
  it before launch.
- A **stream register** is `{stream_id, register, schema}`. `register` is `buf_space_available` (the
  increment-on-write count) or `remote_src` (a plain read-write register). Its address is
  `base + stream_id * stride + registers[register].offset`, from `archs.<arch>.stream_registers`, and `stream_id` is
  below its `count`. Its value is the word's low `registers[register].bits` bits: the bits above
  `buf_space_available`'s count can hold other state.
- A **schema** is an integer (`u8` to `u32`, `i8` to `i32`), `struct:<Name>` or `union:<Name>` (in
  `archs.<arch>.types`), `enum:<Name>` (in `enums`), `bytes` or `pad`. Every `struct:`, `union:` and `enum:` schema
  resolves to a type of that kind: the writer throws otherwise. A class template's name includes its arguments, with
  no space after a comma, for example `struct:HybridMeshPacketHeaderT<36>`.
- A **packed table** is a schema of `enum:<Name>` with `bits_per_entry`: `num_elements` entries of that many bits,
  entry 0 at the lowest bits, each holding a value of the enum. The routing table's first-hop direction tables are
  packed `compressed_routing_values`, and a scatter write's `chunk_encoding` is packed `NocScatterWriteChunkEncoding`,
  one per chunk; entries from `chunk_count` on are unused.
- A **tagged union** is a struct member whose bytes hold one of several types, picked by another member of the same
  struct, its `tag`. The packet header's `command_fields` and `udm_control` are tagged by `noc_send_type`. The
  union's entry in `types` has `"kind": "union"`, `tag_schema` (the `enum:` the tag holds), its members, each at
  offset 0 with the `tag_value` that picks it, and `default_member`, the member every other tag value picks, when it
  has one. To decode one, read the tag, then the member it picks; when it picks none, the bytes are raw.
- The **routing table** (`routing_l1_info_t`) is laid out for the run's routing mode, since one slot of it holds the
  1D routing-path table or the 2D route table. In 1D the slot is `routing_path_table_1d`, and the rest of the slot and
  `exit_node_table`, which only 2D fills, are padding. In 2D the slot is `route_table_2d`. `types` holds only the
  mode's table struct, whose regions are `bytes`, read by rule: entry `i` of the 1D table, the route to the chip `i`
  hops away, starts at word `i * (1 + routing_1d_extension_words)`, and the 2D regions are sized by the mesh shape
  (`mesh_y_size`, `mesh_x_size`), as `route_table_2d_t`'s comment in `fabric_common.h` lays out.

### Routing encodings

`routing_encodings` holds the values the fabric encodes routes with, read from the constants the fabric encodes them
with, so a decoder does not copy them. They are the same in every run; the run's route buffer sizes are in
`fabric_context`. Every packed field below starts at the lowest bits.

- `1d`: a 1D route word holds `hops_per_word` hop fields of `bits_per_hop` bits each, hop 0 first. `hop` names each
  value a field holds.
- `2d`: `action_bits` is the bit each direction, and `local_deliver`, sets in a 2D action byte. The routing table's
  action vectors hold codes of `table_entry_bits` bits, `table_entries_per_byte` to a byte. `y_code_action[code]` and
  `x_code_action[code]` are the action byte a Y or X code widens to: 0 for stop, and for X's reserved code 3.
  `mcast_tree_edge` is one multicast tree edge: `bytes` long, with `child`, `parent` and `output` fields, each
  `{shift, bits}`. `output` is a code of the edge's axis.
- `scatter`: a scatter write's `chunk_encoding` holds one `NocScatterWriteChunkEncoding` per chunk, of
  `chunk_encoding_bits` bits each, for at most `max_chunks` chunks. The header's layout says the same, as a packed
  table.

### Fields

A `fields` object holds what the router kernel is fed, keyed by the field's key. Each field has a `category`
(`lifecycle`, `kernel_params`, `flow_control`, `control_info` or `diagnostics`) and a `kind`:

| Kind | Keys besides `category` and `kind` |
| --- | --- |
| `l1` | An L1 region's keys |
| `stream` | A stream register's keys |
| `number` | `value`, an integer |
| `flag` | `value`, a boolean |
| `enum` | `value`, an integer, and `schema`, the `enum:<Name>` that names it |

A field is null when it is a buffer or stream register the builder did not allocate (the kernel is fed address 0, or
the unused stream id, 32). A field the builder does not emit in this configuration (for example a 2D-only field on
1D) is left out.

[FIELDS.md](FIELDS.md) lists every field: its kind, schema and category, the argument that feeds it, and what it
means.

## Computed facts

These need logic beyond reading one argument, so they are code in the collector or the chip pass rather than table
entries.

- **Router shape** (`shape`): the VC count and the receivers per VC come from the builder's VC shape, and the senders
  per VC and the ERISC count from what the kernel is fed. The collector throws when the receivers the shape lists
  differ from the ones the kernel runs.
- **Channel status** (`status`): `active` when an ERISC runs the channel's step. Otherwise the reason, checked in the
  order the builder applies them: `vc_not_serviced` (the kernel does not run the VC on this router), `mux` (a VC0
  sender other than the worker channel, in mux mode; the tensix mux carries its traffic) or `trimmed` (the applied
  channel trimming profile turned it off). The collector throws when it knows no reason.
- **Serviced by** (`serviced_by`): the ERISCs whose `IS_SENDER_CHANNEL_<c>_SERVICED` or
  `IS_RECEIVER_CHANNEL_<c>_SERVICED` is set, as `erisc<n>`. Empty when the kernel does not run the VC.
- **Sender producer** (`producer`): `"worker"` for the worker channel, `"tensix_mux"` for it in mux mode, the
  sibling's path when a sibling's downstream edge lands on the channel, and null when nothing feeds it.
- **Sender credits** (`credits`): `completed` always, and `acked` only on VC0 with bubble flow control. Each is a
  stream register, or `{array, index}` into the router's credit counter arrays, as the mesh's `credit_transport`
  backing for the VC says. A stream register is null when the builder did not allocate one.
- **Credit transport** (`meshes.M<n>.credit_transport`): each VC's backing (`stream_register` or `l1_counter`) and the
  reasons the builder put it on counters.
- **Credit counters** (`credit_counters`): the router's four L1 counter arrays, always reserved. `index_space` says
  whose sender channels index them: the router's own (`own_sender_compact`) or its peer's (`peer_sender_compact`).
- **Receiver forwards on** (`forwards_on`): the VC whose downstream edges the receiver's step is given, `vc<n>`. VC1
  for receiver 0 of a router that crosses VC0 traffic over to VC1. Null when no ERISC runs the step, or when the
  step forwards to no sibling (VC2, and VC0 in speedy mode).
- **Downstream edges** (`intra_chip_downstream_edges`): the router's persistent connections to its siblings' sender
  channels on the same chip and routing plane. `edge<n>` is the kernel's `EDGE_<n>`. `downstream_channel` is the
  channel the edge lands on, and `through_tensix_mux` is set when the edge ends at the sibling's tensix mux instead.
- **Link** (`link`): the edge capability, the peer router (null when the channel connects to no active router),
  whether the link crosses hosts, whether it wraps a torus axis, and whether it is a dispatch link.
- **Local sync** (`local_sync`): the router that leads the chip's startup sync, the number of routers and a mask
  with bit N set for the router on Ethernet channel N. Null when the chip has no routers.
- **Leftover L1** (`leftover_l1`): the L1 from the end of the channel buffers to the end of loadable L1, or null when
  there is none.
- **Architecture areas** (`archs.<arch>.areas`): the heartbeat word with its magic, mask and period, and the HAL's
  areas, typed from the HAL. On Blackhole the heartbeat word is base firmware's own heartbeat, which the router
  writes over while it runs.
- **Base firmware link status** (`base_fw_port_status`, `base_fw_train_status`, `base_fw_retrain_count`,
  `base_fw_rx_link_up` and `base_fw_postcode` in `archs.<arch>.areas`): the words base firmware keeps about its
  Ethernet link, each a `u32` at the address the HAL gives. Each is null where the HAL gives none, as on Wormhole.
- **Core registers** (`archs.<arch>.core_registers`): `soft_reset`, `wall_clock_low` and `wall_clock_high`, each a
  `u32` at its HAL address. The array keeps the read order: reading the low word latches the high word.
  `soft_reset.reset_masks` is UMD's reset bit for each of the HAL's physical ERISCs on the Ethernet core, one on
  Wormhole and two on Blackhole, whether or not the ERISC mode runs a processor on it.
- **Physical ERISC** (`physical_erisc`): the ERISC on the Ethernet core that the processor runs on, from the HAL,
  which indexes `reset_masks`. It is the processor's index, except in Blackhole single-ERISC mode, where base
  firmware keeps ERISC0 and the one processor runs on ERISC1. The HAL compiles the same value into Blackhole kernels
  as `PHYSICAL_AERISC_ID`.
- **Build identity** (`run.git_sha`, `run.git_dirty`): the commit tt-metal was built from, and whether tracked files
  had uncommitted changes, from the version header CMake refreshes on every build (`cmake/version.cmake`). With
  uncommitted changes, the SHA doesn't say exactly what code ran. `git_sha` is empty when the source tree is neither a
  git checkout nor a `git archive` export.

## What the manifest doesn't cover

**UDM.** A UDM run (`run.udm_mode`) writes a manifest, but its UDM-specific parts aren't described:

- the tensix mux and relay cores, and their L1;
- a receiver's path into its relay: the connection, the relay's free slots stream (id 30), the relay's buffer count and
  the router's L1 word for it;
- which routers have a relay.

The VC0 worker channel's `producer` may also read `"worker"` where a tensix mux feeds it. No test writes a UDM
manifest.

## Arguments the manifest leaves out

Every named argument and define the router kernel is fed is read by a field table or by the collector, or is listed
in [FIELDS.md](FIELDS.md#arguments-the-manifest-leaves-out) with a reason. Otherwise the collector throws.

## Updating the manifest

A fabric change that the manifest does not account for fails with a message starting with `Fabric manifest:`, which
names what to update. By kind of change:

| Change | What fails | What to edit |
| --- | --- | --- |
| New named compile-time argument | The collector: "... is fed to the router, but neither a field table nor the collector reads it" | A field table entry in `fabric_manifest_fields.hpp`, or an entry with a reason in `k_unrecorded_args` |
| New define | The collector: "the define ... is fed to the router, but the collector does not read it" | Read it in the collector and list it in `k_collector_defines`, or add it to `k_unrecorded_defines` with a reason |
| Removed argument | The collector: "missing fabric router named compile-time argument ..." | Delete its table entry. An unrecorded entry for it is not detected; delete it by hand |
| Collector code reads a new argument | `get_named_arg`: "the collector reads ..., which no field table reads" | Add it to `k_collector_args` |
| New struct in L1 | The build, when a field's content names a struct with no `StructLayout`; the writer, when its schema does not resolve | A `StructLayout<T>` and an entry in `DescribedStructs` (`fabric_struct_layouts.hpp`) |
| New packet header size | Nothing: every header `PACKET_HEADER_CONFIGS` lists has its layout checked at compile time | Nothing |
| New packet header type, or a member added to one | The build, for a type with no `StructLayout` or a layout that misses bytes | Its `StructLayout` in `fabric_packet_header_layouts.cpp`, and a `PacketHeaderStructs` entry if it holds a struct `DescribedStructs` leaves out |
| New tagged union | The build, when a layout lists it with no `UnionLayout`; the writer, when its schema does not resolve | A `UnionLayout<U>` and an entry in `DescribedUnions` (`fabric_struct_layouts.hpp`), and `LAYOUT_TAGGED_UNION` in the enclosing layout |
| A member added to `routing_l1_info_t` | The build, for a member that changes an offset or the size, in either routing mode | `routing_l1_info_head` or `routing_l1_info_tail`, or the mode's list (`routing_l1_info_members_1d` or `_2d`), in `fabric_struct_layouts.hpp` |
| A member added to a packet header command struct | The build, for a member that changes an offset or the size | Its `StructLayout` in `fabric_struct_layouts.hpp`. A member added in the tail padding of a struct that lists `LAYOUT_TAIL_PAD` is not detected: it reads as padding |
| New enum | The writer, when a schema names it | An entry in `DescribedEnums` (`fabric_struct_layouts.hpp`) |
| New enum value | Nothing for enums enchantum reads; the build for `EDMStatus`, and for `NocSendType` when no `NocCommandFields` member is picked by it | `FABRIC_MANIFEST_EDM_STATUSES` for `EDMStatus`; for `NocSendType`, the member it picks in `UnionLayout<NocCommandFields>`, and in `UnionLayout<UDMControlFields>` unless the default `write` is right |
| New or renamed field category, field kind or Z port role | The schema check, on the name in `vocabulary` | Its list in `fabric_manifest_schema.json` (`category`, `kind` or `z_port_role`) |
| A routing encoding value changes, or a direction is added | Nothing: `routing_encodings` reads the constants, and lists every direction | Nothing |
| A wider 1D hop field | The build, until `RoutingFieldsConstants::LowLatency::HopAction` names every value the field holds; then the schema check | `HopAction` (`fabric_common.h`), and the schema's `hop` names |
| New computed fact | Nothing | The model, the collector or chip pass, the writer, the tests, a paragraph above, and the schema |
| A change to the JSON's shape | The schema check | `fabric_manifest_schema.json` |
| A field table or unrecorded list change | `ManifestDocs.UpToDate` | Regenerate `FIELDS.md` (below) |

### Regenerating FIELDS.md

`FIELDS.md` is generated from the field tables and the unrecorded lists in `fabric_manifest_fields.hpp`. To update
it, run the docs test with `TT_METAL_FABRIC_MANIFEST_WRITE_DOCS` set to the file to write:

```bash
TT_METAL_FABRIC_MANIFEST_WRITE_DOCS=tt_metal/fabric/debug/visualizer/manifest/FIELDS.md \
    ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ManifestDocs.*
```

### Running the tests

`tests/scripts/run_fabric_manifest_tests.sh` runs the tests that need no fabric hardware: the device-free unit tests,
the manifest fixtures using mock clusters, and the schema check on every manifest they write. It then runs the route
decoder's unit tests, and its check on the same manifests ([decode/README.md](../decode/README.md#running-the-tests)),
then capture's ([capture/README.md](../capture/README.md#running-the-tests)).

To check other manifests against the schema, pass their files or directories to the check, which needs `jsonschema`:

```bash
python3 tests/tt_metal/tt_fabric/fabric_manifest/check_manifest_schema.py generated/fabric
```

## Future work

### UDM

What it would take to cover the parts "What the manifest doesn't cover" lists:

- **A UDM mock fixture** in `run_fabric_manifest_tests.sh`, so that every run writes a UDM manifest and checks it
  against the schema, and a writer that can't describe a UDM run fails CI.
- **The relay path.** The receiver's connection to its tensix relay, written as one more of the receiver's outputs,
  with the relay's free slots stream, its buffer count and the router's L1 word for it. The three UDM arguments
  [FIELDS.md](FIELDS.md) lists as not yet described would become fields. This changes the builder code that
  records connections, so it needs a fabric owner's review.
- **Which routers have a relay.** A per-router flag, since a UDM run's dispatch-link routers don't have one.
- **The worker channel's producer**, written as `"tensix_mux"` where a tensix mux feeds it in a UDM run.
- **The tensix mux and relay cores**, with their L1.

### In-depth docs

This README covers the manifest's shape and how it is built at a high level. Manifest generation needs in-depth docs,
one per part, like decode's `docs/`:

- the collector and its field tables: how each argument is read, and the unrecorded lists;
- the chip pass: router keys, peers, cores, links and sibling producers;
- the struct layouts: how a `StructLayout` describes a struct, and how the layout checks catch a struct that changed;
- the writer: how the model becomes JSON.

Each would name the fabric code it reads, so that a fabric change can update the doc with it.
