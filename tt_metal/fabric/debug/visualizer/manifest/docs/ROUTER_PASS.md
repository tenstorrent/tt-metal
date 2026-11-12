# Router pass

The router pass (`fabric_manifest_router_pass.cpp`) describes one router while its builder still exists. Each chip's
`FabricBuilder` runs it on every router when it publishes the chip ([BUILDER_HOOKS.md](BUILDER_HOOKS.md)), and the
result, a `manifest::Router`, waits in `PublishedChips` until the manifest is written.

Its input is the router's `ManifestRouterInputs`, which the router builder fills from what only it can see:

| Input | What it is |
| --- | --- |
| `erisc_builder` | The router's `FabricEriscDatamoverBuilder`: its config, channel allocator, trimming profile, and the semaphore ids some fields read |
| `connections` | The connections the router made to its siblings, as `establish_connections_to_router` recorded them in the chip's `ConnectionRegistry` |
| `vc_shape` | The router's VCs and how many receiver channels each has |
| `location` | The router's Ethernet channel, the direction it faces, and whether it's on a dispatch link |
| `chip_facts` | The chip's routing facts, such as each direction's edge capability |
| `args` | A `RouterArgs` over what `create_kernel` built each RISC's kernel with: its named compile-time arguments, defines, processor, NoC and opt level |
| `packet_header` | The run's `PacketHeaderConfig` (`FabricContext::get_packet_header_config()`), which picks the header type each ring buffer slot starts with |
| `stream_assignment` | The router's mesh's `StreamAssignment` (`FabricBuilderContext::get_stream_assignment`), which each stream register field's stream id is checked against |
| `hal` | The HAL, for stream register addresses and physical ERISCs |

Plain values are read through the field tables ([FIELD_TABLES.md](FIELD_TABLES.md)). This doc covers the rest: the
facts that take logic (computed facts), and the checks that come with them.

The router pass knows only this router and its builder. What needs the rest of the chip or the fabric is left for the
topology pass ([TOPOLOGY_PASS.md](TOPOLOGY_PASS.md)): the router's routing plane and cores, its link's peer and whether
the link wraps or crosses hosts, and which sibling feeds
each sender channel.

## Steps

`collect_manifest_router` runs, in order:

1. **Check the kernel inputs.** Every argument and define is read or listed as left out
   ([FIELD_TABLES.md](FIELD_TABLES.md)).
2. **The router's shape**, from its arguments.
3. **The channel context.** What every channel needs: the channel allocator, whether 2D routing is on, the trimming
   profile the builder applied, and the channel buffer size.
4. **Channels**, VC by VC: each sender, then each receiver.
5. **Downstream edges**, VC by VC.
6. **Live downstreams**, in 2D.
7. **Fields**: the router's, per ERISC.
8. **The rest**: identity, link, credit counters and leftover L1.

## Where values come from

The router pass reads a value from the kernel's arguments wherever an argument carries it, and from the builder only
where none does:

- **From the arguments, once per router.** A value the builder gives every ERISC alike: the shape's counts, whether 2D
  routing is on, which sender channels are worker channels, the credit counter bases, the slot size, where each VC
  starts in the free-slots stream table, where VC1's teardown semaphores start, and what the live downstreams are
  computed from.
- **From the arguments, per ERISC.** The kernel runs on one or more ERISCs, and the builder gives each its own named
  compile-time arguments. What differs by ERISC is read and recorded per ERISC: every field, which ERISCs run each
  channel, and the two facts that depend on an ERISC's own first-level ack and forwarding settings (a sender's
  `acked` credits and a receiver's `forwards_on`).
- **From the builder**, where no argument carries the value: the receivers per VC, the trimming profile, which sender
  channels the tensix mux carries, the channel allocator's buffers, the connections it recorded (the edges), and the
  leftover L1.

| Reader | Reads | Used for |
| --- | --- | --- |
| `RouterArgs::read_router_arg` | An argument every ERISC is given, which they must agree on | The values read once per router |
| `RouterArgs::read_erisc_arg` | One ERISC's argument | Every field, first-level acks, and a receiver's forwarding and VC2 crossover |
| `RouterArgs::find_eriscs_servicing_sender`, `_receiver` | Each ERISC's `IS_*_CHANNEL_<c>_SERVICED` | Who runs each channel |
| `RouterArgs::has_define` | A define every ERISC shares | VC gates, VC0 crossover |

`RouterArgs` refuses a name the field tables don't read and the router pass's lists don't name, so the lists stay true
to the code. It also fails when an argument is missing, or when ERISCs disagree on one read once per router. Each
failure names the line that read it ([FIELD_TABLES.md](FIELD_TABLES.md)).

## Fields and regions

The router's, each channel's and each edge's `fields` come from the field tables, one set per ERISC: FIELD_TABLES.md's
"Reading a table" says how an entry becomes a field.

The router pass also registers regions of its own, where no single argument gives the location: ring buffers, the
credit counter arrays, leftover L1, and credits on stream registers. Each is registered with its location from the
builder (TYPE_REGISTRY.md's "Where regions are registered" lists them), and each L1 region records whether fabric init
clears it.

**Active.** Every L1 region says whether anything writes it on this run (`active`, [FORMAT.md](FORMAT.md)). A field's
region follows its `written_when` gate (FIELD_TABLES.md's "Reading a table"). Of the router pass's own regions, the
ring buffers are always active, the credit counter arrays are active when the kernel uses them (below), and leftover L1
never is.

**Stream registers.** Each stream register field names its role (FIELD_TABLES.md). The router pass resolves it for the
channel or edge, checks the stream id it's fed, and describes the register (`describe_stream_register`):

| Role | The id must be | Active when |
| --- | --- | --- |
| `sender_free_slots` | `StreamAssignment::id(role, vc, channel)`, the channel within its VC, or the unused id exactly when `has` is false | Always |
| `vc2_sender_free_slots` (a VC2 sender channel's `free_slots`) | VC2's pinned id (`StreamRegAssignments::IncrementOnWrite`) | Always |
| `receiver_pkts_sent` | `id(role, receiver channel, 0)`, keyed by the receiver channel's compact index | Always |
| `downstream_free_slots` | `id(role, vc, edge - 1)`, the edge's compact index | Always |
| `multi_risc_teardown_sync` | Its pinned id (`StreamRegAssignments::Scratch`) | `NUM_ACTIVE_ERISCS` > 1: `sync_erisc` writes it only between two ERISCs |
| `eth_retrain_link_sync` | Its pinned id | Any ERISC has `ENABLE_CONTEXT_SWITCH`: the master sets it, and the other ERISC acks it. A `written_when` gate reads only the same ERISC, so the router pass decides this one |

A mismatch fails the router pass. The register's address is the stream's space-available register for a count and
its remote-source register for a scratch role, from the HAL, and its `read_mask` is the role's
(`get_stream_register_read_mask`, from `fabric_common.h`'s masks).

## The router

**Identity** (`identity`): the router's Ethernet channel and the direction its builder built it facing, from its
location. The topology pass adds its routing plane, which makes its key, and its cores.

**Link** (`link`): the edge capability of the direction the router faces, from the chip's routing facts, and whether
the link is a dispatch link, from its location. The peer, `cross_host` and `wrap` are added by the topology pass.

**Shape** (`shape`):

| Key | From |
| --- | --- |
| `num_vcs`, `receivers_per_vc` | The builder's VC shape |
| `senders_per_vc` | `ACTUAL_VC<v>_SENDER_CHANNELS` |
| `num_active_eriscs` | `NUM_ACTIVE_ERISCS` |
| `channel_trimming_overrides_applied` | Whether the builder applied a channel trimming profile to the router |
| `has_tensix_extension` | `FABRIC_TENSIX_EXTENSION_MUX_MODE`: the router's tensix mux carries its VC0 senders other than the worker channel. False on a dispatch link, even in mux mode |

Whether VC0 runs bubble flow control is a field, `vc0_bubble_flow_control`, so it's recorded per ERISC.

**Credit counters** (`credit_counters`): the router's four L1 arrays of credit counters, at the bases in the ERISC
kernel's arguments (`TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR` and the other three). The kernel sends each pair to the
peer in one packet, so it relies on the arrays being back to back and the same size, and the size comes from that
spacing. Each array is `uint32_t`s, one per sender channel, indexed by the channel's compact index. The `to_sender`
arrays count this router's own sender channels, and the `receiver` arrays the sender channels of the router on the other
end of the link. The model's `k_credit_counter_arrays` says which, and where `L1CreditCounters` holds each array; the
writer records which as `indexed_by`. `credit_counter_region` gives the region a `CounterRef` points into. The arrays
are active when any of the router's ERISCs is fed `VC<n>_USES_COUNTER_CREDITS` (`uses_credit_counters`): the same
arguments the kernel's `any_vc_uses_counter_credits` reads, so the manifest and the kernel decide it from one place,
the builder's credit transport plan. A router on two TX queues requires counters on every VC, so the arrays it zeroes
by role in `router_init.hpp` are zeroed exactly then. With this router VC0 and VC1 always use counters, so the arrays
are always active.

**Leftover L1** (`leftover_l1`): the L1 from the end of what the router places (the builder's
`get_allocated_l1_end_address`) to the end of the L1 the router may load into (`max_l1_loading_size`), as `std::byte`,
or null when nothing is left. With the streaming profiler on, `LinkSyncL1` sits above `max_l1_loading_size`, so it is
not leftover. It's never active, since nothing is placed there; capture still reads its bytes, so a stray write there
shows.

**ERISCs** (`eriscs`): one per RISC the kernel runs on, in RISC order, each with its processor, its physical ERISC
from the HAL, and the opt level its kernel was compiled and linked at. Each ERISC's fields are under the router's
`fields`, keyed by ERISC. The physical ERISC is the processor's index, except in Blackhole single-ERISC mode, where base
firmware keeps ERISC0 and the one processor runs on ERISC1. It indexes the `soft_reset` reset masks in
`arch.core_registers`. The opt level is what `create_kernel` passed to `CreateKernel` (the builder's
`get_kernel_opt_level`): `o3`, or `os` where the router has both VC1 and VC2 and would not fit in code space at O3.

## Channels

Channels are listed by VC and by channel within the VC. Each also has two indices the kernel uses:

- **Compact index**: the channel's index over this router's own counts, VC by VC (VC0's senders, then VC1's, and so
on). Most per-channel arguments are indexed by it, as are the credit counter arrays.
- **Fabric position** (senders only): the channel's index in the fabric's free-slots stream table, which is laid out
for the fabric's maximum counts. VC0 starts at 0, and each other VC where `VC<v>_FABRIC_POSITION_START` says.

**Status** (`status`): `active` when an ERISC runs the channel's step. Otherwise the reason, checked in the order the
builder applies them:

1. `vc_not_serviced`: the kernel doesn't run the VC on this router. VC0 always runs, VC1 only under
   `FABRIC_2D_VC1_SERVICED`, and VC2 only under `FABRIC_2D_VC2_SERVICED`.
2. `mux`: the tensix mux carries the channel's traffic, as the builder's `is_sender_channel_carried_by_mux` says: in
   mux mode the router runs only its worker channel.
3. `trimmed`: the applied channel trimming profile turned it off (a sender it doesn't use, or a receiver whose data it
   doesn't forward).

A channel no ERISC runs, for none of these reasons, fails the router pass: the manifest would otherwise call a broken
channel idle.

**Serviced by** (`serviced_by`): the ERISCs whose `IS_SENDER_CHANNEL_<c>_SERVICED` or `IS_RECEIVER_CHANNEL_<c>_SERVICED`
is set, as `erisc<n>`, where `<c>` is the compact index. Empty when the kernel doesn't run the VC.

**Ring buffer** (`ring_buffer`): the channel's packet slots, from the channel allocator, each the ERISC builder's
`channel_buffer_size` bytes (fed as `CHANNEL_BUFFER_SIZE`). Registered as the run's packet header type, from
`packet_header` through `visit_packet_header_type`, with one element per slot: `element_size` is the slot size, each
slot's header is typed, and the payload after it is raw. Null when the channel has no slots.

### Senders

**Producer** (`producer`): what feeds the channel. A worker channel (`SENDER_CH_<c>_IS_WORKER_CHANNEL`) is fed by
`"worker"`, or by `"tensix_mux"` for VC0 on a router with a tensix extension. A channel fed by a sibling router gets that sibling's path from the
topology pass. Otherwise null.

**Credits** (`credits`): where the channel's credits come back to it, keyed by ERISC: what each ERISC is told to read.
`completed` always, and `acked` only where the ERISC is fed `ENABLE_FIRST_LEVEL_ACK_VC<v>` set. The builder feeds it for
every VC: VC0's follows deadlock avoidance, and VC1's and VC2's are always 0 (the host rejects first-level acks on them).
Each is the channel's element of the `to_sender` counter array, written as `{array, index}`: the router takes every VC's
credits through L1 counters, and `router_sender.hpp` `static_assert`s it. The stream registers the builder also names
(`TO_SENDER_<c>_PKTS_COMPLETED_ID`, `TO_SENDER_<c>_PKTS_ACKED_ID`) and `VC<v>_USES_COUNTER_CREDITS` are left out for
that reason. The mesh's `credit_transport` records the builder's plan ([DESCRIBE.md](DESCRIBE.md)).

### Receivers

**Forwards on** (`forwards_on`): keyed by ERISC, the VC whose downstream edges that ERISC gives the receiver's step, as
`vc<n>`. Null on an ERISC that doesn't run the step, or where the VC's forwarding is off (`DISABLE_RX_CH<v>_FORWARDING`,
which the builder emits per VC and sets for VC0 on a speedy router and for VC2 always): the channel only delivers.
Otherwise `receiver_channel_forwards_on_vc` (`fabric_common.h`), which the kernel calls too: VC1 where the VC crosses
over to it, else its own. VC0 crosses over under `FABRIC_2D_VC0_CROSSOVER_TO_VC1`, VC2 under `VC2_CROSSOVER_TO_VC1`,
and VC1 never.

## Downstream edges

`intra_chip_downstream_edges` lists the router's persistent connections to its sibling routers' sender channels, on the
same chip and routing plane, by VC. They come from the connections the builder recorded as it made them
(`establish_connections_to_router`), so each edge says what the builder chose.

- **Edge number** (`edge<n>`): the kernel's `EDGE_<n>`. In 2D it's the sibling's direction's compact index among this
router's other directions, plus one. In 1D there's one edge, `edge1`.
- **Sibling**: the Ethernet channel and direction of the router the edge lands on, from the record.
- **Landing channel**: the sibling's sender channel the edge lands on, within the edge's VC, from the record. The writer
turns this and the sibling's direction into `downstream_channel`.
- **Through the tensix mux** (`through_tensix_mux`): whether the connection ends at the sibling's tensix mux rather than
its ERISC, from the record.
- **Fields**: the edge's free-slots stream and teardown semaphore, per ERISC. The kernel takes an edge's teardown
semaphore by its rank among the router's edges, VC0's before VC1's, so VC1's start after the adapter's VC0 edge count.

The kernel never forwards on VC2, so VC2 has no edges. Two edges in one slot fail the router pass, since they'd share
a key.

## Live downstreams

`live_downstreams` is, for VC0 and VC1, the directions the router forwards packets to: what the kernel's
`live_downstreams<VC>()` (`router_downstream.hpp`) computes at compile time. Both call `Routing2DCodec::live_downstreams`
(`fabric_common.h`) with the same arguments, read here once per router: `MY_DIRECTION`, the VC's
`VC<v>_DOWNSTREAM_EDM_SIZE`, `IS_INTERMESH_ROUTER` and `IS_INTRAMESH_ROUTER_ON_EDGE`. `live_downstreams` decides from the
last two whether the router can turn.
The rule is shared rather than copied, so it has no row in "Rules the manifest copies".

The model holds each VC's directions as a mask of action bits, as `live_downstreams` returns it; the writer names them. A
VC whose downstream array is empty (the builder gives VC1 size 0 on a router with no VC1 receiver) has no directions.
In 1D `live_downstreams` is null: a 1D packet's route says only whether to forward, to the router's one downstream.

## Rules the manifest copies

A few facts are decided by a rule in the kernel or the builder that no argument or builder getter carries. The router
pass copies each rule. If fabric changes one, update the row's code here too. Only the channel status has a check: a
channel no ERISC runs, for none of the reasons below, fails the router pass.

| Rule | Where fabric decides it |
| --- | --- |
| A channel's status reason, in the builder's order: the VC isn't run, then mux, then trimmed | The kernel's VC gate, `is_vc_configured` (`fabric_erisc_router_ct_args.hpp`, from `FABRIC_2D_VC<v>_SERVICED`); `FabricEriscDatamoverBuilder`'s constructor, which applies mux servicing before `apply_channel_trimming_overrides` |
| Which VCs the kernel runs: VC0 always, VC1 under `FABRIC_2D_VC1_SERVICED`, VC2 under `FABRIC_2D_VC2_SERVICED` | `is_vc_configured` (`fabric_erisc_router_ct_args.hpp`), which `runs_vc` (`router_common.hpp`) combines with each receiver channel's serviced flag |
| Which input says each VC's receiver crosses over to VC1: `FABRIC_2D_VC0_CROSSOVER_TO_VC1` for VC0, `VC2_CROSSOVER_TO_VC1` for VC2, nothing for VC1 | `is_vc_crossover_to_vc1` (`fabric_erisc_router_ct_args.hpp`) |
| VC0's senders start at position 0 in the free-slots stream table; VC1's and VC2's at `VC<v>_FABRIC_POSITION_START` | `fabric_position_for_compact_sender` (`fabric_erisc_router_ct_args.hpp`) |
| A worker channel's producer: the local worker, or on VC0 the tensix mux when the router has a tensix extension | The mux kernel's connection to the router's worker channel, which the tensix builder sets up (`FabricTensixDatamoverMuxBuilder`) |
| An edge's slot: the sibling's direction's compact index among this router's other directions | `get_receiver_channel_compact_index`, which the adapter's `add_downstream_connection` uses too |
| An edge's teardown semaphore: its rank among the router's edges, VC1's after VC0's (offset by `NUM_DOWNSTREAM_SENDERS_VC0`) | `RuntimeArgs::vc1()` (`router_init.hpp`), whose teardown semaphores start at `NUM_DOWNSTREAM_SENDERS_VC0`, each edge indexing them by its dense index (`router_downstream.hpp`) |
| The four credit counter arrays are back to back and the same size, so each is as big as the gap between the first two bases | `num_words_consumed_per_counter` in `FabricEriscDatamoverConfig`'s constructor (`erisc_datamover_builder.cpp`), which nothing exposes |

## Adding a computed fact

1. Add the fact to the model (`fabric_manifest_model.hpp`) where it sits.
2. Compute it here, from `ManifestRouterInputs`: read the argument that carries it, or the builder's own state, before
   copying a rule. If it reads an argument or define no table reads, add the name to `k_router_pass_args` or
   `k_router_pass_defines`, and remove it from the unrecorded lists if it's there. If it's a region, register it with
   its type ([TYPE_REGISTRY.md](TYPE_REGISTRY.md)).
3. Write it in the writer (`fabric_manifest.cpp`), and add it to `fabric_manifest_schema.json` and
   [FORMAT.md](FORMAT.md).
4. Describe it in this doc, with the check that keeps it honest, if any. If it copies a rule, add a row to "Rules the
   manifest copies" saying where fabric decides it.
