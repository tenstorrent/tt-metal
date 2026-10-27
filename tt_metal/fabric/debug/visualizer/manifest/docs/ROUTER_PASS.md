# Router pass

The router pass (`fabric_manifest_router_pass.cpp`) describes one router while its builder still exists. Each chip's
`FabricBuilder` runs it on every router when it publishes the chip ([BUILDER_HOOKS.md](BUILDER_HOOKS.md)), and the result, a `manifest::Router`, waits in `PublishedChips` until the manifest is written.

Its input is the router's `ManifestRouterInputs`: the router's ERISC builder, chip-level routinf facts, etc. Plain values are read through the field tables ([FIELD_TABLES.md](FIELD_TABLES.md)). This doc covers the rest: the facts that take logic (computed facts), and the checks that come with them.

The router pass knows only this router and its builder. What needs the rest of the chip or the fabric is left for the
topology pass ([TOPOLOGY_PASS.md](TOPOLOGY_PASS.md)): the router's key and cores, its link's peer and whether the link
wraps or crosses hosts, which sibling router each downstream edge lands on, and which sibling feeds each sender channel.

## Steps

`collect_manifest_router` runs, in order:

1. **Check the kernel inputs.** Every argument and define is read or listed as left out

   ([FIELD_TABLES.md](FIELD_TABLES.md)).
2. **The router's shape.** It must agree with the number of ERISCs given arguments.
3. **The channel context.** What every channel needs: the channel allocator, which sender channel is each VC's worker

   channel, whether 2D routing and the channel trimming capture are on, speedy VC0, mux mode, the trimming profile the
    builder applied, and the channel buffer size.
4. **Channels**, VC by VC: each sender, then each receiver.
5. **Downstream edges**, VC by VC.
6. **Fields**: the router's, then each ERISC's.
7. **The rest**: identity, link, credit counters and leftover L1.

## How arguments are read

The kernel runs on one or more ERISCs, and each gets its own named compile-time arguments. Most values must be the same
on all of them, so how a value is read says which ERISCs it comes from:

| Reader | Reads from | Used for |
| --- | --- | --- |
| `read_emitted_value` | Every ERISC, which must agree | Router, receiver and edge values, and the values the computed facts use |
| `read_sender_channel_value` | The ERISCs that run the channel. One that doesn't may have the same value or 0 | Sender channel fields |
| `get_named_arg` | One ERISC | ERISC fields, and who services each channel |
| `has_define` | The defines every ERISC shares | VC gates, VC0 crossover |

Every reader refuses a name the field tables don't read and the router pass's lists don't name, and fails when the
argument is missing ([FIELD_TABLES.md](FIELD_TABLES.md)).

## The router

**Identity** (`identity`): the router's Ethernet channel. Its key and cores are added by the topology pass.

**Link** (`link`): the edge capability of the direction the router faces, from the chip's routing facts, and whether
the link is a dispatch link, from its location. The peer, `cross_host` and `wrap` are added by the topology pass.

**Shape** (`shape`):

| Key | From |
| --- | --- |
| `num_vcs`, `receivers_per_vc` | The builder's VC shape. The kernel is only fed the total number of receivers |
| `senders_per_vc` | `ACTUAL_VC<v>_SENDER_CHANNELS` |
| `num_active_eriscs` | `NUM_ACTIVE_ERISCS` |
| `channel_trimming_overrides_applied` | Whether the builder applied a channel trimming profile to the router |
| `vc0_bubble_flow_control` | `ENABLE_DEADLOCK_AVOIDANCE` |

The receivers the VC shape lists must add up to the kernel's `NUM_RECEIVER_CHANNELS`. Otherwise the manifest would list
receivers the kernel doesn't loop over, or leave out ones it does.

**Credit counters** (`credit_counters`): the router's four L1 arrays of credit counters, from the kernel's
`TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR`, `TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR`,
`LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR` and `LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR`. The kernel sends each pair to the peer in one packet, so it relies on the arrays being back to back and the same size, and the size comes from  
that spacing. The router pass fails if they aren't. Each array is `u32`s, one per sender channel, indexed by the
channel's compact index. The `to_sender` arrays count this router's own sender channels, and the `receiver` arrays the
sender channels of the router on the other end of the link. The writer records which as `indexed_by`.

**Leftover L1** (`leftover_l1`): the L1 from the end of the channel buffers to the end of loadable L1
(`max_l1_loading_size`), as bytes, or null when the buffers reach the end.

**ERISCs** (`eriscs`): one per RISC the kernel runs on, in RISC order, each with its processor, its fields, and its
physical ERISC from the HAL. The physical ERISC is the processor's index, except in Blackhole single-ERISC mode, where
base firmware keeps ERISC0 and the one processor runs on ERISC1. It indexes the `soft_reset` reset masks in
`arch.core_registers`.

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
2. `mux`: a VC0 sender other than the worker channel, in mux mode. The tensix mux carries its traffic.
3. `trimmed`: the applied channel trimming profile turned it off (a sender it doesn't use, or a receiver whose data it

   doesn't forward).

A channel no ERISC runs, for none of these reasons, fails the router pass: the manifest would otherwise call a broken
channel idle.

**Serviced by** (`serviced_by`): the ERISCs whose `IS_SENDER_CHANNEL_<c>_SERVICED` or `IS_RECEIVER_CHANNEL_<c>_SERVICED`
is set, as `erisc<n>`, where `<c>` is the compact index. Empty when the kernel doesn't run the VC.

**Ring buffer** (`ring_buffer`): the channel's packet slots, from the channel allocator, each `CHANNEL_BUFFER_SIZE`
bytes. Written as bytes, one element per slot, since a slot holds a packet rather than a described struct. Null when
the channel has no slots.

### Senders

**Producer** (`producer`): what feeds the channel. Each VC's worker channel is fed by `"worker"`, or by `"tensix_mux"` for
VC0 in mux mode. A channel fed by a sibling router gets that sibling's path from the topology pass. Otherwise null.

**Credits** (`credits`): where the channel's credits come back to it. `completed` always, and `acked` only on VC0 with
bubble flow control (first-level acks). How depends on the VC's backing:

- **L1 counters**, when the kernel's `VC<v>_USES_COUNTER_CREDITS` is set: the channel's element of the `to_sender`
counter array, written as `{array, index}`.
- **A stream register** otherwise: `TO_SENDER_<c>_PKTS_COMPLETED_ID` or `TO_SENDER_<c>_PKTS_ACKED_ID`, its
`buf_space_available` register. Null when the builder gave it the unused stream id.

The mesh's `credit_transport` says each VC's backing, and why ([DESCRIBE.md](DESCRIBE.md)).

### Receivers

The kernel runs VC0's receiver step on channel 0 and VC1's on channel 1. VC2's runs on channel 2 when VC1 has senders,
and on channel 1 otherwise. A serviced receiver whose compact index isn't the one the kernel runs fails the router pass.

**Forwards on** (`forwards_on`): the VC whose downstream edges the receiver's step is given, as `vc<n>`. Its own,
except that receiver 0 forwards on VC1 under `FABRIC_2D_VC0_CROSSOVER_TO_VC1`. Null when no ERISC runs the step, and
for the speedy steps, which forward to no sibling: VC2's always, and VC0's in speedy mode.

## Downstream edges

`intra_chip_downstream_edges` lists the router's persistent connections to its sibling routers' sender channels, on the
same chip and routing plane, by VC. They come from the receiver's downstream adapter in the ERISC builder.

- **Edge number** (`edge<n>`): the kernel's `EDGE_<n>`. In 2D it's the sibling's direction's compact index among this
router's other directions, plus one. In 1D there's one edge, `edge1`.
- **Landing channel**: the sibling's sender channel the edge lands on, by the builder's rule for the edge's VC and the
two routers' directions. The topology pass turns this and the sibling's direction into `downstream_channel`.
- **Core**: the NoC core the connection writes to. The topology pass compares it with the sibling's ERISC to set
`through_tensix_mux`.
- **Fields**: the edge's free-slots stream and teardown semaphore. The kernel takes an edge's teardown semaphore by its
rank among the router's edges, VC0's before VC1's.

The kernel never forwards on VC2, so VC2 has no edges. The checks keep the manifest numbering edges the way the kernel does:

- the adapter must name as many edges as the kernel's `NUM_DOWNSTREAM_SENDERS_VC<v>`;
- VC1 edges need `FABRIC_2D_VC1_ACTIVE`, which builds them;
- no two edges may share a slot, and the slots must match the mask the kernel is given.

## Adding a computed fact

1. Add the fact to the model (`fabric_manifest_model.hpp`) where it sits.
2. Compute it here, from `ManifestRouterInputs`. If it reads an argument or define no table reads, add the name to

   `k_router_pass_args` or `k_router_pass_defines`, and remove it from the unrecorded lists if it's there.
3. Write it in the writer (`fabric_manifest.cpp`), and add it to `fabric_manifest_schema.json` and

   [FORMAT.md](FORMAT.md).
4. Describe it in this doc, with the check that keeps it honest, if any.
