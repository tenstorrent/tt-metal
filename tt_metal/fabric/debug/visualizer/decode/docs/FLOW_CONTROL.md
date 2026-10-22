# Flow control

`[flow_control.py](../flow_control.py)` reads what a router's channel counts and credits say:

- for a sender channel, how many of its packets it hasn't sent yet, how many its producer hasn't been credited for yet,
and the credits coming back from the peer's receiver;
- for a receiver channel, how many of its packets wait.

They're counts, read from stream registers and L1 words. Which packets they count isn't in the snapshot: the router
keeps its slot positions in its own variables (see "No slot state" in [RING_BUFFER_SLOTS.md](RING_BUFFER_SLOTS.md)).
They're facts, not verdicts: whether a count means a hang is left to the reader.

## What it takes and returns

`get_channel_flows(snapshot, decoded_routers, router_path)` takes the snapshot, every router of the snapshot decoded by
`[routers.py](../routers.py)`, and the router to read. It needs the other routers because two of a sender channel's
values live on them:

- **The sibling.** A packet that crosses a chip comes in over one router's link, into that router's receiver channel.
The receiver forwards it into a sender channel of another router on the same chip, which sends it out over its own
link. The first router is the sibling: it feeds the sender channel, and the manifest names it as the channel's
`producer`. The sibling keeps its own count of the channel's free slots, which gives the channel's uncredited count.
- **The peer,** the router across the link a sender channel sends on. Its receiver returns the sender's credits, and
under L1-counter credits, it keeps their running totals.

For example, on T3K, `M0/C0/S1` forwards packets that came in over its link into `M0/C0/E1`'s sender channel `ch3`, and
E1 sends them on to its peer, the router across its link. So for E1's `ch3`, S1 is the sibling, and S1's
`intra_chip_downstream_edges` has the edge into that channel.

It gives each active channel's flow, by the channel's path in the router. A channel no ERISC runs has none: its control
words were never initialized.

Every count is a `Count`:


| Field             | What it holds                                                                                                                     |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `before`, `after` | The count, from the register read before the router's L1 and the one after. A count from L1 is read once, so the two are the same |
| `not_observable`  | When the snapshot can't give the count, why. `before` and `after` are then `None`                                                 |


A `SenderFlow`:


| Field           | What it holds                                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `depth`         | The channel's slots                                                                                                                    |
| `producer`      | The manifest's: `worker`, `tensix_mux`, a sibling router's path, or `None` when nothing feeds it                                       |
| `connection`    | On a worker channel, its connection word: unused, open or close requested. `None` elsewhere                                            |
| `producer_core` | On a worker channel, the worker's core from `conn_info.worker_xy`, and whose it is: `current` while a worker is connected, else `last` |
| `unsent`        | Packets the producer wrote that the router hasn't sent yet                                                                             |
| `uncredited`    | Packets the producer wrote that it hasn't been credited for yet: the unsent ones, and those sent but not yet credited                  |
| `credits`       | The credits coming back from the peer's receiver, a `CreditFlow` per credit type                                                       |
| `speedy_batch`  | On a speedy VC0, the batch credits are processed in. `None` elsewhere                                                                  |
| `within_bounds` | Whether `0 <= unsent <= uncredited <= depth`, wherever the values were read                                                            |
| `moved`         | Whether a register's count changed between before and after: traffic moved during the capture                                          |


A `ReceiverFlow` is its `depth`, `waiting` and `unacked` (see "The counts" below), and `moved`.

## A sender slot's life

A packet holds its sender slot from when its producer writes it until the router credits the producer back:

```
producer writes the slot -> router sends it over Ethernet -> peer acks or completes it -> router credits the producer
|---------- unsent ----------|------------------ sent, not yet credited -------------------|
|------------------------------------------ uncredited ------------------------------------|
```

Two directions of credit meet at the router:

- **Credits in** come from the peer's receiver: a completion when it has freed one of its slots, and an ack, with
first-level ack, when the packet has landed there. They're the channel's `credits`.
- **Credits out** go to the producer. The router turns each credit that comes in into one that goes out: at the ack
with first-level ack, else at the completion (`send_credits_to_upstream_workers`). `uncredited` measures what hasn't
gone out yet.



## The counts

Each count belongs to one kind of channel. "The router" is the router holding the channel.


| Count        | Channel                                      | Read from                                                                                                                                | Goes up                                               | Goes down                                                                            |
| ------------ | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `unsent`     | Every sender                                 | the router's `free_slots` register for the channel: `depth - free_slots`                                                                 | the producer, as it writes a packet                   | the router, right after the Ethernet send                                            |
| `uncredited` | A sender fed by a sibling                    | the sibling's register for its edge into the channel (`intra_chip_downstream_edges.<vc>.<edge>.fields.free_slots`): `depth - free_slots` | the sibling, as it forwards a packet into the channel | the router, as it credits the sibling (`notify_persistent_connection_of_free_space`) |
| `uncredited` | The worker channel, with no worker connected | the router's L1: `buffer_index_sem.write_counter - conn_info.edm_read_counter`, mod 2^32                                                 | each worker, as it writes                             | the router, as it credits                                                            |
| `waiting`    | A receiver, without first-level ack          | the router's `pkts_sent` register for the channel                                                                                        | the peer's sender, over Ethernet, after each send     | the router, once it has forwarded or delivered the packet                            |
| `unacked`    | A receiver, with first-level ack             | the router's `pkts_sent` register for the channel                                                                                        | the same                                              | the router, at the ack                                                               |


The credits below are a sender channel's too: a receiver channel returns credits, and keeps none of its own to report.

- **Every sender's** `free_slots` **starts at its depth** (`init_ptr_val` in the router), and so does a sibling's view of
it: `RouterStaticSizedChannelWriterAdapter` starts its edge `free_slots` at the depth.
- **First-level ack** is on for VC0 exactly when the router's `shape.vc0_bubble_flow_control` is, and never on VC1. A
receiver reports the one of `waiting` and `unacked` that `pkts_sent` counts, and the other says why it isn't
observable.
- **A channel nothing feeds** has no `uncredited`: there's no producer to credit.
- **Reads can race while traffic moves.** Each register is read at its own moment, so `within_bounds` can fail while
packets flow. `moved` says when they did. In a hang, the counts hold still.



## The worker channel

The worker channel (VC0's channel 0, and VC2's) is fed by workers, each connecting, sending and closing in turn. Its
producer is `worker`, or `tensix_mux` in mux mode. The router credits a worker by bumping its read counter, an absolute
count:

- **While a worker is connected,** the router writes the read counter into the worker's L1, and the worker keeps its
write counter in its own variables. So `uncredited` isn't observable. `conn_info.edm_read_counter` and
`buffer_index_sem` are stale then: their values from when this worker connected.
- **While none is connected** (`connection` unused), the router writes its read counter into
`conn_info.edm_read_counter` on every credit (`copy_read_counter_to_worker_location_info`), and `buffer_index_sem`
holds the last worker's write counter, which it wrote as it closed (`close_start`). Both are current, so `uncredited`
is their difference. This is what the next worker reads when it connects (`open_start`).
- **A** `connection` **stuck at close requested** is a worker waiting in `close_finish` for a router that hasn't processed
the close. The router keeps sending what's left in the ring after a close.
- `free_slots` **at the depth doesn't mean a worker has room.** Sent packets hold their slots until they're credited,
and a worker counts its room from its read counter.

On every other channel, decode doesn't read `conn_info` or `buffer_index_sem`. A sibling connects once and its words
never change, and the manifest has no `buffer_index_sem` there: nothing writes it, and on Wormhole it holds whatever L1
held.

## Credits coming back

A sender gets credits back from the receiver at the other end of its link: `completed` always, which frees a slot of
the peer's receiver, and `acked` with first-level ack, which frees the sender's own slot. Each VC carries them one of
two ways, which the mesh's `credit_transport` names, with its reasons. The reference's shape says the same, and decode
refuses one that disagrees.


|                                                                 | Stream register                                         | L1 counter                                                                                                                                     |
| --------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| The receiver                                                    | Adds the credits to the sender's register over Ethernet | Keeps a running total in its own L1 (`credit_counters/receiver_*`), and copies the whole block into the sender's `credit_counters/to_sender_*` |
| The sender                                                      | Reads the register, then subtracts what it processed    | Reads the total, less what it processed, which it keeps in its own variables                                                                   |
| `pending`: landed, not yet processed                            | The register                                            | Not observable                                                                                                                                 |
| `total_landed`: landed since the router started                 | Not observable: the register is used up                 | The sender's `to_sender_*` element                                                                                                             |
| `peer_total_sent`: sent by the peer's receiver since it started | Not observable                                          | The peer's `receiver_*` element, when the peer is in the snapshot                                                                              |
| `on_link`: `peer_total_sent - total_landed`, signed             | Not observable                                          | Normally 0. Positive: a copy in flight, or a broken link if it stays. Negative: the sender was read after its peer, while credits flowed        |


- **Indices.** The sender's `to_sender_`* arrays are indexed by its own sender channels, and the peer's `receiver_*`
by its peer's, which are the same channels (`index_space`). The receiver indexes its totals by the sender channel a
packet came from (`ReceiverChannelCounterBasedResponseCreditSender`).
- **Reading `on_link`.** The totals are 32-bit counters that wrap, so `on_link` is their difference as a signed 32-bit
value: a wrap between them still subtracts right. Capture reads each router at its own moment, so while credits flow,
the two totals come from different moments, and a nonzero `on_link` means something only when neither router's counts
moved. A negative one can only be that timing: a sender can't have landed more than its peer sent.
- **Speedy VC0** processes completions only every `sender_credit_amortization` sends, so a `pending` up to that batch is
normal. `speedy_batch` says so.
- **Across snapshots,** `total_landed`'s change is the credits returned in between: a progress signal. A `pending` above
the speedy batch that doesn't change means the sender isn't processing what landed.

In the three real snapshots (Wormhole T3K 2D, and Blackhole 1D and 2D, all idle), every count is 0: nothing unsent,
uncredited or waiting, and every credit total, stream register and `on_link` is 0. Every worker channel is unused, with
(0, 0) as the last worker's core: no worker had connected.

## The names it reads

The meaning of these names is in decode's code (case 2 of "What decode reads, and what it knows" in the
[README](../README.md)). A renamed one fails with `ManifestError`:

- `enums.ConnectionValue`: `unused_connection_value`, `open_connection_value` and `close_connection_request_value`;
- `meshes.M<n>.credit_transport.<vc>`: `backing`, `stream_register` or `l1_counter`, and `reasons`;
- the router's `link.peer`, `shape.vc0_bubble_flow_control`, `fields.speedy_vc0`, `fields.sender_credit_amortization`,
`credit_counters.<array>.index_space`, and `intra_chip_downstream_edges` with each edge's `downstream_channel`;
- each channel's `status` and `ring_buffer.num_elements`;
- a sender's `producer`, its `credits`, `completed` and `acked`, and its `free_slots`, `connection`, `conn_info`
(`worker_xy`, `edm_read_counter`) and `buffer_index_sem` (`write_counter`);
- a receiver's `pkts_sent`.



## What it doesn't cover

- **Which packets the counts are.** That needs the slot positions in L1.
- **The sender's count of the peer receiver's free slots,** which is what lets it send. It's a local variable (`num_free_slots`), under both transports.
- `pending` **under L1 counters,** and `uncredited` **on a connected worker channel.** Both need values the router or worker keeps in local variables.
- **A peer on another rank.** Its credit totals are in that rank's snapshot.



## Files cited


| File                                                                    | Symbols                                                                                                                                                                                         |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`       | `send_credits_to_upstream_workers`, the sender step's sends and credit processing, `get_credits_init_val`, the `init_ptr_val` calls at startup, the receiver step's `pkts_sent`                 |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` | `notify_persistent_connection_of_free_space`, `notify_worker_of_read_counter_update`, `increment_local_read_counter`, `copy_read_counter_to_worker_location_info`, `teardown_worker_connection` |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp`      | `ReceiverChannelCounterBasedResponseCreditSender`, `SenderChannelFromReceiverCounterBasedCreditsReceiver`, `SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver`               |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`      | `WorkerToFabricEdmSenderImpl`: `open_start`, `open_finish`, `close_start`, `close_finish`, `get_num_free_write_slots`                                                                           |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_adapter.hpp`           | `RouterStaticSizedChannelWriterAdapter`                                                                                                                                                         |


