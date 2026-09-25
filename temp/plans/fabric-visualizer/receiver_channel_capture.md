# Receiver channel debug capture

What to capture per ERISC router receiver channel so a hang snapshot can show where every
received packet is, which stage is stuck, and which output is blocking it. Companion to
`sender_channel_capture.md`. Mode-dependent behavior (fused flush/completion, speedy, UDM,
first-level ack, TX queues, ...) is in `router_modes_and_variants.md`.

## Background: a slot's lifecycle

One receiver channel per VC; a ring of `M = RECEIVER_NUM_BUFFERS` slots in router L1. The producer
is the upstream router's sender channel(s) across the Ethernet link. Each packet leaves by one or
both of:

- **forward**: NoC write into the sender channel of the router facing the next direction on this
chip (persistent `EdmToEdmSender`, one per direction per VC);
- **local delivery**: NoC writes/atomics to the destination worker (or, in UDM mode, to a
credited local relay).

`ReceiverChannelPointers` (`edm_fabric_flow_control_helpers.hpp`) already keeps four free-running
`ChannelCounter`s. Ordering:

```text
completed <= flushed <= sent <= acked <= arrived
```


| Event                            | Where                            | Effect                                                                                                                                                                                                                          |
| -------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **arrived**                      | upstream `send_next_data`        | Payload lands over Ethernet. upstream sends remote `+1` to the receiver's `to_receiver_pkts_sent_id` register                                                                                                                   |
| **acked** (first-level ack only) | `run_receiver_channel_step_impl` | Ack sent upstream; `-1` on local `to_receiver_pkts_sent_id`; `ack_counter++`. Upstream frees the slot to *its* producer                                                                                                         |
| **sent**                         | same                             | Head (receiver channel consumer pointer) packet admitted; NoC writes issued to all selected outputs (tagged with the slot's trid); `wr_sent_counter++`. Without ack, `-1` on the `to_receiver_pkts_sent_id` stream happens here |
| **flushed**                      | same                             | All writes for the slot's trid have **left this core** (`..._write_with_transaction_id_sent`); slot bytes no longer needed. Not delivered; the router never waits for destination acks. `wr_flush_counter++`                    |
| **completed**                    | same                             | Completion credit sent upstream; `completion_counter++`. Only now does upstream's `num_free_slots` for this VC go up                                                                                                            |


Variants:

- `fuse_receiver_flush_and_completion_ptr`: flushed and completed collapse into `completion_counter`.
- Speedy path (`fabric_erisc_router_speedy_path.hpp`): local delivery only, no ack, ping-pong trid,
completions batched by `RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL` (`increment_n`). A
`sent - completed` gap up to that batch size is normal there.



### Admission (why the head packet may not move)

All-or-nothing (`admit_2d_dispatch` in 2D, `can_forward_packet_completely` in 1D). The head packet is
sent only if all of:

- every forward direction in its action has credit (`worker_credits_stream_id != 0` on that
direction's adapter; intermesh exit checks the boundary adapter);
- local relay has space (UDM mode only; plain local delivery has no credit check);
- the slot's trid is flushed (`enable_trid_flush_check_on_noc_txn`).

One full output stalls the whole receiver channel, including traffic bound elsewhere (head-of-line
blocking).

## Pitfalls

- `to_receiver_pkts_sent_id` **is not occupancy.** In ack mode it drops on ack. A receiver holding
`M` acked-but-stuck packets reads 0, making the upstream sender look like it refuses to feed an
empty receiver. Occupancy is `arrived - completed`.
- **Two different "free slots" registers per downstream link.** Our adapter's
`worker_credits_stream_id` (on this router) = producer-view free slots in the downstream sender
channel (`-1` on forward, `+n` on downstream's ack/completion). The downstream router's
`sender_channel_free_slots_stream_id` = `N - unsent` there. The gap is the downstream channel's
in-flight region plus credits in transit.
- **Flushed is not delivered.** Stuck writes past "left the core" are only visible in the NIU
outstanding counters.
- Ack and completion sends share the Ethernet TX queue with this router's sender forwards.



## What exists today


| Source                                                                                                                                                   | Gives                                                                                                             | Gap                                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `to_receiver_pkts_sent_id` stream reg                                                                                                                    | `arrived - acked` (ack mode) or `arrived - sent`                                                                  | Not occupancy; no progress                                  |
| `ReceiverChannelPointers` counters                                                                                                                       | All four positions                                                                                                | Router memory, compiler-chosen address (possibly registers) |
| Each downstream adapter's `worker_credits_stream_id` reg                                                                                                 | Free slots per downstream sender channel                                                                          | No progress info                                            |
| Receiver channel buffer in L1                                                                                                                            | Head packet header at the `sent` index: route/action bits, `src_ch_id`                                            | Need `sent` to locate it                                    |
| Counter-credit VCs: `local_receiver_ack/completion_counters_base_address` (this L1) and upstream `to_sender_remote_ack/completion_counters_base_address` | Free-running ack/completion counts per upstream sender channel, at CT-arg addresses, on **both** ends of the link | Stream-reg-credit VCs only hold unprocessed deltas          |
| NIU status regs `NIU_MST_WRITE_REQS_OUTGOING_ID(trid)`, `NIU_MST_REQS_OUTSTANDING_ID(trid)`                                                              | Per-trid writes not yet left / not yet acked                                                                      | Only if the dump can read NoC status regs                   |
| Telemetry RX heartbeat                                                                                                                                   | Router loop liveness                                                                                              | Per ERISC, not per channel                                  |




## Fields to publish



### Per receiver channel (one per VC)


| Field       | Type                     | Mirrors                      | Updated at                           |
| ----------- | ------------------------ | ---------------------------- | ------------------------------------ |
| `acked`     | `uint32_t`, free-running | `ack_counter.counter`        | ack increment (ack mode only)        |
| `sent`      | `uint32_t`, free-running | `wr_sent_counter.counter`    | sent increment                       |
| `flushed`   | `uint32_t`, free-running | `wr_flush_counter.counter`   | flush increment (omit in fused mode) |
| `completed` | `uint32_t`, free-running | `completion_counter.counter` | completion increment                 |




### Per output (per VC x direction, plus local / relay)


| Field             | Type                     | Updated at                                                                                                                                                         |
| ----------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `forwarded_count` | `uint32_t`, free-running | `forward_payload_to_downstream_edm` (all 1D/2D/intermesh forwards funnel through it), `execute_chip_unicast_to_relay`, `execute_chip_unicast_to_local_chip[_impl]` |


The credit stream gives current space; `forwarded_count` gives progress, and shows which outputs a
receiver actually feeds.

## Derived on host

With `s = read_stream_reg(to_receiver_pkts_sent_id)`:


| Quantity                               | Formula                                                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| arrived                                | `acked + s` (ack mode) or `sent + s`                                                           |
| occupancy                              | `arrived - completed`                                                                          |
| head index                             | `sent mod M`                                                                                   |
| gap sizes                              | `arrived - acked`, `acked - sent`, `sent - flushed`, `flushed - completed`                     |
| head packet route and blocking outputs | decode header at `slot[sent mod M]`; outputs whose credit stream is 0                          |
| downstream free slots                  | each adapter's `worker_credits_stream_id`                                                      |
| downstream producer index              | downstream router's `forwarded_count + unsent` for that sender channel (no need to store ours) |
| progress                               | counters changed between peeks                                                                 |




## Cross-router checks


| Check                                                                                   | Mismatch means                                               |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| upstream `downstream_write_idx == arrived mod M`                                        | Incoherent sample, or payload/stream update lost on link     |
| upstream `downstream_free_slots ≈ M - occupancy`                                        | Completions in transit or lost                               |
| our `forwarded_count` into direction D tracks D-router sender channel producer position | Forward/credit disagreement between routers on the same chip |
| counter-credit VCs: receiver ack/completion counters == upstream received copies        | Credits stuck or lost on the link                            |




## Occupancy definition

Same rule as `sender_channel_capture.md`, measured at the owning router:

> A slot is **occupied** from when its producer's write signal lands until the owner sends the
> credit that returns it to the producer.

For a receiver channel:
- start: **arrived** (upstream's remote `+1` on `to_receiver_pkts_sent_id` after the payload);
- end: **completed** (completion credit sent upstream). Not flushed: a flushed slot's data is dead,
  but upstream still cannot write it until the completion goes out;
- **occupancy = `arrived - completed`** (y / M).

Rejected alternatives:
- `arrived - flushed` (receiver's local "data no longer needed"): under-reports; a full ring can
  look partly empty while upstream is blocked.
- `M - upstream.num_free_slots` (upstream's view): needs another router's state sampled at a
  different moment, and counts packets on the wire and credits in flight. Reported as the sync check
  below instead.

Visual: stacked bar of length occupancy.

| Segment | Size | Style |
|---|---|---|
| not yet acked (ack mode) | `arrived - acked` | |
| waiting admission | `acked - sent` (or `arrived - sent`) | |
| NoC draining | `sent - flushed` | |
| dead but held (credit not sent) | `flushed - completed` | hatched; makes TX-queue stalls visible |

Notes:
- A packet between upstream forward and upstream credit occupies a slot upstream **and** here.
  Totals are slots, not packets.
- A payload still on the Ethernet wire (stream `+1` not landed) is not counted.
- Speedy path: `flushed` is not tracked; completions are batched, so the "dead but held" segment can
  legitimately reach `RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL`.

## Producer/owner sync check

For one link and VC:
- producer view: `up_occ = M - downstream_free_slots` (upstream router, per VC; sender doc field);
- owner view: `rx_occ = arrived - completed`.

The difference decomposes exactly into two non-negative transit terms:

```text
up_occ - rx_occ = (up_forwarded - arrived)            packets on the wire
                + (completed - up_completions_done)   credits in flight back
```

`up_forwarded` = sum of `forwarded_count` over the upstream sender channels on this VC (sender doc).

| Observation | Result |
|---|---|
| `diff < 0` | Error if repeated across peeks (impossible), else incoherent sample |
| `diff > 0`, changing or counters moving | Normal transit |
| `diff > 0`, identical across all peeks, no movement in `up_forwarded`, `arrived`, `completed` | **Warning** |

Attributing a warning:

| Check | Non-zero and stuck means |
|---|---|
| `up_forwarded - arrived` | Payload or its stream `+1` never landed on the receiver |
| counter-credit VCs: `local_receiver_completion_counters[src]` vs upstream `to_sender_remote_completion_counters[src]` | Completion credit lost on the link (works with no new instrumentation) |
| stream-credit VCs: upstream `to_sender_packets_completed` stream | Credit arrived upstream but its router is not servicing that sender channel |

Example message: "Link E→W VC0: upstream sees 8/8 full, receiver holds 3/8; 5 completions sent but
unprocessed upstream (stream = 5)."

The same check applies to each forward output (receiver → downstream sender channel over NoC):
producer view `N - worker_credits_stream` on this router vs owner view `unsent + in_flight` on the
downstream router; transit terms are writes/`-1` signals in flight and `+n` credits back. Split with
this output's `forwarded_count` vs the downstream channel's signaled writes (`forwarded_count +
unsent`).

Suppress on the speedy path until the batch threshold is exceeded.

## Decoder checks

Same post-kill, non-atomic caveats as the sender doc. Flag as "incoherent sample" unless repeated
across peeks.

- `completed <= flushed <= sent <= acked <= arrived` (drop terms that don't apply to the mode)
- `0 <= occupancy <= M`
- `0 <= s <= M`
- `0 <= worker_credits_stream <= N_downstream`



## Reading a stuck receiver


| Growing / stuck gap                            | Likely cause                                                                                                                                      |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `arrived - acked`                              | Acks not going out: Ethernet TX queue busy                                                                                                        |
| `acked - sent` (head not admitted)             | Decode head header: a selected forward direction has 0 credits (follow to that router's sender channel), UDM relay full, or slot trid not flushed |
| `sent - flushed`                               | NoC writes can't leave: backpressure toward local destination or downstream router; check NIU outgoing counters                                   |
| `flushed - completed`                          | Completions not going out: Ethernet TX queue busy                                                                                                 |
| receiver `completed` != upstream received copy | Credits lost on link (counter-credit VCs)                                                                                                         |
| occupancy `M`, stream 0                        | Not empty. Receiver full of acked packets blocked downstream (the ack-mode pitfall)                                                               |




## Implementation notes

- **Compile-time gated** debug-snapshot tier. One L1 store at each existing counter increment, plus
one per output forward.
- **Hook both paths**: `run_receiver_channel_step_impl` (`fabric_erisc_router.cpp`) and
`run_receiver_channel_step_speedy` (`fabric_erisc_router_speedy_path.hpp`, `increment_n` for batched
completions).
- Mirroring vs placing `ReceiverChannelPointers` at a fixed address: equivalent cost once stores
must be volatile; mirroring keeps the hot-path struct untouched.
- **Blackhole data cache**: use volatile stores; confirm a UMD peek sees them.
- **Placement**: fixed region in the debug-snapshot leftover, registered as a named region in the
topology manifest.



## Topology manifest requirements

The cross-router edges cannot be drawn without:

- per receiver (VC, my direction): which downstream router and sender channel each compact
downstream index feeds (host slot map, VC crossover);
- which VCs use counter credits vs stream-register credits;
- `M`, trid range per receiver channel, fused / first-level-ack / speedy / UDM flags.



## Optional additions


| Field                                     | Why                                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------ |
| `eth_txq_busy` snapshot (per ERISC)       | Shared cause of the `arrived - acked` and `flushed - completed` gaps                 |
| last-admit-failure bitmask (per receiver) | Which outputs lacked space / trid unflushed. Only if header decode proves unreliable |




## Out of scope

- Per-packet tracing (see packet-trace plan).
- Destination-side delivery confirmation (needs NIU ack counters or destination kernel state).
- Tensix mux/relay/UDM extension internals.

