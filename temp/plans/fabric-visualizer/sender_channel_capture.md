# Sender channel debug capture

What to capture per ERISC router sender channel so a hang snapshot can show where every
packet is, and which stage is stuck. Mode-dependent behavior (speedy, bubble flow control,
first-level ack, TX queues, ...) is in `router_modes_and_variants.md`.

## Background: a slot's lifecycle

A sender channel is a ring of `N` slots in router L1. The producer is a worker (open/close
connection) or the upstream router (persistent connection). Each slot moves through three
stages:

```text
   freed              forwarded             written
     v                    v                    v
 ... [ in flight: sent, awaiting credit ][ unsent: waiting ][ free ... ]
```

| Event | Where | Effect |
|---|---|---|
| **written** | producer adapter (`edm_fabric_worker_adapters.hpp`) | Producer fills `slot[producer_idx]`, remote `-1` on router's unsent stream |
| **forwarded** | `send_next_data` (`fabric_erisc_router.cpp`) | Router sends slot over Ethernet, advances consumer index, local `+1` on unsent stream. Slot is **not** reusable yet |
| **freed** | `send_credits_to_upstream_workers` | Downstream receiver confirmed; producer gets a credit and may reuse the slot |

"Confirmed" = whatever produces the credit: the **ack** when `enable_first_level_ack`, otherwise
the **completion**. The downstream receiver is the neighbor router's receiver channel on this
link (next hop, not final destination).

The producer only ever sees the freed position (read counter for workers, `+n` stream credit for
persistent peers). It stalls in `wait_for_empty_write_slot` (watcher waypoint `FWSW`) when
`written - freed == N`.

## What exists today and why it is not enough

| Source | Gives | Gap |
|---|---|---|
| Unsent stream reg `sender_channel_free_slots_stream_id` | `N - value` = unsent count | Nothing about in-flight |
| `sender_buffer_channel.next_packet_buffer_index` | consumer index | Private member, compiler-chosen address; wrapped index |
| `local_read_counter` (worker interface) | freed count, worker mode only | Not at fixed address; **does not exist** for persistent mode |
| Parked `edm_read_counter` in `EDMChannelWorkerLocationInfo` | freed count | Only synced on teardown / while disconnected; stale while connected |
| `SenderChannelProducerCursor` | producer position | Written by worker **only on close**; router never updates it |
| `local_write_counter` (worker interface) | — | Written but never read in the ERISC router (dead); used only by Tensix mux/relay kernels |
| `outbound_to_receiver_channel_pointers` | downstream write ptr, free slots | Router memory, not fixed address |
| Telemetry TX/RX heartbeats | router loop liveness | Per ERISC, not per channel. TX heartbeat counts "no unsent packets" as idle, so it **keeps ticking** when everything is forwarded but credits never return |

Wrapped indices alone are ambiguous: `freed_idx == consumer_idx` means either 0 or `N` in
flight. Counts are required.

## Fields to publish

### Per sender channel

| Field | Type | Updated in | Update |
|---|---|---|---|
| `forwarded_count` | `uint32_t`, free-running | `send_next_data` | `+= 1` |
| `in_flight` | `uint32_t`, in `[0, N]` | `send_next_data` / `send_credits_to_upstream_workers` | `+= 1` / `-= n` |

`send_credits_to_upstream_workers` is the single point where credits are returned for both
worker and persistent modes, so one hook covers both.

### Per VC (shared by all sender channels on that VC)

| Field | Type | Updated in | Update |
|---|---|---|---|
| `downstream_write_idx` | `uint32_t`, wrapped to `RECEIVER_NUM_BUFFERS` | `send_next_data` | mirrors `advance_remote_receiver_buffer_pointer` |
| `downstream_free_slots` | `uint32_t` | `send_next_data` / completion processing | mirrors `outbound_to_receiver_channel_pointers.num_free_slots` (`-1` on send, `+n` on **completion**, not ack) |

`outbound_to_receiver_channel_pointers` is indexed by `VC_RECEIVER_CHANNEL`, so these are one
per link+VC, not per sender channel. The visualizer should draw them once, fed by several sender
channels.

## Derived on host

With `unsent = N - read_stream_reg(sender_channel_free_slots_stream_id)`:

| Quantity | Formula |
|---|---|
| consumer index (next slot to send) | `forwarded_count mod N` |
| earliest in-flight index | `(forwarded_count - in_flight) mod N` |
| producer index (next slot producer writes) | `(forwarded_count + unsent) mod N` |
| occupied (producer's view) | `in_flight + unsent` |
| producer free slots | `N - in_flight - unsent` |
| channel progress | `forwarded_count` changed between peeks |

The producer index is the position the producer has **signaled** (written and done its `-1`). A
producer mid-write may already be one slot ahead in its own `buffer_slot_index`. That is the
correct definition for a hang snapshot.

Index-from-count assumes counts start at 0 at router start and have not wrapped 2^32 with a
non-power-of-two `N`; not a concern for debug runs.

## Occupancy definition

One rule for every buffer (shared with `receiver_channel_capture.md`), measured at the router that
owns the buffer:

> A slot is **occupied** from when its producer's write signal lands until the owner sends the
> credit that returns it to the producer.

"Reusable" only matters to the producer (the owner never writes its own slots), and the credit is
what grants reuse. Using the owner's counters keeps a snapshot single-router; disagreement with the
producer's view is reported separately (see sync check below), not folded into the definition.

For a sender channel:
- start: producer's `-1` on `sender_channel_free_slots_stream_id`;
- end: `send_credits_to_upstream_workers` (on ack with `enable_first_level_ack`, otherwise on
  completion);
- **occupancy = `unsent + in_flight`** (x / N).

Visual: stacked bar of length occupancy.

| Segment | Size |
|---|---|
| unsent | `N - stream value` |
| in flight, awaiting credit | `in_flight` |

Notes:
- A packet between forward and credit occupies a slot here **and** in the downstream receiver. Totals
  are slots in use, not packets; do not sum across buffers.
- A producer mid-write (signal not yet landed) is not counted.

## Producer/owner sync check

The producer's view of a buffer can only exceed the owner's view by items in transit. A persistent
gap in a quiescent snapshot means something was lost or stuck.

The pairs that involve a sender channel:

| Pair | Producer view | Owner view (this doc) | Legitimate transit |
|---|---|---|---|
| upstream receiver → this sender channel (persistent, same chip, NoC) | `N - worker_credits_stream` on the upstream router's adapter for this direction | `unsent + in_flight` | NoC writes + `-1` signals in flight; `+n` credits in flight back |
| this sender channel → downstream receiver (Ethernet) | `M - downstream_free_slots` (per VC, this router) | receiver `arrived - completed` | see `receiver_channel_capture.md` |
| worker → this sender channel | worker `write - read` | `unsent + in_flight` | Not capturable (worker counters at non-fixed L1); skip |

Rule, with `diff = producer_occ - owner_occ`:

| Observation | Result |
|---|---|
| `diff < 0` | Error if repeated across peeks (impossible), else incoherent sample |
| `diff > 0`, changing or counters moving | Normal transit |
| `diff > 0`, identical across all peeks, no counter movement on either side | **Warning**: lost/stuck signal or credit |

For the NoC pair, split a warning by comparing the upstream receiver's `forwarded_count` into this
direction with this channel's signaled writes (`forwarded_count + unsent`): a gap there means a
write/`-1` never landed; otherwise a `+n` credit never landed.

## Decoder checks

The dump is post-kill and non-atomic (stream reg and L1 words read at different times; router may
still be running). Flag violations as "incoherent sample" unless they repeat across all peeks.

- `0 <= in_flight <= N`
- `0 <= unsent <= N`
- `in_flight + unsent <= N`
- `0 <= downstream_free_slots <= RECEIVER_NUM_BUFFERS`

## Reading a stuck channel

| Snapshot (stable across peeks) | Likely cause |
|---|---|
| `unsent > 0`, `forwarded_count` not moving, `downstream_free_slots == 0` | Downstream receiver full (check sibling sender channels on same VC, and the neighbor's receiver) |
| `unsent > 0`, not moving, `downstream_free_slots > 0` | Ethernet TX queue busy, or bubble flow control threshold (`BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS`) not met on an injection channel |
| `unsent == 0`, `in_flight > 0`, not moving | Acks/completions not returning from downstream; producer stalled at `FWSW`. TX heartbeat still ticks here |
| `unsent == 0`, `in_flight == 0`, producer at `FWSW` | Credit never reached producer (lost NoC write, or parked while disconnected and not picked up) |
| `in_flight + unsent == N` | Channel full from producer's view |

## Implementation notes

- **Gate at compile time** (debug-snapshot tier). Cost is one L1 store per forwarded packet and
  per credit batch, on the hottest path.
- **Hook both send paths**: `fabric_erisc_router.cpp` and `fabric_erisc_router_speedy_path.hpp`
  (calls `update_write_counter_for_send` and `send_credits_to_upstream_workers` too).
- The dead `update_write_counter_for_send()` call in `send_next_data` sits exactly where
  `forwarded_count` / `in_flight` need updating; it can be replaced in the ERISC router. Do not
  remove `local_write_counter` from the shared struct; Tensix mux/relay kernels use it.
- **Blackhole data cache**: ERISC L1 data cache (`ENABLE_RISC_CPU_DATA_CACHE`) may keep stores out
  of L1. Use volatile stores into the debug region and confirm a UMD peek sees them.
- **Placement**: fixed region in the allocator-owned debug-snapshot leftover; register it as a
  named region in the topology manifest so the decoder does not hardcode addresses.

## Optional additions

| Field | Why |
|---|---|
| `connection_epoch` (per channel, `+1` on connect) | Distinguish "same worker still stuck" from "new worker, same symptom" |
| `flags` (per channel): connected, persistent | Label open/close vs persistent and current connection state; worker xy already in `EDMChannelWorkerLocationInfo` |
| `eth_txq_busy` snapshot (per ERISC) | Confirms the TX-queue cause, the one stall reason the fields above cannot show |

## Out of scope

- Worker-side live producer pointer (lives in worker L1 at a non-fixed address).
- Receiver channel capture (separate doc).
- Tensix mux/relay/UDM extensions.
