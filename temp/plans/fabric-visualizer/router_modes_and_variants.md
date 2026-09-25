# Router modes and variants

Compile-time modes and behaviors that change what the sender/receiver capture fields mean, which
fields exist, or how the checks in `sender_channel_capture.md` and `receiver_channel_capture.md`
must be interpreted. The decoder should read the active modes from the launch snapshot / topology
manifest and adjust derivations, segments, and warning thresholds accordingly.

Flags are in `fabric_erisc_router_ct_args.hpp` unless noted. **Status**: *verified* = read in code;
*verify* = inferred, confirm before relying on it.

## Summary

| # | Mode | Flag | Affects | Status |
|---|---|---|---|---|
| 1 | Fused flush + completion | `FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR` | receiver | verified |
| 2 | First-level ack off | `ENABLE_FIRST_LEVEL_ACK_VCx` (per VC) | receiver, sender | verified |
| 3 | Speedy receiver | `ENABLE_SPEEDY_VC0` (`super_speedy_mode`) | receiver | verified (VC0); verify (VC2) |
| 4 | Receiver forwarding disabled | `is_receiver_channel_forwarding_disabled`, `disable_rx_ch0_forwarding` | receiver | verified |
| 5 | UDM mode | `UDM_MODE` | receiver | verified |
| 6 | Trid flush check at admission | `enable_trid_flush_check_on_noc_txn` | receiver | verified (hard-coded `false`) |
| 7 | Fixed source channel ID | `skip_src_ch_id_update` (= Tensix mux mode) | receiver | verified |
| 8 | 1D vs 2D, intermesh, VC crossover | `FABRIC_2D`, `is_intermesh_router*`, downstream VC arrays | receiver, manifest | verified |
| 9 | Speedy sender | `ENABLE_SPEEDY_VC0` | sender | verified |
| 10 | Bubble flow control | `ENABLE_DEADLOCK_AVOIDANCE` | sender | verified |
| 11 | Tensix mux in front of router | `fabric_tensix_extension_mux_mode` | sender | verify |
| 12 | Separate TX queues | `SENDER_TXQ_ID != RECEIVER_TXQ_ID` (`multi_txq_enabled`) | both | verified |
| 13 | Spin-wait on TX queue | `ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA`, `ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK` | both | verified |
| 14 | Stream-register vs counter credits | `VCx_USES_COUNTER_CREDITS` (per VC) | both | verified |
| 15 | Two ERISCs per router | `NUM_ACTIVE_ERISCS == 2` (Blackhole) | both | verify (step split) |
| 16 | Context switch | `ENABLE_CONTEXT_SWITCH` | both | verified |
| 17 | Channel trimming | `is_sender_channel_serviced`, `is_receiver_channel_serviced` | both | verified |

## Receiver

### 1. Fused flush + completion

- **What changes**: only `completion_counter` moves. It advances when the slot's trid is flushed
  **and** the completion is sent, in one step (`fabric_erisc_router.cpp`, fused branch of
  `run_receiver_channel_step_impl`). `wr_flush_counter` is unused.
- **Capture**: publish `completed` only; `flushed` absent.
- **Visual**: "dead but held" segment is always 0. A flushed slot waiting on a busy TX queue is
  indistinguishable from one still draining to the NoC; both fall into `sent - completed`.
- **Disambiguation**: read the NIU outgoing counter for the head completion slot's trid. Zero means
  NoC done, so the slot is waiting on the TX queue.

### 2. First-level ack off

- **What changes**: `ack_counter` unused; `to_receiver_pkts_sent_id` is decremented on **sent**. On
  the sender side, credits to the producer come on **completion** instead of ack.
- **Capture**: `acked` absent.
- **Derivation**: `arrived = sent + stream`.
- **Visual**: "not yet acked" segment merges into "waiting admission" (`arrived - sent`).
- **Sender**: `in_flight` covers forward → completion (longer than forward → ack).

### 3. Speedy receiver

- **What changes** (`run_receiver_channel_step_speedy`): local delivery only, no ack, no forwarding,
  ping-pong between two trids instead of per-slot trids, completions batched by
  `RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL` via `completion_counter.increment_n`. Static asserts
  require no deadlock avoidance, no VC0 first-level ack, trimmed non-worker VC0 senders, RX0
  forwarding disabled.
- **Capture**: `acked`, `flushed` absent; hook `sent` and `completed` in the speedy function.
- **Checks**: `sent - completed` up to the batch size is normal. Occupancy "dead but held" and
  producer/owner sync warnings need a threshold of the batch size.
- **Verify**: VC2 instantiates `ActualSpeedy*State<true>` unconditionally
  (`fabric_erisc_router.cpp`); confirm whether VC2 always runs the speedy steps.

### 4. Receiver forwarding disabled

- **What changes**: `sent` advances without issuing forward writes.
- **Capture/checks**: per-output `forwarded_count` stays 0 by design; do not warn.

### 5. UDM mode

- **What changes**: local delivery goes through `local_relay_interface`, a credited adapter
  (`edm_has_space_for_packet`), instead of direct NoC writes. Admission can stall on relay space.
- **Capture**: treat the relay as one more output: `forwarded_count` plus its credit stream.
- **Checks**: add a producer/owner pair for receiver → relay (relay side is out of scope; producer
  view only).

### 6. Trid flush check at admission

- **What changes**: `enable_trid_flush_check_on_noc_txn` is `constexpr false`, so admission never
  waits on the slot's trid today.
- **Checks**: drop "slot trid not flushed" from blocked-head causes, or show it only if the flag
  becomes true.

### 7. Fixed source channel ID

- **What changes**: with `skip_src_ch_id_update` the header's `src_ch_id` is not written by the
  upstream sender, and the receiver credits one fixed upstream sender channel.
- **Checks**: per-upstream-channel credit attribution is trivial; do not trust `src_ch_id` decoded from
  dumped headers.

### 8. 1D vs 2D, intermesh, VC crossover

- **1D**: single downstream, `can_forward_packet_completely`; routing fields rewritten per hop;
  write-only / forward-only / write-and-forward combinations. Head-header decode differs.
- **2D**: action-map decode, all-or-nothing `admit_2d_dispatch` across LIVE directions.
- **Intermesh**: landing receivers rebuild the route before decode; exit chips forward on the
  boundary adapter only (`intermesh_egress`).
- **VC crossover**: the selected downstream array determines the carrier VC, so a receiver can feed
  another VC's sender channel.
- **Manifest**: must map each receiver (VC, direction, compact index) to the downstream router and
  sender channel it actually feeds, including crossover.

## Sender

### 9. Speedy sender

- **What changes** (`run_sender_channel_step_speedy`): completions are only processed after
  `SENDER_CREDIT_AMORTIZATION_FREQUENCY` sends, and credits to the worker are batched the same way.
  No first-level ack, no bubble flow control.
- **Checks**: `in_flight` (drops on credit) and `downstream_free_slots` lag by up to the batch size.
  Sync warnings need that threshold.
- **Hooks**: speedy send and credit sites (same file).

### 10. Bubble flow control

- **What changes**: on traffic-injection channels (`sender_channel_is_traffic_injection_channel`),
  sending requires `num_free_slots >= BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS`
  instead of `> 0`.
- **Checks**: "downstream free slots > 0 but not sending" is expected below the threshold. The
  stuck-cause table must use the per-channel threshold.

### 11. Tensix mux in front of router

- **What changes**: the sender channel's producer is a Tensix mux, not a worker.
- **Verify**: whether the mux connects persistently or open/close, and which credit mode it uses.
- **Checks**: replaces the "worker → sender channel" row; mux internals out of scope.

## Both

### 12. Separate TX queues

- **What changes**: receiver acks/completions and sender forwards use different Ethernet TX queues.
  The code comment in the ct args also says multi-TXQ requires counter credits on every VC.
- **Capture**: `eth_txq_busy` per queue, not per ERISC.
- **Checks**: "TX queue busy" causes must name the queue (sender vs receiver).

### 13. Spin-wait on TX queue

- **What changes**: the router spins inside the send/ack/completion call until the queue frees,
  instead of skipping and retrying next loop.
- **Checks**: a TX-queue stall shows as the **whole router loop hung** (heartbeats stop, no counter
  moves anywhere on that ERISC), not as one gap growing. Different signature from the default.

### 14. Stream-register vs counter credits

- **What changes**: ack/completion credits travel via L1 counters at CT-arg addresses (both ends) or
  via stream registers (upstream holds only unprocessed deltas).
- **Capture**: none extra. The sender's freed hook (`send_credits_to_upstream_workers`) is
  mode-independent.
- **Checks**: counter-credit VCs allow direct receiver vs upstream counter comparison (lost credits
  on link); stream-credit VCs only show "arrived upstream but unprocessed".

### 15. Two ERISCs per router

- **What changes**: on Blackhole, sender and receiver steps may run on different ERISCs (separate
  stacks, possibly separate data caches). Telemetry already splits by `MY_ERISC_ID`.
- **Capture**: each field needs a single owning ERISC; per-ERISC items (heartbeat,
  `eth_txq_busy`) double.
- **Verify**: exact split of sender/receiver steps and VCs across ERISCs.

### 16. Context switch

- **What changes**: speedy state is copied into locals for the loop and back to persistent storage
  around context switches (`speedy_state_copy_in/out`).
- **Capture**: unaffected if fields are mirrored with volatile stores at update time; do not rely on
  the persistent copies.

### 17. Channel trimming

- **What changes**: unserviced sender/receiver channels have no state and are never stepped.
- **Decoder**: show as absent, not as zero or stuck. Exclude from sync checks.

## Decoder requirements

The launch snapshot / manifest must record, per router: the flags above (per VC where applicable),
`N` per sender channel, `M` per receiver channel, amortization frequencies, bubble threshold, trid
ranges, TX queue IDs, `NUM_ACTIVE_ERISCS`, and serviced-channel masks. Without them the decoder
cannot tell a mode-expected gap from a hang.
