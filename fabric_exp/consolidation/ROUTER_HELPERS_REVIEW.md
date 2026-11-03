# Router helper files: review against the sender/receiver cleanup rules (2026-11-03)

This is the main session's own pass over the v4 router files outside the sender, receiver and downstream code, in
v4txo8 (sender final) and v4rxnsf2 (receiver N/S fix). It applies the rules the user set for the sender and receiver:

- facts live in their owners;
- nothing positional (features per channel, not channel numbers);
- no loose helpers or duplicated facts;
- no one-off structs;
- no study pointers in comments (variant names, RESULTS / LEDGER ids, V4_*.md, dates, cycle numbers);
- NoC only through named API helpers;
- compiler-only primitives from the allowed list.

For the user's review first; then it goes to the sender consolidation agent (V4_SENDER_CONSOLIDATION.md stages)
where it fits.

## A. Router-local files (router/*.hpp, fabric_erisc_router_v4.cpp)

### A1. router_common.hpp (308 lines)
- **Sender facts in the common file.** `sender_free_slots_streams`, `sender_connection_info_addrs` and
  `sender_connection_semaphore_addrs` are hand-written 10-entry lists of host args, one per channel number.
  - They belong to their owner: SenderChannel reads its own, or the ct args file provides each as one array.
  - They also sit under a "Downstream routers (2D)" section header that describes the next block, not them.
- **Downstream facts by VC number and position.**
  - `downstream_sender_free_slots_stream(vc, compact)` re-derives the host's channel layout:
    `VC1_FABRIC_POSITION_START - 1`, `my_direction ± 1`, and a hard-coded `sender_channel_1_...` for 1D.
  - `local_downstream_credit_stream(vc, compact)` branches `vc == 0 ? ... : ...` over two hand lists.
  - Both are downstream facts: they belong in `DownstreamTraits<VC>` (router_downstream.hpp). Better still, the host
    names the downstream's channel, so the kernel doesn't reconstruct the layout.
- **VC presence in two places.** `has_vc1` is here but `has_vc2` is in router_init.hpp. Keep the VC presence facts
  together; the roles move here under C2.
- **A grab-bag.** The file mixes three things: router facts (arch, VCs, roles, stream ids), generic utilities
  (Present / Each / for_each_index / popcount / bit helpers), and the placement primitives. Optional: split the
  utilities and placement primitives into a router-independent header, so router_common.hpp holds facts only.
- **Unnamed heartbeat values.** The interval (every 64 rounds, `0x3F` in router_round.hpp) and UMD's signature
  (`0xDCBA0000`) should become named constants (UPPER_SNAKE).

### A2. router_init.hpp (402 lines)
- **Stream init by number** (the largest positional item left).
  - `init_link_streams` uses `to_receiver_packets_sent_streams[0]`, `to_sender_packets_acked_streams[0..1]`,
    `sender_free_slots_streams[0..1]` with `SENDER_NUM_BUFFERS_ARRAY[0..1]`.
  - `init_2d_streams` uses `[1]`, `[2]`, acked `[2..4]`, and positions `i + 2` for `i < 8`.
  - Proposed: each channel owner names its streams and whether the far end can touch them, which is the reason for
    "before the first sync". Init walks the serviced channels' owners, with no channel numbers in init.
- **Counter-block zeroing written three times** in `init_txqs`, as loops of reinterpret_casts over
  `local_receiver_ack/completion_counters` and `to_sender_remote_ack/completion_counters`. Make it one helper per
  block, owned by the flow-control/credit owner.
- **Duplicated casts.** Each of these is cast again in several files; give each one owner in router_common.hpp:
  - the termination signal: router_round.hpp has `termination_signal()`; router_init.hpp casts it three more times;
  - the routing table: router_routing.hpp has `l1_routing_table()`; router_init.hpp and router_single.hpp cast it
    again;
  - the telemetry block: router_telemetry.hpp has `telemetry_block()`; router_init.hpp casts it twice.
- **`run_handshakes` is `__attribute__((optimize("Os")))`.** That is a compiler attribute outside the allowed list
  (volatile, keep_in_memory, read groups, order_point, FORCE_INLINE / noinline). Either keep it as code space for a
  once-run path (main does the same) and add it to the allowed list, or drop it. **User's call.**
- **`TeardownTrids` restates the transaction-id tracker type** for teardown. Check that it is the receiver's own
  tracker type, and use the receiver's alias if so.

### A3. fabric_erisc_router_v4.cpp (286 lines)
- **Role setup is inline in kernel_main.** C2 / Q6 already moves `wait_for_upstream_routers` and
  `wait_for_connect_request`. Proposed extension: move all role setup into the roles, so kernel_main stays the
  phase skeleton.
  - Sender: zero the connection semaphores and producer cursors, `init_worker_interfaces`, and the
    SenderChannelBuffers init go into `sender::` init / connect.
  - Receiver: `init_downstream_connections`, the ReceiverChannelBuffers init and the downstream opens go into
    `receiver::` init / open.
- **Header comment.** "a worker on this chip (channel 0 of a VC)" is positional wording, and wrong after C1. The long
  "Bring-up status" paragraph would read better as a list of supported configurations next to the `brought_up` gate.
- **`wait_for_connect_request`'s 100000 polls** should be a named constant.
- **Two inits named alike.** `init_telemetry()` (the FabricTelemetry block) and `telemetry::init()` (the counters)
  sit next to each other. Rename one, or make one call the other.

### A4. router_round.hpp (190 lines)
- **Sender state in a shared struct.** `RoundCounters` holds two sender-only fields (`send_yielded`,
  `sent_at_round_start`) in a struct both loops use; they are the sender's round state. The `send_yielded` misnaming
  is already in the C plan.
- **Misplaced comment.** The comment above `RoundCounters` ("The heartbeat, then the context switch: ...")
  describes `end_round`.
- **Study pointer:** "RESULTS §97" in the header comment.

### A5. router_local_delivery.hpp (195 lines)
- **Command-buffer ownership.** `local_write_state_preset` and `local_write_cmd_buf_is_exclusive` go to
  CommandBuffers (C2, planned).
- **Repeated literals and constants:**
  - `local_inline_write` passes `0xF`: use `INLINE_WRITE_ALL_BYTES`;
  - `chunk_encoding_mask = 0b11 // main: CHUNK_ENCODING_MASK` duplicates main's constant: use main's.
- **Counter bookkeeping outside the API.** `count_local_writes` updates `noc_nonposted_writes_num_issued/acked`
  directly, mirroring the API's internals. Prefer an API call, or name the dependency in one helper the API owner
  can see.
- **`channel_trimming_usage_recorder.set_noc_send_type_used`** is repeated in every arm. If the recorder is
  idempotent (it is a bitmask), call it once before the dispatch. Check the -O3 images: it should be identity with
  the capture off.
- **Study pointers:** LEDGER E1-8 / E1-10 / E1-17.

### A6. router_telemetry.hpp (193 lines)
- **Verb pairs.** `note_sent` / `count_received` use different verbs for the same kind of call. Pick one pair.
- **Repeated code.** `(bytes + 3) >> 2` appears twice (needs a words helper), and the perf-recorder buffer cast
  twice (needs a helper like `telemetry_block()`).
- **Study pointers:** "V4_TELEMETRY.md", "offline check 2026-10-29".

### A7. router_single.hpp (73 lines)
- **Chip coordinates built twice.** `run` casts the routing table and builds `ChipCoordinates` itself, as
  `receiver::run` does. Proposed: one `ChipCoordinates` factory (e.g. `this_chip()`) in router_routing.hpp, used by
  both.
- **Study pointer:** "V4_SINGLE_ERISC.md".

### A8. router_routing.hpp (174 lines)
Clean. Only the routing-table cast should go to the shared owner (A2).

### A9. Comments, all router files
Study pointers in comments, about 32 across 10 files:

| File | Count |
|---|---:|
| router_sender.hpp | 10 |
| router_receiver.hpp | 7 |
| router_downstream.hpp | 4 |
| router_common.hpp | 3 |
| router_local_delivery.hpp | 2 |
| fabric_router_flow_control.hpp | 2 |
| router_telemetry.hpp, router_single.hpp, router_round.hpp, router_init.hpp | 1 each |

C4 covers comments in the sender. Extend it to every router file: each pointer becomes a one-line why that stands
on its own upstream, where fabric_exp/ won't exist.

## B. Push scope: settled by the user (2026-11-03)

The push is a **code-shape preview only**: just the router code (router/*.hpp and the kernel file) on a branch, for
the team to read. Nobody will build or run it. So none of the following is needed:

- rebasing the shared library headers;
- a curated host patch;
- the kernel slot;
- defaults;
- a build check.

The drift noted before still matters only if a runnable branch is ever wanted:

| File | Lines differing from main |
|---|---:|
| fabric_erisc_datamover_channels.hpp | 114 |
| fabric_edm_packet_transmission.hpp | 298 |
| fabric_router_flow_control.hpp | 40 |
| fabric_erisc_router_ct_args.hpp | 100 |

Plus study switches in the host builder, and the v2 kernel slot.
