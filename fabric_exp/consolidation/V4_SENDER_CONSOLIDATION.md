# v4 sender consolidation: every fact in its owner (design note, 2026-11-03)

The user reviewed the final sender build `v4txo8` and found its facts and helpers spread around instead of owned. This
note proposes the owners, the shape and the stages. It is a proposal only: no code, no builds, no device jobs. The
user reviews it before anything is built.

What it is based on:
- **The sender build `v4txo8`** (variants/v4txo8): router_sender.hpp (964 lines, cited `L:`), router_single.hpp,
  router_common.hpp, router_round.hpp, router_telemetry.hpp, fabric_erisc_router_v4.cpp (cited `K:`),
  fabric_erisc_datamover_channels.hpp, fabric_erisc_router_ct_args.hpp (cited `ct:`).
- **Its history:** V4_SENDER_OWNERSHIP.md (§1.0 names, §3 structure, the user's decisions, §12 stages X-1..X8 and the
  final round).
- **The receiver pattern:** v4rxg3's router_receiver.hpp (`ReceiverChannel<VC>`, `ReceiverChannelStates` owning the
  placement rule; V4_RECEIVER_OWNERSHIP.md §18) and v4rxnsf2's router_downstream.hpp (`ForwardCommands<VC>`, three
  forms chosen in one place; V4_NS_PRESETS.md §9).
- **Command-buffer audits:** V4_BOUNDS.md §4 (per-ERISC buffer map, F7) and V4_NS_PRESETS.md §2.

Rules applied: host vocabulary; `<Kind>Channel` / `<Kind>ChannelState` / `<Kind>ChannelProgress`; UPPER_SNAKE for
tuning and host constants, lower_snake for derived traits; no one-off structs; facts come from their owners; features
per channel, not channel numbers; readability is not traded for perf (memory feedback-readability-anchor).

None of the sketches below was compiled. Where a C++ spelling might trip GCC (in-class constant expressions over
the trimmed ct arrays; L:52-53 records one such case), the stage's compile settles the spelling. The content stays
the same.

## 0. Summary

**The six review points, answered:**
1. **Loose facts.** Every loose fact goes to one owner:
   - per channel: `SenderChannel<CH>`;
   - per VC: `SenderVc<VC>`;
   - router-wide masks: `SenderVcs`;
   - command buffers: `CommandBuffers` in router_common.hpp, shared with the receiver;
   - the producer link: the `WorkerChannel<CH>` and `ForwardingChannel<CH>` kinds;
   - the library interfaces: `SenderChannelInterfaces`.

   `SENDER_TXQ` goes; the host's `sender_txq_id` is used directly, as the receiver uses `receiver_txq_id`.
2. **`worker_credit_cmd_bufs_distinct`.** Yes, it is the same kind of fact as the receiver's
   `forward_cmd_bufs_consistent`: command-buffer ownership. There is a third one, `local_write_cmd_buf_is_exclusive`.
   The preset conditions disagree too (V4_BOUNDS F7: the worker credit's lacks the DPRINT/watcher guard). One
   `CommandBuffers` table replaces all three checks. It reads the host's ids, records which kinds of command each buffer
   carries on this ERISC, and states one rule.
3. **`worker_channel_positions`.** It holds the host's worker-role positions in a VC, serviced or not. Three things
   use it:
   - `worker_channels` (positions ∩ serviced);
   - `reads_free_slots_at_pass_start` (any position: the lead's grouping, kept since X1, even where the worker is
     trimmed);
   - the one-worker static_assert.

   It becomes `SenderVc`'s private `worker_positions`, computed from each channel's own VC.
4. **`forwarding_index`.** It is dead code. Its only user, `UpstreamCredits`, went in v4txo3a (X3a); init now reads the
   interface's cached address (L:819). It is deleted.
5. **`first_sender_channel_by_vc`.** A VC's channels are found by each channel's VC (`vc_of_sender_channel`, already
   `SenderChannel<CH>::vc`), and its worker channel by role (`SenderVc<VC>::worker_channel`). Then
   `first_sender_channel_by_vc` and `sender_channel_count_by_vc` have no user, and they leave ct.
6. **The worker interface code.** It is wrapped in three owners:
   - **`WorkerChannel<CH>`**: the connection and the credit. One shared cold `open_or_close_worker_connection` sits
     beside it, as ForwardCommands' forms share `forward_stateless`.
   - **`ForwardingChannel<CH>`**: the upstream credit and the static connection.
   - **`SenderChannelInterfaces`**: storage, `start()` and `of<CH>()`. It is built only on ERISCs that serve sender
     channels: v4txo8w's change, which falls out of this shape.

**Stages** (§3):

| stage | content | codegen |
| --- | --- | --- |
| C0 | merge v4txo8 + v4rxnsf2 | identity per parent |
| C1 | sender facts into their owners | byte-identical |
| C1b (option) | the pass-start rule from the serviced workers | worker-trimmed images change |
| C2 | `CommandBuffers` | byte-identical on every coverage image |
| C3 | the producer kinds and the interfaces owner, with v4txo8w | loops identical; init and sender-less images change |
| C4 | comments and names | byte-identical |

On coverage images only C3, and C1b if chosen, are expected to change code. C2 also changes DPRINT/watcher builds,
where the worker credit stops presetting (F7).

## 1. Inventory

Every free-floating fact, helper, constexpr and free function in the sender files, plus the command-buffer facts on
the receiver side that the new owner absorbs. "Kept" items are listed in §1.6.

### 1.1 router_sender.hpp: channels and their features (L:47-180)

| item | L: | what it is for | users | proposed owner | why |
| --- | --- | --- | --- | --- | --- |
| `SENDER_TXQ` | 47 | the TX queue the sender sends on (= ct `sender_txq_id`, host SENDER_TXQ_ID) | `send` (4 uses) | none: `sender_txq_id` used directly | a second name for one host value; the receiver uses `receiver_txq_id` directly |
| `WORKER_CREDIT_NOC` | 49 | the NoC of producer credits and connection handshakes (= library `worker_handshake_noc`) | `preset_worker_credit`, `credit_worker`, `credit_upstream`, the `WorkerConnection` and `WorkerInterfaces` types | `SenderChannel<CH>::producer_noc`; the interface types name the library constant (C3, with its users) | a channel's producer-link fact, as the receiver's `DownstreamTraits<VC>::noc` |
| `remote_free_slots_to_send_by(ch)` | 54 | bubble flow control: 2 free remote slots for an injection channel on a deadlock-avoiding router, else 1 | `SenderChannel::remote_free_slots_to_send` only | inline in `SenderChannel<CH>` (reads the `_all_` array, as today) | one user, a channel fact |
| `worker_channel_positions(vc)` | 61 | the host's worker-role positions in VC, serviced or not | `SenderVc::worker_channels`, `::reads_free_slots_at_pass_start`, the one-worker assert | `SenderVc<VC>` (private `worker_positions`) | a VC fact with one user |
| `worker_read_counter_addr(ch)` | 72 | a channel's read counter in its connection info (a worker's credits, the source word of each credit write) | `SenderChannel::read_counter_addr`; K:139 init (every position, serviced or not) | `SenderChannelInterfaces::read_counter_addr(ch)` (C3) | the connection info is the interface's; both users read it there |
| `channels_in(first, count)` | 77 | the serviced channels in a positional range | `SenderVc::sender_channels` | deleted: `SenderVc` collects the serviced channels whose VC is VC | point 5: each channel's VC, not a range |
| `SenderChannel<CH>` | 86 | a sender channel's facts | everywhere | kept; worker-only members move to `WorkerChannel<CH>` (`connection_semaphore_addr`, `read_counter_addr`, `producer_credit_cmd_buf`, the live-check assert) | a forwarding channel has no connection semaphore, read counter or credit buffer |
| `SenderVc<VC>::first_level_ack` | 125 | the VC's first-level acks | nobody (every user reads `SenderChannel<CH>::first_level_ack`) | deleted | dead; the channel reads the host's per-VC flag |
| `SenderVc<VC>` | 119 | a VC's features | visits, the masks | kept, extended: `worker_positions`, `has_worker_channel`, `worker_channel` | the VC owns which of its channels is the worker |
| `static_assert(MAX_NUM_VCS == 3)` | 132 | the three-term unions below | - | `SenderVcs` | it guards those terms |
| `sender_channels`, `worker_channels`, `forwarding_channels`, `sender_vcs`, `channels_read_at_pass_start` | 133-144 | router-wide masks (unions over the VCs) | `SenderChannelStates`, `pass`, `init_forwarding`, `total_sent`, `any_channel_holds_packet`, K:163, router_single | `SenderVcs` (`sender_vcs` becomes `SenderVcs::serviced`) | one router-wide owner, as `SenderVc` owns `reads_free_slots_at_pass_start` |
| `worker_credit_cmd_bufs_distinct()` + assert | 145-159 | each serviced worker channel's credits on a buffer of its own, none the atomics buffer | its static_assert | `CommandBuffers` (C2) | point 2 |
| `static_assert(!tensix_mux_mode \|\| forwarding_channels == 0)` | 162 | mux mode: no router forwards into this one | - | `SenderVcs` | a router-wide fact about the masks |
| `forwarding_index<CH>()` | 165 | was: a forwarding channel's dense index into `UpstreamCredits::addr` | **none since v4txo3a** | deleted | dead (point 4) |
| `RemoteReceiverChannel<VC>::rx` | 174 | the host's index of VC's receiver channel (= ct `receiver_channel_by_vc[VC]`) | the remote channel's arrays | kept | the same fact has a second owner on the receiver, `receiver_channel_of(vc)` (Q5) |

### 1.2 router_sender.hpp: state, interfaces, registers (L:186-337)

| item | L: | what it is for | users | proposed owner | why |
| --- | --- | --- | --- | --- | --- |
| `SenderVcState<VC>::worker` typed `SenderChannelState<first_sender_channel_by_vc[VC]>` | 258 | the VC's worker channel state | visits, connection checks, presets | `SenderChannelState<SenderVc<VC>::worker_channel>`, present iff `has_worker_channel` | point 5: by role, not position. The same type, so the layout is the same |
| `WorkerInterfaces` | 265 | the library tuple of every sender channel's interface | storage, K:119/124/156 | `SenderChannelInterfaces::All` | part of the interfaces' owner |
| `WorkerInterfaceStorage`, `worker_interface_storage` | 267-270 | aligned bytes for the tuple (no global constructors) | `start_worker_interfaces`, `worker_interface` | `SenderChannelInterfaces` (private storage) | a one-off struct wrapping one array |
| `start_worker_interfaces()` | 272 | builds the tuple in place | K:195 (every ERISC) | `SenderChannelInterfaces::start(...)`, merged with K's `init_worker_interfaces`, called only on sender ERISCs | the owner builds itself; v4txo8w falls out |
| `worker_interface<CH>()` | 276 | channel CH's interface | credit, preset, connection event, `init_forwarding` | `SenderChannelInterfaces::of<CH>()` | part of the owner |
| `holds_packet<CH>` | 289 | a channel's free-slots register says it holds a packet (anything but its slot count) | `channels_holding_packets` | `SenderChannel<CH>::holds_packet` | a channel fact (its slot count), as `ReceiverChannel<VC>::slot_addr` |
| `read_connection_semaphores<WORKERS>` | 317 | the pass's read group: the worker channels' semaphores | `channels_with_packets` | kept; reads `WorkerChannel<CH>::connection_semaphore_addr` | pass mechanics |

### 1.3 router_sender.hpp: credits and connections (L:422-559, 859-887)

| item | L: | what it is for | users | proposed owner | why |
| --- | --- | --- | --- | --- | --- |
| `worker_credit_preset` | 436 | presets on two ERISCs (= `command_buffer_presets`) | preset, credit, restore | deleted: `CommandBuffers::presets` | one preset condition for every use; it gains the DPRINT/watcher guard (F7) |
| `static_assert(!enable_read_counter_update_noc_flush)` | 437 | the credit write is unflushed (a later, larger count may overwrite the word) | - | `WorkerChannel<CH>` | the worker credit's rule |
| `preset_worker_credit<CH>` (noinline) | 442 | programs the channel's credit buffer for its connected worker | connection event, restores | `WorkerChannel<CH>::preset_credit` | the worker channel kind |
| `credit_worker<CH>` (noinline) | 457 | absolute count into the read counter, and into the worker's L1 while connected | `return_slots`, connection event | `WorkerChannel<CH>::credit` | the worker channel kind |
| `credit_upstream` | 488 | +n on the upstream router's free-slots register | `return_slots` | `ForwardingChannel<CH>::credit` | the forwarding channel kind |
| `WorkerConnection` | 494 | the library's slot-count-independent connection type | `worker_connection_event` | the shared function's parameter type, spelled out | an alias with one user |
| `worker_connection_event` (noinline, one copy) | 500 | a changed connection semaphore: open (cache the credit address) or close (write the credits, signal teardown) | `connection_event<CH>` | `open_or_close_worker_connection`, beside `WorkerChannel` | shared by every worker channel (X5b); the receiver's forms share `forward_stateless` the same way |
| `connection_event<CH>` (noinline) | 521 | the shared event, then the opening preset and credit | `check_connection` | `WorkerChannel<CH>::connection_event` (private) | the worker channel kind |
| `check_connection<CH>` | 538 | every pass: the semaphore against the connection state | `check_worker_connection<VC>` | `WorkerChannel<CH>::check_connection` | the worker channel kind |
| `return_slots<CH>` | 551 | slots back to the producer, by kind | visits | kept; calls `WorkerChannel<CH>::credit` or `ForwardingChannel<CH>::credit` | a visit step, by feature |
| `restore_worker_credit_preset(s)` | 861-876 | presets again after the base firmware ran | run, single::run, after_round | kept (per VC); calls `WorkerChannel<worker_channel>::preset_credit` | the worker state's place is per VC (loop copy or memory) |
| `check_worker_connection<VC>` | 880 | every pass, per VC | `pass` | kept (per VC); calls `WorkerChannel<worker_channel>::check_connection` | as above |
| `init_forwarding<MASK>` | 815 | each forwarding state's upstream address from its interface | run, single::run (both name the mask) | `start_forwarding_channels(states)`, reading `ForwardingChannel<CH>::upstream_free_slots_addr()` | callers stop naming the mask |

### 1.4 Outside router_sender.hpp

| file | item | what it is for | proposed | why |
| --- | --- | --- | --- | --- |
| K:119 | `using sender::WorkerInterfaces` | init's interface type | deleted | the owner's |
| K:124-141 | `init_worker_interfaces` (noinline) | builds each interface (connection info, semaphore, starting credits, credit source word) | body moves into `SenderChannelInterfaces::start` | the interfaces' construction is the owner's |
| K:144-166 | `wait_for_connect_request`, `wait_for_upstream_routers` | init: each static upstream connection | into router_sender.hpp's init block, iterating `ForwardingChannel` (Q6) | the receiver's opens live in router_downstream.hpp's init block |
| K:163 | `static_assert(... forwarding_channels ...)` | a static upstream connection is a forwarding channel's | `SenderVcs` | a router-wide fact about the masks |
| K:195 | `sender::start_worker_interfaces()` on every ERISC | the tuple's default construction | gone (the sender's `start` builds it) | v4txo8w: 336 B .bss and +188 B code on bh_vc1's E1 at -O3 (V4_SIZE.md 2026-11-03) |
| K:7 | header "a worker on this chip (channel 0 of a VC)" | - | "a VC's worker channel" | positional wording (rev 4 §1 already listed it) |
| ct:229-231 | `first_sender_channel_by_vc`, `sender_channel_count_by_vc` | a VC's channel range | deleted after C1 | no users |
| ct:225-226 | the `is_vc_crossover_to_vc1` comment | - | back above its array (X2 put the new arrays between them) | misplaced comment |
| ct:232 | `receiver_channel_by_vc` | VC → the host's receiver channel | kept (`RemoteReceiverChannel::rx`) | Q5 |
| router_common:105-125 | `sender_connection_info_addrs`, `sender_connection_semaphore_addrs` under the "Downstream routers" header | - | their own section header | misplaced header |
| router_common:306 | `command_buffer_presets` | presets on two ERISCs | `CommandBuffers::presets` | one owner |
| router_init:18-19 | `is_sender_role`, `is_receiver_role` | this ERISC's roles | router_common.hpp | `CommandBuffers` needs them; both programs use them |
| router_round:106 | `end_round(..., bool send_yielded)` | receives "the round attempted a send" (sent, or yielded); X0's rename gave it the field's name | `send_attempted` (C4) | a wrong name |
| router_round:100-101 | `RoundCounters::send_yielded`, `sent_at_round_start` | sender fields in the shared round record | kept | used by `before_round` / `after_round`; shared object, documented |
| library | `EdmChannelWorkerConnection::sender_sync_noc_cmd_buf` | stores the credit buffer id; no kernel reads it (other kernels pass 0 "unused") | left | library; host-cleanup list (rev 4 Q6) |

### 1.5 Command-buffer facts on the receiver side (v4rxnsf2)

| file:line | item | what it is for | proposed |
| --- | --- | --- | --- |
| downstream:56-57 | `DownstreamTraits<VC>::data_cmd_buf`, `sync_cmd_buf` | VC's forward buffers (host RX_CH_<n>_FWD_DATA / FWD_SYNC_CMD_BUF_ID) | kept as traits, read from `CommandBuffers::forward_data[VC]` / `forward_credits[VC]` |
| downstream:102 | `vc_forwards_to_routers(vc)` | VC0, and VC1 on VC1 routers | router_common.hpp (`CommandBuffers` needs it) |
| downstream:106-117 | `forward_cmd_bufs_consistent<VC>()` | data and credit buffers differ (across VCs too), neither the local-write or atomics buffer | `CommandBuffers` rule |
| downstream:122-128 | `forwards_can_preset` | the forwards' preset condition, with the consistency term | `CommandBuffers::forward_presets` (Q2 on the consistency term) |
| downstream:131-139 | `uses_vc0_forward_cmd_bufs<VC>()` | VC's forwards on VC0's two buffers | `CommandBuffers::forwards_on_vc0_bufs(vc)` (used by `can_share_vc0_destination` and PartialPreset's assert) |
| local_delivery:26 | `LOCAL_WRITE_CMD_BUF` | the local-write buffer | `CommandBuffers::local_writes` (10 use sites; pure rename) |
| local_delivery:29-33 | `local_write_state_preset` | local-write preset condition | `CommandBuffers::presets` |
| local_delivery:36-48 | `local_write_cmd_buf_is_exclusive` + assert | the local-write buffer is no forward's and not the atomics buffer | `CommandBuffers` rule |

### 1.6 Kept where they are, and why

- **The pass mechanics stay free templates:** `read_free_slots_registers`, `channels_holding_packets`,
  `channels_with_packets`, `send<CH>`, `visit_worker`, `visit_channel_with_work`, `visit_forwarding_channels`,
  `visit_channels`, `visit_vc`, `forwarded`, `sent_by_worker`, `total_sent`, `any_channel_holds_packet`, `pass`,
  `before_round`, `after_round` and `run`. They are the pass's steps, as the receiver's `receive` / `handle` / `step` /
  `step_channel` / `pass`. They recurse over their owners' masks.
- **The state types stay:**
  - `RemoteReceiverChannelProgress<VC>`, `SenderChannelProgress` and `SenderChannelState<CH>`;
  - `SenderChannelStates`, with `loop_local_vc` and the placement accessors.

  `SenderChannelState<CH>` keeps its producer-link member present by feature: `connection_state` or
  `upstream_free_slots_addr`. **No `WorkerChannelState` / `ForwardingChannelState`.** A split would rename types that
  `SenderVcState` and `Each<forwarding, ...>` hold. It would risk layout for no reader gain, since the feature already
  decides the member.

## 2. Proposed shape

### 2.1 Channel and VC facts: `SenderChannel<CH>`, `SenderVc<VC>`, `SenderVcs` (C1)

```cpp
// Sender channel CH: its slots, registers and counters, and its features (the host's, per channel or per VC).
template <size_t CH>
struct SenderChannel {
    static constexpr size_t vc = vc_of_sender_channel(CH);  // the host's layout (ct args)
    // A worker channel takes a worker on this chip (the host's role, SENDER_CH_<i>_IS_WORKER_CHANNEL); the others are
    // forwarding channels, which the routers on this chip facing other directions write into.
    static constexpr bool worker = is_sender_channel_worker[CH];
    static constexpr uint32_t num_slots = SENDER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot_addr = channel_allocs::Entry<SENDER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last_slot_addr = first_slot_addr + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t free_slots_stream = sender_free_slots_stream<CH>;
    // Counters the far router's receiver writes over the link.
    static constexpr uint32_t completion_counter_addr = ...;       // as today
    static constexpr uint32_t first_level_ack_counter_addr = ...;  // as today
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[vc];
    // Bubble flow control: on a deadlock-avoiding ring or torus an injection channel (the worker channel; on a torus a
    // channel fed from the other axis) sends only with two free remote slots. (The untrimmed array: L:52-53.)
    static constexpr uint32_t remote_free_slots_to_send =
        enable_deadlock_avoidance && sender_channel_is_traffic_injection_channel_all_[CH]
            ? BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS
            : 1;
    // The NoC of its producer's credits and connection handshake (the library's resolution of SENDER_CH_<i>_ACK_NOC_ID).
    // [C3, with the credit code that reads it; until then WORKER_CREDIT_NOC stays.]
    static constexpr uint8_t producer_noc = tt::tt_fabric::worker_handshake_noc;

    // Whether its free-slots register (raw) says it holds a packet: anything but its slot count. An over-fill must read
    // as holding packets, or the channel is never served (LEDGER E0-10, RESULTS §93).
    static FORCE_INLINE bool holds_packet(uint32_t free_slots_register) {
        return (free_slots_register & stream_count_mask) != num_slots;
    }

    static_assert(num_slots > 0 && num_slots <= stream_count_mask);
    static_assert(sender_channel_uses_counter_credits(CH), "the sender reads its credits from L1 counters");
};

// VC's sender features, from its channels' (each channel's VC and role are the host's): its serviced channels, its
// worker channel and its forwarding channels, and where its free-slots registers are read.
template <size_t VC>
struct SenderVc {
private:
    // The host's worker-role positions among VC's channels, serviced or not.
    static constexpr uint32_t worker_positions = [] {
        uint32_t mask = 0;
        for (size_t ch = 0; ch < NUM_SENDER_CHANNELS; ch++) {
            mask |= (vc_of_sender_channel(ch) == VC && is_sender_channel_worker[ch]) ? 1u << ch : 0u;
        }
        return mask;
    }();
    static_assert(popcount(worker_positions) <= 1, "a VC takes at most one worker");

public:
    static constexpr uint32_t sender_channels = [] {
        uint32_t mask = 0;
        for (size_t ch = 0; ch < NUM_SENDER_CHANNELS; ch++) {
            mask |= (vc_of_sender_channel(ch) == VC && is_sender_channel_serviced[ch]) ? 1u << ch : 0u;
        }
        return mask;
    }();
    static constexpr uint32_t worker_channels = sender_channels & worker_positions;
    static constexpr uint32_t forwarding_channels = sender_channels & ~worker_channels;
    // VC's worker channel, where this router serves one (named only under has_worker_channel).
    static constexpr bool has_worker_channel = worker_channels != 0;
    static constexpr size_t worker_channel = has_worker_channel ? lowest_bit(worker_channels) : 0;
    // Its free-slots registers are read in the pass's first read group when VC has a worker position (with the worker
    // channel's connection semaphore, where this router serves it); otherwise right before VC's own visits, so a packet
    // landing during the earlier VCs' visits does not wait a pass (V4_SENDER_SHAPE.md). [C1b: from the served worker.]
    static constexpr bool reads_free_slots_at_pass_start = worker_positions != 0;
};

// Every VC this router sends on, together: one bit per channel (per VC for `serviced`).
struct SenderVcs {
    static_assert(MAX_NUM_VCS == 3, "the sender's features and its pass name VCs 0, 1 and 2");
    static constexpr uint32_t serviced =  // the VCs with a serviced sender channel (was sender_vcs)
        (SenderVc<0>::sender_channels != 0 ? 1u : 0u) | (SenderVc<1>::sender_channels != 0 ? 2u : 0u) |
        (SenderVc<2>::sender_channels != 0 ? 4u : 0u);
    static constexpr uint32_t sender_channels = ...;               // the union, as today's three-term ORs
    static constexpr uint32_t worker_channels = ...;
    static constexpr uint32_t forwarding_channels = sender_channels & ~worker_channels;
    static constexpr uint32_t channels_read_at_pass_start = ...;   // as today
    // In Tensix mux mode the mux is the only producer: no router forwards into this one.
    static_assert(!tensix_mux_mode || forwarding_channels == 0);
    // Every channel the host has wait for a static upstream connection is a forwarding channel (was K:163).
    static_assert(...);
};
```

Notes:
- **Equivalence.** `vc_of_sender_channel(ch) == VC` over `ch < NUM_SENDER_CHANNELS` equals today's ranges
  `[first_sender_channel_by_vc[VC], + count)` whenever the host sets serviced and worker flags only on real channels.
  The C1 byte compare is the check. If it ever fails on a config, the loop also bounds by the VC's count.
- **The `0` in `worker_channel`.** It is never named: `SenderVcState` uses `Present<has_worker_channel, ...>`, which
  names the type without instantiating it.
- **Readers.** `pass`, `visit_vc` and the per-VC helpers already read `SenderVc<VC>::...`. They swap
  `lowest_bit(SenderVc<VC>::worker_channels)` (three sites) for `SenderVc<VC>::worker_channel`.
- **State.** `SenderChannelStates` holds `Each<SenderVcs::serviced, SenderVcState>` and
  `Each<SenderVcs::forwarding_channels, SenderChannelState>`.

### 2.2 The worker channel by role (C1)

```cpp
template <size_t VC>
struct SenderVcState {
    RemoteReceiverChannelProgress<VC> remote = RemoteReceiverChannelProgress<VC>::start();
    [[no_unique_address]] Present<SenderVc<VC>::has_worker_channel, SenderChannelState<SenderVc<VC>::worker_channel>, 2>
        worker{};
};
```

Where a VC has a worker channel, its index is the one `first_sender_channel_by_vc[VC]` gave (the host's rule puts the
worker first), so the type is the same and so is the layout. Where it has none, the member is `Absent<2>` as today.

### 2.3 One command-buffer owner for both ERISCs: `CommandBuffers` (C2, router_common.hpp)

The ids are the host's (`SENDER_CH_<i>_ACK_CMD_BUF_ID`, `RX_CH_<n>_FWD_DATA_ / FWD_SYNC_ / LOCAL_WRITE_CMD_BUF_ID`). The
owner names each kind of command's buffer and which kinds keep presets, builds one table of what each buffer carries
on this ERISC, and states one rule. Users read their buffer here; their traits keep naming it
(`WorkerChannel<CH>::credit_cmd_buf`, `DownstreamTraits<VC>::data_cmd_buf`).

```cpp
// This ERISC's roles: on two ERISCs one each, on one ERISC both (from router_init.hpp).
constexpr bool is_sender_role = any_sender_channel_serviced;
constexpr bool is_receiver_role = any_receiver_channel_serviced;
// The VCs whose forwards go to downstream routers here: the host lists downstream routers for VC0 and VC1 (from
// router_downstream.hpp).
constexpr bool vc_forwards_to_routers(size_t vc) { return vc == 0 || (vc == 1 && has_vc1); }

// Which NoC command buffer each kind of command uses on this ERISC (the host's ids), which kinds keep their buffer
// programmed between commands (a preset, through the stateful NoC API), and the rule they follow. A buffer that a kind
// keeps preset carries that kind only: any other command on it rewrites the preset (VC2's credit on VC0's credit
// preset, job 142; an inline write on the local-write preset, job 351). The atomics buffer carries no preset and no
// write from L1: atomics need the return address the firmware keeps there (job 366); upstream-router credits,
// stateless forward credits and teardowns, whole inline-register writes and increments, go on it too. Each ERISC's
// buffers are its own: two ERISCs issue on different NoCs (V4_BOUNDS.md §4). The table holds the router's own
// commands; an API that ignores its buffer argument is outside it (Blackhole's emulated L1 inline write always issues on
// write_cmd_buf, job 142).
struct CommandBuffers {
    // The host's buffer of each kind.
    static constexpr uint8_t atomics = write_at_cmd_buf;
    static constexpr uint8_t local_writes = tt::tt_fabric::local_chip_data_cmd_buf;              // receiver
    static constexpr const auto& worker_credits = sender_channel_ack_cmd_buf_ids;               // sender, by channel
    static constexpr const auto& forward_data = receiver_channel_forwarding_data_cmd_buf_ids;   // receiver, by VC
    static constexpr const auto& forward_credits = receiver_channel_forwarding_sync_cmd_buf_ids;

    // Which kinds keep presets: none on one ERISC (both roles' commands share its buffers) or with DPRINT or the
    // watcher (their traffic shares them); forwards also not where every path uses one NoC, or in UDM mode (the relay's
    // stateless forwards share the data buffer).
    static constexpr bool presets =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
        NUM_ACTIVE_ERISCS > 1;
#else
        false;
#endif
    static constexpr bool forward_presets = presets && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode;

    // The kinds of command, one bit each; each worker channel's credits are a kind of their own (WORKER_CREDITS << CH).
    enum Kind : uint32_t {
        ATOMICS = 1u << 0, LOCAL_WRITES = 1u << 1, FORWARD_DATA = 1u << 2, FORWARD_CREDITS = 1u << 3,
        WORKER_CREDITS = 1u << 4,
    };

    // The kinds each of the four buffers (WR, RD, WR_REG, AT) carries on this ERISC. The forwards of every VC that
    // forwards to routers are one kind: VCs on the same buffers share their presets.
    static constexpr std::array<uint32_t, 4> kinds_on = [] {
        std::array<uint32_t, 4> kinds{};
        kinds[atomics] |= ATOMICS;
        for (size_t ch = 0; ch < NUM_SENDER_CHANNELS && is_sender_role; ch++) {
            kinds[worker_credits[ch]] |= is_sender_channel_worker[ch] ? WORKER_CREDITS << ch : 0u;
        }
        if (is_receiver_role) {
            kinds[local_writes] |= LOCAL_WRITES;
            for (size_t vc = 0; vc < MAX_NUM_VCS; vc++) {
                if (vc_forwards_to_routers(vc)) {
                    kinds[forward_data[vc]] |= FORWARD_DATA;
                    kinds[forward_credits[vc]] |= forward_presets ? FORWARD_CREDITS : 0u;  // stateless: on atomics
                }
            }
        }
        return kinds;
    }();
    // The kinds kept preset here.
    static constexpr uint32_t preset_kinds =
        presets ? ~(ATOMICS | (forward_presets ? 0u : FORWARD_DATA | FORWARD_CREDITS)) : 0u;

    // Whether VC's forwards use VC0's two buffers (true where VC forwards to no router here).
    static constexpr bool forwards_on_vc0_bufs(size_t vc) {
        return !vc_forwards_to_routers(vc) ||
               (forward_data[vc] == forward_data[0] && forward_credits[vc] == forward_credits[0]);
    }
};
static_assert(
    [] {
        for (uint32_t kinds : CommandBuffers::kinds_on) {
            if ((kinds & CommandBuffers::preset_kinds) != 0 && popcount(kinds) != 1) {
                return false;
            }
        }
        return true;
    }(),
    "a command buffer kept preset carries one kind of command (the host's *_CMD_BUF_ID)");
static_assert(CommandBuffers::kinds_on[CommandBuffers::atomics] == CommandBuffers::ATOMICS,
              "only atomics, inline register writes and teardowns go on the atomics buffer");
```

What the table gives on today's configs (V4_BOUNDS §4, V4_NS_PRESETS §2):

| ERISC | WR (0) | RD (1) | WR_REG (2) | AT (3) |
| --- | --- | --- | --- | --- |
| two-ERISC E0 | VC0's worker credits | - | VC2's worker credits | atomics |
| two-ERISC E1 | local writes | forward credits (preset forwards) | forward data | atomics |
| one ERISC (nothing preset) | worker credits (VC0), local writes, forward data | - | worker credits (VC2) | atomics |

**What it replaces, and how the old facts map:**
- **`worker_credit_cmd_bufs_distinct`.** On a sender ERISC with presets, the first assert makes each worker
  channel's buffer its own and keeps it off the atomics buffer. The second assert keeps worker credits off the atomics
  buffer on every ERISC. Today's distinctness check on one ERISC, where nothing is preset, is dropped: there is no
  preset there to protect.
  - The new check runs over worker *positions*, not serviced workers. That is stricter and true: the host assigns by
    channel identity, whatever the trimming (rev 4 decision 1).
- **`forward_cmd_bufs_consistent<0>() && <1>()`.** The first assert, under `forward_presets`: the data and credit
  buffers are disjoint across VCs, and neither is the local-write or atomics buffer. That is the same condition.
- **`local_write_cmd_buf_is_exclusive`.** The first assert, under `presets`. It no longer checks VC2's unused
  forward ids (VC2 does not forward). It no longer checks forward credit ids where forwards are stateless (their
  credits go on the atomics buffer).
- **The preset conditions.** `command_buffer_presets`, `local_write_state_preset`, `forwards_can_preset` and
  `worker_credit_preset` become `CommandBuffers::presets` and `::forward_presets`. The worker credit gains the
  DPRINT/watcher guard (F7).

**Consumers after C2:**
- `ForwardCommands<VC>` selects on `CommandBuffers::forward_presets`;
- `DownstreamTraits<VC>` reads `forward_data[VC]` / `forward_credits[VC]`;
- `can_share_vc0_destination` and PartialPreset's assert read `forwards_on_vc0_bufs`;
- `local_write_init` and `local_write` read `presets` and `local_writes`;
- `WorkerChannel<CH>` reads `worker_credits[CH]` and `presets` (C3; in C2 these are still `SenderChannel`'s
  `producer_credit_cmd_buf` and the preset helpers).

**Open choice (Q2).** Today the forwards fall back to stateless when their ids are inconsistent; the other two checks
stop the build. The sketch stops the build for all three, so one rule holds everywhere. No current config differs: the
two-ERISC ids are fixed at data 2, credit 1, local 0, AT 3. The alternative keeps the fallback, with `forward_presets`
also requiring "the forward kinds alone on their buffers"; then the first assert excludes forwards.

**Ahead of the 1D line VCs.**
- A line VC1 worker's credit buffer (RD, LINE_VCS §9 item 4) is one more `WORKER_CREDITS << ch` entry.
- A line VC that forwards on VC0's buffers adds to `FORWARD_DATA` / `FORWARD_CREDITS`; `forwards_on_vc0_bufs` covers
  it.

### 2.4 Producers: `SenderChannelInterfaces`, `WorkerChannel<CH>`, `ForwardingChannel<CH>` (C3)

```cpp
// Every sender channel position's library interface: its producer's connection (connection info, connection semaphore,
// the producer's credit address, cached when it connects). At a fixed location, not in run()'s frame: the out-of-line
// connection path reaches them by channel. Storage and an explicit start: the kernel runs no global constructors.
struct SenderChannelInterfaces {
    using All = decltype(tt::tt_fabric::EdmChannelWorkerInterfaces<tt::tt_fabric::worker_handshake_noc,
                                                                    SENDER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_SENDER_CHANNELS>{}));

    // Channel `ch`'s read counter in its connection info: the credits returned so far, which a worker reads when it
    // connects, and the source word of each credit write to it.
    static constexpr uint32_t read_counter_addr(size_t ch) {
        return sender_connection_info_addrs[ch] + offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, edm_read_counter);
    }

    // Builds every position's interface, serviced or not (once, at init, on an ERISC that serves sender channels): its
    // connection info, its connection semaphore (the runtime args'), its starting credits (0 for a worker channel, whose
    // credits are counts; the slot count for a forwarding channel) and a worker's credit source word (its read
    // counter). Today's start_worker_interfaces() and K's init_worker_interfaces(), in one.
    __attribute__((noinline)) static void start(const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& connection_semaphores);

    template <size_t CH>
    static FORCE_INLINE auto& of() {
        return std::launder(reinterpret_cast<All*>(storage))->template get<CH>();
    }

private:
    alignas(All) static inline unsigned char storage[sizeof(All)];
};

// What a worker channel's connection-semaphore change does, the same for every worker channel (one copy: the
// connection type does not depend on the channel's slot count). Opening caches the worker's credit address; closing
// writes the credits returned so far (`returned`) into the worker's connection info and signals its teardown semaphore.
// Returns the value the semaphore holds while nothing changes. (Today's worker_connection_event, renamed.)
__attribute__((noinline)) uint32_t open_or_close_worker_connection(
    tt::tt_fabric::EdmChannelWorkerConnection<tt::tt_fabric::worker_handshake_noc>& connection,
    uint32_t semaphore, uint32_t expected, uint32_t returned);

// A worker channel: a worker on this chip connects at run time (its connection semaphore, checked every pass) and is
// credited with the absolute count of slots returned. The count goes into the channel's read counter, which therefore
// always holds it; while a worker is connected, a 4-byte posted write copies that word into the worker's L1 on the
// channel's own credit buffer (CommandBuffers). Where presets are kept, that buffer stays programmed for the
// connection (at the connection, and again after the base firmware ran), so a credit only issues the write.
template <size_t CH>
struct WorkerChannel {
    static_assert(SenderChannel<CH>::worker);
    static_assert(!sender_ch_live_check_skip[CH], "the host checks every worker channel's connection");
    // Unflushed: the read counter may be rewritten before an earlier write read it; the worker then sees the newer count.
    static_assert(!enable_read_counter_update_noc_flush);

    static constexpr uint32_t connection_semaphore_addr = sender_connection_semaphore_addrs[CH];
    static constexpr uint32_t read_counter_addr = SenderChannelInterfaces::read_counter_addr(CH);
    static constexpr uint8_t credit_cmd_buf = CommandBuffers::worker_credits[CH];

    static FORCE_INLINE bool connected(const SenderChannelState<CH>& state) {
        return state.connection_state != tt::tt_fabric::connection_interface::unused_connection_value;
    }

    // The connection, from the semaphore the pass's read group read; a change is handled out of line. Returns whether
    // it changed. (Today's check_connection<CH>, verbatim.)
    static FORCE_INLINE bool check_connection(SenderChannelState<CH>& state, uint32_t semaphore);

    // `returned` slots in all: into the read counter and, while connected, into the worker's L1. Out of line: once per
    // completion group, or per ack frame. (Today's credit_worker<CH>, verbatim.)
    __attribute__((noinline)) static void credit(uint32_t returned, bool connected);

    // The credit buffer programmed for the connected worker, where presets are kept. (Today's preset_worker_credit<CH>.)
    __attribute__((noinline)) static void preset_credit();

private:
    static FORCE_INLINE auto& interface() { return SenderChannelInterfaces::of<CH>(); }

    // The shared event, then, where it opened the connection, the credit buffer preset and the credits returned so far
    // sent. (Today's connection_event<CH>.)
    __attribute__((noinline)) static uint32_t connection_event(uint32_t semaphore, uint32_t expected, uint32_t returned);
};

// A forwarding channel: a router on this chip facing another direction writes into it. That upstream router connects
// once at init and is credited +n on its free-slots register.
template <size_t CH>
struct ForwardingChannel {
    static_assert(!SenderChannel<CH>::worker);
    // Whether init waits for its upstream router to connect (SENDER_CH_<i>_WAIT_STATIC_CONNECTION; a channel a trimming
    // override forced on has none).
    static constexpr bool waits_for_upstream_router = sender_ch_wait_static_connection[CH];

    // The upstream router's free-slots register, as the interface cached it when that router connected.
    static FORCE_INLINE uint64_t upstream_free_slots_addr() {
        return SenderChannelInterfaces::of<CH>().cached_worker_semaphore_address;
    }

    // +n on the upstream router's free-slots register. (Today's credit_upstream.)
    static FORCE_INLINE void credit(uint64_t upstream_free_slots_addr, uint32_t n) {
        noc_inline_dw_write<InlineWriteDst::REG, true>(
            upstream_free_slots_addr, pack_value_for_inc_on_write_stream_reg_write(n), INLINE_WRITE_ALL_BYTES,
            SenderChannel<CH>::producer_noc);
    }
};
```

**Users after C3:**
- `return_slots<CH>`: `WorkerChannel<CH>::credit(returned, WorkerChannel<CH>::connected(state))` or
  `ForwardingChannel<CH>::credit(state.upstream_free_slots_addr, n)`.
- `check_worker_connection<VC>` and `restore_worker_credit_preset<VC>`: `WorkerChannel<SenderVc<VC>::worker_channel>`.
- `read_connection_semaphores`: reads `WorkerChannel<CH>::connection_semaphore_addr`.
- `start_forwarding_channels(states)`: reads `ForwardingChannel<CH>::upstream_free_slots_addr()`.
- `wait_for_upstream_routers()`: recurses over `SenderVcs::forwarding_channels`, gated by
  `ForwardingChannel<CH>::waits_for_upstream_router`.

**Section order in router_sender.hpp:**
1. channels and their features;
2. state;
3. producers (this block);
4. free-slots registers;
5. sending;
6. visits;
7. where the channel states live, and the loop;
8. init.

The producers block comes before the free-slots section, which reads `WorkerChannel<CH>`.

**Is `ForwardingChannel` warranted?** It is the smallest kind: one host flag, one init accessor and the credit. It
makes `return_slots` read as "each kind's `credit`", and it owns the static-connection facts that live in the kernel
file today. Folding it back into `return_slots` and the init loop is the alternative (Q1).

### 2.5 The kernel's sender init after C3

```cpp
    // Phase: local setup.
    RuntimeArgs args;
    read_runtime_args(args);
    DownstreamConnections downstreams{};
    if constexpr (is_sender_role) {
        ...  // zero the serviced channels' connection semaphores and producer-cursor blocks (unchanged)
    }
    ...
    if constexpr (is_sender_role) {
        auto sender_channels = SenderChannelBuffersFromAllocs<...>::make();
        sender_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));
        sender::SenderChannelInterfaces::start(args.connection_semaphores);
    }
    ...
    // Phase: upstream connections.
    if constexpr (is_sender_role) {
        sender::wait_for_upstream_routers();
    }
```

The interfaces are built only where the ERISC serves sender channels. That is v4txo8w's guard, by construction rather
than by an `if constexpr` inside a helper that every ERISC calls. On a sender-less ERISC nothing references the
storage, so it leaves the image, as v4txo8w measured (§12, "Cleanup item").

### 2.6 What is byte-identical and what changes codegen

| stage | expected | why | risk to flag |
| --- | --- | --- | --- |
| C1 | **byte-identical** on every image (-O3, -Os) | constexpr values and the types instantiated are unchanged; `holds_packet` stays FORCE_INLINE | spelling only: a lambda or `_all_` read inside a class template may need the private-helper fallback (L:52-53's GCC quirk); no codegen either way |
| C1b (option) | changes **only** where a VC has a worker position that this router does not serve (T1, T6, T7 shapes; trim replays with VC0's worker trimmed) | the VC then reads its registers right before its visit instead of in the first group (rev 4 §3 #4, §6.3) | by design; no census row has these shapes |
| C2 | **byte-identical** on every coverage image | constexpr facts and static_asserts only; coverage has no DPRINT/watcher case (ALL_INSTR = telemetry + bandwidth + code profiling), and the forwards' consistency term is true wherever forwards preset | DPRINT/watcher builds change by design: the worker credit becomes stateless there (F7) |
| C3 | **loops identical per function**; images differ in (a) init: the interfaces built in one `start` at the sender's setup point; (b) sender-less images: v4txo8w's class (storage and construction gone; -O3 operand order in receiver-buffer init); (c) address constants | bodies move verbatim into static members and one function; symbols are renamed | **a readable shape that may move code:** renaming the out-of-line functions can change their emission order, and so the start address of `sender::run`. Loop alignment moved rows by up to 6 cycles/packet in the §96 jitter study. If an address moves, the core census decides (§3), and the numbers go to the user; the loose shape does not come back by itself |
| C4 | **byte-identical** | comments, names and a parameter name | none |

## 3. Stages

The names `v4txc*` are free (variants and cov_m builds checked). Every stage gets the sender pass's protocol
(V4_SENDER_OWNERSHIP §9, §12):
- `bc_tx.sh NAME "O3 Os" PREV`: coverage_txo.py on every case, trim replays and T1-T9 included, with the swap guard
  before and after; the ok-set check; bytecmp against the previous stage;
- for stages that are not identical:
  - per-function classification (`funccmp.py`, `insn_norm_cmp2.py`, `layoutcmp.py` with RENAMES);
  - the register-map guard `loopvars_maps/structure/guard_builds.py` (loop size and sp stores/loads per role);
  - the headroom check `loopvars.py --headroom PREV NEW --role sender|receiver` and the overlays `hr_tx.sh NAME`;
- `size_report.py` against main and the lead, logged in V4_SIZE.md.

For a byte-identical stage the guard and headroom follow from the bytes; the guard is still recorded (cheap, offline).
Device runs are the main session's to queue.

| stage | variant | content | identity target | guard / headroom | lines: sender.hpp / v4.cpp / other | device |
| --- | --- | --- | --- | --- | --- | --- |
| C0 merge | v4txc0 | v4txo8 + v4rxnsf2 (below) | see below | sender guard against v4txo8, receiver guard against nsf2: loop +0, S+0, L+0; headroom: sender = `headroom_v4txo8.txt`, receiver = nsf2's (= nsa: bh_torus N/S 8/12) | 964 / 286 / downstream 618 (nsf2 621 less INLINE_WRITE_ALL_BYTES) | validation 13/13 + inline/atomic + UDM (its overlap images are new combinations); no perf A/B if the targets hold |
| C1 facts | v4txc1 | §2.1, §2.2; deletes `SENDER_TXQ`, `remote_free_slots_to_send_by`, `worker_channel_positions`, `channels_in`, the five loose masks, `forwarding_index`, the dead `SenderVc::first_level_ack`, ct's `first_sender_channel_by_vc` / `sender_channel_count_by_vc`; `holds_packet` into `SenderChannel`; K:163's assert into `SenderVcs`; the ct comment fix | **byte-identical to C0** | implied | ~960 / 286 / ct -7 | none |
| C1b (option, Q3) | v4txc1b | `reads_free_slots_at_pass_start = has_worker_channel`; `worker_positions` keeps only the one-worker assert | differs only on worker-trimmed images (listed by the compare) | guard on T1, T5, bh_trim_gtest(_1erisc), bh_2d_vc2_trim_w0(*) | +0 | trimming gtests 63/63, both modes |
| C2 command buffers | v4txc2 | §2.3; role flags and `vc_forwards_to_routers` to router_common; the receiver reads its buffers from the owner (`LOCAL_WRITE_CMD_BUF` goes) | **byte-identical to C1** on every coverage image | implied | ~945 / 286 / common ~365 (+55), downstream ~585 (-33), local_delivery ~181 (-14), init -2 | none; optional: one watcher-enabled validation run for F7 |
| C3 producers | v4txc3 | §2.4, §2.5 (+v4txo8w); `init_worker_interfaces` into `start`; the two waits into router_sender.hpp's init block (Q6); `WORKER_CREDIT_NOC` becomes `SenderChannel::producer_noc` (the interface types name the library constant); `worker_read_counter_addr` into `SenderChannelInterfaces` | sender::run, single::sender_pass, receiver::run, single::receiver_pass identical per function; only (a)-(c) of §2.6 differ | guard C2→C3 on every role: loop +0, S+0, L+0; `hr_tx.sh` cells = C0's; size: sender-less E1 -336 B .bss, -188 B code (bh_vc1, -O3; V4_SIZE.md), least free E1 stack expected back near v4rxg3's 4,236 B; check bh_vc1_vc2 at -O3 (the VC2 8/10 allocation was 16 B over there: hypothesis, the E1 saving may fit it) | ~985 / ~240 (~960 / ~265 if the waits stay) | validation (connection path moved: VC2 connect/disconnect, inline/atomic, trimming); core census A/B C2 vs C3, two passes in one job, **only if** a loop's bytes or start address moved |
| C4 comments | v4txc4 | the file header's per-pass list by features, not VC numbers; v4.cpp's "channel 0 of a VC"; `end_round(..., send_attempted)`; router_common's connection-info section header | **byte-identical to C3** | implied | ±0 | none |

**Stage 0, the merge, file by file** (22 files compared against v4rxg3; only two overlap):
- **Base:** v4txo8, every file.
- **From v4rxnsf2:**
  - router_downstream.hpp, without its `INLINE_WRITE_ALL_BYTES` definition (v4txo8 moved it to router_common.hpp;
    v4txo8's own copy differs from v4rxg3 only by that removal);
  - the additive NoC API: dataflow_api.h, blackhole/ and wormhole/noc_nonblocking_api.h (v4txo8 does not touch them).
- **fabric_erisc_router_v4.cpp:** v4txo8's, with nsf2's one line, `ForwardCommands<0>::preset(args.vc0())` for
  `preset_downstream_cmd_bufs<0>(args.vc0())`.
- **Reproducibility:** a `stage_c0.py` writes the variant, as stage_f2.py and the stage_x*.py scripts do.
- **Builder:** coverage_txo.py, because the sender reads `SENDER_CH_<i>_IS_WORKER_CHANNEL` and follows the job-464
  credit-buffer rule.

**Stage 0 identity targets:**
- **Images neither parent changed against v4rxg3** are identical to v4rxg3 and to both parents.
- **Images only v4txo8 changed** are identical to v4txo8's: every E0 image, every one-ERISC and UDM image, and the
  single-downstream E1 images that carry the interface storage. nsf2 changes none of them (V4_NS_PRESETS §9.1: 2,469
  images identical to v4rxg3).
- **Images only nsf2 changed** have receiver functions identical to nsf2's: the 394 multi-downstream E1 images, for
  those without v4txo8's E1 storage change.
- **Images both changed** (multi-downstream E1 with the interface storage) are checked per function: `receiver::run`,
  the ForwardCommands presets and `init_downstreams` / `open_downstreams` equal nsf2's; `kernel_main`'s interface part
  and the storage equal v4txo8's (`funccmp.py`).
- **The bytecmp passes:** C0 against v4txo8 should differ only in images nsf2 changed. C0 against nsf2 should differ
  only in images v4txo8 changed. nsf2's cov_m build may predate the job-464 rule, but E1 code does not read the sender
  args.

**Line totals.**
- **Before:** router_sender.hpp 964, v4.cpp 286, router_common 308, router_downstream 618 (after C0),
  router_local_delivery 195.
- **After C4:** about 960-985 / 240-265 / 365 / 585 / 181.
- **Reading:** this pass moves facts to owners; it does not shrink the code. The sender files end within about ±3% of
  today. The new block is `CommandBuffers` (~55 lines in router_common), which replaces three checks and four preset
  conditions spread over three files.
- These are estimates; each stage logs its real counts.

**Order.**
- C1 comes first: it is identical and removes the positional facts the later stages would otherwise read.
- C2 comes before C3, because `WorkerChannel::credit_cmd_buf` reads `CommandBuffers`.
- C1b can go anywhere after C1, or nowhere.
- C4 comes last.

**Device budget:**
- C0 and C3 validation;
- one core census job only if C3 moves a loop;
- the trimming gtests if C1b is chosen.

## 4. Questions for the user

1. **Names:**
   - `SenderVcs`, with `serviced`, `sender_channels`, `worker_channels`, `forwarding_channels` and
     `channels_read_at_pass_start` (alternative: `SenderChannels`);
   - `WorkerChannel<CH>` and `ForwardingChannel<CH>` (or fold the latter, §2.4);
   - `SenderChannelInterfaces`, `open_or_close_worker_connection`, `SenderChannel::producer_noc`;
   - `CommandBuffers` with kinds `ATOMICS`, `LOCAL_WRITES`, `FORWARD_DATA`, `FORWARD_CREDITS`, `WORKER_CREDITS`.

   Accept, or other words?
2. **Command buffers:** one rule asserted for all three uses, so the forwards no longer fall back to stateless on
   conflicting host ids (no current config differs)? Or keep the forwards' fallback?
3. **Pass-start grouping:** keep the worker-position rule (identical; the lead's grouping on worker-trimmed shapes), or
   take the served-worker rule rev 4 planned (C1b)?
4. **DPRINT/watcher:** the worker credit's preset gets the guard the other presets have (F7). Only DPRINT/watcher
   builds change. OK?
5. **The receiver mirror, now or later?**
   - The VC → receiver-channel index has two owners: ct `receiver_channel_by_vc` (sender) and `receiver_channel_of`
     (receiver).
   - The receiver's router-wide facts (`receiver_vcs`, `runs_vc`, `any_first_level_ack_channel`) are still loose.
   - Mirroring the sender's shape there (one index owner, a `ReceiverVcs`) would happen in this pass or after.
6. **Sender init:** move `wait_for_upstream_routers` / `wait_for_connect_request` from the kernel file into
   router_sender.hpp's init block (like router_downstream.hpp's opens), or keep the kernel file's init skeleton?
7. **v4txo8w:** adopt it as part of C3, as §2.5 does?

## 5. Implementation (stages C0-C4, 2026-11-03)

The user approved the design with the answers below (main session, 2026-11-03); an implementation agent built the
stages. Every stage has a variant (`variants/v4txc*`), a script that makes it from its parent
(`loopvars_maps/sender_consolidation/stage_c*.py`; from C1 on, a `c*.patch` applied with no fuzz; `--check` rebuilds it
in scratch and compares) and its outputs in the same directory.

**The user's answers:**
1. Names: `SenderChannels` (router-wide masks), `WorkerChannel<CH>`, `ForwardingChannel<CH>`,
   `SenderChannelInterfaces`, `CommandBuffers`.
2. Command-buffer conflicts stop the build (static_assert); no stateless fallback on conflicting host ids.
3. The served-worker pass-start rule (C1b), with the trimming validation in both ERISC modes and a short A/B.
4. The DPRINT/watcher guard on the worker credit's preset.
5. The receiver mirror later (TODO.md): only `CommandBuffers` covers both ERISCs now.
6. `wait_for_upstream_routers` / `wait_for_connect_request` into router_sender.hpp.
7. v4txo8w in C3.

**Protocol per stage:** `bc_txc.sh` (coverage_txo.py on every case at -O3 and -Os, swap guard before and after, ok-set
check, bytecmp against the parent); for stages that change code, the per-function split (`fnsplit.py`, with
`--renames` for renamed functions), the register-map guard (`structure/guard_builds.py`) and the loop-function address
check (`loopaddr.py`); `size_report.py` against main, v4rxg3 and v4txo8 (`size_summary.py` for V4_SIZE.md's columns).

### 5.0 C0: the merge (v4txc0)

`stage_c0.py` as §3 planned (it checks the premises first: nsf2 changes only router_downstream.hpp, the NoC API and
v4.cpp against v4rxg3; v4txo8 leaves the NoC API alone).

Identity, on all 2,863 built configs at both levels (sets in `c0_vs_txo8_*.txt`, `c0_vs_nsf2_*.txt`,
`nsf2_vs_g3_*.txt`, `txo8_vs_hrnew_*.txt`, `both_*.txt`):
- **C0 against v4txo8 differs on exactly the 394 images nsf2 changed** against v4rxg3 (the same set at -O3 and -Os;
  all E1). Every E0, one-ERISC and UDM image is byte-identical to v4txo8's.
- **C0 against nsf2 differs on exactly the images v4txo8 changed** against hrnew (v4rxg3 under the job-464 rule):
  2,427 at -O3, all 2,863 at -Os (v4txo8's interface storage reaches every -Os E1).
- **Only nsf2 changed** (68 images at -O3, none at -Os): byte-identical to nsf2.
- **Both changed** (326 at -O3, 394 at -Os; multi-downstream E1 with the interface storage), per function
  (`fn_*_c0_*.txt`, `mergecheck_v4txc0.txt`):
  - every receiver function but `receiver::run` (ForwardCommands presets, `init_downstreams`, `open_downstreams`)
    equals nsf2's; every telemetry function equals v4txo8's;
  - `receiver::run` and `kernel_main` are exact merges on every image: C0's change from nsf2 equals v4txo8's change
    from hrnew, and C0's change from v4txo8 equals nsf2's change from v4rxg3 (instruction multisets, registers
    normalised). On the 38 telemetry / all-instrument / code-profiling images, `receiver::run`'s change from nsf2 is
    v4txo8's X6 telemetry layout (field offsets +8) and the `round_end_impl` call target;
  - `_start` differs only by crt layout constants (end of text, bss bounds); the .bss holds v4txo8's interface storage.
- **Register-map guard:** sender images are byte-identical to v4txo8 (guard +0 by identity); receiver roles against
  nsf2 (`guard_v4rxnsf2_v4txc0_*.txt`): loop +0, S+0, L+0 on every family (60 at -O3, 71 at -Os). Headroom follows:
  the sender's is `headroom_v4txo8.txt`, the receiver's nsf2's.
- **Size** (`size_v4txc0_*.txt`): VC1 E0+E1 22,596 B at -O3 (v4txo8 23,056, nsf2 22,536), 21,068 B at -Os; largest
  one-ERISC and least free stack as v4txo8.
- **Lines:** router_sender.hpp 964, v4.cpp 286, router_downstream.hpp 618, router_common.hpp 308.
- **Device** (jobs 525 / 526): see §5.5.

### 5.1 C1: facts into their owners (v4txc1)

`stage_c1.py` (`c1.patch`, 438 lines): §2.1 and §2.2 as planned, with the user's name `SenderChannels` for the
router-wide masks. Its members: `serviced`, `worker_channels`, `forwarding_channels`, `read_at_pass_start`,
`serviced_vcs` (was `sender_vcs`). `SenderVc<VC>` collects its channels by each channel's VC (`vc_of_sender_channel`
over `NUM_SENDER_CHANNELS`: on every captured config `NUM_SENDER_CHANNELS` is the sum of the per-VC counts and no flag is
set past it, so this equals the old ranges). `holds_packet` and the bubble flow-control threshold live in
`SenderChannel<CH>` (the threshold still reads the untrimmed array, GCC's constant-expression quirk). K:163's assert
moved into `SenderChannels`.

- **Byte-identical to C0 on all 2,863 images at -O3 and -Os** (`cmp_v4txc1_*.txt`). Guard and headroom follow.
- Lines: router_sender.hpp 972 (+8), v4.cpp 285 (-1), ct 964 (-4).

### 5.2 C1b: the served-worker pass-start rule (v4txc1b)

`stage_c1b.py` (`c1b.patch`): `reads_free_slots_at_pass_start = has_worker_channel`; `worker_positions` keeps the
worker-channel mask and the one-worker assert.

- **Changes exactly 112 images at each level** (`c1b_vs_c1_*.txt`): the T1 and T7 shapes, VC0's worker trimmed while
  VC2's worker is served (bh_2d_vc2_trim_w0, bh_torus_vc2_trim_w0, two ERISCs and one). Everything else is
  byte-identical to C1, including T5 (VC1 with VC0's worker trimmed), T9 (1D), the trimming-gtest replays
  (bh_trim_gtest) and every untrimmed image. Where VC0's forwarding registers were the first read group's only
  registers, reading them in the group or right before VC0's visit gives the same code.
- Per function (`fn_c1_c1b_*.txt`): `sender::run` (two ERISCs) or `single::sender_pass` (one ERISC), and `_start`.
- Register-map guard (`guard_v4txc1_v4txc1b_*.txt`), sender role (the receiver roles of the one-ERISC images are +0):
  - -O3: two ERISCs loop -2, S+0, L+0; one ERISC +3, S+1, L+1 (12 torus images +0, S+1, L+1);
  - -Os: two ERISCs +16, S+6, L+6 (the upstream-credit address and value spilled around `return_slots<1>/<2>`; 12 torus
    images +7); one ERISC +6, S+2, L+2. These VC2 configurations build at -O3, where the loop is shorter.
- Size: T1/T7 E0 -8 B at -O3, +64 B at -Os; one ERISC +12 / +24 B.
- **Device: none (main session, 2026-11-03, option (c)).** No device configuration produces the changed images. The
  trimming gtests' replays and the exp_2d_trim rows run VC0 only; their images are byte-identical to C1 (validated as
  C0, job 525). T1/T7 exist only as mock captures through the host's trimming override (every router alike), never run
  on the device. Measuring them would need a VC2 row with a VC0-worker trimming override, a new device combination;
  not built. Prepared and not run: `txc_trim_val_runs.sh` (the trimming gtests on one variant, both modes) and
  `txc_trim_ab_runs.sh` / `txc_trim_report.py` (an A/B on exp_2d_trim's forwarding rows, which these shapes don't
  reach).

### 5.3 C2: one command-buffer owner (v4txc2)

`stage_c2.py` (`c2.patch`): §2.3 as planned. `CommandBuffers` in router_common.hpp holds the host's buffer of each
kind, `presets` (the DPRINT/watcher guard now covers the worker credit too) and `forward_presets`, the kinds-per-buffer
table and the two static_asserts (a conflict stops the build: user's answer 2). It replaces
`worker_credit_cmd_bufs_distinct`, `forward_cmd_bufs_consistent`, `local_write_cmd_buf_is_exclusive`,
`command_buffer_presets`, `local_write_state_preset`, `forwards_can_preset`, `worker_credit_preset`,
`uses_vc0_forward_cmd_bufs` (now `CommandBuffers::forwards_on_vc0_bufs(vc)`) and `LOCAL_WRITE_CMD_BUF`.
`is_sender_role` / `is_receiver_role` and `vc_forwards_to_routers` moved to router_common.hpp. GCC accepts the trimmed
host arrays here (`CommandBuffers::worker_credits` and friends are references to them).

- **Byte-identical to C1b on all 2,863 images at -O3 and -Os** (`cmp_v4txc2_*.txt`): the asserts hold on every coverage
  config, and no coverage case builds with DPRINT or the watcher. Those builds change by design: the worker credit no
  longer presets there.
- Lines: router_sender.hpp 955 (C1b 971), router_common.hpp 397 (308), router_downstream.hpp 578 (618),
  router_local_delivery.hpp 176 (195), router_init.hpp 400 (402).

### 5.4 C3: the producers (v4txc3)

`stage_c3.py` (`c3.patch`): §2.4 and §2.5 as planned, with v4txo8w (user's answers 6 and 7). Differences from the
sketch: `start()` takes the runtime args' connection-semaphore array (router_sender.hpp does not see `RuntimeArgs`);
`start_forwarding_channels(states)` sits right before `run()` (its users are `run()` and `single::run`), and the init
section after the loop holds `wait_for_connect_request` / `wait_for_upstream_routers`; parameters that would shadow the
kinds' members are named `worker_connected` and `register_addr`. The file's sections follow §2.4's order.

- **Per function** (`fn_c2_c3_*.txt`, `fnsplit.py --renames c3/renames.tsv`): on every image (2,331 differ at -O3, all
  2,863 at -Os) the differing functions are init only: `SenderChannelInterfaces::start` (was `init_worker_interfaces`
  plus the tuple's construction), `wait_for_upstream_routers`, `kernel_main` and `_start` (crt layout). Every loop
  function (`sender::run`, `single::sender_pass` / `receiver_pass` / `run`, `receiver::run`) and every out-of-line
  sender function (`WorkerChannel<CH>::credit` / `preset_credit` / `connection_event`,
  `open_or_close_worker_connection`) equals C2's, renamed calls aside.
- **Register-map guard** (`guard_v4txc2_v4txc3_*.txt`): loop +0, S+0, L+0 on every role at both levels (spill names
  renamed only).
- **Headroom** (`headroom_v4txc3.txt`): all 40 cells equal C1b's and v4txo8's (`hrcells.py`).
- **Loop start addresses move** (`loopaddr_c2_c3_*.txt`): `sender::run` -60 B (1D, ring), -100 B (2D, torus), -120 B
  (2D + VC2, Galaxy express), -160 B (VC1); `single::*` +16 B (2D one ERISC, mux, UDM), +8 B (1D); VC1's `receiver::run`
  -4 B; the 64 B line offset changes in each. The bodies are identical, so the core census decides (§3); queued with
  report 3.
- **Size:** VC1 E0+E1 -348 B at both levels (22,248 B at -O3); bh_vc1's E1 free stack 3,896 -> 4,236 B, back to
  v4rxg3's (no interface storage on a sender-less E1); largest one-ERISC 16,684 B.
- **Lines:** router_sender.hpp 1,034 (C2 955: the producers block with its comments, the init section), v4.cpp 232
  (285); the two together +26.

### 5.5 C4: comments and names (v4txc4)

`stage_c4.py` (`c4.patch`): router_sender.hpp's header by features (the per-pass list, the channel kinds named);
v4.cpp's "a VC's worker channel"; `end_round(..., send_attempted)` with its description moved above it from
`RoundCounters` (and `idle_interval` named SWITCH_INTERVAL); the sender channels' connection blocks under their own
section header in router_common.hpp; `SenderChannels`' static-connection assert on the host's trimmed array; comment
lines over 120 columns this pass made, and three older ones in router_sender.hpp, rewrapped.

- **Byte-identical to C3 on all 2,863 images at -O3 and -Os.**
- Lines: router_sender.hpp 1,041, router_common.hpp 401.

### 5.6 Device

| job | variant | what | result |
| --- | --- | --- | --- |
| 525 | v4txc0 | variant_val (13 runs) | 13/13: two ERISCs mesh 96, val1d 143, ring 19, Y-torus 12, mux 19, VC2 10, NeighborExchange 7, trimming 63 (3 skipped); one ERISC mesh 96, ring 19, VC2 10, NeighborExchange 7, trimming 63 |
| 526 | v4txc0 | inline write + flushed atomic, UDM | 1D 2/2, 2D 2/2 (main's control too); UDM unit 8/8, nightly 68/68 |
| Slurm 169770 (bh-lb-120-a08u28) | v4txc4 (= C3's bytes) | variant_val, inline / atomic, UDM | 13/13 (trimming 63 + 3 skipped in each mode); 1D 2/2, 2D 2/2; UDM 8/8, 68/68 |
| 527 / 528 (a09u32) | v4txc2 vs v4txc3 | core census and VC1 split + inter-mesh, passes c3a / c3b in reversed order | every row within ±0.3 cycles/packet (`txo_report.py --chain v4txc2,v4txc3 c3a c3b`): 2D, one ERISC, VC2 on (study and v8 splits), VC1 split means 158.4 / 158.3, inter-mesh 332.9 / 332.9. The moved loop addresses change nothing measurable |

### 5.7 run_handshakes' optimize("Os") (user decision, 2026-11-03)

Measured on v4txc4 with VC1+VC2 admitted (probes `v4txc4p` / `v4txc4pn`, `handshake_os/`): without the attribute, -O3
`run_handshakes` grows 1,136-1,204 B -> 2,720-3,116 B on two-ERISC E0 (image +576..+796 B) and 280-340 B -> 440-812 B on
one ERISC (+0..+148 B); E1 compiles it to nothing; least free stack is unchanged. VC1+VC2 E0+E1 would go 24,816 ->
25,572 B. **The user keeps the attribute.** Rule: `__attribute__((optimize("Os")))` is an allowed primitive for once-run
init code only; it is not applied anywhere else without asking. The code says why next to it (stage A9).

## 6. Part A: the helper files (ROUTER_HELPERS_REVIEW.md, user-approved 2026-11-03)

Stages after C4, same protocol (§5). `stage_a1.py` ... `stage_a9.py` with `a1.patch` ... `a9.patch`.

### 6.1 A1: router_common.hpp (v4txa1)

- The hand-listed per-channel host-arg arrays leave router_common.hpp: the ct args file provides each as one array
  (`sender_free_slots_streams` by fabric position, `sender_connection_info_addrs`, `sender_connection_semaphore_addrs`,
  `local_downstream_credit_streams[VC][entry]`), built from its scalars, which stay (fabric_router_adapter.hpp names one
  of them). `SenderChannel<CH>::free_slots_stream` reads its own.
- `downstream_sender_free_slots_stream(vc, compact)` / `local_downstream_credit_stream(vc, compact)` become
  `DownstreamTraits<VC>::local_credit_stream`, `forwarding_position` and `remote_credit_stream`.
- `has_vc2` next to `has_vc1`; `HEARTBEAT_INTERVAL`, `HEARTBEAT_SIGNATURE`.
- The router-independent helpers (compile-time loops, mask helpers, `Present` / `Each`, the placement primitives) moved
  to a new router_utils.hpp; router_common.hpp holds facts only.
- **Byte-identical to C4 on all 2,863 images at both levels.** A first version stated "the other directions, skipping a
  router's own" once, as a helper both `downstream_compact_index` and the downstream position used. It changed
  `receiver::run`'s instruction schedule on 26 capture / replay images (bh_2d_allinstr, bh_2d_trim, the trimming-gtest
  replays), where the index is computed at run time; neither FORCE_INLINE nor a template form restored identity. The
  runtime form stays as it was, and `DownstreamTraits::forwarding_position` states the rule from the downstream's side
  (compile time only).

### 6.2 A2: router_init.hpp (v4txa2)

- **Stream registers by owner** (the main session's decision): `receiver::init_channel_streams()` (each receiver channel
  the router has: its packets-received register) and `sender::init_channel_streams()` (each sender channel the router
  has, at its fabric position: free-slots register at its slot count, first-level-ack and completion registers where
  used), both before the ERISCs' first sync; `init_context_switch_state()` in router_round.hpp. `init_link_streams` /
  `init_2d_streams` (by channel number, the 2D group after the first sync) are gone.
  - Behavior change, from every captured config's args (`a2/stream_init_scan.py`, 3,124 headers including T1-T9 and
    the trimming-gtest replays): on 3,104 the set of registers written (stream id and value) is the same, only the 2D
    group moves before the first sync; on 20 (bh_neighbor) position 1's free-slots register, which no channel owns
    there (VC0 has only its worker channel) and whose id nothing else uses, is no longer written.
  - A far router writes only this router's receiver channels' packets-received registers (all still set, serviced or
    not) and, under stream credits, this router's own sender channels' ack / completion registers (all still set). A
    free-slots register is written only by this chip's producers through a connection to that channel. Trimming does
    not change the set (the router walks the host's channel counts, not the serviced flags).
- `zero_credit_counters(block)`: one helper per L1 credit-counter block in `init_txqs`.
- One owner each, in router_common.hpp: `termination_signal()`, `l1_routing_table()`, `telemetry_block()`;
  router_init.hpp's three termination casts, two routing-table casts and two telemetry casts, and router_round.hpp's
  `router_state()`, use them.
- `TeardownTrids`: not the receiver's tracker type. The receiver tags its slots' commands with `CompletionGroupTrids`
  over the same IDs (`RX_CH_TRID_STARTS[CH]`, `NUM_TRANSACTION_IDS`); teardown uses the library's tracker only to wait for
  all of them. Kept, with that said.
- **Full coverage:** every image changes (init). Per function (`fn_a1_a2_*.txt`): only init (`init_txqs`, the two
  `init_channel_streams`, `kernel_main`, `_start`); every loop function equals A1's. Loop start addresses shift -12..+8 B
  (`loopaddr_a1_a2_*.txt`).
- Register-map guard (`guard_v4txa1_v4txa2_*.txt`): loop +0, S+0, L+0 on every role.
- Device: jobs 529 (variant_val) and 530 (inline / atomic, UDM) on a09u32, §6.6.

### 6.3 A3: the kernel file (v4txa3)

- Role setup into the roles; kernel_main is the phase skeleton: `sender::reset_connections` and
  `sender::set_up_channels`; `receiver::set_up_channels` (was router_init.hpp's `init_downstream_connections` plus the
  buffer init), `receiver::open_connections` (with `wait_for_downstream_muxes`, moved from router_init.hpp) and
  `receiver::preset_forward_commands`. router_receiver.hpp includes router_init.hpp for `RuntimeArgs`.
- The header's bring-up paragraph is a list at the bring-up gate; `POLLS_PER_ROUTING_RUN`; `init_telemetry` is
  `init_telemetry_block`, apart from `telemetry::init` (the counts).
- **Full coverage:** 728 images differ at -O3, 651 at -Os (`fn_a2_a3_*.txt`); per function only init: the renamed call
  in `kernel_main` everywhere, and on one-ERISC images (172 at -O3, 158 at -Os) GCC now inlines `kernel_main` into
  `_start` (2 / 7 images the other way). Every loop function equals A2's. That inlining moves one-ERISC loop start
  addresses by +584..+756 B (`loopaddr_a2_a3_*.txt`); two-ERISC loops stay put. The census at the end covers it.


### 6.4 A4-A7: round state, local delivery, telemetry, chip coordinates (v4txa4)

- A4: `RoundCounters` keeps what every loop's round needs (heartbeat, idle rounds); the sender's fields go to
  `sender::SenderRound { RoundCounters counters; bool send_yielded; uint32_t sent_at_round_start; }` (the same layout for
  the sender as before).
- A5: `INLINE_WRITE_ALL_BYTES` in `local_inline_write`; the scatter chunk mask named `CHUNK_ENCODING_MASK` (the library's
  is local to its delivery function, so not shareable); `count_local_writes` says it is the one place the router touches
  the API's counters. **The trimming recorder's hoist (one call before the dispatch) was tried and reverted**: it
  changed `single::receiver_pass`'s loop on 50 capture / replay images, and the review asked for it only if identity
  held.
- A6: `note_sent` / `note_received`; `words_of(bytes)`; `perf_buffer()`.
- A7: `this_chip()` in router_routing.hpp, used by `receiver::run` and `single::run`.
- **Full coverage:** 2,285 images differ at -O3, 2,323 at -Os. **The loops change** (`loopbody_a3_a4_*.txt`,
  `layout_a3_a4_O3.txt`, `op_a3_a4_*.txt`, `guard_v4txa3_v4txa4_*.txt`):
  - `sender::run` (566 images at -O3, 604 at -Os): the same opcode at every position (one compare's operands swapped,
    or a register permutation); 2 trimming-gtest images reallocate (+2 loop instructions, S+1, L+1).
  - `single::run`: the prologue's routing-table loads reordered (`this_chip`); on 70 images the loop's same-opcode
    operand swaps.
  - `receiver::run` (912 images; counted with loopbody.py before its name fix in §6.9, which on telemetry images compared
    `telemetry::round_end_impl<..., receiver::run lambdas>` in place of the loop: the count, not the finding): frame
    offsets (the 8 B smaller `RoundCounters`) and, on the VC1, VC2, torus-VC2 and
    trimmed families, register reallocation. Guard at -O3: VC2 receivers -22..-56 loop instructions with S-2..-6 /
    L-2..-6 (torus VC2: S+2, L-4); VC1 receivers -4..+28 with S-4..0 / L-3..+2; 2D trim and trimming-gtest images
    -4..-28. At -Os: VC1 +1 (L+2), torus VC2 -37, 2D VC2 +4, trim +1..+10. Two-ERISC 1D, ring, torus and 2D receivers
    are +0.
  - Single-change probes (`v4txa4p1` = A3 + `this_chip` only, `v4txa4p2` = A3 + the round split only; bh_vc1, bh_2d_vc2,
    bh_torus_vc2, bh_2d, glx_express at -O3): each alone changes `receiver::run`'s loop (98 and 134 images); the round
    split alone gives all of A4's loop changes on those cases (and `sender::run`'s operand swaps). `this_chip` via a
    `tt_l1_ptr`-typed routing-table reference (`v4txa4p3`) changes the same 98: the factory's by-value return, not the
    reference type. On bh_2d its effect is a callee-saved register renaming around one stream read.
  - So A4 trades loop identity for the ownership (both pieces are frame-layout changes the register allocator follows).
    The final census (C4 vs A9, core + VC1) decides; if a row regresses, dropping the round split and / or `this_chip`
    restores the receiver's loops (the probes show which).
- Size: VC1 E0+E1 +72 B at -O3 (22,320), +32 B at -Os; stack unchanged.
- Sender headroom on A9 (= A4's bytes; `headroom_v4txa9.txt`): all 40 cells equal C3's and v4txo8's.

### 6.5 A9: study pointers out of the comments (v4txa9)

- Every RESULTS / LEDGER / design-note / job / date / tool reference in router/*.hpp and the kernel file is gone; each
  became a one-line why or was dropped where the sentence already says why; shortened paragraphs rewrapped
  (`reflow.py`). `run_handshakes`' attribute has its why (§5.7). Two stale pointers to a sentence router_sender.hpp no
  longer has ("The loop function takes no closures") now state the reason.
- **Byte-identical to A4 on all 2,863 images at -O3 and -Os.**
- Grep, zero hits:
  `grep -nE "RESULTS|LEDGER|\.md\b|§|20[0-9]{2}-[0-9]{2}|\bv[1-4][a-z0-9]*[0-9]\b|\bv4[a-z]+|\bjobs? [0-9]+|offline check|loopvars|headroom|measured|\bE[01]-[0-9]+|fabric_exp|study" router/*.hpp fabric_erisc_router_v4.cpp | grep -v FABRIC_ERISC_ROUTER_V4`

### 6.6 Device (Part A)

| job | variant | what | result |
| --- | --- | --- | --- |
| 529 (a09u32) | v4txa2 | variant_val | 13/13: two ERISCs mesh 96, val1d 143, ring 19, Y-torus 12, mux 19, VC2 10, NeighborExchange 7, trimming 63 (3 skipped); one ERISC mesh 96, ring 19, VC2 10, NeighborExchange 7, trimming 63 (3 skipped) |
| 530 (a09u32) | v4txa2 | inline write + flushed atomic, UDM | 1D 2/2, 2D 2/2 (main's control too); UDM unit 8/8, nightly 68/68 |
| 531 (a09u32) | v4txa9 | variant_val | 13/13 (trimming 63 + 3 skipped in each mode) |
| 532 (a09u32) | v4txa9 | inline write + flushed atomic, UDM | 1D 2/2, 2D 2/2; UDM 8/8, 68/68 |
| 533-536 (a09u32) | v4txc4 vs v4txa9 | census: core, VC1 split + inter-mesh, census set; passes a9a / a9b reversed | see §6.7 |

### 6.7 A9's census and the this_chip() revert (user decision, 2026-11-03)

`txo_report.py --chain v4txc4,v4txa9 a9a a9b`, cycles/packet, each pass in parentheses:
- **Flat** (±0.1): 2D (ew16, bew1024, ew4352), one ERISC, 1D, ring, torus, N/S, scatter, VC1 split 1-hop and burst
  rows, inter-mesh (mean 333.0 = 333.0).
- **VC2 rows** +0.2..+0.8 (VC2 sender v8-vc02ew1024 269.0 -> 269.7, v8-vc02bew1024 288.9 -> 289.6; vc2on-ew4352
  132.1 -> 132.5); vc2on-torusns -1.1 (167.1 -> 166.0); VC2-alone rows -0.2.
- **VC1 regressions beyond noise** (tight passes): VC1Mesh2HopMerge 16 B 137.5 -> 140.9 (+3.4), 4352 B 142.9 -> 145.3
  (+2.4), VC1MeshMulticast 4352 B 149.8 -> 151.5 (+1.8), VC1Mesh2Hop +0.5 / +0.8; VC1 mean 158.3 -> 159.0.
- **Mapping (guard, A3 -> each probe, -O3, bh_vc1 / bh_2d_vc2 / bh_torus_vc2):** `this_chip()` alone (`v4txa4p1`) gives
  every receiver loop change A4 has on these families: VC1 receivers +28 loop instructions (S-4, L-3), +21, +4, -4;
  2D VC2 -29..-46; torus VC2 -30..-56. The round split alone (`v4txa4p2`): loop +0, S+0, L+0 on every role (frame
  offsets and operand order only). So the VC1 rows trace to `this_chip()` (hypothesis until A11's A/B): its by-value
  return of `ChipCoordinates` into the loop function's local changes the register allocation of VC1's receive path.
- **The user reverted `this_chip()`** (A11, §6.9): ship the receive loops that were measured.

### 6.8 A10: the kernel file as the phase skeleton (v4txa10, stage_a10.py / a10.patch on v4txa9)

- **Includes:** the kernel file includes the kernel API and the router headers only; each router header includes what it
  uses (router_common.hpp the shared library headers, the others their own). The file-scope
  `using namespace tt::tt_fabric;` is gone from the kernel file; router_common.hpp has it inside `namespace fabric_router`
  (one line, with why: the router's code names the library's types unqualified). The `#ifdef FABRIC_2D` include moved to
  router_routing.hpp (its user), `<limits>` to router_init.hpp.
- **Supported configurations** (router_common.hpp, end): the bring-up gate became a list comment and one `static_assert`
  per unsupported feature, each with its own message; `udm_` / `vc2_` / `speedy_config_supported` moved there. No
  "brought up" / "so far" wording.
- **`finish_phase<Phase>()`** (router_init.hpp): `sync_erisc` plus the phase's postcodes, from one table of
  `(Phase, EDMStatus)` in main's order. Two of main's codes are misnamed for where they fire (`EDM_VCS_SETUP_COMPLETE`,
  `ROUTING_TABLE_INITIALIZED`): they stay at their phase end, marked "main's code, kept for tooling". `*edm_status() =
  STARTED` and the WAYPOINTs stay in kernel_main.
- **`init_router()`** (router_init.hpp): the prologue (trimming-recorder reset, `INITIALIZATION_STARTED`, router state
  `INITIALIZING`, telemetry block and counters init, code-profiling clear). **Deviation from the design:**
  `set_l1_data_cache<ENABLE_RISC_CPU_DATA_CACHE>()` is kept. With the cache off it is not a no-op: it sets bit 3 of CSR
  0x7c0, which disables the data cache; dropping it would leave the base firmware's setting.
- The teardown master's second `INITIALIZING` stays, with its why ("the router state reads INITIALIZING until the first
  round starts running", as main does).
- **`DownstreamConnections`** is built and returned by `receiver::set_up_channels(args)`; a sender-only ERISC gets an
  empty `Present<false, ...>` (no connections, no init); `single::run` still gets the connections.
- One comment on `#define FABRIC_ERISC_ROUTER_V4` (it selects the ct-args header's transaction-ID count).
- kernel_main reads: `init_router(); init_txqs();` both roles' stream init, `finish_phase<STREAMS_SET>`; runtime args,
  sender connection reset, `STARTED`; receiver then sender channel setup, `finish_phase<SETUP_DONE>`; handshake,
  downstream opens, forward-command presets, upstream connections, each followed by its `finish_phase`; the teardown
  master's `INITIALIZING`; `run_loop(downstreams)`; `teardown()`.
- **Full coverage** (`fn_a9_a10_*.txt`, `loopbody_a9_a10_*.txt`): every loop function (`sender::run`, `receiver::run`,
  `single::run`, `single::sender_pass`, `single::receiver_pass`) identical to A9 on all 2,863 images at -O3 and -Os;
  only `_start` / `kernel_main` differ (-O3: 1,336 both, 1,174 `_start`, 20 `kernel_main`, 333 with no function
  differing; -Os 1,245 / 1,155 / 138 / 325). Loop functions moved on 1,400 / 1,343 images (`loopaddr_a9_a10_*.txt`). Size as A9 (V4_SIZE.md).
- Kernel file 203 -> 103 lines; router_common 246 -> 320, router_init 339 -> 393.

### 6.9 A11: the this_chip() revert (v4txa11, stage_a11.py / a11.patch on v4txa10; user decision, 2026-11-03)

- `this_chip()` is gone from router_routing.hpp; `receiver::run` and `single::run` build `my_chip` in place, as A3
  did, from `l1_routing_table()` (A2's owner).
- **Offline check against C4 (full coverage, `loopclass2_c4_a11_*.txt`): not identical.** `single::run` and the one-ERISC
  passes are identical on every image. `receiver::run` at -O3: 688 identical, 170 frame offsets only (-16), 44 same
  opcodes, 102 OTHER; at -Os 715 / 0 / 281 / 8. `sender::run`: same opcodes on 590 / 610, 2 OTHER at -O3.
  Guard C4 -> A11 (`guard_v4txc4_v4txa11_*.txt`): -O3 bh_2d_allinstr receiver loop -1 (12 images), bh_trim_gtest sender
  +2 (S+1, L+1; 2 images); every other cell +0; -Os all +0. The VC1 receivers' +28/+21/+4/-4 and the VC2 -22..-56 are
  gone.
- **Tool fix:** loopbody.py (and loopclass.py through it) matched run functions by substring, so on telemetry images
  "receiver3run" also matched `telemetry::round_end_impl<..., receiver::run lambdas>` and compared that in place of the
  loop. It now matches mangled-name prefixes. Stages whose per-function compare (fnsplit) shows no loop function
  changing (C3, A2, A3, A10) do not depend on it.
- **Mechanism, by single-change probes** (each built on the affected families at -O3 and -Os, loops compared with
  loopclass):
  1. **The round split** (`v4txa4p2` = A3 + `SenderRound`, 21 families): every OTHER and every frame-offset receiver
     loop on the non-telemetry families (`RoundCounters` moves in the frame: `mv a0,sp` <-> `addi a0,sp,K`, offsets
     -16 also through an index register), the sender's operand swaps, and the bh_trim_gtest sender +2. Guard C4 ->
     `v4txa4p2`: +0 on every cell but that one.
  2. **Reading the table through `l1_routing_table()` in `receiver::run`** (`v4txa11p1` = A11 with A3's in-place
     `reinterpret_cast` on that one line): ring, torus and express receivers. A11 vs the probe: one commutative `add`
     with its operands swapped (`index_add` in `accumulate_first_level_acks`; 115 images), or two values trading s10 /
     s11 (8 images at -Os); the same instructions otherwise. With the cast in place, these loops equal `v4txa4p2`'s on
     all 240 images at both levels. On telemetry images it also renames registers (38 OTHER at -O3, 20 same-opcode at
     -Os).
  3. **A6's `words_of()` / `perf_buffer()`** (`v4txa11p2` = `v4txa11p1` with both written back as the inline
     expressions): telemetry-on images at -O3 only (28 OTHER, 6 same-opcode): register renaming in the
     bandwidth-counting block, and in allinstr images the round-end block's placement. With both written back, the loops
     equal `v4txa4p2`'s on all 82 telemetry receivers at both levels.
  So A11 reverts `this_chip()`'s by-value return (the VC1 / VC2 reallocations are gone), but the loops still differ
  from C4's by (1), (2) and (3). (2) and (3) are the same kind of effect as `this_chip()`: an inline helper in place of
  the expression changes the register allocator's choices. They are not attributed to any perf row (no A/B).
- Size: VC1 E0+E1 22,232 at -O3 (-16 B vs C4), 20,760 at -Os (+40 B).
- **User decision (2026-11-03): A11 stays as it is** (option c of three: (a) restore the cast in `receiver::run`, (b) also
  write the telemetry helpers back inline). `l1_routing_table()` and the telemetry helpers stay. The remaining receive-loop
  differences from C4 are operand order and register names (all three pieces) and the round split's frame offsets:
  the class accepted at X7 (V4_SENDER_OWNERSHIP.md). Outside that class: on bh_2d_allinstr images at -O3 the
  telemetry helpers also move the round-end block (guard: receiver loop -1, 12 images), and the round split's
  bh_trim_gtest sender +2 (S+1, L+1; 2 images). Neither family is in a perf row. A11 is measured itself: device round
  544-549 on a09u32 (§6.10), census set included.
- The probe builds (`v4txa4p2*`, `v4txa11p1*`, `v4txa11p2t`) were deleted after this record; the probe variants stay in
  variants/ (each is its parent plus the one change named above).

### 6.10 A11 on the device (jobs 544-549, a09u32, 2026-11-03; TT_FABRIC_NO_BW_TELEMETRY=1)

| job | what | result |
| --- | --- | --- |
| 544 | variant_val v4txa11 | 13/13: two ERISCs mesh 96, val1d 143, ring 19, Y-torus 12, mux 19, VC2 10, NeighborExchange 7, trimming 63 (3 skipped); one ERISC mesh 96, ring 19, VC2 10, NeighborExchange 7, trimming 63 (3 skipped) |
| 545 | inline write + flushed atomic, UDM | 1D 2/2, 2D 2/2 (main's control too); UDM unit 8/8, nightly 68/68 |
| 546-549 | v4txc4 vs v4txa11: VC1 split + inter-mesh, core, census set; passes a11a / a11b reversed | below |

`txo_report.py --chain v4txc4,v4txa11 a11a a11b` (cycles/packet, passes in parentheses):
- **Flat.** Every row is within ±0.3 of C4; VC1 mean 158.3 -> 158.4, inter-mesh mean 333.1 = 333.1. VC2 rows ±0.1
  (A9: +0.2..+0.8), vc2on-torusns 167.1 = 167.1 (A9: -1.1).
- **A9's VC1 regressions are gone:** VC1Mesh2HopMerge 16 B 137.5 -> 137.5 (A9 140.9), 4352 B 142.9 -> 142.9 (A9 145.3),
  VC1MeshMulticast 4352 B 149.9 -> 150.0 (A9 151.5), VC1Mesh2Hop 231.2 / 254.1 -> 231.2 / 254.2. A9 and A11 differ in
  their loops only by `this_chip()` (A10's loops are A9's), and C4 reads the same in both rounds (137.5, 142.9,
  149.8 / 149.9). So the revert removing the loss attributes A9's VC1 rows to `this_chip()`, as §6.7 proposed.
- ew4352-1e 199.9 -> 200.1 (+0.2, both passes tight). The one-ERISC loops are identical to C4's; on 24 of the 28
  bh_2d_1erisc images every loop function sits 12 B lower (the 64 B line offset changes; 4 images +748 B), from A10's
  kernel_main. Placement as the cause is a hypothesis (no A/B); +0.2 is within the 1-2 cycles/packet the user accepts.

### 6.11 Ring-multicast write after the sender pass (PARKED by the user, 2026-11-03)

Found in the 400G refresh (jobs 550/551): RingMulticast write (all_to_all, full ring, both directions) runs +0.9 to
+3.4 cycles/packet slower on v4txa11 than on v4rxg3 at every size (4352 B 147.5 -> 150.8; tight passes; main flat;
scatter, HalfRing, Unidir and Ring1Hop flat). The A11 round's census (txo*) never ran the row (its ring1 row is 1-hop),
so §6.10's "flat" does not cover it.
- **Intervals (same job, ring sweep, passes reversed):** 560: the whole step is in v4rxg3 -> v4txo8; v4txo8 -> v4txc4
  -> v4txa11 flat. 565 (arms X0, X1, X3, X3k, X3a, X3b, X4, X8): two stages, X3 -> X3k (+1.02 / +1.06 / +0.07 / +0.02
  / +2.77 at 256 / 1024 / 2048 / 3072 / 4352 B) and X3k -> X3a (+1.47 / +1.74 / +1.22 / +0.72 / +0.89); X0 -> X3 and
  X3b -> X8 flat; no other ring row moves more than 0.5.
- **Offline (the device images of 565):** every ring test at every size runs the same router images (all 72 builds
  CHANNEL_BUFFER_SIZE 4400). E1's .text is identical X3 -> X3k (36/36 configs) and X3k -> X3a (32/36; the other 4 loop
  +0). E0's sender loop shrinks: 327 insns / 31 stack loads / 13 stores (X3) -> 324 / 28 / 12 (X3k) -> 317 / 22 / 12
  (X3a); non-stack loads / stores unchanged (37 / 26); the same values stack-held. Per function: X3k `send` 78 -> 54
  plus the packed progress word's operations 21; X3a `restore_worker_credit_preset` 9 -> 5, `return_slots` 9 -> 7. No
  stage adds E0 work. Loop start -20 B (X3k), +416 B (X3a).
- **Bound context (delay A/B on v4rxv2i3, job 349, same 147.5 base):** RingMulticast write E0 slack 11.7, E1 4.6;
  scatter E0 30.8, E1 0.
- **Depth A/Bs (fabric_exp/pcs3/ringmcw4352.yaml, row_ab_runs.sh, X3 / X3k / X3a, passes reversed, slots logged):**

| TT_FABRIC_EXP_SLOTS_1D (worker, forwarding, receiver) | job | X3 | X3k | X3a | X3 -> X3k | X3k -> X3a |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 26,25,32 (the default) | 568 | 147.06 | 149.85 | 150.77 | +2.79 | +0.92 |
| 33,34,16 (shallow receiver) | 568 | 146.94 | 149.87 | 150.74 | +2.93 | +0.87 |
| 16,16,51 (deep receiver) | 568 | 149.94 | 149.97 | 150.00 | +0.03 | +0.02 |
| 16,25,32 (worker 16) | 569 | 147.14 | 149.92 | 150.84 | +2.78 | +0.92 |
| 26,16,32 (forwarding 16) | 569 | 147.70 | 150.19 | 151.14 | +2.49 | +0.94 |

- **Hypotheses:**
  - H1 (worker/forwarding arbitration under bubble flow control, set by the far receiver's depth): **not supported** as
    stated: the step does not grow at receiver 16.
  - H3 (a producer-side credit loop, worker or forwarding window): **not supported** for either window alone.
  - H2 (E0 code placement changing a latency-critical step): **open**; its A/B needs a pad variant (X3k with the loop
    start moved back), not built.
  - Only 16,16,51 removes the step, by X3 slowing 2.9 and X3a speeding 0.8 (all at 150.0). Receiver 51 or its
    interaction with the 16/16 sender windows: open.
- **Next A/B, noted, not run:** sender windows held at 16/16, receiver 32 vs 51, in one job (labels rabrmw{c,e}).
- Related, with the user, no action: the channel-trimmed MeshMulticast replay step (+2.4 write, +1.1 scatter) sits in
  v4txo8 -> v4txc0 (C0, the v4txo8 + v4rxnsf2 merge; jobs 562, 566, 567); the VC2 default-slot step (X0..X4, job 507)
  stays parked (V4_SENDER_OWNERSHIP.md §X4 bisect).

### 6.12 The channel-trimmed MeshMulticast step at C0, and v4txa11w (user go, 2026-11-03)

The step (jobs 562, 566, 567): trimmed MeshMulticast replay write +2.4, scatter +1.1, all at v4txo8 -> v4txc0 (C0, the
v4txo8 + v4rxnsf2 merge); untrimmed MeshMulticast and Mesh3HopEW flat.
- **Offline (no device; scratchpad/c0trim):** the device's MeshMulticast capture (generated/reports/channel_trimming_capture,
  567's last pass) imported as the replay does (TT_METAL_FABRIC_TRIMMING_PROFILE) in a pinned-root mock on a09u32's
  descriptor (cov_m cases x11_mcast_trim, x11_mcast_full; builds c0t_v4txo8_O3, c0t_v4txc0_O3, 124 configs each).
  - E0 identical at C0 everywhere; only E1 changes.
  - The forward-command mode (from DWARF: forwards_can_preset, forwards_have_one_destination) follows direction and
    VC0_DOWNSTREAM_EDM_SIZE only: E/W and one-entry N/S receivers FullPreset (identical to v4txo8's stateful preset);
    mid-row N/S receivers with two entries (E and W) Stateless -> PartialPreset at C0. Trimming changes neither the
    choice nor any serviced E1 image (every trimmed E1 .text is one of the untrimmed run's); it changes E0 (26 of 34).
  - N/S two-entry receiver loop, v4txo8 -> v4txc0: 596 -> 550 instructions, command-buffer register stores 78 -> 68,
    command-buffer ready reads 13 -> 17, stack loads/stores 3/3 unchanged.
  - Why multicast and not 3-hop: Mesh3HopEW's replay uses E/W receivers only (identical at C0); MeshMulticast lands on
    the mid-row N/S receivers, which forward each packet E and W (two forwards back to back).
- **Hypotheses:**
  - H-C0a (open): PartialPresetForwards::forward waits for both command buffers before the data command when either
    is busy; the second of two back-to-back forwards then waits on the first's credit buffer.
  - H-C0b: **WRONG for the write row.** It said the trimmed row is E1-bound where the untrimmed one is E0-bound (job 349:
    E0 slack 0, E1 26.2). Job 577 (v4txa11d, bnd_delay_runs.sh trim, passes x11ta / x11tb): trimmed write #1 is
    E0-bound (TX slack 4.0, RX 20.2); it holds for the scatter row only (RX-bound, RX slack 7.2). C0 changed only E1,
    so the write row's +2.4 reaches E0 indirectly (credit or latency loop): hypothesis.
- **v4txa11w (user go 2026-11-03; stage_a11w.py; v4txa11 untouched):** PartialPresetForwards::forward with a decoupled
  slow path. Fast path (both buffers ready) as today: trid, data command, credit command. Slow path: wait for the data
  buffer, data command, wait for the credit buffer, credit command. The commands are written once (data_command,
  credit_command). The two ready reads and the `node` load go in begin_read_group() / end_read_group() if that alone
  (v4txa11g) leaves every image identical; otherwise v4txa11g is its own arm.
- **Predictions (written before any build or run; hypotheses):**
  - Trimmed MeshMulticast scatter replay (E1-bound): faster on v4txa11w, by up to the C0 step (~1), if the wait order
    is what costs (H-C0a).
  - Trimmed MeshMulticast write replay (E0-bound): moves only if a credit or latency loop carries E1's forward timing
    to E0; no move would leave the C0 write step unexplained by the wait order.
  - Mesh3HopEW (capture and replay), the census core set's E/W and one-ERISC rows, SlsRowMerge and Sls3HopEW: no change
    (their receivers are FullPreset or stateless, expected byte-identical; the offline check below says which).
  - SlsMulticast and untrimmed multicast (E0-bound, job 349): no change expected; SlsTurn4 (one forward per packet at
    the turn): no change expected.
  - VC1 rows: the inter-mesh landing routers are PartialPreset (intermesh); their rows are VC1-window-bound (V4_BOUNDS
    §1b), so no change expected; a VC1 multicast that fans out could move.
- **Offline results (full coverage, cov_m v4txa11g_O3 / _Os, v4txa11w_O3; stage_a11w.py, w_check.py, w_loops.py,
  fwd_path.py in loopvars_maps/sender_pass):**
  - **Scope (both arms):** only two-ERISC E1 images with PartialPresetForwards change: 378 of 394; E0 1,054,
    FullPreset E1 610 and one-ERISC 757 images identical (all loadable sections). Same counts at -Os for g.
  - **Read group alone (v4txa11g): mixed, so its own arm (user rule).** On the trimmed-replay N/S two-entry receiver
    the fence puts a link field load ahead of the first ready read and one between the two reads, and the second
    read's first use moves to the next instruction (use distance 2 -> 1: a stall on an MMIO read, likely); loop +2
    (an extra branch), loads and spills unchanged. Over the 378 images: loop +1 to +9, nothing else.
  - **Shape (v4txa11w, without the read group):** the fast path keeps today's two back-to-back ready reads (same use
    distance) and the same command stores, the posted-write counter update interleaved with the data command's
    register writes instead of after them. The slow path is code of its own per downstream entry: receiver loops
    (static) +22 to +60 instructions on 2D / torus / Galaxy receivers and +115 to +126 on some VC1 receivers, command
    register stores +19 to +72, ready reads +4 to +13, stack loads/stores 0 to -7 (no new spill). Trimmed-replay N/S
    receiver: 550 -> 600.
  - **Size (-O3, largest E1 per case):** w +108 (bh_torus), +144 (bh_2d), +160 (bh_2d_vc2), +220 (glx_2d), +316
    (glx_express), +512 B (bh_vc1: E0+E1 22,232 -> 22,744, 3,880 B left); E0 and least free stack unchanged. g +0
    to +28 B.
- **Device round 1 (2026-11-03, a09u32), no result:** 578 (variant_val v4txa11w) stopped at device open. UMD waited on
  ETH training for core (29, 25) on device 0 from 5 s to the 300 s timeout; there were 0 compile commands and no test
  started, so v4txa11w was swapped in but never built or run. The hang capture hit the same wait (0 configs). The
  broker then held 579 (v4txa11g) for "eth/fabric fault on a present mesh", and it was killed before it started;
  580-584 were cancelled. Not a v4txa11w or v4txa11g result: validation and perf rerun after the device recovers.
- **Device round 2 (a09u32, after a reset; TT_FABRIC_NO_BW_TELEMETRY=1):**
  - **Validation clean, both arms:** 587 / 588 variant_val 13/13 each (mesh 96, val1d 143, ring 19, Y-torus 12, mux 19,
    VC2 10, NeighborExchange 7, trimming 63 (3 skipped); one ERISC mesh 96, ring 19, VC2 10, NeighborExchange 7,
    trimming 63); 589 inline write + flushed atomic 1D/2D 2/2 each, UDM 8/8 and 68/68 each.
  - **Perf** (590-593, arms v4txa11 / v4txa11w / v4txa11g in one job, passes w11a / w11b reversed;
    row_ab_report.py w11a|w11b, txo_report.py --chain):

| row | v4txa11 | v4txa11w | v4txa11g |
| --- | ---: | ---: | ---: |
| trimmed MeshMulticast scatter replay (#1) | 174.5 / 174.6 | 174.5 / 174.5 | 174.5 / 174.5 |
| trimmed MeshMulticast write replay (#1) | 139.4 / 139.3 | 139.6 / 139.6 | 139.4 / 139.1 |
| trimmed Mesh3HopEW write / scatter replay | 141.6 / 148.3 | 141.5-141.6 / 148.3 | 141.5-141.6 / 148.3 |
| SlsMulticast, SlsTurn4, SlsRowMerge, Sls3HopEW (1024, 4352 B) | - | within ±0.1 | within ±0.3 |
| core set (ew16, bew1024, ew4352, ew4352-1e, VC2 on, v8 vc02) | - | +0.0 every row | +0.0 every row |
| VC1Mesh2Hop 16 / 4352 B | 231.3 / 254.0 | +0.7 / +1.0 | +0.0 / +0.3 |
| VC1MeshMulticast 4352 B | 149.9 | +0.7 | +0.2 |
| VC1 inter-mesh 1_1..1_3 | - | +0.2 to +0.7 | +0.1 to +0.5 |
| VC1 mean / inter-mesh mean | 158.3 / 333.1 | +0.2 / +0.3 | +0.0 / +0.2 |

  - **Predictions:**
    - trimmed scatter replay faster on w (H-C0a): **WRONG** (174.55 -> 174.5). Decoupling the slow path does not
      recover the C0 step, so H-C0a (the paired wait) is **not supported**.
    - trimmed write replay: no move to w's benefit (+0.25), as allowed; the C0 write step stays unexplained.
    - E/W, core, row merge, 3-hop, turn, untrimmed multicast unchanged: held.
    - VC1 rows unchanged: **WRONG for w** (VC1Mesh2Hop +0.7 / +1.0, VC1MeshMulticast +0.7, inter-mesh +0.2 to +0.7;
      tight passes). Cause: hypothesis only (the slow path's code per entry on the multi-downstream VC1 receivers).
  - **Cost against gain:** w costs +108 to +512 B of E1 code and buys nothing measured, with small VC1 losses. g (the read
    group alone) is flat. Neither is proposed for adoption; the C0 trimmed-replay step is open (H-C0a not supported,
    H-C0b wrong for the write row).
