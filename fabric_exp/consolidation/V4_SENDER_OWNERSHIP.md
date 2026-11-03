# v4 sender: the owner's words and values (2026-11-01; revision 4, 2026-11-02)

The sender's counterpart of V4_RECEIVER_OWNERSHIP.md. The rule is the same: every concept has one owner, and the
kernel uses that owner's words and values (the user's ten points; memory feedback-readability-anchor). This note is a
proposal only. No code was written for it.

What it is based on:
- **The lead `v4rxv2i3`** (router_sender.hpp; router_single.hpp, sender half; fabric_erisc_router_v4.cpp, sender
  init). Lines are cited as `L:`.
- **Stage 1**, `v4tx1` / `v4tx1b`: validated 13/13 in job 346, held back over vocabulary.
- **The state-model variants** `v4txr1`..`v4txr6`. `v4txr3` (= A + B + C) is cited as `T:`.
- **Host** (two-ERISC Blackhole):
  - erisc_datamover_builder.cpp, cited as `B:`;
  - builder/fabric_builder_config.hpp, cited as `F:`;
  - the library: fabric_erisc_datamover_channels.hpp, fabric_router_flow_control.hpp, tt_fabric_utils.h,
    fabric_connection_interface.hpp;
  - main's kernel, fabric_erisc_router.cpp.
- **Every captured named-arg map in the coverage cases**: 4,478 maps, of which 2,840 are sender maps (two-ERISC E0
  maps and one-ERISC maps). The checks were made by scratch scripts (`argfacts.py`, `q3.py` in the session
  scratchpad, own/).

Every earlier measurement cited below is a data point tied to the structure it was measured in. It is not a constraint
on a restructure (user, 2026-11-01). All of them were measured at -O3, which production v4 VC1 routers will now use as
well (§5 Q9).

**Revision 2 (2026-11-01, the user's review: every trimmed shape).** §6 is new. It covers what the host emits for
trimmed channels, the trimmed shapes found in the captures and the ones missing, each derivation per shape, and how to
capture the missing shapes on the mock. Rows changed by it: §1 role and `reads_at_pass_start`, §2 #4, #5, #6, #7, #10,
§3 #4, §4 (identity checks), §5 Q2 and Q9.
- **WRONG** (revision 1): "`Channel::worker = serviced && !LIVE_CHECK_SKIP` matches today's rule on every channel of
  every map". Revision 1 checked only the 2,840 sender maps of the coverage cases, and none of them is a trimmed replay.
  The trimming gtests' replay captures (`~/.tt-metal-cache-trim-v4trm-{two,one}`, 584 + 292 maps) contain 16 sender
  maps where the rule fails: a VC0 forwarding channel that a global override forced on has no upstream router, and
  LIVE_CHECK_SKIP is 0 on it (§6.2).
- **WRONG** (revision 1, §4 X1): "identical (the facts are equal in all 2,840 sender maps)". The role fact is not equal
  on those 16 maps.

**Revision 3 (2026-11-01, the user's decisions).**
- Worker role: option (a), a new host arg. The host patch is
  fabric_exp/variants/v4txo_host/sender_worker_channel_arg.patch: one named arg per sender channel,
  `SENDER_CH_<i>_IS_WORKER_CHANNEL`, from the builder's own worker-type position (`get_worker_connected_sender_channel()`
  and VC2's channels). The patch puts that position in one builder lambda, `is_worker_channel`, which
  `should_force_internal_sender_skip` now uses too. The arg is emitted for every channel position, serviced or not.
- The kernel's rule: `worker = serviced && SENDER_CH_<i>_IS_WORKER_CHANNEL`, with
  `static_assert(!worker || !LIVE_CHECK_SKIP)`.
- Offline builds inject the arg through copies of the build scripts: offline_build_txo.py and coverage_txo.py.
- The missing trimmed shapes are captured on the mock from the host's real emission (§6.4, results).
- VC1 runs at -O3 in production (§5 Q9).

**Revision 4 (2026-11-02): the receiver pass's lessons applied, for review before the sender pass starts.** Changes are
marked **[r4]** where they sit. Doc only: no code, no device.
- **Base.** The sender pass runs on the receiver lead: v4rxf1..v4rxf4. Their router_sender.hpp is identical across the
  four, and equals v4rxv2i3's except two lines: first-level acks from `is_vc_first_level_ack_enabled[vc]`, and the mux
  assert without a worker (`!tensix_mux_mode || all_forwarding == 0`; it fixed T8).
- **New sections:**
  - §7: E0 register-budget inventory (loopvars.py --live, v4rxf1_O3, 1,667 sender images);
  - §8: the sender's placement rule, three proposals with predicted headroom, and a calibration probe;
  - §9: codegen discipline per stage, the receiver's causes generalized;
  - §10: style differences between the receiver lead and the sender's planned shape;
  - §11: what the 1D line VCs need from the sender.
- **Changed:**
  - §1.0: names by content and keying (per VC vs per channel), UPPER_SNAKE constants, no short aliases. They supersede
    revision 3's proposed names where they differ.
  - §2 #4: the corrected host rule ("the first sender channel of each VC whose mode has a worker"). It supersedes
    v4txo_host's patch, which marks every VC2 channel.
  - §2 #7: a third credit buffer (RD) for a 1D VC1 worker.
  - §2 #20-#22: new rows.
  - §3: read-group primitives, cold paths, comments, explicit `visit_vc<0..2>` (rev 3's recursion withdrawn).
  - §4: the stage table is replaced.
  - §5: new open questions (rev 3's are decided, listed at the end).
- **Batching is parked** (V4_SEND_BATCH.md, user decisions 2026-11-02). It does not land in this pass. Revision 3's
  "batching lands after X3" is withdrawn.

## 1. Glossary

### 1.0 Revision 4 names [r4]

These rules come from the receiver's final shape (v4rxf1..v4rxf4, V4_RECEIVER_OWNERSHIP §16). Where this table and the
tables below (revision 3) disagree, this one holds.

**Keying.**
- **Per VC (`<VC>`)** where the sender has one thing per VC: its features, its remote receiver channel and its
  remote receiver progress, its visit, and its placement (§8).
- **Per channel (`<CH>`)** where a VC has several (one worker and forwarding channels): ring bounds, free-slots
  register, counters, sends, credits, connection checks.
- Host per-channel arrays are reached through `RemoteReceiverChannel<VC>::rx` (the receiver's `ReceiverChannel<VC>::rx`
  pattern).

**Constants.** UPPER_SNAKE for tuning constants we choose and for host-provided values; derived per-type traits stay
lower_snake.

**State types.** Named by content: one flat progress struct per kind of state. No Cursor/State/Record stacks, and no
near-synonyms (Worker / WorkerCredit / WorkerConnections / WorkerInterfaces; FarRing / FarRingOf / FarRingTraits).

**Channel-kind naming (user, 2026-11-02; the receiver applies it in v4rxg1, V4_RECEIVER_OWNERSHIP §18).** Each channel
kind has `<Kind>Channel` (its facts), `<Kind>ChannelState` (its state) and `<Kind>ChannelProgress` (where it is in its
slots). The kinds use host vocabulary: Receiver, Sender, RemoteReceiver, DownstreamSender. Identifiers say "channel"
and "slot", never "ring" or "cursor" (`first_slot_addr`, `slot_after`, `index_add`); comments may still say a
channel's slots wrap.

| revision 3 or lead | rev 4 name | keyed by | what it is |
| --- | --- | --- | --- |
| `Vc<V>` | `SenderVc<VC>` | VC | the VC's sender features: `sender_channels` and `worker_channels` (masks from the serviced set), `first_level_ack`, `reads_free_slots_at_pass_start`; static_assert at most one worker channel |
| `Channel<CH>` | `SenderChannel<CH>` | channel | a sender channel's ring (`num_slots`, `first_slot_addr`, `last_slot_addr`), `free_slots_stream`, `completion_counter_addr`, `first_level_ack_counter_addr`, `connection_semaphore_addr`, `worker`, `producer_credit_cmd_buf`, `remote_free_slots_to_send`, and `vc` |
| `FarRingTraits<RX>` (keyed by the host's receiver index) | `RemoteReceiverChannel<VC>` | VC | the far router's receiver channel of VC: `rx` (the host's index, used only for its per-channel arrays), slots, `packets_received_stream` |
| `FarRing {free_slots, next_slot}` / `RemoteReceiver` (rev 3) / `RemoteReceiverProgress` (rev 4 draft) | `RemoteReceiverChannelProgress {free_slots, next_slot_addr}` | VC | this router's progress into the remote receiver channel: free slots and the next slot to fill. The receiver's `ReceiverChannelProgress` vocabulary, and the order of its `DownstreamSenderChannelProgress {free_slots, next_slot_addr}` |
| `ChannelState<CH>` {slot, sent, completed, acked, expected_connection \| upstream_credit}; lead `Worker`, `ForwardingRecord` | `SenderChannelProgress {next_slot_addr, sent, completed, first_level_acked}` (rev 4 draft: `SendProgress`) inside `SenderChannelState<CH> {progress; connection_state (worker) \| upstream_free_slots_addr (forwarding)}` | channel | a flat per-packet progress plus the producer link. It mirrors `ReceiverChannelState<VC> {progress, sources}` |
| `WorkerChannels`, `ForwardingChannels`, `FarRings` (rev 3: `Each<...>` by role) | `SenderChannelStates : Each<sender_channels, SenderChannelState>`, `RemoteReceiverChannelProgresses = Each<sender_vcs, ...>` | channel / VC | one container per kind, filled from the serviced set. Where each lives is the container's rule (§8), as the receiver's `ReceiverChannelStates` owns its rule |
| `WorkerConnections`, `WorkerInterfaceStorage`, `worker_interface<CH>()` | `worker_interface<CH>()` at its fixed location (A) | channel | the library interface; no wrapper struct |
| `sending_vcs` | `sender_vcs` (beside the receiver's `receiver_vcs`) | - | VCs with a serviced sender channel |
| `in_flight` | `forwarding_in_flight` | - | forwarding channels with packets not yet completed (or first-level acked) |
| `pending`, `pending_at_start` | `holding`, `holding_at_pass_start` | - | channels whose free-slots register says they hold a packet |
| `semaphores[CH]` | `connection_semaphores[CH]` | - | |
| `RoundCounters::attempted` | `send_yielded` (router_round.hpp, shared) | - | |
| `far_ring_start`, `worker_start(interface)` (unused parameter), `workers_start`, `far_rings_start` (by-value aggregates) | `sender_channel_progress_start<CH>()`, `remote_receiver_channel_progress_start<VC>()`, constexpr like the receiver's `progress_start<VC>()` | channel / VC | no by-value state aggregates (§9) |
| `txq` | `SENDER_TXQ` (= ct `sender_txq_id`, as the receiver's `LOCAL_NOC`) | - | host-provided |
| `tt::tt_fabric::worker_handshake_noc` at call sites | `WORKER_CREDIT_NOC` | - | host-provided |
| `0xF` / `0xf` literals (L:386, L:403) | `INLINE_WRITE_ALL_BYTES`, moved from router_downstream.hpp to router_common.hpp | - | shared constant |
| `BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS` (ct) | the one shared host-device constant (decided) | - | |
| `FIRST_VC` (receiver only; rev 4 draft: in router_common.hpp, used by both programs) | the sender's own `loop_local_vc`, derived (lowest set bit of `sender_vcs`), owned by `SenderChannelStates` with the receiver's accessors: `loop_local_progress()`, `progress_for<VC>(loop_copy)`, `after_step<VC>()` (V4_RECEIVER_OWNERSHIP §18). **P-B confirmed by the user (2026-11-02), with the remote receiver progress packed (§8.4, decision at the end)** | VC | derived trait, lower_snake |
| short names `w`, `r`, `far`, `src`; aliases `using C = Channel<CH>`, `FarT`, `W` | `worker`, `state`, `remote`, `slot_addr`; full type names, as the receiver writes them | - | lesson 1 (`me` became `my_chip`) |
| `std::atomic_signal_fence` at L:240/243, 279/282, 301/308, 467/474 | `begin_read_group()` / `end_read_group()` (router_common.hpp) | - | four read groups |
| fences at L:286 (after the header stamp) and L:312 (after the free-slots update) | not read groups: §3 #7 | - | |

The current name below is the lead's, or v4txr3's where v4tx1 introduced it ("-" means the name does not exist there).

### Features (per VC, per channel)

| current | meaning | owner's term (cite) | proposed |
| --- | --- | --- | --- |
| `vc == 0 ? .. : vc == 1 ? .. : ..` (L:77-81); `first_channel_of_vc`, `channel_count_of_vc`, `far_receiver_of_vc` (T:90-98) | a VC's sender channels and the remote receiver channel it sends into | `ACTUAL_VC<n>_SENDER_CHANNELS` (B:1357-1359); ct `VC<n>_RECEIVER_CHANNEL`, `vc_of_receiver_channel` | per-VC ct arrays `first_sender_channel_of_vc`, `sender_channel_count_of_vc`, `receiver_channel_of_vc` (`[MAX_NUM_VCS]`), next to v4rxo2's `is_vc_*` arrays (§2 #1-#2) |
| `vc_takes_worker {true,false,true}` (T:99); `CH == 0 \|\| CH == VC2_LOCAL_CHANNEL_START` (v4.cpp init) | which channel position takes a worker (worker-type), for any channel, trimmed or not | the host's convention: `get_worker_connected_sender_channel()` returns 0, "sender channel 0 is always for local worker" (fabric_builder_helpers.hpp:146-149); VC2's channel is worker-type (fabric_builder_config.hpp:61; B:1205-1212). It is not emitted as its own arg. `SENDER_CH_<i>_LIVE_CHECK_SKIP` (B:1477) is the connection protocol, and it differs from the role under trimming overrides (§6.2) | ct `sender_ch_is_worker_channel[ch]` from the new host arg `SENDER_CH_<i>_IS_WORKER_CHANNEL` (§2 #4); `Channel::worker = serviced && is_worker_channel`; `Vc<V>::worker_channels` = the VC's serviced worker channels |
| `vc_first_level_ack {ACK0, ACK1, false}` (T:100); L:80-81 | first-level acks per VC | `ENABLE_FIRST_LEVEL_ACK_VC<n>` (B:1347, B:1353; VC2 from the host patch); v4rxo2 ct `is_vc_first_level_ack_enabled` | deleted; read from the ct array |
| `Vc<V>::channels` | the VC's serviced sender channels | `IS_SENDER_CHANNEL_<i>_SERVICED` (B:1391) | `Vc<V>::sender_channels` |
| `Vc<V>::worker` | the VC's worker channel (a mask) | as above | `Vc<V>::worker_channels` (static_assert: at most one) |
| `Vc<V>::far_receiver`, `Channel::far_receiver` | the receiver channel on the router across the link | `REMOTE_RECEIVER_*` (ct :231-234, :860); main's `remote_receiver_channels` | `remote_receiver_channel` |
| `Vc<V>::first_level_ack`, `Channel::first_level_ack` | the VC's packets are first-level acked | `ENABLE_FIRST_LEVEL_ACK_VC<n>`; the receiver's `Channel::first_level_ack` | kept |
| `Vc<V>::reads_at_pass_start` | the VC's free-slots registers are read in the pass group. Today that means VCs whose convention takes a worker (VC0, VC2), even when the worker channel is trimmed | - (the kernel's policy, §3 #4) | `reads_free_slots_at_pass_start = (Vc<V>::worker_channels != 0)`: from the serviced set (§6.3) |
| `Channel::ack_counter` | the counter where the far receiver writes this channel's first-level acks | `TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR` (B:1544); the library's `acks_received_counter_ptr` (flow_control:144) | `first_level_ack_counter` |
| `Channel::completion_counter` | the same, for completions | `TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR` (B:1546); `completions_received_counter_ptr` | kept |
| `Channel::credit_cmd_buf` (T:158); `worker_credit_cmd_bufs = {WR, WR_REG}` (T:139) | the command buffer that carries a worker channel's credits | `SENDER_CH_<i>_ACK_CMD_BUF_ID` (B:1505): AT for every channel (B:534), and no device code reads it (the interface stores it as `sender_sync_noc_cmd_buf`, never used) | `Channel::producer_credit_cmd_buf`, from the host arg; the table is deleted (§2 #7, Q1) |
| `far_slots_needed_by`, `Channel::far_slots_needed` | bubble flow control: an injection channel sends only with 2 free remote slots | `ENABLE_DEADLOCK_AVOIDANCE` (B:1340), `SENDER_CH_<i>_IS_INJECTION` (B:1495), `bubble_flow_control_protected_receiver_min_slots` (F:111) | `remote_free_slots_to_send` (§2 #6) |
| `channels_in(first, count)` | the serviced channels in a range | - | `serviced_sender_channels(first, count)` |
| `pass_start_channels` | the channels whose registers the pass group reads | - | `channels_read_at_pass_start` |
| v4tx1b: `lead_vc`, `lead_channels`, `is_lead_channel`, `Channel::lead`; v4txr1: `credit_preset_channels` | the one channel that owned the preset write buffer | - | gone with B: each worker channel owns its buffer |
| v4tx1b: `register_channels`, `record_channels`, `register_vcs`, `RegisterChannels`, `Records`, `Connections`, `visit_register_channel(s)`, `visit_record_channel(s)` | placement classes | - | role names, as in v4txr1: `WorkerChannels`, `ForwardingChannels`, `visit_worker(s)`, `visit_forwarding(_channels)` (§3 #1) |

### State

| current | meaning | owner's term | proposed |
| --- | --- | --- | --- |
| `Each<MASK, T>`, `NotHeld` | one `T<I>` for each bit I of MASK | router_common's `Present` / `Absent` (the presence vocabulary) | name kept; moves to router_common.hpp beside `Present` (Q8) |
| `FarRing {free_slots, next_slot}`, `FarRingOf`, `FarRings` | this router's view of a remote receiver channel | main's `OutboundReceiverChannelPointers {num_free_slots, remote_receiver_channel_address_ptr}` (edm_fabric_flow_control_helpers.hpp:199) | `RemoteReceiver {free_slots, next_slot}`, `RemoteReceivers` |
| `FarRingTraits<RX>` | a remote receiver channel's slots and its packets-received register | `REMOTE_RECEIVER_NUM_BUFFERS_ARRAY`, `TO_RECEIVER_<i>_PKTS_SENT_ID` | `RemoteReceiverChannel<RX>` |
| `far_ring_start<VC>`, `start_far_rings`, `far_rings_start` | | - | `remote_receivers_start` (the per-VC helper has one user and is folded into it) |
| `ChannelState::acked` (lead: `Worker::acked`, `ForwardingRecord::acked`) | first-level acks taken | the library's `acks_received_and_processed` | `first_level_acked` |
| `ChannelState::completed` | completions taken | `completions_received_and_processed` | kept |
| `ChannelState::expected_connection` (lead: `Worker::expected_connection`) | the value the connection semaphore holds while nothing changes (unused or open) | connection_interface values (fabric_connection_interface.hpp:14-16) | `connection_state` |
| `ChannelState::upstream_credit` (lead: `ForwardingRecord::upstream_credit_addr`) | the upstream router's free-slots register (NoC address) | the library's `notify_persistent_connection_of_free_space` | `upstream_free_slots_addr` |
| `UpstreamCredits {addr}`, `forwarding_index` (lead: `record_index`) | init only: the cached address, from kernel_main into run | the interface's `cached_worker_semaphore_address` | deleted: `start_channels` reads the kernel interface's cached address (§3 #2) |
| `WorkerInterfaceStorage` (T:255) | an aligned byte array wrapped in a struct | - | no struct: `alignas(WorkerInterfaces) inline unsigned char worker_interface_storage[...]` |
| v4.cpp `WorkerInterfaces` (decltype of `EdmChannelWorkerInterfaces`) | every sender channel's interface; its name collides with the sender's `WorkerInterfaces` | - | `SenderChannelInterfaces` |
| `in_flight` | the forwarding channels with packets not yet completed (or first-level acked) | - | `forwarding_in_flight` |
| `pending`, `pending_at_start` | the channels whose register says they hold a packet | - | `holding`, `holding_at_pass_start` |
| `RoundCounters::attempted` | a send yielded this round (TX queue busy) | main's `did_something` (idle context switch) | `send_yielded` |

### Send, credits, connections

| current | meaning | owner's term | proposed |
| --- | --- | --- | --- |
| `txq` | the sender's TX queue | `SENDER_TXQ_ID` (B:1433) | the alias is dropped; ct `sender_txq_id` is used directly |
| `send<CH>(.., far, .., acks, attempted)` | | - | parameters renamed `remote`, `first_level_acks`, `send_yielded` |
| `observe_sent<CH>` (T:347); inline in the lead (L:292-295, 319-321) | per packet: the channel-trimming capture and telemetry | the receiver's `note_received<CH>` | `note_sent<CH>` |
| "completion batch" (L:333, comments) | the slots one completion credit returns | the receiver's completion group | "completion group" |
| `return_slots<CH>` | slots back to the channel's producer | main's `send_credits_to_upstream_workers` (fabric_erisc_router.cpp:1492) | kept |
| `credit_worker<CH>` | the worker's read counter: written into its connection info, and while connected also into the worker's L1 | library `notify_worker_of_read_counter_update`, `edm_read_counter`; `NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR` (B:1386) | `update_worker_read_counter<CH>` |
| `credit_upstream` | +n on the upstream router's free-slots register | library `notify_persistent_connection_of_free_space` | `add_upstream_free_slots` |
| `worker_credit_preset` | presets on two ERISCs | router_common `command_buffer_presets` | deleted; `command_buffer_presets` is the one owner |
| `preset_worker_credit<CH>`, `restore_worker_credit_presets` | | stateful NoC API (`ncrisc_noc_write_set_state`) | `preset_read_counter_write<CH>`, `restore_read_counter_write_presets` |
| lead `WorkerCredit<CH> {idle_word, owns_write_buffer, preset, source}` | | - | gone in v4tx1b / B (each worker channel's credit comes from its own read counter) |
| `worker_connection_event<CH>` | a changed semaphore: open, or close | library `establish_worker_connection` / `teardown_worker_connection` | `open_or_close_worker_connection<CH>` |
| `check_connection<CH>`, `check_connections<W>` | every pass, from the group's semaphores | main's `check_worker_connections` (tt_fabric_utils.h:47) | `check_worker_connection<CH>`, `check_worker_connections<W>` |
| `semaphores[CH]` | | `connection_live_semaphore` | `connection_semaphores[CH]` |
| `interfaces_of` (T:791), `worker_channels_start` (T:783) | one user each | - | folded into `start_worker_interfaces` and `run`'s init |
| `pass` visiting `visit_vc<0>`, `<1>`, `<2>` | | - | `visit_vcs<sending_vcs>` (recursion over the mask; §3 #3) |

### Telemetry, init

| current | meaning | owner's term | proposed |
| --- | --- | --- | --- |
| `telemetry::perf_recorder` | the low-resolution bandwidth recorder | `PERF_TELEMETRY_MODE` (B:904), `LOW_RESOLUTION_BANDWIDTH` | `perf_telemetry` |
| `telemetry::any_option` | | - | `any_telemetry` |
| `dynamic_stats`, `count_sent`, `Counters::tx_*`, C's `total_tx_*` | | `FABRIC_TELEMETRY_*`, FabricTelemetry `tx_bandwidth` | kept |
| `init_worker_interfaces`: local `vc2_worker`, `CH == 0 \|\| vc2_worker` | | `LIVE_CHECK_SKIP` | `Channel` role (§2 #4) |
| v4.cpp header "channel 0 of a VC" | | - | "a VC's worker channel" |

Kept as they are (**[r4]** except where §1.0 renames them: `Channel`, `ChannelState`, `SenderState`'s members): `Channel` (`vc`, `num_slots`, `first_slot`, `last_slot`, `free_slots_stream`, `connection_semaphore`,
`worker`), `ChannelState`, `slot`, `sent`, `holds_packet`, `channels_with_packets` and its three recursive helpers (they
live beside their one caller), `producer_connected`, `worker_read_counter`, `visit_worker`, `visit_forwarding`,
`visit_workers`, `visit_forwarding_channels`, `visit_vc`, `start_channels`, `add_sent`, `total_sent`,
`any_channel_holds_packet`, `run`, `wait_for_upstream_routers`, `SenderState`, `sender_pass`.

There is no sender use of crossover, completion groups as a structure, or transaction IDs: those are receiver concepts.
The sender only names them in comments, in the receiver's words.

## 2. Ownership: facts the sender decides or copies

| # | fact | sender today | true owner | host provides? | sender receives it as | static_assert kept |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | VC count | three-way ternaries (lead); `std::array<bool, 3>` x2 (T) | `builder_config::MAX_NUM_VCS` (F:46) | yes: `MAX_NUM_VCS` (B:1321) | `MAX_NUM_VCS`-sized ct arrays | v4rxo2's `MAX_NUM_VCS == 3` beside the per-VC arg lists, only there |
| 2 | sender channels per VC; remote receiver channel per VC | ternaries; ct derives `VC<n>_LOCAL_CHANNEL_START` | the host's dense layout (B:1357-1359, B:1309-1310) | yes (counts, positions) | ct arrays (§1), next to `vc_of_sender_channel` / `vc_of_receiver_channel` | `receiver_channel_of_vc` inverts `vc_of_receiver_channel` |
| 3 | VC sent on | `sending_vcs` from serviced channels | `IS_SENDER_CHANNEL_<i>_SERVICED` (B:1391; trimming clears it, B:2047-2058) | yes | unchanged (derived from the serviced set). Not from the receiver's VC-serviced defines (v4rxo2 `is_vc_serviced`) or from `has_vc1` / `has_vc2` (counts and defines, untouched by trimming) | - |
| 4 | worker or forwarding role | worker-type position (VC0's channel 0, VC2's channel) ∩ serviced | the host's convention (fabric_builder_helpers.hpp:146-149; fabric_builder_config.hpp:61; B:1205-1212), which it does not emit as an arg of its own | **not exactly** (revised, §6.2). `SENDER_CH_<i>_LIVE_CHECK_SKIP` is the connection protocol. Under trimming overrides the host stops forcing skip on serviced channels without an upstream router (B:1198). So a VC0 forwarding channel that a global override forced on, with no upstream, reads skip = 0 (16 replay maps). `serviced && !skip` would make it a second VC0 worker channel. Revision 1 called this rule exact: **WRONG** | **decided, option (a)**: the new host arg `SENDER_CH_<i>_IS_WORKER_CHANNEL` (host patch fabric_exp/variants/v4txo_host/sender_worker_channel_arg.patch), the builder's worker-type position for every channel position. The kernel reads it into a ct array, and `Channel::worker = is_sender_channel_serviced[ch] && sender_ch_is_worker_channel[ch]`. Init code asks the arg alone (a trimmed channel's role). **[r4] Corrected host rule:** the first sender channel of each VC whose routing mode has a worker (FULL_2D, LINE, NEIGHBOUR: yes; INTER_MESH: no). The VC2-line host implements it (LINE_VCS_HOST_DESIGN.md §9), and the live-check skip uses the same rule. The main tree's host, from v4txo_host's patch, still marks every VC2 channel; main fixes it before the sender pass. **v4txo_host's patch is superseded**, and offline_build_txo.py's injection must take the corrected rule (identical today: VC2 has one channel outside line mode) | `!worker \|\| !sender_ch_live_check_skip[CH]` (holds on every map, trimmed ones included, and on T1-T9); at most one worker channel per VC; the mux assert without a worker: `!tensix_mux_mode \|\| forwarding_channels == 0`. The lead's L:108 (`has_worker && ...`) and T:134 (`worker_channels != 0 && ...`) both fail to compile a mux router whose channel is trimmed (T8, §6.4); `popcount(worker_channels) <=` the number of producer-credit buffers. **No assert that a VC with serviced channels has a worker** |
| 5 | first-level acks per VC | `vc == 0 ? ACK0 : vc == 1 ? ACK1 : false` | B:1346-1353 | VC0 and VC1 yes. ACK0 == DEADLOCK_AVOIDANCE in every map. Trimming can turn both off together on a router whose VC0 is terminal-only or source-only after trimming (B:1241-1247); the trim replays have both values. ACK1 = 0. VC2 comes from the pending host patch (`ENABLE_FIRST_LEVEL_ACK_VC2 = 0`) | `is_vc_first_level_ack_enabled[vc]` (v4rxo2 ct; the user's decision 2026-11-01). It is used only by serviced channels of the VC | the bring-up gate's `!ACK_VC1` (and `!ACK_VC2`). That both ends of a link agree under trimming is the host's invariant; the kernel cannot see the far router's args (§6.3) |
| 6 | bubble flow control | `enable_deadlock_avoidance && is_injection[ch] ? 2 : 1` | the host (F:111, F:117: VC0 only) | the rule yes: `ENABLE_DEADLOCK_AVOIDANCE` (trimming can turn it off, B:1245) and `SENDER_CH_<i>_IS_INJECTION`. The host already folds the VC rule in: inj = 0 on every VC1/VC2 channel in all maps. Trimmed channels keep their inj flags (78 trimmed channels with inj = 1 in the replays), but `Channel<CH>` exists only for serviced ones. The **2** is copied twice: ct `BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS` (:545) and F:111, joined only by a "must match" comment | the rule from the host args as today; the 2 from one shared constant in a host-device header (hostdevcommon/fabric_common.h) or a named arg (Q4) | - |
| 7 | producer-credit command buffers | lead: VC0 credits on WR (emulated L1 inline write, preset), VC2 on WR_REG (`noc_semaphore_set_remote`). T: the kernel's `{WR, WR_REG}` by order among *serviced* worker channels (with VC0's worker trimmed, VC2's gets WR) | the host (it owns the receiver's buffers: `RX_CH_<n>_*_CMD_BUF_ID`) | the arg exists (`SENDER_CH_<i>_ACK_CMD_BUF_ID`, B:1505) but is AT for every channel, trimmed replays included (B:534), and unused on the device | **host patch**, by channel identity, independent of trimming: the worker-type channel 0 gets WR, VC2's channel gets WR_REG, all others keep AT. With VC0's worker trimmed, VC2's still gets WR_REG and WR stays unused. Correct for every trimmed shape (§6.3). The kernel reads `sender_channel_ack_cmd_buf_ids[CH]`. **[r4] (decided)** channel 0 → WR, VC2's worker → WR_REG; plus RD (buffer 1, unused on E0) for a 1D VC1 worker in the multi line-VC work (LINE_VCS_HOST_DESIGN §9 item 4). That is one distinct buffer per worker channel, assigned by identity, with up to three on a 1D router with three line VCs | over the *serviced* worker channels: pairwise distinct buffers, none equal to a serviced forwarding channel's or to the teardown atomics buffer (`write_at_cmd_buf`); ack NoC == `worker_handshake_noc` |
| 8 | TX queue | `txq = sender_txq_id` | the host | yes: `SENDER_TXQ_ID` (B:1433; 0 on E0, 1 on E1 for two ERISCs; 0 on one ERISC) | ct, no alias | the gate's `multi_txq_enabled` on two ERISCs |
| 9 | stream ids | router_common lists the 10 free-slots ids by hand; the `fabric_position_for_compact_sender` mapping | the host (`SENDER_CHANNEL_<i>_FREE_SLOTS_STREAM_ID`, `VC<n>_FABRIC_POSITION_START`, `TO_RECEIVER_<i>_PKTS_SENT_ID`) | yes | unchanged | `MAX_NUM_SENDER_CHANNELS == 10` beside the hand list (one arg per channel) |
| 10 | remote receiver slot facts | `FarRingTraits` from ct allocations; `last_slot` derived | the host's channel allocations (positional args; ct :231-234, :860). The allocators take no trimming input (`FabricRemoteChannelsAllocator` copies the static allocator's receiver slots, fabric_remote_channels_allocator.cpp:13-24; nor does the stream map, fabric_stream_assignment.hpp:121-122), so a trimmed router sees the untrimmed remote slots | yes | unchanged (renamed). Instantiated only for VCs in `sending_vcs` | `num_slots <= stream_count_mask`; new: `num_slots > 0` for every VC in `sending_vcs` |
| 11 | credit transport | the sender assumes L1 counters | `VC<n>_USES_COUNTER_CREDITS` (B:1304-1306) | yes | unchanged | new: `sender_channel_uses_counter_credits(CH)` in `Channel` (today it is implied by multi-TXQ and the VC2 gate) |
| 12 | completion and ack counter address = base + 4·CH | `Channel` computes it | the library's `SenderChannelFromReceiverCounterBasedCreditsReceiver` (flow_control:141-150) and the receiver's credit sender (`ptr[src_id]`) | n/a (layout) | an additive constexpr address helper beside the credit receiver (outside the listed files; Q5), or a comment naming the owner | - |
| 13 | connection semaphore address | ct `SENDER_CH_<i>_CONNECTION_SEMAPHORE_ADDR` (B:1377, study arg); the runtime arg too | the host config | yes (twice) | ct (as today) | watcher ASSERT (runtime == ct) until the host drops the runtime copy (Q6) |
| 14 | connection info and read counter | `LOCAL_SENDER_CH_<i>_CONN_INFO_ADDR` + `offsetof(EDMChannelWorkerLocationInfo, edm_read_counter)` | host + fabric_edm_types.hpp | yes | unchanged | - |
| 15 | the read-counter scratch word | lead: VC0's credit source | the host (`NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR`, B:1386) | yes | unused after B (each worker channel's own read counter); host cleanup once v4 is the only router (Q6) | - |
| 16 | source-channel stamp | `skip_src_ch_id_update = mux mode` | the host: `SKIP_SRC_CH_ID_UPDATE` (B:1362) | yes | v4rxo2's ct already reads it (shared with the receiver) | - |
| 17 | packet header fields, payload size, header size | the header's own accessors and members | fabric_edm_packet_header.hpp | n/a | unchanged (no copies on the sender) | - |
| 18 | connection protocol values | the library's `connection_interface` | fabric_connection_interface.hpp | n/a | unchanged | - |
| 19 | a worker's starting credits (0; a forwarding channel's: the ring size) | `CH == 0 \|\| vc2_worker ? 0 : N` | follows the role (#4) | yes (via #4) | `Channel` role | - |
| 20 **[r4]** | which VC's send state the loop carries in registers (§8) | implicit: VC0's worker and every remote receiver are loop locals | the kernel (a placement rule) | n/a | `FIRST_VC`, one definition in router_common.hpp, shared with the receiver's rule; absent where FIRST_VC sends nothing (the receiver's `first_vc_progress_in` returns a static `Absent` there) | - |
| 21 **[r4]** | worker channels per line VC (1D, N = 2..3) | n/a (the gate refuses line VCs) | the host (`SENDER_CH_<i>_IS_WORKER_CHANNEL`, one per line VC's first channel) | in the VC2-line host only | #4's rule, unchanged | at most one worker per VC; `popcount(worker_channels) <=` 3 buffers |
| 22 **[r4]** | first-level acks on a VC other than VC0 (1D line rings) | `!ENABLE_FIRST_LEVEL_ACK_VC1` in the bring-up gate | the host (`ENABLE_FIRST_LEVEL_ACK_VC<n>`, per VC from its routing and the topology) | VC1 on 1D rings in the VC2-line host | `is_vc_first_level_ack_enabled[vc]` (v4rxf already); the gate's `!ACK_VC1` moves to the host's table (LINE_VCS §9 item 3) | - |

## 3. Structure

| # | item | current shape | why (data points) | proposed | identical? |
| --- | --- | --- | --- | --- | --- |
| 1 | **role / placement rule** (**[r4] superseded by §8**: placement becomes a stated rule owned by one helper, picked by the user from measured proposals) | lead: `Workers` (loop locals), `ForwardingTable` (kept in memory per visit), `WorkerConnections` (the loop's interface copies; frame objects, because they escape to the out-of-line calls), `FarRings` (loop locals), `in_flight` (its own local). v4tx1: the same placement under class names. T: interfaces at a fixed location | forwarding records in memory (v4pl1 vs v4sh2b): VC1 split -6.2%, 3-hop 151.8 -> 146.0, one-ERISC 2D -2.2%, VC2 on +0.5..+1.9% (V4_PLACEMENT). Send path in memory: rejected, 3-hop 145.9 -> 159.0 (V4_STATE §11). Fixed-location interfaces (v4txr1, static only): loop -7 instructions, -6 stack accesses. Connection state in memory (v4txr4): the claim failed on 2D+VC2 (static) | placement follows the role, stated once in the state comment: worker channel state in loop locals (every pass); forwarding channel state in memory between visits; worker interfaces at a fixed location; remote receivers in loop locals. One object per placement (keep_in_memory acts on a whole object). No class vocabulary | the names: yes. The fixed location (A): no, device A/B |
| 2 | **`Each<>` and one-off structs** | lead: `Workers`, `FarRings`, `WorkerConnections`, `ForwardingTable`, `UpstreamCredits`. v4tx1: `Each<>` for states, remote receivers and interfaces. T: plus `WorkerInterfaceStorage` | the lead's structs are one-offs per role (V4_SENDER_SHAPE 1c) | `Each` in router_common.hpp; `WorkerInterfaceStorage` becomes an aligned array; `UpstreamCredits` is deleted (`start_channels` reads the kernel interface's cached address; `wait_for_upstream_routers` keeps caching it, as today) | moving `Each`: yes. `UpstreamCredits`: loop yes, init changes (check) |
| 3 | **visit structure** (with room for batching) | T: `pass` = one group read (the registers of the VCs that read at pass start, plus every worker's connection semaphore), then `check_connections`, then `visit_vc<0>`, `<1>`, `<2>`. Each `visit_vc` = its own group read where the VC reads later, its worker, then its forwarding channels behind one test | one group for every VC (v4sh2): VC1 forwarding +0.6..+2.2%, rejected. Semaphores in the group (v4rxv2i1): VC2 on -11.6 / -9.0, VC1 1-hop +1.2 (a moved stall, §33). Joint worker visit (v4rxc7w): torus +11.5. Visit-only-when-busy (v4txr6): +15..+45 instructions, skip. Worker-only batching (P1 K2/K4 probe): small rows -2..-3.6, 4352 B rows +2..+2.8, and the worker took 57-83% of a shared link. Index slots (v4sbi): +1.7..+1.9 on small writes | the ownership stages leave the inside of `visit_vc<VC>` as T (renames only). **[r4]** `pass` = group read, `check_worker_connections`, then three explicit calls `visit_vc<0>`, `<1>`, `<2>` with `static_assert(MAX_NUM_VCS == 3)`, as the receiver's `pass()` steps its VCs. Revision 3's `visit_vcs<sending_vcs>` recursion is withdrawn. `visit_vc<VC>` also owns the placement rule (§8), as the receiver's `step_channel<VC>`. **[r4] Batching is parked**, so nothing restructures `visit_vc` in this pass | explicit calls: yes (same calls, same order) |
| 4 | **`reads_at_pass_start`** | lead and T: VC0 and VC2 read their registers in the pass group, VC1 right before its visit. This is a VC-number rule: it holds even when VC0's worker is trimmed | §29-§32 (the semaphore read rides in the group) and v4sh2 (VC1 read early waits a pass); data points | derived from the serviced set: a VC reads in the pass group iff it has a serviced worker channel, whose connection semaphore rides in that group; otherwise right before its own visit. A uniform alternative (every VC reads its registers and its worker's semaphore right before its own visit) is for an A/B in the batching work, not here | yes on every shape with VC0's worker serviced. **No** where VC0's worker is trimmed and VC0 forwards; same position when no other VC has a worker; with VC2's worker live, one group becomes two (§6.3). Per-image check on the trim captures |
| 5 | **single-ERISC sender half** (**[r4] superseded by #10**) | `SenderState {connections, workers, forwarding, in_flight, far}` (T: no connections); `single::run` repeats `run`'s start (interfaces, remote receivers, channels) and its sender-round work (restore presets, `total_sent`, telemetry arguments) | one ERISC runs each role's pass out of line over a record (V4_SINGLE_ERISC) | `sender::start(...)` and the sender-round helpers (restore, sent-this-round), used by both `run` and `single::run` (two users, so they live in router_sender.hpp). `run` keeps separate local objects (placement per object) | likely (same inlined code); check |
| 6 | **worker credit** | lead: two shapes (VC0 preset or emulated inline write; VC2 `noc_semaphore_set_remote`); T: one shape on the channel's own buffer, preset on two ERISCs | B: compatible with every user of E0's buffers (V4_BOUNDS §4) | B, with the buffer from the host (§2 #7) | no (device A/B) |
| 7 **[r4]** | **fences** | ten bare `std::atomic_signal_fence` calls (L:240, 243, 279, 282, 286, 301, 308, 312, 467, 474) | read groups keep their loads together, so their latencies overlap (V4_STATE §7) | four read groups, each `begin_read_group()` / `end_read_group()` (router_common.hpp, documented once): the free-slots and semaphore group (240/243), the pre-send group (279/282), the post-send group (301/308), the visit's counter group (467/474). L:286 (after the header's `src_ch_id` stamp) and L:312 (after the free-slots register update) close no read group. Removing each alone is checked offline first; if every image is identical they go, otherwise they get a named primitive (Q3) | primitives: yes (the receiver's F1 bisection: identical). 286/312: check |
| 8 **[r4]** | **cold paths per channel** | `worker_connection_event<CH>` and `credit_worker<CH>` are instantiated per worker channel: about 95 + 25 instructions per extra worker (§ "Planned" below) | the interface type is templated on the slot count; the VC0-only preset | one noinline worker connection event for every worker channel, through the library refactor of the interface's connection state; the credit's cold part (disconnected: the read counter only; one ERISC: a whole command) one function taking the channel's addresses and buffer; thin per-channel wrappers | loop identical expected; byte compare plus the sender headroom check (a cold call's signature can move the caller's allocation) |
| 9 **[r4]** | **comments** | cycle numbers and variant names in code: L:23-24 (LEDGER E0-11 numbers), L:29 (162 -> 188), L:42-43 (+5 to +16), L:264 (5 to 18 cycles, v4s, v4s2), L:335 (141.8 -> 130.5), L:716 (+0.6% to +2.2%, v4sh2) | the receiver's C1 rule | one line per why plus one doc pointer; the State paragraph becomes one rule comment on the state containers pointing to §8 and the headroom check | identical |
| 10 **[r4]** | **single-ERISC sender half** (replaces #5's form) | `SenderState {connections, workers, forwarding, in_flight, far}` | the receiver's `ReceiverState {states, credits, frame_due}` | `SenderState {SenderChannelStates states; RemoteReceiverProgresses remote; uint32_t forwarding_in_flight}`. `sender_pass` passes the FIRST_VC copies in place (`first_vc_send_state_in(states)`), as the receiver's `receiver_pass` | layout-only expected (classify, §9) |

## 4. Proposed stages

The names `v4txo*` are free (v4txo_host holds only the superseded host patch).

**Trimmed shapes in every identity check.** The per-image compare runs on every coverage case:
- the coverage cases, whose real trimmed replays are `bh_trim_gtest` / `bh_trim_gtest_1erisc` (316 + 158 configs;
  full builds such as `v4tx1b_O3` include them, so the compare must never be restricted with `--cases`);
- the mock trim cases of §6.4 (T1-T9, captured; bases `trimx_*`).

A stage marked "identical" is identical on those too, or it is not identical.

**[r4] Stage table, revised.** Revision 3's table is replaced. The base is the receiver lead's final variant (v4rxf4,
or whichever lands last); its sender equals v4rxf1's. Every stage gets the §9 protocol:
- coverage at -O3 and -Os on every case (trim replays and T1-T9 included);
- per-image classification against the previous stage;
- the sender headroom check from X3 on (§7.4);
- size_report.py, logged in V4_SIZE.md.

"Identical" means byte-identical .text on every image built by both. "Layout-only" means every difference falls in the
receiver's classes:
- (a) stack or frame offsets, and frame-relative address arithmetic;
- (b) init stores before the loop;
- (c) frame size.

Anything else stops the stage for a single-change bisection.

| stage | variant | content | expected | device |
| --- | --- | --- | --- | --- |
| X0 renames | v4txo0 | §1.0 names on the lead's structure (`SenderChannel<CH>`, `SenderVc<VC>`, the rev 4 state and field names, constants UPPER_SNAKE, no short aliases); the four read groups as `begin_read_group()` / `end_read_group()`; `INLINE_WRITE_ALL_BYTES` and `FIRST_VC` move to router_common.hpp | identical (layoutcmp.py with RENAMES for the mangled names) | - |
| X0f | v4txo0f | L:286 and L:312: each fence removed alone, offline | identical: delete; not identical: keep it under a named primitive (Q3) | - |
| X1 structure | v4txo1 | v4tx1's structure: `Each` containers from the serviced set (the common `Each`, plus a const `get` overload), `SenderVc` / `SenderChannel` features, `visit_vc<VC>` with the three explicit calls, one producer-credit path (`return_slots`), `reads_free_slots_at_pass_start` from serviced workers, `send_progress_start<CH>()` constexpr starts (no by-value aggregates) | identical within the layout-only classes on single-VC routers. v4tx1 vs the lead was loop-count equal with register renames, so expect the same and classify it. The worker-trimmed VC0 + VC2 shapes (T1, T6, T7) change by design (§6.3) | A/B only if a production family changes beyond the classes |
| X2 host facts | v4txo2 | §2 #1-#5, #8, #11, #16, #22: the role from `SENDER_CH_<i>_IS_WORKER_CHANNEL` (corrected rule), per-VC ct arrays, the shared bubble-flow-control constant, the counter address helper in fabric_router_flow_control.hpp, the counter-credit static_assert | identical (needs main's host fix; offline: offline_build_txo.py with the corrected rule) | - |
| X3 placement | v4txo3 | §8's chosen rule, including A (worker interfaces at a fixed location; `UpstreamCredits` gone); `visit_vc<VC>` owns it; `keep_in_memory` written out at each memory-resident object | not identical by design: per-function compare; headroom check against X2 | **A/B** (§8.1 rows), after the calibration probe |
| X4 credit buffers (B) | v4txo4 | each worker channel's credit on `SENDER_CH_<i>_ACK_CMD_BUF_ID`'s buffer (WR / WR_REG, RD reserved for line VC1), preset on two ERISCs; the connection-open credit on that buffer; VC2's credit posted and preset; tx1b's read-counter source | not identical: one-worker loops expected equal in count; VC2 loops -4..-8 (EXPECTATIONS.txt, not re-checked) | **A/B**: VC2 on, VC0 + VC2, VC1 split; VC2 connect/disconnect and the inline-write gtests in validation |
| X5 cold paths | v4txo5 | §3 #8: one worker connection event (library refactor of the interface's connection state), the credit's cold part shared | loop identical expected; init/cold code smaller (about 95 + 25 instructions per extra worker) | - |
| X6 per-packet hook (C) | v4txo6 | `note_sent<CH>`; telemetry's sent packets from the channels' `sent`, 32-bit round counts, 64-bit totals once per round | identical with telemetry and capture off; telemetry-on images change (functional check of the host-visible totals, never scored) | functional only |
| X7 single | v4txo7 | §3 #10 | layout-only expected | - |
| X8 comments | v4txo8 | §3 #9 | identical | - |

After X8:
- the census (`census_runs.sh`, `census_vc1_runs.sh`);
- `variant_val_runs.sh` (13/13), the trimming gtests (63/63), and the inline-write gtests.

Device budget: the calibration probe (about 15 min), one A/B job for X3 and X4 (two passes in reversed order), the
census, the validation.

Ordering, and why:
- Renames and structure come first, so the placement compare (X3) sees only the placement.
- Placement comes before B, because B's credit path reads the worker's connection state, which §8 places.
- Cold paths come after B, because the shared event takes the channel's credit buffer as an argument (a host fact
  after X4).

v4txr6 is not staged. Batching is parked.

## 5. Open questions for the user [r4]

Revision 3's ten questions are decided ("User decisions on the open questions", at the end). Q9 (-O3 for v4's VC1
routers) is in the host since job 399; the VC1+VC2 fit at -O3 is still open on the host side (26,156 B before about
480 B of runtime args).

1. **Placement rule (§8).** Which proposal: P-A (today's role rule, stated), P-B (the receiver's FIRST_VC rule per VC;
   recommended), or P-C (first worker in registers, remote receiver progress always in registers)? Run the
   calibration probe (§8.1, about 15 min) before the pick? The receiver's estimates missed by 2-4x (§13 there).
2. **FIRST_VC.** Use the receiver's constant (VC0) for both programs, or "the first VC this router sends on"? The
   constant is one shared fact, and is absent where VC0 sends nothing; the second adapts to trimming. Recommended:
   the shared constant.
3. **L:286 and L:312**, the two fences that close no read group. If removing them is identical (X0f), delete them. If
   not, keep them under a third primitive documented with the others: name `order_point()`, or another?
4. **Names (§1.0).** `SenderChannel<CH>`, `SenderVc<VC>`, `RemoteReceiverChannel<VC>`,
   `RemoteReceiverProgress {free_slots, next_slot_addr}`, `SendProgress {next_slot_addr, sent, completed,
   first_level_acked}`, `SenderChannelState<CH>`, `SenderChannelStates`, `sender_vcs`. Accept, or other words?
5. **Credit cold part (§3 #8).** Share it too (one function with the channel's addresses and buffer as arguments, about
   25 instructions per extra worker), or share only the connection event (about 95)?
6. **Headroom tooling (§7.4).** Extend loopvars.py --headroom and the synthetic overlays to the sender as study tooling
   inside this pass (no CI), before X3?
7. **`Each`'s const `get`** (router_common.hpp, shared). Add the overload the sender's const readers need
   (`total_sent`, `any_channel_holds_packet`), or make those readers non-const?
8. **Stage order.** Placement (X3) before credit buffers (X4), as proposed?

## 6. Channel trimming (revision 2)

The rule for every derivation: what the sender runs follows from the **serviced set** alone. That covers channels,
visits, connection checks, credit paths, credit buffers, `sending_vcs` and remote receivers. Nothing assumes that a VC
with traffic has a worker, or that a VC compiled in (counts, defines) is sent on.

### 6.1 What the host emits for a trimmed router (B = erisc_datamover_builder.cpp)

| arg | trimmed behaviour | cite |
| --- | --- | --- |
| `IS_SENDER_CHANNEL_<i>_SERVICED` | starts from the per-VC counts (mux mode: only the worker channel, B:777-781). A capture or override clears the bit of each channel it did not see; it never sets one the counts lack. Not applied on express fabrics (B:826-830) | B:2047-2058 |
| global overrides | per VC, `force_enable_sender_channels` (VC-relative indices; `[]` trims the whole VC) or `force_enable_all_*`. They clear the VC's bits and set only the listed ones, on every router. With no capture, the baseline is all channels enabled | compute_mesh_router_builder.cpp:51-118, 128-170; channel_trimming_import.cpp:330-370 |
| `ACTUAL_VC<n>_SENDER_CHANNELS`, `NUM_SENDER_CHANNELS`, channel allocations, stream ids | **untouched by trimming** (counts from the topology; the allocators and the stream map take no trimming input) | B:1179-1187; fabric_remote_channels_allocator.cpp:13-24; fabric_stream_assignment.hpp:121-122 |
| `SENDER_CH_<i>_LIVE_CHECK_SKIP` | `has_static_peer \|\| forced`. Under trimming overrides `forced` is off (B:1198), so a serviced channel without an upstream router (one an override forced on) reads 0 | B:1197-1213, B:1477-1479 |
| `SENDER_CH_<i>_WAIT_STATIC_CONNECTION` | `has_static_peer && serviced`: 0 on every trimmed channel | B:1488-1491 |
| `SENDER_CH_<i>_IS_INJECTION`, `_ACK_CMD_BUF_ID`, `_ACK_NOC_ID`, connection info and semaphore addresses | emitted for every channel, trimmed or not; unchanged by trimming (ACK_CMD_BUF = AT in all replay maps) | B:1371-1380, B:1495-1507 |
| `ENABLE_DEADLOCK_AVOIDANCE`, `ENABLE_FIRST_LEVEL_ACK_VC0`, `ENABLE_SPEEDY_VC0` | DA and ACK0 together turn off where VC0 is terminal-only or source-only after trimming; speedy may turn on | B:1236-1247 |
| `IS_RECEIVER_CHANNEL_<i>_SERVICED`, `DISABLE_RX_CH<n>_FORWARDING`, open gates | receiver side only; the sender reads none of them (§6.3) | B:1401-1419, B:1570-1587 |

### 6.2 Evidence: which trimmed shapes the captures hold

Revision 1 scanned only the coverage cases' own caches (4,478 maps, 2,840 sender maps):
- those hold no trimmed replay;
- `bh_2d_trim` / `bh_1d_trim` are capture builds with every channel serviced;
- the only unserviced channels there are mux and UDM routers' (the host services only the worker channel).

The trimming gtests' replays (V4_TRIM.md §4; coverage cases `bh_trim_gtest` / `_1erisc`, whose maps live in
`~/.tt-metal-cache-trim-v4trm-{two,one}`) add 292 sender maps per ERISC mode:
- all VC0 only: 2D mesh / torus X with 4 VC0 channels, 1D with 2;
- no VC1, no VC2, no mux, no Z.

Scripts: `own/trimshapes.py`, `own/mism.py` in the session scratchpad.

The replay caches (`~/.tt-metal-cache-trim-v4trm-{two,one}`) were deleted and then recaptured by broker jobs 383/384
(re-ingested with identical config hashes, 584 / 292 maps). Re-run on the recapture: every count below is unchanged in
both ERISC modes, including the 16 forced-on maps per mode.

| sender shape (per ERISC mode) | maps | notes |
| --- | ---: | --- |
| untrimmed sender (worker + all forwarding) | 136 | 24 with DA on |
| worker live, forwarding trimmed per direction (2D: 1 of 3 live) | 40 | |
| worker live, all forwarding trimmed | 47 | 7 speedy, 3 DA |
| **VC0 worker trimmed, forwarding live** (forwarding-only VC0) | 3 | 2 with DA (torus X) |
| nothing serviced on the sender | 66 | two ERISCs: E0 runs `run_idle`; one ERISC: empty sender pass |
| of the above: a forwarding channel forced on with no upstream (serviced, LIVE_CHECK_SKIP 0, WAIT 0) | 16 | 12 in 2D, 4 in 1D; it breaks `serviced && !LIVE_CHECK_SKIP` (§2 #4) |

Checked on all of them, including the trim replays:
- `!worker || !LIVE_CHECK_SKIP` holds;
- worker-type position ∩ serviced equals today's kernel rule;
- WAIT is 0 on every trimmed channel;
- trimmed channels keep their injection flags (78 per mode with inj = 1).

**Missing from the replays**, all captured on the mock since (T1-T8, §6.4):
- VC2's worker trimmed;
- VC0's worker trimmed with VC2's live (the credit-buffer case);
- no serviced worker on a VC2 router;
- VC1 partly or fully trimmed;
- VC0's worker trimmed on a VC1 or VC1+VC2 router;
- torus Y + VC2 with the worker trimmed;
- a mux router with its one channel trimmed.

### 6.3 Each derivation, per trimmed shape

| item | rule (from the serviced set) | worker trimmed, VC0 forwards | forwarding trimmed per direction | VC2 worker trimmed | VC0 worker trimmed, VC2 live | VC1 trimmed (part / all) | nothing serviced |
| --- | --- | --- | --- | --- | --- | --- | --- |
| role | `worker = serviced && SENDER_CH_<i>_IS_WORKER_CHANNEL` (§2 #4) | no worker channel | each serviced non-worker channel forwards; the 16 forced-on ones are idle forwarding channels (no producer, WAIT 0, upstream address 0, never hold a packet) | no worker | one worker, VC2's | VC1 never has a worker | - |
| VC masks, `sending_vcs` | `Vc<V>::sender_channels` from serviced bits; a VC is sent on iff it has one | VC0 sent on | VC0 sent on | VC2 not sent on (`has_vc2` stays true: the receiver layout) | VC0 and VC2 | VC1 if any left; none: not sent on | none: `run_idle` (two ERISCs) or an empty pass (one ERISC) |
| pass group | registers of VCs with a serviced worker, plus those workers' semaphores | VC0 reads right before its visit (T and the lead read it in the group: identical position when no other worker exists) | unchanged | VC0 as untrimmed | the group holds VC2's register and semaphore; VC0 reads before its visit (T: one group of both) | VC1 reads before its visit, as today | - |
| connection checks | over serviced worker channels | none | unchanged | VC0's only | VC2's only | unchanged | - |
| producer credits | by role; forwarding credits only to channels with an upstream (WAIT) | upstream only | serviced forwarding channels only | VC0 worker + upstream | VC2 worker + upstream | unchanged | - |
| credit buffer (§2 #7) | T: by order among serviced workers; host identity: channel 0 → WR, VC2's → WR_REG | none | unchanged | WR (both) | T: WR; host: WR_REG. Both are distinct from AT, and nothing else on E0 uses either | unchanged | - |
| remote receiver | instantiated per VC in `sending_vcs`; slots from the allocation (no trimming input) | VC0's | VC0's | none for VC2 | VC0's and VC2's | VC1's if sent on | none |
| first-level acks | per VC flag (`is_vc_first_level_ack_enabled`), used by serviced channels only; ACK0 may be off after trimming | per VC0 flag | per VC0 flag | - | VC0 forwarding: ACK0; VC2: its flag (0) | ACK1 = 0 | - |
| bubble flow control | DA (may be off after trimming) && the serviced channel's IS_INJECTION | the remaining injection (turn) channels, if DA | unchanged | - | unchanged | VC1 never | - |

Not checkable in the kernel: that both ends of a link agree under trimming (DA / ACK0, B:1236-1247), and that the far
receiver channel of every VC this router sends on is serviced on the far router. Both are capture-consistency
properties the host keeps. A global override that force-enables sender channels without the far receivers would break
the second. Device evidence: the trimming gtests' 11 capture-then-replay scenarios pass on v4trm and on the lead
(V4_TRIM.md §4; V4_RECEIVER_SHAPE.md §34, 63/63 in both ERISC modes).

The sender reads nothing derived from this router's receiver-side trimming. `VC<n>_RECEIVER_CHANNEL` comes from the
counts (ct :182-185), the remote receiver slots from the allocation, and `TO_RECEIVER_<i>_PKTS_SENT_ID` from the host's
per-channel stream ids. It must not use v4rxo2's `is_vc_serviced` (the receiving VCs) or `has_vc1` / `has_vc2` (counts
and defines) to decide what it sends on.

### 6.4 Capturing the missing shapes on the mock (no device)

**Route 1: real host emission (preferred).** `coverage.py mock` runs test_tt_fabric on the mock cluster, CPU only.
test_tt_fabric sets only the capture flag and the profile path itself (tt_fabric_test_common.hpp:215-238). It leaves
`TT_METAL_FABRIC_TRIMMING_OVERRIDE` alone, so a case whose env names an override yaml builds every router through the
builder's own override path (§6.1). New cases go into a copy of coverage.py, or the main session adds them;
coverage.py is shared, so it is not edited here.

```yaml
channel_trimming_overrides:
  vc0:
    force_enable_sender_channels: [1, 2, 3]   # VC-relative; [] trims the whole VC
```

| id | base case (and its `_1erisc` twin) | override | shape |
| --- | --- | --- | --- |
| T1 | bh_2d_vc2 | vc0: [1, 2, 3] | VC0 worker trimmed, VC0 forwards, VC2 worker live (the credit-buffer case) |
| T2 | bh_2d_vc2 | vc2: [] | VC2 worker trimmed |
| T3 | bh_2d_vc2 | vc0: [1, 2, 3]; vc2: [] | no serviced worker on a VC2 router |
| T4 | bh_vc1 | vc1: [0], and separately vc1: [] | VC1 partly / fully trimmed |
| T5 | bh_vc1 | vc0: [1, 2, 3] | VC0 worker trimmed on VC1 routers |
| T6 | bh_vc1_vc2 | vc0: [1, 2, 3] | VC1+VC2 with VC0's worker trimmed (needs the VC1+VC2 gate line to build) |
| T7 | bh_torus_vc2 | vc0: [1, 2, 3] | bubble flow control with the worker trimmed, VC2 live |
| T8 | bh_2d_mux | vc0: [] | mux router with nothing serviced |
| T9 | bh_1d | vc0: [1] | 1D worker trimmed |

Caveats for route 1:
- An index at or past a VC's count on any router is a TT_FATAL (compute_mesh_router_builder.cpp:66-75). VC0 has 4
  channels on 2D mesh routers here (Z routers would have 5; none on this cluster) and 2 on 1D.
- A global override trims every router alike. Per-direction shapes need a capture-format profile
  (`TT_METAL_FABRIC_TRIMMING_PROFILE`), written by hand or edited from a `bh_2d_trim` capture.
- The VC1 cases run as two mock ranks; the env passes to both.

**Captured (2026-11-01, route 1).** The tool is `coverage_txo.py mock_pinned VARIANT CASE...`, CPU only, no device:
- It runs the mock against a runtime root of its own. That root holds `tt_metal/fabric`, `tt_metal/hw` and
  `tests/tt_metal/tt_fabric`, copied at the tree's base state; a swap active during the copy is undone from its backup.
  Everything else is linked to the tree.
- The capture router goes in the v2 slot; the capture needs every router compile to succeed.
- Afterwards the paths are rewritten to `cov_m/root`, so the cases build like any other mock capture, and the cache is
  pruned to what `build` reads: the router configs' headers and the precompiled standard-library headers (`pch/`).

Cases live in `~/.tt-metal-v3r-builds/cov_m/cases/<name>/`: cmds.json, mock.log, and `cache/`. The cache holds the
config headers and the `pch/` directories, about 2.1 GB over the 17 cases. **It must stay**: `build` copies its headers
and the commands include its PCH. Exclude it from cache cleanups (it is not a `~/.tt-metal-cache-*` directory). Overrides: `fabric_exp/consolidation/trimx/*.yaml`.

| id | case | configs (E0 + E1) | sender shape on every router (from the host's emission) |
| --- | --- | ---: | --- |
| T1 | bh_2d_vc2_trim_w0 / _1erisc | 80 / 40 | VC0 no worker, 3/3 forwarding; VC2 worker live |
| T2 | bh_2d_vc2_trim_w2 / _1erisc | 80 / 40 | VC0 worker + 3/3 forwarding; VC2 worker trimmed |
| T3 | bh_2d_vc2_trim_w02 / _1erisc | 80 / 40 | no serviced worker: VC0 3/3 forwarding only; VC2 trimmed |
| T4 | bh_vc1_trim_vc1p, bh_vc1_trim_vc1 | 94, 96 | VC0 worker + 3/3; VC1 1/3, VC1 0/3 |
| T5 | bh_vc1_trim_w0 / _1erisc | 90 / 40 | VC0 no worker, 3/3; VC1 3/3 |
| T6 | bh_vc1_vc2_trim_w0 / _1erisc | 92 / 48 | VC0 no worker, 3/3; VC1 3/3; VC2 worker live |
| T7 | bh_torus_vc2_trim_w0 / _1erisc | 80 / 40 | VC0 no worker, 3/3; VC2 live; 16 of 40 sender maps with DA and ACK0 on |
| T8 | bh_2d_mux_trim_all | 40 (one ERISC) | mux router, nothing serviced |
| T9 | bh_1d_trim_w0 / _1erisc | 72 / 36 | 1D, VC0 no worker, 1/1 forwarding |

Notes on the captures:
- Capture router: `variants/v4rxv2i3v` (the lead + v4rxc7v's VC1+VC2 gate line) for T1-T7 and T9; `variants/mainv2cap`
  for T8. The lead fails T8 at its mux static_assert, router_sender.hpp:108, which requires a worker (§2 #4).
- The VC1 cases run as two ranks; as for the base VC1 cases, the collection drops commands the other rank's output cut
  into (90-96 configs per case; 90 for the untrimmed bh_vc1_vc2). Three cut commands that slipped through in
  bh_vc1_trim_w0_1erisc (a path split mid-token, a lost define) were removed from its cmds.json; the raw file is kept as
  `cmds_with_cut_commands.json`.
- In every override capture, serviced forwarding channels with no upstream read `LIVE_CHECK_SKIP = 0` (48-96 per case):
  more evidence that LIVE_CHECK_SKIP is not the role.
- `!worker || !LIVE_CHECK_SKIP` and WAIT = 0 on trimmed channels hold on all of them.
- Not captured: per-direction shapes (a global override trims every router alike).

**Bases for identity checks** (coverage_txo.py build, the 17 cases above). Built from pinned copies of the variants:
`dataflow_api_raw.h`, `hostdev/fabric_edm_packet_header.hpp` and `buf_rw_note.h` at the tree's HEAD content, so a swap
in the live tree cannot reach the build:
- `trimx_v4rxv2i3_{O3,Os}`: the lead;
- `trimx_v4rxv2i3v_{O3,Os}`: the lead + the VC1+VC2 gate line.

Results:

| base | built | fails (every one by the router, none by the capture) |
| --- | ---: | --- |
| trimx_v4rxv2i3_O3 / _Os | 515 / 608 each | T6 (65): the bring-up gate excludes VC1+VC2. T8 (28): router_sender.hpp:108's mux assert |
| trimx_v4rxv2i3v_O3 / _Os | 580 / 608 each | T8 (28): the same mux assert |

What the bases show:
- The lead builds every other trimmed shape: VC0's worker trimmed with VC2 live, VC2's worker trimmed, no worker at
  all, VC1 partly or fully trimmed, torus with bubble flow control, and 1D.
- **New lead defect:** a mux router whose one channel trimming removes does not compile. Its `static_assert(!tensix_mux_mode
  || (has_worker && ...))` assumes a worker. Main's router builds it. The fix is the serviced-set rule (§2 #4):
  `!tensix_mux_mode || forwarding_channels == 0`.
- Pinned variants (scratch): copies of `variants/v4rxv2i3` and `v4rxv2i3v` plus the three HEAD files.

**Route 2: `synth` (fallback, compile coverage only).** SYNTH (coverage.py:268-330) edits named args by exact
replacement. A faithful trimmed shape needs `IS_SENDER_CHANNEL_<i>_SERVICED = 0` plus everything the builder derives
from it:
- `WAIT_STATIC_CONNECTION` (B:1488-1491);
- `LIVE_CHECK_SKIP` without the forced skip (B:1198);
- the open gates;
- `DISABLE_RX_CH<n>_FORWARDING`;
- the DA / ACK0 / speedy relaxation (B:1236-1247).

That copies several host rules by hand, so use it only if route 1 fails test setup.

## 7. E0 register budget: inventory (revision 4, 2026-11-02, offline) [r4]

**Method.**
- `loopvars.py --live --role sender` on every sender image of the v4rxf1 coverage build (v4rxf1_O3, -O3): 1,667
  images, every case including the trim replays and T1-T9.
- The loop: on two ERISCs, `sender::run`'s pass loop on E0; on one ERISC, `single::sender_pass`.
- **Carried:** live at the pass loop's head. **Peak:** the most values live at one instruction of the pass loop.
- The sender is the same from v4rxf1 to v4rxf4.
- Script and data: `loopvars_maps/sender_budget/` (`sender_inventory.py BUILD`, `inv_v4rxf1_O3.txt`, and the per-image
  carried values in `inv_v4rxf1_O3.json.gz`).

**Budget.** 27 allocatable registers: s0-s11, a0-a7 and t0-t6. One difference from E1: the sender's pass calls one
out-of-line function on a warm path, the worker credit, once per completion group (once per ack frame under
first-level acks). A value live across that call sits in s0-s11 or is saved around the call. The connection event and
the per-round work are cold.

### 7.1 Demand by family (two-ERISC E0 unless noted; ranges over a family's images)

| family | pass loop (instructions) | carried: registers | carried: stack slots (of which by design) | peak registers | hot values carried on the stack |
| --- | ---: | ---: | --- | ---: | --- |
| 1D, Galaxy 1D | 213 | 14 | 15 | 20 | - |
| ring, Galaxy ring | 279 | 13 | 16 | 20 | - |
| 2D, Galaxy 2D, express mesh | 455 | 13 | 23 (forwarding states 15, round 2, interfaces 2) | 22 | - |
| torus, Galaxy torus | 597 | 13 | 27 | 22 | - |
| Galaxy express (5 VC0 channels) | 752-753 | 15 | 32 | 23-24 | - |
| neighbour (speedy) | 85 | 17 | 9 | 19 | - |
| **VC1** (VC0 worker + 3 VC1 forwarding) | 832 | 15 | 39 (forwarding states 30) | 24 | VC1's remote receiver next slot (22 of 22 images) |
| **2D + VC2** (two workers) | 596 | 15 | 31 | 24 | both workers' next slot, both remote receivers' next slot (28 of 28) |
| **torus + VC2** | 596-735 | 14-15 | 31-33 | 24 | the same four (28 of 28) |
| T1: 2D + VC2, VC0's worker trimmed | 471 | 16 | 24 | 24 | - |
| T3: no serviced worker | 340 | 17 | 21 | 26 | - |
| T5: VC1, VC0's worker trimmed | 699 | 18 | 37 | 27 | - |
| T9: 1D, worker trimmed | 107 | 19 | 10 | 25 | - |
| one ERISC (`sender_pass`; state in its record) | 124-863 | 2-6 | 0-1 | 9-12 | - |

What the columns hold:
- **Carried registers on 2D:**
  - worker channel state 3 (next slot, sent, connection state; first-level-ack families carry all 5);
  - remote receiver 2 (free slots, next slot);
  - `in_flight` 1, the pass counter 1, invariants 3, promoted / unnamed 3.
- **"By design" slots:** objects the source puts in memory: forwarding channels' states (`keep_in_memory` per
  visit), the round counters (out-of-line `end_round`), the interface copies (they escape).
- **Per-packet:** the send's header pointer and source slot are stored per packet on 2D (sp+8/12, router_sender.hpp:298-299).
  The receiver's §10 recorded the same.
- **Worker-trimmed shapes** carry 12-15 hoisted invariants. With the worker's state gone, GCC fills the free registers
  with constants. That is GCC's choice, not demand.

### 7.2 Growth (measured between families)

- **+1 worker channel** (2D → 2D + VC2; torus → torus + VC2):
  - carried registers +2 (13 → 15) and peak +2 (22 → 24);
  - **four hot values move to the stack** (each worker's next slot, each remote receiver's next slot).
  - A worker channel with its VC's remote receiver is about 7 values: send progress 4, connection state 1, remote
    receiver 2. About 2 of them fit.
- **+1 forwarding-only VC** (2D → VC1: three forwarding channels and a remote receiver):
  - carried registers +2 and peak +2;
  - one hot value on the stack (VC1's remote next slot);
  - the forwarding states add 15 slots, in memory by design.
- **Per extra carried value** (the receiver's x1 / x2 / fd probes): not measured. The headroom check reads E1 only (§7.4).
- **1D line VCs** (N = 2..3) don't build today (the gate). By the growth above, three line VCs carry three workers and
  three remote receivers, about +14 values over today's 1D (14 carried, peak 20): over the budget under today's
  placement.
- **Summary:** single-worker, single-remote-receiver routers have 3-7 registers of headroom at the peak. Every
  multi-VC router is at or over the budget, and GCC chooses which send-path values go to the stack.

### 7.3 Measured facts that carry over from the receiver

| fact (where) | consequence for the sender |
| --- | --- |
| Packing hot counters loses on this ISA (RV32IM + Zba/Zbb: no bitfield insert or extract): B2-packed +7..+12, P2-packed +17..+25 (receiver §14) | no packed send progress or counters |
| Copying a channel's state in and out of memory costs about 1.9 cycles per field per pass for every channel copied. `keep_in_memory` after a whole-object write-back stores every field (receiver §13) | copy only a VC that has work (from the pass group). Copy each field where first used; write back only fields that change; never assign a whole object back |
| A slot address derived from a loaded index in front of the header load: about +4 per packet (receiver §13). The sender's P3 index form lost too: +1.7..+1.9 (V4_SEND_BATCH) | keep `next_slot_addr` as a field |
| S2-explicit's stability slope: 0 on single-channel routers, +4..+7 with two extra values on two-channel routers (receiver §14) | the target for the sender's rule: flat on single-VC routers, and nothing carried that grows per VC |
| The sender's own precedent: the send path parked in memory +3.7..+16 (v4pl3); copied in and out 1.6-8 (v4pl4) (V4_STATE §10-11) | FIRST_VC's send path stays in registers |
| Two busy VCs on one link total about 115 cycles/packet, the same as one VC (V4_SEND_BATCH, jobs 389/390) | the sender's per-packet work is the ceiling for multi-VC fan-in. Placement must not add to a busy VC's packet path |

### 7.4 The headroom check for the sender (to add to loopvars.py; specification, not built)

- **Role and images.** `--headroom BASE NEW --role sender` selects:
  - on two-ERISC configs, `active_erisc` images (`sender::run`);
  - on one-ERISC configs, `subordinate_active_erisc` images (`single::sender_pass`).
  - Today `headroom_builds` globs `subordinate_active_erisc.elf` and analyses the receiver role only.
- **Hot** (moving one to the stack fails the check):
  - the carried send progress (FIRST_VC's under P-B, every worker's under P-A);
  - remote receiver progress in registers;
  - `forwarding_in_flight`, `holding`;
  - the connection semaphores and the connection state;
  - the pass counter;
  - the post-send TX-queue status;
  - the header pointer and payload size.
- **Not hot:**
  - members of objects the source keeps in memory (forwarding channels' states; later VCs' states under P-B);
  - the round counters and the worker interfaces;
  - invariant constants and pointers;
  - saves around the out-of-line worker credit and connection event.
  - `INVARIANT` gains the sender's names: `interface`, `connections`, `upstream_free_slots_addr`.
- **Synthetic overlays** (a sender branch in `loopvars_maps/packed/synth_overlays.py`):
  - `x1` / `x2`: one or two opaque values carried pass to pass in `sender::pass`;
  - `wk`: a worker-sized state, five opaque values updated once per pass (a line VC's worker);
  - `rr`: two values updated per send (a line VC's remote receiver).
- **Families:** 1D, ring, 2D, torus, 2D + VC2, torus + VC2, VC1, Galaxy express, T1, T5.
- **Self-test:** BASE → BASE moves nothing. The first measured run is recorded as the known answers, as the receiver's
  were.

## 8. The sender's placement rule (revision 4; proposals for the user's pick) [r4]

**Today (the lead and v4rxf).** The rule is implicit:
- worker channels' send progress and connection state, and every remote receiver's progress, are loop locals;
- forwarding channels' states stay in memory between visits, touched only when the channel has a packet or packets in
  flight (`keep_in_memory`);
- the interface copies are frame objects (they escape).

§7 measured where this lands: single-worker routers fit; two-worker routers spill four hot values; VC1 spills one;
line VCs would spill more.

**P-A: today's role rule, stated** (plus A: interfaces at a fixed location).
- Rule: "worker channels and remote receivers in registers; forwarding channels in memory between their visits".
- Readability: one line; no channel singled out.
- Predicted:
  - identical on single-worker routers;
  - two-worker routers unchanged (the four spills stay, GCC's choice);
  - every added worker about +7 values.
- Stability: it fails the "survives added state" test by measurement already (2D + VC2, torus + VC2).

**P-B: the receiver's rule, per VC (recommended before the probe; the probe's measurements in §8.3 change this).**
- Rule: FIRST_VC's send state — its worker channel's send progress and connection state, and its remote receiver
  progress — is carried through the loop as a local copy. Every later VC's state (its remote receiver progress, its
  worker channel's state) and every forwarding channel's state stays in memory between visits.
- `visit_vc<VC>` owns it, as the receiver's `step_channel<VC>`:
  - FIRST_VC works on the loop's copies;
  - a later VC is visited only when the pass group shows it a packet, packets in flight, or a connection change (its
    semaphore rides in the group);
  - its fields are copied into locals where first used and written back where they change, then `keep_in_memory`.
- One accessor gives the FIRST_VC copies: `first_vc_send_state_in(states)`, used by `run` for its copy and by
  `single::sender_pass` in place.
- Readability: the same sentence as the receiver ("the first VC's state is carried in registers; every later VC's
  stays in memory between its visits"). One rule for both ERISCs.
- Predicted (estimates from §7.3's costs; this is the method that missed by 2-4x on the receiver):
  - single-VC routers (1D, ring, 2D, torus, Galaxy, neighbour, mux): 0, layout-only;
  - VC2 compiled in, VC0 traffic (VC2-on ew4352: E0-bound, slack 0): 0 to -8. VC0's four values go back to
    registers; the gain is bounded by the VC2-on / off gap, 134.6 vs 126.9;
  - VC0 and VC2 both busy: VC2's visits copy 5-7 fields, about +10..+13 per VC2 packet. **WRONG (§8.3): VC2's
    contended rate improved (-1.7 to -45.5); VC0's worsened by +14.4 to +15.0.** VC2 alone is window-bound
    at 2 slots today;
  - VC1 split rows (vc1_alloc.yaml: intra-mesh VC0 traffic on VC1-built routers, VC1 idle; the 1-hop and burst rows
    are E0-bound): 0 to -2 (VC1's spilled next slot leaves the loop);
  - VC1 traffic runs only on the inter-mesh rows, which are bound by VC1's windows (V4_BOUNDS §1b), so VC1's copy
    (about +3..+5 per VC1 visit) should be within noise there. It would show on an E0-bound VC1 row, and none is
    measured today;
  - one ERISC: 0 (its state is already in a record).
- Headroom: carried registers stay flat per added VC or line VC (+1 gate value per pass).
- Risk: a busy later VC pays the copy on every visit. With two equally busy line VCs, one of them pays about 2 cycles
  per field per visit; the probe measures it.

**P-C: by role count.**
- Rule: the first serviced worker channel (in channel order) keeps its state in registers; every other worker
  channel's state is in memory (gated as in P-B); every remote receiver progress is a loop local.
- Readability: two rules (workers by order, remote receivers always in registers). Better than P-A's implicit spill,
  worse than P-B.
- Predicted:
  - VC2-on VC0 rows: 0 to -6 (VC2's worker leaves; both remote receivers stay, so 1-2 spills may remain);
  - VC0 + VC2 busy: +6..+9 per VC2 packet; **WRONG (§8.3): -0.3 to +1.2, no cost;**
  - VC1 rows as today;
  - line VCs: +2 registers per line VC (its remote receiver).

**Not proposed:**
- packed state (lost on E1);
- every channel's state in memory every pass (the receiver's P1/P2: +12..+20);
- the send path in memory (v4pl3).

### 8.1 Calibration probe before the pick (study only; about 15 minutes of device time)

- **Variant:** P-B as it would read, on the receiver lead.
- **Rows:**
  - ew4352 with VC2 on (E0-bound, VC2 idle);
  - vc02ew4352 at v8 slots (VC0 + VC2 busy);
  - VC1 split EW 1-hop and 2HopMerge (VC1 idle);
  - VC1Inter 1_0 and 1_1 (VC1 traffic);
  - ew4352 with VC2 off (control);
  - one-ERISC ew4352.
- Two passes in reversed order, plus the sender headroom check offline (§7.4).
- The same probe gives P-C's numbers if built as a second arm (+1 build).

### 8.2 The probe (approved 2026-11-02; study only, no adoption build)

**Variants** (sender only, on v4rxf1, whose sender equals the final receiver chain's). Script:
/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/probe/make_probe.py.
- **`v4txpb` (P-B).** `FIRST_VC = 0`. VC0's worker state and far ring stay loop locals, as on the lead. `LaterVcs` (in
  memory, `keep_in_memory` once per pass per VC) holds VC1's far ring, and VC2's far ring and worker state.
  - A later VC is visited only when the pass's reads show it work: VC2 when its register shows a packet or its
    `in_flight` bit is set; VC1 when its group or `in_flight` does.
  - The far ring is copied in at the visit and written back field by field only when it changed (`write_back_changed`).
  - VC2's worker visit is `visit_kept_worker`: `visit_worker`'s logic in `visit_forwarding`'s in-memory form (fields taken
    where first needed, written where they change, the channel's bit in `in_flight`).
  - VC2's connection check compares the group's semaphore with the state in memory every pass.
- **`v4txpc` (P-C).** The first serviced worker (VC0's; VC2's where VC0's is trimmed) is a loop local. Every far ring is
  a loop local. A later worker (VC2's) is kept in memory as in P-B.
- **P-A** is the lead's codegen, so the lead (`v4rxf1`) is its control.
- Offline: both build the 13 key cases (534 of 534 images), including T1, T3, T5, T8, the trim gtests and one ERISC.

**Predictions** (written before the device runs; cycles/packet against the lead; estimates from §7's inventory and the
receiver's ~1.9 cycles per field copied):

| row | bound (V4_BOUNDS) | lead (earlier jobs) | P-B | P-C | reasoning |
| --- | --- | --- | --- | --- | --- |
| ew16, bew1024, ew4352 (2D, VC2 off) | E0 | 115.8, 116.4, 126.9 | 0 | 0 | no VC1/VC2: `LaterVcs` empty. Offline: 2D and 1D images byte-identical to the lead in both probes; ring images (both) and VC1 images (P-C) differ only by commuted compare operands (same instructions, same order) |
| ew4352, one ERISC | RX half | 198.3 | 0 | 0 | the record gains an empty member |
| VC2 on: ew4352 | E0 (slack 0) | 134.6 | -1 to -6 | 0 to -4 | VC0's four spilled send-path values return to registers (bounded by the VC2-on/off gap of 7.7); VC2's connection state now loaded from memory every pass (+1..+2) |
| VC2 on: torusns | both | 166.9 | 0 to -4 | 0 to -3 | as above; E1 may bind |
| v8 vc02ew256 / 1024 / 4352: VC0 sender | E0 (contended) | 4352: 190.7 (job 340, v4rxv2i3) | -2 to -6 per VC0 packet. **WRONG: +14.4 to +15.0** | -1 to -4 (measured -0.3 to -1.7) | VC0's registers back |
| v8 vc02: VC2 sender | E0 (contended) | 4352: 368.7 | +10 to +13 per VC2 packet. **WRONG: -2.7, -1.7, -45.5** | +6 to +9. **WRONG: +0.1, -0.3, +1.2** | VC2's copy in/out: far ring 2 loads + up to 2 stores, worker 3-4 loads + 2-4 stores, write-back compares |
| v8 vc02: the link combined | E0 | 4352: about 126 | +2 to +5 (measured +4.4, +4.5, 0.0) | +1 to +4. **WRONG: -0.6, -0.5, 0.0** | half the packets each way |
| VC1 split 1-hop and burst rows (VC0 traffic, VC1 idle) | E0 | 142.0-142.2 | 0 to -2 | 0 | P-B: VC1's spilled far-ring slot leaves the loop. P-C: identical code to the lead on VC1 routers (no VC2) |
| VC1 split 2-hop, merge, multicast; suite mean | RX / both | mean 159.3 | 0 to -1 | 0 | |
| VC1Inter 1_0 (direct, VC1 traffic) | RX | 208.3 / 209.1 | 0 (+3..+5 hidden behind E1) | 0 | |
| VC1Inter 1_1..1_3 | VC1 windows | 346-414 | within noise. **WRONG: 16 B rows +4.2 to +8.8 (4352 B rows +0.3 to +2.6)** | 0 (measured 0) | |

### 8.3 Probe results (2026-11-02; jobs 422, 425, 429, 432)

**Runs.**
- Rows: txp_runs.sh. Job 422 ran the routers in the order lead, P-B, P-C; job 425 in the order P-C, P-B, lead. Job 432
  (`TXP_SET=vc2`, the VC2 rows below) ran both orders.
- PC samples: bnd_pcs_runs.sh, job 429.
- Report: txp_report.py.
- The two passes agree within 0.3 (VC1 inter-mesh 16 B rows within 0.5).
- Cycles/packet, mean of the two passes; deltas against the lead. About 30 minutes of device time in all.

**Rows without both VCs busy.**

| row | lead | P-B | P-C | predicted P-B / P-C |
| --- | ---: | ---: | ---: | --- |
| ew16, bew1024, ew4352 (2D, VC2 off) | 115.8, 116.5, 126.9 | 0.0 | 0.0 | 0 / 0 |
| ew4352, one ERISC | 198.3 | 0.0 | 0.0 | 0 / 0 |
| VC2 on, VC2 idle: ew4352 | 134.6 | -2.8 | -3.3 | -1..-6 / 0..-4 |
| VC2 on: torusns | 166.9 | +0.2 | +0.1 | 0..-4 / 0..-3 |
| VC2 alone (v8, window-bound): vc2ew256, 1024, 4352 | 223.7, 226.5, 240.8 | +4.5, +3.0, +4.7 | +0.9, -0.1, +1.4 | not predicted |
| VC1 split: 1-hop and burst rows (8 rows) | 141.9-142.2 | -0.0..+0.1 | -0.1..+0.1 | 0..-2 / 0 |
| VC1 split: mean of 13 rows | 159.2 | -0.0 | -0.0 | 0..-1 / 0 |
| VC1Inter 1_0 (16 B, 4352 B) | 208.3, 209.1 | 0.0, 0.0 | 0.0, 0.0 | 0 / 0 |
| VC1Inter 1_1, 1_2, 1_3 at 16 B | 346.1, 346.1, 375.1 | +8.8, +5.6, +4.2 | +0.2, +0.0, -0.3 | noise (**WRONG**) / 0 |
| VC1Inter 1_1, 1_2, 1_3 at 4352 B | 397.1, 395.5, 412.4 | +0.3, +0.6, +2.6 | -0.0, -0.1, +0.0 | noise / 0 |

**VC0 + VC2 on one link (v8 split: VC0 17/10/16, VC2 8/10), per VC and combined.**
- Where each rate comes from:
  - VC0's contended rate: from vc02ew* (4M packets each; VC0 finishes first).
  - VC2's contended rate: from vc02bew* (VC0 40M, VC2 2M; VC2 finishes first).
  - Combined while both send: 1/(1/VC0 + 1/VC2). This is an estimate from the two rows, made as in V4_RECEIVER_SHAPE
    §36.
- The vc02ew rows' VC2 rate (lead 265.3, 267.2, 307.3) includes VC2's tail after VC0 finishes, so it is not used.

| size | quantity | lead | P-B | P-C |
| --- | --- | ---: | ---: | ---: |
| 256 | VC0 sender, VC2 busy | 201.9 | 216.8 (+15.0) | 200.2 (-1.7) |
| 256 | VC2 sender, VC0 busy | 281.7 | 279.1 (-2.7) | 281.8 (+0.1) |
| 256 | combined | 117.6 | 122.0 (+4.4) | 117.0 (-0.6) |
| 256 | VC0's share of the packets | 58% | 56% | 58% |
| 1024 | VC0 sender | 201.3 | 215.7 (+14.4) | 199.8 (-1.6) |
| 1024 | VC2 sender | 283.7 | 282.1 (-1.7) | 283.5 (-0.3) |
| 1024 | combined | 117.8 | 122.2 (+4.5) | 117.2 (-0.5) |
| 4352 | VC0 sender | 190.4 | 205.2 (+14.8) | 190.1 (-0.3) |
| 4352 | VC2 sender | 369.9 | 324.4 (-45.5) | 371.1 (+1.2) |
| 4352 | combined | 125.7 | 125.7 (0.0) | 125.7 (0.0) |
| 4352 | VC0's share | 66% | 61% | 66% |

- Other figures from the same rows:
  - The profiler's link figure for vc02ew (the longest sender's cycles over all packets, VC2's tail included). Lead:
    132.7, 133.6, 153.6. P-B: +1.2, +1.3, -5.9. P-C: 0.0, -0.3, +0.5.
  - vc02bew's VC0 rate (about 90% of its run with VC2 idle). Lead: 139.0, 139.0, 139.9. P-B: -6.8, -6.8, -2.2. P-C:
    -5.7, -5.7, -2.6.
  - From these, VC0 alone with VC2 compiled in at 256 B is about 134 (lead), -8 (P-B) and -6 (P-C). Derived, not
    measured.
- At 4352 B the combined rate is pinned at 125.7 for all three routers: about the 2D VC0-alone rate (126.9), so the
  link or the far side bounds it. P-B moves 5% of the packets from VC0 to VC2. At 256 and 1024 B (E0-bound), P-B loses
  4.4 to 4.5 cycles per packet combined (+3.8%).

**What each pair isolates** (by source).
- **P-B vs P-C.** The two variants differ only in the far rings' placement flags (`vc1_far_kept`, `vc2_far_kept`; the
  file diff is those lines).
  - On 2D + VC2 the only difference is VC2's far ring in memory. Measured costs: combined +4.4 / +4.5 at 256 / 1024 B;
    at 4352 B the share moves (VC0 +14.8, VC2 -45.5, combined unchanged); VC2 alone +3.0 to +4.7.
  - On VC1 builds the only difference is VC1's far ring in memory. Measured cost: the window-bound 16 B VC1 inter-mesh
    rows +4.2 to +8.8.
- **P-C vs the lead.** On 2D + VC2, VC2's worker state moves to memory with gated visits (`visit_kept_worker`); the far
  rings are unchanged. Measured: VC2-on ew4352 -3.3; VC0 contended -0.3 to -1.7; combined -0.6 to 0.0; VC2 alone -0.1
  to +1.4.
  - VC1 and single-VC images are unchanged (identical, or the same instructions with commuted compare operands).
- **Mechanisms (hypotheses, not isolated below the placement):**
  - **P-B's combined loss at 256 / 1024 B:** the far ring's copy-in and write-back on every VC2 visit (2 loads, 2
    compares, up to 2 stores). The static map shows the local copy's `free_slots` itself spilled. PC samples put
    5.3% (256 B) and 6.5% (4352 B) of E0's overlap samples on those lines, about 6.5 cycles per combined packet at
    256 B. That is consistent with the loss but does not isolate it.
  - **P-B's share shift at 4352 B:** the two worker visits compete for the TXQ by timing (`send` returns at
    `txq_busy`), so a VC2 visit that reaches its TXQ check a few cycles later finds the TXQ free more often. The
    isolating A/B is a fixed delay before the lead's VC2 visit (0/4/8/16 cycles) on vc02ew4352 and vc02bew4352. Not
    run.
  - **VC0's gain with VC2 compiled in** (-2.8 / -3.3 at 4352 B): VC0's spilled slot and far-ring values return to
    registers. Both probes carry every VC0 value in registers.
- **The receiver's estimate method does not carry over to contended sender rows.** It counts fields copied at about
  1.9 cycles each. On these rows it predicted the wrong sign for P-B on both VCs and a cost for P-C that did not
  appear: a later VC's visit cost shows up as a VC0/VC2 share change and, at small packets, a combined loss.
  Contended rows need per-VC rates measured.

**Register maps** (loopvars.py --live --role sender on the sampled E0 image: 2D + VC2, the vc02 rows' build, kernel
hash 12364934172383773789 in each run cache; loopvars_maps/sender_budget/txp_live_<router>.txt; the row reports
txp_report_ab.txt, txp_report_cd.txt and the overlap-phase splitter txp_phase_lines.py beside them):

| | lead | P-B | P-C |
| --- | --- | --- | --- |
| pass loop (round loop) instructions | 596 (665) | 605 (669) | 583 (641) |
| carried at the pass head | 46: 15 registers, 31 slots | 46: 12 registers, 34 slots | 44: 13 registers, 31 slots |
| peak live | 55: 24 registers | 57: 20 registers, 37 slots | 53: 22 registers |
| VC0's send state | sent, far free slots, expected connection in registers; slot and far next slot on the stack | all in registers | all in registers |
| VC2's send state | sent, far free slots in registers; slot and far next slot on the stack | in memory by design (`later.vc2` in run's frame) | worker in memory by design; far free slots in a register, far next slot on the stack |
| VC1 builds (§7 inventory) | VC1's far next slot on the stack | VC1's far ring in memory by design | as the lead |

Across the 13 key cases (quick builds txpq_*): P-B leaves no hot value on the stack in any family; P-C spills VC2's far
next slot on 2D + VC2 and torus + VC2, as the lead does.

**PC samples** (job 429).
- Target: E0 of chip 1 chan 5, the busiest sender, overlap phase only: the traffic window's first VC0/VC2 fraction,
  bins with at least 30% of the peak send density.
- Sample counts: 486, 505 and 469 at 256 B (about ±2.5 cycles per entry); 254, 291 and 229 at 4352 B (too few to
  split).
- Cycles per combined packet at 256 B (share × combined rate). They split cycles within E0 only and do not name a
  bound.

| | lead | P-B | P-C |
| --- | ---: | ---: | ---: |
| VC0 worker visit | 41.4 | 32.3 | 35.2 |
| VC2 worker visit | 33.2 | 37.7 | 35.0 |
| pass body (group reads, connection semaphores, gates; P-B: the far ring's copy and write-back) | 24.5 | 30.0 | 26.7 |
| rest (unmapped, round, forwarding gates, connection checks) | 18.5 | 22.0 | 20.1 |

**Calibration verdict.**
- **P-A (the lead):** the reference. It spills four VC0/VC2 values. With VC2 compiled in, its VC0 rows are 2.8 to 3.3
  slower at 4352 B than both probes (about 6 to 8 at 256 B, derived).
- **P-B:**
  - Gains: the cleanest maps (no hot value on the stack; flat per added VC) and the VC2-on VC0 gain.
  - Losses: E0-bound fan-in +3.8% combined at 256 / 1024 B; VC2 alone +3 to +4.7; VC1 inter-mesh 16 B +4 to +9.
  - At 4352 B, VC0 +15 under contention, with the share moving to VC2.
  - All of P-B's losses come from the far rings in memory (the P-B vs P-C pair).
- **P-C:** flat or better on every measured row: worst +1.4 (VC2 alone, 4352 B), best -3.3 (VC2-on ew4352). VC1 code is
  the lead's. It keeps the lead's one far-ring spill on 2D/torus + VC2, and every added VC's far ring adds 2 carried
  values (line VCs).

**Recommendation (the user picks): P-C**, written as the receiver's rule plus one clause:
- "The first worker channel's send state is carried in registers; a later worker channel's state stays in memory
  between its visits."
- "Every remote receiver's progress (far ring) is carried in registers."
- That is P-B for worker state, with the far rings kept as loop locals (only E0's workers have per-visit state worth
  moving; the far rings are where P-B lost).
- Readability: one sentence more than P-B, and no channel singled out.
- Watch item for the line VCs: each adds a far ring, about 2 values. The §7.4 headroom check guards it. If line VCs
  spill a hot far-ring value, measure P-B's far-ring placement again on those rows: it cost 3 to 9 cycles on
  latency-bound rows here.
- Separate from the pick: the VC0/VC2 split on one link moved 5% of the packets with a few cycles of VC2 visit timing.
  Every placement is timing-sensitive there (memory: robust over peak). The delay-sweep A/B above would test the TXQ
  hypothesis before any fairness design.

**The user picked P-B (2026-11-02).** §8.4 measures how much packing its far rings costs.

### 8.4 P-B with each far ring packed into one word (`v4txpbk`; study only, no adoption build)

**The variant.** `v4txpbk` is v4txpb (P-B) with every VC's far ring (to be renamed
`RemoteReceiverChannelProgress {free_slots, next_slot_addr}`) held as one `uint32_t`:
- the first VC's loop-local copy is one word;
- a later VC's copy-in is one load, and its write-back is one compare and one store.

Changed: router_sender.hpp only:
- `FarRing` becomes `{word}` with `free_slots()`, `next_slot()` and `add_free_slots(n)`;
- new `far_ring_word`;
- new `take_next_slot<RX>`, which replaces the send's advance and its `free_slots--`;
- `write_back_changed` compares and stores the one word;
- static_asserts in `FarRingTraits`.

**Encoding** (chosen before any build):
- **Low 16 bits: free slots. High 16 bits: the next slot's address / 16.**
  - Completions stay a plain add.
  - A send takes its slot with the same add that advances the next slot: `advance = ((buffer size / 16) << 16) - 1`.
  - The wrap test compares the whole word against a constant: the next slot is the last one exactly when the word is
    at least `(last slot / 16) << 16`, because free slots < 65536 and slots are at least 16 B apart. No field is
    extracted.
  - The wrap adds a second constant (`first - last` in the high half, -1 in the low, modulo 2^32).
  - The opposite order (next slot low, free slots high) would make the room test a plain compare. But completions
    would need a shift (+1), the address two instructions, and the wrap test an extraction.
- **Address / 16, not a slot index.**
  - Every slot address is 16 B aligned, so the address is one shift.
  - An index needs a multiply by the buffer size or a table in front of the send. The receiver's lesson (§7.3): an
    address derived from an index in front of the header load cost +4.
- **The address in one instruction: `word >> 12`.** The free count is below 4096, so the low half's top 4 bits are
  zero and `word >> 12` equals `(word >> 16) * 16`.
- **The room test** masks the low half with `zext.h` (the target, `-mcpu=tt-bh`, has Zbb), then compares.
- **static_asserts per far ring:**
  - `first_slot % 16 == 0` and `channel_buffer_size % 16 == 0`;
  - `last_slot / 16 < 2^16` (L1 below 1 MiB);
  - `num_slots < 4096`.

**Per-operation instruction estimate against P-B** (written before the builds and runs):

| operation | where | P-B | packed | delta |
| --- | --- | --- | --- | ---: |
| room test (`free_slots >= needed`) | every visit with a packet, every VC | compare/branch on the register | `zext.h`, then compare/branch | +1 |
| address for the send | every send | the register | `srli 12` | +1 |
| advance and wrap, take one free slot | every send | compare with the last slot, select, add the buffer size; separate `addi -1` | compare the word with a constant, select one of two constants, one add | -1 (0 if the second constant is not hoisted) |
| completions | visits with completions | add | add | 0 |
| later VC's copy-in | every gated visit of VC1 or VC2 | 2 loads | 1 load | -1 |
| later VC's write-back | gated visits that change the ring | up to 2 compares and 2 stores | 1 compare, 1 store | -1 (completions only) to -2 (a send: both fields change) |
| carried registers | first VC (and any register-kept ring) | 2 per far ring | 1 | -1 register per ring |

- Net, per packet:
  - the first VC's send: about +1 (room test +1, address +1, the decrement folded into the advance -1);
  - a later VC's visit with a send: about -2;
  - the pass loop of the 2D + VC2 image: about +5 to +10 instructions (each send instance and room test +1, the VC2
    copy -3).
- Cycles: ALU instructions are about 1 cycle each on the in-order ERISC. The receiver's packing estimates missed by 2
  to 4x (§7.3: B2-packed +7..+12, P2-packed +17..+25), but those inserted and extracted fields on hot counters; this
  packing only adds and extracts.

**Predictions** (cycles/packet against P-B; written before the device runs):

| row | P-B (§8.3) | packed vs P-B | reasoning |
| --- | ---: | --- | --- |
| ew16, bew1024, ew4352 (2D, VC2 off; E0-bound) | 115.8, 116.5, 126.9 | +1 to +4. **WRONG: +0.7, +0.4, +0.1** | only the first VC: room test and address +2, decrement -1; the receiver's 2-4x miss in the range |
| ew4352, one ERISC (RX half) | 198.3 | 0 (measured -0.1) | the record loads and stores one word fewer per send; hidden behind the RX half |
| VC2 on, VC2 idle: ew4352 (E0) | 131.8 | +1 to +4. **WRONG: -0.6** | as the 2D rows; an idle VC2 is never copied |
| VC2 on: torusns (both) | 167.2 | 0 to +3 (measured -0.2) | |
| VC2 alone, v8 (window-bound) | 228.2, 229.5, 245.5 | -1 to -3. **WRONG: -3.9, -3.5, -3.6** | the VC2 visit's copy and write-back shrink by 2 to 3 instructions on the credit loop's path (P-B paid +3..+4.7 for the far ring in memory) |
| vc02: VC0 sender, VC2 busy | 216.8, 215.7, 205.2 | 0 to +3. **WRONG: -12.0, -11.5, -13.9** | VC0 +1 per packet; shorter VC2 visits |
| vc02bew: VC2 sender, VC0 busy | 279.1, 282.1, 324.4 | -1 to -4 at 256 / 1024 B. **WRONG: +2.2, +1.2**; 4352 B not predicted (+42.0) | 4352 B: the VC0/VC2 share moves with timing (§8.3) |
| combined at 256 / 1024 B (E0-bound) | 122.0, 122.2 | -1 to +1. **WRONG: -3.5, -3.6** | per VC0 + VC2 packet pair, about -1 instruction |
| combined at 4352 B | 125.7 | 0 (measured 0.0) | pinned (link or far side) |
| VC1 split: 1-hop and burst (VC0 traffic, E0-bound) | 141.9-142.3 | +1 to +4. **WRONG: +0.5 to +1.0** | the first VC pays |
| VC1 split: 13-row mean | 159.1 | 0 to +2 (measured +0.2) | |
| VC1Inter 1_0 (RX-bound) | 208.3, 209.1 | 0 (measured 0.0) | |
| VC1Inter 1_1..1_3 at 16 B | 354.9, 351.7, 379.3 | -1 to -4. **WRONG: -7.7, -6.0, +2.4** | VC1's copy and write-back shrink by 2 to 3 per visit (P-B paid +4.2..+8.8 for VC1's far ring in memory) |
| VC1Inter 1_1..1_3 at 4352 B | 397.5, 396.1, 414.9 | 0 to -2. **WRONG: +3.4, +1.1, -3.1** | |

Register map (same 2D + VC2 E0 image as §8.3): carried registers 12 → 11 (VC0's far ring in one); VC2's far ring a
single word in `later.vc2`; peak live unchanged ±2; no new hot value on the stack. **Partly WRONG (static check below
and the device image): 10 registers, peak -5.**

**Static check (offline, before the device runs).**
- Build: quick build `txpq_v4txpbk_O3`, 13 key cases, 534 of 534 images. Every static_assert holds.
- Image: the 2D + VC2 E0 image `bh_2d_vc2__10740105522294516597`, P-B / packed, by instruction count per inline visit
  (loopvars_maps/sender_budget/txp_linecount.py) and `loopvars.py`.

| | P-B | packed | estimate |
| --- | ---: | ---: | --- |
| pass loop (round loop) instructions | 605 (669) | 589 (650) | +5 to +10. **WRONG: -16** |
| `visit_worker<0>` (VC0's worker visit with its send) | 88 | 83 | about +1. **WRONG: -5** |
| VC0's far-ring advance with the free-slot decrement | 10 + 1 | 7 (`take_next_slot`) + 1 (`>> 12`) | -1. **WRONG: -3** (P-B materializes the last, first and size constants at the advance) |
| room test | 3 | 3 (`zext.h` included) | +1. **WRONG: 0** |
| `visit_kept_worker<4>` (VC2's worker visit) | 107 | 98 | |
| pass body (the far ring's copy-in and write-back) | 72 | 67 | -2 to -3 per visit (-5 in code) |
| forwarding visits (1, 2, 3) | 111, 111, 112 | 111, 111, 112 | |
| sp accesses in the loop | 99 (70 loads, 29 stores) | 84 (60, 24) | |
| carried at the pass head | 46: 12 registers, 34 slots | 42: 10 registers, 32 slots | 11 registers. **WRONG: 10** |
| peak live | 57 (20 registers, 37 slots) | 52 (20 registers, 32 slots) | ±2. **WRONG: -5** |
| VC2's far-ring copy | `pass:far2.free_slots` spilled (2 stores, 3 loads) | `pass:far2.word` in a register | |

The device predictions above stay as written. The static count suggests the rows come out at or below P-B, not
above.

**Device results (2026-11-02; jobs 445 and 446).**
- Runs: txp_runs.sh. Job 445: passes e (rows) and g (`TXP_SET=vc2`), order P-B, packed. Job 446: passes f and h,
  order packed, P-B.
- About 23 minutes of device time (631 s and 723 s); no hangs; every test passed; the VC1 split's 50 finishes in both passes.
- The passes agree within 0.4, except vc02bew1024's VC2 sender (1.2).
- P-B's rows reproduce §8.3 (jobs 422/425/432) within 0.3, so the lead and P-C columns from §8.3 are comparable.
- Reports: loopvars_maps/sender_budget/txp_report_ef.txt and txp_report_gh.txt. Cycles/packet, mean of the two
  passes.

| row | lead (§8.3) | P-C (§8.3) | P-B | packed | packed vs P-B |
| --- | ---: | ---: | ---: | ---: | ---: |
| ew16 (2D, E0-bound) | 115.8 | 115.8 | 115.8 | 116.5 | +0.7 |
| bew1024 | 116.5 | 116.5 | 116.5 | 116.8 | +0.4 |
| ew4352 | 126.9 | 126.9 | 126.9 | 127.0 | +0.1 |
| ew4352, one ERISC | 198.3 | 198.3 | 198.3 | 198.2 | -0.1 |
| VC2 on, VC2 idle: ew4352 | 134.6 | 131.4 | 131.8 | 131.2 | -0.6 |
| VC2 on: torusns | 166.9 | 167.1 | 167.1 | 166.9 | -0.2 |
| VC2 alone v8: 256 B | 223.7 | 224.6 | 228.2 | 224.2 | -3.9 |
| VC2 alone v8: 1024 B | 226.5 | 226.4 | 229.5 | 226.0 | -3.5 |
| VC2 alone v8: 4352 B | 240.8 | 242.3 | 245.5 | 241.9 | -3.6 |
| vc02bew VC0 sender (about 90% VC0 alone, VC2 compiled in): 256, 1024 B | 139.0, 139.0 | 133.3, 133.4 | 132.2, 132.3 | 134.6, 134.6 | +2.4, +2.3 |
| vc02bew VC0 sender: 4352 B | 139.9 | 137.3 | 137.8 | 137.0 | -0.8 |
| VC1 split: 1-hop and burst (8 rows) | 141.9-142.2 | 141.9-142.2 | 141.9-142.3 | 142.6-143.1 | +0.5 to +1.0 |
| VC1 split: 2-hop merge 16 B, 4352 B | 139.3, 144.0 | 139.3, 144.0 | 139.6, 144.2 | 138.5, 143.5 | -1.1, -0.7 |
| VC1 split: mean of 13 | 159.2 | 159.2 | 159.2 | 159.4 | +0.2 |
| VC1Inter 1_0 (16 B, 4352 B) | 208.3, 209.1 | 208.3, 209.1 | 208.3, 209.1 | 208.3, 209.1 | 0.0 |
| VC1Inter 1_1, 1_2, 1_3 at 16 B | 346.1, 346.1, 375.1 | 346.3, 346.2, 374.8 | 354.9, 351.6, 379.6 | 347.2, 345.7, 382.0 | -7.7, -6.0, +2.4 |
| VC1Inter 1_1, 1_2, 1_3 at 4352 B | 397.1, 395.5, 412.4 | 397.1, 395.5, 412.4 | 397.6, 396.1, 414.7 | 400.9, 397.2, 411.6 | +3.4, +1.1, -3.1 |
| VC1Inter mean of 8 | 336.2 | 336.2 | 339.0 | 337.8 | -1.2 |

**Fan-in at v8, per VC and combined** (as §8.3: VC0's contended rate from vc02ew, VC2's from vc02bew, combined =
1/(1/VC0 + 1/VC2)):

| size | quantity | lead (§8.3) | P-C (§8.3) | P-B | packed | packed vs P-B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | VC0 sender, VC2 busy | 201.9 | 200.2 | 216.8 | 204.8 | -12.0 |
| 256 | VC2 sender, VC0 busy | 281.7 | 281.8 | 279.1 | 281.2 | +2.2 |
| 256 | combined | 117.6 | 117.0 | 122.0 | 118.5 | -3.5 |
| 256 | VC0's share | 58% | 58% | 56% | 58% | |
| 1024 | VC0 / VC2 | 201.3 / 283.7 | 199.8 / 283.5 | 215.7 / 282.6 | 204.2 / 283.8 | -11.5 / +1.2 |
| 1024 | combined | 117.8 | 117.2 | 122.3 | 118.8 | -3.6 |
| 4352 | VC0 / VC2 | 190.4 / 369.9 | 190.1 / 371.1 | 205.2 / 324.5 | 191.3 / 366.5 | -13.9 / +42.0 |
| 4352 | combined | 125.7 | 125.7 | 125.7 | 125.7 | 0.0 |
| 4352 | VC0's share | 66% | 66% | 61% | 66% | |

- Derived from vc02bew (not measured): VC0 alone with VC2 compiled in at 256 and 1024 B is about 129.4 packed, against
  126.4 for P-B (+3.0), 128.3 for P-C and 134.3 for the lead.

**Register map** (`loopvars.py --live --role sender` on the device image, kernel hash 12364934172383773789 in each
run cache: loopvars_maps/sender_budget/txpk_live_v4txpb.txt and txpk_live_v4txpbk.txt; the same numbers as the quick
build above).

| | P-B | packed |
| --- | --- | --- |
| pass loop (round loop) instructions | 605 (669) | 589 (650) |
| carried at the pass head | 46: 12 registers, 34 slots | 42: 10 registers, 32 slots |
| peak live | 57: 20 registers, 37 slots | 52: 20 registers, 32 slots |
| VC0's far ring | 2 registers (free slots, next slot) | 1 register (`far.vc0.word`) |
| VC2's far ring | `later.vc2.far` (2 slots); its in-visit copy's free slots spilled | `later.vc2.far.word` (1 slot); its in-visit copy in a register |
| VC0's worker state | registers | registers |
| VC1 image (quick build `bh_vc1__10354273653968310346`) | 828 (894); 14 registers carried; peak 66; VC1's far ring 2 slots | 819 (888); 13 registers; peak 62; 1 slot |

**What the pair isolates.**
- Only the far ring's representation changes, for every VC at once: VC0's register copy and the later VCs' memory
  copies. So the deltas belong to the packing as a whole, not to one VC's ring.
- Mechanisms (hypotheses):
  - **Multi-VC rows recover:** VC2 alone -3.5..-3.9, fan-in combined -3.5, VC0 under contention -12, and the 4352 B
    share back to the lead's. The likely cause is the later VC's visit getting shorter: one load and one compare-store
    instead of two of each, and the in-visit copy no longer spilled. Static evidence: VC2's visit 107 → 98
    instructions, pass body 72 → 67, sp accesses 99 → 84.
  - **The first VC pays a little:** +0.1..+0.7 on the 2D rows, +0.5..+1.0 on the VC1 1-hop and burst rows, about +2.4
    for VC0 with VC2 compiled in at small packets. Statically its visit is 5 instructions shorter. Possible causes:
    - the extra `zext.h` and `srli` sit on the dependency chain in front of the room test's branch and the TXQ
      command;
    - a layout or stall move.

    Not isolated: a variant packing only the later VCs' rings (VC0's ring kept as two registers) would separate the two.
  - **The window-bound VC1 inter-mesh rows move both ways** (16 B: -7.7, -6.0, +2.4; 4352 B: +3.4, +1.1, -3.1;
    mean -1.2). This is timing on the credit loop, as on every placement so far (§8.3); I make no cause claim.

**How much worse packing gets.**
- **Single-VC routers (2D, 1D-like E0-bound rows):** +0.1 to +0.7.
- **VC1 routers with VC1 idle:** +0.5 to +1.0.
- **VC2 compiled in, VC2 idle:** -0.6 at 4352 B and about +2.4 at 256/1024 B (vc02bew VC0, about 90% VC0 alone).
- **Every row with a later VC's traffic is better than unpacked P-B:**
  - VC2 alone -3.5..-3.9;
  - fan-in combined -3.5 / -3.6 / 0.0;
  - VC0 under contention -11.5..-13.9.
- **Against the lead and P-C on those rows:**
  - fan-in combined +0.9 / +1.0 / 0.0 vs the lead, and +1.5 / +1.6 / 0.0 vs P-C;
  - VC2 alone +0.5 / -0.5 / +1.1 vs the lead.
- **VC1 inter-mesh mean:** 337.8 (lead 336.2, P-B 339.0).
- **Net:** packing does not make P-B worse on multi-VC traffic; it removes most of P-B's far-ring-in-memory cost
  (§8.3). The price is up to about +1 per packet on the first VC alone (+2.4 with VC2 compiled in at small packets).

**The estimate method, again:** the per-operation counts missed in sign:
- the advance was cheaper packed;
- the room test cost nothing extra;
- the copy savings showed up 1.5-4x larger on the device than counted.

Both probes (§8.3, §8.4) say the same thing: predictions for this loop need the static build (instruction counts per
visit, sp accesses, carried registers) before any per-row estimate.

## 9. Codegen discipline per stage (revision 4; the receiver's causes, generalized) [r4]

| cause the receiver hit (V4_RECEIVER_OWNERSHIP §15-§16) | where it can bite the sender | rule for the sender pass |
| --- | --- | --- |
| a field of memory state re-read inside a hot function changed the -O3 loop (`credit_completions` read `progress.next_index`) | `visit_forwarding` reads `r.acked` at L:527, 530, 532 and 536, and `r.slot` / `r.sent` around `send` (v4rxf line numbers) | each memory field is copied once where first needed, passed by value, and written back once where it changes |
| a by-value helper initialising the loop's copy changed -O3 instrumentation loops and -Os allocation (`first_vc_progress_copy`) | `far_rings_start()`, `workers_start()` and `worker_start()` return state by value; `worker_connections()` copies the interfaces by value | initialise from references (one accessor); constexpr starts per channel and per VC |
| flat vs nested struct moved one load (RingProgress; access-path types, in RTL) | `SenderChannelState {progress; ...}` nested vs flat | decide the nesting once (Q4); verify it in X1; a classified move is accepted only by the user |
| a larger zero-initialised object took a `memset` call at -Os, and -Os re-allocated the loop | `SenderChannelStates` vs the lead's smaller per-role objects | check -Os at X1 and X7; member initialisers matching today's init stores |
| a dead field in an object kept in memory grew the frame and moved -Os allocation (v4rxe1: `states.ch0.cursor`) | FIRST_VC's state inside `SenderChannelStates` while the loop works on its copy (P-B) | the accepted layout-only shape; classify (a)/(b)/(c) |
| an empty member permuted VC1 stack slots (V4_SENDER_SHAPE §4) | `Present<>` members | `[[no_unique_address]]` throughout; classify |
| instrumentation builds (codeprof, allinstr) differ when production does not | telemetry and capture hooks in `send` | compare every case; a difference only there is reported, not waived |
| the codec's uniform volatile decode cost +2.2 and was accepted for one stated rule | the sender's header reads (payload size, `src_ch_id` stamp) use the header's own accessors | no per-path plain/volatile choices |

**Per-stage protocol:**
- Offline builds go through coverage_txo.py with the non-fabric files pinned (the swap guard), at both levels, on every
  case.
- Classify with layoutcmp.py (with RENAMES for renamed symbols).
- Any difference outside (a)/(b)/(c) gets single-change A/Bs before the next stage.
- Record the sender headroom check at X1 (the base) and run it from X3 on.
- Log size_report.py in V4_SIZE.md.
- Device A/B only for the stages marked so.

## 10. Style: the receiver lead vs the sender's planned shape (revision 4) [r4]

| item | receiver (v4rxf) | sender (lead / revision 3) | revision 4 |
| --- | --- | --- | --- |
| type names | `ReceiverChannel<VC>`, `ReceiverChannelState<VC>`, `RingProgress` | `Channel<CH>`, `Vc<V>`, `ChannelState<CH>`, `FarRing`; rev 3 `RemoteReceiver` | §1.0 |
| aliases | full type names | `using C`, `FarT`, `W` | full names |
| short names | `my_chip` (was `me`) | `w`, `r`, `far`, `src` | §1.0 |
| the pass | three explicit `step_channel<VC>` calls + `static_assert(MAX_NUM_VCS == 3)` | explicit `visit_vc<0..2>` (lead); recursion (rev 3) | explicit + static_assert |
| grouped reads | per-VC values in a braced `std::array` (0 where a VC does not run) | per-channel recursion into `registers[CH]` | the recursion stays (a VC has several channels), with the read-group primitives |
| placement note | one rule comment on `ReceiverChannelStates`, pointing to the doc and the headroom check | a State paragraph with cycle numbers | one rule comment (§3 #9) |
| constants | UPPER_SNAKE (`IDLE_PASSES_BEFORE_PARTIAL_CREDIT`, `LOCAL_NOC`, `INLINE_WRITE_ALL_BYTES`, `FIRST_VC`) | `txq`, `0xF` / `0xf` literals | §1.0 |
| shared constants in a role file | `INLINE_WRITE_ALL_BYTES` in router_downstream.hpp, `FIRST_VC` in router_receiver.hpp | - | both to router_common.hpp |
| file header | the steps, "Channels and VCs", "State, by placement class", a sections list | a per-pass list and a state list with measurements | the receiver's layout |
| `Each` | non-const `get<I>()` | const `of<I>()` used by const readers | Q7 |
| VC-number names | `vc0_register_empty` (telemetry) | lead: `vc0_channels`, `vc0_forwarding`, `vc1_forwarding`, `vc2_worker`, `has_vc2_worker`, `sends_vc0..2` | sender: gone with `SenderVc<VC>`. `vc0_register_empty` is noted for the receiver's next pass |
| start values | constexpr `progress_start<VC>()` | by-value aggregates | constexpr per channel / VC |

## 11. 1D line VCs: what touches the sender (LINE_VCS_HOST_DESIGN.md §9, read-only) (revision 4) [r4]

| need | sender part | in this pass? |
| --- | --- | --- |
| a worker channel per line VC (the first channel of each LINE VC) | §2 #4 with the corrected host rule; at most one worker per VC | yes, X2 |
| a third producer-credit buffer (RD) for a 1D VC1 worker | §2 #7 | the rule in X4; used when line VCs come up |
| first-level acks per VC on the sender (a 1D VC1 ring acks) | `is_vc_first_level_ack_enabled[vc]` (in v4rxf); the gate's `!ACK_VC1` moves to the host's table | the read is done; the gate moves with bring-up |
| bubble flow control per line VC | `SENDER_CH_<i>_IS_INJECTION` from the host's `bubble_flow_control_enabled_on_vc(VcRouting)` | already host-read |
| free-slots streams per line VC's channels | `sender_free_slots_stream<CH>` by flat position (the host's `sender_flat_base`) | already host-read |
| up to three worker channels and three remote receivers on one 1D router | placement (§8) | the rule must hold them: P-B carries 0 extra registers per line VC; P-A about +7 values per worker |
| the gate (`!has_vc1 \|\| is_2d`, `vc2_config_supported`'s `is_2d`) | fabric_erisc_router_v4.cpp | with bring-up (the host's table is the check) |
| the 1D downstream branches, `VC<n>_SERVICED`, `VC<n>_ROUTES_AS_LINE`, `RX_CH_<r>_FORWARDS_ON_VC` | receiver and router_common | the receiver's pass |

## User decisions on the open questions (2026-11-02)

All seven remaining questions are answered as recommended; Q2 (worker role: option (a), host arg
SENDER_CH_<i>_IS_WORKER_CHANNEL, in the host since job 399) and Q9 (-O3 for v4's VC1 routers, in the host since job
399) were decided earlier.
1. Producer-credit command buffers come from the host's SENDER_CH_<i>_ACK_CMD_BUF_ID: channel 0 → WR, VC2's channel →
   WR_REG, every other channel AT (independent of trimming).
2. Names: "remote receiver" for the far ring; the library's "read counter" (workers) and "free slots" (upstream
   routers) for producer credits.
3. Bubble flow control's 2: one shared host-device constant.
4. The counter address helper lives in fabric_router_flow_control.hpp, the layout's owner.
5. Host leftovers (runtime-arg connection semaphores, NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR): clean up once v4
   is the only router.
6. Stages A, B and C are kept (fixed-location worker interfaces, per-worker credit buffers, one per-packet hook).
7. `Each` moves to router_common.hpp beside `Present`.
Start: once the receiver ownership pass lands (its S3 placement decision is pending), on top of its final variant.
Batching stays parked until the slot decisions (V4_SEND_BATCH.md); P3 is dropped.

## Planned: one copy of the cold worker-connection path (user, 2026-11-02)

Measured on the lead (-O3, share of .text instructions): init and control code copied per VC or channel is 11-18% of
each image. On E0 that is mostly `worker_connection_event<CH, WorkerInterface>`, ~95 instructions per worker channel,
plus the worker credit's cold part, ~25 per worker. Adding VC2 costs E0 +126 of these.
- What forces a copy per channel: the interface type StaticSizedSenderChannelWorkerInterface<NOC, NUM_BUFFERS>
  (CRTP, templated on the slot count), and the VC0-only credit preset (`if constexpr (CH == 0)`).
- Plan, folded into this pass next to X3:
  - move the slot-count-independent connection state into a non-templated part of the interface (shared library
    file; main's router unchanged);
  - make the preset take the channel's credit command buffer (a host fact after X3) as an argument;
  - one noinline `worker_connection_event` for every worker channel, behind thin per-channel wrappers.
- Cost: none per packet (connect-time path). Checks: byte comparison of the loop plus the register-map guard
  (loopvars.py), since a cold call's signature can shift the caller's allocation.

## User decision: packed P-B (2026-11-02)

The sender pass uses P-B with every VC's remote receiver progress packed into one word, as §8.4's `v4txpbk` encodes
it (low 16 bits free slots, high 16 bits next slot address / 16; no index, no multiply). The user asked for a proper
type with operations rather than bit masking at the call sites: `RemoteReceiverChannelProgress<VC>` (keyed by VC so
its slot constants are its own) with `has_room(n)`, `next_slot_addr()`, `take_slot()` and `add_free_slots(n)`; the
encoding, its constants and its static_asserts live in the type.

Safety of the encoding (2026-11-02, for the sender pass):
- Alignment: the host rejects payload sizes that are not L1-aligned (16 B on BH, fabric_context.cpp
  validate_and_apply_packet_size) and places channel buffers at eth_word_l1_alignment = 16
  (erisc_datamover_builder.hpp:153); the type static_asserts first slot and buffer size % 16.
- Range: the remote ring is always in the far ERISC's L1 (BH MEM_ETH_SIZE 512 KiB), so address / 16 <= 0x7FFF, half the
  field; static_assert on the last slot. Not valid for Tensix L1 (1.5 MiB): a Tensix downstream must not reuse it.
- The free-slot half never borrows or carries into the address half: slots < 4096 (static_assert; also what `>> 12`
  needs), every take follows a room test, completions never exceed the slots outstanding. These are protocol
  invariants: add ASSERTs in take_slot() and add_free_slots() (compiled out unless asserts are on).
- Slot addresses are compile-time today. If they ever become runtime args, the bounds need an init-time check.
- The sender pass's full coverage build (every config, trim captures and slot-size variants included) compiles the
  static_asserts on every config.

## 12. The sender ownership pass on v4rxg3 (2026-11-02)

The base is the receiver lead v4rxg3. The placement is packed P-B, with `loop_local_vc` derived and owned by
`SenderChannelStates`. Names follow the channel kind (§1.0). Stages follow §4, one change each.

### X-1: the host's worker-role rule (main tree; job 448)

- **Rule:** `SENDER_CH_<i>_IS_WORKER_CHANNEL` marks the first sender channel of each VC whose routing mode has a
  worker:
  - VC0: its worker-connected channel (the host's convention, channel 0);
  - VC1 (inter-mesh): none;
  - VC2 (neighbour exchange): its first channel.

  Before, it marked every VC2 channel. The live-check skip (`should_force_internal_sender_skip`) calls the same
  `is_worker_channel`, so it follows.
- **Change:** tt_metal/fabric/erisc_datamover_builder.cpp, the `is_worker_channel` lambda and its comment.
  - Patch pinned at fabric_exp/variants/host_rebuild_20261102/1_worker_role_first_channel.patch.
  - Applied and libtt_metal rebuilt by host_rebuild_20261102.sh in broker job 448 (CCACHE_DISABLE=1, 43 s).
  - Backup: ~/.tt-metal-host-rebuild-backup/erisc_datamover_builder.cpp.pre20261102.
- **The args are identical on every current config** (VC2 has at most one sender channel outside line mode):
  - Every captured coverage config, 5,831 named-arg headers in cov_m's 81 cached cases plus the 876 of the trim-gtest
    caches, has `ACTUAL_VC2_SENDER_CHANNELS` 0 or 1.
  - The 262 configs whose capture already carries `IS_WORKER_CHANNEL` (the four VC0-receiver-trimmed cases) match the
    new rule on every channel.
  - The 2,480 argdump records all have VC2 count 0.
  - Before and after the rebuild, the host's router args were dumped on the mock (argdump's `TT_FABRIC_EXP_DUMP_ARGS`
    hook, cases bh_2d, bh_2d_vc2, bh_neighbor; 80 router ERISCs each). CT args, named CT args and defines are
    identical on all 240. bh_2d_vc2 sets the flag on channels 0 and 4 both times.
- No device run: no router config's args change.

### The sender headroom check (built before X0; §7.4)

- **Command:** `loopvars.py --headroom BASE NEW --role sender` (images or builds).
  - Sender images: two-ERISC E0 `active_erisc` images (`sender::run`) and one-ERISC `subordinate_active_erisc` images
    (`single::sender_pass`).
  - Not hot (per role, `DESIGNED_MEMORY` / `INVARIANT`): the channel states and forwarding records kept in memory,
    the round counters, the worker interfaces and their copies, the upstream routers' addresses, and saves next to an
    out-of-line call.
  - The receiver path is unchanged, and the self-test passes.
- **Overlays:** `loopvars_maps/sender_budget/synth_overlays_tx.py` adds synthetic state, declared in `sender::run`
  and updated after each `pass()`:
  - `tx1` / `tx2`: one or two carried values;
  - `twk`: a five-value worker state;
  - `trr`: a two-value remote receiver progress.
- **Runner:** `loopvars_maps/sender_pass/hr_tx.sh BASE` builds the four overlays at -O3 on the families and compares
  each with BASE's full build. Families: bh_1d, bh_ring, bh_2d, bh_torus, bh_2d_vc2, bh_torus_vc2, bh_vc1,
  glx_express, T1 (bh_2d_vc2_trim_w0), T5 (bh_vc1_trim_w0). The base run on X0 is the recorded known answer.

### X0: names and the one-block regroup (v4txo0)

- **Content:** the §1.0 names on the lead's structure, in v4txo0:
  - `SenderChannel<CH>` with `first_slot_addr`, `last_slot_addr`, `completion_counter_addr`,
    `first_level_ack_counter_addr`, `connection_semaphore_addr` and `remote_free_slots_to_send`;
  - `RemoteReceiverChannel<VC>`, keyed by VC, with `rx`;
  - `RemoteReceiverChannelProgress {free_slots, next_slot_addr}`;
  - the fields `next_slot_addr`, `first_level_acked`, `connection_state` and `upstream_free_slots_addr`;
  - `forwarding_in_flight`, `holding`, `connection_semaphores`, and `send_yielded` (`RoundCounters`,
    router_round.hpp);
  - `SENDER_TXQ` and `WORKER_CREDIT_NOC`; `INLINE_WRITE_ALL_BYTES` moved to router_common.hpp;
  - no short names or type aliases.
- **Read groups:** the four read groups use `begin_read_group()` / `end_read_group()`. Send's two other fences stay
  until X0f.
- **Regroup:** the receiver's file layout, and one block, "Where the channel states live, and the loop", above
  `pass()` and `run()`. Moves only.
- **Scripts:** `loopvars_maps/sender_pass/stage_x0_names.py` and `stage_x0_regroup.py`.
- **Identity:** `bc_tx.sh` (coverage_txo.py, every case, swapguard before and after): 2,863 of 2,863 built configs
  byte-identical to v4rxg3 at -O3 and at -Os (3,124 configs, ok-sets equal; `cmp_v4txo0_{O3,Os}.txt`). Size: equal
  to v4rxg3.

### Sender headroom on X0 (the recorded known answers; `loopvars_maps/sender_pass/headroom_v4txo0.txt`)

Cells: configs passing (no new hot value on the stack), per family and overlay.

| family | tx1 | tx2 | twk | trr |
| --- | --- | --- | --- | --- |
| 1D | 10/10 | 10/10 | 0/10 | 10/10 |
| 2D | 28/28 | 28/28 | 0/28 | 28/28 |
| 2D + VC2 | 28/28 | 0/28 (VC0 worker `sent`) | 0/28 | 0/28 (VC0 worker `sent`) |
| T1 (VC0 worker trimmed, VC2) | 0/28 (VC2 far next slot) | 0/28 | 0/28 | 0/28 |
| ring | 10/10 | 0/10 (VC0 far next slot) | 0/10 | 0/10 |
| torus | 12/12 | 0/12 (VC0 worker slot, send's header and slot address) | 0/12 | 0/12 |
| torus + VC2 | 16/28 | 0/28 | 0/28 | 0/28 |
| VC1 | 22/22 | 22/22 | 0/22 | 22/22 |
| T5 (VC1, VC0 worker trimmed) | 22/22 | 22/22 | 22/22 | 22/22 |
| Galaxy express | 28/28 | 0/28 (VC0 worker slot, send's header and slot address) | 0/28 | 0/28 |

A worker-sized addition (`twk`, a line VC's worker) spills a hot value on every family except T5, so a line VC's worker
does not fit in registers next to today's loop-local state.

### X1: structure, under today's rule (v4txo1)

- **Content:**
  - features from the serviced set: `SenderVc<VC>` (`sender_channels`, `worker_channels`, `forwarding_channels`,
    `first_level_ack`, `reads_free_slots_at_pass_start`) and `SenderChannel<CH>::worker`;
  - the masks `sender_channels`, `worker_channels`, `forwarding_channels`, `sender_vcs` and
    `channels_read_at_pass_start`;
  - `SenderChannelState<CH> {SenderChannelProgress progress; connection_state | upstream_free_slots_addr}`, layout
    as the lead's records;
  - `SenderVcState<VC> {remote; worker}`, and `Each` containers (with a const `get`);
  - `SenderChannelStates`, which owns the rule (today's: every VC's send state loop-local) through
    `loop_local_progress()`, `progress_for<VC>(loop_copy)` and `after_visit<VC>()`;
  - `visit_vc<0..2>`, `check_worker_connection<0..2>`, and one producer-credit path (`return_slots<CH>`).
  - Script: `stage_x1.py`, with the new sections in `loopvars_maps/sender_pass/x1/`.
- **Static, full coverage against X0:**
  - Every E1 image is identical. 1,695 E0 and one-ERISC images differ (2,380 role analyses; `guard_x0_x1_{O3,Os}.txt`,
    renamed spills normalized).
  - -O3 loop deltas:
    - 1D -2 (stack S-1 L-1);
    - ring -2 (the forwarding visits' `reported_acks` spill is gone);
    - torus +1 (S+1);
    - 2D + VC2 and torus + VC2 without acks +2, with the lead's four hot spills (VC0 and VC2 slot and far next slot)
      now named as two;
    - torus + VC2 with first-level acks (12 images) -10;
    - Galaxy express +1 or +2 (S+1): **a new spill, `return_slots<0>:returned`**. This is the absolute count handed to
      the out-of-line worker credit under first-level acks, stored next to that call; the forwarding visits'
      `reported_acks` spills are gone;
    - one ERISC -2 to -4;
    - 2D and VC1: +0, names only.
  - -Os: only Galaxy express changes (-2).
  - The device result for each, including the Galaxy express spill, is below (stage A/B).
- **Headroom on X1** (`headroom_v4txo1.txt`) against X0's cells:
  - better: torus + VC2 at tx1 (28/28 from 16/28), tx2 and trr (12/28 from 0/28); torus at tx2 (12/12 from 0/12);
    Galaxy express at tx2 (28/28 from 0/28);
  - worse: VC1 at tx2 and trr (0/22 from 22/22: VC0's worker slot and send's header and slot address move);
  - every other cell is unchanged.

### X2: host facts (v4txo2)

- **Content:**
  - the worker role from `SENDER_CH_<i>_IS_WORKER_CHANNEL` (ct array `is_sender_channel_worker`; X-1's corrected
    host rule);
  - per-VC ct arrays `first_sender_channel_by_vc`, `sender_channel_count_by_vc` and `receiver_channel_by_vc`;
  - `sender_channel_counter_addr()` in fabric_router_flow_control.hpp;
  - static_asserts: the channel uses counter credits; the host checks every worker channel's connection; one
    worker per VC.
  - Init asks the arg for a channel's role.
  - Script: `stage_x2.py`.
  - The shared bubble-flow-control constant (#6) goes with X4's host patch.
- **Identity:** byte-identical to X1 on all 2,863 built configs at -O3 and at -Os. So X2 is no arm of the device A/B,
  and its headroom is X1's.
- offline_build_txo.py's injection now uses the corrected rule (identical: no config has two VC2 channels).

### Stage A/B on the device (2026-11-02; jobs 451-456)

- **Arms:** one per codegen stage, in chain order: v4rxg3 (= X0), X1 (= X2: identical), X3 (P-B), X3k (packed
  remote receiver progress), X3a (A: interfaces at a fixed location, UpstreamCredits gone).
- **Jobs:** three row sets, each with every arm in one job (`txo_runs.sh`, `TXO_SET=core|census|vc1`):
  - pass s1 (jobs 451-453): arms in chain order;
  - pass s2 (jobs 454-456): reversed.
- The passes agree within 0.3 (VC2 contended rates within 0.8). Report: `txo_report.py --chain ...`,
  `loopvars_maps/sender_pass/txo_report_s1s2.txt`.

Cycles/packet, mean of the two passes. Each delta is the stage against its predecessor.

| row | v4rxg3 | X1 | X3 | X3k | X3a | X1 | X3 | X3k | X3a |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ew16 | 116.0 | 116.0 | 116.0 | 116.6 | 116.6 | 0.0 | 0.0 | +0.7 | 0.0 |
| bew1024 | 116.7 | 116.7 | 116.7 | 117.1 | 117.1 | 0.0 | 0.0 | +0.4 | 0.0 |
| ew4352 | 127.0 | 127.0 | 127.0 | 127.1 | 127.1 | 0.0 | 0.0 | +0.1 | 0.0 |
| ew4352, one ERISC | 200.0 | 200.0 | 204.8 | 201.8 | 201.8 | 0.0 | **+4.8** | -3.0 | 0.0 |
| lin1, ring1, torusns, ns4352, ewscatter | 124.4, 140.9, 149.2, 127.2, 142.3 | = | = | = | torusns -0.2 | 0.0 | 0.0 | ring +0.1 | ±0.2 |
| VC2 on: ew4352 / torusns | 134.7 / 166.9 | 134.8 / 167.0 | 132.2 / 167.1 | 132.1 / 167.2 | 132.2 / 167.1 | +0.1 | -2.5 / +0.1 | -0.1 | 0.0 |
| VC2 alone v8: 256, 1024, 4352 | 224.1, 227.0, 240.9 | = | 228.0, 229.3, 246.4 | 225.2, 228.0, 243.0 | 225.5, 228.6, 243.5 | 0.0 | +2.3..+5.6 | -1.3..-3.4 | +0.3..+0.6 |
| vc02ew VC0 sender (VC2 busy): 256, 1024, 4352 | 200.8, 200.4, 190.8 | = | 215.7, 214.9, 202.2 | 203.6, 203.1, 191.5 | 206.3, 205.7, 192.6 | +0.1 | +11.3..+14.8 | -10.6..-12.1 | +1.1..+2.7 |
| vc02bew VC2 sender (VC0 busy): 256, 1024, 4352 | 286.1, 287.3, 368.6 | 286.1, 288.4, 368.4 | 280.3, 282.9, 332.4 | 285.5, 287.9, 365.8 | 284.7, 287.3, 361.8 | 0.0 / +1.1 / -0.1 | -5.6..-36.0 | +5.0..+33.3 | -0.6..-4.0 |
| vc02bew VC0 sender (mostly alone, VC2 compiled in): 256, 1024, 4352 | 139.0, 139.1, 140.0 | +0.1 | 131.0, 131.5, 138.1 | 130.4, 131.0, 138.0 | 131.2, 131.5, 138.0 | +0.1 | -7.6..-8.2, -1.9 | -0.5 | +0.5..+0.8 |
| VC1 split: 1-hop and burst | 142.0-142.3 | ±0.2 | ±0.2 | 142.6-142.9 | = | ±0.2 | ±0.2 | +0.6..+0.8 | ±0.3 |
| VC1 split mean (13) | 159.1 | 159.1 | 159.0 | 159.3 | 159.3 | 0.0 | 0.0 | +0.2 | 0.0 |
| VC1Inter 1_1 / 1_2 / 1_3, 16 B | 346.5 / 347.7 / 375.8 | ±0.2 | 354.4 / 351.9 / 380.4 | 347.4 / 346.7 / 382.5 | 347.4 / 346.8 / 382.3 | ±0.2 | +7.8 / +4.3 / +4.7 | -7.0 / -5.1 / +2.1 | ±0.2 |
| VC1Inter 1_1 / 1_2 / 1_3, 4352 B | 396.8 / 396.2 / 413.8 | ±0.4 | +0.5 / +1.6 / +2.1 | +3.0 / +0.5 / -3.7 | ±0.3 | | | | |
| VC1Inter mean (8) | 337.0 | 336.9 | 339.5 | 338.2 | 338.3 | -0.1 | +2.6 | -1.3 | +0.1 |

- **Fan-in combined** (1/(1/VC0 + 1/VC2) from vc02ew's VC0 and vc02bew's VC2; an estimate, as §8.3):

  | size | v4rxg3 | X3a |
  | --- | ---: | ---: |
  | 256 B | 118.0 | 119.6 (+1.6) |
  | 1024 B | 118.1 | 119.9 (+1.8) |
  | 4352 B | 125.7 | 125.7 |

- **X1 (and X2):** no device effect. Every row is within ±0.2 of v4rxg3, except vc02bew1024's VC2 rate (+1.1;
  256 B 0.0, 4352 B -0.1).
  - The Galaxy express spill `return_slots<0>:returned` (next to the out-of-line worker credit) has no row on this
    P150 host, so the device shows nothing for it. A Galaxy row would need the Slurm harness.
- **X3 (P-B) reproduces the probe** (§8.3, P-B against the lead):
  - VC0 under contention +11..+15;
  - the share moves to VC2 at 4352 B;
  - VC2-on ew4352 -2.5;
  - VC2 alone +2..+6;
  - VC1 inter-mesh 16 B +4..+8.
  - **New:** one ERISC +4.8 (the probe measured 0 there; hypothesis below).
- **X3k (packed) reproduces §8.4** against P-B:
  - multi-VC rows recover;
  - first VC +0.1..+0.7;
  - VC1 split 1-hop and burst +0.6..+0.8;
  - one ERISC -3.0.
- **X3a (A):** small. VC0 under contention +1.1..+2.7; VC2 under contention -0.6..-4.0; VC2 alone +0.3..+0.6. Every
  other row is within ±0.3.
- **X3a against v4rxg3 (the pass so far):**
  - first-VC rows: ew16 +0.6, bew1024 +0.4, ew4352 +0.1;
  - one ERISC +1.8;
  - VC2-on ew4352 -2.5;
  - fan-in combined +1.6 / +1.8 / 0.0;
  - VC2 alone +1.4..+2.6;
  - VC1 split 1-hop and burst +0.6..+0.8;
  - VC1 inter-mesh mean +1.3 (1_3 at 16 B +6.5, a window-bound row);
  - every census row within ±0.2.
- **One-ERISC hypothesis (not isolated):** X3's `visit_vc` takes the loop-local VC's remote receiver progress into a
  local and puts it back after the visit. On two ERISCs both are the loop's register copy. On one ERISC the "loop
  copy" is the state record in memory, so each pass gains a load and a store; the packed word halves them (X3k
  -3.0). The probe's P-B visited that VC's far ring in place. The isolating change is `stage_x3b.py`: the loop-local
  VC visited on its state in place, a later VC still taken and put back. **Settled for the unicast and scatter rows
  by the X3b A/B below. mcast-1e moves the other way.**

### X3b: the loop-local VC visited on its state in place (v4txo3b; user-approved fix)

- **Content:** X3a plus `visit_channels<VC>` (a VC's worker and forwarding channels against a given remote progress).
  `visit_vc` calls it on the loop-local VC's state in place; a later VC is taken into a local, visited, and put back
  where it changed (`after_visit`, now only for later VCs). Script: `stage_x3b.py`.
- **Static against X3a** (full coverage, `guard_v4txo3a_v4txo3b_{O3,Os}.txt`):
  - two ERISCs: every E1 image is identical; E0 bytes differ, but loop size and stack stores/loads are +0/+0/+0 on
    every family at -O3 (bh_trim_gtest: 2 images -2/-1/-1), and only names change at -Os;
  - one ERISC: sp stores/loads -1/-1 (mux, UDM, neighbour -2/-2). The loop grows +3..+16 (torus +23..+25, Galaxy
    express +28..+32). The in-place accesses go through the state record's pointer, which the sp-only guard does not
    count.
- **Device:**
  - Job 460 (X3a against X3b, passes b1/b2 reversed, same job): two-ERISC rows unchanged (ew16, bew1024, ew4352,
    VC2-on ew4352: 0.0; vc02ew256 / vc02bew256 within ±0.2).
  - Job 461 adds v4rxg3 on the one-ERISC rows (passes o1/o2). The passes agree within 0.1. Cycles/packet:

| row | v4rxg3 | X3a | X3b | X3a vs v4rxg3 | X3b vs X3a | X3b vs v4rxg3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ew4352, one ERISC | 200.0 | 201.8 | 199.9 | +1.8 | -1.9 | -0.1 |
| ewscatter, one ERISC | 242.5 | 244.4 | 242.4 | +1.9 | -1.9 | -0.1 |
| mcast, one ERISC | 331.6 | 325.8 | 327.7 | -5.8 | +1.9 | -3.9 |

- **Reading:**
  - The take-and-put-back of the loop-local VC's remote progress on one ERISC cost about 1.9 on the unicast and
    scatter rows. Visiting in place recovers it; the A/B changes only that, so this is settled for those rows.
  - On mcast, X3b gives back 1.9 of X3a's 5.8 gain. Cause unknown (hypothesis: the in-place accesses through the
    record pointer around send's read groups, which the static count shows as +16 loop instructions).
  - X3b stays at or below v4rxg3 on every one-ERISC row measured.

### Fences (X0f)

- Removing the fence after send's header stamp (old L286) alone: byte-identical on all 2,863 configs at -O3 and -Os.
  It goes.
- Removing the fence after the free-slots register update (old L312) alone changes E0 code (98 of 224 images on the
  quick set). It stays as `order_point()` (router_common.hpp, documented with the read-group primitives).
- Script: `stage_x3f.py`, an identity stage on the placement stage's variant.

### X3f: send's fences (v4txo3f)

- The fence decisions above applied to X3b (`stage_x3f.py`): byte-identical to v4txo3b on all 2,863 configs at -O3
  and -Os (`cmp_v4txo3f_{O3,Os}.txt`).

### The host's ACK_CMD_BUF_ID rule against the lead (job 464; offline, 2026-11-02)

- X4's host change (`variants/host_rebuild_20261102b`) gives a worker channel's credits a command buffer of their own:
  `SENDER_CH_<i>_ACK_CMD_BUF_ID` = WR for VC0's worker channel, WR_REG for VC2's, the config's (AT) for every other
  channel. It is in the live build from job 464 on, so every run of v4rxg3 and main since then used it.
- Check (`ackrule_check.sh`, `ack_main_os.sh`, `diff_src.py`): v4rxg3 built on full coverage under the old and the new
  rule (`OFFLINE_OLD_ACK_CMD_BUF=1` skips the offline injection).
  - -O3: 1,168 images identical (every E1 image), 1,695 differ (every image that runs the sender). The old-rule build
    is byte-identical to the pinned v4rxg3 build (2,863 of 2,863).
  - Every changed instruction of the 1,695 (disassembly with addresses stripped, mapped by addr2line) is init: the
    worker interface's constructor storing the field it does not use (`sender_sync_noc_cmd_buf`,
    fabric_erisc_datamover_channels.hpp:475-479, :579), its call site in the router's init, and crt start-up constants
    after the text shifts (firmware_common.h:62 in 122 images). No sender, receiver or round line.
  - -Os: the same images differ, at the same constructor, call-site and crt lines (`ack_src_hr_Os.txt`).
  - Main's router as shipped (`variants/mainx5pin`: main's v1 router in the v2 slot, the live library, and the
    base-state `tt_metal/hw/inc` files it includes; `ack_main_pin.sh`). -O3: 1,776 of 2,872 images differ; -Os: 1,465
    of 2,843. Every changed instruction is the worker interface's constructor and its init
    (fabric_erisc_datamover_channels.hpp:521-525, the counter reset it calls, router :3537/:3539), crt constants, or
    one address constant after the code moved by 4 B: the jump-table address of the local-delivery switch
    (fabric_edm_packet_transmission.hpp:332, 56 images at each level; same instruction, new immediate). No loop logic
    changes. (A first main build, `mainx5base`, read those headers from the live tree while another job's swap was
    active; one config failed to compile and its images are not trusted. Deleted.)
- Coordinator's decision: accepted for v4rxg3, no reruns (its multi-VC rows before and after 464 agree within 0.12%).
  Every sender-pass variant from X4 on is built offline under the new rule (as the live host), and v4txo3fn is X3f
  rebuilt under it (X4's static reference).

### X4: worker credits on the host's command buffer (B; v4txo4)

- Content: `stage_x4.py`'s docstring. Against v4txo3fn (full coverage, `guard_v4txo3fn_v4txo4_{O3,Os}.txt`):
  - every E1 image identical; every family's sender loop +0 except:
  - -O3: 2D + VC2 and torus + VC2 sender loop +6 (VC0 worker trimmed: +8), stack loads +2;
  - -Os: the same families +4 / +7, loads +2; 1D and ring -9 / -10 with 4 fewer stack stores and loads (a spilled
    credit address, and the ring's acks, now stay in registers).
- No device arm: the user replaced the stage A/Bs from X3f on with one census of the final build (2026-11-02). If a
  row regresses there, the X4 arm bisects it.

### X5: one cold worker-connection path (v4txo5, v4txo5b)

- First cut (v4txo5, `stage_x5.py`): the library's worker interface gets a base, `EdmChannelWorkerConnection<NOC>`
  (connection state, `cache_producer_noc_addr`, the teardown check, `close_connection`), and the router one
  out-of-line `worker_connection_event` for every worker channel. Its inline connection check grew the loops: -O3 +6 on
  most families, one ERISC +12 with the check's state spilled around the call (`guard_v4txo4_v4txo5_*.txt`). Not kept.
- v4txo5b (`X5_NAME=v4txo5b stage_x5.py`): a per-channel out-of-line `connection_event<CH>` (the channel's opening
  credit) calling the one shared event. Against X4 on full coverage: sender loop +0 and stack stores/loads +0 on every
  role analysis (2,200 at -O3, 2,282 at -Os; `guard_v4txo4_v4txo5b_*.txt`). Code: two-worker E0 -56 B, one worker
  +72 B (quick set).
- Device validation of v4txo5 (jobs 473, 474): variant_val 13/13, inline write and flushed atomic 1D/2D, UDM 8/8 and
  68/68. The final build is validated again below.

### X6: telemetry's sent count from the channels (C; v4txo6)

- Content: `stage_x6.py`'s docstring. The per-packet hook adds only the words; the round's packets come from
  `total_sent` (the channels' `sent`), passed to `round_end`.
- A round is at most 32 passes over at most 10 sender channels, so a round's packets and 16 B words stay far below
  2^32: the 32-bit round counts cannot wrap, and the functional check does not try.
- Full coverage against X5b (`bc_v4txo6.log`): 2,623 of 2,863 images identical at -O3 and at -Os; the 240 that differ
  are exactly the telemetry-on cases (`*_telemetry`, `bh_2d_bwtelemetry`, `*_allinstr`), both ERISCs. Their sender
  loops shrink (-18 to -54 instructions; VC1 all-instruments -92), with +2 to +7 stack loads (`guard_v4txo5b_v4txo6_*`).
  Telemetry-on images are not scored; the functional check (`tel_txo_runs.sh`) is in the device round.

### X7: the round's sender work in two helpers (v4txo7)

- `before_round` / `after_round` (router_sender.hpp), used by `sender::run` and `single::run`, which no longer repeat
  the round's start, end_round, telemetry and preset restore (`stage_x7.py`).
- Quick set (216 images): 188 differ only in a `beq`'s two registers; the 28 one-ERISC images also swap the two loads
  feeding an add in `total_sent` at the round's start. ~~"in run's round code ... nothing in a pass"~~ **WRONG**
  (2026-11-02): the `beq` is in the pass (`visit_worker`, router_sender.hpp:581, inlined into `pass`); the quick check
  mapped it to the right line but I misread the chain.
- Full coverage, X7 + X8 (v4txo8) against X6, `insn_norm_cmp2.py`: at every address the same opcode; only registers and
  operand order differ (callee-saved registers exchanged, `beq`/`add` operands swapped, two loads feeding an add
  exchanged). -O3: 1,536 images identical, 708 identical after sorting symmetric operands, 619 with registers renamed;
  -Os: 1,520 / 136 / 1,207. The guard: sender and receiver loop +0, stack stores/loads +0 on every role analysis
  (1,844 at -O3, 1,860 at -Os; `guard_v4txo6_v4txo8_*.txt`). Accepted as identity (coordinator, 2026-11-02, on the
  quick-set reading); the census measures it.

### X8: comments (v4txo8)

- `stage_x8.py v4txo7 v4txo8`: the two comments with cycle numbers or variant names now point to RESULTS §98 only;
  "ring bounds" reads "slot bounds". Comments only, the same line count (no `__LINE__` moves).

### Final round on the device (v4txo8 = X8; jobs 496-501, 2026-11-03)

The user replaced the intermediate stage A/Bs with one round on the final build.
- **Validation** (all pass):
  - variant_val 13/13 (two ERISCs: mesh, val1d, ring, Y-torus, mux, VC2, NeighborExchange, trimming; one ERISC:
    mesh, ring, VC2, NeighborExchange, trimming); trimming 63 passed, 3 skipped, in each mode (job 496);
  - inline write and flushed atomic, 1D and 2D; UDM 8/8 and nightly 68/68 (job 497).
- **Telemetry (X6's functional check, job 498; `tel_txo_runs.sh v4rxg3 v4txo8`, `tel_txo_report.py`):** the perf
  recorder's per-core `packets_sent` and `eth_words_sent` equal main's on every reading, for v4rxg3 and v4txo8, two
  ERISCs and one: 50 readings on exp_4352 (2,741,518 packets), 154 on exp_1d_4352 (10,184,202). The dynamic-telemetry
  gtest passes for main, v4rxg3 and v4txo8 in both modes.
- **Census A/B against v4rxg3** (jobs 499-501, passes f1 / f2 in reversed order, same job; cycles/packet;
  `txo_report.py --chain v4rxg3,v4txo8 f1 f2`). `d-` rows: VC2 on with each router's own VC2 slots; `v8-`: the
  17,10,16,8,10 split.

| row | v4rxg3 | v4txo8 | delta | passes (v4txo8 f1 / f2) |
| --- | ---: | ---: | ---: | --- |
| bew1024 | 116.6 | 117.1 | +0.5 | 117.0 / 117.1 |
| d-vc02bew4352 (link) | 136.5 | 135.4 | -1.1 | 135.4 / 135.4 |
| d-vc02bew4352 VC0 sender | 143.3 | 142.2 | -1.1 | 142.2 / 142.2 |
| d-vc02bew4352 VC2 sender | 991.3 | 996.6 | +5.4 | 996.5 / 996.8 |
| d-vc02ew4352 (link) | 438.7 | 447.4 | +8.8 | 447.4 / 447.4 |
| d-vc02ew4352 VC0 sender | 163.6 | 165.8 | +2.2 | 165.8 / 165.7 |
| d-vc02ew4352 VC2 sender | 877.3 | 894.9 | +17.6 | 894.9 / 894.9 |
| d-vc2ew4352 | 857.3 | 875.0 | +17.7 | 875.0 / 875.0 |
| ew16 | 115.9 | 116.6 | +0.6 | 116.6 / 116.6 |
| ew4352 | 127.0 | 127.1 | +0.1 | 127.1 / 127.1 |
| ew4352-1e | 200.0 | 199.9 | -0.1 | 199.9 / 199.9 |
| ewscatter | 142.3 | 142.3 | -0.0 | 142.2 / 142.3 |
| lin1 | 124.4 | 124.4 | -0.0 | 124.4 / 124.4 |
| ns4352 | 127.2 | 127.3 | +0.0 | 127.3 / 127.3 |
| ring1 | 140.9 | 141.0 | +0.1 | 141.0 / 140.9 |
| torusns | 149.2 | 149.0 | -0.2 | 149.0 / 149.0 |
| v8-vc02bew1024 (link) | 132.4 | 125.0 | -7.5 | 125.0 / 125.0 |
| v8-vc02bew1024 VC0 sender | 139.1 | 131.2 | -7.8 | 131.2 / 131.2 |
| v8-vc02bew1024 VC2 sender | 287.3 | 288.1 | +0.8 | 288.4 / 287.7 |
| v8-vc02bew256 (link) | 132.4 | 124.7 | -7.7 | 124.7 / 124.7 |
| v8-vc02bew256 VC0 sender | 139.0 | 130.9 | -8.1 | 130.9 / 130.9 |
| v8-vc02bew256 VC2 sender | 286.0 | 285.3 | -0.6 | 285.6 / 285.1 |
| v8-vc02bew4352 (link) | 133.3 | 131.3 | -2.0 | 131.3 / 131.3 |
| v8-vc02bew4352 VC0 sender | 140.0 | 137.9 | -2.1 | 137.9 / 137.9 |
| v8-vc02bew4352 VC2 sender | 368.5 | 368.5 | -0.1 | 368.5 / 368.5 |
| v8-vc02ew1024 (link) | 134.6 | 134.4 | -0.1 | 134.6 / 134.3 |
| v8-vc02ew1024 VC0 sender | 200.4 | 199.2 | -1.1 | 199.1 / 199.4 |
| v8-vc02ew1024 VC2 sender | 269.2 | 268.9 | -0.3 | 269.1 / 268.6 |
| v8-vc02ew256 (link) | 133.6 | 133.4 | -0.2 | 133.4 / 133.4 |
| v8-vc02ew256 VC0 sender | 200.9 | 199.9 | -1.1 | 199.9 / 199.8 |
| v8-vc02ew256 VC2 sender | 267.2 | 266.8 | -0.3 | 266.8 / 266.9 |
| v8-vc02ew4352 (link) | 153.4 | 153.7 | +0.3 | 153.7 / 153.7 |
| v8-vc02ew4352 VC0 sender | 190.8 | 190.8 | +0.0 | 190.8 / 190.8 |
| v8-vc02ew4352 VC2 sender | 306.8 | 307.4 | +0.6 | 307.4 / 307.4 |
| v8-vc2ew1024 (link) | 226.7 | 225.8 | -0.9 | 225.8 / 225.8 |
| v8-vc2ew1024 VC2 sender | 226.7 | 225.8 | -0.9 | 225.8 / 225.8 |
| v8-vc2ew256 (link) | 223.8 | 223.7 | -0.1 | 223.7 / 223.8 |
| v8-vc2ew256 VC2 sender | 223.8 | 223.7 | -0.1 | 223.7 / 223.8 |
| v8-vc2ew4352 (link) | 240.6 | 241.9 | +1.4 | 241.9 / 241.9 |
| v8-vc2ew4352 VC2 sender | 240.6 | 241.9 | +1.4 | 241.9 / 241.9 |
| vc2on-ew4352 | 134.7 | 132.1 | -2.6 | 132.1 / 132.1 |
| vc2on-torusns | 167.0 | 167.1 | +0.2 | 167.1 / 167.1 |
| vc1: MEAN | 159.1 | 159.2 | +0.2 | 159.2 / 159.2 |
| vc1: VC1BurstEW_N32_G500_size_16 | 142.2 | 142.8 | +0.5 | 142.7 / 142.8 |
| vc1: VC1BurstEW_N32_G500_size_4352 | 142.3 | 142.9 | +0.5 | 142.9 / 142.9 |
| vc1: VC1BurstEW_N8_G500_size_16 | 142.1 | 142.9 | +0.8 | 142.8 / 143.0 |
| vc1: VC1BurstEW_N8_G500_size_4352 | 142.2 | 142.8 | +0.6 | 142.8 / 142.8 |
| vc1: VC1Mesh1HopEW_size_16 | 141.9 | 142.6 | +0.7 | 142.6 / 142.6 |
| vc1: VC1Mesh1HopEW_size_4352 | 142.0 | 142.8 | +0.8 | 142.7 / 142.9 |
| vc1: VC1Mesh1HopNS_size_16 | 142.0 | 142.6 | +0.6 | 142.6 / 142.6 |
| vc1: VC1Mesh1HopNS_size_4352 | 142.0 | 142.7 | +0.7 | 142.7 / 142.7 |
| vc1: VC1Mesh2HopMerge_size_16 | 138.7 | 138.2 | -0.6 | 138.2 / 138.2 |
| vc1: VC1Mesh2HopMerge_size_4352 | 143.5 | 143.1 | -0.4 | 143.1 / 143.1 |
| vc1: VC1Mesh2Hop_size_16 | 239.2 | 238.6 | -0.6 | 238.5 / 238.6 |
| vc1: VC1Mesh2Hop_size_4352 | 258.2 | 257.2 | -1.0 | 257.2 / 257.1 |
| vc1: VC1MeshMulticast_size_4352 | 151.3 | 150.9 | -0.4 | 150.9 / 150.9 |
| vc1i: MEAN | 336.8 | 338.2 | +1.4 | 338.3 / 338.2 |
| vc1i: VC1Inter_0_0_to_1_0_size_16 | 209.0 | 209.1 | +0.0 | 209.1 / 209.1 |
| vc1i: VC1Inter_0_0_to_1_0_size_4352 | 209.9 | 209.9 | +0.0 | 209.9 / 209.9 |
| vc1i: VC1Inter_0_0_to_1_1_size_16 | 346.8 | 347.6 | +0.8 | 347.6 / 347.5 |
| vc1i: VC1Inter_0_0_to_1_1_size_4352 | 397.1 | 400.8 | +3.6 | 400.8 / 400.7 |
| vc1i: VC1Inter_0_0_to_1_2_size_16 | 347.4 | 346.7 | -0.6 | 346.8 / 346.7 |
| vc1i: VC1Inter_0_0_to_1_2_size_4352 | 395.8 | 397.5 | +1.7 | 397.6 / 397.4 |
| vc1i: VC1Inter_0_0_to_1_3_size_16 | 375.7 | 382.2 | +6.5 | 382.3 / 382.2 |
| vc1i: VC1Inter_0_0_to_1_3_size_4352 | 413.1 | 412.2 | -0.9 | 412.3 / 412.1 |

- **Reading** (attribution of the regressions below: hypotheses until the bisect):
  - every 1D, ring, torus, N/S, scatter and one-ERISC row is within ±0.2;
  - VC0 under VC2 contention is faster (vc02bew VC0 -2.1..-8.1; VC2-on ew4352 -2.6);
  - first-VC rows ew16 +0.6 and bew1024 +0.5, and the VC1 split's 1-hop and burst rows +0.5..+0.8: the same deltas as
    X3a against v4rxg3 in jobs 451-456 (X3k for the VC1 rows);
  - **VC2 with its default slots regresses: VC2 alone at 4352 B +17.7 (+2.1%), VC2 under VC0 contention +17.6 and
    +5.4.** These rows were not in the stage A/Bs. Bisect with the X4 arm: job 507 (passes g1 / g2).
  - VC1 inter-mesh 1_1 / 1_2 at 4352 B (+3.6 / +1.7) and 1_3 at 16 B (+6.5): already in the stage data at X3a (jobs
    453 / 456: 1_1 +3.6 = X3 +0.5, X3k +3.0; 1_2 +1.8 = X3 +1.6, X3k +0.5; 1_3 16 B +6.5 = X3 +4.7, X3k +2.1). Not new.
  - VC2 alone at v8 slots: at X3a +1.4 / +1.6 / +2.6 (256 / 1024 / 4352 B; X3 +3.7..+5.6, X3k -1.3..-3.4); now
    -0.1 / -0.9 / +1.4.
- **Bisect with the X4 arm (job 507; arms v4rxg3, v4txo4, v4txo8; passes g1 / g2 reversed, same job):**

| row | v4rxg3 | X4 (v4txo4) | X8 (v4txo8) | X4 vs v4rxg3 | X8 vs X4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| d-vc2ew4352 (VC2 alone, default slots) | 857.3 | 875.0 | 875.0 | +17.7 | +0.0 |
| d-vc02ew4352 VC2 sender | 877.2 | 894.9 | 894.9 | +17.7 | +0.0 |
| d-vc02ew4352 VC0 sender | 163.6 | 165.8 | 165.8 | +2.2 | +0.1 |
| d-vc02ew4352 link | 438.6 | 447.4 | 447.5 | +8.9 | +0.0 |
| d-vc02bew4352 VC2 sender | 991.5 | 996.4 | 996.4 | +5.0 | -0.0 |
| d-vc02bew4352 VC0 sender | 143.3 | 142.2 | 142.2 | -1.1 | -0.0 |
| v8-vc2ew4352 | 240.5 | 241.9 | 241.9 | +1.4 | -0.0 |
| ew16 | 115.9 | 116.6 | 116.6 | +0.6 | +0.0 |
| bew1024 | 116.6 | 117.1 | 117.1 | +0.5 | -0.0 |

  - X5b..X8 move no row (all within 0.1): every regression is between v4rxg3 and X4 (stages X0..X4).
  - ew16 / bew1024: X3k in jobs 451-456 (+0.7 / +0.4).
  - **The VC2 default-slot rows: which of X0..X4 moves them is open** (no stage ran them). Hypotheses, not tested:
    the placement (X3 / X3k: VC2, a later VC, kept in memory; at v8 slots X3 cost VC2 alone up to +5.6 and X3k gave
    back 3.4), or X4's VC2 credit path (preset on WR_REG). The rule stopped the bisect here; a follow-up split is the
    user's call.

### Cleanup item: worker interface storage on ERISCs without workers (2026-11-03)

- Stage A builds the worker interfaces at their fixed location unconditionally (`start_worker_interfaces()` in
  kernel_main). An ERISC that serves no sender channel keeps the storage and its construction: bh_vc1's E1 at -O3
  (336 B .bss, +188 B code; least free stack 4,236 -> 3,896 B), and at -Os every receiver-only E1 (48-336 B; bh_vc1
  4,092 -> 3,752 B).
- Candidate fix (`stage_x8w.py`, v4txo8w; not adopted, the user's call): build them only where the ERISC serves sender
  channels. Full coverage at -O3 and -Os: every image of an ERISC that serves a sender channel is identical to v4txo8;
  the changed images are all sender-less (VC1 E1 storage gone; elsewhere operand order in receiver-buffer init at
  -O3, the storage at -Os); receiver loops +0 on every role analysis (`guard_v4txo8_v4txo8w_*.txt`).

### Headroom across the stages (`headroom_<variant>.txt`; cells: configs passing)

| family | overlay | X0 | X1 | X3 | X3k | X3a | X3b |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1D, 2D, VC1 T5 | all but twk | pass | pass | pass | pass | pass | pass |
| 2D + VC2 | tx1 | 28/28 | 28/28 | 28/28 | **0/28** | 0/28 | 0/28 |
| T1 (2D + VC2, VC0 worker trimmed) | tx1, tx2, trr | 0/28 | 0/28 | 28/28 | 28/28 | 28/28 | 28/28 |
| T1 | twk | 0/28 | 0/28 | 28/28 | 0/28 | 0/28 | 0/28 |
| ring | tx2, trr | 0/10 | 0/10 | 0/10 | 10/10 | 10/10 | 10/10 |
| torus | tx2 / trr | 0 / 0 | 12 / 0 | 12 / 0 | 12 / 12 | 12 / 12 | 12 / 12 |
| torus + VC2 | tx1 | 16/28 | 28/28 | 16/28 | 0/28 | 0/28 | 0/28 |
| torus + VC2 | tx2 / trr | 0 / 0 | 12 / 12 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| VC1 | tx2 | 22/22 | 0/22 | 22/22 | 22/22 | 22/22 | 22/22 |
| VC1 | twk | 0/22 | 0/22 | 0/22 | **22/22** | 22/22 | 22/22 |
| VC1 | trr | 22/22 | 0/22 | 22/22 | 0/22 | 0/22 | 0/22 |
| Galaxy express | tx2 / trr | 0 / 0 | 28 / 0 | 28 / 0 | 28 / 28 | 28 / 28 | 28 / 28 |
| every family but T5, T1 at X3 and VC1 from X3k | twk | 0 | 0 | 0 | 0 | 0 | 0 |

- X4, X5b and X8 (the final build): every cell equal to X3b (`headroom_v4txo{4,5b,8}.txt`).
- **The cells that fail from X3k on:**
  - 2D + VC2 and torus + VC2 at tx1: the only hot value moved is `visit_vc<2>:remote.word`, the later VC's working
    copy of its packed remote progress inside its visit (VC2's packet path; the state itself is in memory by design);
  - VC1 at trr: VC0's worker next slot and send's header and slot address.
- **What packing buys** (X3 to X3k): ring, torus and Galaxy express take two more values (tx2/trr), and VC1 takes a
  worker-sized state (twk).
