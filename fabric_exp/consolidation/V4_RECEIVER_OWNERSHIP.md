# v4 receiver: the owner's words and values (2026-11-01)

Base `v4rxbt34` (byte-identical to the lead `v4rxv2i3`, 2,170 images at -O3 and -Os). This note answers the user's ten
points on v4rxbt34. The rule: every concept has one owner, and the kernel uses that owner's words and values. New variants
are named `v4rxo*` (the name is free). Step 1 is this note only; no code yet.

Checked against:
- the host's named args: `erisc_datamover_builder.cpp` (lines cited as `B:`) and `compute_mesh_router_builder.cpp:901-938`
  (the defines, `C:`);
- `builder/fabric_builder_config.hpp:46` (`MAX_NUM_VCS = 3`);
- the header `tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp` (`H:`);
- the codec `hostdevcommon/fabric_common.h` (`Routing2DCodec`, `R:`);
- the VC2 line work (`tt-metal-vc2line`, `vc2l_p3b`);
- every captured router named-arg map in the coverage cases (4,478 maps).

## 1. Glossary

### 1.0 Current names: channel kinds (user, 2026-11-02; v4rxg1-v4rxg3, §18)

**Rule.** Each channel kind has `<Kind>Channel` (its facts), `<Kind>ChannelState` (its state) and
`<Kind>ChannelProgress` (where it is in its slots). The kinds use host vocabulary: Receiver, Sender, RemoteReceiver,
DownstreamSender. Identifiers say "channel" and "slot", never "ring" or "cursor"; comments may still say a channel's
slots wrap. The tables after this one are revision history (the names they propose were adopted through v4rxf4).

| v4rxf4 | v4rxg3 | file | note |
| --- | --- | --- | --- |
| `RingProgress` | `ReceiverChannelProgress` | receiver | {packets_ready, next_slot_addr, next_index, uncredited, idle_passes_with_partial_group} |
| `ring_add` | `index_add` | receiver | |
| `completion_group_size_of_ring` | `completion_group_size_for` | receiver | |
| `using ReceiverChannelStates = Each<...>` | `struct ReceiverChannelStates : Each<receiver_vcs, ReceiverChannelState>` | receiver | owns the placement rule (one comment, one block above pass() and run()) |
| `FIRST_VC` (0) | `ReceiverChannelStates::loop_local_vc` | receiver | derived trait: the lowest VC whose receiver channel runs (0 where none runs) |
| `FirstVcProgress`, `first_vc_progress_in` | `loop_local_progress()`, `progress_for<VC>(loop_copy)`, `after_step<VC>()` | receiver | run() copies the loop-local progress into `loop_copy`; one ERISC steps on it in place |
| the branch in `step_channel` | - | receiver | step_channel keeps only its `runs_vc` gate |
| static_asserts: little-endian; a sender channel fits in a byte | removed | receiver | `retire(..., uint8_t source)` and `SlotSources::set(uint32_t, uint8_t)` take `source_channel()`'s `uint8_t` as it is |
| `DownstreamCursor {next_slot, free_slots}` | `DownstreamSenderChannelProgress {free_slots, next_slot_addr}` | downstream | field order as the sender's `RemoteReceiverChannelProgress` |
| `Relay::cursor`, `Downstreams::cursors` | `progress` (by entry in `Downstreams`) | downstream | |
| `DownstreamLink::ring_base`, `RelayArgs::ring_base` | `first_slot_addr` | downstream | |
| `DownstreamArgs::ring_bases`, the init args' `ring_bases` | `first_slot_addrs` | downstream, init | |
| `next_ring_slot` | `slot_after` | downstream | |

Kept: `DownstreamTraits<VC>` (a VC's downstream routers, not one channel); `NamedProfiler` (main's library). Left for
the sender pass: `producer_cursors` (router_init.hpp args, the sender channels' producer cursor blocks, and its
`cursor` local in fabric_erisc_router_v4.cpp); `FarRings` / `far_rings_start` (router_single.hpp's sender half).

### router_receiver.hpp

| current | meaning | owner's term (cite) | proposed |
| --- | --- | --- | --- |
| `completion_group_size<N>` | slots per completion group of an N-slot ring | the kernel's own policy (the host's RECEIVER_CREDIT_AMORTIZATION_FREQUENCY is speedy-only; v4 does not use it) | `completion_group_size_of_ring<N>` (frees the member name) |
| `Channel::group_size` | that size for this channel (1 under first-level acks) | - | `Channel::completion_group_size` |
| `group_of<G>(i)` | slot i's completion group | - | folded into its one user, the trid map |
| `group_end<G>(i)` | the index just past slot i's group | - | folded into its one user, `slots_due_for_credit` |
| `GroupIds<G,FIRST,N>::of(i)` | slot i's trid (its completion group's) | ct args `RX_CH_TRID_STARTS`, `NUM_TRANSACTION_IDS` | `CompletionGroupTrids<...>::of_slot(i)` |
| `Channel::TransactionIds` | the channel's trid map | same | `Channel::CompletionTrids` |
| `SlotRecords::group_sources(f, n)` | the sender channels of one completion group's slots | H `src_ch_id` | `SlotSources::completion_group_sources(f, n)` |
| `CompletionCounts::uncompleted` | slots retired but not yet credited | completion credit | `uncredited` (matches `oldest_uncredited`) |
| `CompletionCounts::idle_passes` | passes without a packet while a partial group waits | - | `idle_passes_with_partial_group` |
| `release_after_idle_passes` (4) | how many such passes before a partial group is credited | - | `idle_passes_before_partial_credit` |
| `completion_wait_checks` (32) | the most status reads a group's proof waits for | the NoC API's "trid sent" | `completion_proof_max_checks` |
| `vc_has_first_level_acks` `{VC0, VC1, false}` | first-level acks, per VC | `ENABLE_FIRST_LEVEL_ACK_VC0/1` (B:1347, B:1353); VC2 line adds `..._VC2` | deleted; read from ct args (§2 #3) |
| `Channel::acks` | the channel's packets are first-level acked | same | `Channel::first_level_ack` (the sender's name) |
| `channel_acks<CH>()`, `any_channel_acks` | a channel that runs here has first-level acks | - | `runs_with_first_level_ack(ch)`, `any_first_level_ack_channel` |
| `acknowledge_arrivals<CH>` | first-level ack of every packet that arrived since the last pass | the credit sender's `ack_counters`; main's `send_ack_credit` | `accumulate_first_level_acks<CH>` (a verb, after review) |
| `accumulate_ack` | adds 1 to a sender channel's ack count | the credit sender's `ack_counters` | the credit sender's new `accumulate_ack_credit(src)` (§3 #10, approved) |
| `vc_delivers_only` `{speedy, false, true}` | the channel only delivers, per VC | `DISABLE_RX_CH<vc>_FORWARDING` (B:1585-1587) = ct `is_vc_forwarding_disabled[MAX_NUM_VCS]` | deleted (§2 #4) |
| `Channel::routes` | decodes a route and may forward | forwarding on (`!DISABLE_RX_CH<vc>_FORWARDING`) | `Channel::forwards` |
| `Channel::lands` | packets come from another mesh | `IS_RECEIVER_CHANNEL_<i>_INTERMESH_INGRESS` (B:1469) | `Channel::intermesh_ingress` |
| `Channel::downstream_vc` | the VC of the downstreams it forwards on | crossover, `FABRIC_2D_VC0_CROSSOVER_TO_VC1` (C:933) | `Channel::forwards_on_vc` |
| `Channel::relays_delivery` | UDM: delivers through the relay | `UDM_MODE`; the host's "local tensix relay" | `Channel::delivers_through_relay` |
| `vc_is_received` | traffic of this VC reaches the router | `FABRIC_2D_VC1_SERVICED`, `..._VC2_SERVICED` (C:924, C:930) | deleted; ct `is_vc_serviced` (§2 #2) |
| `due(..., count&)` | whether the oldest uncredited slots are due for credit, and how many | - | `slots_due_for_credit(...)`, which returns the count (0 = none) |
| `release_completions<CH>` | the due, proof and credit steps | completion credit | `credit_completions<CH>` |
| `credit_slots` | adds slots to their sender channels' counts | `accumulate_completion_credit` | `accumulate_completion_credits` |
| `writes_left`, `writes_left_within<N>` | every write tagged with the trid has left | `ncrisc_noc_nonposted_write_with_transaction_id_sent` | `trid_writes_sent`, `trid_writes_sent_within<N>` |
| `frame_due` | the credit counts changed since the last frame | `flush_completion_credits` | `credit_frame_due` |
| `group_txq_status`; "register group" (comments) | the TX-queue status, read with the pass's grouped reads | - | `txq_status_at_pass_start`; "the pass's grouped reads" |
| `Tables` | VC0's slot records, plus VC1's and VC2's {cursor, records} | - | see §3 #2 |
| `SlotRecords<N>`, `SlotRecordsOf<CH>`, `record`, `source_of` | each slot's sender channel, one byte per slot | H `src_ch_id` | `SlotSources<N>`, `SlotSourcesOf<CH>`, `set`, `source_of` |
| `start<CH>()` | a channel's cursor before its first packet | - | `cursor_start<CH>()` |
| `slot_retired`, `low_bytes` | helpers with one user each | - | folded into `retire` and `SlotSources` |
| `PacketInfo` (`load_word`, `decode`, offset 40, shifts 16/24) | size, send type and sender channel, read with one load | H:432-450 `PackedPayloadAndSendType` (its comment already names `src_ch_id`) | moves to the header (§3 #3) |
| `source_channel()` | the far sender channel; 0 in mux mode | `SKIP_SRC_CH_ID_UPDATE` (B:1362) | name kept; it reads the host's arg (§2 #8) |

Kept as they are: `Cursor`, `Position` (`ready`, `slot_addr`, `index`), `CursorOf<CH>`, `ChannelState<CH>`, `Credits`,
`ring_add`, `oldest_uncredited`, `retire`, `receive`, `handle`, `step`, `pass`, `run`, `note_received`,
`record_forwarded`, `read_packets_received`, `send_credit_frame`.

### router_downstream.hpp, router_routing.hpp, router_local_delivery.hpp, shared files

| file | current | owner's term | proposed |
| --- | --- | --- | --- |
| downstream | `DownstreamTraits<VC>`, with `VC == 0 ? ... : ...` ternaries | `VC<n>_DOWNSTREAM_EDM_SIZE`, `DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY[vc]`, `VC<n>_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_<k>` | name kept; reads per-VC tables (§2 #16) |
| downstream | `forwards_are_stateful<VC>` and its `VC == 0` term | stateful NoC API | `forwards_keep_preset<VC>`; `VC == 0` becomes the named `forward_preset_vc` (§3 #1) |
| downstream | `vc1_links_vc` (init), `IfRelay`, `NoRelay`, `NoDownstreams`, `downstreams_for` | - | deleted (§3 #1) |
| downstream | `RelayTraits`, `relay`, `relay_cursor`, `admit_relay`, `forward_to_relay` | the host's "local tensix relay" | one `Relay` {link, cursor, `admit`, `forward`} (§3 #1) |
| downstream | `any_used_up` | - | `any_without_free_slot` |
| routing | `Routing2D<VC, LANDS>`, `land()` | `fabric_set_2d_intermesh_landing_route` | `Routing2D<VC, INTERMESH_INGRESS>`, `set_landing_route()` |
| routing | `exits` | R `action_is_intermesh_exit`; `IS_INTRAMESH_ROUTER_ON_EDGE` | `intermesh_exits` |
| routing | `column_only`, `action()` | R:278-297 `decode_action<MY_DIR>`, `decode_action_y_first` | deleted if the codec is called (§3 #9) |
| routing | `before_forward` | main's `update_packet_header_for_next_hop` | `update_for_next_hop` |
| local delivery | `deliver<uint8_t CH>` | - | `size_t CH`, like every other channel template |
| common | `HeaderWord` | - | deleted (no user since bt34) |
| common | the `PacketHeaderBase` alias | H | deleted along with `PacketInfo` |
| common | `downstream_direction_table`, `max_downstreams` | R:302 `fwd_dirs<MY_DIR>()` (the same table) | read from the codec |
| common | `heartbeat_addr` (Blackhole 0x7CC70) | `eth_fw_api.h:21` `MEM_SYSENG_ETH_HEARTBEAT` | uses it (Wormhole keeps main's literal) |
| init | `wait_for_channel_transactions<0>`, `<1>` with VC1, `<VC2_RECEIVER_CHANNEL>` with VC2 | - | one loop over the receiver channels |
| single | `ReceiverState{cursor0, tables, credits, frame_due}`, `d0` / `d1` | - | follows §3 #1 and #2 |

## 2. Ownership: facts the kernel decides or copies

| # | fact | kernel today | true owner | host provides? | kernel receives it as | static_assert kept |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | VC count | `std::array<bool, 3>`, three times | `builder_config::MAX_NUM_VCS` | yes: `MAX_NUM_VCS` (B:1321; ct :89) | `MAX_NUM_VCS` | `MAX_NUM_VCS == 3` next to the per-VC arg lists (there is one named arg per VC) |
| 2 | VC serviced | `vc_is_received`, from the defines | C:913-931 | yes: `FABRIC_2D_VC1/VC2_SERVICED`; VC0 is always serviced | ct `is_vc_serviced[MAX_NUM_VCS]` | - |
| 3 | first-level acks per VC | receiver `{ACK0, ACK1, false}`; sender `vc == 0 ? ... : false` | B:1346-1353 | VC0 and VC1 yes; **VC2 no** (VC2 line: `ENABLE_FIRST_LEVEL_ACK_VC2`) | ct `is_vc_first_level_ack_enabled[MAX_NUM_VCS]`; **host patch** emits `ENABLE_FIRST_LEVEL_ACK_VC2 = 0` | bring-up gate in `brought_up`: `!...[1]`, `!...[2]` (not brought up here) |
| 4 | delivers only | `{super_speedy_mode, false, true}` | B:1569-1587 (speedy, the capture, VC2) | yes: `DISABLE_RX_CH<vc>_FORWARDING` | `Channel::forwards = !is_vc_forwarding_disabled[vc]`. In all 4,478 maps DIS0 == speedy, DIS1 = 0 and DIS2 = 1, so the result is byte-identical. VC2 line's `!vc2_line_mode` edit becomes unneeded (its host already sets DIS2 by mode). | speedy implies DIS0 (`speedy_config_supported`, already in v4.cpp); the channel's contract assert becomes a tautology and is dropped (B:1592 `TT_FATAL` owns it) |
| 5 | forward VC (crossover) | `downstream_vc`, from the define | C:933-938 | yes: `FABRIC_2D_VC0_CROSSOVER_TO_VC1` | ct `vc0_crossover_to_vc1`; `forwards_on_vc = (vc == 0 && crossover) ? 1 : vc` | `!vc0_crossover_to_vc1 \|\| has_vc1` |
| 6 | intermesh ingress | per channel | B:1469 | yes | already read; rename only | - |
| 7 | relay delivery (UDM) | `udm_mode` | B:1365 | yes | unchanged | `udm_config_supported` |
| 8 | skip src_ch_id (mux) | ct `= fabric_tensix_extension_mux_mode` (inherited from main) | B:1115-1120 | yes: `SKIP_SRC_CH_ID_UPDATE`, which ct ignores today | ct reads `NAMED_CT_ARG("SKIP_SRC_CH_ID_UPDATE")`; equal to MUX_MODE in all 4,478 maps | - |
| 9 | header word layout (offset 40; shifts 16 and 24) | `PacketInfo`; the dead `HeaderWord` | H:414-450 | n/a (layout) | H `PackedPayloadAndSendType` gains `src_ch_id`, `load_word(hdr)` and `unpack(word)` (additive, device-safe) | the offsetof asserts move into the header |
| 10 | src_ch_id fits a byte (SlotSources) | `MAX_NUM_SENDER_CHANNELS <= 0x100` | H `uint8_t src_ch_id` | n/a | `static_assert(sizeof(src_ch_id) == 1)` kept with SlotSources | yes |
| 11 | 2D route buffer `[Y \| X]` and the decode rule | `Routing2D::action` reimplements it | R:278-297 | n/a (codec) | call `decode_action<my_direction>` / `decode_action_y_first`; optional, see §3 #9 | - |
| 12 | 1D next-hop rule | `before_forward` reimplements it, reading extension words only on a refill | main `update_packet_header_for_next_hop` (transmission :642) | n/a | **defer**: a lean helper in the header would change 1D code; for now, a comment naming the owner | - |
| 13 | compact downstream order | `downstream_direction_table` | R:302 `fwd_dirs<MY_DIR>()` | n/a | from the codec (constexpr, identical) | - |
| 14 | at most 4 downstreams | `max_downstreams = 4` | R `fwd_dirs().size()`; the builder's `num_downstream_edms_2d_*` | n/a | `fwd_dirs<...>().size()` | - |
| 15 | the downstream sender channel this router feeds (`below ? d : d+1`, VC1 base `VC1_FABRIC_POSITION_START - 1`) | router_common, the same as main's kernel | the host's wiring rules (`get_packed_downstream_sender_channel_ids`) | only under capture (`PACKED_DOWNSTREAM_VC<n>_SENDER_CHANNEL_IDS`) | **defer**: a mainline-wide host change; note it | - |
| 16 | per-VC downstream count, slots and credit streams; capture ids | `VC == 0 ? A : B` ternaries | B named args | VC0 and VC1 yes; VC2 only in VC2 line | per-VC arrays in ct, for the VCs the host lists downstream args for | `VC < count` of those |
| 17 | live downstreams (an E/W router inside a mesh does not turn N/S) | `live_downstreams` | the codec's decode rule (an E/W router reads X; `widen_x` sets only E/W bits) | n/a | kept; the comment cites the codec | - |
| 18 | which VCs' forwards use the forward command-buffer preset | `VC == 0` inside `forwards_are_stateful` | the host's cmd-buf ids (`RX_CH_0/1_FWD_DATA_CMD_BUF_ID` are both 2; V4_BOUNDS §4 #1) | yes (ids) | the buffer-2 fix, option (c), user-confirmed (§3 #1) | VC0 is stateful only when every sharing VC's live-downstream mask equals its own |
| 19 | the heartbeat word and the 0xDCBA signature | literals | Ethernet FW `MEM_SYSENG_ETH_HEARTBEAT` (Blackhole); 0xDCBA is UMD's | n/a | the Blackhole constant; Wormhole literal with its source | - |
| 20 | completion group size, 4 idle passes, 32 proof checks, group of 1 under first-level acks | receiver | the receiver (policy) | no | kept in the receiver; the comment says the per-slot rule exists for bubble flow control (the host pairs ACK_VC0 with deadlock avoidance, B:1246) | `group_size <= 4` |
| 21 | `NUM_TRANSACTION_IDS`, `RX_CH_TRID_STARTS` | ct (main's "TODO: pass from host") | the kernel side (ct) | no | unchanged (one owner: the receiver and teardown both read ct) | - |
| 22 | `VC2_RECEIVER_CHANNEL`, `vc_of_receiver_channel` | ct derives them from the host's counts | the host's dense layout | indirectly | unchanged (one place, ct) | - |
| 23 | scatter chunk mask `0b11` ("main: CHUNK_ENCODING_MASK") | local delivery | H:175-191 (2-bit encoding, `set_chunk_encoding`) | n/a | **optional**: the header gains a `get_chunk_encoding(encs, i)` reader (additive); a variable shift may change scatter code | - |
| 24 | receiver channels 0..2 in `pass` | three explicit reads and steps | `MAX_NUM_RECEIVER_CHANNELS` (B:1320) | yes | the scalars stay (§3 #7) | `MAX_NUM_RECEIVER_CHANNELS == 3` beside `pass` |

## 3. Structure

| # | item | current shape | why (measured where it is a placement choice) | proposed | identical? |
| --- | --- | --- | --- | --- | --- |
| 1 | **VC0 vs VC1 downstreams** (point 9) | (a) `run(Downstreams<0>&, Vc1Downstreams&)`, with a VC0 stand-in object when there is no VC1 (`vc1_links_vc`); (b) the UDM relay lives inside VC0's (`IfRelay`); (c) only VC0 presets (`VC == 0`); (d) init is per VC under `if has_vc1`; (e) traits use ternaries | (a) history: it avoids instantiating `DownstreamTraits<1>` (count > 0) before `Present` existed; (b) UDM is VC0-only (`!has_vc1`); (c) VC0 and VC1 forwards share data cmd buf 2, so only one VC can keep a preset; (e) the host's per-VC args. Placement: both VCs stay in memory by reference (in registers, 2D scatter +5 cycles/packet, RESULTS §98 v4b/v4c) | `DownstreamConnections` {`Present<..,Downstreams<0>,0> vc0`, `vc1`, `Present<udm_mode, Relay> relay`; `of<VC>()`}; `Downstreams<VC>` is the same for every VC; traits from per-VC tables; init loops over the VCs that have downstream args. **Buffer-2 fix, option (c)** (the VC2 line rule, generalized to every VC): a VC whose one live downstream equals VC0's forwards stateful on VC0's preset; where more than one VC shares the preset, every forward sets its own credit address; where the masks differ, VC0 falls back to stateless (a static check). VC1's forwards move from NoC VC 1 to EDM_NOC_VC. Checked offline on a synthetic VC1 mesh larger than 2x2 (no such router on this host) | **no**: one reference instead of two, the stand-in disappears, UDM changes. Expected (hypothesis): loop instructions equal up to register names on configs the fix does not touch. Per-function compare, then device A/B |
| 2 | **`Tables` / `ChannelState`** (point 10) | VC0's cursor is a loop local; `Tables` {VC0's slot records; VC1's and VC2's {cursor, records}} | GCC keeps an aggregate with a run-time-indexed member in memory. VC0's cursor inside an aggregate: +4 stack accesses (§5, offline). One rule for every cursor (v4rx2, §6): VC1 split -0.4%, inter-mesh -1.0%, torus Y with VC2 +0.7% (scatter +9.4). The register-pressure hypothesis was NOT CONFIRMED (v4rx2m); those rows followed where GCC placed the TX-queue status's first use. These are data points tied to the v4rx1/v4rx2 structure | **B2 (recommended)**: one rule. Every channel's cursor is a loop local (`Present`); every channel's slot sources go in one object in memory, `ChannelSlotSources` {ch0, ch1, ch2}. **B1' (fallback)**: placement unchanged; `Tables` becomes `ChannelStates`, and a comment states the VC0 rule | B2 **no** (device A/B); B1' yes |
| 3 | `PacketInfo`'s home (point 7) | a struct in the receiver that re-encodes offsets and shifts | one load instead of three (uncached L1) | in the header, additive: `PackedPayloadAndSendType` gains `uint8_t src_ch_id`, `load_word()` and `unpack()`; `load()` becomes `unpack(load_word())` (same behaviour; main's users unchanged) | likely (same load and shifts; check inlining at -Os) |
| 4 | one-use generic helpers (point 5) | `low_bytes`, `group_of`, `group_end`, `slot_retired`; the dead `HeaderWord` | - | folded into their users; deleted | yes |
| 5 | `me` built twice | `run` and `single::run` each cast `ROUTING_TABLE_BASE` | - | `my_coordinates()` beside `l1_routing_table()` | likely |
| 6 | teardown special-cases VCs | `<0>`, `<1>` if VC1, `<VC2_RECEIVER_CHANNEL>` if VC2 | - | `for_each_index<NUM_RECEIVER_CHANNELS>` (the same set and order) | likely |
| 7 | the pass names channels 0..2 | `arrived0/1/2`, `step<0..2>` | the grouped reads stay scalars: a record across the fences stays in memory (+7 to 10 cycles/packet, §12) | kept; static_assert on `MAX_NUM_RECEIVER_CHANNELS`; `any_first_level_ack_channel` becomes a constexpr loop | yes |
| 8 | `slots_due` (point 8) | `bool due(..., count&)` | - | `slots_due_for_credit()` returns the count. §38: flat on most rows, +0.4 to +1.7 on VC1 inter-mesh (bound by VC1's slot windows; the cause is not isolated); the user accepts it | **no**, by design |
| 9 | route decode duplicates the codec | `Routing2D::action`; plain load on E/W non-exit routers | volatile Y/X pair, so the two loads issue together (LEDGER E1-19) | call the codec (its loads are always volatile) | **maybe not** on E/W non-exit routers (plain becomes volatile); optional |
| 10 | first-level ack writes the credit sender's members | `accumulate_ack` writes `ack_counters` and `ack_counters_base_ptr` | `accumulate_completion_credit` has no ack twin | `accumulate_ack_credit(src)` in `ReceiverChannelCounterBasedResponseCreditSender` (fabric_router_flow_control.hpp, **outside the listed scope**; additive) | likely |

## 4. Stages

Each stage is a cumulative variant. After each one: the coverage build at -O3 and -Os, loadable-bytes compare against
`v4rxv2i3_O3+v4rxbt0tv*_O3` and `v4rxbt0base_Os+v4rxbt0tv*_Os`, and `size_report.py` (logged in V4_SIZE.md). A stage
expected to be identical that is not gets single-change A/Bs before the next stage starts.

| stage | variant | content | expected |
| --- | --- | --- | --- |
| 0 inline-write fix | v4rxo0 | the user's option (i), §6 | identical except the images that contain the inline-write arm |
| R1 renames | v4rxo1 | §1 renames (the receiver, downstream, routing, local delivery, single; the sender untouched) | identical to v4rxo0 |
| H1 host-owned facts | v4rxo2 | §2 #1-#5, #8, #9, #13, #14, #16, #18, #19; header accessor; `forwards` from the host. Offline: `ENABLE_FIRST_LEVEL_ACK_VC2=0` injected by copies `offline_build_rxo.py` and `coverage_rxo.py` (names free; offline_build.py untouched) | identical |
| S1 small structure | v4rxo3 | §3 #4-#7 (and #10 if the user approves it) | identical |
| S2 per-VC downstreams | v4rxo4 | §3 #1 | not identical: per-function compare |
| S3 channel state | v4rxo5 | §3 #2, B2 (or B1') | B2 not identical |
| S4 slots due for credit | v4rxo6 | §3 #8 | not identical |
| (S5, optional) | - | §3 #9, §2 #23 | maybe |
| C1 comment trim | v4rxo7 | cut history (variant names, cycle numbers); keep each "why" in a line with one doc pointer; placement notes only at the choices they explain | identical to v4rxo6 |

Device work, for S2 to S4 only. About 1 h 45 min of the 2 h budget:
1. One row A/B job, with S2, S3 and S4 against the lead, two passes in reversed order. Rows: §38's set (2D E/W and N/S
   4352 and scatter, 1D, ring, torus N/S, VC2-on rows, VC1 split and inter-mesh), plus one-ERISC 2D. `udm_runs.sh`
   also runs, because S2 moves the relay.
2. The full census (`census_runs.sh`, `census_vc1_runs.sh`) on the final variant.
3. `variant_val_runs.sh` (13/13) on the final variant, plus, from stage 0 on, the inline-write and flushed-atomic gtests
   for 1D and 2D (`bnd_inline_runs.sh`'s sequence) for every stage that changes codegen.

**Host patch.** The H1+ variants read `ENABLE_FIRST_LEVEL_ACK_VC2`, and a missing named arg does not compile. So the patch
must be applied, with the host rebuilt, before any device job of those variants. The patch is one line, after B:1353:
`named_args["ENABLE_FIRST_LEVEL_ACK_VC2"] = 0;` (neighbour exchange: one hop, no bubble flow control). VC2 line's host
uses the same name. It will be written as a patch file inside the v4rxo2 directory.

## 5. Decisions for the user

1. Channel state: B2 (one rule, device A/B) or B1' (a rename, identical).
2. The host patch `ENABLE_FIRST_LEVEL_ACK_VC2`. It gates device runs from H1 on.
3. The sender's VC2 `false` for first-level acks: make it read `is_vc_first_level_ack_enabled` now (a one-line,
   identical change; otherwise a VC2 ring build would ack on E1 but not on E0), or leave it for the sender's migration.
4. Crossover: `forwards_on_vc = (vc == 0 && vc0_crossover_to_vc1) ? 1 : vc` follows the host's name. VC2 line's phase 3
   instead makes every channel cross over, so it would then need its own host flag for VC2. Alternative: every channel
   crosses over (today's behaviour).
5. §3 #10: an additive `accumulate_ack_credit` in fabric_router_flow_control.hpp (outside the listed files).
6. Optional and deferred items: §3 #9 (codec decode), §2 #12 (1D next hop in the header), §2 #15 (downstream sender
   channel from the host), §2 #23 (chunk-encoding reader).
7. The comment-trim policy in C1.

## 6. Stage 0: the inline-write hang fix (V4_BOUNDS §4-§5)

**Option (i), `v4rxo0`: WRONG, ruled out.** The design: the emulated L1 inline write goes on `write_at_cmd_buf` (3), on
the claim that buffer 3's users all program their full command. The claim is false.
- Offline, as intended: only `deliver_other` changed (buffer 0 → 3), and every E0 image was identical.
- Device (job 366):
  - InlineWrite OK; FusedAtomicInc FAILED with a data mismatch (receiver status 0xabc00003, words 1 vs 10);
  - E1 then did not return to base firmware at teardown.
- Mechanism, by code: Blackhole firmware's `noc_init_one` presets buffer 3's `NOC_RET_ADDR_LO/MID/COORDINATE` to the
  core's atomic-return scratch (`MEM_NOC_ATOMIC_RET_VAL_ADDR`). `noc_fast_atomic_increment` (program_ret_addr=false,
  INCR_GET) relies on that preset. The emulated write overwrote it, so every later local atomic returned its old value
  into the inline write's destination. On a stateful E1, every command buffer holds preset state.

**Option (ii′), `v4rxo0b`: the user's choice, adopted.**
- Library (additive; defaults unchanged):
  - Blackhole `noc_fast_spoof_write_dw_inline<…, emulated_write_on_cmd_buf>` issues the emulating write on the
    `cmd_buf` it is passed;
  - `noc_inline_dw_write<…, emulated_write_on_cmd_buf>(…, cmd_buf = write_at_cmd_buf)` (dataflow_api.h / _raw.h; the
    flag is passed on Blackhole only).
- Router: v4's `deliver_other` handles NOC_UNICAST_INLINE_WRITE itself (`local_inline_write`, router_local_delivery.hpp).
  It issues on `local_write_cmd_buf` with the flag set, then calls `local_write_init()`, which restores the preset
  (NOC_CTRL only, the preset's one register).
- The shared transmission arm is untouched: main's router and Wormhole are unaffected, and the DRISC restore is kept.
- Offline (-O3/-Os; 60 cases plus the 17 new trim cases):
  - against the lead, only E1 images containing the arm differ: `deliver_other` +5 instructions, plus one layout constant
    in `_start`;
  - every E0 image is identical.
- Device (job 385):
  - job-351 sequence, 1D and 2D: main's router 2/2 PASSED; v4rxo0b 2/2 PASSED, no hang;
  - validation 13/13;
  - spot rows, lead / v4rxo0b in cycles/packet, two passes: ew4352 126.8 / 126.8, ewscatter 139.6-139.7 / 139.7,
    ring1 140.9 / 140.9, torusns 149.2 / 149.1-149.2.

## 7. Step 2: stages and identity (offline; -O3 and -Os; swap guard clean on every build)

Every build pins its non-fabric files (dataflow_api.h, dataflow_api_raw.h, BH/WH noc_nonblocking_api.h, and from H1 on
fabric_edm_packet_header.hpp). Builds go through coverage_rxo.py / offline_build_rxo.py, which inject
ENABLE_FIRST_LEVEL_ACK_VC2=0 and VC2_CROSSOVER_TO_VC1=0. Cases: the 60 lead-base cases, the 17 new trim cases, and the
trim gtests (restored).

| stage | variant | result vs previous stage | cause (single-change A/B) |
| --- | --- | --- | --- |
| R1 renames | v4rxo1 | identical (1,696 / 515 images, both levels) | - |
| H1 host-owned facts | v4rxo2 | -O3 identical; -Os: `receiver::run` / `single::receiver_pass` change (stack-slot assignment, +0 or +1 instruction) | the PacketInfo → header accessor move alone (v4rxo2p2 = R1 + move reproduces H1 exactly). On the old stage 0 a second -Os effect showed: `tt_buf_rw::note<>` (an empty asm-only library function) inlined or not; it needed all of H1's parts together and disappeared after the rebase |
| S1 small structure | v4rxo3 | identical; bh_2d_mux_trim_all now builds (+28 images, the sender's mux-trim fix) | `my_coordinates()` (return by value) changed `run()` at -O3, and `l1_routing_table()` (a plain reference drops `tt_l1_ptr`) changed code-profiling builds; both dropped (item #5 not done) |
| S2 per-VC downstreams + buffer-2 fix (c) | v4rxo4 | `kernel_main` -26 instructions (median), `init_downstreams` +4, `receiver::run` same count on 2D (local-state stack layout), -30 on VC1 | by design |
| S3 one cursor rule (B2) | v4rxo5 | multi-channel routers (VC1, VC2): `receiver::run` -17..+38 (median 0), `single::receiver_pass` layout | by design |
| S4 slots due for credit | v4rxo6 | `receiver::run` -37..0 (median -12), `single::receiver_pass` -31..+2 (median -9) | by design (§38) |
| C1 comment trim | v4rxo7 | identical to S4 (2,239 images, both levels) | - |
| codec decode (A/B arm) | v4rxo6c | `receiver::run` 0..+8 (median +1) | by design |

**Buffer-2 fix (c), checked offline on synthetic inner routers.** Case rxo_vc1_inner (private dir cov_rxo): bh_vc1's 32
intra-mesh E1 configs with IS_INTRAMESH_ROUTER_ON_EDGE=0, as the host would emit them for a VC1 mesh larger than 2x2.
- A compile-time probe confirms the rule:
  - inner E/W routers: VC0 keeps the preset, VC1 rides it, and the credit register is set per forward;
  - N/S routers and the real bh_vc1 edge/intermesh routers: stateless.
- In the code: on an inner W-facing router both forward sites are stateful on buffer 2 (trid, source, destination), and
  each sets its own credit address on buffer 1. Before the fix, VC1 was stateless on VC0's preset buffer.
- On an E-facing router, `receiver::run` drops 1,034 → 992 instructions.
- Not verified: device behaviour (no inner VC1 router on this host; a Galaxy run comes with the VC2 line work).

**Size (v4rxo7 vs the lead, V4_SIZE.md):** VC1 E0+E1 -500 B (-O3) / -384 B (-Os); E1 least free stack +80 / +224 B.

**Not done (deferred, listed in §2):** the 1D next-hop rule in the header, the downstream sender channel from the host,
the chunk-encoding reader, and `my_coordinates()` (§3 #5; it changed codegen, see S1).

## 8. Step 2 on the device (host rebuilt, job 399; jobs 400-403; cycles/packet)

v4rxo7 = S4 + C1 (final candidate); v4rxo6c = S4 + codec decode. Single-change A/B arms: v4rxo4 (S2), v4rxo5 (S3).

| check | result |
| --- | --- |
| VC0 rows, lead / v4rxo7 / codec (job 400, two passes reversed) | ew4352 126.9 / 126.8 / 127.0; ns4352 127.1 / 127.2 / 127.2; lin1 124.4 all; ring1 140.8-140.9 all; torusns 149.2 / 149.2 / 149.1-149.2; hop3 146.2 / 146.2 / 146.7; ewscatter 139.7 / 140.1 / 142.3 (both passes) |
| VC2 on (job 400) | ew4352 134.6 → 134.9; **torusns 166.9 → 171.1; torusns_scatter 209.2 → 228.3** |
| one ERISC ew4352 (job 401) | 198.3 → 198.0 |
| VC1 census, one pass (job 401) | vc1 mean 159.3 → 157.2 (2HopMerge -9.9 / -5.5, 2Hop -4.2 / -4.4, multicast -3.5; 1-hop and bursts ±0.3); vc1i mean 336.7 → 334.0 (1_1..1_3 -1.6 to -6.9; 1_0 +0.7) |
| UDM fixtures (job 401) | 8/8 and 68/68 |
| inline write then flushed atomic, 1D and 2D (job 401) | main 2/2, 2/2; v4rxo7 2/2, 2/2 |
| validation (job 402) | 13/13 |

**The torus-Y + VC2 loss is S3 (B2), by an isolating A/B** (job 403, two passes reversed). torusns: lead 166.9-167.0,
S2 166.9-167.0, S3 171.1. torusns_scatter: lead 209.2-209.3, S2 209.1-209.2, S3 228.3. S2's `receiver::run` on these
configs differs from the lead's only by register names; S4 changes nothing more for them.

Mechanism, from E1 PC samples (chip 5 / channel 11, the busiest receiver; S2 vs S3 on torusns_scatter with VC2):
- with every channel's cursor a loop local, GCC spills VC0's long-latency loaded values right after their loads;
- each spill store waits for its load (the load's first use moves to the store);
- spill-store cycles 5.2 → 24.0 cycles/packet (+18.8; the row is +19.1):
  - the TX-queue status (`eth_txq_reg_read` ETH_TXQ_STATUS): S2 keeps it in a callee-saved register, S3 stores it at
    once (`sw a5,12(sp)`, 7.4 cycles/packet);
  - the scatter header's chunk_count and chunk_encoding (`lbu …38/39`): S2 spills only the encoding (4.3), S3 spills
    both (`sw …24(sp)` 7.9 and `sw …28(sp)` 6.5).
- This is §6's v4rx2 pattern on today's code.

Per the user's rule, no fallback was built. The options are the user's to choose:
- B1' (placement as before: VC0's cursor alone a loop local, a rename only);
- B2 with VC1/VC2 cursors kept in memory by `keep_in_memory` at their step (a documented primitive; v4rx2m-like,
  §6 data: removed the spills there but moved the stall);
- something else.

**Codec decode (A/B arm):** ewscatter +2.2 and hop3 +0.5 against S4 (a single change, so the cost is the codec decode's).
Hypothesis for the mechanism (not isolated further): the codec reads the column byte as a volatile load and adds a
`zext.b`; `run` grows by 4 instructions on an inner E/W router. ~~Not adopted.~~ Adopted by the user on 2026-11-02 as
it is: v4rxo6c's change (the codec's volatile decode on every router, exit routers' destination mesh read volatile
beside it, v4's own byte logic gone), cost accepted (ewscatter +2.2, hop3 +0.5). No mixed plain/volatile scheme and no
generic-over-volatility codec. The one rule the code states: header bytes are written by the link's DMA, so the router
reads them through the codec's volatile API; the codec owns the decode. It goes in the final chain after the S3
decision, and the census notes its cost on its rows.

**Not run:** the full census (census_runs.sh two1 / two2 / one). It waits for the B2 decision, since the final variant
changes with it.

## 9. Third shape: placement by access frequency (v4rxo8; jobs 404-406; cycles/packet)

The user's third shape, built on S2 (v4rxo4; files changed: router_receiver.hpp, router_single.hpp):
- every channel's per-packet Position (index, slot_addr, ready) is a loop local (`PositionOf<CH>`);
- the per-credit / per-pass CompletionCounts and the slot sources of all channels are one memory object
  (`Completions` of `ChannelCompletions<CH>`), `keep_in_memory(completions)` at the end of each pass;
- one Placement note in the source cites this section.

A/B, one job per pass, two passes in reversed router order (jobs 404 and 405; every row within 0.3 across passes, the
third shape's within 0.1; a range shows the two passes):

| row | lead v4rxv2i3 | S2 v4rxo4 | S3 (B2) v4rxo5 | third shape v4rxo8 | third shape - lead |
| --- | --- | --- | --- | --- | --- |
| ew4352 | 126.9 | 126.9 | 126.9 | 127.2 | +0.3 |
| ewscatter | 139.7 | 139.7 | 139.7-139.8 | 143.5 | **+3.8** |
| torusns, VC2 on | 166.9 | 166.9-167.0 | 171.1 | 173.1 | **+6.2** |
| torusns_scatter, VC2 on | 209.3 | 209.0-209.3 | 228.3 | 220.2 | **+10.9** |
| ew4352, one ERISC | 198.3 | 198.3 | 198.3 | 198.3 | 0 |
| VC1Mesh2HopMerge 16 / 4352 | 139.4 / 144.2 | 139.2 / 144.0 | 129.4 / 138.7 | 138.2 / 143.1 | -1.2 / -1.1 |
| VC1Mesh2Hop, multicast | 239.4, 258.0; 153.2 | 239.0, 257.8; 151.5 | 234.3, 253.8; 148.3 | 239.4, 258.1; 151.6 | 0, +0.1; -1.6 |
| VC1Inter 1_0 16 / 4352 | 208.3 / 209.1 | 208.3 / 209.1 | 208.3 / 209.1 | 211.7 / 212.7 | **+3.4 / +3.6** |
| VC1Inter 1_1 16 / 4352 | 346.1 / 397.0 | 346.1 / 397.1 | 342.6 / 392.4 | 346.9 / 394.6 | +0.8 / -2.4 |
| VC1Inter 1_2 16 / 4352 | 347.3 / 396.0 | 346.3 / 395.4 | 342.7 / 390.1 | 346.2 / 391.1 | -1.1 / -4.9 |
| VC1Inter 1_3 16 / 4352 | 376.0 / 414.3 | 375.1 / 412.6 | 369.4 / 404.2 | 368.2 / 405.8 | -7.8 / -8.5 |
| VC1 census means (job 406, two passes reversed; vc1 13 rows / vc1i 8 rows) | 159.3 / 336.8 | 159.1 / 336.2 | 157.1 / 332.4 | 159.0 / 334.7 | -0.3 / -2.1 |

**Acceptance (within 1-2 cycles/packet of the lead on every row): fails** on ewscatter, torusns, torusns_scatter and
VC1Inter 1_0 (not in the brief's row list; equal in both passes). The VC1 1-hop and burst rows stay within ±0.3.
Per the user's rule, no fallback was built.

**Bound side.** The A/B changes E1 only. The E0 images of S2 and the third shape have byte-identical .text on all 36
two-ERISC configs of each cache (rxoEa, rxoEva), so the loss comes from the receiver's change.

**Mechanism, torusns_scatter.** Evidence: E1 PC samples on chip 5 / channel 11, S2 against the third shape, by
instruction and by source line (router_receiver.hpp lines mapped by diff). The register map is loopvars.py (§10) on the
same config's images.
- Register map. Against S2, the third shape adds three values to callee-saved registers: channel 1's Position
  (s0 slot_addr, s4 index, s8 ready). S2 kept them in its memory record (sp+72..80).
- Two values leave registers:
  - the TX-queue status (S2: s10 over the whole loop). The third shape stores it to sp+4 right after its read
    (`sw a5,4(sp)`, tt_eth_api.h:31) and reloads it in send_credit_frame;
  - VC0's uncredited count (S2: s7). It now lives in `completions` (sp+60): a load and a store in retire per packet
    (:307) and a store in credit_completions (:598).
- The chunk count stays in a register. The encoding byte is spilled in the lead, S2 and the third shape (register
  map). It costs about the same in the samples: 4.3 cycles/packet in S2, 4.1 in the third shape; the lead was not
  sampled.
- PC samples. sp-relative instructions take 11.8 cycles/packet in S2 and 22.4 in the third shape (+10.6; the row is
  +10.9). Where the extra cycles show:
  - The pass-start register reads (two stream registers, two TX-queue registers): 13.7 → 19.8 (+6.1). The spill store
    waits for the TX-queue read, 8.3 cycles/packet at that one instruction. In S2 the first wait is 4.1 cycles, at the
    first use of a stream register (step :629).
  - send_credit_frame's reload and use of the status: +3.0 (:664 in S2 numbering).
  - VC0's count through memory: retire +3.1 at its new line, against -1.2 for S2's channel-1 count (:317).
  - Channel 1's Position in registers saves its record accesses: receive and handle -3.9 (:435/:449/:457/:461).
- This is S3's TX-queue status spill: the same instruction in S3 took 7.4 cycles/packet. It is the one of S3's three
  spills the third shape keeps.
- Net by line: the status lines +9.1, the count through memory about +1.9, channel 1's record -3.9. The other +3.8
  is spread over lines the change does not touch. The largest are routing :92 +2.3, NoC command writes :174 +2.3 and
  the trid wait :1483 -2.5; moved stalls show up this way.
- Hypothesis (not isolated: no variant changes only one piece, and none was built): the TX-queue status spill is
  the main cost, and the completion count through memory the smaller one.

**ewscatter (E-W VC0 configs, no VC1/VC2 on E1).** Static evidence only; this row has no PC samples. On all 36 E1
images, the only stack difference from S2 is the completion counts. VC0's uncredited count and
idle_passes_with_partial_group are now in `completions`, and credit_frame_due is spilled. The S2 loop has 3-6 stores
and 3-6 loads through sp; the third shape has 9 stores and 8 loads. Hypothesis: the uncredited count's load and store
on every packet's retire, on the scatter path, where E1 runs longer per packet.

**torusns** (+6.2) has no PC samples. The TX-queue status spill sits in the same 12 of the 36 E1 configs in S3 and the
third shape (§10). S3 has the same spill and was +4.2 on this row.

## 10. Register-map tool: loopvars.py (study tooling, not CI)

`fabric_exp/consolidation/loopvars.py`. The header documents usage and limits; golden data is in `loopvars_golden/`.
It replaces the scratch txstate/regmap.py with a self-contained rewrite.
- **Loop.** The two-ERISC loop is the largest strongly connected region of receiver::run / sender::run. One ERISC uses
  the whole single::receiver_pass / sender_pass.
- **Method A, instructions.** Every sp-relative load and store in the loop, with its addr2line inline chain and source
  line.
- **Method B, DWARF.** Each variable's location over the loop, from the concrete DIE and its inlined callees, using
  .debug_loclists. Aggregates are split into members by their type.
- **Cross-check.** Each access is matched to the value B places at that slot (slot method) or in the stored or loaded
  register (register method), at the instruction or the next one.
- **Flagged, never guessed.** Accesses neither method names ("unknown"); a value whose scope is not in the access's
  inline chain; values B puts on the stack that no access touches ("DWARF only"). GCC's location lists are
  incomplete, so unknown accesses are expected; they are mostly caller-saved temporaries around deliver_other's call
  (router_local_delivery.hpp:189).
- `--compare LABEL=CACHE ...` maps every router image of each config in kernel caches: E1 as receiver, E0 as sender,
  a one-ERISC image as both.
- `--self-test` checks the known answers and the recorded outputs. The recording pins the tool's hash and the
  toolchain binaries (sfpi 7.91.0, binutils 2.44.20250202, GCC 15.1.0). A normal run warns until the self-test has
  passed for the current pair. The test passed under several Python hash seeds.

**Known answers** (lead, S2 and S3 E1 images of the PC-sampled torus-Y + VC2 config 14894759742743696472):
- S3 stores the TX-queue status (sp+12, `eth_txq_reg_read`, tt_eth_api.h:31), the chunk count (sp+24,
  router_local_delivery.hpp:140) and the encoding (sp+28, :141). These are the PC samples' three spills.
- The lead and S2 store neither the status nor the count.
- **The lead and S2 do store the encoding byte** (sp+8, at :102 and :141). S2's samples measure it at 4.3
  cycles/packet. So the brief's "none on the same paths" holds for two of the three values; the test pins what the
  images show.
- Everything S3 stores that S2 does not: the status, the count, the encoding after each chunk shift (:162, 1.1
  cycles/packet), a NoC command word at `ncrisc_noc_set_transaction_id` (:1485, 1.0), and channel 1's encoding (:141).
  The test pins this list exactly.

**Maps on the A/B candidates.** Data: `loopvars_maps/rxo8_ab/` (per-config compare output per cache, the four full maps
of the PC-sampled config, and the E1 PC sample line reports for S2, S3 and the third shape).

| caches | E1 receiver: sp stores / loads in the loop per config | E0 sender |
| --- | --- | --- |
| VC2 torus (rxoEva, 36 E1 + 36 E0) | lead = S2: 14/18 (24), 12/22 (8), 14/25 (4); S3: 12/19, 13/21, 21/30, 18/31; third shape: 13/14 (24), 11/19 (8), 13/23 (4) | identical in all four builds |
| E-W VC0 (rxoEa) | lead = S2 = S3: 3/3 (12), 6/6 (24); third shape: 9/8 (36) | identical |
| one ERISC (rxoE1a, 36 images) | same loop size and sp counts in all four builds (receiver_pass 569-660 instructions, 3-4 stores / 4-5 loads); the pass state is reached through the state record `r` (memory, not stack) | sender_pass identical in all four (551 instructions, 6/6) |

- **Receiver.** The per-packet VC0 spills in each build:
  - lead and S2: the encoding byte (12 of 36 VC2 configs; deliver's scatter path);
  - S3: the encoding (12), the TX-queue status (12) and the chunk count (4);
  - third shape: the encoding (4), the TX-queue status (the same 12 configs as S3), and VC0's counts in
    `completions` (by design, all 36; E-W loops add stores at retire :307, credit_completions :595/:598,
    slots_due_for_credit :529/:533, run :754).
  - Counted by DWARF value. A count by the store's own source line differs where scheduling moves a spill: for
    example, the status store's innermost line is tt_eth_api.h:31 in only 4 of S3's 12 configs.
- **Sender (all builds; E0 unchanged).** The loop keeps on the stack:
  - its records: forwarding records sent / slot / completed, far.next_slot, round, workers' slot;
  - one per-packet value, the worker channels' header pointer (`send<0>` / `send<4>` header and src at sp+12/16,
    router_sender.hpp:299). Recorded here only; not acted on.
- **Host note for the sender pass (from the VC2 line agent, 2026-11-02).** The host's worker-role arg
  (`SENDER_CH_<i>_IS_WORKER_CHANNEL`) marks every VC2 sender channel as a worker. That is wrong once VC2 has a
  forwarding channel (line mode). It is harmless today, since nothing reads it yet. The main session fixes the host
  rule (the first channel of each VC whose mode has a worker) before the sender pass. When the sender starts reading
  it, re-run the guard's sender maps on the VC2 configs.
- **VC1 census caches (cenvc1-*, 48 E1 + 48 E0 configs).** The lead's images carry other config hashes (host args
  differ), so they are not compared per config; S2, S3 and the third shape are.
  - Receiver sp stores / loads per config: S2 1-2 / 5 (16 configs) and 10-15 / 23-27 (32); S3 1-2 / 5 (16) and 6-8 /
    14-18 (32); third shape 7 / 8 (16) and 13-15 / 18-20 (32).
  - The third shape keeps VC0's counts in `completions` on all 48 and channel 1's on 32. S2's channel-1 Position
    record (32 configs) is gone. There is no TX-queue status spill on these configs.
  - Sender: 36 / 82 on all 48, identical in the three builds.

## 11. Frame-first pass order (v4rxo4f, v4rxo5f; jobs 407-409; cycles/packet)

The idea (user, 2026-11-02): shorten the live range that causes the pressure instead of moving named state. Both
uniform shapes spill the TX-queue status. On first-level-ack routers it is read in the pass's opening register group
but used only by send_credit_frame at the pass's end, so it holds a register across every channel step.

**The order**, in `pass()`:
1. The grouped register reads, fenced as before: every channel's packets-received register, plus the TX-queue status
   where a channel has first-level acks.
2. The credit frame the previous pass left due, sent with the status just read.
3. The channel steps. They set `credit_frame_due` for the next pass.

The rule the code states: nothing read at the start of a pass is held across the channel steps. Credit timing barely
moves: a frame now goes out after the next pass's register reads instead of at the end of its own pass.
`send_credit_frame(credits, queue_seen_free)` now takes whether the grouped read saw the queue free. When it did not,
or the router has no first-level acks, it reads the queue live, as before.

**Edges.**
- **Round end, termination, router commands.**
  - Before: every pass closed with its own frame, so a frame due from a round's last pass was tried before
    `end_round` and before the loop could exit. `end_round` may hand the core to the base firmware for a context
    switch; the loop exits on termination, and in one-ERISC mode on a router command.
  - Teardown (router_init.hpp) sends no credits; a frame the busy queue refused in the last pass was dropped.
  - With frame-first alone, the round's last frame would wait across the context switch and be dropped at exit.
  - So `run()` and `single::run()` call `send_credit_frame_before_round_end` (live queue check) once per round, before
    `end_round`. That keeps today's guarantee exactly.
- **One ERISC.** router_single.hpp runs the same `pass()` as the receiver half, so it gets the same order, plus the same
  round-end call.
- **Routers without first-level acks.** There is no status in the group; the frame's send is the same code, moved to
  the start of the pass. The static check is below.

Variants (on the stage-0 chain, without the codec change):
- v4rxo4f: S2 plus frame-first, the control.
- v4rxo5f: B2 (S3) plus frame-first, the candidate.
- Each differs from its base in router_receiver.hpp and router_single.hpp only.

A/B (cycles/packet, two passes in reversed order, every row within 0.2 across passes; jobs 407 and 408; the VC1
census is job 409):

| row | lead | S2 | S3 (B2) | S2 + frame-first | B2 + frame-first | B2 + frame-first - lead |
| --- | --- | --- | --- | --- | --- | --- |
| ew4352 | 126.9 | 126.9 | 126.9 | 127.0 | 127.0 | +0.1 |
| ewscatter | 139.7 | 139.7-139.8 | 139.7 | 140.6 | 140.6 | +0.9 |
| ring1 | 140.9 | 140.9 | 140.9 | 145.0 | 145.0-145.1 | **+4.1** |
| torusns, VC2 on | 166.9 | 166.9-167.0 | 171.1-171.2 | 167.2-167.3 | 170.6-170.7 | **+3.7** |
| torusns_scatter, VC2 on | 209.2-209.3 | 209.1-209.2 | 228.2-228.3 | 206.5-206.6 | 216.8-216.9 | **+7.6** |
| ew4352, one ERISC | 198.3 | 198.3 | 198.3 | 203.8-203.9 | 203.8-203.9 | **+5.5** |
| VC1Mesh2HopMerge 16 / 4352 | 139.5 / 144.2 | 139.3 / 144.0 | 129.8 / 138.8 | 134.6 / 140.7 | 131.1 / 139.1 | -8.4 / -5.1 |
| VC1Mesh2Hop 16 / 4352; multicast | 239.4 / 257.9; 153.4 | 239.2 / 257.8; 151.5 | 234.3 / 253.8; 148.3 | 238.3 / 257.7; 150.1 | 233.4 / 252.6; 148.7 | -6.0 / -5.3; -4.7 |
| VC1Inter 1_0 16 / 4352 | 208.3 / 209.1 | 208.3 / 209.1 | 208.3 / 209.1 | 210.8 / 211.6 | 210.8 / 211.5 | **+2.5 / +2.4** |
| VC1Inter 1_1 16 / 4352 | 346.1 / 397.0 | 346.2 / 397.2 | 342.5 / 392.3 | 349.5 / 398.9 | 345.5 / 393.1 | -0.6 / -3.9 |
| VC1Inter 1_2 16 / 4352 | 347.3 / 395.9 | 346.2 / 395.3 | 342.7 / 390.0 | 348.9 / 396.6 | 344.9 / 391.8 | -2.4 / -4.1 |
| VC1Inter 1_3 16 / 4352 | 376.0 / 414.2 | 375.3 / 412.6 | 369.5 / 404.1 | 377.4 / 414.2 | 368.9 / 405.7 | -7.1 / -8.5 |
| VC1 means, vc1 / vc1i (job 409) | 159.3 / 336.7 | 159.1 / 336.3 | 157.2 / 332.3 | 158.4 / 338.5 | 157.2 / 334.0 | -2.1 / -2.7 |

**Acceptance (B2 + frame-first within 1-2 cycles/packet of the lead on every row): fails** on ring1, torusns,
torusns_scatter and one-ERISC ew4352, and narrowly on VC1Inter 1_0 (+2.5 / +2.4, both placements). The VC1 1-hop and
burst rows stay within +0.5; the B2 gains on 2HopMerge, 2Hop, multicast and VC1Inter 1_1..1_3 remain. Per the user's rule, no fallback; S4, the codec and C1 are not carried.

**The order alone** (S2 → S2 + frame-first, same placement): ring1 +4.1, one-ERISC ew4352 +5.5, VC1Inter 1_0
+2.5, ewscatter +0.9, torusns +0.3, torusns_scatter -2.6, 2HopMerge -4.7 / -3.3. The order is not free on its own.

**Static check, routers without first-level acks** (coverage builds at -O3, 2,974 configs each, both variants build the
same 2,713 as their bases; tool: per E1 image, the pass loop = receiver::run's largest loop with the round's end_round
call cut, and the round loop that includes the round-end send; one ERISC: single::receiver_pass and single::run):
- Two ERISCs, 506 images. Pass loop:
  - -1 or -2 instructions with no stack change on most 2D images (2D 172, 2D with all instrumentation and Galaxy 2D
    50, BW telemetry 32);
  - -3 with 4 fewer stack accesses on 1D (52);
  - +1 to -14 on VC1;
  - -9 to -28 on 2D / torus with VC2, with -4 to +3 stack accesses;
  - -1 with +2 stack accesses on 2D trim (12);
  - +5 to +10 only on BW-telemetry and code-profiling builds.
  The round loop grows +2 to +27 instructions: the round-end send, once per 32 passes. Per pass the order costs these
  routers about nothing; the device agrees (ew4352 +0.1).
- One ERISC, 264 images: single::receiver_pass is +3 instructions (174; +2 / +7 / -1 on the rest). The pass now
  starts by loading `credit_frame_due` from the state record and testing it at once (`lbu`, `beqz`); single::run grows
  +20 (the round-end send). On the device this row loses 5.5 (no samples; hypothesis: the flag's load-use and the
  live TX-queue test now at the pass's head).
- With first-level acks the pass loop shrinks: -8 to -31 instructions on two ERISCs, -4 to -33 on one.

**Mechanism, torusns_scatter** (E1 PC samples, chip 5 / channel 11, B2 + frame-first against S2; register map by
loopvars.py on the same config):
- Gone: the TX-queue status spill and the chunk count spill. Stack loads and stores take 10.2 cycles/packet (S2 11.8,
  S3 32.0); the encoding byte is still spilled, as in the lead and S2 (3.8).
- **Still spilled: the round's pass counter** (`n` in run's `for`, at sp+12: a load, decrement, store and branch every
  pass; 5.4 cycles/packet at those four instructions, +3.2 by line against S2). Outside the pass loop the slot holds
  `credit_frame_due`, which inside the loop sits in a temporary register (t2). B2's register pressure spills one
  value either way: the status in S3, the pass counter here.
- **New with the order:** the frame's busy test now follows the TX-queue status read at once. `slli a2,a3,0xf`, the
  status's first use, waits 5.6 cycles/packet; send_credit_frame's lines go 2.6 → 7.8. The old order's first wait, at a
  stream register's first use (step :629), drops 2.5. The read's latency used to hide behind the channel steps; the
  frame now waits for it.
- Hypothesis (not isolated; no variant changes one piece): the loss is the exposed status read (about +3 net here)
  plus the pass counter through memory (about +3), with the rest spread (NoC command writes :174 +3.0).

**ring1** (E1-bound, V4_BOUNDS §3; +4.1 with either placement). No samples. Register map: the frame-first variants keep
`credit_frame_due` on the stack on all 36 ring configs (stores 4 → 6: after each send and at the round's end).
Hypothesis: the same exposed status wait before the frame (ring1 sends a frame most passes) plus the flag through the
stack.

**Earlier answer (missed before this probe; flagged here).** V4_RECEIVER_SHAPE.md §7 tried frame-first as `v4rx1f`
(job 284, rejected): +4 on first-level-ack rows with two ERISCs and +5.2 to +9.7 on one ERISC, with the same hypothesis
(the frame waits for the next pass's group). This probe reproduces it on today's code (ring1 +4.1, one ERISC +5.5); the
new part is the register map (the status no longer spills; the pass counter does under B2).

**What this says about the idea.** Shortening the status's live range works as intended: the status no longer
spills. But it trades a register for the read's latency, and B2's pressure spills the next value (the pass counter).
The comment pair the coordinator asked for (a pass first sends the credits the previous pass left due; a round ends by
sending any still due, before the core may yield or the loop exits) is not applied to the measured variants, since
the probe stops here.

Register maps of every variant, by loopvars.py:
- VC2 torus, E-W / ring and one-ERISC caches: all five variants. VC1 census caches: S2, S3 and both frame-first
  variants; the lead's VC1 configs carry other hashes.
- Senders are identical in all builds.
- B2 + frame-first's receivers on the 32 larger VC1 configs: 7-9 stores / 15-19 loads through sp (S3 6-8 / 14-18,
  S2 10-15 / 23-27).
- Files: `loopvars_maps/frame_first/`, with the sampled config's maps, the PC line report and the static checks
  (ffstatic.py).

## 12. Register budget: inventory and proposals (2026-11-02, offline; for the user's pick)

User's direction: readability, then stability, then perf; acceptance starts at 3-5 cycles/packet on any row. S2 is at
parity only because VC0 keeps its registers in today's code shape. Native worker fan-in (line VCs) and flush-defer
will add state.

**Method.** `loopvars.py --live` (new): register and stack-slot liveness over E1's receiver loop, from the
instructions, named by DWARF. Run on every E1 coverage image (-O3) of the lead, S2 and S3.
- **Carried:** the values live at the pass loop's head.
- **Peak:** the most values live at any one instruction of the pass loop.
- **Classes:** from DWARF names and the defining instruction's source line.
- Data and scripts: `loopvars_maps/budget/` (inventory per build, live ranges and hot-point maps of the sampled torus
  config).

**Budget.** The hot path has no call: `deliver_other` is cold and `end_round` runs once per round, and GCC saves and
restores around both. So all 27 allocatable registers are usable: s0-s11 (12) plus a0-a7 and t0-t6 (15).

**Demand by family** (S3 = every cursor a loop local, the uniform rule; S2 today; lead equal to S2 except where noted):

| family | carried, S3 (registers + slots) | peak, S3 | peak, S2 | carried classes (S3) |
| --- | --- | --- | --- | --- |
| 1D, ring | 18 (17-18 + 0-1) | 26-28 | 26-28 | channel 4, pass 2, invariants 7, promoted 4, other 1 |
| 2D, Galaxy 2D | 19 (19 + 0) | 27-28 | 27-28 | channel 4, pass 2, invariants 4-7, promoted 5-8 |
| torus | 17-18 (16 + 1-2) | 29-30 | 29-30 | channel 4, pass 2, invariants 6-7, promoted 5 |
| 2D + VC2 | 20-23 (18-20 + 1-3) | 30-32 | 34-35 | channel 9, pass 2, invariants 3-5, promoted 5-7 |
| torus + VC2 | 19-22 (16-20 + 1-3) | 30-34 | 34-35 | channel 8-9, pass 2, invariants 3-8, promoted 2-7 |
| VC1 | 20-21 (16-17 + 4-5) | 28-32 | 28-36 | channel 4-9, pass 2, invariants 5-9, promoted 2-5 |
| one ERISC (receiver_pass; state in its record) | 3-8 | 15-21 | 15-21 | pointers, promoted 1-5 |

- **Channel:** each cursor's ready, slot_addr, index and uncredited, plus idle_passes on grouped rings.
- **Pass:** the pass counter and `credit_frame_due`.
- **Invariants:** hoisted constants (stream, NoC and TX-queue register bases, credit block, 0x01010101), pointers
  (downstreams) and the coordinates.
- **Promoted:** memory GCC keeps in registers across the loop by its own choice: the NoC API's software write counters
  (`noc_nonposted_writes_num_issued/acked`, `noc_posted_writes_num_issued`), the downstream rings' next-slot
  pointers, the credit counters' addresses.
- **Per-pass reads, not carried:** each channel's packets-received count, live from the pass start to its step, and,
  on first-level-ack routers, the TX-queue status, live from the pass start to the frame.
- **Per-packet temporaries at the peak:** about 10-12 (header pointer and word, route word or action, payload size,
  NoC addresses, trid, chunk count and encoding).
- The peak is always inside `deliver`'s NoC programming, and there all 27 registers are in use on every two-ERISC
  family. Single-channel routers sit at the budget (0-3 values on the stack). Two-channel routers are 1-10 over, and
  GCC chooses what goes to the stack. One ERISC, where every channel's state lives in a record between passes, peaks
  at 15-21: 6-12 registers of headroom.

**Each value at the hot points** (frequency; live range as the share of the pass loop's instructions where it is live,
S3 torus + VC2, `live_ranges_s3_torusy_vc2.txt`; cost in memory; cost with its latency exposed):

| value | frequency | live range | kept in memory | latency exposed |
| --- | --- | --- | --- | --- |
| cursor `ready`, `uncredited` | per packet; per credit | carried, 86-91% | ~2-3 per packet per field | - |
| cursor `index`, `slot_addr` | per packet | carried (46-56% under the cursor's name, the rest under aliases: `group`, `header`) | ~2-3 per packet per field | - |
| `idle_passes_with_partial_group` | per idle pass, grouped rings | carried | ~1-2 per idle pass | - |
| `credit_frame_due` | per pass | carried, 81% | ~5 per pass if used at once | - |
| the pass counter `n` | per pass | carried, 90% | 5.4 per pass (measured) | - |
| channel k's packets-received count | per pass | pass start to step k (channel 1: 50%) | a store right after the group waits for it, ~4 per pass | first use at once: 4-7 |
| TX-queue status (first-level acks) | per pass | pass start to the frame, 87% | 7.4-8.3 (spill right after the read) | 5.6 |
| header word, route bytes, chunk count, encoding | per packet | receive to handle / deliver, 12-42% | 4.3-7.9 (spill right after the L1 load) | the L1 load, ~8 |
| NoC software counters, downstream next slot (promoted) | per packet on the write / forward path | carried, 48-71% | ~2 per update (load + store) | - |
| invariant constants, pointers, coordinates | per pass | carried, 88-98% | rebuilt 1-2 per use, or a stack reload ~2 | - |

**Growth** (measured between families; S3):
- Per added receiver channel: carried channel state +4 or +5 (4 → 8-9); +1 per-pass value (its packets-received
  count); peak +1 to +4.
- Per added line VC: a forwarding channel (+5 to +6), plus its downstream ring's promoted pointers (+1 to +2, GCC's
  choice). On today's placement two line VCs look like torus + VC2, the case that lost (S3: TX-queue status spill;
  third shape; frame-first: pass counter spill).
- Per flush-defer record: 0 carried (the record is in memory). +1 load and 1-2 short temporaries at the step's start,
  before the hot path; the resolver is out of line (a cold call, saved around like `deliver_other`).
- So the growth that makes today's placement fragile is the cursor: +4-5 registers per channel. Everything else is
  flat or small.

**Cost of each class when it is not in a register** (measured unless marked):
- Per-packet field kept in memory: about 2-3 cycles/packet per field. The third shape's VC0 uncredited count: retire's
  load + store about 2, credit's store about 1, line-level net +1.9.
- Per-pass value in memory used at once: about 5. The B2 + frame-first pass counter: lw / addi / sw / bnez 5.4.
- A long-latency value spilled right after its read, so the store waits for it: 6-8. TX-queue status 7.4 / 8.3;
  scatter chunk count 7.9; encoding 4.3-6.5.
- The same read used at once instead of late (exposed): TX-queue status 5.6 (frame-first).
- Invariants rebuilt at each use: about 1-2 per use. v4rxc1 rebuilt several and lost 7-9 cycles/packet (V4_RECEIVER_SHAPE
  §23).
- Grouped local-memory loads: about 1 cycle each, about 2 if used at once; stores about 1, not measured (V4_STATE §7,
  RESULTS §17).
- Precedent (V4_STATE §10-11): on the sender, parking the send path in memory cost +3.7 to +16 cycles/packet (v4pl3).
  Copying it into locals at the visit and back once (v4pl4, the shape below) recovered about 60% and still cost
  1.6-8, VC1 included. "Parking costs a few cycles" was marked WRONG there. The user then kept the sender's send path
  with the compiler. The same estimate method is used below, so the predictions carry the same risk.

**Proposals** (predictions are estimates, counted accesses times the costs above; E0- and wire-bound rows have E1
slack of about 20 cycles, so they move 0 under any of these):

**WRONG (§13, the P2 probe, jobs 410-411):**
- The predictions below missed by 2-4x on every E1-bound row, and the VC1 sign was wrong.
- The "about 20 cycles of E1 slack" was also wrong: the E0-bound ew4352 moved +1.7 when E1 gained about 10-12
  cycles.
- The per-field cost (about 2 cycles) held. The access counts did not: see §13.

1. **P1: every channel's state in memory between steps** (main's shape).
   - Rule: each receiver channel's cursor lives in memory between steps, one object for every channel. A step loads
     its channel's cursor as a group into a local, works on the local, stores it back, and ends with
     `keep_in_memory`.
   - Code: `template <size_t CH> struct ChannelState { Cursor cursor = cursor_start<CH>(); SlotSourcesOf<CH> sources{}; };`
     `step<CH>(ChannelState<CH>& state, ...) { Cursor cursor = state.cursor; ...today's body on cursor...;`
     `state.cursor = cursor; keep_in_memory(state); }`. `pass(states, credit_frame_due, credits, downstreams, me)`.
     One-ERISC `ReceiverState` holds the same `states`.
   - Readability against S2: better. One rule and one state object, no channel singled out, and the placement note
     in one place.
   - Cost: about 4-5 loads and 4-5 stores per handled packet on the active channel, plus 2-3 loads per idle channel
     per pass.
   - Predicted:
     - E1-bound first-level-ack rows (ring1, torus N/S, torus with VC2): +4 to +10;
     - 2D / torus scatter: +4 to +9;
     - VC1 rows (the lead already keeps VC1's cursor in memory): -2 to +3;
     - one ERISC: 0 (already this shape).
   - Demand: about one-ERISC's 15-21 plus 3-5 pass-level values, i.e. 18-26 of 27. Growth +1 per channel and per line
     VC (the packets-received count), 0 per flush-defer record.
2. **P2: P1 with the cursor trimmed.**
   - Rule: P1, with `slot_addr` derived from `index` by the existing `Channel<CH>::slot_addr(index)`. A cursor is
     {index, ready, uncredited} plus idle_passes on grouped rings, touched only on idle passes.
   - Readability: better than P1. The ring index is the cursor, and `advance` is one line.
   - Cost: 3 loads and 3 stores per handled packet (2 + 2 + the idle counter on grouped rings), plus a multiply or
     shift-add per header address.
   - Predicted:
     - first-level-ack E1-bound rows: +3 to +8;
     - scatter: +3 to +7;
     - VC1 rows: -2 to +2;
     - one ERISC: 0 to -1.
   - Demand: P1's, minus one register while a step runs.
   - Not trimmed: non-first-level-ack `ready` (the kept count). Replacing it with the pass's packets-received read
     depends on a stream-register update being visible to the next pass's read; not verified, so not proposed.
3. **P3: one rule by count, not by channel.**
   - Rule: a router that runs one receiver channel keeps that cursor a loop local; a router that runs several keeps
     every channel's cursor in memory between steps (P2's form).
   - Code: `using ChannelStates = std::conditional_t<receiver_channels_run == 1, LocalCursor, InMemoryCursors>`, plus
     the two step forms.
   - Readability: worse than P1/P2 (two placements, one condition); better than S2 (the condition is a count, not a VC).
   - Predicted:
     - single-channel rows (1D, ring, 2D, torus, Galaxy): 0;
     - two-channel E1-bound rows (torus with VC2): +3 to +8;
     - VC1: -2 to +2.
   - Demand: single-channel routers stay at the budget (26-30 against 27), with no headroom for added state there;
     multi-channel routers are flat (+1 per channel).
4. **P4 (a probe after P1/P2, not a shape on its own): amortize the state traffic.**
   - Rule: a step that has loaded its cursor handles up to K ready packets (for example 4) before storing it.
   - E1-bound rows always have a backlog, so the per-packet traffic drops to about 1/K: P2's cost becomes about +1 to
     +3.
   - Risks:
     - scheduling changes: a channel holds E1 for up to K packets;
     - a pass's frame carries up to K completions, so credits leave later. Held credits collapsed rings under bubble
       flow control (RESULTS §98).
   - Needs its own A/B.

Not proposed:
- S3 or the third shape plus fixes (growth stays +3 to +5 per channel);
- frame-first (§11);
- making the NoC API's counters memory-resident (the API's globals; they are flat, 2-3 values).

**Before the pick, the cheapest isolating measurement (my suggestion; not run).** A P2 probe (study only) on 4-5 rows
calibrates these estimates:
- rows: ring1, torusns with VC2, ewscatter, VC1 2HopMerge, one-ERISC ew4352;
- about 15 minutes of device time, with loopvars headroom checks offline.
v4pl4's precedent says the estimate method can miss by 2-3x.

**What transfers to the sender pass.** `--live --role sender` on S2's E0 images:
- carried: 13-15 registers, plus 23-39 slots that are its records in memory by design;
- peak: 22-24 registers.
- The sender already has headroom because its forwarding records live in memory (the class rule, V4_STATE §9). Its
  send path stays with the compiler (v4pl4's 1.6-8 cost).
- A second worker channel adds a worker's state (5 values, V4_STATE §3), the same per-channel growth as a receiver
  cursor. Whatever the user picks here (P1/P2's "in memory between visits, copied in and out") is the shape the sender
  would take for its two-worker routers, at v4pl4's measured price.
- The same inventory (`inventory.py` with `--role sender`) applies.

## Planned: one copy of the per-VC downstream init (user, 2026-11-02)

Measured on the lead: `init_downstreams<VC>` / `open_downstreams<VC>` are copied per VC with forwarding,
~70-110 instructions each (VC1 adds +122 on E1). They are templated only because Downstreams<VC> sizes its arrays from
DownstreamTraits<VC>. The element DownstreamLink is non-templated and `open_downstream` is already one shared
out-of-line function.
- Plan: one non-template function taking (links, count, the VC's runtime args), called per VC.
- When: a small stage after the placement decision lands, or with the 1D line-VC router work if code space gets
  tight.
- Cost: none per packet. Checks: byte comparison of the loop plus the register-map guard.

## Planned: single-source receiver rings (user, 2026-11-02; not built)

A receiver ring all of whose packets come from one far sender channel needs no per-slot sources:
- no `SlotSources` record of each retired slot's sender channel;
- no group-sources test when a group is credited; every credit goes to that one channel, a compile-time constant.

Rings fed by several far sender channels keep today's path.
- Needs one host fact: the far sender channels per VC after channel trimming (today the kernel learns a packet's
  source only from its header).
- When: after the receiver lead is settled.
- Checks: coverage compare (only single-source rings change), the register-map guard and headroom, and the census.

## Planned: N/S floor stages (user, 2026-11-02; not built)

Background: V4_NS_FLOOR.md §2-§3.
- Order: after the v4rxf chain and after the command-buffer-2 fix (c).
- Each stage gets its own A/B on the Galaxy N/S rows, with E/W controls plus rings and tori, and the headroom check.

**N/S fix 1: partial presets on routers with several live downstreams.**
- The forward keeps its invariant fields preset on its data command buffer (control word and the others). It writes
  only the per-packet fields: source, destination address, coordinate, length and go. Local writes already work this
  way, with `noc_write_with_state<..., CQ_NOC_SNDL, ...>`.
- The credit gets its own command buffer: buffer 1, free on multi-downstream routers.
  - Preset: control word, AT_DATA (the -1 increment), AT_LEN_BE and TARG_ADDR_MID.
  - Per forward it writes only TARG_ADDR_LO (the downstream's free-slots register), TARG_ADDR_COORDINATE and go.
  - Its ready read pairs with the data buffer's.
- It does not share buffer 2 with the data write. A write's destination lives in RET_ADDR_* and an inline write's in
  TARG_ADDR_*, and their control words differ, so nothing carries over.
- Expected (estimate only): part of the ~37 E1 cycles. The credit might fall from ~18 to perhaps 6-8, and the data
  writes need fewer register writes.
- **Design note: V4_NS_PRESETS.md (2026-11-02).** Offline drafts on v4rxg3: v4rxnsa (option (a), multi-downstream
  routers only) and v4rxnsb (option (b), every forward), with one additive inline-write API form; code shapes, buffer
  ownership per router shape, static numbers, predictions, measurement plan and open questions. **Outcome
  (2026-11-03): built as v4rxnsf2 (compile-time ForwardCommands: single-downstream routers byte-identical to the lead,
  multi-downstream partial presets), validated; Galaxy N/S rows at 2 KiB slots 172.9 -> 136.8 (V4_NS_PRESETS §8.9).**

**N/S fix 2 (probe): the next packet's route word loaded ahead into the channel's state.** Design note:
V4_ROUTE_PRELOAD.md (2026-11-02; with the strict flush-defer).
- At the end of a channel's step, and only when the pass's count shows the next slot holds a packet, issue the L1 load
  of the next packet's route word.
- Store it into the channel's state after the step's remaining work, so the store does not stall on the load.
- The next pass's admission reads the word from local memory instead of waiting on the header load (the 7.9 route
  word plus the 17.2 admission stall, V4_NS_FLOOR.md §2).
- Precedent: v4rx1rg (V4_RECEIVER_SHAPE.md §14) stored header fields at ack time and lost on rings (+5.8: routing
  fields spilled and reloaded twice). That cause is gone: 1D routing is one word now, and the guard catches spills.
  Expectations are still modest.

## 13. P2 calibration probe (v4rxp2; jobs 410-411; study only)

**Variant.** v4rxp2 = S2 (v4rxo4: the stage-0 chain, S2's downstreams, no codec) with P2 written as it would read:
- every channel's state (`ChannelState<CH>`: cursor and slot sources) in memory between steps, one object
  (`ChannelStates`);
- a step copies the cursor into a local (`Cursor cursor = state.cursor`), works on the copy, then writes it back
  (`state.cursor = cursor; keep_in_memory(state)`);
- the cursor is {ready, index} plus the completion counts; the slot's address is `Channel<CH>::slot_addr(index)`;
- the one-ERISC `ReceiverState` holds the same `ChannelStates`.

Offline: it builds the same 2,713 of 2,974 configs as S2. Two-ERISC E0 images are byte-identical, so the A/B changes
E1 only.

**Rows** (cycles/packet, two passes in reversed order, every row within 0.2 across passes):

| row | lead | S2 | P2 | P2 - lead | predicted (§12) |
| --- | --- | --- | --- | --- | --- |
| ew4352 (E0-bound control) | 126.9 | 126.9 | 128.6 | **+1.7** | 0 |
| ewscatter | 139.7-139.8 | 139.7-139.8 | 152.2 | **+12.5** | +3 to +7 |
| ring1 | 140.9 | 140.9 | 153.0 | **+12.1** | +3 to +8 |
| torusns, VC2 on | 166.9-167.0 | 166.9 | 185.6 | **+18.7** | +3 to +8 |
| torusns_scatter, VC2 on | 209.1-209.3 | 209.3 | 228.9-229.0 | **+19.8** | +3 to +7 |
| ew4352, one ERISC | 198.3 | 198.3 | 200.1 | **+1.8** | 0 to -1 |
| VC1Mesh2HopMerge 16 / 4352 | 139.4 / 144.2 | 139.2 / 144.0 | 157.8 / 160.2 | **+18.4 / +16.0** | -2 to +2 |
| VC1Mesh2Hop 16 / 4352; multicast | 239.3 / 258.0; 153.1 | 239.0 / 258.0; 151.5 | 251.2 / 268.8; 166.7 | +11.9 / +10.8; +13.6 | -2 to +2 |
| VC1Inter 1_0 16 / 4352 | 208.3 / 209.1 | 208.3 / 209.1 | 225.5 / 226.2 | +17.2 / +17.1 | |
| VC1Inter 1_3 16 / 4352 | 375.9 / 414.1 | 375.4 / 412.3 | 368.4 / 408.7 | -7.5 / -5.4 | |
| VC1 means, vc1 / vc1i | 159.3 / 336.7 | 159.1 / 336.3 | 165.9 / 341.4 | +6.6 / +4.7 | |

**Where the cycles go** (E1 PC samples, torusns_scatter with VC2, chip 5 / channel 11, against S2's from job 403;
per-line deltas with router_receiver.hpp mapped by diff):
- The copy-in loads (step's first line, :626): 7.1 cycles/packet. There are 8 of them: VC0's four fields and the idle
  VC2 channel's four, every pass. About 0.9 each.
- The write-back stores (the step's last line, :642): 7.9 cycles/packet for 8 stores. About 1.0 each.
- Together 15.0 of the row's +19.7: **about 1.9 cycles per field per channel per pass**, as §12 estimated per field.
- The slot address derived from a loaded index:
  - `Channel::slot_addr`'s multiply +2.5 and `next_index` +2.0, against `advance` -3.3;
  - the header word's load now waits behind it: `load_word` +3.2.
  - Net about +4 on the per-packet critical path.
- Spread elsewhere: NoC register writes +2.9, the stream register's first use +2.4, deliver +2.4. S2's channel-1
  record accesses are gone: -4.8.

**Register maps** (loopvars.py, the same config; `loopvars_maps/p2_probe/`):
- The cursors leave registers as intended. Carried values: 22 (15 registers + 7 slots) → 21 (11 + 10). Every cursor
  field is in its slot (VC0's at sp+92..104, VC2's at sp+124..136).
- The peak stays 35 (27 registers + 8 slots). GCC fills the freed registers with promoted memory and constants, as
  §12 expected (on ring routers the promoted values go from 4 to 8).
- The encoding byte is still spilled (4.2).

**What the miss says about the estimates.**
- The per-field cost held: about 1.9 measured against 2-3 estimated.
- The access count was wrong in three ways:
  1. `keep_in_memory` after the write-back makes the object an input and an output of the asm, so GCC stores every
     field every pass, changed or not.
  2. Every channel pays the copy every pass, idle or not. I had counted 2-3 loads for an idle channel and no stores.
  3. Deriving the slot address from a loaded index puts a load and a multiply in front of the header load: about +4
     per handled packet that I had counted as free.
- The VC1 sign: the lead already keeps VC1's cursor in memory, but only touches it where used. P2 adds VC0's full
  copy every pass on rows where VC0 is idle, and VC1's full copy where the lead touched a few fields.

**Re-estimates with the calibrated costs** (about 1.9 per field copied; still estimates):
- **P1** (five fields, slot address stored, copy every pass):
  - one channel about +10;
  - two channels about +19;
  - three about +29;
  - E0-bound rows +1 to +2.
  - The cost grows about 9.5 per channel per pass: the register demand is flat, the cycles are not.
- **P2:** as measured (+12 one channel, +16 to +20 two).
- **P2 written to copy only what a step changes and to test an idle channel from the pass group first** (not built):
  - the busy channel about +8 per handled packet (4 loads, 3-4 stores, the address derivation; less without it);
  - each idle channel +2 to +4 per pass;
  - so one channel about +8, two channels about +10 to +14.
- **P3:** single-channel rows 0; multi-channel rows as the line above (+10 to +14).
- **P4** (K packets per step): the busy channel's cost divided by the backlog. One-channel E1-bound rows about +2 to
  +4; idle channels' cost unchanged, so torus with VC2 about +4 to +7.
- Every form that keeps per-packet state in memory between steps lands above 3-5 cycles/packet on E1-bound rows.
  Exceptions: P3 on single-channel rows (unchanged, but no headroom there), and possibly P4 on single-channel rows.

Study only: no adoption build, no further variants; the user picks the shape.

## 14. Packed cursors and the headroom test (jobs 415-417; probes for the user's pick, no adoption build)

The user's question: a cursor's bounded counts fit one 32-bit word. `index` < num_slots ≤ 255, `ready` ≤ num_slots,
`uncredited` ≤ num_slots, and the idle passes ≤ idle_passes_before_partial_credit. Only the slot address is wide.

**Variants.** All are on the stage-0 chain with S2's downstreams, without the codec.
- **S2-explicit (v4rxs2x).** S2's placement with the rule stated:
  - the first channel's cursor is a loop local;
  - `ChannelMemory` holds every channel's slot sources and the later channels' records, with
    `keep_in_memory(memory.chN)` after each later channel's step.
  - Byte compare against S2: 1,538 of 1,696 shared images are identical (every single-channel image). The 158 that
    differ are E1 images with a second channel: 2D and torus with VC2, 14 of 22 VC1 images, and one ERISC with VC2.
- **B2-packed (v4rxb2p).** Every cursor a loop local (B2), with the counts packed:
  `class Cursor { uint32_t slot_addr; ... uint32_t counts_; }`.
  - Named operations: `ready()`, `index()`, `uncredited()`, `idle_passes()`, `add_ready`, `set_ready`, `credit`,
    `count_idle_pass(limit)` (saturating, so the byte never carries), `clear_idle_passes`, and
    `retire(next_index, next_slot_addr)`.
  - The layout is stated in one place (ready 0-7, uncredited 8-15, idle 16-23, index 24-31), with explicit
    shift/mask accessors and no bitfields.
  - The slot address is kept as a field (P2 measured deriving it at about +4).
- **P2-packed (v4rxp2p).** Every channel's state in memory between steps (`ChannelStates`), packed the same way.
  - A step is skipped without touching memory when nothing arrived and the channel's bit in `channels_with_work`
    (one loop local for all channels) is clear.
  - Otherwise the step copies the cursor, works on it, and writes back only if the counts word or the slot address
    changed, with `keep_in_memory` only on that path.
  - The work bit is needed: skipping on "nothing arrived" alone would strand ready packets (first-level acks consume
    the register) and uncredited slots (completions with no new arrival), and the far sender would wait forever.

**Device** (cycles/packet, lead / S2 / S2-explicit / B2-packed / P2-packed, two passes in reversed order; every row
within 0.2):

| row | lead | S2 | S2-explicit | B2-packed | P2-packed |
| --- | --- | --- | --- | --- | --- |
| ew4352 (E0-bound) | 126.9 | 126.9 | 126.9 (0) | 127.9 (+1.0) | 130.5 (+3.6) |
| ewscatter | 139.8 | 139.7-139.8 | 139.7-139.8 (0) | 147.0 (+7.2) | 162.6 (+22.8) |
| ring1 | 140.9 | 140.9 | 140.9 (0) | 152.5-152.6 (+11.7) | 162.6 (+21.7) |
| torusns, VC2 on | 166.9-167.0 | 166.9-167.0 | 167.0 (0) | 173.9-174.0 (+7.0) | 185.3 (+18.4) |
| torusns_scatter, VC2 on | 209.1-209.3 | 209.2-209.3 | 209.1-209.2 (0) | 219.5 (+10.3) | 226.8 (+17.6) |
| ew4352, one ERISC | 198.3 | 198.3 | 198.3 (0) | 199.4 (+1.1) | 215.3-215.4 (+17.0) |
| VC1Mesh2HopMerge 16 / 4352 | 139.5 / 144.3 | 139.2 / 144.0 | 139.2 / 143.9-144.0 (0) | 148.3 / 151.7 (+8.8 / +7.4) | 156.4 / 159.6 (+16.9 / +15.3) |
| VC1Inter 1_0 16 / 4352 | 208.3 / 209.1 | 208.3 / 209.1 | 208.3 / 209.1 (0) | 215.1 / 216.2-216.3 (+6.8 / +7.2) | 233.6 / 234.3 (+25.3 / +25.2) |
| VC1 means, vc1 / vc1i | 159.3 / 336.7 | 159.2 / 336.2-336.3 | 159.1-159.2 / 336.3 | 162.5-162.6 / 340.0 | 166.5 / 347.5 |

**Where the packed variants' cycles go** (E1 PC samples on the busy receiver 0/6/E1, against the lead's from job 358;
`loopvars_maps/packed/pcs_*`):
- **B2-packed, ring1:** the cursor's accessors take 17.1 cycles/packet (`Cursor::retire` 4.2, `retire_slot` 4.2,
  `index` 3.3, `field` 3.2, `add_ready` 1.2, `credit` 1.0).
- **P2-packed, ewscatter:** the accessors take 18.6, plus the step's copy, compare and write-back lines (step 17.2,
  against 2.5 in the lead).
- Hypothesis (not isolated): every update of a packed count is a read-modify-write of the same register, so the
  updates serialize; the first use of a freshly loaded word is a shift and mask. Packing trades registers for a
  dependency chain on every packet.
- The stack traffic does fall (2D loop: 6 loads and 6 stores → 1 and 1), but the ALU chain costs more.

**Headroom test** (`loopvars.py --headroom`, a standing guard check; its known answers in the self-test: S2 → S3 moves
exactly S3's six hot values, S2 → S2 none).
- Synthetic additions, each built separately on the 1D, ring, 2D, torus, 2D + VC2, torus + VC2 and VC1 cases
  (276 images; generator `loopvars_maps/packed/synth_overlays.py`):
  - **x1 / x2:** one or two values carried from pass to pass, opaque to GCC (`asm volatile("" : "+r"(v))` once per
    pass);
  - **fd:** a flush-defer-sized record per channel in memory ({NoC address, value, waiting}), tested at the step's
    start and settled out of line.
- **Stability criterion:** no hot value newly on the stack. Stores into objects kept in memory, invariant pointers and
  coordinates, and saves around an out-of-line call do not count.
- **Second line VC:** not shaped offline. A 1D config with two forwarding receiver channels needs about 30 host args
  and receiver/sender paths that 1D does not build today. Its closest equivalents are already in the test: VC1 (a
  second forwarding channel) and torus + VC2 (a second channel beside first-level acks).

Families that pass, out of 7 (images passing in brackets where a family passes only in part):

| variant | + 1 value | + 2 values | + flush-defer record |
| --- | --- | --- | --- |
| S2-explicit | 3 (fails ring 0/10, torus 0/12, 2D 24/28, torus + VC2 16/28: TX-queue status, NoC command words, encoding) | 0 (1D header word; 2D/ring/torus NoC words, chunk count; VC2 idle count; VC1 frame flag) | 6 (torus + VC2 4/28: VC0's idle count, TX-queue status) |
| B2-packed | 5 (2D + VC2 and torus + VC2 8/28: the VC2 channel's counts word; TX-queue status on 4) | 3 (ring: TX-queue status; torus: NoC words; VC2 configs: channel 1's header and slot address) | **7** |
| P2-packed | 3 (ring: TX-queue status; torus: NoC words, encoding; torus + VC2 16/28; VC1 20/22) | 3 | 2 (the stepped cursor's word spilled on 1D, 2D, ring, 2D + VC2, torus + VC2) |

**Stability cost on the device** (the leading variant, S2-explicit, against itself + 2 values, v4rxs2xh2; same job,
two passes in reversed order, job 417):
- ew4352 -0.1, ring1 -0.1, ewscatter +2.5, VC1Inter 1_0 +0.6;
- torusns with VC2 +5.1, torusns_scatter with VC2 +6.6, VC1Mesh2HopMerge +5.4 / +4.4.
- So the guard is conservative on ring: it flags moved values, and ring1 does not move. It is right on the
  two-channel rows: torus + VC2 and VC1 pay +4 to +7.

**Readability.**
- **S2-explicit:** S2 plus one stated placement rule and two `keep_in_memory` lines. The rule names a channel count
  ("the first channel's cursor in registers, later channels' in memory"), not a VC.
- **B2-packed:** the uniform rule plus a small cursor type. Its operations read well at the call sites
  (`cursor.retire(...)`, `cursor.credit(count)`), but every count is now a method and the type needs the layout note
  and the saturating idle count.
- **P2-packed:** the most code. A work mask, a copy, a compare, and a conditional write-back.

**For the sender's two-worker routers.**
- Packing hot per-packet counters into one word costs ALU chains here (about 17 cycles/packet on E1); the sender's
  per-packet worker and far-ring counters would see the same.
- Memory-between-visits with a compare-and-write-back cost +17 to +25 here, against v4pl4's 1.6-8 for a plain copy
  in/out.
- The headroom guard carries over with a small change: a role argument and E0 image selection in `--headroom`
  (today it reads E1 receiver loops only), plus a sender overlay in the synthetic generator.

## 15. S2-explicit adopted; the readability polish (v4rxe1) and its identity check (2026-11-02, offline)

The user adopted S2-explicit, on condition of a readability polish that builds byte-identical code; then S4, the codec
change and C1 on top. The user also allowed the polish as a layout-only change, on conditions: every difference from
v4rxs2x at -O3 and -Os must fall into (a) stack/frame offsets and frame-relative address arithmetic, (b) the initial
stores of `states.ch0.cursor` before the loop, or (c) frame size; the headroom results must match v4rxs2x's; and the
frame growth goes into V4_SIZE.md. Anything else: report and stop.

**v4rxe1:**
- `ChannelStates` holds one uniform `ChannelState<CH>` {cursor, slot sources} per channel.
- `first_channel = 0`. `run()` copies the first channel's cursor into its loop local once, through
  `first_channel_cursor_in(states)`; one ERISC passes that cursor in place.
- `step_channel<CH>` owns the placement: the first channel works on the loop's local; a later channel works on its
  state in memory, followed by `keep_in_memory`.
- One rule comment, on `ChannelStates`, pointing here (§14) and to the headroom check. "First channel" is the name
  throughout.
- `pass()` keeps the three explicit steps and the static_assert: the group's packets-received values are scalars, and a
  fold over the channels would gather them into a record across the fences.

**Identity** (full coverage, 2,974 configs per level, every case including the trim captures and bh_trim_gtest; bases
v4rxs2x_O3 and v4rxs2x_Os built today; classifier `loopvars_maps/polish/layoutcmp.py`):
- Builds the same 2,713 configs as v4rxs2x at both levels.
- One-ERISC images are byte-identical: ReceiverState's layout is unchanged.
- **-O3:** 1,024 of 2,713 images differ, all two-ERISC E1. Other functions differ only by code addresses (moved
  calls and targets).
  - `receiver::run`: 880 images fall within (a)+(b)+(c): loop instructions equal with frame offsets normalized; the
    setup adds the initial stores.
  - **74 outside the classes**, all in instrumentation builds:
    - bh_2d_codeprof 28;
    - bh_2d_allinstr 20, bh_1d_allinstr 10, bh_vc1_allinstr 6: register choice and order of a few loop instructions
      differ (e.g. `lw a5 / li a3,4` become `lw a3 / li a5,4`);
    - bh_ring_allinstr 10: the same instruction multiset in a different order.
  - Plus `_start` in 6 bh_2d_codeprof images: the code-profiling buffer offsets shift by 32.
- **-Os: the polish is not layout-only.**
  - 446 images add a `memset` call in the setup: -Os zero-initializes the larger `states` object with a call.
  - 326 images change register allocation or scheduling inside the loop, across production families:
    - bh_trim_gtest 38; bh_2d_vc2 28 and its trims 20 each;
    - bh_torus_vc2 16 and trim 16; bh_1d / trim / telemetry / delay / allinstr 10 each;
    - glx_1d 12; bh_neighbor 4; and others.
  - The rest (182) are within (a)+(b)+(c).
- Cause, as for the frame:
  - the first channel's `ChannelState` now carries a cursor field that `run()` copies once and never reads again;
  - `ChannelStates` stays in memory (its slot sources are indexed at run time; later channels' `keep_in_memory`);
  - so the frame grows by 16-32 B, the setup changes, and -Os's allocator lays out the loop differently.

**Headroom** (`loopvars_maps/polish/headroom_e1.txt`):
- The pass/fail counts match v4rxs2x in all 21 cells (7 families x {+1 value, +2 values, flush-defer record}).
- In two failing cells a different value moves:
  - +1 on ring: v4rxs2x moves `pass:txq_status`, v4rxe1 the first channel's `uncredited` count;
  - +2 on 1D: v4rxs2x moves the header word, v4rxe1 the `uncredited` count.

**Size** (V4_SIZE.md): two-ERISC E1 code +28 B at -O3 and +12 B at -Os. E1 frame +16 B (1D, ring, 2D, Galaxy 2D) to
+32 B (express, VC1). The least free E1 stack is unchanged (3,700 / 3,556, set by one-ERISC configs).

**Stopped here** as the conditions require: -O3 instrumentation builds and -Os production builds change inside the
loop beyond offsets. The chain stages are prepared offline but not built or measured:
- v4rxe2: S4 applied cleanly;
- v4rxe3: the codec decode with the one-rule comment;
- v4rxe4: the C1 trims.
They rebase onto whichever base the user picks.

## 16. Naming and structure polish (v4rxf1) and its identity check (2026-11-02, offline)

The user accepted v4rxe1 and approved a naming and structure polish. Its condition: at -O3 in production families the
hot loop must be unchanged, and any in-loop change is reported with its cause and the work stops. Base for every
compare: v4rxe1. Artefacts: `loopvars_maps/structure/`.

**v4rxf1** (on v4rxe1; the sender is untouched):
- Names: `me` -> `my_chip`; `Coordinates` -> `ChipCoordinates`. Constants: IDLE_PASSES_BEFORE_PARTIAL_CREDIT,
  COMPLETION_PROOF_MAX_CHECKS, FIRST_VC; LOCAL_NOC, LOCAL_WRITE_CMD_BUF, LOCAL_NOC_VC, INLINE_WRITE_ALL_BYTES,
  DOWNSTREAM_EDM_COUNT_BY_VC. Derived traits stay lower_snake.
- The receiver is keyed by VC:
  - `ReceiverChannel<VC>` (was `Channel<CH>`), `step<VC>`, `get<VC>()`;
  - `receiver_channel_of(VC)` is used only for the host's per-channel arrays and the trim capture, as
    `ReceiverChannel<VC>::rx`.
- Containers:
  - `Each<MASK, T>` with `get<I>()` in router_common.hpp, for the sender to reuse;
  - `ReceiverChannelStates = Each<receiver_vcs, ReceiverChannelState>`;
  - `ReceiverChannelState<VC>` = {progress, sources}, with one rule comment pointing to §14 and the headroom check;
  - the SlotSources invariant comment.
- `pass()` steps every VC uniformly through `step_channel<VC>(first_vc_progress, states, ...)`.
- The fold is adopted: the grouped reads go into a local `std::array` by VC, in a braced initializer over all three
  VCs. `read_packets_received<VC>()` is 0 where a VC's channel does not run, so there is no uninitialized entry.
  Adopted because the loop code is unchanged (below).
- One flat `RingProgress` {packets_ready, next_slot_addr, next_index, uncredited, idle_passes_with_partial_group}.
  This is v4rxe1's field order, which keeps the layout.
- `first_vc_progress_in(states)`: one accessor, used by run() for its copy and by single::receiver_pass in place. On a
  router without the first VC's channel it returns a function-local static `Absent`.
- Placement primitives: `begin_read_group()` / `end_read_group()`, functions documented beside `keep_in_memory`. They
  replace the receiver's and the downstream admit group's bare fence pairs.
  - The sender's fences are left for the sender pass: router_sender.hpp lines 240, 243, 279, 282, 286, 301, 308, 312,
    467, 474.
  - Forward bisection stage F1 (every non-receiver file of v4rxf1 plus the receiver renames) is byte-identical to
    v4rxe1, so the primitives cost nothing.

**Causes found on the way** (single-change A/Bs on the quick case set; each excluded change rebuilt and classified):
1. `credit_completions` read `progress.next_index` inside the completion code: the -O3 loop changed in production
   families. **Fixed**: the next index is copied once at the top of `credit_completions` (as the pass's step left it)
   and passed by value.
2. `run()` initialised its first-VC copy through a by-value helper (`first_vc_progress_copy`). This changed the -O3
   loop in the 74 instrumentation images (bh_2d_codeprof 28, bh_2d_allinstr 20, bh_1d_allinstr 10, bh_vc1_allinstr 6,
   bh_ring_allinstr 10) and the -Os loop's register allocation in 46 images.
   - A/B v4rxf1i1: initialising from the reference instead makes all of them identical.
   - **Fixed**: the single accessor above.
3. **The flat RingProgress** changes `single::receiver_pass` in the UDM images at -O3 (bh_2d_udm 8, bh_2d_udm_gtest 8).
   - What changes: one load of the first VC's `next_slot_addr` moves from a join block into one of its predecessors,
     and registers are renamed around it. Static +1 instruction. Per path the loads are 2 -> 2, 2 -> 1 and 3 -> 3: no
     path longer, one path one load shorter.
   - Not the cause (each reverted alone, then all together; the difference stayed):
     - the fold (scalars);
     - the snapshot form (copy at the call site);
     - `Each` (three `Present` members instead);
     - nesting the progress without sub-object references;
     - nesting plus v4rxe1's completion signatures;
     - bare fences;
     - the step_channel form.
   - The cause, by single-change A/B (v4rxf1i3 vs v4rxf1r2): v4rxe1 reaches the position fields through a reference to
     a position sub-object (`Position&` in advance, the acks, retire and receive). With that, the code is identical;
     with every access rooted at the whole progress object, it changes.
   - GCC's optimized GIMPLE of the two is the same statements on the same memory; only the access-path types and
     the SSA names differ (`gimple_one.py`). The move happens in RTL.
   - A flat struct has no position sub-object, so this is inherent to the flattening.

**Identity** (full coverage, 2,974 configs per level, the same 2,713 built as v4rxe1 at both levels; .text compare,
classifier `layoutcmp.py` with RENAMES for the renamed mangled names):

| option | -O3 | -Os |
| --- | --- | --- |
| v4rxf1 (flat RingProgress) | 2,697 identical; 16 UDM images change `single::receiver_pass` (cause 3); their other functions only move | 2,713 identical |
| v4rxf1n (RingProgress = {position, counts}, `RingPosition&` in advance / acks / retire / receive) | 2,713 identical | 2,713 identical |

**Stopped here**, as the condition requires, for the user's decision: flat (accepting the UDM change) or nested
(identical everywhere). **Decision (user, 2026-11-02): flat (v4rxf1).** The 16 UDM images' load move is accepted: no
path gets longer. v4rxf1n is deleted; its diff is `loopvars_maps/structure/ab_v4rxf1n.diff`.

Not yet run on the chosen base: the headroom check, the chain rebase (S4, the codec decode with its one-rule
comment, C1, scripts prepared), the census and the validation.

Tools:
- loopvars.py's INVARIANT now also matches `my_chip`; the self-test passed after the edit.
- `layoutcmp.py` takes `RENAMES="old=new;..."`.
- `synth_overlays.py` accepts the VC-keyed step.

## 17. The chain on v4rxf1 (v4rxf2-v4rxf4), census and validation (2026-11-02; jobs 420-421, 423-424, 427-428)

**Stages** (each a single change on the previous; scripts `loopvars_maps/structure/`; non-fabric files pinned as in
v4rxf1; no swap active during any build):
- v4rxf2 = v4rxf1 + S4: `slots_due_for_credit` returns the count.
- v4rxf3 = v4rxf2 + the codec decode: `Routing2DCodec::decode_action` / `decode_action_y_first` through the volatile
  route buffer, with the one-rule comment. The codec lives in tt_metal/hostdevcommon (committed, unchanged), so
  nothing extra is pinned.
- v4rxf4 = v4rxf3 + C1, the comment trim: 64 lines, all comments. The pass() hunk has no counterpart, since v4rxf1's
  pass comment carries no history.

**Identity** (full coverage at -O3 and -Os; the same 2,713 configs build at every stage):

| check | -O3 | -Os |
| --- | --- | --- |
| v4rxf2 vs v4rxe2 (S4 on v4rxe1, prepared) | 2,697 identical; the 16 accepted UDM images | 2,713 identical |
| v4rxf3 vs v4rxe3 (codec, prepared) | 2,697 identical; the same 16 UDM images | 2,713 identical |
| v4rxf4 vs v4rxf3 (C1) | 2,713 identical | 2,713 identical |
| v4rxf3 vs v4rxf2 (what the codec changes) | 1,357 images, all in 2D-routing families (2D, torus, VC1, Galaxy 2D / express, trim gtest); 1D untouched | 1,359 |
| v4rxf2 vs v4rxf1 (what S4 changes) | 1,482 images, every family | 1,438 |

**Guard** (`guard_builds.py`: loopvars on every image whose .text differs; `guard_f1_f2_O3.txt`, `guard_f2_f3_O3.txt`):
- S4 shrinks the receiver loop in most families:
  - -8 to -32 instructions;
  - production 1D, 2D, trim and Galaxy 2D / 1D drop spills: `credit_frame_due` (-3 stores / -3 loads), and on 1D
    also the header word (-6 / -6).
- S4 grows the loop on:
  - VC2 two-ERISC configs: +5 to +8 instructions, no stack change;
  - 6 VC1 configs: +36 instructions, +1 / +1 stack accesses.
- The codec: E/W-facing 2D routers +1 instruction. In 8 of 28 bh_2d configs (and their delay, trim and gtest
  variants) it brings back the `credit_frame_due` spill that S4 removed (+3 / +3).
- Golden maps of the final variant (census caches, lead vs v4rxf4): `loopvars_maps/v4rxf4_golden/`.
  - Two ERISCs: v4rxf4's receivers spill the first VC's idle count and `credit_frame_due` (the lead: its idle count
    and frame_due) and the header word on 36 configs (the lead: the route word on 72).
  - VC1: the VC1 channel's header word and volatile header (the codec's read) on 20 configs (the lead: 4); VC1 split
    -0.3 on the device.

**Headroom** (7 families x {+1, +2 values, flush-defer record}):
- v4rxf1 matches v4rxe1 in all 21 cells, pass/fail and the values that move (`headroom_f1.txt`).
  - The +1/+2 probe builds are byte-identical to v4rxe1's.
  - The fd probe's only difference is its own record index: the probe indexes by VC now, so VC2's record is host
    channel 1 on VC2-without-VC1 configs.
- v4rxf4 (`headroom_f4.txt`) differs from v4rxf1 both ways:
  - better: +1 on bh_2d 24 -> 28 and bh_torus 0 -> 4; +2 on bh_2d 8 -> 28, bh_2d_vc2 8 -> 28, bh_torus_vc2 0 -> 16;
  - worse: +2 on bh_vc1 14 -> 10; fd on bh_2d_vc2 28 -> 12 and bh_vc1 22 -> 16. The first VC's
    `idle_passes_with_partial_group` moves.
  - Cause, by single-change stage: S4. v4rxf2 (S4 alone) reproduces every changed cell (`headroom_f2.txt`), and the
    codec stage changes none.

**Size** (V4_SIZE.md): v4rxf4 VC1 E0+E1 22,996 B at -O3 (v4rxe1 +12, the lead -344) and 22,156 at -Os. Least free
stack unchanged (1,604 / 3,700 at -O3).

**Census** (census_runs.sh two1 / two2 / one and census_vc1_runs.sh; passes f4a and f4b in reversed router order;
TT_FABRIC_NO_BW_TELEMETRY=1; every test executed with 0 failures). Suite means, lead v4rxv2i3 -> v4rxf4:
- 2D 4352: 142.3 -> 143.3
- 1D: 150.4 -> 150.6
- mix: 126.3 -> 127.2
- bursts: 522.8 -> 522.9
- ring: 179.9 -> 179.9
- Y-torus: 152.7 -> 153.5
- one ERISC: 2D 271.3 -> 272.6; 1D 299.7 -> 299.6; ring 356.8 -> 356.9
- mux: 371.2 -> 371.1
- trimmed 2D: 149.3 -> 150.2
- VC1 split: 159.3 -> 159.0
- inter-mesh: 336.8 -> 336.9

Row A/B (row_ab_runs.sh, both orders, tags rxf4a/b, rxf4va/b):
- ewscatter 139.7 -> 142.2;
- hop3 146.2 -> 146.7;
- ew4352, ns4352, lin1, ring1, torusns: within 0.1;
- VC2 on: ew4352, torusns, torusns_scatter within 0.2 (§8's S3 loss is gone).

**Rows beyond the decided costs.** Attribution job: the lead, v4rxf1, v4rxf2 and v4rxf4 in the same job, two passes
reversed (census_runs.sh two1 sfa/sfb; one-ERISC rows rabsf1a/b). Cycles/packet, lead -> v4rxf1 -> v4rxf2 -> v4rxf4:

| row | lead | v4rxf1 (polish) | v4rxf2 (+S4) | v4rxf4 (+codec) | codec step |
| --- | ---: | ---: | ---: | ---: | ---: |
| E/W scatter | 139.8 | 139.9 | 140.2 | 142.3 | +2.1 |
| E/W atomic | 127.6 | 127.6 | 128.0 | 130.1 | +2.1 |
| E/W fused | 146.9 | 146.9 | 147.3 | 150.1 | +2.8 |
| FWD atomic | 131.6 | 131.6 | 132.3 | 136.5 | +4.2 |
| FWD fused | 151.9 | 151.9 | 152.6 | 156.8 | +4.2 |
| FWD scatter | 144.1 | 144.1 | 144.8 | 148.7 | +3.9 |
| multicast scatter | 168.7 | 168.7 | 169.3 | 173.1 | +3.8 |
| hop3 write | 146.3 | 146.3 | 146.3 | 146.8 | +0.5 |
| N/S scatter / atomic / fused | 144.5 / 131.9 / 152.0 | same | +0.5 / +0.5 / +0.4 | = v4rxf2 | 0 |
| writes (E/W, FWD, N/S, multicast) | | 0 | 0 | 0 to +0.6 | |
| one ERISC: ew4352 | 198.3 | 198.3 | 198.0-198.1 | 200.0 | +1.9 |
| one ERISC: ewscatter | 240.7 | 240.7 | 240.5 | 242.5 | +2.0 |
| one ERISC: mcast | 330.0 | 330.0 | 329.8 | 331.6 | +1.8 |

- The polish (v4rxf1) costs nothing on any row.
- S4 costs +0.3 to +0.7 on the small-packet scatter, atomic and fused rows (FWD +0.7, multicast scatter +0.6; decided
  +0.4 on scatter). Writes do not move, and one ERISC is 0.2-0.3 better.
- The codec stage (a single change) costs more than its decided figures on rows not measured at the decision:
  - forwarded small-packet rows +3.9 to +4.2;
  - multicast scatter +3.8;
  - E/W fused +2.8;
  - one-ERISC E/W rows +1.8 to +2.0, writes included.
  - Its decided rows reproduce: ewscatter +2.1, hop3 +0.5.
- Mechanism, hypothesis (not isolated): on E/W-facing routers the column byte is now a volatile load plus `zext.b` (§8),
  where v4 read it as a plain load the compiler could schedule early. A forwarder needs the route before it admits
  and forwards, so the load's latency lands in its per-packet path. N/S routers already read both bytes volatile, and
  their rows do not move.

**Validation:**
- 13/13 (job 424): mesh 96, val1d 143, ring 19, Y-torus 12, mux 19, VC2 10, NeighborExchange 7, channel-trimming
  gtests 63 (3 skipped); one ERISC: mesh 96, ring 19, VC2 10, NeighborExchange 7, trimming 63.
- UDM 8/8 and 68/68.
- Inline write then flushed atomic, 1D and 2D: main 2/2, 2/2; v4rxf4 2/2, 2/2 (job 423).

**Lead candidate:** v4rxf4. It is within the decided costs on every census suite, except the codec's cost on the rows
above, which exceeds the figures it was adopted on. The user decides whether that cost is accepted as the codec's.
Untested alternative: v4rxf4 without the codec stage (v4rxf2 + C1), built and measured only on request.

**Decision (user, 2026-11-02): v4rxf4 is the receiver lead; the codec's measured cost is accepted.** No plain
(non-volatile) codec variant for E/W routers (discussed: it would likely restore v4rxf2's E/W code, but reverses the
uniform-volatile rule and adds a shared-API overload). N/S routers stay volatile in any case: plain Y/X reads let GCC
load X only after testing Y (RESULTS §98, v3vd vs v3vc; LEDGER E1-19). Follow-up: N/S fix 2 (the next route word /
header loaded ahead into channel state) is to be measured on the codec's rows (forwarded atomic / fused / scatter,
multicast scatter, E/W fused, one-ERISC E/W) as well as the N/S rows.

## 18. Channel-kind naming, and the placement rule owned by ReceiverChannelStates (v4rxg1-v4rxg3; 2026-11-02; job 447)

**Decisions (user, 2026-11-02).**
- v4rxf4 is the receiver lead, codec cost included; no plain E/W codec variant (§17).
- Name by channel kind (§1.0).
- Drop the little-endian and fits-in-a-byte static_asserts; the `uint8_t` source carries the second.
- `loop_local_vc` replaces FIRST_VC, and `ReceiverChannelStates` owns the rule.

**Stages** (each a single change; receiver files only; the sender code untouched; non-fabric files pinned as in v4rxf4;
scripts `loopvars_maps/structure/stage_g{1,2,3}.py`):
- v4rxg1 = v4rxf4 + the renames of §1.0 + both asserts out (`retire(..., uint8_t source)`,
  `SlotSources::set(uint32_t, uint8_t)`, no cast).
- v4rxg2 = v4rxg1 + `struct ReceiverChannelStates : Each<receiver_vcs, ReceiverChannelState>` with `loop_local_vc`
  (still 0), `loop_local_progress()`, `progress_for<VC>(loop_copy)` and `after_step<VC>()`.
  - Gone: FIRST_VC, FirstVcProgress, `first_vc_progress_in` and the branch in `step_channel`.
  - The rule's one block (comment, struct, step_channel) sits directly above pass() and run(), after step.
  - `DownstreamSenderChannelProgress` becomes {free_slots, next_slot_addr}.
- v4rxg3 = v4rxg2 + `loop_local_vc` derived: the lowest set bit of `receiver_vcs`, or 0 where none runs.
  `progress_for` takes `ReceiverChannelProgress&`. In g2 it took the loop copy's type, because with VC fixed at 0 a
  VC0-trimmed router's loop copy is the stand-in below.
- No receiver channel (`receiver_vcs == 0`): `loop_local_progress()` (a member template, so an absent channel's state
  is named only in template-dependent code) returns the states object, empty there, as a stand-in no step uses.
  - run() is not a template, so it is compiled on every router, including E0 and `run_idle` ERISCs; it needs some lvalue
    there.
  - It copies with `auto loop_copy = states.loop_local_progress();`: copy-init from a reference, the form §16 found
    keeps the instrumented loops identical.
  - No static object.

**Identity** (full coverage at -O3 and -Os):

| stage | vs | -O3 | -Os |
| --- | --- | --- | --- |
| v4rxg1 | v4rxf4 | 2,713 of 2,713 identical | 2,713 identical |
| v4rxg2a (experiment: g2 without the field order) | v4rxg1 | 2,713 identical | 2,713 identical |
| v4rxg2 | v4rxg1 | 1,601 differ, all from the field order (below) | 1,601, the same classes |
| v4rxg3 | v4rxg2 | 2,779 of 2,863 identical (all 2,713 original configs); 84 differ, all in the VC0-receiver-trimmed captures | the same 84 |

- **g2 is layout-only** (`fieldswap.py`, OTHER 0 at both levels). g2a shows that the rule and the regroup are
  byte-identical, so every g2 difference is the field order:
  - receiver::run (940 images), single::receiver_pass (649) and sparse_multicast_hop (230): the same instructions in
    the same order, each changed one only moving a memory offset by 4 bytes on the same base register (the two fields
    trade places);
  - init_downstreams (1,710) and init_relay (56): the same stores in address order.
  - Guard: 2,256 role analyses, loop size and stack stores / loads +0 / +0 / +0 on every one (`guard_g1_g2_O3.txt`).
    The stored-value names differ only by renamed locals.
- **g3**: no shape of the existing captures (T1-T9, bh_trim_gtest, every production case) has VC0's receiver trimmed
  while another runs, so four were captured on the mock.
  - Capture: CPU only, `mock_rxtrim.py` (coverage_txo.py's `mock_pinned`, as §6.4 of V4_SENDER_OWNERSHIP.md),
    capture router v4rxf4, override `trimx/vc0_receiver_trimmed.yaml` (VC0 `force_enable_receiver_channels: []` on
    every router).
  - Cases (configs logged / built):

| case | logged | built |
| --- | ---: | ---: |
| bh_vc1_trim_rx0 | 96 | 44 |
| bh_vc1_trim_rx0_1erisc | 46 | 22 |
| bh_2d_vc2_trim_rx0 | 80 | 56 |
| bh_2d_vc2_trim_rx0_1erisc | 40 | 28 |

  - Every receiver config there has channel 0 unserviced and channel 1 (VC1's, or VC2's) serviced.
  - Build checks only: the far senders still send on VC0.
  - The cases now belong to cov_m's case set: a full build is 3,124 configs, 2,863 of which build.
- **g3's 84 changed images** (`guard_g2_g3_O3.txt`, `_Os.txt`): every router whose lowest running receiver VC is VC1
  or VC2.

| shape | images | -O3 loop / sp stores / sp loads | -Os |
| --- | ---: | --- | --- |
| VC2 only, two ERISCs (loop_local_vc 2) | 28 | -36 / -7 / -9; VC2's in-memory state spills gone (`run:states`, `credit_completions<2>:waiting`, `arrived`) | -12 / -6 / -9 |
| VC1 only, two ERISCs (loop_local_vc 1) | 14 | -16 to +3 / -5 to -6 / -10 to -11; VC1's state spills gone; on 8 of them the header word is spilled instead | -9 to -24 / -2 to -3 / 0 to -5; VC1's state spills gone, but on 10 of them forward-path values spill instead (the NoC write value, destination address, size) |
| VC2 only, one ERISC | 28 receiver_pass (+28 sender_pass, moved only) | +1 / 0 / +1 | -1 / 0 / 0 |
| VC1 only, one ERISC | 14 receiver_pass (+14 sender_pass, moved only) | +1 / 0 / +1 | -1 / 0 / 0 |

  - On one ERISC every state stays in memory either way; the change is that the loop-local VC no longer gets
    `keep_in_memory` after its step.
  - The 8 + 8 unchanged receiver images of the VC1 cases are routers without VC1 serviced: `receiver_vcs` is 0 there,
    so nothing is stepped in either stage.

**Headroom** (`headroom_g3.txt`): matches `headroom_f4.txt` in all 21 cells, pass / fail and the values that move. The
only difference is the local's name (`first_vc_progress` -> `loop_copy`). The probe families hold no g3-changed shape.

**Size**: equal to v4rxf4 at every stage (V4_SIZE.md).

**Device**: no production image's loop changes (g1 identical; g2 layout only; g3 only the mock-captured trimmed
shapes), so no census or row A/B.
- Validation on v4rxg3, 13/13 (job 447), which includes the channel-trimming gtests on two and one ERISC, as asked for
  a stage that changes trimmed images: 63/63 each, 3 skipped as before.
- It also runs g2's field order on hardware.

**Lead**: v4rxg3, the same per-packet code as v4rxf4 on every production config.
