# V4 feature spec: what the generic loop does that v3q's specialized programs do not

Semantic spec of the router behaviors that live only in v3q's generic loop (`run_fabric_edm_main_loop` and
`kernel_main`), written so each can be rebuilt as a component of a new kernel. It describes what the code does; it
proposes nothing. Read-only study of the tree at branch `aagarwal/fabric-router-study` (2026-10-26); nothing was run.

Requested items → sections: first-level ack §2; deadlock avoidance §3; express/Z §4; intermesh §5; VC2 §6; UDM §7;
mux §8; single ERISC and Wormhole §9; speedy path §10; per-round and rare work §11; instrumentation §12; data cache and
TXQ spin flags §13; everything else the generic loop does (`runs`, `shape_supported`) §1 and §14; obligations summary
§15; direct register access §16; anomalies and open questions §17.

## 0. Sources, abbreviations, terms

| Tag | File |
| --- | --- |
| V3 | `fabric_exp/variants/v3q/fabric_erisc_router_v3.cpp` |
| SE | `fabric_exp/variants/v3q/tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_sender_erisc.hpp` |
| RE | `fabric_exp/variants/v3q/tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_receiver_erisc.hpp` |
| SP | `fabric_exp/variants/v3q/tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_speedy_path.hpp` |
| PT | `fabric_exp/variants/v3q/fabric_edm_packet_transmission.hpp` (variant override) |
| CH | `fabric_exp/variants/v3q/fabric_erisc_datamover_channels.hpp` (variant override) |
| CT | `fabric_exp/variants/v3q/fabric_erisc_router_ct_args.hpp` (variant override) |
| FC | `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp` |
| FH | `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp` |
| FA | `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_adapter.hpp` (`EdmToEdmSender`) |
| AU | `tt_metal/fabric/hw/inc/edm_fabric/adapters/fabric_adapter_utils.hpp` (`connection::open_start/finish`) |
| EU | `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp` |
| SR | `tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp` |
| TT | `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp` |
| TX | `tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h` |
| TB | `tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp` |
| CP | `tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_code_profiling.hpp` |
| TR | `tt_metal/fabric/hw/inc/edm_fabric/fabric_trimming.hpp` |
| FU | `tt_metal/fabric/hw/inc/tt_fabric_utils.h` |
| HS | `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_eth_handshake.hpp` |
| API2D | `tt_metal/fabric/hw/inc/tt_fabric_api.h` |
| CODEC | `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` (`Routing2DCodec`, `RouterStateManager`) |
| TUN | `tt_metal/hw/inc/internal/ethernet/tunneling.h` |
| ER | `tt_metal/hw/inc/internal/ethernet/erisc.h` |
| ETHR | `tt_metal/hw/inc/internal/ethernet/tt_eth_ss_regs.h`, `tt_eth_api.h` |
| FW | `tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h` |
| BHN | `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h` |
| DF | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` |
| EDB | `tt_metal/fabric/erisc_datamover_builder.cpp` (EDBH: `.hpp`) |
| CMRB | `tt_metal/fabric/compute_mesh_router_builder.cpp` |
| BC | `tt_metal/fabric/builder/fabric_builder_config.hpp` |
| FBC | `tt_metal/fabric/fabric_builder_context.cpp` |
| FCX | `tt_metal/fabric/fabric_context.cpp` |
| INJ | `tt_metal/fabric/builder/injection_policy.cpp` |
| MAIN | main's router at the merge base, `git show 453ee7bdfa7:tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` |

Terms. A **pass** is one visit of every serviced channel (one iteration of the inner `for`, V3:2167). A **round** is
`iterations_between_ctx_switch_and_teardown_checks` passes (32 by default, EDBH:154) plus the once-per-round work
(V3:2136-2480). The **far ring** of a VC is the far router's receiver channel of that VC, tracked locally by
`OutboundReceiverChannelPointers` (FH:198-239). A **slot** is one packet buffer. A **completion** is the far receiver's
credit saying a far-ring slot is free; an **ack** (first-level ack) says a packet has arrived in the far ring.
**Counter credits** = L1 counters carried by Ethernet packets; **stream credits** = Ethernet remote writes to stream
registers (CT:336-399, FC:17-93).

## 1. Which program runs, and every exclusion condition (feature 13 first)

kernel_main (V3:3172-3936) dispatches: `receiver_erisc::run_kernel` when `receiver_erisc::runs` (V3:3173-3179),
`sender_erisc::run_main_loop` when `sender_erisc::runs` (V3:3895-3896), otherwise the generic loop (V3:3897-3920).

### 1.1 `sender_erisc::runs` (SE:127-130) and `shape_supported` (SE:108-124)

| Condition | Excludes |
| --- | --- |
| `MY_ERISC_ID == 0` | ERISC 1 |
| `NUM_ACTIVE_ERISCS > 1` | every one-ERISC build: Wormhole, Blackhole with 2-ERISC disabled, the mux and UDM (tensix extension forces one ERISC, EDB:106-127) |
| `!any_receiver_channel_serviced` | an ERISC 0 that services a receiver channel (one-ERISC builds) |
| `any_sender_channel_serviced` | an ERISC 0 with no sender channel |
| `!super_speedy_mode` | the speedy path (§10) |
| `is_blackhole_arch` | Wormhole |
| `multi_txq_enabled` | single-TXQ builds (`SENDER_TXQ_ID == RECEIVER_TXQ_ID`) |
| `!ENABLE_RISC_CPU_DATA_CACHE` | data-cache builds (§13) |
| `!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA` | spin-wait-send builds (§13) |
| `!vc2_serviced` (`FABRIC_2D_VC2_SERVICED`) | VC2 (§6) |
| `shape_supported`: for every channel, "serviced and visited by the generic loop" equals "visited by this program" | a serviced VC0 channel 4 (express, 5-wide VC0): the program visits VC0 channels `< min(ACTUAL_VC0, 4)` in 2D, `< 2` in 1D (SE:66-68), the generic loop also visits channel 4 (SE:100-102) |
| `shape_supported`: visited channel 0 has the liveness check, every other visited channel skips it (`sender_ch_live_check_skip[ch] == (ch != 0)`) | a forwarding-type channel 0, a worker-type forwarding channel |
| `shape_supported`: `sender_channel_uses_counter_credits(ch)` | stream-register credit transport on any visited VC (single-TXQ; VC1 without multi-TXQ or express, FBC:52-56) |
| `shape_supported`: `0 < SENDER_NUM_BUFFERS_ARRAY[ch] <= 0x7FF` | zero-slot or oversized channels |

SE does implement bubble flow control (SE:132-140, 371-383), first-level ack (SE:483-494, 731-740), BW and perf
telemetry, channel-trimming capture and the idle context switch; it does not implement VC2, the speedy path, the
data cache or spin-wait sends.

### 1.2 `receiver_erisc::runs` (RE:53-73)

| Condition | Excludes |
| --- | --- |
| `ARCH_BLACKHOLE && !FABRIC_2D_VC2_SERVICED` (preprocessor; `run_kernel` is not even defined otherwise, V3:2851-2853) | Wormhole, VC2 |
| `NUM_ACTIVE_ERISCS == 2 && MY_ERISC_ID == 1` | one-ERISC builds, ERISC 0 |
| `!any_sender_channel_serviced`, `is_receiver_channel_serviced[0]` | mixed ERISCs; an ERISC 1 without receiver channel 0 |
| `!super_speedy_mode` | speedy path |
| `multi_txq_enabled` | single TXQ |
| `!enable_deadlock_avoidance` | rings, tori, express protected rings (§3) |
| `!ENABLE_FIRST_LEVEL_ACK_VC0 && !ENABLE_FIRST_LEVEL_ACK_VC1` | first-level ack (§2) |
| `!udm_mode` | UDM relay (§7) |
| `!skip_src_ch_id_update` (= mux mode, CT:197-198) | the mux (§8) |
| `!enable_context_switch` | an ERISC that is the context-switch master (host never sets it on ERISC 1, EDB:154-158) |
| `!enable_trid_flush_check_on_noc_txn` | (constexpr false, CT:768; never excludes) |
| `!FORCE_ALL_PATHS_TO_USE_SAME_NOC` | Blackhole one-ERISC (already excluded) |
| `local_chip_noc_equals_downstream_noc && edm_to_local_chip_noc == noc_index` | builds whose local writes or forwards use another NoC than this ERISC's |
| `is_2d_fabric || ROUTING_FIELDS_TYPE == LowLatencyRoutingFields` | the 1D non-low-latency `RoutingFields` header (unreachable from the host: `ROUTING_MODE` always has `ROUTING_MODE_LOW_LATENCY`, FCX:396) |
| every serviced receiver channel uses counter credits and has forwarding enabled | stream credits; trimming builds with `DISABLE_RX_CHx_FORWARDING` (§14.1) |
| `static_assert(!runs \|\| local_write_cmd_buf_is_exclusive)` (RE:414-423) | (compile error if the local-write command buffer were shared) |

RE does implement intermesh landing, intermesh exit, VC0-to-VC1 crossover, VC1 pass-through, deferred flushed atomics,
BW telemetry, code profiling and trimming capture (RE:981-1227). Per the study's offline build census
(`SHAPE_INVENTORY.md:24-31`): BH ring/torus run SE on ERISC 0 and the generic loop on ERISC 1; express runs the
generic loop on both (ERISC 0 by `shape_supported`; ERISC 1 presumably by deadlock avoidance on the express torus,
not verified); VC2 runs generic on both; mux, UDM, one-ERISC BH and every Wormhole build run the generic loop on one
ERISC.

### 1.3 Baseline the features modify (generic loop, no feature on)

Sender channel visit (`run_sender_channel_step`, V3:1411-1469): invalidate cache (§13); worker channel (channel 0,
liveness-checked) runs `run_worker_sender_step` (V3:1280-1342), router-to-router channels run
`run_forwarding_sender_step` (V3:1344-1409). Each: (1) has-unsent-packet from the channel's free-slots stream register
(grouped read for VC0 channels 0..3, V3:996-1022; own read otherwise); (2) `sender_channel_try_send` (V3:1100-1234):
if far ring has room and a packet is unsent, read payload size + TXQ status, stamp `src_ch_id` into the header in L1,
yield if the TXQ is busy, else Ethernet data command (`eth_send_packet_bytes_unsafe` on `sender_txq_id`), advance far
slot and local slot, decrement far free count, wait TXQ idle, remote write `+1` to the far receiver's packets-received
stream register on the same TXQ, add `+1` to the local free-slots register (the BH overlapped form does this `+1`
before the remote write, V3:1203-1213; the other form after, V3:1221-1230); (3) completions
(`sender_channel_process_completions`, V3:1236-1278): add to far free count, return the same count as credits to the
upstream producer; (4) worker channel only: connection open/teardown poll (V3:1336-1340).

Receiver channel visit (`run_receiver_channel_step_impl`, V3:1471-1861): read packets-received register; if non-zero,
decode, admit (all needed downstreams have a free slot), assign the slot's transaction ID (trid), forward / deliver
locally, `wr_sent_counter++`, packets-received register `-1`; then completions: batched (V3:1779-1836) — one credit
frame for up to 4 slots once the batch's slots show no outgoing NoC writes on their trid.

## 2. First-level ack

**Gate.** Named args `ENABLE_FIRST_LEVEL_ACK_VC0` / `_VC1` (CT:200-201). Host: VC0 = `final_enable_deadlock_avoidance`
(EDB:1242), so it is on exactly when bubble flow control is; VC1 is always 0 (EDB:1343-1348: code size over the
ACTIVE_ETH config window, and it would have to agree across every VC1 link). Templated per call site as
`enable_first_level_ack`: VC0 steps get `_VC0` (V3:2213-2307), VC1 steps get `_VC1` (V3:2312-2381). VC2 and the speedy
path never ack (SP:30-31 static_assert).

**What an ack is vs a completion.**

| | Ack | Completion |
| --- | --- | --- |
| Meaning | packet is present in far receiver slot (L1) | far slot free again: every NoC command that reads it has left the NIU |
| Receiver sends it | when `packets_received > 0`, before processing the packet (V3:1503-1528) | after the slot's trid shows no outgoing writes (`transaction_flushed` = `_sent`, TT:105-113) |
| Credit transport | counter mode: `ack_counters[src]++`, write L1, send whole credit block (FC:46-50); stream mode: remote `+1` on `to_sender_packets_acked_streams[src]` (FC:87-89) | counter: `completion_counters[src]+=n` (FC:31-41); stream: remote `+n` on `..._completed_streams[src]` (FC:76-82) |
| Sender uses it for | upstream credits (worker read counter or upstream router free-slots register) (V3:1266-1277) | far-ring free count only (V3:1257-1263; upstream credit suppressed by `if constexpr (!enable_first_level_ack)`) |

**Receiver, per pass (V3:1500-1529).** `pkts = get_ptr_val(packets_received)`. If `pkts > 0` and (unless
`ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK`) the receiver TXQ is idle: invalidate cache; decrement the
packets-received register by 1 (this is where the register is consumed under FLA, not at retire: V3:1740-1742 is
skipped); read `src_ch_id` from the header of slot `ack_counter.index` (or use the fixed id under mux, §8) and store it
in `src_chan_ids[idx]`; `receiver_send_received_ack`; `ack_counter++`. At most one ack per pass. A packet is
processed only once acked: `unwritten = !wr_sent_counter.is_caught_up_to(ack_counter)` (V3:1529).

**Receiver completions under FLA.** Batching is disabled (`rx_batch_completion && !enable_first_level_ack`,
V3:1779), so the per-slot branch runs (V3:1837-1859): if `completion_counter` is behind `wr_sent_counter`, the next
slot's trid is "sent", and the TXQ is idle (or spin flag), send one completion to `src_chan_ids[idx]`, clear the slot's
trid, `completion_counter++`. At most one completion per pass. In counter mode each ack and each completion is a
separate Ethernet frame on `receiver_txq_id` carrying the whole ack+completion counter block
(`update_sender_side_credits`, FC:59-65; layout CT:633-650), so FLA costs two frames per packet.

**Sender, per visit (V3:1243-1278).** Completions first: add to far free count and to `completions_processed`. Then,
if `has_acks_pending()` (counter mode: `packets_sent != acks_processed`, FC:173-175; stream mode: always true,
FC:209-211), read acks (`*acks_received_counter_ptr - processed`, or the acked stream register), add to processed (stream
mode: decrement the register, FC:195-197), and `send_credits_to_upstream_workers(acks)`. MAIN's comment explains the
order (MAIN:1284-1286): completions are read before acks so the acks seen are never fewer than the completions seen.

**Worker credits under FLA.** Same mechanism as without FLA, fed by ack counts (V3:930-951): the worker interface's
`local_read_counter += n`; if connected, `noc_inline_dw_write<L1, posted=true, flush=false>` of the absolute counter to
the worker's cached semaphore address, sourced from the scratch word `NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR`
(CH:591-600, DF:2376-2412); if not connected, the counter is copied to the local connection info `edm_read_counter`
(CH:508-510) for the next worker to read at open. Forwarding channels instead post `+n` to the upstream router's
free-slots register (`notify_persistent_connection_of_free_space`, CH:602-607, `noc_inline_dw_write<REG, posted>`).
Note: with `flush=false` the scratch word may be rewritten before an earlier write has read it; because the value is
an absolute monotonic counter, the worker can only see a newer value (DF:2372-2373 states the caller obligation).

**State.** Receiver: `ack_counter`, `src_chan_ids[]` per receiver channel (FH:244-272). Sender:
`acks_received_and_processed` (counter mode) per channel (FC:142-182).

**Other effects.** `skip_idle_per_channel` requires `!ENABLE_FIRST_LEVEL_ACK_VC0` (V3:1045), so FLA disables the
idle-channel skip. `can_defer_local_atomic_flush` requires `!enable_first_level_ack` (V3:279-289).

## 3. Deadlock avoidance (bubble flow control) on rings and tori

**Gate.** `ENABLE_DEADLOCK_AVOIDANCE` (CT:174). Host (EDB:1238-1241): `!is_inter_mesh &&
need_deadlock_avoidance_support(...)`, forced off when channel-trimming says VC0 is terminal-or-source only.
`need_deadlock_avoidance_support` (FCX:305-333): Ring always; Torus per axis (not on the non-torused axis of TORUS_X /
TORUS_Y); express meshes: per axis, iff the mesh has a protected ring on that axis; otherwise false. Per-channel
`SENDER_CH_i_IS_INJECTION` (CT:499-514; EDB:1452-1457) from INJ:100-113: only VC0 participates
(`bubble_flow_control_enabled_on_vc`, BC:117; CMRB:388-389). Non-express (INJ:23-46): the worker channel is injection;
a producer channel is injection only on Torus and only when the producer's axis differs from this router's (a turn);
Z-facing routers mark no producer as a turn. Express: classified per ring effect (INJ:62-98). Host validation: the far
VC0 receiver must have at least 2 slots (EDB:1244-1267).

**Sender ("two free far slots" rule).** `use_bubble_flow_control = is_injection[ch] && enable_deadlock_avoidance`
(V3:1122-1123). Such a channel sends only when `num_free_slots >= 2`
(`BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS`, CT:514; V3:1129-1131); other channels need 1. Enforced
per send attempt on the shared far-ring free count of the VC. `static_assert(!bubble || first_level_ack)` (V3:1124-1126).

**Receiver differences under deadlock avoidance** (these are why RE excludes it; RE:49-52 lists the exclusions without
a reason):
1. First-level ack on VC0 (§2) and therefore per-slot completions (no batching).
2. `NUM_TRANSACTION_IDS = 8` (CT:709-716) instead of 16 (lone receiver channel) or 4; trid ranges per receiver channel
   start at `RX_CH_TRID_STARTS[ch] = ch * NUM_TRANSACTION_IDS` (`tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp:26-34`).
3. Strict-flush atomics wait for every trid of the receiver channel's range to be acked
   (`flush_write_to_noc_pipeline`, PT:120-133) instead of one global `ncrisc_noc_nonposted_writes_flushed` (PT:105-118).
4. No deferred flushed atomics (`can_defer_local_atomic_flush` needs `!enable_deadlock_avoidance`, V3:282).
5. Forward credit writes: `forward_payload_to_downstream_edm<enable_deadlock_avoidance=true, ...>` makes the stateful
   credit write re-program `addr_lo` on every forward (`noc_inline_dw_write_with_state<update_addr_lo=true>`, FA:326-331)
   instead of firing the pre-set state. The code gives no reason.
6. `send_credits_to_upstream_workers<enable_deadlock_avoidance>` passes the flag into
   `notify_persistent_connection_of_free_space`, whose implementation ignores it (CH:602-607).

**Dateline.** None. No dateline code exists in v3q, MAIN, the fabric hw headers or the builders (grep).

**Mux.** BC:111-114 says the 2-slot rule must also match `fabric_router_mux_extension.cpp` (the mux enforces it
for injection it performs; not examined further).

## 4. Express / Z direction

**Gate.** No kernel flag; it is shape: `ACTUAL_VC0_SENDER_CHANNELS == 5` (worker + 3 mesh + Z, BC:55-66;
`SHAPE_INVENTORY.md:13`), `VC0_DOWNSTREAM_EDM_SIZE` up to 4 (BC:89-93), `ACTUAL_VC1_SENDER_CHANNELS` up to 4,
`MY_DIRECTION == 4` for Z-facing routers, VC1 counter credits (FBC:53). Host: `express_routing_enabled(mesh)` disables
speedy, trim bypass and channel remapping (EDB:1215-1228, 1273-1278); express + mux is fatal (CMRB:299-306); express + UDM
is a static_assert in the header (`tt_metal/fabric/fabric_edm_packet_header.hpp:1345-1358`). `FABRIC_EXPRESS_ENABLED` only sizes worker
headers; "router behavior does not otherwise specialize on FABRIC_EXPRESS_ENABLED" (CMRB:920-922).

**Sender channel 4.** Visited every pass after channels 0..3 and before VC1 (V3:2298-2308), only `if constexpr
(is_2d_fabric && ACTUAL_VC0_SENDER_CHANNELS > 4)`. It is outside the grouped read (group is `min(ACTUAL_VC0, 4)`,
V3:996), so it reads its own free-slots register with `get_ptr_val` (masked to `REMOTE_DEST_WORDS_FREE_WIDTH` on BH,
V3:1376-1381; SR:43-50). It is never idle-skipped (`skip_idle_channel` requires `ch < 4`, V3:1027-1029); idle skip of
channels 1..3 is unaffected. It is a forwarding (router-to-router) channel in this shape; it is the Z-ingress channel
(see stream mapping below). FLA stream mode has a fifth acked stream `TO_SENDER_4_PKTS_ACKED_ID` (CT:40-42, 731-746).

**Decode.** `Routing2DCodec::decode_action<my_direction>` (CODEC:278-297): E/W-facing routers read only the X byte
`route_buffer[MESH_Y_SIZE + x]`; N/S/Z-facing routers read the Y byte `route_buffer[y]` and fall back to the X byte when
it is 0 (Y first). Bits: E=0x01 W=0x02 N=0x04 S=0x08 Z=0x10 LOCAL=0x20 (CODEC:172-178). Destination-major maps are not
mutated in transit (PT:680-685).

**Downstream slots.** Compact index = direction with self removed, base order {E, W, N, S, Z} (V3:317-331;
`fwd_dirs`, CODEC:302-335). Z-facing row: {E, W, N, S} (V3:173-176). `live_eth_mask_2d` (V3:539-562): slot i is live iff
its compact index < `DOWNSTREAM_EDM_SIZE`; E/W intramesh routers (`!is_intermesh_router`) additionally drop the N and S
slots (`live & ~0b0110`), so they can forward only to the opposite X direction and Z. Bits outside LIVE are ignored.

**Admission** (`admit_2d_dispatch`, V3:588-680). Non-UDM, non-crossover fast path: LIVE == 1 slot → test that slot's
local free-slots copy register only if its action bit is set; otherwise if no Ethernet bit is set → admit; else read
every LIVE slot's local copy register (up to 4, `vc0_local_free_slots_stream_id<compact>`, V3:217-233) and require
non-zero for each slot whose action bit is set. Otherwise (UDM or crossover): per used LIVE slot,
`downstream_edm_interfaces[i].edm_has_space_for_packet()` (FA:182-186). Local delivery needs no admission except UDM.

**Forwarding** (`forward_2d_dispatch`, V3:682-745). One unmodified full-packet copy per used LIVE slot, in compact slot
order 0..3, all with the same trid; local delivery afterwards (V3:1717-1729). Stateful NoC API only when
`use_2d_single_live_stateful_noc` (V3:575-586: E/W router, LIVE == 0b0001, no VC1 build, no UDM, no intermesh flags, no
forced NoC, no DPRINT/watcher).

**Credit stream mapping for the 5-wide shape** (V3:179-206). Downstream sender channel at the destination router =
`my_direction` if `my_direction > ds_direction` else `my_direction + 1`. Hence a Z-facing router (4) feeds channel 4
of every cardinal router; E/W/N/S feed channel 1/2/3/4 of the Z-facing router. VC1 uses the same rule offset by
`VC1_FABRIC_POSITION_START - 1` (V3:195-206). The free-slots table is fabric-flat (CT:101-139).

**Z-facing routers (`my_direction == Z`).** Only the decode (Y first), the compact map row and the stream mapping above
differ; there is no other Z-specific code in the generic loop. VC2 has no Z receiver (BC:61-63).

## 5. Intermesh: landing rewrite, exit, VC0-to-VC1 crossover, VC1 pass-through

**Gates.**

| Item | Kernel gate | Host |
| --- | --- | --- |
| Landing rewrite | `IS_RECEIVER_CHANNEL_i_INTERMESH_INGRESS` (CT:451-454) | `is_inter_mesh && i < 2` (EDB:1433-1435) |
| Intermesh exit | `IS_INTRAMESH_ROUTER_ON_EDGE` (CT:434) | intramesh-facing channel on a chip that has intermesh channels (EDB:199-211) |
| E/W keep N/S arms; no stateful forward | `IS_INTERMESH_ROUTER`, `IS_INTERMESH_ROUTER_ON_EDGE` (CT:176, 433) | EDB:1337, 1417 |
| VC1 channels exist | define `FABRIC_2D_VC1_ACTIVE` | `sender_counts[1] > 0` (CMRB:887-891) |
| VC1 serviced | define `FABRIC_2D_VC1_SERVICED` | intramesh + `requires_vc1_full_mesh`, or intermesh + pass-through (CMRB:893-906) |
| VC0→VC1 crossover | define `FABRIC_2D_VC0_CROSSOVER_TO_VC1` | intermesh router + `requires_vc1_full_mesh` (CMRB:914-919) |
| VC1 first-level ack | `ENABLE_FIRST_LEVEL_ACK_VC1` | always 0 (EDB:1348) |
| Deadlock avoidance | — | never on intermesh routers (EDB:1239, CMRB:274-276) |
| Opt level | — | `-Os` when VC1 receiver exists (EDB:1943-1953) |

**Receiver per packet (generic, V3:1569-1621).**
1. Landing (ingress channel only, before decode): `fabric_set_2d_intermesh_landing_route(header, routing_table,
   MESH_Y_SIZE, MESH_X_SIZE)` (API2D:197-260) rewrites `route_buffer` in place: final mesh != this mesh → maps toward
   `exit_node_table[final_mesh]`; final mesh == this mesh and unicast → maps to `dst_start_chip_id`; multicast →
   rebuild the multicast maps rooted at this chip. Then `decode_action_y_first` regardless of facing (V3:1581-1582).
2. Exit test (edge routers only): `action == ACTION_LOCAL_DELIVER && dst_start_mesh_id != my_mesh_id`
   (CODEC:338-340). If true: `boundary_dir = inter_mesh_direction_table.get_original_direction(dst_mesh)`; it must be
   `< COUNT`, not `my_direction`, and map to a compact index inside the downstream array; otherwise `ASSERT(false)` and
   the packet is never admitted (the channel then retries it every pass; V3:1603-1610). If addressable: admission =
   that downstream has space; forward = one stateless `forward_payload_to_downstream_edm<DA, false>` of the unmodified
   packet, no local delivery (V3:1706-1714).
3. Otherwise ordinary admission and dispatch (§4).
4. A pending deferred atomic may not be bypassed by an intermesh egress forward (stateless credit write uses
   `write_at_cmd_buf`, V3:1645-1660).

**Crossover.** Receiver channel 0 is called with the VC1 downstream array (V3:2223-2243) and skips the VC0 fast
admission path (`#if !defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)`, V3:598). The VC0 adapters built in kernel_main also
point their remote credit at the downstream's VC1 sender channel (V3:3514-3518) and are opened when receiver 0 is
serviced (V3:3784-3785), but receiver 0 forwards only through the VC1 adapters. Trimming capture uses
`PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS` (V3:753-754). When VC1 is also serviced, receiver channels 0 and 1 forward
through the same `downstream_edm_noc_interfaces_vc1` objects (shared slot cursors and local credit copies; V3:2224-2231,
2362-2369), so both receiver channels must run on the same ERISC. Which downstream sender channel the opened VC0
adapters register with on a crossover router was not traced (uncertain).

**VC1 pass-through.** Receiver channel 1 (V3:2362-2381) is the generic receiver step with `ENABLE_FIRST_LEVEL_ACK_VC1
= 0`, the VC1 downstream array, its own trid range, `rx_idle_steps_ch1`. VC1 forwards are always stateless
(`use_2d_single_live_stateful_noc` is false in VC1 builds). Sender side: VC1 channels `ACTUAL_VC0 + 0..2` (+3 when
`ACTUAL_VC1 > 3`) each pass after VC0 (V3:2312-2361), no grouped read. In VC1 builds the sender/receiver steps are not
`FORCE_INLINE` (code size, V3:1289-1291, 1353-1355, 1420-1422).

## 6. VC2

**Gate.** Define `FABRIC_2D_VC2_SERVICED` when `sender_counts[2] > 0` (CMRB:908-912). Host: rtoption
`enable_fabric_vc2`, Blackhole, not UDM, not mux, 2D (FBC:381-390). One worker-type sender channel at
`VC2_LOCAL_CHANNEL_START = ACTUAL_VC0 + ACTUAL_VC1` (CT:98-99), receiver channel `VC2_RECEIVER_CHANNEL` = 2 if VC1 has
senders else 1 (CT:185). Counter credits always (FBC:54-56). Free-slots stream fixed at 30 (shared with the tensix relay
id; VC2 and UDM/mux are exclusive; `tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp:28-29`). Receiver never forwards
(`DISABLE_RX_CH2_FORWARDING = 1`, EDB:1540). Workers connect through the private `append_fabric_vc2_connection_rt_args`
(`tt_metal/fabric/fabric_vc2_connection.hpp:14-23`). Initial worker read counter 0 (V3:2587-2591).

**Sender, per round** (V3:2150-2163): connection check for the VC2 channel (`check_worker_connections`, FU:46-70) once,
before the passes. **Per pass** (V3:2385-2402): `run_sender_channel_step_speedy<VC2_LOCAL_CHANNEL_START, ...,
amortization 1, MANAGE_CONNECTION_LIVENESS=false>` (SP:93-224, §10.2): no connection poll inside the pass, no bubble
flow control, no FLA; completions are read whenever a packet is outstanding and credits returned per completion
(amortization 1).

**Receiver, per pass** (V3:2403-2420): `run_receiver_channel_step_speedy<VC2_RECEIVER_CHANNEL,
producer=VC2_LOCAL_CHANNEL_START, ..., amortization 1>` (SP:229-323, §10.3): local delivery only, completion credited
to sender channel index `VC2_LOCAL_CHANNEL_START` regardless of `src_ch_id`, state `SpeedyReceiverState<2>`.

**State.** `persistent_speedy_sender_state_vc2`, `persistent_speedy_receiver_state_vc2` (statics, V3:2108-2111),
copied into locals at loop entry and back at loop exit (V3:2119-2134, 2488-2494) so they survive PAUSE/RUN.
Third trid tracker (V3:3671-3680), included in teardown (V3:2761-2765).

**Observed:** the speedy receiver alternates trids with `current_write_trid = 1 - current_write_trid` (SP:302),
starting from `RX_CH_TRID_STARTS[2]` (SP:48), which is `2 * NUM_TRANSACTION_IDS` (8 or 16). The flip therefore
alternates between 8 and 249 (uint8), outside the 4-bit trid range, and the flushed check then reads
`NIU_MST_WRITE_REQS_OUTGOING_ID(249)`, a different status register (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:74, 107`). Main's speedy header is
identical (tree `fabric_erisc_router_speedy_path.hpp:332`). Correct only for VC0 (start 0). Not verified on hardware.

## 7. UDM mode: local tensix relay

**Gate.** Named `UDM_MODE`, `LOCAL_RELAY_NUM_BUFFERS` (CT:209-210), `TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID` (CT:80),
`NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS = 1` (CT:144); define `UDM_MODE` selects `UDMHybridMeshPacketHeader` (FCX:346-350;
fabric_edm_packet_header.hpp:1331-1362). Host: `FabricTensixConfig::UDM`, set in `connect_to_local_tensix_builder`
(CMRB:681-700): `udm_mode = true`, relay slot count, one local tensix connection. 2D only (CMRB:683, header
static_assert for 1D). Tensix extension forces one ERISC (EDB:118-126).

**Runtime args** (V3:3353-3375): `has_local_tensix_relay_connection`, then (UDM and set) relay buffer base, NoC x, NoC
y, registration id, location-info address, relay free-slots stream id, connection buffer-index id.

**Init.** `local_relay_interface = RouterToRouterSender<LOCAL_RELAY_NUM_BUFFERS>(persistent=true, x, y, base, n,
registration, location info, channel_buffer_size, buffer index, 0, teardown=nullptr, 0, remote stream = relay's
free-slots stream (runtime), local stream = TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID, data cmd buf = RX0's, sync cmd buf
= RX0's)` (V3:3597-3618); constructor sets the local copy register to `LOCAL_RELAY_NUM_BUFFERS` (FA:112). One ERISC:
`setup_edm_noc_cmd_buf` for it unless forced NoC (V3:3622-3626). Wait for `edm_local_tensix_sync` to reach 1 (relay
ready, §8) before opening (V3:3758-3761). Opened after the downstream routers, with the same
`open<false, posted, worker_handshake_noc>` (V3:3790-3796). **Observed:** the teardown pointer is `nullptr`, so
`open_start` sends address 0 as the teardown semaphore address and `open_finish` writes 0 to L1 address 0
(AU:16-63; FA:270-289). Effect not established.

**Receiver per packet.** Admission: the fast path is skipped (`if constexpr (!udm_mode)`, V3:599); generic path
requires `!local_deliver || relay.edm_has_space_for_packet()` plus every used LIVE downstream (V3:651-679). Local
delivery goes through `forward_to_local_destination` (V3:520-537) → `execute_chip_unicast_to_relay` (PT:596-623): the
whole packet (header + payload) is written to the relay's next slot with the packet's trid on
`edm_to_downstream_noc` via the forwarding command buffers (stateless, posted), then `-1` on the relay's free-slots
register and `-1` on the local copy — the same NoC commands as a router-to-router forward. No direct local NoC writes
or atomics happen on this router in UDM; the deferred-atomic path is off (`!udm_mode`, V3:282). The relay's writes share
the slot's trid, so completions wait for the relay copy to leave the NIU (TT:105-113 checks both NoCs if they differ).

**Relay credits back.** The relay returns slots by incrementing `TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID` (the local copy
register) over the NoC — implied by the adapter protocol (FA:27-44); relay kernel not examined.

## 8. Mux / tensix extension

**Gate.** `FABRIC_TENSIX_EXTENSION_MUX_MODE` (CT:197) → `skip_src_ch_id_update` (CT:198);
`NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS` (CT:144); `EDM_LOCAL_TENSIX_SYNC_PTR_ADDR` (CT:280); `REMOTE_WORKER_SENDER_CHANNEL`
(CT:206). Host: `has_tensix_extension = (tensix config == MUX) && !dispatch link` (CMRB:286-297, 545); MUX mode services
only the worker sender channel and every receiver channel (EDB:765-786); every downstream connection made to a
`FabricTensixDatamoverBuilder` increments `num_downstream_tensix_connections` (EDB:1926-1929); tensix extension
disables 2-ERISC (EDB:118-126); BH runs the one-ERISC kernel on physical ERISC 1 when 2-ERISC system mode is on
(CMRB:952-958). The kernel ignores the host's `SKIP_SRC_CH_ID_UPDATE` (EDB:1357) and uses the mux flag instead; the host
value is false when the link has no remote routing direction (EDB:1109-1116) while the kernel's is not (observed
discrepancy).

**Router changes.**
1. Sender does not stamp `src_ch_id` (V3:1154-1156, 1171-1173).
2. Receiver never reads `src_ch_id`; at main-loop entry it stores `REMOTE_WORKER_SENDER_CHANNEL` in receiver channel 0's
   `src_chan_ids[0]` (V3:2090-2092), and every ack/completion goes to that sender index (V3:1515-1516, 1816-1818,
   1849-1850). Batched completions send one credit of `batch` (V3:1816-1818). Only channel 0 gets the fixed id; another
   receiver channel would read an unset entry (whether mux ever has VC1 was not traced).
3. Downstream connections of the receiver channels may point at tensix mux cores instead of routers; the kernel builds,
   opens and forwards to them with the same `RouterToRouterSender` adapters and runtime-arg layout (V3:3465-3549). No
   code path distinguishes them.
4. Before opening downstream connections, wait until the L1 word at `EDM_LOCAL_TENSIX_SYNC_PTR_ADDR` equals
   `NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS` (`wait_for_notification`, V3:3756-3761 and V3:3118-3122 in `receiver_erisc::run_kernel`; FU:73-89). The mux / relay
   kernels `noc_semaphore_inc` that word on each upstream router when ready
   (`tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_mux_extension.cpp:323-328`, `.../fabric_router_relay_extension.cpp:577-579`). The router never clears
   it (who does was not traced). The wait checks immediate termination (not on WH) and calls `run_routing()` each
   iteration on WH and on physical ERISC 0 of BH (FU:76-87).
5. The worker channel's producer is the mux core; to the router it is an ordinary worker connection (§11.4).

## 9. Single-ERISC operation and Wormhole

**Gate.** `NUM_ACTIVE_ERISCS == 1` (CT:428). Host: Wormhole always; Blackhole when 2-ERISC is disabled
(`TT_METAL_DISABLE_FABRIC_TWO_ERISC`, `TT_METAL_DISABLE_MULTI_AERISC`), with the tensix extension, or single-processor
dispatch (EDB:106-127, 168-175). In one-ERISC mode both sender and receiver channels are serviced (EDB:180-195).

**Interleaving.** One generic loop, fixed order per pass (non-speedy, V3:2207-2422): invalidate cache; grouped
free-slots read of VC0 channels 0..3; sender 0; receiver 0; sender 1; (2D) senders 2, 3; sender 4; VC1 senders then
receiver 1; VC2 sender then receiver. Speedy: sender 0 speedy, receiver 0 speedy, then the VC1/VC2 blocks
(V3:2169-2206, 2310-2422). In two-ERISC generic builds each ERISC compiles only its serviced steps
(`is_*_channel_serviced` guards, V3:1434, 1888).

**TX queues.** Single TXQ: `RECEIVER_TXQ_ID == SENDER_TXQ_ID == 0` (receiver TXQ becomes 1 only when 2-ERISC is enabled,
EDB:299-301). Data sends, packets-received remote register writes, acks and completions share TXQ 0, each guarded by a
TXQ-idle check (exceptions: §17 items 2 and 3). `initialize_state_for_txq1_active_mode` is skipped (V3:3202). Credit transport is stream registers for
VC0 (and VC1 unless express) because counters are tied to multi-TXQ (FBC:52-53). ETHR:55 notes the remote-register-write
command is TXQ0-only, which is consistent with multi-TXQ requiring counter credits (TXQ1 carries only data frames).

**NoC and command-buffer setup** (`setup_edm_noc_cmd_buf`, FA:169-180: pre-program the data cmd buf's destination and
the sync cmd buf's credit target + value `-1`).

| Build | When | What |
| --- | --- | --- |
| One ERISC, not forced NoC, not single-live stateful | at adapter construction, before handshake and open (V3:3527-3533) | every connected VC0 downstream |
| One ERISC, single-live stateful | same point (V3:3540-3547) | VC0 downstream 0 only |
| One ERISC, UDM | same point (V3:3622-3626) | relay |
| Two ERISCs, receiver serviced, not forced NoC | after the VCS_OPENED sync (V3:3815-3858) | every connected VC0 downstream, then downstream 0 again if single-live, then every connected VC1 downstream |
| VC1 downstreams, one ERISC | never | (VC1 forwards are stateless) |

Implicit assumption (not verified): on Wormhole the pre-programmed state survives the `run_routing()` calls of the
handshake and connection waits and the idle context switches (base firmware runs in between, ER:20-46).

**Blackhole one-ERISC.** `FORCE_ALL_PATHS_TO_USE_SAME_NOC = (BH && one ERISC)` (EDBH:255; EDB:1416): all router NoC
traffic on NoC 1 (base firmware may use NoC 0, EDB:307-314); forward data cmd buf becomes `WR_CMD_BUF` (0), the same as
the local-write cmd buf (EDB:518-523); stateful forwarding is disabled (V3:419-424, 581) and no command-buffer
pre-programming happens. Kernel runs on physical ERISC 1 when 2-ERISC system mode is on (CMRB:952-958), where
`run_routing*` and the handshake's context switches compile to nothing (ER:26, 54; HS:41, 72).

**Wormhole-only paths.**

| Place | Wormhole behavior |
| --- | --- |
| `eth_txq_is_busy` (TUN:73-83) | `ETH_TXQ_CMD != 0`; BH reads CMD (BH-55 workaround) then the STATUS ongoing bit |
| `get_ptr_val` (SR:43-59) | full register value; BH masks to `REMOTE_DEST_WORDS_FREE_WIDTH` |
| `sender_channel_try_send` (V3:1145, 1163-1178) | not the BH grouped path: payload size, busy check (yield), `did_something`, stamp, then send; no overlapped completion read (V3:1094-1098) |
| static-connection wait (V3:2540-2553) | no termination check and no periodic `run_routing()` |
| handshake, `wait_for_notification` (HS:33-44; FU:76-87) | no termination check; `run_routing()` (full NoC sync + RTOS switch) |
| `worker_handshake_noc` (CT:777-781) | `SENDER_CH_0_ACK_NOC_ID` (NoC 0); BH: `noc_index` |
| connection open (V3:3682-3686) | posted writes; BH non-posted |
| idle context switch (ER:50-57) | `rtos_context_switch_ptr()` (real switch to base firmware); BH physical ERISC 0: link-status check only |
| heartbeat address (V3:249-253) | 0x1F80; BH 0x7CC70 (`MEM_SYSENG_ETH_HEARTBEAT`, FW:21) |
| teardown `noc_clear_packet_tags` (V3:2787-2791) | skipped |
| `can_defer_local_atomic_flush`, BH global flush (V3:281; PT:105-119) | never |
| `enable_trid_flush_check_on_noc_txn` | constexpr false everywhere (CT:768; MAIN's tree CT the same) — dead |
| IRAM | no IRAM-specific code in v3q or MAIN; WH router `.text` lives in `ERISC_APP_IRAM_CODE` and VC1 builds overflow it at -O3/-O2 (`V3_REVIEW.md:477`), hence -Os for VC1 (EDB:1949-1952) |

## 10. The speedy path

**Selection.** `ENABLE_SPEEDY_VC0` (CT:175) → `super_speedy_mode` (CT:874). Host (EDB:1273-1278;
`tt_metal/fabric/channel_trimming_import.cpp:171-175`): not express, and (`ACTUAL_VC0_SENDER_CHANNELS == 1 && !deadlock_avoidance`, e.g.
NeighborExchange has one VC0 sender, BC:52) or a trimming capture says VC0 is worker-only non-forwarding, or a
terminal-only router whose exact peer is worker-only (CMRB:446-528). With speedy: `DISABLE_RX_CH0_FORWARDING = 1` (unless
DRISC offload, EDB:1523-1525); amortization `SENDER_CREDIT_AMORTIZATION_FREQUENCY = max(min(2, slots/2), 1)`,
`RECEIVER_... = max(min(4, slots/2), 1)`, both capped by packet size from the capture (EDB:1542-1579). Static asserts
(SP:27-37; CT:875-880): no deadlock avoidance, no VC0 FLA, no serviced VC0 sender beyond channel 0, RX0 forwarding
disabled when serviced, non-zero amortization, `NUM_TRANSACTION_IDS >= 2`, no data cache (SP:261).

**10.1 Loop.** Per pass: invalidate cache; `run_sender_channel_step_speedy<0, ..., SENDER_AMORT, liveness in helper>`
if sender 0 serviced; `run_receiver_channel_step_speedy<0, producer 0, ..., RECEIVER_AMORT>` if receiver 0 serviced
(V3:2169-2206); then VC1/VC2 as usual. State in statics copied in/out around each `execute_main_loop` (V3:2106-2134,
2483-2494).

**10.2 Sender step (SP:93-224).** `can_send = far has space && free_slots != N && (TXQ idle unless spin)`. Send:
payload size; `eth_send_packet_bytes_unsafe`; `update_write_counter_for_send<false>` (worker write counter);
advance far pointer and the local slot (index-based); far free count `-1`; perf record; wait TXQ idle; remote `+1`
packets-received; `sender_amort_counter++`; BW counter; local free-slots `+1`. **Does not set `did_something`** (the
idle context-switch counter is not reset by speedy sends). Completions are read only when `sender_amort_counter >=
AMORT`; credits go to the worker only when `completion_count >= AMORT`, then `sender_amort_counter -= completion_count`.
So the credits of the last `< AMORT` packets are held until more packets are sent (observed). Connection poll every
call (`speedy_connection_liveness_check_period = 1`, SP:198-222) when `MANAGE_CONNECTION_LIVENESS_IN_SPEEDY_HELPER`.

**10.3 Receiver step (SP:229-323).** If packets-received != 0: local delivery of the slot with
`execute_chip_unicast_to_local_chip_impl(…, current_write_trid, …)` (no forwarding, no UDM relay, no src id), BW counter,
trimming, `wr_sent++`, register `-1`, `unacked_sends++`, `did_something = true`. Ping-pong trids: once `unacked_sends
>= AMORT` and no flush is pending, the current trid becomes the pending one with the batch count and the current trid
flips. Every call: if pending and that trid has no outgoing writes (`_sent`), `completion_counter += batch`,
`receiver_send_completion_ack<false>(…, producer_sender_channel_index, batch)` — **with no TXQ-idle check** (SP:312-315;
same in MAIN) — and clear pending. Completions of the last `< AMORT` packets are held until more arrive (observed).

## 11. Per-round and rare work

**11.1 Round skeleton (generic, V3:2136-2480).** Top: `continue_running_main_run_loop` (V3:1972-1986: immediate
termination; plus `command != RUN` on the context-switch master and on one-ERISC builds); `did_something = false`;
`tx_progress = rx_progress = 0`; perf window open; BW timestamp; VC2 connection check. 32 passes. End: drain a pending
deferred atomic of receiver 0 (V3:2424-2429); BW cycles; `update_telemetry`; heartbeat; context switch; perf window
close and write. ERISC 0 wraps rounds in the command loop (V3:2497-2515); other ERISCs loop until termination
(V3:2516-2520).

**11.2 Idle context switch.** Gates: `ENABLE_CONTEXT_SWITCH` (CT:330), `IDLE_CONTEXT_SWITCHING` (CT:423, host
`WAIT_FOR_IDLE` type, EDB:1411-1412), `SWITCH_INTERVAL` (CT:167-172, 0 with DPRINT). Host: ERISC 0 of BH 2-ERISC and
Wormhole get `enable_context_switch = true`; BH ERISC 1 false; BH one-ERISC = 2-ERISC system mode flag (EDB:129-166).
Interval/type: default 10000 / WAIT_FOR_IDLE (EDBH:357-358); BH 2-ERISC 32 / WAIT_FOR_IDLE (CMRB:552-555); dispatch links
16 / INTERVAL (CMRB:867-872). Behavior (V3:2450-2469): master, idle mode: a round with `did_something` resets
`did_nothing_count`; otherwise `if (did_nothing_count++ > SWITCH_INTERVAL)` reset and switch — i.e. after
`SWITCH_INTERVAL + 2` consecutive idle rounds. Interval mode: same counter without the reset, every `SWITCH_INTERVAL + 2`
rounds. `did_something` is set by a send attempt that reached the TXQ check (BH: also when it yielded on a busy TXQ,
V3:1153; WH: only when sending, V3:1170) and by every admitted receiver packet (V3:1634) and speedy receiver packet; not
by completions, acks, credits, connection events or speedy sends. SE uses the equivalent "attempted or sent count
changed" (SE:786-789). Non-master: the non-master handshake every round (V3:2468).

**11.3 Coordinated context switch / retrain sync** (V3:80-96, 785-864). Master = `enable_context_switch`. Shared stream
scratch register `ETH_RETRAIN_LINK_SYNC_STREAM_ID` (CT:82; SR:24-38). One ERISC: the switch is a plain
`run_routing_without_noc_sync()` (V3:857-858). Two ERISCs, master: write RETRAIN_INTENT; spin for INTENT_ACK; run;
write RETRAIN_COMPLETE; spin for COMPLETE_ACK; write NORMAL_EXECUTION (V3:789-828). Non-master, at the end of every
round: if the register reads RETRAIN_INTENT, write INTENT_ACK, spin for RETRAIN_COMPLETE, write COMPLETE_ACK, spin for
NORMAL_EXECUTION (V3:830-853). Every spin exits on immediate termination. Only the master initializes the register
(NORMAL_EXECUTION, V3:3226-3230); a one-ERISC build with `ENABLE_CONTEXT_SWITCH = 0` is a non-master and polls a register
nothing initialized (observed). What "run" does: `run_routing_without_noc_sync` → WH: `rtos_context_switch_ptr()`; BH
physical ERISC 0: `update_boot_results_eth_link_status_check()` (calls the firmware link-status check, debounced by a
timestamp, FW:470-485); BH physical ERISC 1: nothing (ER:50-57). Full `run_routing()` (static-connection,
handshake and notification waits) = `ncrisc_noc_full_sync<1>` + `aerisc_context_switch` (heartbeat[1], `service_eth_msg`,
link check) + counters re-init on BH ERISC 0 (ER:20-46; FW:488-504). **Only physical ERISC 0 can talk to
base firmware**, so the master role must sit there.

**11.4 Router commands** (`RouterStateManager` in the routing table L1, CODEC:891-918: `state` written by device,
`command` by host). ERISC 0 command loop (V3:2497-2515; SE:801-809): RUN → `state = RUNNING`, run rounds until a non-RUN
command or termination; PAUSE → (two ERISCs) master handshake start, `execute_pause_command`, master handshake finish —
so ERISC 1 is parked in its non-master handshake for the whole pause; any other command directly from RUN →
`ASSERT(false)` and the loop re-reads (spins). `execute_pause_command` (V3:1954-1970, ERISC 0 only): loop until RUN or
termination: `state = PAUSED`; PAUSE → `run_routing_without_noc_sync()`; DRAIN → `run_drain_step` (state DRAINING,
`ASSERT(false)` not implemented, spin while DRAIN, V3:1944-1952); RETRAIN → `run_retrain_step` (state RETRAINING, one
`run_routing_without_noc_sync()` "placeholder", spin while RETRAIN, V3:1933-1942). ERISC 1 never reads commands. ERISC 0
writes `state = INITIALIZING` at kernel start and again just before the loop (V3:3188-3190, 3885-3887).

**11.5 Termination.** `got_immediate_termination_signal` (FU:23-31): termination word == IMMEDIATELY_TERMINATE (2) or
the launch mailbox's `exit_erisc_kernel`. Checked once per round and in every wait loop. GRACEFULLY_TERMINATE (1) is
"non functional" (`tt_metal/fabric/fabric_edm_packet_header.hpp:34-43`) and no router code reads it; the router keeps running.

**11.6 Heartbeat.** Every 64 rounds `*FABRIC_KERNEL_HEARTBEAT_ADDR = 0xDCBA0000 | counter` (V3:2446-2448; also SE:582-584,
RE:1399-1401), on both ERISCs of a two-ERISC router (each with its own counter). BH address is the firmware's
`eth_status.heartbeat[0]` (V3:246-253).

**11.7 Worker connection (worker channel and VC2).** `check_worker_connections` (FU:46-70): not established and
semaphore is open (1) or close-requested (2) → established; `cache_producer_noc_addr` (reads `worker_xy`,
`worker_semaphore_address` from the location info, CH:531-539) and `notify_worker_of_read_counter_update()` with flush
(FU:38-44). Established and semaphore == 2 → not established; `teardown_worker_connection<posted=true>` (CH:514-529):
semaphore = 0 (unused), `edm_read_counter = local_read_counter`, `noc_semaphore_inc` +1 on the worker's teardown
semaphore. Polled every visit of the worker channel (V3:1336-1340); per round for VC2.

**11.8 Static (router-to-router) connections.** Init only, after sync 5 (V3:2525-2573, 3869-3876): for each serviced
channel with `SENDER_CH_i_WAIT_STATIC_CONNECTION` (host: has a static peer, EDB:1446-1451), spin until the connection
semaphore reads open or close-requested (BH: with termination check; BH ERISC 0: `run_routing()` every 100000 polls),
then cache the upstream's credit address. Never torn down. Single-TXQ builds iterate every channel tuple entry
(V3:2566-2572).

**11.9 Teardown** (`teardown`, V3:2722-2800). Two-ERISC sync T1; for each serviced receiver channel
`all_buffer_slot_transactions_acked` (`noc_async_write_barrier_with_trid` for every trid of the range, on both NoCs if
they differ, TT:115-124); `ncrisc_noc_counters_init()` (software counters re-read from hardware, since the router's
NoC calls do not maintain them); sync T2; if `WAIT_FOR_HOST_SIGNAL` and local handshake master:
`notify_subordinate_routers(mask, self, termination_signal_addr, *termination_signal)` (forwards the termination word to
the chip's other routers, FU:111-126); `noc_async_write_barrier(); noc_async_atomic_barrier()`; BH:
`noc_clear_packet_tags(noc_index)` ("tagged write completion does not reset command-buffer transaction IDs", BHN:242-246);
sync T3; restore TXQ sequence-update timeouts (teardown master); ERISC 0 writes `EDMStatus::TERMINATED`.

**11.10 ERISC sync primitive** (`wait_for_other_local_erisc`, V3:2698-2720). Stream scratch register
`MULTI_RISC_TEARDOWN_SYNC_STREAM_ID`: ERISC 0 writes 0x0fed, spins until `(reg & 0x1FFF) == 0x1bad`, writes 0; ERISC 1
spins for 0x0fed, writes 0x1bad. No termination check by design (V3:2698-2702). Count: six in init (kernel_main V3:3232, 3688,
3750, 3809, 3861, 3878; `receiver_erisc::run_kernel` V3:2900, 3056, 3114, 3126, 3137, 3144) and three in teardown; both ERISCs must make the same
number in the same order whichever program each runs.

**11.11 Init sequence (generic kernel_main, V3:3172-3887)**, with the duties that only one ERISC may do marked ★:
trimming reset; `state = INITIALIZING`; `set_l1_data_cache<ENABLE_RISC_CPU_DATA_CACHE>()` (disables the BH data cache
when false, risc_common.h:149-175); `initialize_fabric_telemetry` (memset, static info, router_state per ERISC);
`ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD = 32` on the sender TXQ; ★ teardown master saves and sets
`ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT` on sender and receiver TXQs (V3:2669-2687; host 1000 on BH 2-ERISC, CMRB:556-560);
multi-TXQ: ★ the receiver-servicing ERISC runs `eth_enable_packet_mode(receiver_txq)` (TX:76-134: packet resend on
both TXQs, MAC DA for TXQ1, per-TXQ config tables, RX queues packet mode, MAC RX routing), zeroes the local receiver
ack/completion counters and sets accept-ahead on TXQ1; the sender-servicing ERISC zeroes the to-sender counters
(V3:2802-2815, 3202-3210); stream registers 0/1 init (before the handshake, so the far side cannot touch them first,
V3:3215-3224); ★ master sets the retrain register; **sync 1**; 2D stream registers; code-profiling clear; runtime args;
zero serviced channels' connection semaphores and `SenderChannelProducerCursor` blocks (V3:3377-3392); `STARTED`;
`edm_read_counter = 0` for every channel; downstream adapters, relay, receiver-channel header zeroing (all skipped by
an ERISC 0 that runs SE; done by both ERISCs when both run the generic loop, same values); sender headers zeroed;
worker interfaces constructed (`edm_read_counter` = 0 for worker/VC2 channels, `SENDER_NUM_BUFFERS` otherwise,
V3:2577-2612); trid trackers; **sync 2**; ★ Ethernet handshake on TXQ0 (ERISC with `ENABLE_ETHERNET_HANDSHAKE`: ERISC 0 on
BH 2-ERISC, EDB:149-158), neighbor ids into telemetry, `REMOTE_HANDSHAKE_COMPLETE`, then with `WAIT_FOR_HOST_SIGNAL` the
chip-local handshake (master waits for `num_local_edms - 1` notifications, notifies all; others notify master and wait),
`LOCAL_HANDSHAKE_COMPLETE`, wait for host `READY_FOR_TRAFFIC`, master forwards it (V3:3691-3748); **sync 3**; tensix
ready wait (§8); open downstream connections (and relay) (V3:3763-3807); **sync 4**; two-ERISC cmd-buf setup (§9);
**sync 5**; static connections (§11.8); **sync 6**; ERISC 0 `state = INITIALIZING`.

## 12. Instrumentation

| Feature | Gate (kernel / host) | Per packet | Per pass | Per round | Init |
| --- | --- | --- | --- | --- | --- |
| Dynamic BW telemetry | `FABRIC_TELEMETRY_BANDWIDTH` = `ENABLE_FABRIC_TELEMETRY` && stats mask BANDWIDTH (CT:597-613) / rtoptions telemetry settings per chip, channel, ERISC (EDB:866-884, 886-896) | `update_bw_counters` after each send (V3:1215-1217, 1227-1229; SP:168-170) and each admitted receive (V3:1636-1638; SP:280-282): packets++ and words += ceil(bytes/4) into tx (one-ERISC or ERISC 0) and rx (one-ERISC or ERISC 1) (V3:953-966) | — | timestamp at round start, `update_bw_cycles(delta, tx, rx)`: elapsed and active (if progress) cycles (V3:2146-2149, 2431-2435, 968-983); stored to L1 in `update_telemetry` (V3:904-927) | telemetry struct init (V3:2817-2849) |
| Other dynamic stats | `FABRIC_TELEMETRY_ANY_DYNAMIC_STAT` (ROUTER_STATE, HEARTBEAT_TX/RX) | — | — | `update_telemetry` (V3:866-928): router state copy; TX heartbeat++ if tx progress or no sender channel active (reads every sender free-slots register, V3:773-783); RX heartbeat++ if rx progress or receiver 0's packets-received register is 0 | same |
| Perf telemetry recorder ("bandwidth telemetry" in the test harness) | `PERF_TELEMETRY_MODE` = 1 (LOW_RESOLUTION_BANDWIDTH) / rtoption `enable_fabric_bw_telemetry`, set by `tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp:250-255` in perf modes unless `TT_FABRIC_NO_BW_TELEMETRY` (EDB:898-902) | `record_packet_send` per send: packets++, words += bytes/16 (TB) | — | window opened at round start, closed at round end; if an event was captured or any sender channel is active, accumulate (cycles, words, packets) into the L1 buffer at `PERF_TELEMETRY_BUFFER_ADDR` (V3:2142-2144, 2471-2480; TB) | buffer reserved always on BH (EDB:249-253) |
| Code profiling | `CODE_PROFILING_ENABLED_TIMERS` bit RECEIVER_CHANNEL_FORWARD, `CODE_PROFILING_BUFFER_ADDR` / rtoption `enable_fabric_code_profiling_rx_ch_fwd` (EDB:904-916) | — | per receiver step with a packet present: wall clock around the forward section; adds duration and count at the timer's L1 slot (V3:1534-1540, 1753; CP) | — | `clear_code_profiling_buffer` (V3:3261-3263) |
| Channel-trimming capture | `ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE`, `RESOURCE_USAGE_CAPTURE_OUTPUT_L1_ADDRESS`, `PACKED_DOWNSTREAM_VC{0,1}_SENDER_CHANNEL_IDS` (CT:847-867) / rtoption capture (EDB:280-292, 1282-1289, 1513-1517) | send: channel used, min/max packet size (V3:255-261); receive: receiver channel saw data (V3:1639), NoC send type used (PT:305), forwarded-to sender mask (V3:747-763; 1D V3:436, 453, 486) | — | — | `reset` (V3:3180-3182) |
| Channel-trimming replay | `DISABLE_RX_CHx_FORWARDING` (CT:841-845) | see §14.1 | | | |
| Packet recorder | `fabric_packet_recorder.hpp` included (V3:55) | **never used** (no call sites in v3q or MAIN) | | | |

The trimming recorder's methods are empty when capture is off (TR:72-81); calls in hot paths are unconditional in the
source. All of these are also in SE and RE except VC2/speedy-specific call sites; RE has code profiling, SE does not
need it.

## 13. Data cache and TXQ spin-wait flags

All three are constexpr-false on the host (`enable_risc_cpu_data_cache = false`, EDB:1134;
`eth_txq_spin_wait_* = false`, EDB:1076-1077).

**`ENABLE_RISC_CPU_DATA_CACHE`** (CT:202). Init: `set_l1_data_cache<flag>()` enables (true) or disables (false) the BH
L1 data cache (`tt_metal/hw/inc/internal/tt-1xx/risc_common.h:149-175`); every program calls it (V3:3192; V3:2876 in `receiver_erisc::run_kernel`). `router_invalidate_l1_cache<flag>()`
(`fence` on BH, `tt_metal/hw/inc/internal/tt-1xx/cache.h:12-16`) sits at: top of each pass (V3:2170, 2208), each sender step (V3:1435), each receiver
step (V3:1889), before reading a header (V3:1511, 1543), 1D forward (V3:425), every downstream space check (FA:184),
counter-credit reads (FC:155, 163), termination check (FU:26), producer address caching and teardown (CH:516, 533),
every init wait loop (V3:2545, 2710, 2716; FU:81). Effects on code paths: disables the overlapped send/completion read
(V3:1095-1097), excludes SE (SE:129); the speedy receiver static-asserts it off (SP:261). RE does not exclude it.

**`ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA`** (CT:409). Off: a busy TXQ makes the send yield (V3:1159-1161, 1165-1169;
SP:122-124). On: spin until idle before the data command (V3:1174-1177; SP:143-147); the BH grouped path and the
overlapped completion read are not used (V3:1095-1097, 1145); excludes SE.

**`ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK`** (CT:410-411). Off: acks and completions are only attempted when the
receiver TXQ is idle (V3:1507-1509, 1769-1771, 1809-1811, 1844-1846; RE:887-891). On: the attempt spins inside
`receiver_send_completion_ack<true>` / `receiver_send_received_ack<true>` (FC:255-275) or before the batched flush
(V3:1828-1831; RE:916-919). The speedy receiver always calls `<false>` with no prior check (SP:312).

## 14. Other generic-loop behaviors the specialized programs skip

**14.1 Receiver channel with forwarding disabled** (`is_receiver_channel_forwarding_disabled`, CT:841-845; host
EDB:1518-1540: capture shows neither forwarding nor local delivery on that VC, or speedy on VC0, or always for VC2). The
generic step still reads, decodes and admits the packet but the whole trid / forward / local-delivery block is compiled
out (V3:1640); it then `wr_sent++` and decrements the register: the packet is consumed and credited without any NoC
command. RE requires forwarding enabled (RE:64-67).

**14.2 1D `RoutingFields` header** (V3:426-440): forward if not the last multicast value, deliver if the hop distance is
terminal; `update_packet_header_for_next_hop` decrements distance or range (PT:625-634). Unreachable from the current
host (FCX:396).

**14.3 1D low-latency path in the generic loop** (V3:441-491; also implemented by RE): WRITE_ONLY deliver;
FORWARD_ONLY forward; WRITE_AND_FORWARD forward then deliver, except sparse multicast: deliver, advance
`write_idx += counts[chip_idx]`, `chip_idx++`, then forward the mutated header. Forward = shift the route field
(`>> FIELD_WIDTH`, refill from the extension words when empty, PT:642-678) in L1, then NoC write. Admission
(V3:301-315): WRITE_ONLY needs nothing, else downstream space.

**14.4 Dead or unused state.** `port_direction_table` is declared uninitialized and passed through but never read
(V3:3859); `!fuse_receiver_flush_and_completion_ptr` branch is dead (`static_assert(fuse == 1)`, CT:180; V3:1755-1778);
`enable_trid_flush_check_on_noc_txn` dead (CT:768; V3:1629-1632); `ENABLE_INTERRUPTS` read, unused (CT:331).

**14.5 Behaviors shared by all programs that a new kernel must keep** (not features, but obligations): deferred
flushed atomics (`can_defer_local_atomic_flush`, V3:263-289; PT:152-216) — nothing may use `write_at_cmd_buf`
(atomics, inline writes, stateless forward credits, `noc_inline_dw_write`) while one is staged; drain at end of round
and on an idle pass. Completions only after the slot's trid has no outgoing writes ("sent", not "acked", TT:105-113;
BHN:743-746); strict-flush atomics wait for acks (BHN:739-750).

## 15. Ordering rules, protocol obligations, single-owner duties (summary)

1. `src_ch_id` is written into the L1 header before the Ethernet data command reads it (V3:1154-1156, 1183); not under
   mux.
2. Data command, then (after the TXQ is idle) the remote packets-received increment on the same TXQ (V3:1183, 1207-1213,
   1222-1224). Every TXQ command is issued only after an idle check (TUN:73-83, BH-55: read CMD before STATUS).
3. Receiver: packets-received register is decremented after all NoC commands of the packet (non-FLA, V3:1739-1742) or at
   ack time (FLA, V3:1512). Ack before processing (FLA). Completions in slot order, after the trid is "sent".
4. Sender: read completions before acks (MAIN:1284-1286). Far-ring slots return on completions; upstream credits on
   completions (no FLA) or acks (FLA).
5. 2D: forwards (compact order) before local delivery; 1D write-and-forward: forward first except sparse multicast.
6. Worker credits are absolute counters; when no worker is connected they go to `edm_read_counter` in local L1.
7. Init: stream registers before the handshake; six syncs; downstream opens after the tensix-ready wait; stateful
   command-buffer state programmed last for the single live downstream. Teardown: three syncs; trid barriers before
   counter re-init; BH clears packet tags.
8. Single owner: base-firmware hooks and the retrain-sync master role (physical ERISC 0); the Ethernet handshake
   (ERISC 0); `eth_enable_packet_mode` and the TXQ1 setup (receiver ERISC); seq-update timeout set/restore and
   `TERMINATED` (teardown master = ERISC 0); router commands (ERISC 0). Both ERISCs write the heartbeat.
9. Code size: VC1 builds compile at -Os and outline the steps; VC1 FLA does not fit the ACTIVE_ETH window (EDB:1343-1348).

## 16. Direct hardware register access (ERISC builds)

| Location | What | Existing API with the same operation |
| --- | --- | --- |
| V3:1000-1004 `read_free_slots_raw` | `NOC_STREAM_READ_REG(free-slots stream, SPACE_AVAILABLE)` unmasked, grouped | No function in DF/BHN/TUN/ethernet; macro is in `internal/tt-1xx/blackhole/noc/noc_overlay_parameters.h:70`. Fabric wrapper `get_ptr_val` (SR:43-59) reads the same register and masks |
| V3:1141-1151 | raw TXQ pointer: dummy `ETH_TXQ_CMD` read, `ETH_TXQ_STATUS` read (BH-55) inside a fenced group with the header read | `internal_::eth_txq_is_busy` (TUN:73-83) does the same two reads; `eth_txq_reg_read` (tt_eth_api.h:30) |
| V3:1197-1200 | same pair plus an L1 counter read, grouped | as above |
| V3:1191-1192, 1211-1213 | remote register write: `ETH_TXQ_DEST_ADDR`, `ETH_TXQ_REMOTE_REG_DATA`, `ETH_TXQ_CMD = START_REG` | `internal_::eth_write_remote_reg_no_txq_check` (TUN:135-141); fabric `remote_update_ptr_val` (SR:99-107) |
| V3:1674-1676; RE:1107-1109 | `NOC_STATUS_READ_REG(NIU_MST_WR_ACK_RECEIVED)`, `NIU_MST_REQS_OUTSTANDING_ID(trid)` ×2 (deferred-atomic proof) | No raw-count getter. `ncrisc_noc_nonposted_writes_flushed` (BHN:739-741) compares the ack count to the software counter; `noc_available_transactions(noc, trid)` (BHN:597-599) returns `MAX - outstanding(trid)`; `ncrisc_noc_nonposted_write_with_transaction_id_flushed` (BHN:748-751) is the boolean |
| RE:76-79, 645-653 | `NOC_STREAM_READ_REG` packets-received / local credit copies (grouped) | as row 1 |
| RE:290-296 | raw volatile read of a stream register by computed address (runtime index) | `get_ptr_val(uint8_t)` (SR:52-59) |
| RE:339-340, 393; FA:347 | raw volatile store `pack(-1)` to a local stream update register | `increment_local_update_ptr_val(stream, -1)` (SR:71-74); no hw/inc function |
| RE:789-794 `writes_outgoing` | raw read of `NOC_STATUS(NIU_MST_WRITE_REQS_OUTGOING_ID(trid))` | `ncrisc_noc_nonposted_write_with_transaction_id_sent` (BHN:743-746) returns `== 0`, the only use made of the value |
| EU:52-62 (generic loop stateful forwards, RE `forward_payload`) | `noc_cmd_buf_ready` spin + `NOC_CMD_BUF_WRITE_REG(NOC_PACKET_TAG, trid)`, then `ncrisc_noc_write_with_state<skip_cmdbuf_chk>` | `ncrisc_noc_set_transaction_id` (BHN:1454-1458) is exactly the ready-wait + tag write |
| SR:24-38 (`read/write_stream_scratch_register`, used by both sync protocols) | `NOC_STREAM_READ/WRITE_REG(..., STREAM_REMOTE_SRC_REG_INDEX)` | none in hw/inc |
| SR:65-118 (`increment_local_update_ptr_val`, `init_ptr_val`) | `NOC_STREAM_WRITE_REG` | none in hw/inc |
| TX:76-134 `eth_enable_packet_mode` | `eth_reg_write` / `eth_txq_reg_write` to TXQ/RXQ control, MAC DA and routing registers at addresses defined in TX:17-31 | `eth_reg_write` is the tt_eth_api.h API; the register addresses are local to TX |
| V3:2680-2693, 2808, 2881, 3197; SE:323-324, 347-348 | `eth_txq_reg_read/write` (seq timeout, accept-ahead, TXQ status) | these are the API (tt_eth_api.h:26-32) |
| V3:2114, 2447; SE:583; RE:1398-1400 | heartbeat: raw store to an L1 firmware word (not a register) | BH macro `MEM_SYSENG_ETH_HEARTBEAT` (FW:21); no function |
| PT:224-282; FA:308-310, 387-389 | raw `NOC_CMD_BUF_WRITE_REG` sequences | `COMPILE_FOR_DRISC` only — not compiled into ERISC routers |

All other NoC traffic goes through API calls: `noc_async_write_one_packet_with_trid[_set_state|_with_state]`,
`noc_write_init_state` / `noc_write_with_state`, `noc_inline_dw_write[_set_state|_with_state]`,
`noc_semaphore_inc[_set_state|_with_state]`, `noc_async_write_barrier_with_trid` (DF; PT; FA; RE:425-470).

## 17. Observed anomalies and open questions (not verified on hardware)

1. VC2 speedy receiver trid flip leaves the 4-bit trid range (§6).
2. Generic batched completions with **stream-register** credits issue up to 4 remote register writes back to back with
   one TXQ-idle check before the first (V3:1820-1833 with FC:80-82). Stream credits are used on single-TXQ builds
   (WH, BH one-ERISC, mux, UDM). MAIN sends one completion per pass with a check. Whether the TXQ tolerates this is
   not established here.
3. Speedy receiver sends completions without any TXQ check (SP:312-315; same in MAIN).
4. UDM relay adapter's null teardown pointer writes L1 address 0 at open (§7).
5. `flush_write_to_noc_pipeline`'s non-deadlock-avoidance fallback loops trids `0..N-1` without the receiver channel's
   `RX_CH_TRID_STARTS` offset (PT:134-146), so a receiver channel other than 0 waits on channel 0's trids.
6. Mux: only receiver channel 0's fixed `src_chan_ids[0]` is set (§8). Host/kernel disagree on the skip flag (§8).
7. One-ERISC BW telemetry counts every sent and every received packet in both tx and rx (V3:953-966; same in MAIN).
8. Non-master one-ERISC builds poll an uninitialized retrain register (§11.3).
9. Batch size differs between the generic loop (`>= 8` slots → 4, V3:1783) and RE (quarter ring, max 4, RE:150); the
   generic check reads only the batch's last slot (and first when idle), RE reasons per command buffer.
10. Speedy sends do not set `did_something` (§10.2); speedy credits for the last `< AMORT` packets are held (§10.2-10.3).
11. An unaddressable intermesh exit stalls its receiver channel forever after `ASSERT(false)` (§5); DRAIN/RETRAIN
    without PAUSE spins on `ASSERT(false)` (§11.4).
12. Which router the opened VC0 adapters of a crossover router register with, who clears
    `EDM_LOCAL_TENSIX_SYNC_PTR_ADDR`, and whether the host signals termination to every router or only the local
    master were not traced.
