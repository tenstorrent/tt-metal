# Router map: fabric ERISC router v2, per ERISC (Blackhole, two ERISCs)

What the router does today, per ERISC, per pass and per packet, and what it must do. Written as the starting spec
for whole-router redesigns (PLAN.md phase 0a).

**Sources and conventions**
- Source: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp` as in the tree (= `variants/s79mc`,
  which is `s79m` minus two unused cold-helper templates), plus the headers it uses. Line numbers below are the tree's.
- Build resolved as the best one: `TT_FABRIC_EXP_FLAGS=13129088` (bits 7, 8, 10, 12, 14, 19, 22, 23), no DRISC
  offload, phase profiler off, delays 0, BW telemetry off, keepalive `EXP_SEQ_UPDATE_TIMEOUT=1000`, `-O3`.
- Evidence ELFs: `fabric_exp/variants/s79m/elf/` (320 configs, 640 ELFs). s79m and s79mc compile to the same hot loop.
  - **Hot E0** `10733501891889981293`: 2D, EAST-facing (`MY_DIRECTION=0`), eth channel 11, ERISC 0. Four VC0 sender
    channels: worker ch0 with 17 slots, forwarding ch1–3 with 16 each. Far receiver ring 16 slots of 4448 B (96 B
    header + 4352 B payload). Only ch3 has an upstream router (`SENDER_CH_3_WAIT_STATIC_CONNECTION=1`).
  - **Middle E1** `10856749301170935304`: 2D, WEST-facing, eth channel 4, ERISC 1. One receiver channel, 16 slots,
    16 transaction IDs. Two downstream connections, one live forward direction (EAST).
- Costs are ERISC core cycles at 1.35 GHz; cycles/packet = 1.35e9 / packets per second. "§n" is `RESULTS.md` §n.
  Profiled phase costs run 5–16% long (timer boundaries, §72, §73); read them as shares.
- "(inferred)" marks anything not measured in RESULTS and not read directly from code or disassembly.
- "Pass" = one inner-loop iteration. "Visit" = one channel step inside a pass.

---

## 0. The router in one page

- **E0 (TX)** runs the sender channels. Every pass it reads the four VC0 free-slot stream registers together, then
  visits the worker channel (ch0) and the three forwarding channels (ch1–3). A visit with a packet sends it over
  Ethernet TXQ0, then reads completions and returns credits upstream.
- **E1 (RX)** runs the receiver channel. Every pass it reads the packets-received stream register. With a packet it
  decodes the header, checks downstream space, NoC-writes the packet locally or to the next router, retires the slot,
  and every 4 packets sends one 96 B completion-credit frame back over TXQ1.
- **Both** run an outer loop every 32 passes: termination, router commands (E0), context switch to base firmware
  (E0 leads, E1 follows), heartbeat.
- **1-hop E/W 4352 B write: 132 cycles/packet, E0 binds** (§72, §79). Of E0's ~130 per packet only ~50 is the send.
  About 70 is visiting channels: the worker channel's own per-pass work (~63 per pass) plus ~16 for three idle
  forwarding channels (§73).
- **Multi-hop and multicast bind on E1, or on both ERISCs and the credit loop.** Forwarding costs E1 ~90 per packet
  (decode 30 + forward 60, §72). Scatter, atomics and fused writes are bound by E1's local delivery (§72).
- **Only bit 22 still selects code in v2.** Bits 7, 8, 10, 12, 14, 19 and 23 are hard-wired in v2 and its headers,
  and so are the E1 changes of §53, §57, §63 and §74. v2 with all flags off is therefore not main (§1.1).

---

## 1. Roles and configurations

### 1.1 What the experiment flags still select in v2

`grep fabric_exp_` over v2 and its headers finds only these uses:

| Flag / knob | Still a switch in v2? | Where | Effect with it on |
| --- | --- | --- | --- |
| bit 22 `EXP_RX_BATCH_COMPLETION` | **yes** | ct_args `NUM_TRANSACTION_IDS`; v2 `can_defer_local_atomic_flush()` (l.300), completion block (l.1806); transmission `flush_write_to_noc_pipeline` (l.111) | Single receiver channel and no deadlock avoidance: 16 trids, trid = slot index. Completions batched up to 4, with a 4-idle-poll hold in 2D. Flushed atomics deferred (2D, 2-ERISC). Strict flush on BH = one global check. With bit 22 off: 4 trids (8 with deadlock avoidance), one completion per slot, and a blocking per-trid flush loop |
| bits 7, 8, 10, 12, 14, 19, 23 | **no**, hard-wired | `sender_channel_try_send`, `read_grouped_free_slots`, `skip_idle_per_channel`, `sender_channel_overlaps_send_completion` | Always on, subject only to their structural conditions: counter credits, BH, no spin-wait, no data cache, no first-level ack |
| E1 §53 bits 27/28, §56 variant A, §57 bits 26/28, §63 deferral, §74 two-slot check | **no**, hard-wired | `admit_2d_dispatch`, `use_2d_single_live_stateful_noc`, `PackedHeaderWord40`, completion hold, deferral block (l.1664–1720) | Always on where their conditions hold |
| bits 0, 5, 6 (speedy path), TX/RX delays | yes | speedy header, `fabric_experiment_spin` | Benchmark-only, off |
| `EXP_SEQ_UPDATE_TIMEOUT` | yes | `exp_set_seq_update_timeouts` (E0 only) | Every study run used 1000 (firmware default 8000). Removes the one-way Ethernet floor above 4114 B frames (§15, §16) |

### 1.2 Configuration switches and the code they select

| Configuration | Selected by | What changes in the code | Study ran it? |
| --- | --- | --- | --- |
| **2D vs 1D** | `FABRIC_2D` define, `IS_2D_FABRIC` | **2D:** 4 VC0 senders (5 with express/Z). Destination-major action byte decoded from `route_buffer`; the header is never modified in transit. `admit_2d_dispatch` / `forward_2d_dispatch`; deferral of flushed atomics. **1D:** 2 senders. `LowLatencyRoutingFields` (WRITE_ONLY / FORWARD_ONLY / WRITE_AND_FORWARD); the header's routing field is shifted per hop before forwarding (`update_packet_header_for_next_hop`); `can_forward_packet_completely` reads the adapter's runtime stream id; no deferral (flushed atomics spin) | 2D: yes, everything. 1D: §76–§81 only (bandwidth, per-ERISC split, 1D sanity 181/181) |
| **E/W vs N/S 2D router** | `MY_DIRECTION` | **E/W:** decode reads the X byte only (1 load). Live forward mask trimmed to the opposite direction (`trim_ns_for_intramesh_ew`, l.563), so `LIVE=0b0001`, one stream read for admission, and **stateful single-live NoC forwarding** (`use_2d_single_live_stateful_noc`). **N/S:** Y then X (2 loads), up to 3 live directions, all live downstream registers read together, stateless forward per copy | Both |
| **Edge router / downstream count** | `NUM_DOWNSTREAM_SENDERS_VC0`, `VC0_DOWNSTREAM_EDM_SIZE`, runtime `has_downstream_edm` mask | Which downstream adapters are built and opened at init. Also which forwarding channels have an upstream router: `SENDER_CH_n_WAIT_STATIC_CONNECTION` | Yes (2x4 mesh: every router is on an edge) |
| **Forwarding channels with no upstream** | `IS_SENDER_CHANNEL_n_SERVICED=1` with `WAIT_STATIC_CONNECTION=0`. Host: `WAIT_STATIC_CONNECTION` = `has_static_peer`, set only when an upstream router builds a connection; a non-worker channel without one gets `LIVE_CHECK_SKIP` forced (`should_force_internal_sender_skip`, `erisc_datamover_builder.cpp` l.1193–1209, 1433–1446) | Channel still visited every pass. Its upstream credit address is never cached (`establish_edm_connection` runs only for wait-static channels) | Yes. Every 2D E0 in the saved set polls 1–2 such channels; the hot E0 polls 2 (ch1 from W, ch2 from N) |
| **Worker channel** | ch0 is worker-type (`SENDER_CH_0_LIVE_CHECK_SKIP=0`), `IS_SENDER_CHANNEL_0_SERVICED` | Per-pass connection poll; upstream credit = absolute read counter written to the worker's L1. Routers where no worker ever connects still poll it every pass (§73, §76) | Always serviced. Trimmed builds (ch0 compiled out) ran only in the v1 era (§8, §17, §23, §26–§29) |
| **Forwarding channel count** | `ACTUAL_VC0_SENDER_CHANNELS`, `IS_SENDER_CHANNEL_n_SERVICED` | The grouped read covers the first `min(4, N)` VC0 channels. A 5th VC0 channel (express/Z) is read alone every pass, without idle skip | 2 (1D) and 4 (2D) only |
| **Idle skip (bit 19 form)** | `skip_idle_per_channel`: every serviced forwarding channel among 1..3 is live-check-skip, uses counter credits, no first-level ack. `skip_idle_bit_test` needs power-of-two slots below 2^11, all equal | Idle forwarding visit = one bit test of the grouped value plus sent == processed. Otherwise a masked compare | On in all 2D and 1D-line E0s. Off on ring (first-level ack) |
| **VC1 present** | `FABRIC_2D_VC1_ACTIVE` / `_SERVICED`, `ACTUAL_VC1_SENDER_CHANNELS` | 3–4 more sender visits per pass on E0, ungrouped and without idle skip. A second receiver step per pass on E1. Steps lose `FORCE_INLINE`. 4 trids, no deferral. **`-Os` build** (`get_kernel_opt_level`) | **No.** The two-2x2 harness exists (`fabric_exp/vc1`) but RESULTS has no VC1 numbers |
| **VC2 / intermesh** | `FABRIC_2D_VC2_SERVICED`, `IS_INTERMESH_ROUTER*`, `IS_RECEIVER_CHANNEL_n_INTERMESH_INGRESS`, `FABRIC_2D_VC0_CROSSOVER_TO_VC1` | VC2 runs speedy steps plus a per-outer-iteration connection check. Intermesh ingress rewrites the landing route and decodes Y-first. Edge routers detect intermesh exit and forward to the boundary direction. Crossover forwards VC0 into VC1 adapters. All disable stateful single-live forwarding | No |
| **Speedy (neighbor exchange)** | `ENABLE_SPEEDY_VC0` (one VC0 sender and no deadlock avoidance, or a trim capture saying worker-only) | Separate steps (`fabric_erisc_router_speedy_path.hpp`): no grouped read, no forwarding. Sender completions read every 2 sends, worker credits every ≥2 completions. RX uses 2 ping-pong trids and acks per batch of 4. `DISABLE_RX_CH0_FORWARDING=1` | 1D neighbor exchange: §10, §45, §76 (at the wire, 125.0) |
| **Deadlock avoidance / bubble flow control** | `ENABLE_DEADLOCK_AVOIDANCE` (Ring; Torus axes; never intermesh), `SENDER_CH_n_IS_INJECTION` | Injection channels send only with ≥2 free far slots. Forces first-level ack on VC0. 8 trids. No completion batching, no deferral, no idle skip, no speedy | **Yes, 1D ring only** (§76 half-ring rows; 48 ring ELFs in the saved set: `ENABLE_DEADLOCK_AVOIDANCE=1`, `ENABLE_FIRST_LEVEL_ACK_VC0=1`). Never 2D torus. Only bandwidth and E0/E1 split measured |
| **First-level ack** | `ENABLE_FIRST_LEVEL_ACK_VC0` (= deadlock avoidance); VC1 always 0 | E1 sends an ack frame per received packet before processing it. E0 returns upstream credit on the ack; completion only frees the far slot. Completion one slot at a time | 1D ring only (§76). RX ack cost 22 (§3, v1) |
| **Counter vs stream-register credits** | `VCn_USES_COUNTER_CREDITS` (counters on every VC when multi-TXQ; VC1 also under express; VC2 always) | **Counters:** E1 keeps absolute per-source counters in L1 and sends the whole ack+completion block as one Ethernet write (96 B in 2D, 32 B in 1D) on TXQ1; E0 reads its L1 counter. Needed by bits 7, 19 and 23. **Stream:** each credit is an Ethernet remote register write; E0 reads and decrements a stream register every visit | Counters only (two ERISCs ⇒ multi-TXQ) |
| **Single vs two ERISC** | `NUM_ACTIVE_ERISCS`, `MY_ERISC_ID`, `SENDER_TXQ_ID != RECEIVER_TXQ_ID` | See §3.13 | Two only |
| **UDM relay, tensix mux extension** | `UDM_MODE`, `FABRIC_TENSIX_EXTENSION_MUX_MODE` | UDM: local delivery becomes a router-to-router send to a tensix relay, with relay-space admission. Mux: `src_ch_id` is fixed, no per-packet stamp. Both force single-ERISC on BH (`is_fabric_two_erisc_enabled`) | No |
| **Context switch** | `ENABLE_CONTEXT_SWITCH` (E0), `IDLE_CONTEXT_SWITCHING`, `SWITCH_INTERVAL=32` | E0 yields to base firmware on the 34th consecutive outer iteration with no send (`did_nothing_count++ > 32`, ~1,100 passes) and coordinates with E1 through stream scratch register 30 | Always (E0 master) |

**Compiled out in this build** (one line each): DRISC offload (`FABRIC_DRISC_OFFLOAD`); phase profiler (calls
compile to nothing at mode 0); BW and perf telemetry, code profiling; channel-trimming capture; data cache
(`ENABLE_RISC_CPU_DATA_CACHE=0`, so every `router_invalidate_l1_cache` is a no-op); spin-wait TXQ modes; stack paint;
debug print and watcher (which would turn off the stateful NoC APIs); Wormhole branches.

**Not run in the study**: single ERISC, VC1/VC2, intermesh, express/Z, UDM, tensix mux, stream-register credits, 2D
deadlock avoidance, data cache, Wormhole, and `-Os` builds of these configs (§8 notes only the opt-level effect on
the v1 router).

---

## 2. What the router must do (obligations)

These are protocol facts. A redesign may move or batch them, but not drop them. **[rule]** marks standing study rules
rather than protocol needs.

**Sender side (E0 today)**
1. Detect a packet in a sender-channel slot. The producer writes the slot, then decrements the channel's free-slot
   stream register over the NoC. Unsent packets exist exactly when the register is below `num_buffers`.
2. Send only if the far receiver ring has a free slot. All VC0 channels share one far ring, so there is one credit
   count and one far slot pointer per VC. Injection channels under bubble flow control need ≥2 free.
3. Never write a TXQ command while the queue is busy. The busy check needs the BH-55 dummy `CMD` read before
   `STATUS`. Removing the can-send check hung the router (§11).
4. Stamp `src_ch_id` in the header (counter credits route completions by it), then send header + payload to the far
   slot: 4 TXQ register writes.
5. After the data command, increment the far receiver's packets-received stream register on the same TXQ, so it
   lands after the data: busy check plus 3 TXQ register writes.
6. Mark the slot consumed: +1 on the channel's free-slot stream register.
7. Track the far receiver's completions (or acks), free far slots, and return slot credits upstream:
   - worker: absolute read counter, NoC-written to the worker's L1, or copied into `worker_location_info` while
     disconnected;
   - router: +N on the upstream router's free-slot stream register.
8. Worker connection protocol: see an open request and cache the worker's credit address; see a close request,
   copy the read counter for the next producer, mark the connection unused, and increment the worker's teardown
   semaphore. **[rule]** Poll the connection on every pass, never from the outer loop only.
9. Router-to-router (static) channels: wait at init for each upstream router to connect.

**Receiver side (E1 today)**
1. Detect: packets-received stream register > 0.
2. Read the header: route (1D field or 2D action byte), send type, payload size, `src_ch_id`.
3. Admission: every target has room. Each downstream router needs a free slot (local copy in a stream register);
   in UDM the relay needs space.
4. Tag the slot's NoC writes with a transaction ID, so the slot can be freed once its writes have left the NIU.
5. Forward: NoC-write header + payload into the downstream sender-channel slot, then decrement the downstream's
   free-slot register over the NoC (the doorbell). Decrement the local copy and advance the downstream slot index.
   1D: update the routing field first.
6. Deliver locally per send type: unicast write, atomic inc, inline write, fused write + atomic, scatter (2–4 chunks,
   the last may be an atomic), sparse multicast.
7. Atomic with flush: STRICT means all earlier writes on this NoC are acked before the atomic issues; RELAXED means
   this channel's own writes (§70–§74).
8. Retire the slot: decrement packets-received, advance the read index.
9. Completion: once a slot's writes are sent, credit the sender channel that sent it (`src_ch_id`) over Ethernet on
   the receiver TXQ, after a TXQ busy check.
10. With first-level ack: ack each packet on receipt.
11. **[rule]** No RX work on E0.

**Both ERISCs**: honour immediate termination and launch-message exit; E0 handles router state commands (RUN, PAUSE,
DRAIN, RETRAIN) and context switches to base firmware, E1 follows the handshake; heartbeat; teardown (wait for every
trid acked, reset NoC counters, notify subordinate routers, sync the ERISCs, report `TERMINATED`).

---

## 3. Structure, per ERISC

### 3.1 `kernel_main` init (l.2885–3594)

| # | Step | E0 | E1 | Notes |
| ---: | --- | :-: | :-: | --- |
| 1 | Router state = INITIALIZING; data cache config; zero the telemetry block | ✓ | ✓ | |
| 2 | TXQ0 accept-ahead = 32; keepalive override on TXQ0/TXQ1 (restored at teardown) | ✓ | accept-ahead only | Keepalive only on the teardown master (E0) |
| 3 | TXQ1 packet mode, accept-ahead; zero the local receiver ack/completion counter arrays | | ✓ | `initialize_state_for_txq1_active_mode` |
| 4 | Zero the to-sender remote ack/completion counters | ✓ | | `..._sender_side` |
| 5 | Initialise stream registers: packets-received, free-slots ch0/ch1 = num_buffers, retrain sync | ✓ | ✓ | Both ERISCs write the same registers |
| 6 | Sync E0/E1 (stream scratch 31) | ✓ | ✓ | 6 syncs in init, 3 in teardown |
| 7 | 2D: initialise the remaining stream registers (free-slots ch2..) | ✓ | ✓ | |
| 8 | Read runtime args: connection semaphores, producer-cursor addresses, downstream EDM info, teardown semaphores, relay | ✓ | ✓ | |
| 9 | Zero each serviced channel's connection semaphore and producer cursor; `edm_read_counter = 0` | ✓ | ✓ | |
| 10 | Build channel objects: slot tables, worker interfaces, downstream adapters (`EdmToEdmSender`, per-slot address table) | ✓ | ✓ | |
| 11 | Init sender channels (zero headers), worker interfaces (`-Os`, noinline) | ✓ | | |
| 12 | Trid tracker init | ✓ | ✓ | |
| 13 | Ethernet handshake with the peer (sender or receiver side); host-signal / local-handshake master protocol; wait READY_FOR_TRAFFIC | ✓ | | E0 only (`ENABLE_ETHERNET_HANDSHAKE`) |
| 14 | Open downstream router connections (NoC writes into the downstream's connection info + open semaphore) | | ✓ | Only for receiver-serviced ERISCs |
| 15 | `setup_edm_noc_cmd_buf` per downstream: pre-program the data cmd buf coordinate and the sync cmd buf address/value; then downstream 0 again for E/W single-live | | ✓ | Makes the stateful forward valid. One data and one sync cmd buf are shared by all downstreams, so stateful works only with one live downstream |
| 16 | `wait_for_static_connection_to_ready`: spin on each wait-static channel's connection semaphore (context switch every 100k polls on E0), then cache the upstream credit NoC address | ✓ | | Hot E0: ch3 only |
| 17 | Sync, then the main loop | ✓ | ✓ | |

### 3.2 Outer loop (l.2552–2575 and the `execute_main_loop` lambda, l.2140–2550)

E0: `while (!terminated) switch (command)`. RUN runs `execute_main_loop`. PAUSE does the coordinated context-switch
start, `execute_pause_command` (noinline), then finish. E1: `while (!terminated) execute_main_loop()`.

Inside `execute_main_loop`, per outer iteration (every 32 passes):

| Action | ERISC | Touches | Class | Notes |
| --- | --- | --- | --- | --- |
| `continue_running_main_run_loop`: immediate-termination word, launch-message read pointer, then dependent launch exit flag | both | 3 L1 loads, 2 dependent | essential (cadence is a choice) | §17: ~25 instructions per 32 passes on E0 |
| Router command pending (`state_manager.command`) | E0 | L1 load | essential | E1 skips it (not the retrain master) |
| `did_something = false` | E0 | local-memory byte store | overhead | Set per send (1 `sb`), read below. GCC removed it entirely from E1 |
| 32 inner passes | both | | | |
| Drain a pending deferred atomic | E1 | register test | essential when set | |
| Heartbeat: halfword counter +1 (stack); every 64th iteration write `0xDCBA0000 \| n` to `0x7CC70` | both | stack load/store, L1 store | overhead | |
| Idle context switch: on the 34th consecutive outer iteration with no send → `run_coordinated_context_switch_to_base_firmware` (stream-scratch handshake with E1, `run_routing_without_noc_sync`) | E0 | stack counter, stream scratch | essential (base firmware needs the core) | Never fires under load |
| Non-master retrain check: read stream scratch 30; on RETRAIN_INTENT run the ack handshake | E1 | stream register read | essential | Per outer iteration |

Cost: back-edge plus outer loop ~2–3 cycles per pass on E0 (§73).

### 3.3 Inner loop: pass order by configuration (l.2187–2479)

| Configuration | Order within one pass |
| --- | --- |
| **2-ERISC generic, E0** (hot) | grouped read of VC0 free-slot registers 0–3 → ch0 worker step → ch1 → ch2 → ch3 → (ch4 if 5-wide VC0) → (VC1 senders) |
| **2-ERISC generic, E1** | receiver ch0 step → (VC1 receiver step). The sender steps are `if constexpr` empty; E1 does no grouped read |
| **Single ERISC** | grouped read → sender 0 → receiver 0 → senders 1–3 → (VC1 senders, receiver 1) on one core |
| **Speedy** | sender 0 speedy step (E0); receiver 0 speedy step (E1). No grouped read, no forwarding channels |
| **VC2** | adds a VC2 speedy sender and receiver step per pass, and a VC2 connection check per outer iteration |

At most one packet per channel per pass on both ERISCs.

### 3.4 E0 per pass (hot E/W config, s79m disassembly `0x8d88–0x910c`, 226 instructions)

Costs are §73 (s71, 1-hop E/W 4352 B; s79m differs only by the mask reload and a per-pass store, §79).
Idle pass = no packet anywhere and nothing in flight: 34 instructions (from the disassembly).

| Action (in order) | Touches | How often | CT / RT decisions | Measured | Class + reason |
| --- | --- | --- | --- | --- | --- |
| **Grouped read**: 4 × `lui; lw` of `STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE` for ch0–3, fenced back to back | 4 stream registers | every pass | Stream IDs and addresses compile-time (rematerialised `lui`) | loop top 9.1 / packet, 8.0 / pass; 4 grouped reads ≈ 4 when used later (§17) | **mixed**: ch0's read is essential. ch1/ch2 have no upstream router in this config, so 2 of 4 reads can never see a packet |
| **ch0 packet + credit test**: `andi 0x7FF`, `== 17`?, far free slots `!= 0` | registers | every pass | 17 and 0x7FF immediates | 17.6 / packet (~15.7 / pass), mostly waiting for the grouped reads (§73) | essential detection; the wait is overhead |
| ch0, no packet: completion check if `sent != processed` (stack loads of both counts) | 2 stack loads; L1 counter if pending | passes without a ch0 packet | | in "completion test" 5.7 / packet | overhead (bookkeeping) |
| ch0 send path (§3.5) | | on a packet | | 29.0 + 50.2 | |
| ch0 completion test: `beqz` on the delta read in the send | register | every sending pass | | 5.7 / packet | overhead |
| ch0 credit return (§3.6) | | 0.22 / packet | | 7.6 / packet (34.2 each) | essential |
| **ch0 connection check**: `beqz established`; `lw` semaphore pointer (stack) → `lw` semaphore (L1) → compare with close-request | stack + dependent L1 load, used at once | every pass | | 14.8 / packet (~13 / pass) | essential **[rule]**; the latency chain is overhead |
| **ch1–ch3 idle tests**: per channel, the grouped bit (16 slots → bit 4) all-free and `sent == processed` | registers; ch3's sent count is a stack load and its test path is out of line | every pass | bit-test form chosen at compile time | 18.5 / packet (16.0 / pass, ~5 per channel), 11 instructions, 4 taken branches | **overhead**: 2 of the 3 channels can never carry traffic in this config |
| back-edge `addi; bnez` | register | every pass | 32 iterations compile-time | ~2–3 / pass incl. outer | overhead |

### 3.5 E0 per packet: the send (`sender_channel_try_send`, l.1081–1218)

Same code for every sender channel. Shown for ch0 in the hot config (`0x8dc0–0x8e68`).

| Step (in order) | Touches | CT / RT | Measured | Class + reason |
| --- | --- | --- | --- | --- |
| **Can-send** (step entry): far free slots > 0 (≥2 if bubble flow control) and unsent packet | registers | bubble flow control compile-time per channel | in the 17.6 test | essential |
| **Pre-send group** (fenced): slot address ← stack; payload size ← L1 header +40 (depends on the slot address); TXQ0 `CMD` (dummy, BH-55) + `STATUS` reads | stack, L1, 2 eth registers | | **pre-send 29.0 / packet**, 9–10 instructions (§73) | essential: size and busy check. The stack → L1 chain used at once is overhead: the slot address is router state kept on the stack |
| `did_something = 1` | local-memory byte store | | in pre-send | overhead (context-switch heuristic) |
| Stamp `src_ch_id` (byte store, header +43) | L1 store | skipped in mux mode | in pre-send | essential (counter credits) |
| TXQ busy → **return false** (yield; completions still processed) | | | never busy in practice (§11, §72) | essential |
| **Data command**: `TRANSFER_START_ADDR`, `DEST_ADDR`, `TRANSFER_SIZE_BYTES` (size + 96), `CMD = START_DATA` | 4 eth register writes | | **send 50.2 / packet**, 34–35 instructions (§72, §73; §20 breakdown) | essential |
| Advance the far slot pointer: compare with far last (stack), add stride (stack); base (stack) on wrap | 2–3 stack loads | All 3 values compile-time-known, held on the stack | in send | essential state; the stack loads are overhead |
| Advance the channel slot cursor by stride (bit 12): last, stride, cursor on the stack; first on wrap | 3 stack loads + 1 store | first/last/stride compile-time-known | in send | essential state; loads are overhead |
| **Overlap group** (bit 23, fenced): TXQ `CMD` + `STATUS`, then the channel's L1 completion counter | 2 eth registers + 1 L1 load | needs counter credits, BH, no spin-wait, no data cache | in send | essential (busy check and credit poll), overlapped |
| Far free slots −1; channel free-slot stream register +1 (`sw 64 → 0xffb40438`) | register; stream register write | stream ID compile-time (s77s) | in send | essential |
| TXQ still busy → out-of-line spin | eth registers | | spin 0 everywhere (§72) | essential |
| **Pointer update**: `DEST_ADDR` = far packets-received update register (constant reloaded from the stack), `REMOTE_REG_DATA = 1<<6`, `CMD = START_REG` | 3 eth register writes, 1 stack load | | in send | essential (the far E1's doorbell) |
| `packets_sent++` (lean completions, bit 7) | stack load/add/store | | in send | overhead (bookkeeping that lets idle visits skip the L1 read) |
| Completion delta = counter − processed (processed on the stack) | stack load | | in send | essential |

### 3.6 E0 completion processing and upstream credit (`sender_channel_process_completions`, l.1227–1266)

| Step | Touches | How often | Measured | Class |
| --- | --- | --- | --- | --- |
| No send this visit: read the L1 completion counter only if `sent != processed` | 2 counts (registers or stack), L1 load | idle visits with packets in flight | in visits | overhead (polling), but it is how credit comes back |
| Delta ≠ 0: far free slots += delta; processed += delta | registers / stack | per completion frame (0.22–0.44 / packet, §72) | in credit returns | essential |
| **Worker credit** (`increment_local_read_counter` + `notify_worker_of_read_counter_update`): read counter (stack) += delta; store it into the L1 source word; poll NoC0 cmd-buf ready; 6–7 cmd-buf register writes (address from 3 stack slots, coordinate masked by a stack constant); posted-write counter +1 (local memory) | stack ×4, L1 store, NoC register read + writes, local-memory RMW | per completion frame on ch0 | **7.6 / packet, 34.2 each** (§73) | essential (the worker waits for it) |
| Worker not connected: copy the read counter into `worker_location_info.edm_read_counter` | L1 store | rare | | essential (the next producer reads it) |
| **Router credit** (`notify_persistent_connection_of_free_space`): pack +N; poll NoC0 cmd buf 3; write `AT_DATA`, `CTRL`, address and coordinate (2 stack loads of the cached upstream address, 2 stack constants), byte-enable, `CMD_CTRL`; posted counter +1 | stack ×4, NoC registers, local-memory RMW | per completion frame on ch1–3 | 8.4–15.5 / packet on forwarding routers (§72) | essential |
| With first-level ack: completions only free far slots; acks (`has_acks_pending` → L1 ack counter) drive the upstream credit | L1 | per ack | ring: credit returns 27.9 / packet at 1.7 / packet (§76) | essential |

Note: the worker credit write does not wait for the previous credit write to read its L1 source
(`enable_read_counter_update_noc_flush = false`). It is safe because the value is an absolute counter, so a newer
value only arrives earlier (inferred).

### 3.7 E0 connection handling (`check_worker_connections`, inlined)

| Event | Work | How often |
| --- | --- | --- |
| Poll (every pass, ch0 only) | `!established \|\| semaphore == CLOSE_REQUEST`: stack pointer → L1 load | every pass, ~13 (§73) |
| Open seen | Read `worker_location_info` (xy, semaphore address) → cache the NoC address; push the read counter to the worker (NoC inline write, flushed variant) | per connection |
| Close seen | Semaphore = UNUSED; copy the read counter to location info; posted NoC semaphore inc on the worker's teardown address | per connection |
| Forwarding channels | None: live-check skipped; credit address cached once at init | — |

On routers where no worker ever connects, the poll still runs every pass. On the unidirectional multicast forwarder
the connection *body* ran 1.5 times per packet (§76, not investigated).

### 3.8 E1 per pass and per packet (`run_receiver_channel_step_impl`, l.1476–1895; middle E1 disassembly)

Costs: §55 (1-hop E/W segments, before §56/§57's header-load changes), §72 (s71 timed phases), §80 (forward).

| Step (in order) | Touches | How often | CT / RT | Measured | Class + reason |
| --- | --- | --- | --- | --- | --- |
| **Poll**: packets-received stream register (stream 4), used at once | stream register | every pass | | **9.2** (§55) | essential |
| No packet: completion decision (below), back-edge | registers | idle pass, ~10 instructions | | idle pass ~15–17 (inferred) | overhead |
| Slot address: stack table indexed by the read index | stack load (depends on the index register) | per packet | Address = base + idx × 4448 is compile-time-computable; held as a 16-entry table | in decode | overhead (table, not arithmetic) |
| **Header word 40** (payload size, send type, `src_ch_id`), one 32-bit L1 load; route byte(s) from `route_buffer` | 2 L1 loads (E/W); 3 (N/S: Y then X) | per packet | E/W vs N/S compile-time | **decode/admit 17.1–20.1 E/W, 20.1–27.4 N/S** (§55, §72) | essential |
| Store `src_ch_id` into the per-slot array | stack byte store | per packet | skipped in mux mode | in decode | essential (completion routing) |
| **Admission**: local-only → true at compile time. Forward → read the downstream free-slot register(s), used at once (E/W: 1, `lui; lw; slli; beqz`; N/S: all live, issued together) | stream register(s) | per forwarded packet | live mask compile-time | +10 on forward vs local (§80) | essential check. The read latency is overhead: the count is only refilled in batches (§80 follow-up) |
| Trid = slot index (16 trids, bit 22) | register | per packet | compile-time mapping | — | essential (free) |
| Pending deferred atomic? (`bnez`) | register | per packet | compiled only where deferral is allowed | §63: +2 to +3 on every E1-bound row when deferral went in; attributed to code growth and layout, not proven | overhead for non-flush traffic |
| **Forward** (§3.10) and/or **local delivery** (§3.9) | | per packet | | forward 60.0, local write 46.1 (§72) | essential |
| **Retire**: `wr_sent` counter +1, index +1 mod 16, packets-received −1 (stream update write) | registers, stream register write | per packet | | post-local **6.0** (§55) | essential |
| **Completion decision** (bit 22): uncompleted = sent − completed. Progress: check only if > 4. Idle: check if ≥4, or 1–3 after 4 consecutive idle polls (2D hold); 1D: any on idle | registers (`rx_idle_steps` in a register) | every pass | batch target 4, hold 4: compile-time | **~11 / packet** (§55) | overhead (bookkeeping), needed to batch |
| **Completion send** (per batch of ≤4, §3.11) | | 0.25–0.27 / packet | | **15.4 / packet, ~57–59 per batch** (§55, §72) | essential |
| Back-edge | | | | ~2 (§55) | overhead |

Totals: E1 ~124 / packet on 1-hop E/W (§54 slack test, s79-era hold not included); ~115–120 on s71 (§72). E1 runs
about 83 instructions per local-write packet including the amortized completion, so CPI is ~1.5 (inferred from the
disassembly).

### 3.9 E1 local delivery by send type (`execute_chip_unicast_to_local_chip_impl`)

Dispatch: a sparse-multicast check (type 8), then a jump table. The table base and target base are reloaded from the
stack every packet. All writes go on NoC1 cmd buf 0 with the slot's trid; atomics use cmd buf 3.

| Type | NoC work | Measured (E1 local, per packet) |
| --- | --- | --- |
| Unicast write | 1 stateless write: ready poll, `PACKET_TAG`, `CTRL`, `TARG_LO`, `RET_LO`, `RET_COORD`, `AT_LEN_BE`, `CMD_CTRL` + 2 local-memory counter RMWs (issued, acked) | **46–51** (§55, §72; 1D 47.1, §76) |
| Atomic inc (no flush) | 1 atomic | **64.2** (§72) |
| Fused write + atomic (no flush) | write + atomic, 3 more header fields | **89.1** (§72) |
| Scatter (4 chunks) | up to 4 writes, or 2 writes + atomic | **100–110** (§72; 1D 100) |
| Inline write | `noc_inline_dw_write` (cmd buf 3) | not measured |
| Sparse multicast | one write per page of this chip | not measured |
| Multicast write / multicast atomic | `ASSERT(false)`: unsupported on this path | — |

### 3.10 E1 forward (2D E/W single-live, stateful; middle E1 `0xdca4–0xdd68`)

| Step | Touches | Class |
| --- | --- | --- |
| Downstream free-slot register read, used at once (admission) | stream register | essential check; overhead latency |
| Payload size again (`lhu` +40; word 40 was already loaded before admission) | L1 | overhead (duplicate load) |
| Downstream slot index, data cmd-buf id, slot address (16-entry table): 3 stack loads, the address depends on the index | stack | overhead (adapter state on the stack, §80) |
| Ready poll; `PACKET_TAG` (trid), `TARG_LO`, `RET_LO`, `AT_LEN_BE`, `CMD_CTRL` (coordinate pre-programmed) | NoC1 cmd buf 2 | essential (data) |
| Slot index +1 mod 16 (stack RMW); posted counter +1 | stack, local memory | essential state / bookkeeping |
| Sync cmd-buf id (stack); ready poll; `CMD_CTRL` (stateful −1 into the downstream free-slot register); posted counter +1 | NoC1 cmd buf 1 | essential (the downstream E0's doorbell) |
| Local copy of downstream free slots −1 (pointer from the stack) | stream register write | essential |

44 instructions and 2 NoC commands against 26 and 1 for a local write. **Forward 60.0 vs local 46.1** (§72); 66.0
in 1D, 75.0 on the ring (§76). N/S and multi-live forwards use the stateless API (coordinate written per copy). One
`forward_payload_to_downstream_edm` per direction set in the action.

### 3.11 E1 completion batch and credit frame (bit 22, middle E1 `0xd594–0xd6b0`, 72 instructions per batch of 4)

1. Check that the trid of the batch's last slot has left the NIU: NoC1 `NIU_MST_WRITE_REQS_OUTGOING_ID` status read.
   On idle batches with batch > 1, also the first slot.
2. TXQ1 busy check: `CMD` + `STATUS`.
3. Per slot (×batch): load `src_ch_id` (stack), load that channel's completion counter (stack), +1, store to both the
   stack copy and the L1 array.
4. One Ethernet write of the whole counter block: `TRANSFER_START_ADDR` (L1 `0x15be0`), `DEST_ADDR` (`0x15b80`),
   `SIZE = 96`, `CMD`. The block carries the unused ack array too.
5. Completed counter and index += batch; `rx_idle_steps = 0`.

Cost ~57–59 per batch, ~15 per packet (§55, §72). Caveat (§26, still open): while making progress only the last
slot's trid is checked. Local writes (cmd buf 0) and forwards (cmd buf 2) can mix in one batch.

### 3.12 Flushed atomics: deferral (2D, 2-ERISC, bit 22; l.1664–1720, l.1772, l.2480; transmission l.181–215)

| Step | When | Cost |
| --- | --- | --- |
| Atomic with flush: stage it (`noc_semaphore_inc_set_state`); if every earlier non-posted write is already acked (global check), fire now | per flushed atomic | |
| Otherwise record `pending = 1` (STRICT) or `0x10 \| last write trid` (RELAXED) and continue | | |
| Next admitted packet: STRICT → drain (spin until acked, fire). RELAXED → a plain unicast write may bypass while its trid is outside s, s−1; the first bypass does the two-slot exact check (global vs trid_s + trid_p, repeated until equal), later bypasses check the remaining trid(s) (s74) | per packet while pending | NoC status read ~15 each (§71) |
| Idle or blocked pass, and end of the inner loop: drain | | |

Measured: flushed fused every packet 440 (§63), 441.2 (§74) against 155.2 unflushed: ~286 per packet of E1 waiting
for the 4352 B write's ack. Chunked flushed signals (s74): 24–87 per signal (§74). 1D, VC1 and single-ERISC builds
take the old path, which spins on the flush at the atomic (§76).

### 3.13 Paths not in the hot build

| Path | What differs |
| --- | --- |
| **Single ERISC** (`NUM_ACTIVE_ERISCS=1`) | One core runs §3.4–§3.11 in one loop. One TXQ, so **stream-register credits**: no lean completions (`has_completions_pending()` is always true, so every visit reads a stream register), no overlapped send tail (stock tail: spin on busy, then the pointer update), one Ethernet register write per completion credit. NoC1 forced for everything (`FORCE_ALL_PATHS_TO_USE_SAME_NOC`), so no stateful forwarding. No deferral (`!any_sender_channel_serviced`). E0 has ~1 KB of stack free today (§18) |
| **First-level ack / bubble** (ring) | E1 sends an ack frame per packet (TXQ1 busy check, packets-received decremented at ack time), and a completion frame per packet (no batching). 8 trids with a tracker table. E0's worker is an injection channel (needs ≥2 far slots). No idle skip on the forwarding channel |
| **Speedy** | E0: runtime-id stream read, TXQ can-send check, inline send (two busy reads, one of them dead; write-counter update kept), slot advance by table, completions every 2 sends, worker credit every ≥2, connection poll every pass. E1: local delivery on a ping-pong trid; every 4 packets flip trids and send one completion frame **with no TXQ busy check** (`CHECK_BUSY=false`, no can-send test; inherited from main). §76: send 61.4, visits 130.1, at the wire |
| **VC1** | 3–4 ungrouped visits per pass on E0, each a stream read used at once plus completion bookkeeping; v1 measured ~27 per empty visit (§8), v2 not measured. Second receiver step on E1. `-Os` |
| **Stream-register credits** | §3.6 rows read and decrement stream registers; E1 sends one remote register write per credit |

### 3.14 Teardown (`teardown`, `-Os`)

Sync E0/E1 → receivers wait until every trid is acked → `ncrisc_noc_counters_init` → sync → notify subordinate
routers (local handshake master) → write and atomic barriers → clear packet tags → sync → restore the keepalive → E0
writes `TERMINATED`.

---

## 4. E0 inner-loop live state (hot E/W config)

**Method.** Backward liveness over the s79m disassembly of `kernel_main` (all paths from the loop header `0x8d88`,
out-of-line blocks included, exit `0x9110`), plus every stack slot the loop touches. The loop region is 500
instructions: 226 contiguous (`0x8d88–0x910c`) and 274 out of line. The contiguous part has 33 stack loads and 6 stack
stores (matches §79).

**Count.** 19 values sit in registers across the back-edge (10 invariant, 9 loop-carried). The other 8 allocatable
registers serve as per-pass temporaries: the 4 grouped values, then send scratch. **All 27 are in use.** Another 39
distinct values live in stack slots and are reloaded inside the loop. In total ~58 held values, against a 27-register
budget.

| # | Value | Scope | Needed | Compile-time known? | s79m location |
| ---: | --- | --- | --- | --- | --- |
| 1 | Loop counter (32) | shared | every pass | count yes | `t5` |
| 2 | Far receiver free slots (`num_free_slots`) | shared by all 4 channels (one far ring) | every pass (ch0 test) | no | `s4` |
| 3 | Far slot pointer | shared | per send | no | `s6` |
| 4–6 | Far ring base / last / stride | shared | per send (base on wrap only) | **yes** | `20(sp)` / `24(sp)` / `12(sp)` |
| 7 | TXQ0 register base `0xffb90000` | shared | per send | yes | `a5` |
| 8 | did_something address `0xffb01000` | shared | per send | yes | `t6` |
| 9 | L1 completion-counter base `0x16000` | shared | per send; idle visits with packets in flight | yes | `t4` |
| 10 | NoC0 cmd-buf base `0xffb20000` (worker credit) | shared | per worker credit | yes | `a4` |
| 11 | NoC0 cmd buf 3 base `0xffb22000` (router credit) | shared | per router credit | yes | `s0` |
| 12 | NoC SW counter base `0xffb00000` | shared | per credit write | yes | `a0` |
| 13 | `&noc_nonposted_atomics_acked` | shared | teardown of a worker connection | yes | `s8` |
| 14–16 | Constants 1, 2, 4 (`did_something`, `CMD_CTRL`; `START_DATA` / close request; `START_REG` / length 4) | shared | per send | yes | `a2`, `a7`, `s5` |
| 17 | Far packets-received update address `0xffb44438` | shared | per send | yes | `40(sp)` (hoisted, spilled) |
| 18–21 | NoC masks / encodings `0xffffff`, `0x1000000f`; `0x32e0`, `0x107c` (connection paths) | shared | per credit; rare | yes | `36`, `56`, `60`, `64(sp)` |
| 22 | ch0 slot cursor | ch0 | per send | no | `1212(sp)` |
| 23–25 | ch0 first / last / stride | ch0 | per send (first on wrap) | **yes** | `1216`, `1220`, `1224(sp)` |
| 26–27 | ch0 sent / processed | ch0 | per send; idle passes | no | `32(sp)` / `28(sp)` |
| 28 | ch0 read counter (worker credit) | ch0 | per credit | no | `540(sp)` |
| 29–30 | ch0 worker credit NoC address lo / hi | ch0 | per credit | no (per connection) | `512`, `516(sp)` |
| 31 | ch0 credit source word address | ch0 | per credit | yes (CT arg) | `528(sp)` |
| 32 | ch0 connection-semaphore pointer | ch0 | **every pass** | no (runtime arg, fixed) | `520(sp)` |
| 33 | ch0 `worker_location_info` pointer | ch0 | connection events | yes (CT arg) | `504(sp)` |
| 34 | ch0 connection established | ch0 | **every pass** | no | `s10` |
| 35–37 | ch1 sent / processed; ch1 cursor | ch1 | sent/processed **every pass**; cursor per send | no | `s3` / `s11`; `1124(sp)` |
| 38–40 | ch1 first / last / stride | ch1 | per send | **yes** | `1128–1136(sp)` |
| 41–42 | ch1 upstream credit address lo / hi | ch1 | per credit | no (fixed at init) | `464`, `468(sp)` |
| 43–50 | ch2: same 8 values | ch2 | as ch1 | as ch1 | `s7` / `s1`; `1040–1052`, `416/420(sp)` |
| 51–58 | ch3: same 8 values | ch3 | as ch1 | as ch1 | sent `16(sp)`, processed `s9`; `956–968`, `368/372(sp)` |
| — | Grouped values ch0–3 | per channel | live from loop top to each visit, **not** across the back-edge | no | `t1`, `a6`, `a1`, `a3` |
| — | Free-slot stream bases, `64`, `0x7FF`, `17`, bit 4 | — | per pass | yes | rematerialised (`lui` / `li`), not held |

**What the counts say**
- **Runtime state is 26 values.** 17 change while running: loop counter, far free slots, far pointer, 4 slot cursors,
  8 sent/processed counts, ch0 read counter, connection flag. 9 are fixed after init or connection: semaphore pointer,
  worker credit address (2), 3 × upstream credit address (2).
- **About 32 held values are compile-time constants.** Hardware bases and small constants (13), hoisted >12-bit
  immediates (5), far ring base/last/stride (3), 4 × slot first/last/stride (12), 2 compile-time addresses. They are
  held because they live in struct fields (`StaticSizedSenderEthChannel`, `OutboundReceiverChannelPointers`) or
  because GCC hoists a large immediate out of the loop. §32, §77, §78 and §81 show that removing them from the
  structs shifts allocation unpredictably.
- **Needed on every pass: 12 values plus 4 temporaries.** Loop counter, far free slots, ch0 sent/processed (no-packet
  passes), connection flag and semaphore pointer, and 6 forwarding counts; plus the 4 grouped values. Four of the 6
  forwarding counts belong to ch1/ch2, which have no upstream router here and stay at 0 forever.
- **Needed only on a packet:** ~17 per sending channel (~29 distinct across the four). **Only on a credit:** ~15.
  **Only on connection events:** ~4.

---

## 5. Per-packet work budgets (4352 B unless noted)

"Binds" comes from the +20 spin slack test: the ERISC whose interval moves by ~+27–31 (§72, §76). Items are per packet.
**E** = essential, **O** = overhead, **M** = mixed: essential work whose cost is mostly latency or bookkeeping that
could be known earlier.

| Case | Interval | Binds | Binding ERISC's items | E / M / O (per packet) |
| --- | ---: | --- | --- | --- |
| **1-hop E/W write, 2D** | 132.9 (§72); 132.2 s79m (§79); 16 B 121.7 | **E0** (E0 +27.0 / E1 +13.1) | E0 (§73, profiled, sum 155 against 133.7 unprofiled): send 50.2 E; credit returns 7.6 E; connection poll 14.8 E **[rule]**; pre-send 29.0 M (size + busy check behind a stack → L1 chain); packet test 17.6 M (waits on the grouped reads); loop top 9.1 M (2 of 4 reads hit producer-less channels); idle ch1–3 18.5 O; completion test 5.7 O; back-edge ~3 O | **E ~73 / M ~56 / O ~27** profiled; scaled to 133: ~63 / ~48 / ~23. E1 not binding: ~115–120, slack ~13 |
| **1-hop N/S write, 2D** | 133.3 (§72) | **both** (+26.6 / +19.2) | E0 as above (within 0.3, §73). E1: poll 9.2 E; decode/admit 27.4 E (Y then X); local write 50.1 E; retire 6 E; completion 15.3 E; completion decision ~11 O; post-admit header handling ~13 M (§55, partly removed in §56/§57); loop ~2 O | E1: **E ~108 / M ~13 / O ~13** (inferred mix of §55 and §72) |
| **Forwarded write through a router** (E1 forward + next E0 forwarding send) | FWD row 125.7, at the wire (§72); 16 B 111.9 (§79) | wire (4352 B). 16 B not established | Router k's E1: decode/admit 30.3 (incl. ~10 for the downstream register read used at once, §80) + forward 60.0 + completion 15.6 + untimed ~35. Router k+1's E0, forwarding channel: send 47–49 + credit returns 8–16 (0.23–0.44 frames) + visits (~62 rest of visits on the FWD row, §72) | E1 (inferred split): **E ~100 / M ~25 / O ~15**; the forward's +14 over a local write is two NoC commands plus stack-held adapter state (§80) |
| **3-hop middle router, E/W** | 146.9–148.8 (§72, §79) | **E1** (~140), with E0 ~135 and the 16-slot forwarding credit loop ~135–140 close behind (§80) | E1 as in the FWD row. Middle E0: idle worker channel 51.9 / packet = 1.48 passes × ~35 (test ~14, connection ~13, completion test ~3) **all O** except the **[rule]** poll; forwarding channels 110.1 incl. send 47.2 and credits 8.4 (§72, §73) | E1: as FWD. E0: ~33 of 51.9 is pure O, ~19 is the [rule] poll |
| **Multicast write, 2D all-to-all** | 140.8 (§72) | **both** (+23.5 / +22.8) | E1 forwarder: decode 30.7, forward 60.1, local 46.3, completion 16.0 (sum 153, profiled). E0: send 47.8, credit returns 15.3 (0.44), connection 5.9, visits 140.2 at 3.04 visits / packet, ~71 rest of visits | E1: timed phases 153, all E (profiled, ~5–10% long), plus ~13 O of untimed bookkeeping (inferred). E0: ~71 visit overhead shared across flows |
| **Flushed fused write + atomic** (E/W, flush every packet) | 441.2 (§74); unflushed 155.2 | **E1** | Unflushed: decode 20.0 E, local delivery 89.1 E (write + atomic), completion 15.7 E. Flushed: +~286 / packet of E1 waiting for the 4352 B write's NoC ack before firing the atomic; nothing overlaps because the next packet is also fused and drains at admission (§63, §75) | ~286 is **protocol-imposed wait** (CPU idle). Chunked signals cost 24–87 per signal (s74, §74) |
| *atomic_inc, 2D* | 131.6 (§72) | both | E1 local 64.2 | |
| *1-hop scatter, 2D* | 165–175 (§72, §74) | E1 | local 100–110 | |
| *1D line 1-hop write* | 137.6 (§76) | **neither: worker credit loop** (16 worker slots ≈ 2,300 / 16) | E0 ~122 (send 47.6, credits 6.2, visits 121.8 at 2.31 / packet, ~68 overhead); E1 ~129 (decode 15.2, local 47.1, completion 15.5) | |
| *1D 3-hop / linear multicast* | 146.8 / 149.2 (§76) | E1 | forwarder decode 28.5–29.0, forward 58–66, completion 15.6 | |
| *1D half ring (ring, first-level ack)* | 229.9 (§76) | both | forward 75; completion every packet 20; E0 credit returns 27.9 at 1.7 / packet | |
| *1D neighbor exchange (speedy)* | 125.0 (§76) | at the wire | E0 send 61.4, visits 130.1 | |

**Primitive costs used above** (§3, §17): stream-register read used at once 7 (9 in a loop); L1 load 8; stack load 2
alone, +6–14 when issued behind outstanding reads; TXQ busy check (2 reads) 8; NoC status or cmd-buf-ready read 7;
4 stream reads together ~4 when used ≥4–6 instructions later, while a group of 5–6 costs ~14; stateless 64 B NoC write
23; inline 4 B NoC write 24; NoC status read inside a polling walk ~15 (§71).

**Credit-loop floors** (slots × round trip of ~2,100–2,300 cycles; §26, §27, §76, §80): 17 worker slots ~124–130,
16 forwarding slots ~135–140, 1D 16 worker slots ~140. Holding completions for a batch lengthens the loop: §28's
16-idle-poll hold cost 5–65.

---

## 6. Observations for a redesign

1. **E0's cost is per pass, and the worker channel dominates it.** A 1-hop packet is ~50 of send plus ~70 of
   visiting (§72). Of the visiting, the worker channel's own checks are ~63 per pass and three idle forwarding
   channels ~16 (§73). An idle pass is 34 instructions. A sending pass is ~80–100 instructions at a CPI of ~1.6.
2. **Per-pass work scales with serviced channels, including channels that can never carry traffic.** Every 2D E0 on
   the 2x4 mesh services all three forwarding channels, though only 1–2 have an upstream router (the
   `WAIT_STATIC_CONNECTION` args know which). Routers with no connected worker still poll ch0's connection every
   pass: ~35 per pass on the 3-hop middle E0 (§73). VC1 adds 3–4 ungrouped visits and a second receiver step per
   pass, unmeasured on v2.
3. **Most of E0's held state is compile-time-known.** ~32 of ~58 held values are constants (slot bounds and strides,
   far ring geometry, hardware bases, >12-bit immediates). The protocol state is ~26 values, and only 12 are needed
   every pass. The register file is full of things the compiler could fold, and isn't asked to.
4. **The hot paths are latency chains, not instruction counts.** E0 pre-send (stack → L1 header size), connection poll
   (stack → L1 semaphore), and every first use of a register read. E1: stream read → slot table → header; downstream
   credit read used at once. §17's overlap rules (≤4 reads in flight, use after 4–6 instructions) govern most of the
   ~45 cycles per pass that E0 spends beyond its instruction count.
5. **One far ring per VC is shared by all sender channels.** One far free-slot count and one far pointer on E0, a
   `src_ch_id` stamp per packet, a per-slot `src_ch_id` array on E1, and per-source absolute counters in every
   credit frame. Sending order across channels fixes far slot order, so channel fairness lives in E0's loop order.
6. **Every slot credit is tracked twice, in two encodings.** Producer to router: −1 on a stream register (worker or
   upstream E1); E0 +1 after sending. Router back to producer: worker gets an absolute read counter in L1 (NoC
   write), a router gets +N on its stream register. E1 keeps the downstream's free slots in its own stream register
   (−1 per forward, +N from the downstream E0). The far ring's credit comes back as absolute L1 counters over
   Ethernet. Credit form is chosen by channel type at compile time.
7. **Throughput floors are credit-loop floors as much as CPU floors.** Slots × ~2,100–2,300 cycles sets 124–140
   cycles/packet for the worker, forwarding and 1D channels (§26, §27, §76, §80). Batching on E1 trades CPU for loop
   length (§28, §57).
8. **E0 and E1 duplicate bookkeeping.** Each runs its own outer loop with termination checks. Each keeps NoC software
   counters in local memory per command: E1 two RMWs per local write and two per forward, E0 one per credit write.
   E1 writes each completion counter twice (stack copy and L1 array), then ships both arrays (ack array unused
   without first-level ack) in every 96 B frame.
9. **Compile time vs run time is split in a few odd places.**
   - Per-packet run-time work on compile-time facts: E1's slot address comes from a stack table instead of
     base + idx × stride, the jump-table base is reloaded from the stack per packet, and the forward path reloads the
     payload size it already has.
   - Compile-time choices on traffic-dependent facts: batch target 4 and the 4-poll hold are fixed, and so is where
     deferral exists (2D, 2-ERISC, one receiver channel only; 1D and VC1 spin on every flushed atomic).
   - E/W single-live forwarding is a sound compile-time fact under dimension-order routing.
10. **The v2 source is no longer a flag matrix.** Only bit 22 still switches code. The other "best set" bits and every
    E1 change since §53 are unconditional, so "v2 with flags off" is a different router from main, not main.

---

## Appendix A: disassembly landmarks (s79m)

| Item | Address / fact |
| --- | --- |
| Hot E0 `kernel_main` | `0x793c–0x96f0`, frame 1296 B; `.text` 10,092 B |
| Hot E0 inner loop | header `0x8d88` (grouped read), back-edge `0x9108` (`addi t5,-1; bnez`), outer loop `0x9110`; out-of-line ch1/ch2 sends `0x9438`, `0x920c`; TXQ spins `0x9188`, `0x91d4`, `0x92ac`, `0x94dc`; connection open/close `0x92d0`, `0x938c` |
| Hot E0 pre-loop spills | `0x8d3c–0x8d68`: hoisted constants stored to `36–64(sp)`; ch1/ch3 counts reloaded from `44/48/52(sp)` per outer iteration |
| Middle E1 `kernel_main` | `0xc910–0xe378`; `.text` 7,636 B |
| Middle E1 inner loop | header `0xd438` (packets-received read), local write `0xd510–0xd578`, retire `0xd57c`, completion batch `0xd594–0xd6b0`, back-edge `0xd6bc`; idle `0xdbec`; forward `0xdca4–0xdd68`; deferral `0xdacc`, `0xe1a0`, `0xe348` |
| Middle E1 live across the back-edge | 18 registers (10 invariant: stream-4 base, NoC1 base, gp, TXQ1 base, trid status base, posted counter, route offset, constants 3, 4, −64; 8 carried: loop counter, pending atomic, idle steps, `wr_sent` counter/index, completion counter/index, plus one path-local). Stack holds the slot table, `src_ch_id` array, per-source counters, jump-table bases and adapter state |
| Tooling | `objdump -d -l`, compacted, with a backward-liveness pass over the CFG. Scripts in the session scratchpad (`map/tools/`) |
