# Router v3e1: a whole-program redesign of E1 (the receiver ERISC)

v3e1 is v3a with ERISC 1's program rewritten. In two-ERISC mode ERISC 1 (`MY_ERISC_ID == 1`) now runs its own program,
`e1_kernel_main`: its part of kernel_main's init, its own downstream router connections, a new receive loop and
teardown. ERISC 0 and every build outside that scope run v3a's code, unchanged: **E0 is instruction-identical to v3a on
every logged config at -O3, -O2 and -Os** (0/208, 0/208, 0/160 + 0/48 VC1), and so are the 200 E1-processor builds the
program does not cover (ring, speedy, single ERISC).

**v3e1b** (after hardware jobs 224–226: local-delivery, VC1 and -Os fixes) is described in §9; **v3e1c / v3e1d**
(scatter, atomic and fused local delivery, on v3l2be1b) in §10; **v3t / v3tb** (a hot loop that cold edits do not
move, and paths less dependent on the opt level, on v3sb) in §11; **v3v / v3vb** (N/S decode at -Os, word-aligned
completion batches, local writes with a preset NOC_CTRL, on v3tk) in §12, and after hardware jobs 287–289 the NoC API
port, the regression analysis, the controls v3vc / v3vd / v3ve and **v3v2** (the route decode without a serialized
load) in §12.8–§12.12.

Variant: `fabric_exp/variants/v3e1/` (same layout as v3a; only `fabric_erisc_router_v3.cpp` differs from v3a, the three
shared headers and the speedy header are v3a's). Scratch (builds, scripts, disassembly):
`/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3e1/`.

## 1. Scope

`e1_receive_program` (constexpr, next to the new code) selects the program. It needs: Blackhole; two ERISCs and
`MY_ERISC_ID == 1`; no sender channel serviced on this ERISC; receiver channel 0 serviced; not the speedy path; separate
sender/receiver TX queues; no deadlock avoidance and no first-level ack (VC0 or VC1); no UDM, no tensix mux, no
context-switch master role; no VC2; one NoC for local and downstream writes equal to `noc_index`; counter credits and
forwarding enabled on every serviced receiver channel; 1D only with low-latency routing fields. VC0-to-VC1 crossover
(intermesh routers) **is** covered.

Logged configs (536 per level; "E1" = the subordinate processor, as codegen_diff counts it):

| Builds | Count | Code |
| --- | ---: | --- |
| E1, 2D VC0 (E/W and N/S) | 40 | v3e1 program |
| E1, 1D line | 40 | v3e1 program |
| E1, 2D VC1 (32 VC1-serviced + 16 crossover) | 48 | v3e1 program |
| E1, 1D ring (first-level ack), 1D speedy, single-ERISC (mux) 1D/2D | 200 | v3a, identical |
| E0, all | 208 | v3a, identical |

`kernel_main` itself is v3a's, with one dispatch at its top (`if constexpr (e1_receive_program) { e1_kernel_main();
return; }`). Earlier attempts that edited kernel_main's body for E1 (an `if constexpr` around the main-loop call, E1-only
locals) left E0 identical at -O3/-O2 but swapped one compare's operands in 14–44 E0 configs at -Os, so ERISC 1's program
lives in its own function and kernel_main's body is untouched.

## 2. Design

**Structure.** `e1_kernel_main` (FORCE_INLINE into kernel_main) runs the receiver-relevant init steps of kernel_main in
the same order with the same six ERISC syncs, then `run_e1_receive_program`, then v3a's `teardown` (three syncs). The
loop:

```
while (!immediate termination):
    32 passes: e1_rx_step<0>, e1_rx_step<1> (VC1-serviced builds)
    drain a deferred flushed atomic (VC0, 2D)
    telemetry (as v3a, tx_progress = 0), heartbeat, non-master context-switch handshake
```

**State.**

| Object | Where | Contents |
| --- | --- | --- |
| `E1Rx<CH>` | compile time | ring base (`channel_allocs` entry), slot count/size, packets-received stream, batch target, trid layout, downstream VC (VC1 for VC0 under crossover), forwarding command buffers, deferral flag |
| `E1Downstreams<VC>` | compile time | downstream count and slots, command buffers, per compact slot the downstream's credit stream and this router's local copy (the values kernel_main's adapters are built with) |
| `E1RxState` (per channel, `// STABILITY_PROBE_STATE`) | loop-carried | `avail` (packets known present), `slot_addr` (running slot pointer), `sent`, `completed`, `idx`, `completion_idx`, `idle_polls`, `pending_atomic` |
| `src_ids[num_slots]` (per channel) | stack | sender channel of each slot (completion routing), as v3a's `src_chan_ids` |
| `E1DownstreamCursor[ds_size]` (per VC) | stack (escapes to the init calls) | next downstream slot, ring bounds, NoC coordinate and credit address (stateless sends), local-copy register addresses, `free_slots` (known free downstream slots) |

The cursor's fields live in memory by construction (its address goes to the out-of-line init and stores to NoC/stream
registers may alias it), so nothing depends on how the compiler allocates them.

**Per pass, per channel** (`e1_rx_step`):

1. `avail == 0` → read the packets-received register (masked with `0x7FF`, see §5); otherwise no read.
2. Packet present: header word 40 (size, type, `src_ch_id`) and the route byte(s), from `slot_addr`; store `src_ch_id`.
3. Decode: v3a's (intermesh landing intercept, `decode_action`, intermesh-exit intercept).
4. Admission: local-only → admitted. Forward → for each LIVE direction the action uses, test `free_slots`; only when a
   needed count is 0, re-read the local copy register(s) (one direction on E/W; on N/S all live ones as one fenced
   group). VC1 builds: one shared out-of-line body walks the used directions.
5. trid as v3a (tracker). Deferred-atomic resolution: v3a's code, verbatim, before any NoC command of the packet.
6. Forward: per used direction, `e1_forward_payload` — data write tagged with the trid (`send_chunk_from_address_with_trid`,
   stateful on single-live E/W and 1D, stateless otherwise), slot pointer advance, doorbell (stateful inline write on the
   sync command buffer, or stateless `noc_inline_dw_write`), local copy −1, `free_slots` −1. Command buffers and the
   local stream are compile-time.
7. Local delivery: `execute_chip_unicast_to_local_chip_impl`, unchanged (shared header).
8. Retire: `avail−1`, packets-received register −1 (after the packet's NoC commands, as v3a), `sent+1`, next index, next
   slot (`idx == 0 ? first : slot + size`).
9. No progress and an atomic pending → drain (v3a). Then completions (§3).

**Completions** (`e1_rx_completions`): v3a's policy (progress: batch of 4 once more than 4 wait; idle: full batch at
once, partial after 4 idle polls in 2D, at once in 1D). The readiness check is new: one fenced group reads the trid-sent
status (`NIU_MST_WRITE_REQS_OUTGOING_ID`) of the four slots from the oldest uncompleted one, then the receiver TX queue's
CMD (BH-55) and STATUS; the batch is credited only when all four statuses and the queue are clear. Then v3a's frame:
accumulate per slot, one Ethernet write of the counter block.

**E1's downstream connections.** kernel_main builds RouterToRouterSender adapters for every downstream, opens them on
the receiver ERISC and pre-programs the forwarding command buffers. On ERISC 1, `e1_init_downstreams` /
`e1_open_downstreams` / `e1_setup_downstream_cmd_bufs` do the same writes in the same places and order: per connected
downstream, zero its teardown semaphore and `init_ptr_val(local copy, slots)` (adapter construction, before the
handshake); `connection::open_start` then `open_finish` with the same arguments (after the handshake);
`noc_async_write_one_packet_with_trid_set_state<true>` + `noc_inline_dw_write_set_state<true, true>` per connected
downstream, then downstream 0 again when stateful (after the next sync). VC1 the same with VC1's args; with crossover
the VC0 set credits the downstream's VC1 channels, as kernel_main's VC0 adapters do. The forward then only uses the
cursors. The bodies are out of line (`noinline`), run once, and serve both VCs.

**VC1 code size.** VC1 builds route both receiver channels through one out-of-line body each for local delivery
(`e1_local_delivery_shared`), admission (`e1_admit_slots_shared`) and forwarding (`e1_forward_slots_shared`), which walk
the used downstream slots at run time. What is still instantiated per channel is the step's decode, retire and completion
bookkeeping (~450 B each at -O3); delivery, admission and forwarding no longer grow with channels or directions. VC0-only
builds keep the compile-time inline forms (their hot path). VC1 forwards were already stateless.

## 3. Per-packet and per-pass work vs v3a

| Item | v3a (ROUTER_MAP §3.8–§3.11) | v3e1 |
| --- | --- | --- |
| Packet poll | packets-received register read every pass, used at once (~9, §55) | read only when the count from the last read is used up; with a backlog (E1-bound rows) most packets skip it |
| Slot address | stack table indexed by the read index (dependent load) | `slot_addr` register/loop value; advance = add, wrap on `idx == 0` |
| Header loads | word 40 + route byte(s); forward re-loads the payload size | same two loads; forward reuses word 40's size |
| Forward admission | downstream local-copy register read, used at once (+10 vs local, §80) | test of the known-free count; register read only when it is 0 (§80, "reload at zero") |
| Forward | adapter state on the stack: slot index → table → address, index wrap/store, both cmd-buf ids, local-credit pointer (6 loads + 1 store, §80) | cursor: slot address, last slot (and first on wrap), `free_slots`; compile-time command buffers and local stream |
| NoC commands per forward / local write | 2 / 1 | 2 / 1 (identical commands) |
| Deferral | v3a | v3a (verbatim), plus no bypass when the packet leaves by the intermesh exit (§4 c) |
| Completion check | trid of the batch's last slot (+ first on idle), then TX queue (sequential) | trid of every slot of the batch + TX queue, one fenced group |
| Completion frame | v3a | v3a |
| Retire | counter +1, index wrap, register −1 | same plus `avail−1` and the slot pointer |

Removed from E1 entirely: the shared main-loop scaffolding E1 never used (`did_something`, speedy copy-in/out, perf
window, grouped sender reads, sender steps compiled to nothing), the adapters' per-downstream slot tables, and the
sender-side init that ERISC 0 performs identically (worker location info `edm_read_counter`, sender channel header
zeroing, the far receiver's channel view, worker interfaces, `wait_for_static_connection_to_ready`, which has nothing to
wait for on ERISC 1). ERISC 1 no longer reads the sender connection runtime args.

Not changed: local delivery (all send types), the deferral and RELAXED/STRICT handling, the batching policy and frame,
the hold, telemetry, heartbeat, context-switch handshake, teardown, `src_ch_id` routing.

## 4. Semantic equivalence

The program issues the same packets, credits, completions and flushes, in an order the other side accepts:

- **Counts kept by E1 are lower bounds of the registers they cache.** `avail`: the packets-received register only grows
  from the far E0 and only E1 decrements it (−1 per retired packet, still). E1 sets `avail` from a read and decrements it
  per retired packet, so `avail ≤ register` always; a packet is used only when it is known to be there (same visibility
  rule as v3a: the far E0 updates the register after the data). `free_slots`: the local copy register only grows from the
  downstream E0's credits and only E1 decrements it (−1 per forward, still); same argument. Admission decisions equal
  "register != 0" at some time during the pass, as before. Telemetry's RX heartbeat and the init/teardown paths still read
  the registers, which keep v3a's values.
- **Forward and connection setup:** the same NoC commands with the same arguments and order as the adapter (data write,
  slot advance, doorbell, local −1; open_start/open_finish; set_state). The downstream slot sequence starts at slot 0
  after open, as the adapter's does.
- **Completions:** credited only when every slot of the batch has no outgoing write (a superset of v3a's condition), so
  never earlier than v3a would, and never later for uniform batches (if writes on one command buffer leave in order, the
  last slot clear implies all clear). The newest slot is still never in a progress batch (deferral invariant, §63).
- **Flushes:** STRICT and RELAXED deferral code is v3a's; drain points unchanged (admission, idle/blocked pass, end of the
  32 passes). Local delivery gets the pending pointer exactly when v3a passes it.

Intended differences from v3a (all on ERISC 1):

| | v3a | v3e1 | Why |
| --- | --- | --- | --- |
| a. Completion check | last slot (+ first on idle) | every slot of the batch | §26 OPEN: a batch can mix local writes (cmd buf 0), forwards (cmd buf 2) and atomics (no slot read: a batch ending in an atomic slot passed v3a's check at once). Exact, and independent of command-buffer ordering |
| b. VC1 receiver admission | reads **VC0's** local free-slot registers (`admit_2d_dispatch`'s compile-time path ignores the channel; study §53 bit 28) | reads VC1's | v3a/v2 bug on VC1-serviced routers: VC1 traffic was admitted on VC0's downstream credits. Main (v1) read the adapter's own (VC1) register; v3e1 does too |
| c. Deferral bypass | allowed for a plain write with `action == LOCAL_DELIVER` | not when that packet leaves by the intermesh exit | the exit forward is stateless; its doorbell uses `write_at_cmd_buf`, which holds the staged atomic. Only reachable on a non-VC1 edge router (deferral is off with VC1); none is logged |
| d. Sender-side init on ERISC 1 | done by both ERISCs | ERISC 0 only | identical writes; ERISC 1 never uses that state |
| e. Unused state | adapters, worker interfaces, sender channels built on ERISC 1 | not built | — |

No host interface change: v3e1 reads the same compile-time args and a subset of the same runtime args at the same
indices.

## 5. Grouped reads and the `0x7FF` mask

Grouped reads use v3a's form (volatile loads between two `std::atomic_signal_fence(seq_cst)`, results combined after
the second fence), in two places: the completion check (4 trid-sent statuses + TX queue CMD + STATUS) and the N/S credit
reload (all live local-copy registers). They are issued unconditionally (no short-circuit) so the loads go back to back. No new primitive: the fence form is
already v3a's (E0's grouped reads, the two-slot check), it only orders the volatile loads among themselves and before
their uses, and it does not depend on how registers are allocated.
The packets-received and free-slot counts are masked with `0x7FF` (an immediate) instead of the 17-bit field mask: the
counts never exceed the slot count, which is static_asserted ≤ 0x7FF (same reasoning as E0's s79m mask).

## 6. Ledger

| Idea | Status in ledger | Here |
| --- | --- | --- |
| Cached downstream credits (§80) | OPEN | used ("reload only at zero", per-forward register decrement kept; no bulk consume, no idle refresh) |
| Per-command-buffer check for mixed batches (§26) | OPEN, correctness | used, generalized to every slot of the batch, grouped with the TX-queue reads |
| Compile-time forwarding cmd buf (§37) | OPEN | used (and compile-time local credit stream) |
| E1 slot address from a stack table, payload size re-load in forward (ROUTER_MAP §6.9, §55/§56) | observation | removed |
| Packets-received count kept across passes | new (unlike §21's prefetch, nothing is read early: a lower bound, no stale data) | used |
| Deferral (§63/§71/§74) | IN | kept verbatim |
| Remove deferral's placement cost (§68) | OPEN | not done: the no-pending path is still a taken jump in the hot E/W E1 (layout, see risks) |
| Keep the decoded action across retries (§13) | OPEN | not done (retries happen only when blocked) |
| Batching on ring / deadlock-avoidance routers (§76) | OPEN | not done: ring E1 keeps v3a's code |
| Deferral and hold in 1D (§76) | OPEN | not done (1D program keeps v3a's 1D policy) |

## 7. Builds

All builds overlay the Phase 1 snapshot (`offline_build.py`, wrapper `ob.py` in the scratch dir that tolerates the
incomplete `-Os` log entries). `-Os`: the 440 VC0 configs from the `-Os` logs plus the 96 VC1 configs from their `-O2`
commands rebuilt with `--set-opt Os` (the logged `-Os` VC1 command lines are interleaved MPI output). Script:
`build_all.sh` in the scratch dir.

| Build | Result |
| --- | --- |
| v3e1 at -O3 / -O2 | 536/536 each (-Werror) |
| v3e1 at -Os | 440/440 VC0 + 96/96 VC1 |
| v3a baselines, same procedure | 536 / 536 / 440 + 96 |

codegen_diff against v3a (changed / total; instruction and largest-frame deltas over the changed configs):

| Level | E0 | E1 | E1 instructions med/max | E1 frame med/max | Hot E1 mid-W (`10856749301170935304`) |
| --- | --- | --- | --- | --- | --- |
| -O3 | **0/208** | 128/328 | −536 / −1055 | −264 / −368 | 1926 → 1818 instructions, frame 720 → 368 |
| -O2 | **0/208** | 128/328 | −438 / −1008 | −280 / −416 | 1839 → 1737, frame 720 → 368 |
| -Os VC0 | **0/160** | 80/280 | −151 / −513 | −288 / −368 | 1677 → 1524, frame 704 → 336 |
| -Os VC1 | **0/48** | 48/48 | −968 / −999 | −400 / −416 | |

The 128 changed E1s are exactly the program's configs (40 2D VC0, 40 1D line, 48 VC1); the other 200 E1-processor
builds are identical to v3a at every level. `.text` of some E1s at -O3 (v3a → v3e1): mid-W E/W 7,704 → 7,328; N/S
7,804 → 7,556; 1D line 8,620 → 5,992; a VC1 E1 11,136 → 8,288.

**VC1 code size** (largest E0 + largest E1 loadable bytes over the 96 VC1 kernels; E0 is v3a's):

| Level | E0 | E1 v3a | sum v3a | E1 v3e1 | sum v3e1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -O3 | 16,996 | 12,708 | 29,704 (over by 3,080) | 9,064 | **26,060 (fits, 564 spare)** |
| -O2 | 13,920 | 12,232 | 26,152 | 8,408 | 22,328 |
| -Os | 12,344 | 11,660 | 24,004 | 7,816 | 20,160 |

By router type at -O3: VC1-serviced routers E0 16,996 + E1 8,868 = 25,864; intermesh (crossover) routers E0 14,648 +
E1 9,064 = 23,712. So VC1 routers fit the 26,624 B kernel config buffer at -O3 with v3a's E0. Where E1's VC1 bytes went
(-O3, largest E1): local delivery, admission and forwarding shared by both channels (−1.9 KB, one local-delivery body
instead of two); the adapters' construction, open and command-buffer setup lambdas replaced by three run-once bodies
(−1.5 KB); sender-side init dropped. Marking the shared bodies `noclone` as well (one copy per body guaranteed) would
take E1 to 8,800 B; v3e1 does not, since the brief rules out codegen-steering attributes and `noinline` alone fits.

**Probes** (perturb.py on v3e1 and on v3a, -O3). Markers: one DECL (v3a's), two PRELOOP/LOOP pairs (v3a's shared loop,
E1's loop), STATE in v3a's sender channel struct and in `E1RxState`.

| Probe | E0 | E1 | Note |
| --- | --- | --- | --- |
| hook | identical to v3a + hook (0/208 differ) | 328/328 change; median +7 instructions (v3a: +16), max +72 (v3a +72); hot E1: 1,649 of 1,818 differ (v3a: 1,806 of 1,926) | a call in E1's loop, as in v3a |
| statefield | identical to v3a + statefield | the 128 program E1s do not change (the never-read member of `E1RxState` disappears); the 120 single-ERISC E1s change as in v3a | |

**Disassembly check** (hot mid-W E1, -O3): the loop head tests `avail` in a register and reads the packets-received
register only at 0; word 40 and the route byte load from `slot_addr` held in a register; `src_ch_id` is one byte store;
the E/W forward uses NoC1 command buffers 2/1 and the local credit stream as constants, the cursor's slot/last/free
fields from the stack, no index → table chain; retire is the register update, a store to the stream register and the
slot advance (the 4,448 B stride is a spilled constant); the completion check is four address computations, six loads
back to back, then the ORs. The no-pending-atomic path leaves the loop body by a taken branch to the local-delivery
dispatch and comes back by a jump (the §68 shape; not tuned, no hints).

## 8. Risks and what to measure

- **Untested init on hardware.** ERISC 1 opens and programs its downstream connections itself. It is written against
  the adapter code (same calls, arguments, order and sync points), but a mismatch would hang init (downstream ERISC 0s
  wait for their static connections). Validation (stream, mesh, 1D sanity) and the VC1 multi-mesh sanity cover every
  path used here, including crossover (intermesh) routers.
- **Completion check cost.** Six reads per batch in one group (four trid statuses + TX queue) against v3a's one trid
  read and then the TX queue. By §17 a group of 5–6 costs ~14, about what v3a's two dependent reads cost, so roughly
  neutral per batch of 4; it is the price of exactness for mixed batches. If it shows, the alternative is to check the
  last slot plus the newest slot of the other command buffer, tracked per slot.
- **Where the gains should show.** Forwarding E1s (3-hop, multicast, FWD rows, 1D multi-hop): no downstream register read
  per forward while credits are known, no stack index/table chain, no duplicate size load. E1-bound rows with a backlog
  (scatter, fused, atomic, multicast): the packets-received read is skipped while packets are known present. E0-bound
  rows (1-hop E/W write) should not move: E1 drains to an empty ring and reads the register every pass, as before. The
  §80 caveat applies: 3-hop may move less than E1's saving (16-slot credit loop, E0 close behind).
- **Layout.** The common no-pending local-write path is a taken jump out of the loop body and back (like v3a, §68).
  Left alone (no hints); it may move by itself with any edit, as before.
- **Behaviour changes** (all on ERISC 1, §4): exact completion check; VC1 admission on VC1's own credits (VC1 runs may
  differ from v3a here — v3a's VC1 admission is wrong); no deferral bypass through the intermesh exit (no logged config
  reaches it).
- **VC1 performance.** VC1 builds call three shared bodies from the receive loop (calls in the loop, as E0's VC1 sender
  steps). VC1 throughput was slot-bound in §84; not measured with this code.
- **Not built offline:** watcher/DPRINT builds (stateful forwarding is off there, as in v3a; the program then forwards
  statelessly) and Wormhole (program compiled out).
- **Two receive implementations.** v3a's receiver step stays for ring (first-level ack), speedy and single-ERISC builds.
  Moving the ring receiver (and its batching, §76) into the program would let it go.
- **Next, if hardware agrees:** bulk consumption of packets and credits (one register write per refill instead of per
  packet, §80 optional step), refreshing credits on idle passes, ring and 1D deferral/hold (§76), the deferral's
  placement cost (§68) by structure.

---

## 9. v3e1b: local delivery, VC1 and -Os (after hardware jobs 224–226)

Variant: `fabric_exp/variants/v3e1b/` (only `fabric_erisc_router_v3.cpp` differs from v3a; everything outside the E1
program section is v3e1's, i.e. v3a's plus the one-line dispatch). Builds and disassembly in the same scratch dir
(`b1_*`, `mid_g*.dis`, `d1_g*.dis`, path counter `path.py`).

### 9.1 What hardware showed

- Job 224: v3e1 passes RELAXED/STRICT validation; forwarding and multicast improved; E1-bound local delivery regressed
  (atomic +7.7, scatter +4..7.5, fused +3, flushmix4 +7.4 mean, small writes +1..+3.5).
- Job 225: VC1 routers regress on large packets (1-hop E/W 4352 B: 321–332 at -O2/-O3 vs v3a ~254–258; N/S and
  multicast similar; 16 B even).
- Job 226: the whole ~20% -Os penalty is E1's (E0 -Os + E1 -O3 is fine; E1 -Os costs as much as all -Os).

### 9.2 Why v3e1 lost on local delivery (from the disassembly, v3a vs v3e1, hot mid-W E1 at -O3)

Per-packet path, 16 B local write, steady state (a batch of 4 credited every 4 packets; instruction counts from
`path.py` over the actual branch path):

| | v3a | v3e1 |
| --- | --- | --- |
| packet path (no completion) | 65 instr, 3 stack loads; packets-received register read every pass | 65 instr, 7 stack loads; register read skipped while `avail > 0` |
| local write | send-type dispatch: range test, table load, `jr`, jump back | same dispatch (unchanged `execute_chip_unicast_to_local_chip_impl`) |
| completion, per batch | 75 instr; 1 trid status read, then TXQ (CMD+STATUS) only if clear | 81 instr; 4 trid + 2 TXQ reads in one group on **every** attempt |

What moved:

1. **Completion check.** Six status reads per attempt instead of v3a's one (plus the two TXQ reads once it passes). By
   the §33 model (~11 cycles per read used at once, a group of 4 ≈ 26) a successful attempt costs ~+13–15 cycles over
   v3a's sequence and every failed attempt ~+19 (6 reads vs 1). That is ~+3–4 cycles/packet on E1-bound rows, more
   where attempts fail (writes still in flight when uncompleted reaches 5). Atomic batches pay it for nothing: their
   slots have no writes, so v3a's single read passes at once while v3e1 still issues the six-read group. This is the
   bulk of the atomic / flushmix4 / small-write regression.
2. **Atomic arm.** One extra dependent stack load (the acked-counter address was spilled) in front of the flushed
   atomic's register reads.
3. **Retire** grew by ~6 instructions (`avail`, the running slot pointer) while the packet path kept v3a's dispatch, so
   on rows where the packets-received read was not the bottleneck (ring drains to empty every pass) nothing paid for it.
4. **VC1 (job 225).** VC1 receiver rings have 4 transaction IDs for 8 slots (VC0 channel) and 4 slots (VC1 channel).
   The 4-trid check of slots base..base+3 always includes trid(base) = trid(base+4), and slot base+4 is already issued
   when the check runs (uncompleted ≥ 5) — so every batch waited for the **newest** packet's write as well. With
   4352 B packets that is one full write time per batch on an 8-slot ring, which makes the credit loop the
   bottleneck: large packets regress, 16 B does not. On top, v3e1 routed VC1 builds' channel 0 (VC0 traffic) through
   the out-of-line shared bodies (calls per packet for admission, forward and local delivery).

### 9.3 v3e1b changes

**a. Exact completion check from per-slot records (the cheaper exact check).** Each retired slot gets a one-byte
record: its `src_ch_id` (bits 0–5, `static_assert(MAX_NUM_SENDER_CHANNELS <= 64)`) and which command buffers read its
data (bit 6: a local write on the local-write command buffer — write, fused, scatter, sparse multicast; bit 7: a
forward on the forwarding data command buffer). Atomics and inline writes carry their values in command-buffer
registers and set neither bit; nor does a packet that is neither delivered nor forwarded. The record is written at
retire (after the packet's NoC commands). Records are stored as words, so a batch's four records are one word load
(two when the batch straddles a word). The check, per batch:

- no record has a read bit (atomics, inline writes, nothing) → no trid read at all;
- otherwise the newest slot that reads its data (`clz` on the read bits) is read; it proves every older slot on the
  command buffers it uses (in-order per command buffer, the same premise as v3a's last-slot check);
- only if an older slot reads on the *other* command buffer, the newest such slot is read too, in one fenced group
  with the first (two reads).

So uniform batches (all writes, all forwards, all multicast = forward + write on every slot) cost one read of the
newest slot, exactly v3a's read; atomic batches cost none; mixed batches cost two and are exact. The TX queue is read
only after the trids are clear (v3a's order). Because the newest-of-class slots of a uniform batch are base+3, their
trid is shared only with base+7, which is issued only when the ring is full: no waiting on newer packets on 4-trid VC1
rings (§9.2 item 4) for uniform batches; a mixed batch whose older reading slot is base can still wait on base+4 when
trids are shared (correct, conservative, rare). This replaces v3e1's 6-read group and an intermediate `kinds`
(OR-of-slot-kinds) version; the record version needs no loop-carried state and no special case for "nothing" slots.

**b. One credit for a full batch from one sender.** The same word gives the four `src_ch_id`s: if they are equal
(`((w ^ (w >> 8)) & 0x3F3F3F) == 0`), the batch is one `accumulate_completion_credit(src, 4)` instead of four per-slot
updates (same final counters; the frame is sent only after). Mixed sources keep the per-slot loop (over the bytes of the
word, no loads). Loop-free in the common case, so it does not depend on unrolling (see §9.4).

**c. Plain local write issued directly.** `e1_local_delivery` issues `NOC_UNICAST_WRITE` itself (the same NoC command
as the write arm: `noc_async_write_one_packet_with_trid` on the local-write command buffer with the slot's trid); every
other send type still goes through `execute_chip_unicast_to_local_chip_impl`, unchanged. Removes the range test, table
load, indirect jump and jump back from the hottest path.

**d. VC1.** Only the VC1 receiver channel (channel 1) uses the shared out-of-line bodies (`E1Rx<CH>::uses_shared_bodies
= CH == 1`); channel 0 always uses the inline compile-time forms, also on VC1 and crossover routers, so VC0 traffic on
a VC1 router runs the VC0-only code (plus the edge-router intercepts all logged VC1 routers have, as v3a). With (a) the
VC1 large-packet credit delay is gone for uniform batches. Init is table-driven: one `E1DownstreamSetup` entry per VC
and one call each to `e1_init_downstreams` / `e1_open_downstreams` / `e1_setup_downstream_cmd_bufs` (same writes,
same order as v3e1, which matched kernel_main's adapters; it removes v3e1's per-VC constprop clones).

**e. Smaller things.** `e1_slot_add<N>` for slot `base + n` (v3a's `wrap_increment_n<2>` ignores the increment, so its
last-slot index is wrong for 2-slot rings; no logged receiver ring has 2 slots); the record words are zero-initialized;
the send-type shift is masked.

### 9.4 -Os: what changes, and what v3e1b does about it

Comparing -O3 and -Os disassembly of the hot mid-W E1 (`10856749301170935304`, the 1-hop E/W local-delivery E1 of §80)
and the 1D line E1 (`10388088869406661095`), v3a and v3e1b:

1. **No calls appear in the hot path at -Os** (everything on it is `FORCE_INLINE`); inlining is not the issue.
2. **Loops are not unrolled.** v3a's per-slot credit loop costs ~17 instructions per slot at -Os (address
   rematerialized each iteration, index wrap, counter) vs ~12 unrolled at -O3: +26 per batch. v3e1b's uniform batch
   is one credit, no loop: its completion costs 65 instructions per batch at -Os vs 62 at -O3.
3. **No block duplication for jump threading, "simple" block order.** At -O3 the second test of the forward bit (after
   the deferred-atomic check) is threaded into the first; at -Os it is re-tested, and the dispatch path in v3a adds a
   range test, `jr` and a jump back: v3a's -Os local-write path takes 6 taken branches (+ back edge) vs 3 at -O3.
   v3e1b's direct write removes the dispatch, and the record removes v3e1's "nothing" special case, so the -Os path
   has 2 taken branches (+ back edge).
4. **Frequency-blind register allocation.** At -Os the allocator weighs every block equally, so it keeps cold-path
   constants in registers and spills the inner pass counter (`lw`/`addi`/`sw` per pass) and the route-byte offset.
   This is allocation, not structure; v3e1b does not steer it (no pinning, no attributes). It reduces the live set
   instead (no loop-carried `kinds`, no slot table) — the remaining -Os gap on the mid-W path is mostly these spills.
5. **Constants** (NoC control word, coordinate mask, counter addresses) are hoisted and spilled at both levels.

Per-packet instruction counts on the actual branch path (16 B local write, steady state, batch of 4 from one sender,
batch completion amortized over 4 packets; "+" columns are the -Os penalty):

| E1 | v3a -O3 | v3a -Os | + | v3e1 -O3 | v3e1 -Os | + | v3e1b -O3 | v3e1b -Os | + |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mid-W, local write | 83.8 | 104.5 | +25% | 85.3 | 105.0 | +23% | **72.5** | **82.3** | +14% |
| 1D line, local write | 78.5 | 104.0 | +32% | | | | **75.3** | **80.0** | +6% |
| mid-W, forward (packet path only) | 74 | 90 | | 62 | 74 | | 68 | 75 | |

Breakdown for v3e1b mid-W: packet path 57 / 66 instructions (-O3 / -Os; 5 / 7 stack loads, 1 stack store at -Os),
completion 62 / 65 per batch. v3a: 65 / 79 and 75 / 102. Status reads per batch: v3a 1 + 2 (TXQ), v3e1 6 every
attempt, v3e1b 1 + 2 (0 + 2 for atomic batches, 2 + 2 grouped for mixed). Register reads per packet that the counts
do not show: v3a reads the packets-received register every pass and the downstream credit register on every forward;
v3e1/v3e1b only when their cached counts run out.

So v3e1b at -Os is about v3a at -O3 on the mid-W local path (82 vs 84 instructions) and well under it on 1D; the
-Os penalty is roughly halved on 2D and mostly gone on 1D. What is left on 2D is items 3–5 (spilled pass counter and
route offset, one re-test of the forward bit), which only allocation or layout steering would remove.

The forward packet path is 6 instructions longer than v3e1's at -O3 (68 vs 62: one more value live across the forward
— the slot record — gives one spill/reload pair and a reloaded counter address; same at -Os, 75 vs 74). Per batch the
completion saves 3 status reads per credited batch (5 per failed attempt) and ~19 instructions against v3e1, so
forwarding rows should be about even with v3e1.

### 9.5 Builds

| Build | Result |
| --- | --- |
| v3e1b -O3 / -O2 | 536/536 each |
| v3e1b -Os | 440/440 VC0 + 96/96 VC1 (VC1 from the -O2 commands with `--set-opt Os`, as for v3a/v3e1) |
| codegen_diff vs v3a | E0 **0/208** at -O3 and -O2, **0/160 + 0/48** at -Os; E1 128/328 changed, exactly v3e1's 128 program configs; the other 200 E1s identical |
| hot mid-W E1 | 1926 → 1905 instructions at -O3 (v3e1 1818), frame 720 → 432 |

E1 `.text` at -O3 (v3a / v3e1 / v3e1b): mid-W 7,704 / 7,272 / 7,620; N/S 7,804 / 7,452 / 7,908; 1D line
8,620 / 5,920 / 6,712; largest VC1 E1 11,860 / 8,692 / 11,900.

**VC1 code size** (largest E0 + largest E1 loadable bytes over the 96 VC1 kernels; E0 is v3a's):

| Level | E0 (v3a) | E1 v3a | E1 v3e1 | E1 v3e1b | sum with v3a's E0 |
| --- | ---: | ---: | ---: | ---: | --- |
| -O3 | 16,996 | 12,708 | 9,064 | **12,140** | 29,136 (over by 2,512) |
| -O2 | 13,920 | 12,232 | 8,408 | 11,988 | 25,908 (fits) |
| -Os | 12,344 | 11,660 | 7,816 | 11,084 | 23,428 (fits) |

E1 is within the ~12.5 KB target at -O3: it leaves 14,484 B for E0 in the 26,624 B buffer, so it fits with an E0 of
12.8–13.4 KB (v3l/v3u2), but not with v3a's E0 (v3a itself is over by 3,080). Crossover E1s: 9,616. Where the -O3
bytes are (largest VC1 E1): two copies of the send-type dispatch (inline in channel 0: 1,664 B; the shared body for
channel 1: 1,812 B), forward 868, admission 560, completions 656 + 192, init/open/setup 1,572, the rest loop and
teardown. If the E1 budget gets tighter, channel 0's non-write send types can go through the shared body on VC1 builds
(−~1.6 KB, at the cost of a call for atomics/scatter/fused/inline on VC0 traffic of VC1 routers; writes stay inline).

**Probes** (perturb.py, -O3): hook — E0 identical to v3a + hook (0/208), all E1s change (median +6 instructions);
statefield — E0 identical to v3a + statefield, the 128 program E1s do not change (the member is never read), the 120
single-ERISC E1s change as in v3a.

### 9.6 Risks and what to measure

- **Exactness premise** is v3a's (one command buffer's writes leave in issue order), now relied on per command buffer
  with the records saying which buffer each slot used. A send type that read the slot on a third command buffer would
  break it; none does today (all slot-reading local types use `local_chip_data_cmd_buf`, forwards the VC's data
  command buffer).
- **Rows to watch:** E1-bound local delivery (atomic, scatter, fused, flushmix4, small writes) should recover to v3a or
  better (no dispatch on writes, 0–1 trid reads per batch, one credit per uniform batch). VC1 large packets should be
  back to v3a (no wait on the newest write). Multicast on VC1 now does one read per batch. Forwarding should be about
  v3e1 (§9.4). -Os builds should lose about half of v3a's 2D penalty and most of 1D's.
- **VC0 traffic on VC1 routers** runs the VC0-only code for channel 0, but every pass still polls channel 1's
  packets-received register when VC1 is idle (one register read and ~10 instructions per pass, as v3a). Hiding it
  would need either issuing that read before channel 0's step and using it after (allocation-sensitive) or a slower
  idle poll of channel 1 (a VC1 first-packet latency trade-off); neither is done here, both need measurement.
- **VC1 -O3 size** fits only with a smaller E0 than v3a's (above).
- **Mixed batches on shared-trid rings** can still wait for a newer packet when the older reading slot is the batch's
  first (rare pattern; correct).
- Hardware-untested parts carried from v3e1: E1's own downstream connection setup (now table-driven, same writes and
  order), VC1 admission on VC1's credits.

## 10. v3e1c / v3e1d: scatter, atomic and fused local delivery (on v3l2be1b)

Variants: `fabric_exp/variants/v3e1c/` (the proposal) and `fabric_exp/variants/v3e1d/` (control). Both are v3l2be1b
with one change, in the E1 program section of `fabric_erisc_router_v3.cpp`: `e1_local_delivery` and a new helper
just above it, `e1_local_write_uncounted`. Headers, E0, init and everything else are v3l2be1b's. Scratch:
`/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3e1c/`. Builds: `fc_*` =
v3e1c, `fd_*` = v3e1d, `base_*` = v3l2be1b. Path tools: `walk.py`, `scen.py`, `runtab.sh`, `fmt.py`. Tables:
`dis/table9.txt`, `dis/fwd9.txt`.

### 10.1 Gaps (hardware, -O3 cycles/packet)

v3l2 is the lean E0 with v3a's E1, so comparing v3l2 with v3l2be1b isolates the E1 change.

| Row | v3a | v3l2 | v3l2be1b |
| --- | ---: | ---: | ---: |
| EW / NS PureScatter4352 | 165.3 / 176.1 | 165.1 / 177.6 | 167.6 / 180.9 |
| FWD_PureScatter4352 / FWD_PureFused4352 | 165.2 / 155.2 | 164.9 / 153.9 | 171.2 / 160.8 |
| EW / FWD PureAtomic | 129.3 / 129.2 | | 132.3 / 133.0 |
| 1D Linear1HopEW / EWBidir / 3HopEW scatter | 167.1 / 167.1 / 167.0 | | 172.7 / 174.0 / 173.6 |
| EW_Chunk4Fused_Flush / Chunk8 | 156.7 / 137.9 | | 158.9 / 139.2 |

In all these rows the destination's E1 delivers to its own chip. FWD rows mix two flows into the destination's E1;
Chunk rows send 3 or 7 writes followed by a fused write+increment with `flush: true`, which is deferred.

### 10.2 Cause: what a non-write packet costs on v3e1b's path

v3e1b removed only the plain write from `execute_chip_unicast_to_local_chip_impl` (§9.3c). Every other send type
still pays the following, shown here on the mid-W E1 at -O3:

1. **The read-class computation**, `record |= e1_local_read_bits(type)`: `li 281; srl; andi; slli` plus the `or`. v3a
   has no records.
2. **The impl's dispatch.** Without the write case, GCC lowers the impl's switch to a compare chain: a sparse test, a
   range test and per-type compares. That is 7 instructions on the scatter and atomic paths and 4 on the fused path,
   with 2 extra taken branches (out of the write fall-through, then into the chain).
3. **Scatter's software NoC write counters, updated per chunk.** On BH, `noc_async_write_one_packet_with_trid` always
   updates `noc_nonposted_writes_num_issued/acked` (#28758). That costs `lw/addi/sw` for each counter, plus the counter
   addresses: 7 instructions per chunk, 14 for the 2-chunk test packet.
4. **E1's per-packet step overhead over v3a** (the record store and the slot pointer): about +4. The write path pays for
   it with the dispatch it no longer runs; other send types got nothing in return.

Instruction counts by category (mid-W, -O3; "step" is the receive loop and per-packet bookkeeping):

| Path | v3a | v3e1b |
| --- | --- | --- |
| scatter | 105 = step 28 + dispatch 10 + arm 21 + NoC commands 32 + counters 14 | 110 = step 32 + read class 4 + dispatch 7 + arm 21 + NoC 32 + counters 14 |
| atomic | 75 = step 28 + dispatch 10 + arm 8 + NoC 25 + counters 4 | 81 = step 32 + read class 4 + dispatch 7 + arm 8 + NoC 26 + counters 4 |
| fused, flush deferred | 99, 4 taken branches | 100, 8 taken branches |

On E1-bound rows the time between packets roughly follows the instruction path (§9). So the scatter gap grows from W
(+5 instructions) to NS (+8) to 1D (+9), as the hardware gap does (+2.3, +4.8, +5.6 cycles).

### 10.3 The change

**v3e1c.** `e1_local_delivery` is an if / else-if chain in source order: write, scatter, atomic, fused, then
everything else. Write stays first, so its path is unchanged. It serves channel 0, including channel 0 of VC1 routers.
VC1's channel 1 keeps the shared out-of-line body, unchanged.

- **Scatter** issues the same commands as the impl's ERISC scatter arm: up to three writes, then the last chunk as a
  write or a semaphore increment, with the same flush-type mapping and the slot's trid. Two things differ:
  - Each write goes through `e1_local_write_uncounted`, which is `noc_async_write_one_packet_with_trid`'s command with
    `update_counter = false` passed to `ncrisc_noc_fast_write`.
  - The two software counters are incremented once, by the number of writes, after the last write and before the
    optional semaphore increment (its flush test reads them). Nothing reads them between the writes. The router is
    `DM_DEDICATED_NOC` (static_assert), so these are the counters `ncrisc_noc_fast_write` would update. This matches
    what the DRISC scatter path does.
- **Atomic** is the impl's atomic arm: `execute_local_semaphore_inc(addr, val, flush, CH, pending_atomic_flush,
  (trid - 1) & 0xF)`. It sets no read class, as before.
- **Fused** is the impl's fused arm: a counted write with the slot's trid, then
  `execute_local_semaphore_inc(..., trid)`. It sets the LOCAL read class.
- Each arm sets its read class as a constant. `e1_local_read_bits` now runs only for the remaining types (inline write
  and sparse multicast).

There are no hints or attributes. The impl wraps its ERISC arms in `#if !defined(COMPILE_FOR_DRISC)`; the new arms do
not need it, because the router file is built only for the ERISCs (the DRISC kernel is
`fabric_drisc_datamover.cpp`). GCC lowers the chain to compare-and-branch at all three levels: no E1 configuration's hot
loop has a jump table except VC1 channel 1's shared body, which is unchanged.

Rejected shapes (probe builds, mid-W -O3):

- A switch after the write test makes a jump table with its bases spilled: scatter 111.
- A full if-chain over every type is turned into a jump table that includes the write: write 67.
- Early returns move the write off the fall-through: write 58 with 5 taken branches.

**v3e1d** has the scatter arm only; atomic and fused go through the impl, as in v3e1b. It is the control: it costs no
VC1 size and separates the value of the atomic and fused arms on hardware.

### 10.4 Per-packet path counts

**Method.** `scen.py` drives `walk.py`, which finds the shortest path through `objdump -dl` from the receive loop's
head to its back edge. If the loop has several back edges, it takes the shortest over all of them. It follows calls and
returns and resolves jump tables from the ELF's `.data`. The constraints for each case:

- **Avoided:** memory accesses and calls in the packets-received register read, the pending-atomic resolution block,
  the completion body and the forward bodies (or, for the forward case, the local-delivery code). Avoiding only memory
  accesses is what keeps this independent of how GCC attributes ALU and branch instructions to source lines.
- **Required:** a memory access in the wanted send-type arm, and a number of NoC command issues: a store to
  `NOC_CMD_CTRL` (offset 0x40 mod 0x800) from the NoC API. Write 1, scatter 2, atomic 1, fused 2, deferred fused 1,
  forward 1.
- **Fixed branch outcomes:** no pending atomic; the scatter's last chunk is a write; atomic is flushed and issued at
  once (writes already flushed); fused has no flush; for "flush deferred", the flush is set and the writes are not yet
  flushed.

1D has no deferral, so in 1D the flushed atomic and fused completion are counted with the flush already complete.
Completion batches are not included: the completion code is unchanged.

Cells are instructions, with taken branches (excluding the back edge) in parentheses. v3e1c is in bold. The
forward-only case is not measured for v3a.

-O3:

| E1 | path | v3a | v3e1b | v3e1c | v3e1d |
| --- | --- | ---: | ---: | ---: | ---: |
| W | write | 65 (2) | 57 (3) | **57 (3)** | 57 (3) |
| W | scatter (2 chunks) | 105 (5) | 110 (7) | **96 (7)** | 96 (7) |
| W | atomic (issued at once) | 75 (6) | 81 (6) | **72 (7)** | 82 (6) |
| W | fused, no flush | 98 (5) | 100 (6) | **97 (6)** | 101 (6) |
| W | fused, flush deferred | 99 (4) | 100 (8) | **96 (7)** | 101 (8) |
| W | forward only | – | 68 (5) | **68 (6)** | 66 (5) |
| NS | write | 73 (6) | 63 (6) | **63 (6)** | 64 (6) |
| NS | scatter (2 chunks) | 110 (7) | 118 (7) | **105 (7)** | 106 (7) |
| NS | atomic (issued at once) | 82 (7) | 87 (7) | **78 (7)** | 89 (7) |
| NS | fused, no flush | 105 (7) | 105 (7) | **99 (7)** | 108 (7) |
| NS | fused, flush deferred | 105 (6) | 106 (6) | **100 (6)** | 108 (6) |
| NS | forward only | – | 101 (6) | **103 (7)** | 104 (5) |
| 1D | write | 62 (5) | 60 (5) | **62 (5)** | 60 (5) |
| 1D | scatter (2 chunks) | 101 (8) | 110 (8) | **100 (8)** | 97 (8) |
| 1D | atomic (flushed) | 73 (6) | 85 (6) | **80 (7)** | 86 (6) |
| 1D | fused, no flush | 94 (6) | 101 (8) | **101 (8)** | 104 (8) |
| 1D | fused, flushed | 100 (6) | 107 (8) | **107 (7)** | 110 (8) |
| 1D | forward only | – | 66 (7) | **68 (7)** | 68 (7) |

-O2:

| E1 | path | v3a | v3e1b | v3e1c | v3e1d |
| --- | --- | ---: | ---: | ---: | ---: |
| W | write | 65 (2) | 59 (3) | **58 (3)** | 59 (3) |
| W | scatter (2 chunks) | 105 (5) | 111 (7) | **99 (7)** | 97 (8) |
| W | atomic (issued at once) | 75 (6) | 82 (6) | **74 (7)** | 82 (6) |
| W | fused, no flush | 98 (5) | 101 (6) | **98 (6)** | 102 (6) |
| W | fused, flush deferred | 99 (4) | 101 (8) | **97 (7)** | 102 (8) |
| W | forward only | – | 70 (5) | **68 (6)** | 68 (5) |
| NS | write | 73 (7) | 64 (6) | **65 (6)** | 65 (6) |
| NS | scatter (2 chunks) | 110 (8) | 119 (7) | **104 (7)** | 104 (7) |
| NS | atomic (issued at once) | 82 (8) | 88 (7) | **80 (7)** | 89 (7) |
| NS | fused, no flush | 105 (8) | 106 (7) | **101 (7)** | 108 (7) |
| NS | fused, flush deferred | 105 (7) | 107 (6) | **102 (6)** | 108 (6) |
| NS | forward only | – | 103 (6) | **106 (7)** | 107 (5) |
| 1D | write | 64 (5) | 61 (5) | **60 (5)** | 60 (5) |
| 1D | scatter (2 chunks) | 102 (7) | 114 (8) | **98 (8)** | 98 (8) |
| 1D | atomic (flushed) | 74 (6) | 85 (6) | **76 (7)** | 86 (6) |
| 1D | fused, no flush | 95 (5) | 102 (8) | **100 (9)** | 105 (8) |
| 1D | fused, flushed | 101 (5) | 108 (8) | **106 (8)** | 111 (8) |
| 1D | forward only | – | 66 (7) | **67 (7)** | 68 (7) |

-Os:

| E1 | path | v3a | v3e1b | v3e1c | v3e1d |
| --- | --- | ---: | ---: | ---: | ---: |
| W | write | 79 (5) | 66 (2) | **65 (2)** | 65 (2) |
| W | scatter (2 chunks) | 117 (7) | 117 (6) | **102 (4)** | 102 (4) |
| W | atomic (issued at once) | 89 (8) | 91 (9) | **80 (6)** | 92 (8) |
| W | fused, no flush | 113 (9) | 111 (10) | **106 (8)** | 113 (11) |
| W | fused, flush deferred | 112 (6) | 108 (6) | **106 (6)** | 110 (7) |
| W | forward only | – | 73 (3) | **71 (3)** | 71 (3) |
| NS | write | 92 (7) | 74 (4) | **73 (4)** | 73 (4) |
| NS | scatter (2 chunks) | 128 (7) | 127 (8) | **112 (6)** | 114 (6) |
| NS | atomic (issued at once) | 102 (10) | 100 (10) | **90 (8)** | 102 (10) |
| NS | fused, no flush | 126 (11) | 120 (11) | **116 (10)** | 123 (13) |
| NS | fused, flush deferred | 123 (7) | 118 (8) | **116 (8)** | 120 (9) |
| NS | forward only | – | 112 (5) | **109 (5)** | 109 (5) |
| 1D | write | 80 (5) | 64 (5) | **61 (5)** | 62 (5) |
| 1D | scatter (2 chunks) | 117 (6) | 114 (9) | **98 (6)** | 99 (6) |
| 1D | atomic (flushed) | 93 (9) | 89 (10) | **79 (9)** | 91 (11) |
| 1D | fused, no flush | 113 (8) | 104 (10) | **101 (11)** | 106 (12) |
| 1D | fused, flushed | 121 (10) | 111 (11) | **106 (10)** | 112 (12) |
| 1D | forward only | – | 75 (5) | **72 (5)** | 73 (5) |

The largest VC1 E1 (`9530214107180203457`, channel 0 traffic), v3e1b → v3e1c:

| Path | -O3 | -Os |
| --- | --- | --- |
| write | 84 → 85 | 100 → 100 |
| scatter | 134 → 119 | 150 → 137 |
| atomic | 107 → 100 | 125 → 118 |
| fused | 126 → 123 | 143 → 139 |
| fused, flush deferred | 131 → 129 | 149 → 143 |
| forward only | 122 → 121 | 133 → 133 |

Reading the tables (v3e1c against v3e1b at -O3 unless stated):

- **Scatter** is −14 on W, −13 on NS and −10 on 1D, and now at or below v3a on every E1 at every level (W 96 vs v3a's
  105, NS 105 vs 110, 1D 100 vs 101). Where it comes from: the read class becomes one `ori` (−3), the sparse test and
  compare chain are replaced by the arm chain (about −6), and the counters are updated once instead of per chunk (−6).
  About +1 comes back as spills (the record and the last-chunk index).
- **Atomic**: W 72 vs v3a's 75, NS 78 vs 82. 1D is 80 vs v3a's 73: 1D has carried this E1 overhead since v3e1b, and
  it is not in the gap list.
- **Fused**: W 97 vs v3a's 98, NS 99 vs 105. Deferred fused, as in the Chunk rows: W 96 vs 99, NS 100 vs 105.
- **Writes** are unchanged except 1D at -O3, which is +2 (62 vs 60): two register copies and two counter-address
  reloads. That is allocation; the source of the write path did not change. It is gone at -O2 and -Os (60 and 61, vs
  61 and 64) and under the hook probe.
- **Forward-only** paths move by ±2 to 3 instructions: W 68 = 68; NS 103 vs 101; 1D 68 vs 66 at -O3; at -Os 71 / 109 /
  72 vs 73 / 112 / 75. The forward source did not change. In 1D the cause is one more spill/reload pair and one `mv`.
  Under the hook probe the forward paths are equal.
- **v3e1d against v3e1c**: the same scatter gains; atomic and fused stay at v3e1b's counts.

### 10.5 Builds, size, probes

| Build | v3e1c | v3e1d |
| --- | --- | --- |
| -O3 / -O2 | 536/536 each | 536/536 each |
| -Os | 440/440 VC0 + 96/96 VC1 (VC1 from the -O2 commands with `--set-opt Os`) | same |

**PT_LOAD comparison against v3l2be1b** (address, file size, memory size and bytes of every LOAD segment; ERISC0 =
`active_erisc`, ERISC1 = `subordinate_active_erisc`; script `ptload.py`). Both variants, at every level:

- ERISC0 is identical on every config: 208/208 at -O3 and -O2, and 160/160 + 48/48 at -Os.
- ERISC1 changed in exactly the 128 program configs, the set where v3l2be1b's E1 differs from v3a's. The other 200 are
  identical.

**VC1 code size** (largest E0 plus largest E1 loadable bytes over the 96 VC1 kernels; E0 is v3l2b's):

| Level | E0 | E1 v3l2be1b | E1 v3e1c | E1 v3e1d | sum with v3e1c |
| --- | ---: | ---: | ---: | ---: | --- |
| -O3 | 13,200 | 12,140 | **12,240** | 12,140 | 25,440 (1,184 spare of 26,624) |
| -O2 | 12,316 | 11,988 | 12,048 | 11,960 | 24,364 |
| -Os | 10,256 | 11,084 | 11,072 | 11,040 | 21,328 |

So v3e1c spends 100 B of the 1,284 B spare at -O3, and v3e1d spends none. Crossover E1 at -O3: 9,616 → 9,592 (v3e1c)
and 9,588 (v3e1d).

**Probes** (`perturb.py` hook and statefield on v3l2be1b and on v3e1c, -O3, all 536 configs):

- ERISC0 is identical between v3l2be1b with a probe and v3e1c with the same probe (208/208 for each probe).
- statefield leaves the 128 program E1s unchanged, as it does on v3l2be1b.
- With the hook, the paths go from v3e1b to v3e1c as follows (mid-W / NS / 1D):

| Path | mid-W | NS | 1D |
| --- | --- | --- | --- |
| write | 61 → 61 | 68 → 67 | 65 → 65 |
| scatter | 112 → 96 | 119 → 103 | 114 → 101 |
| atomic | 84 → 75 | 92 → 82 | 89 → 83 |
| fused | 104 → 101 | 108 → 104 | 105 → 104 |
| deferred fused | 104 → 100 | 109 → 105 | 111 → 109 |
| forward | 73 → 73 | 108 → 108 | 72 → 72 |

The gains survive the probe. The 1D write and forward differences seen at plain -O3 do not, which confirms they are
allocation.

### 10.6 Risks and what to measure

- **Rows that should close to v3a or better** (v3e1c, at -O3 and -Os):
  - EW, NS and FWD PureScatter4352, and the 1D Linear1HopEW, EWBidir and 3HopEW scatter rows.
  - EW and FWD PureAtomic.
  - FWD_PureFused4352, and EW_Chunk4Fused_Flush and Chunk8.
  - v3e1d should close the scatter rows only. If the atomic and fused rows close on v3e1c but not on v3e1d, the extra
    100 B of VC1 code are paying off.
- **Rows that should not change:** NS_PureAtomic (135.6), UnidirLinearMulticast scatter (233.8), MeshMulticast write
  and Mesh3HopEW write. Their send types and the multicast and forward source are unchanged.
- **Rows to watch:** 1D write rows at -O3 (+2 instructions, allocation) and NS / 1D forwarding rows at -O3 and -O2 (+2
  to 3 instructions, allocation).
- **Merged counters.** They rely on nothing reading `noc_nonposted_writes_*` between a scatter's writes:
  - Nothing does. It is one thread; the semaphore increment's flush test reads the counters after the update; the
    completion check reads trid status registers, not these counters.
  - The perf tests do not exercise every scatter shape. They send 2-chunk packets whose last chunk is a write. 3- and
    4-chunk packets and a semaphore-increment last chunk (flushed or not) are only checked against the impl's arm by
    reading. Run the RELAXED/STRICT validation, plus any scatter-with-semaphore validation that exists.
- **VC1:** channel 1 is unchanged (shared body). VC0 traffic on VC1 routers gets the new arms.

## 11. v3t: a hot loop that cold edits do not move, and paths less dependent on the opt level (on v3sb)

Variants:

- `fabric_exp/variants/v3t/`: the proposal.
- `fabric_exp/variants/v3tb/`: a control. It is v3t with `run_main_loop` inlined (`FORCE_INLINE` instead of
  `noinline`, plus a comment), to measure what v3t's function boundary costs.
- `v3t_strict`, `v3t_hook`, `v3tb_strict`, `v3tb_hook`: made as `prep_round.sh` makes them. The `_strict` and `_hook`
  dirs are built at -O3 on all 536 configs with `--jobs 16`.
- `v3t/derive_edits.py` (build-ignored) applies v3r2's six edits to a joined v3sb or v3t.

All changes are in `fabric_erisc_router_v3_receiver_erisc.hpp`, plus `run_kernel` and the start of `kernel_main` in the
.cpp. E0 is PT_LOAD-identical to v3sb. Scratch: `/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3t/`.
It holds the builds `gt_*` (v3t), `fb_*` (v3tb) and `sb_*` (v3sb), and these tools:

- `rxscen.py`, `walk.py`, `tab.sh`, `cmp.py`: per-path counts.
- `fncmp.py`: compares the main-loop region.
- `stability.sh`, `stab.sh`: the edit test.
- `rtlfirst.py`: finds the first RTL pass where a function's dumps differ.

### 11.1 Why cold edits moved E1's hot loop

1. **The loop shared a function with init.** v3sb inlines `run_main_loop` into `run_kernel`, and that into
   `kernel_main`. Depending on size, `kernel_main` is also inlined into `_start` together with the firmware's launch
   code. Any edit to init, the handshake or teardown therefore changes the function the loop is compiled in.
2. **Edits that change no code near the loop still moved it.** `templates` only turns template helpers into plain
   functions, and `os` only lets cold helpers be inlined into init. They still moved the loop at -O3 and -O2, both in
   v3sb and in a v3t draft whose init and teardown were already out of line. To find out how, I compared GCC's dumps
   with and without `templates` (W config, -O3):
   - The loop function's optimized GIMPLE is identical apart from declaration numbers (`-fdump-tree-optimized`).
   - The first difference is in RTL expansion (`-fdump-rtl-all`, `rtlfirst.py`). The same `x * 17` is expanded as a
     `mul` in one build and as shift + add in the other.
   - GCC expands a multiplication by a constant that is not a power of two through a cached cost search (synth_mult).
     The cache is shared by every function compiled in the unit, so the choice depends on what was compiled earlier.
   - One differently expanded multiply renumbers the pseudo registers, and register allocation and block layout then
     differ across the whole function.

   The loop's code had five such multiplications:

   | Multiplication | Where it comes from |
   | --- | --- |
   | `trid_b * 17` | the deferred-atomic encoding `0x100 \| (trid_b << 4) \| trid_b` |
   | `launch_msg_rd_ptr * 144` | `got_immediate_termination_signal`'s mailbox indexing |
   | `covered * 0x01010101` | the completion check |
   | `index * 3` | the intermesh direction table |
   | `i * 40` | runtime-indexed `DownstreamCursor`s |

So insulation needs both a function boundary and no constant multiplications inside the function. (Revised in RESULTS §96. The boundary-only draft also kept the per-round work, the termination check, heartbeat and
context-switch handshake, inside the loop function, so it did not isolate the multiplies. On the final v3t, with
`t * 17` put back into the loop, seven edits including cold multiplies left the loop code unchanged. What matters is
the function boundary plus per-round work out of line; the multiply effect is a gotcha, not a rule.) (RESULTS §97, the 2x2: the multiply has no effect in any cell; with the
per-round work inlined, cold edits re-lay out even a loop with no multiplies. Per-round work out of line is the
insulation.) With the boundary
alone (a measured draft), the loop function still changed under `templates`, `os` and `heartbeat`. sender_erisc's loop
has no such multiplication, which is consistent with E0 not moving under v3r2.

### 11.2 Changes (v3t)

Insulation:

- **`run_main_loop` is a function of its own** (`noinline`, like `sender_erisc::run_main_loop`).
  - Its only inputs are the downstream connections. It builds its own transaction ID trackers (stateless for every
    logged ring), and takes compile-time addresses and tables itself instead of receiving them.
  - `run_kernel` is unchanged apart from the call. Teardown's trackers carry no state it uses.
- **No multiplication by a constant that is not a power of two inside it** (checked in the optimized GIMPLE of the
  four hot configs):
  - The deferred atomic's "one trid left" state is now `(pending & ~0xF) | trid_b`, the same value.
  - The completion check finds each command buffer's newest reading slot directly (two masks and clz) instead of
    broadcasting with `* 0x01010101`. It is the same check.
  - `DownstreamCursor` is 32 bytes. The local credit register's read address is now the update address plus a constant
    (`stream_credit_read_offset`, static_asserted to be linear) instead of a stored field.
  - The intermesh-exit lookup moved into `intermesh_egress_slot` (noinline; once per exit packet on edge routers).
  - The termination test is `termination_requested`. The heartbeat and context-switch handshake are `end_round`. Both
    are noinline and run once per round.

Less dependence on the opt level (these remove work, or replace something only -O3 does by structure):

- **Completions.**
  - A full batch whose four records have the same read bits needs no newest-slot search: it is the last slot, or no slot
    reads.
  - A full batch from mixed senders gives four credits written out in the source; v3sb used a loop, which -O2 and -Os do
    not unroll.
  - Partial batches, which happen only on passes without progress, keep the loop.
  - Per batch, this is −13 to −16 instructions with one sender and −14 to −27 with mixed senders, at every level
    (§11.4).
- **1D decides once from the route value.** `WRITE_ONLY` goes to local delivery, `FORWARD_ONLY` forwards, and
  `WRITE_AND_FORWARD` does sparse multicast or forward then deliver. In v3sb the deliver, forward and sparse flags were
  computed up front and tested again after the forward.
- **2D packets that are only delivered here** test the live forward bits once (`live_forward_action_bits`), in
  admission and around `forward_2d`. -Os does not merge the re-tests after the deferred-atomic block (no jump
  threading).
- **The rare local send types go through `local_delivery_shared`** (inline writes, multicast, sparse). It is now
  defined for all builds, takes and returns the deferred-atomic state, and takes their dispatch out of the loop.

VC2 (from the coverage builds): `run_kernel`'s definition and its call in `kernel_main` are guarded by
`defined(ARCH_BLACKHOLE) && !defined(FABRIC_2D_VC2_SERVICED)`, the same condition as `receiver_erisc::runs`.

### 11.3 Stability test: v3r2's six edits

Method:

- `derive_edits.py` applies each edit alone, then the hot configs are built at -O3, -O2 and -Os (mid-W, N/S, 1D and
  the largest VC1 E1).
- I compare the main-loop region, which is the largest strongly connected region of the control-flow graph, wherever
  it is inlined (`fncmp.py`).
- Normalization: branch targets inside the region are taken relative to its start. Callees are compared by name
  without template arguments or clone numbers, so a helper that only became a plain function compares equal.
- "moved" means the same code at a different start address.

| Edit | v3sb | v3t | v3tb (inline loop) |
| --- | --- | --- | --- |
| templates | loop changes at -O3/-O2 (W, N/S, 1D) | same | same |
| os | loop changes at -O3/-O2 (all four) | same, moved at -O3/-O2 | changes at -O3/-O2 |
| heartbeat | same | same | same |
| param | same | same | same |
| bittest | same | same | same |
| init | loop changes at all levels (all four) | same, moved | changes at all levels |

- In v3sb the changes move per-path counts by up to ±3 instructions at -O3 and -O2, and up to ±3 at -Os under `init`.
- v3t's loop code is unchanged under all six edits at all three levels on all four configs; only its start address
  moves.
- v3tb shows that the function boundary is needed.

### 11.4 Per-path counts (instructions; v3sb / **v3t** / v3tb)

The paths are the per-packet paths of §10.4, with two additions:

- They are now forced not to wrap the ring, which removes a once-per-ring path that was shorter in some builds.
- "write + completion" adds the batch completion of a full batch of four writes. From one sender it is the credit of 4;
  from mixed senders it is four credits, which is what the FWD rows' destination sees.

The completion costs about a quarter of its delta per packet.

W (mid-W):

| path | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| write 16 B | 57 / **62** / 60 | 58 / **63** / 61 | 65 / **61** / 61 |
| forward only | 68 / **72** / 72 | 68 / **72** / 72 | 71 / **68** / 68 |
| atomic | 72 / **76** / 73 | 74 / **77** / 75 | 80 / **74** / 74 |
| fused | 97 / **101** / 98 | 98 / **101** / 99 | 106 / **99** / 99 |
| fused, flush deferred | 96 / **100** / 98 | 97 / **101** / 99 | 106 / **99** / 99 |
| scatter | 96 / **96** / 94 | 99 / **99** / 96 | 102 / **97** / 97 |
| write + completion, one sender | 119 / **106** / 106 | 119 / **109** / 107 | 130 / **114** / 114 |
| write + completion, mixed senders | 146 / **132** / 131 | 150 / **134** / 133 | 167 / **140** / 140 |

N/S:

| path | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| write 16 B | 63 / **66** / 67 | 65 / **65** / 66 | 73 / **72** / 71 |
| forward only | 103 / **113** / 113 | 106 / **112** / 112 | 109 / **112** / 109 |
| atomic | 78 / **82** / 84 | 80 / **81** / 83 | 90 / **87** / 88 |
| fused | 99 / **105** / 104 | 101 / **104** / 103 | 116 / **113** / 112 |
| scatter | 105 / **100** / 101 | 104 / **101** / 101 | 112 / **110** / 109 |
| write + completion, one sender | 125 / **110** / 112 | 126 / **110** / 113 | 138 / **122** / 122 |
| write + completion, mixed senders | 152 / **136** / 138 | 157 / **136** / 137 | 175 / **148** / 148 |

1D (line). v3tb's -O3 loop is rotated so that the path tool cannot isolate a packet, so those cells are not measured:

| path | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| write 16 B | 62 / **58** / – | 60 / **58** / 62 | 61 / **62** / 63 |
| forward only | 68 / **72** / – | 67 / **72** / 77 | 72 / **72** / 74 |
| atomic | 80 / **72** / – | 76 / **72** / 77 | 79 / **76** / 79 |
| fused | 101 / **97** / – | 100 / **97** / 99 | 101 / **100** / 99 |
| scatter | 100 / **92** / – | 98 / **93** / 97 | 98 / **97** / 98 |
| write + completion, one sender | 121 / **107** / – | 120 / **106** / 109 | 130 / **109** / 111 |
| write + completion, mixed senders | 146 / **130** / – | 151 / **130** / 132 | 166 / **135** / 137 |

Largest VC1 E1 (channel 0 traffic):

| path | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| write 16 B | 85 / **85** / 86 | 85 / **85** / 86 | 100 / **94** / 94 |
| forward only | 121 / **138** / 132 | 120 / **138** / 134 | 133 / **127** / 132 |
| atomic | 100 / **103** / 104 | 100 / **102** / 103 | 118 / **112** / 112 |
| fused | 123 / **124** / 125 | 123 / **124** / 125 | 139 / **133** / 133 |
| scatter | 119 / **120** / 121 | 120 / **121** / 123 | 137 / **131** / 131 |
| write + completion, one sender | 150 / **134** / 135 | 151 / **136** / 137 | 169 / **146** / 146 |
| write + completion, mixed senders | 177 / **160** / 161 | 182 / **160** / 161 | 203 / **173** / 173 |

Reading:

- **-Os** is shorter than v3sb on almost every path. Per-packet paths are 1–7 shorter, completions 15–30 shorter per
  batch. The spread between levels shrinks: the W write goes from 57/58/65 to 62/63/61, and the VC1 write from
  85/85/100 to 85/85/94.
- **-O3 and -O2, local delivery:**
  - Amortized with completions, W and N/S are about even with v3sb: W write with one sender 72.5 vs 73 per packet,
    with mixed senders 79.3 vs 79.5.
  - 1D and the VC1 router are shorter: 1D local paths are −4 to −8 instructions per packet.
- **-O3 and -O2, forward paths are longer on multi-direction E1s:**
  - N/S is +10, +6 at -O2. The VC1 router is +17, +18 at -O2; it is −6 at -Os, the production level for VC1.
  - Three causes:
    - the function boundary: the downstream array's base is a spilled pointer, where v3sb addressed it from the stack
      pointer, and constants are rematerialized rather than reloaded;
    - the one-test change;
    - layout.
  - Counting rematerialized `lui`/`addi` constants as instructions overstates the cycle cost, because a v3sb reload is
    one load.
  - A draft with the forward test only in admission had N/S 109 and VC1 126 at -O3, but W write 66 and -Os worse on
    most paths, so I did not use it.
- **What still depends on the level**, all allocation and layout: the pass counter and route offsets spilled at -Os,
  constants held in registers at -O3 and rematerialized or spilled elsewhere, and taken jumps from block placement.
  None of these is a transformation the source can stop depending on without steering the compiler.

**Forwarded fused gap (+5.6 at -O3):** not closed.

- E1's fused paths do not explain it: local fused delivery is 98–101 against v3a's 98, and the forward is the same as
  for writes.
- v3t shortens what the FWD destination does per batch: the mixed-sender completion is −14 at -O3 and −27 at -Os.
  Hardware will show whether that is where the gap is.

### 11.5 Builds, size, VC2, probes

| Build | v3t | v3tb |
| --- | --- | --- |
| -O3 / -O2 | 536/536 | 536/536 |
| -Os | 440/440 + 96/96 VC1 | same |
| PT_LOAD vs v3sb | E0 identical on every config at all levels (208, 208, 160 + 48); E1 changed exactly in the 128 program configs | same |
| VC2 (`coverage.py build --cases bh_2d_vc2`) | 56/56 at -O3, -O2 and -Os | 56/56 at each |

**VC1 code size** (largest E0 + largest E1; the limit is 26,624 B):

| Level | v3sb | v3t | v3tb |
| --- | --- | --- | --- |
| -O3 | 13,120 + 12,240 = 25,360 | 13,120 + 11,980 = **25,100** | 13,120 + 11,736 = 24,856 |
| -O2 | 12,196 + 12,048 = 24,244 | 12,196 + 11,768 = 23,964 | 12,196 + 11,636 = 23,832 |
| -Os | 10,792 + 11,072 = 21,864 | 10,792 + 10,868 = 21,660 | 10,792 + 10,784 = 21,576 |

The crossover E1 grows from 9,592 to 9,900 at -O3.

**Probes** (`perturb.py`, -O3, all 536 configs):

- E0 is identical between v3sb and v3t with the same probe applied: 208/208 for hook and for statefield.
- statefield leaves the 128 program E1s unchanged, as on v3sb.
- With hook, v3t's paths cost 1–4 more instructions than without it; v3sb's cost 3–5 more.

### 11.6 Risks and what to watch

- **On hardware:**
  - -Os rows: E1 at -Os carried the whole remaining -Os penalty. Paths are shorter now, and completions much shorter.
  - -O3 and -O2 forwarding rows through N/S and VC1 routers: the forward path is longer.
  - FWD_PureFused4352.
  - The completion fast path on every E1-bound row.
  - v3tb against v3t isolates the function boundary.
- **Correctness of the new code:**
  - Exactness of the completion fast path: when all four records have equal read bits, the newest reading slot is the
    batch's last. It is the same slot the general search finds.
  - 1D `NOOP` still does nothing.
  - Sparse write-and-forward keeps its order: deliver, then advance, then forward.
  - `local_delivery_shared` passes the deferred-atomic state by value and returns it. Passing a pointer would have made
    the channel state escape.
- **VC1 credit path** (for the hang investigation): I found nothing that could over-fill a 2-slot VC1 sender channel or
  lose a credit.
  - A forward needs a known-free count above zero. The count is a lower bound of the local copy register, which only
    this ERISC decrements, once per forward.
  - On crossover routers, channels 0 and 1 share one cursor array by reference.
  - One thing for anyone touching this code: copying the cursors per channel (an idea I measured and dropped) would
    give each copy its own free count on crossover routers and over-fill the downstream channel.
- **Coverage** (`coverage.py build --name v3tall`, every case, all three levels): the same results as v3sb, except that
  VC2 now builds.
  - Every case builds except three that also fail on v3sb, outside E1: `*_nofuse` (ct_args static_assert and the
    generic loop) and `wh_vc1` at -O3 and -O2 (link).
  - `bh_2d_vc2` builds 56/56 at each level.

## 12. v3v: N/S at -Os, completion batches and local writes (on v3tk)

Variants:

- `fabric_exp/variants/v3v/`: changes 1 to 3 below. All changes are in `fabric_erisc_router_v3_receiver_erisc.hpp`; the
  .cpp, the shared headers and E0 are v3tk's. Since §12.9 its local writes go through the NoC API.
- `fabric_exp/variants/v3v_raw/`: v3v as first delivered and run on hardware (p14v), with the local write state written
  as raw command-buffer registers.
- `fabric_exp/variants/v3vb/`: a control (raw form). It is v3v_raw with NOC_CTRL written by every local write, as in
  v3tk (`local_write_ctrl_preset = false`), to measure that change alone.
- After the hardware (§12.8 to §12.12), all on the NoC API form: `v3vc`, `v3vd`, `v3ve` (controls without change 1),
  `v3v2` (v3v with the decode fixed) and `v3v3` (v3v2 with a different cut remainder). Then `v3v4` (v3v2 with an
  8-pass partial-batch hold, §12.13; the current best) and `v3v5` (v3v4 with the cut in 1D only).
- DIAGNOSTIC ONLY, never for shipping (§12.15): `v3vc_d8`, `v3vc_d20`, `v3vc_d50`, `v3vc_d100`, `v3tk_d50` (E1 waits
  after each pass without progress).

Scratch: `/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v/` (not durable).
Tools there: `rxscen.py` (v3t's, with `DECODE=x|y` to fix which route byte decides and `ALIGN=al|mis` to fix whether a
completion batch straddles two record words), `stab5.sh` (instructions / taken / loads / estimated load-use stall per
path), `score.py` (deltas between two tables), `ringscen.py` (the generic 1D receiver step with first-level ack),
`scen_a_comp.py` (v3a's write and completion paths), `stall.py`, `stabcheck.sh` (the ordinary-edit check).

### 12.1 Gaps (hardware, cycles/packet; v3t = p12t*, v3a -O3 = s79m)

| Row | v3t -O3 / -O2 / -Os | v3a -O3 / -O2 / -Os |
| --- | --- | --- |
| N/S write 16 B | 117.2 / 116.1 / 123.1 | 123.9 / 129.3 / 156.2 |
| N/S fused 4352 | 162.0 / 161.4 / 173.2 | 165.2 / 171.9 / 200.5 |
| N/S scatter 4352 | 154.8 / 155.6 / 166.7 | 176.1 / 181.8 / 208.4 |
| N/S atomic | 127.3 / 128.7 / 134.4 | 138.9 / 145.9 / 172.4 |
| Half-ring multicast write / scatter (1D) | 226.2 / 226.7 / 240.7; 278.4 / 277.9 / 293.3 | 227.5 / 225.8 / 241.3; 279.0 / 278.2 / 293.0 |
| Forwarded fused 4352 | 159.8 / 159.8 / 159.6 | 155.2 / 159.5 / 189.9 |
| E/W fused 4352 | 155.4 / 155.5 / 154.4 | 155.2 / 159.6 / 190.0 |

### 12.2 Causes

**N/S at -Os.** N/S rows reach their destination on the Y leg. The packet's route is built by
`widen_2d_route_to_chip`: the Y byte at the destination's row is STOP (0) and LOCAL_DELIVER is in the X byte, so the
destination's N/S router decodes the X byte. That is the path that matters for these rows. §11.4's N/S counts took the
other branch (the Y byte decides), which is shorter. On the X path, v3t's N/S write is 66 / 65 / 72 instructions at
-O3 / -O2 / -Os, against W's flat 62 / 63 / 61. The -Os listing (N/S write, v3t) differs from -O2 in five places:

| Item | -O2 | -Os | Instructions |
| --- | --- | --- | ---: |
| Route bytes | `add; add; lbu 48(a); lbu 50(b)` | `lw y,12(sp)` (spilled coordinate), `addi rb,slot,48`, `addi x2,x,2`, `add; add; lbu 0; lbu 0` | +3, one of them a load |
| Deliver-only packet | forward bits tested once (jump threading) | tested again after the deferred-atomic block | +1 |
| Slot record store | `add a,idx,sp; sb 24(a)` | `addi a,idx,64; addi b,sp,16; add; sb -56(a)` | +2 |
| Constants | 4 held in a register | `li 4` before the batch compare | +1 |
| Total | 65 | 72 | +7 |

The route-byte item is the decode itself. `Routing2DCodec::decode_action` reads `route_buffer[y]` and
`route_buffer[MESH_Y_SIZE + x]` through one shared `const volatile uint8_t*`. At -O2 and -O3 GCC folds the route
buffer's offset (48) and MESH_Y_SIZE into the two loads. At -Os it keeps the shared pointer in a register and adds
MESH_Y_SIZE to the coordinate per packet. With the bytes read through a non-volatile header pointer instead, -Os folds
both offsets (measured below; with a volatile header pointer and the same indexing, -Os still computes `x + 2`). W
reads one byte, so the decode is already flat there. The spill, the re-test, the record store and the constant are
register allocation, block layout and frame-offset folding; §12.6 lists what I tried for them.

**Half-ring multicast (1D) at -Os.** The ring routers' E1 does not run the E1 program. Ring builds use first-level ack
and deadlock avoidance, which `receiver_erisc::runs` excludes, so their E1 runs kernel_main's generic loop. That is v3a's
code, the same in every design, which matches the hardware: at each level the half-ring rows of v3a, v3sb and v3t are
within 1.5 of each other (write) and 0.7 (scatter). The 1D multicast code in the E1 program is not involved.

- In the generic step with first-level ack, a WRITE_AND_FORWARD packet is an ack frame, the forward (an out-of-line
  `forward_payload_to_downstream_edm` call), the local write through the send-type jump table and one completion frame.
- Per packet that is 200 / 197 instructions at -O2 / -Os (`ringscen.py`, config 7572880629736378935), with equal load
  counts (42 / 41). The -Os cost is in when the loads are used, not in how many instructions run.
- -Os does not schedule before register allocation (`-fschedule-insns` is off at -Os), and it keeps more loop state in
  stack slots (the loop counter and the written-slot counter). The estimated load-use stall (`stall.py`) is 88 at -O2
  and 101 at -Os for an assumed 5-cycle load latency (+10 to +16 for 3 to 8 cycles), against 3 fewer instructions at
  -Os and the measured +14.
- v3v does not change this path: the 200 E1 ELFs outside the program are PT_LOAD-identical to v3tk's.

**Forwarded fused 4352.** v3a's E1 at -O3 and at -O2 (W config) has the same per-packet paths: write 65, scatter 105,
atomic 75, fused 98, deferred fused 99 at both levels. The one difference is the completion: v3a credits a batch slot by
slot in a loop, which -O3 unrolls. A batch of four credits costs 140 instructions with a write at -O3 and 156 at -O2
(`scen_a_comp.py`), 4 per packet. That matches v3a's -O2 cost on both fused rows (+4.4 E/W, +4.3 forwarded). v3t
already writes the four credits out at every level. Two things cost v3t on these rows:

- **Record words.** The batch's slot records are read as one word when the batch starts on a word boundary, and as two
  words plus a funnel shift otherwise (+10 instructions per batch, +1 load). After any partial batch on an idle pass,
  every later batch straddles two words until the next idle period moves the oldest slot again, so a run's completion
  cost depends on where its warm-up left the ring (3 alignments of 4 straddle). This is a plausible source of the bimodal
  N/S chunk-8 rows in RESULTS §96.
- **Local writes.** The fused write rewrites NOC_CTRL with the same value every time (3 instructions).

Per forwarded fused packet at -O3 (fused path + a quarter of the mixed-sender completion beyond a write): v3t 118.5
aligned, 121.0 straddling; v3a 116.75. With one sender (E/W fused), v3t 112 / 114.5 and v3a 116.75 (v3a credits per slot
either way). So v3t's lead on E/W fused is smaller than its completion advantage suggests, and with mixed senders it is
behind.

**E/W atomic at -O3 (126.6 vs v3sb 124.5).** v3t's W atomic path is 76 against v3sb's 72. The difference is
constants: v3sb inlined the loop into kernel_main and reloaded constants from stack slots (one more load), while v3t
rematerializes them (`lui`/`addi`), and one `j` comes from block layout. That is allocation, so I did not pursue it.

### 12.3 Changes (v3v)

1. **Completion batches end at the end of a record word** (`send_completions`, `SlotRecords::to_word_end`). On rings
   of a multiple of 4 slots with batches of 4, a batch is cut at the end of its oldest slot's record word, so its
   records are always one aligned word load. After a cut, the following batches are whole words again.
   - While packets arrive (uncompleted > 4), the cut batch is 4 - (base mod 4) slots. It is credited exactly like any
     batch: the newest slot is still never in it, and the check uses the same per-buffer newest-reading-slot logic.
   - Behavior difference, idle passes only: a batch that would straddle a word goes out in two parts, and the second
     part waits like any partial batch (at once if 4 slots wait, else after 4 more idle passes in 2D). That is at most
     4 idle passes of extra credit latency for up to 3 slots, at the end of a burst.
   - Rings of 2 to 7 slots (batch 1) and rings that are not a multiple of 4 keep `four_from`.
2. **Local writes set NOC_CTRL once** (`local_write_init_state`, `local_write_with_state`; in v3v_raw
   `set_local_write_ctrl`, `local_write`). Every local write of E1 (plain, fused, scatter chunks) uses
   `local_chip_data_cmd_buf` with the same NOC_CTRL value. run_main_loop programs it once before the loop, and each
   write then writes only the transaction ID, source, destination, coordinate, length and command (6 stores instead of
   7, and no constant). §12.9 gives the NoC API calls.
   - The out-of-line deliveries (sparse multicast through `execute_chip_unicast_to_local_chip_impl`) write the same
     value.
   - `static_assert(!runs || local_write_cmd_buf_is_exclusive)`: on this program, no forward command buffer and not
     `write_at_cmd_buf` is the local write command buffer. In the logged configs these are 0 (local writes), 2 and 1
     (forward data and sync) and 3 (atomics, inline writes).
   - With DPRINT or the watcher, whose NoC traffic shares command buffers, the preset is off and every write goes
     through `ncrisc_noc_fast_write`.
   - The forwarding command buffers already keep their state across the loop this way (`setup_downstream_cmd_bufs`).
3. **2D decode reads the route bytes as header fields** (`decode_action`).
   - It applies the same rule as the codec: the X byte on E/W-facing routers; otherwise the Y byte when it is not 0,
     else the X byte; landing intercept first on an intermesh ingress channel.
   - It reads `packet_header->route_buffer[...]` directly, without the codec's shared volatile pointer, like the other
     header fields this ERISC reads (command fields, `dst_start_mesh_id`). The slot address plus the coordinate then
     takes the byte's fixed offset in the load at every level.
   - At -Os GCC also turns the X load into a branch when the Y byte decides.
4. **Comments.** The run_main_loop, `termination_requested`/`end_round` and `intermesh_egress_slot` comments still
   gave the withdrawn multiply rationale. They now give RESULTS §97's: per-round and rare work out of line.

### 12.4 Per-path counts (instructions; v3t / **v3v**)

Method as in §10.4 and §11.4. For v3t's completions, the straddling batch is in parentheses (`ALIGN=mis`); v3v's
batches never straddle while packets arrive. The paths are the walker's shortest, so the N/S table takes the Y branch.
The N/S destination table below it fixes the X branch (`DECODE=x`), which is the N/S rows' path.

Correction (§12.10): with change 3, the Y branch skips the X byte's load, so the "N/S, shortest" and VC1 tables below
show v3v 2 to 3 instructions shorter than the packets that decide on the X byte. And instruction counts miss what
change 3 costs on the X branch: §12.10 gives the estimated load-use stall and the X-branch VC1 counts.

W (mid-W):

| path | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| write 16 B | 62 / **59** | 63 / **60** | 61 / **58** |
| forward only | 72 / **72** | 72 / **72** | 68 / **68** |
| atomic | 76 / **76** | 77 / **77** | 74 / **75** |
| fused | 101 / **98** | 101 / **98** | 99 / **97** |
| fused, flush deferred | 100 / **97** | 101 / **98** | 99 / **96** |
| scatter | 96 / **90** | 99 / **93** | 97 / **91** |
| write + completion, one sender | 106 (116) / **101** | 109 (119) / **103** | 114 (124) / **108** |
| write + completion, mixed senders | 132 (142) / **125** | 134 (144) / **130** | 140 (150) / **134** |

N/S, destination (the X byte decides: N/S rows):

| path | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| write 16 B | 66 / **62** | 65 / **61** | 72 / **64** |
| atomic | 82 / **81** | 81 / **80** | 87 / **82** |
| fused | 105 / **101** | 104 / **100** | 113 / **104** |
| fused, flush deferred | 104 / **100** | 103 / **99** | 112 / **103** |
| scatter | 100 / **93** | 101 / **93** | 110 / **98** |
| write + completion, one sender | 110 (120) / **103** | 110 (120) / **104** | 122 (132) / **115** |
| write + completion, mixed senders | 136 (146) / **128** | 136 (146) / **131** | 148 (158) / **140** |

N/S, shortest (the Y byte decides: transit on the Y leg):

| path | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| write 16 B | 66 / **60** | 65 / **59** | 72 / **62** |
| forward only | 113 / **110** | 112 / **109** | 112 / **105** |
| atomic | 82 / **79** | 81 / **78** | 87 / **80** |
| fused | 105 / **99** | 104 / **98** | 113 / **102** |
| fused, flush deferred | 104 / **98** | 103 / **97** | 112 / **101** |
| scatter | 100 / **91** | 101 / **91** | 110 / **96** |
| write + completion, one sender | 110 (120) / **101** | 110 (120) / **102** | 122 (132) / **113** |
| write + completion, mixed senders | 136 (146) / **126** | 136 (146) / **129** | 148 (158) / **138** |

1D (line):

| path | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| write 16 B | 58 / **54** | 58 / **54** | 62 / **59** |
| forward only | 72 / **70** | 72 / **70** | 72 / **72** |
| atomic | 72 / **70** | 72 / **70** | 76 / **76** |
| fused | 97 / **92** | 97 / **92** | 100 / **97** |
| fused, flush deferred | 102 / **96** | 102 / **97** | 103 / **100** |
| scatter | 92 / **84** | 93 / **85** | 97 / **91** |
| write + completion, one sender | 107 (115) / **94** | 106 (114) / **95** | 109 (119) / **103** |
| write + completion, mixed senders | 130 (138) / **119** | 130 (138) / **123** | 135 (145) / **128** |

Largest VC1 E1 (channel 0 traffic; -Os built from the -O2 command):

| path | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| write 16 B | 85 / **76** | 85 / **76** | 94 / **86** |
| forward only | 138 / **130** | 138 / **132** | 127 / **122** |
| atomic | 103 / **95** | 102 / **95** | 112 / **106** |
| fused | 124 / **114** | 124 / **114** | 133 / **124** |
| fused, flush deferred | 129 / **119** | 129 / **119** | 137 / **128** |
| scatter | 120 / **107** | 121 / **109** | 131 / **119** |
| write + completion, one sender | 134 (142) / **122** | 136 (144) / **123** | 146 (154) / **135** |
| write + completion, mixed senders | 160 (168) / **146** | 160 (168) / **147** | 173 (181) / **161** |

v3vb is v3v plus 3 per local write at every level and config: write, fused and the completions (which include a
write) +3, scatter +6, the rest equal, within 1 at a few cells (allocation).

Reading:

- **N/S at -Os** (destination path): the -Os excess over -O2 is write +3 (was +7), atomic +2 (+6), fused +4 (+9),
  scatter +5 (+9). The decode change removes 4 to 5 of it at -Os (1 at -O2): with changes 1 and 2 only, the -Os
  excess was still +7 / +6 / +9 / +10.
  - What is left: the spilled coordinate (one stack load), the forward-bit re-test, the record store's frame
    arithmetic and `li 4`.
  - Every level also gains the structural changes: write -4 to -8 and scatter -7 to -12 on the N/S destination path.
- **Write + completion paths:** -4 to -14 against v3t's one-word case and -14 to -22 against the straddling case, at
  every level and config (3 of it is the write's NOC_CTRL).
- **Forwarded fused**, per packet at -O3 (fused + a quarter of the mixed completion beyond a write): v3v 114.5,
  against v3t 118.5 (121.0 straddling) and v3a 116.75. At -O2: 115.5 against 118.75 (121.25) and v3a 120.75.
  - v3v still has more taken branches per packet than v3a's -O3 path (about 8.5 against 4.75). Layout decides these
    and the source does not, so the hardware row is the test.
- **W atomic:** 76 / 77 / 75, unchanged (§12.2; +1 at -Os, allocation). The forward-only path is unchanged on W,
  -2 on 1D at -O3 and -O2 (allocation), and 3 to 8 shorter on N/S and VC1 (the decode).
- **Half-ring:** unchanged (§12.2).

### 12.5 Builds, size, stability, coverage

| Check | Result |
| --- | --- |
| Offline builds, every logged config | -O3 536/536, -O2 536/536, -Os 534/536. The 2 -Os failures are the two garbled logged VC1 commands, which fail the same way for v3tk; the 96 VC1 configs build from their -O2 commands at -Os (96/96). v3vb the same |
| E0 vs v3tk (PT_LOAD, `loadcmp.py`) | identical on every config: 208/208 at -O3 and -O2, 207/207 at -Os, 48/48 VC1 at -Os from the -O2 commands. v3vb the same |
| E1 vs v3tk | changed in exactly the 128 program configs (127 at -Os, plus the 48 VC1 E1s from the -O2 commands); the 200 others (ring, speedy, single-ERISC) identical |
| Ordinary edits (`jitter_edits.py` coldmul, init, templates, os on v3v; four hot configs at -O3/-O2/-Os; `stabcheck.sh`) | run_main_loop never DIFF: coldmul moved everywhere; init moved (VC1 same); templates same everywhere; os moved at -O3/-O2, same at -Os |
| VC1 code size (largest E0 + largest E1) | -O3 13,120 + 11,912 = **25,032** (v3tk 25,100; limit 26,624); -O2 12,196 + 11,736 = 23,932; -Os 10,792 + 10,764 = 21,556. v3vb -O3 25,092 |
| Coverage (`coverage.py build`, every case, -O3; names v3vH, v3vbH, v3tk) | identical to v3tk case by case over 41 cases; the only failures are the inherited `*_nofuse` (4 cases) and `wh_vc1` (15/20). v3vb the same. `bh_2d_vc2` 56/56 at -O3, -O2 and -Os |

### 12.6 Tried and not used (offline counts only)

- **Pending atomic resolved before admission**, so that a deliver-only packet tests the forward bits once at every
  level.
  - Deferral needs a fixed transaction ID per slot. That holds on every logged and coverage config (probe:
    static_assert, 0 hits).
  - -Os: N/S write -2, 1D -2.
  - -O3: W write +4 and N/S +3. The write arm was merged into the fused arm's write and placed behind two taken jumps,
    and the pending block was laid out inline.
- **Route byte indices precomputed per router** (slot + index): both indices were spilled to the stack at every level
  (N/S -O3 write +2).
- **X byte read only when the Y byte is 0**: -5 on the Y branch at -Os, but +2 at -O3 and -O2 on the destination path
  (load, branch, load), so worse for the N/S rows.
- **Row-pointer or volatile direct indexing** for the decode: -2 to -4 at -Os, W -Os +1. Superseded by change 3.
- **Completion split into progress and idle forms**, with idle batches unchanged: 2 to 5 fewer instructions per batch
  than change 1, but the per-packet paths moved by +1 to +3 at some levels from allocation. Change 1 is the simpler
  form.
- **Forward inlined into the generic 1D step** (ring E1, `forward_payload_to_downstream_edm`):
  - -O2 -22 instructions, -Os -11.
  - The estimated -Os load-use stall grew by 9, so the -Os gap does not close.

### 12.7 Risks and what to watch

- **Correctness of the new code:**
  - Preset NOC_CTRL: nothing else on E1 may use `local_chip_data_cmd_buf` during the loop. The static_assert covers
    the forward command buffers and `write_at_cmd_buf`. By reading: the context-switch handshake (E1 is never its
    master), the heartbeat and telemetry issue no NoC commands. Run the RELAXED and STRICT validation, and VC1 (whose
    channel 1 delivers through the out-of-line impl, which writes NOC_CTRL itself).
  - Cut batches: the exactness argument of §11.6 holds per batch, and a cut batch is a partial batch. The newest slot
    is still excluded while packets arrive (batch ≤ 4 < uncompleted).
  - Decode: the same bytes and rule as the codec, read non-volatile after the packet is known present, as the rest of
    the header is. On intermesh ingress channels the landing route is written first, to the same bytes.
  - Unchanged: the deferred atomic stays staged in `write_at_cmd_buf` (local writes never use it); the sparse
    write-and-forward order; the VC1 credit path; 1D NOOP; the intermesh egress bypass rule.
- **Hardware rows:**
  - N/S at -Os (and all levels): NS_PureWrite16/256/1024, NS_PureScatter4352, NS_PureFused4352 / NS_Fused_NoFlush,
    NS_PureAtomic, Mesh1HopNS write and scatter, NS flushmix rows.
  - Forwarded and E/W fused: FWD_PureFused4352 (v3t 159.8, v3a -O3 155.2), EW_PureFused4352, FWD_MixChunk.
  - Everything E1-bound gains the completion change. Check the scatter rows (−6 per packet from the preset alone) and
    the 1D Linear rows.
  - Run-to-run spread: the bimodal N/S chunk-8 flushmix rows of §96 should become unimodal if batch alignment was their
    cause.
  - Half-ring multicast: expected unchanged (the ring E1 is not touched).
  - v3vb against v3v isolates the preset NOC_CTRL; the decode and completion changes are in both.
- **Not closed:**
  - The N/S -Os remainder (+2 to +5 instructions: spill, re-test, record address, constant).
  - The half-ring -Os gap: it needs the ring E1 on a program with explicit per-packet state (first-level ack and the
    deadlock-avoidance credit write in the E1 program), or a reordered generic ring step.
  - E/W atomic vs v3sb (allocation).

### 12.8 Hardware: jobs 287–289 (p14v = v3v_raw, p14vb = v3vb; v3t = p12t / p13k)

All validation passed (RELAXED and STRICT); VC1 34/34 at -O3, -O2 and -Os for v3v. Suite means against v3a -O3
(cycles/packet, lower is better):

| | 4352 | mix | flushmix | 1D |
| --- | ---: | ---: | ---: | ---: |
| v3t -O3 | −8.45 | −4.84 | −3.86 | −6.81 |
| v3v -O3 | −9.83 | −4.85 | −3.85 | −10.52 |
| v3vb -O3 | −7.55 | −3.28 | −2.49 | −8.99 |
| v3t -O2 | −6.60 | −4.46 | −3.87 | −6.26 |
| v3v -O2 | −8.57 | −4.57 | −4.18 | −10.12 |
| v3t -Os | −5.48 | −2.08 | −1.29 | −1.40 |
| v3v -Os | −8.27 | −4.01 | −2.94 | −3.99 |

Gains, from the preset local write state (v3vb loses them):

| Row | v3t | v3v | v3vb |
| --- | ---: | ---: | ---: |
| E/W scatter 4352 | 151.1 | 144.5 | 149.5 |
| Linear1HopEW scatter | 154.3 | 142.5 | 147.9 |
| FWD_PureFused4352 | 159.8 | 157.9 (-O2 158.0, -Os 157.0) | 160.7 |
| EW_PureFused4352 | 155.4 | 153.9 | |
| N/S scatter, -Os | 166.6 | 159.8 | |
| N/S fused, -Os | 173.3 | 166.2 | |
| N/S write 16 B, -Os | 123.1 | 120.7 | |

Half-ring multicast unchanged, as expected (§12.2).

Regressions, in both v3v and v3vb at every level (so from change 1, change 3 or what they move):

| Row | v3t → v3v | Δ |
| --- | --- | ---: |
| Mesh1HopEW write | 128.1 → 130.4 | +2.3 |
| Mesh1HopNS write | 128.3 → 131.0 | +2.7 |
| Mesh3HopEW write | 144.0 → 146.3 | +2.3 |
| EW_PureWrite4352 | 128.4 → 130.5 | +2.1 |
| NS_PureWrite4352 | 128.5 → 131.2 | +2.7 |
| EW_MixSize4 | 117.8 → 120.6 | +2.8 |
| NS_MixSize4 | 120.3 → 124.8 | +4.5 |
| NS_PureAtomic | 127.4 → 134.1 (-O2 128.8 → 134.7) | +6.7 (+5.9) |
| NS_Chunk8Fused_NoFlush | 131.0 → 134.5 | +3.5 |
| Linear3HopEW write, -Os | 148.8 → 152.2 | +3.4 |

The per-packet instruction counts (§12.4) predicted N/S atomic 82 → 81, so the cost is not in instruction counts.

### 12.9 NoC programming

Rule (from 2026-10-25): E1's NoC commands are programmed through the NoC API. No `NOC_CMD_BUF_WRITE_REG` /
`NOC_CMD_BUF_READ_REG` in E1's code. Where the API lacks a call, it is proposed as an API addition, not hand-rolled.
v3v, v3v2, v3v3, v3vc, v3vd and v3ve follow it. The only `NOC_CMD_BUF_WRITE_REG` left in these dirs is the DRISC-only
block of `fabric_edm_packet_transmission.hpp` (`COMPILE_FOR_DRISC`), which no ERISC compiles. v3v_raw and v3vb keep
the raw form they were measured with.

What the E1 program calls:
- **Local writes**, on `local_chip_data_cmd_buf`:
  - `local_write_init_state()`, once before the loop: `noc_cmd_buf_ready`, then
    `noc_write_init_state<cmd_buf, CQ_NOC_mkp>(noc, forward_and_local_write_noc_vc)`, which writes NOC_CTRL.
  - `local_write_with_state<COUNTED>(src, dst, size, trid)`, per write: `ncrisc_noc_set_transaction_id` (it waits
    for the command buffer), then `noc_write_with_state<noc_mode, cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_wait,
    COUNTED, /*posted=*/false>`.
  - The helpers hold the invariant: `local_write_cmd_buf_is_exclusive` (static_assert: no forward command buffer and
    not `write_at_cmd_buf` is the local one). They also hold the DPRINT / watcher fallback (`local_write_state_preset =
    false`): `noc_cmd_buf_ready` and `ncrisc_noc_fast_write` per write.
  - The out-of-line deliveries (`execute_chip_unicast_to_local_chip_impl`) are unchanged:
    `noc_async_write_one_packet_with_trid` and `ncrisc_noc_fast_write`, which program the same state.
- **Forwards**: `noc_async_write_one_packet_with_trid_set_state` and `noc_inline_dw_write_set_state` once per
  downstream (`setup_downstream_cmd_bufs`), then `send_chunk_from_address_with_trid` and
  `noc_inline_dw_write_with_state` / `noc_inline_dw_write` per packet.
- **Atomics**: `noc_semaphore_inc`, `noc_semaphore_inc_set_state` / `noc_semaphore_inc_with_state` (the deferred
  flushed atomic, staged on `write_at_cmd_buf`), `ncrisc_noc_nonposted_writes_flushed`,
  `ncrisc_noc_nonposted_write_with_transaction_id_flushed`.
- **API form against the raw form** (v3v against v3v_raw, the 96 cells of §12.4): instruction counts identical except
  VC1 -O3 fused +1, VC1 -Os atomic −1 and fused +1. Estimated stall −3 to +2 per path.
- **Status reads that stay raw**: unchanged from v3tk, and not command-buffer accesses.
  - The completion check reads NIU_MST_WRITE_REQS_OUTGOING_ID(trid) through a pointer (`outgoing[trid]`), one trid
    or two read together. The API's `ncrisc_noc_nonposted_write_with_transaction_id_sent` compares each read with 0.
    With it, the loop's code changed (tried), so it is not used.
  - The deferral bypass reads NIU_MST_WR_ACK_RECEIVED and NIU_MST_REQS_OUTSTANDING_ID(trid) as counts with
    `NOC_STATUS_READ_REG`, to prove g_out == s_out + p_out. The API has only boolean forms.
  - Proposed NoC API additions: count accessors `ncrisc_noc_nonposted_writes_outgoing_with_transaction_id(noc, trid)`,
    `ncrisc_noc_nonposted_writes_outstanding_with_transaction_id(noc, trid)` and
    `ncrisc_noc_nonposted_writes_ack_received(noc)`. Each returns the register value, so a caller can read several
    status registers back to back and test them together.

### 12.10 Regression analysis

**Change 3 on N/S routers: the X byte's load waits for the Y byte's test.** N/S destination path (Y byte 0, the X
byte decides), atomic, -O3:

```
v3t (codec, volatile)              v3v (plain header fields)
add   a2,s2,s8                     add   a5,s2,s7
add   a5,s2,s7                     lbu   a5,48(a5)    # Y byte
lbu   a4,48(a2)    # Y byte        lw / srli / zext / srli
lbu   a5,50(a5)    # X byte        bnez  a5,...       # Y != 0
srli / zext / zext / srli          add   a5,s2,s9
beqz  a4,...       # Y == 0        lbu   a5,50(a5)    # X byte, issued only now
andi  t1,a5,3                      andi  t1,a5,3      # used by the next instruction
```

- With plain fields GCC may place the X load behind the test that decides whether its value is used. It did so at
  every level. On the N/S rows' path both loads then wait in turn.
- Change 3 alone on the N/S destination path (v3vd against v3vc, §12.11; `stall.py`, 5-cycle loads): 1 to 2 fewer
  instructions and +5 estimated load-use stall per path at -O3 and -O2; 5 to 7 fewer instructions and +3 stall at
  -Os. Against v3t, v3v's atomic went from 82+18 to 81+23.
- E/W routers read one byte; there the plain field is scheduled earlier (W stall −2 at -O3 and -O2).
- The largest VC1 E1 also decodes Y first: same effect on its X branch (counts in §12.11).

Fit to the rows:
- NS_PureAtomic +6.7 / +5.9 (-O2). An atomic has no payload, so the per-packet path is what the row measures.
  EW_PureAtomic is not among the regressions, and change 1 acts the same on both directions.
- N/S rows regress more than their E/W twins: MixSize4 +4.5 against +2.8, PureWrite4352 +2.7 against +2.1, Mesh1Hop
  +2.7 against +2.3. That excess (0.4 to 1.7) is N/S-only.
- NS_Chunk8Fused_NoFlush +3.5: the fused path's stall is +5.
- Confidence: high for NS_PureAtomic, medium for the N/S excess. v3vd against v3vc measures it; v3ve and v3v2 carry
  the fix.

**Change 1 (batch cut) and credit timing.** A batch is cut only when the oldest uncredited slot is in the middle of a
record word, which needs an earlier partial batch.
- In 2D a partial batch goes out after 4 passes without progress with 1 to 3 slots waiting.
- An idle pass of v3t's W loop at -O3 is 14 instructions: one stream-register read, 3 taken branches. Read from the
  listing: test avail, read the packets-received register, test the pending atomic, completion test, back edge.
  Estimated 20 to 30 cycles.
- A steady 4352 B stream runs about 128 cycles per packet. E1's write path is about 75 cycles (62 + 15 stall), which
  leaves about 2 idle passes per packet. So batches of 4 go out on an idle pass or with the next packet, word-aligned.
  Partial batches, and so cuts, come only at the start and end of a stream or after a gap of about 100 cycles. In that
  steady state v3t and v3v send the same batches.
- A pass-level model (`sim/credsim3.py` in the scratch dir) tests this. It has a closed loop of 16 sender slots, a
  credit frame of 50 cycles, +10 for a straddled batch, and a newest-slot write busy 85 cycles after issue. Policies:
  T = v3t, V = v3v, P = v3v3, G = cut only while more than 4 wait. Its absolute numbers are not calibrated; only the
  differences mean anything:

| idle pass, arrival jitter | cuts per 1000 packets (V) | cycles/packet T / V / P / G |
| --- | ---: | --- |
| 10 cycles, ±20, credit-bound | 70 | 133.61 / 133.93 / 133.87 / 133.65 |
| 10 cycles, ±20, link-bound | 69 | 125.91 / 126.24 / 126.25 / 125.87 |
| 15 cycles, ±20, credit-bound | 22 | 134.90 / 134.70 / 134.66 / 134.95 |
| 20 cycles, ±60 | 3 | 143.30 / 142.91 / 142.88 / 142.88 |
| 25 cycles, ±60 | 0.4 | 143.59 / 143.08 / 143.09 / 143.06 |

- Only with idle passes of 10 cycles or less does the cut cost anything: 11 to 14% more credit frames and +0.3
  cycles/packet. At the estimated 20 to 30 cycles, cuts are rare, and V is 0.4 to 0.5 faster than T (no straddled
  batches).
- The prompt remainder (P, v3v3) is no better than V when cuts do cost. In the model the slots after a cut wait for
  their newest slot's write, which is still in flight, so they gain little from going on the next pass. The cost is
  the extra frame, not the 4-pass wait.
- G (cut only while more than 4 wait) keeps T's idle passes and V's realigned batches. Its source forms changed the
  per-packet code, though (§12.12).
- 1D sends a partial batch at the first idle pass, so misaligned batches are common there, and change 1 matters more
  both ways. The gain shows on E1-bound 1D rows (v3vb, which has no preset: Linear1HopEW scatter −6.4, 1D suite −2.2).
  The single -Os link-bound row (Linear3HopEW +3.4) could be the cost, or the jitter band (§96: single rows move by
  up to ±6 when the loop only moves).

**Not explained offline: the common E/W part (+2.1 to +2.8).**
- Change 1's credit timing does not fit in steady state (above).
- Change 3's E/W part is one byte read, with the same instruction count and 2 less stall.
- Code movement (§96) is possible. It is less likely for a shift that shows at every level in two variants with
  different layouts.
- EW_MixSize4 (+2.8, the largest) is the row where mixed sizes give E1 idle gaps, so change 1 is most plausible there.
- The controls separate it: v3vd against v3v is change 1 alone; v3vd against v3vc is change 3 alone (E/W and N/S).

| Row | Δ | Likely cause | Confidence |
| --- | ---: | --- | --- |
| NS_PureAtomic | +6.7 (-O2 +5.9) | change 3: X load behind the Y test, +5 stall per packet | high |
| NS_MixSize4, NS_PureWrite4352, Mesh1HopNS | +4.5, +2.7, +2.7 | change 3 for the excess over the E/W twin (0.4 to 1.7); the rest as E/W | medium |
| NS_Chunk8Fused_NoFlush | +3.5 | change 3 (fused path +5 stall) | medium |
| EW_PureWrite4352, Mesh1HopEW, Mesh3HopEW | +2.1 to +2.3 | not found; change 1's timing does not fit in steady state | low |
| EW_MixSize4 | +2.8 | change 1 plausible (idle gaps, partial batches, cuts) | low |
| Linear3HopEW write, -Os | +3.4 | -Os only; no decode in 1D, so change 1 or code movement | low |

### 12.11 The controls, v3v2 and v3v3

All are on the NoC API form (§12.9), with every change in the receiver header:

| Dir | Change 2 (preset) | Change 3 (decode) | Change 1 (cut) | Measures |
| --- | --- | --- | --- | --- |
| `v3vc` | yes | no (codec) | no | the preset alone |
| `v3vd` | yes | plain fields | no | change 3 as run (with v3vc) |
| `v3ve` | yes | fixed | no | the fixed decode without the cut |
| `v3v` | yes | plain fields | yes | change 1 (with v3vd); the API port (with p14v) |
| `v3v2` | yes | fixed | yes | the proposal |
| `v3v3` | yes | fixed | yes, prompt remainder | the 4-pass-wait hypothesis (with v3v2, E/W rows only) |

- **Fixed decode** (v3v2, v3ve). E/W routers read the X byte as a plain header field, as v3v does. Routers that decode
  Y first (N/S, Z, intermesh ingress) read both bytes volatile and unconditionally, as the codec does, so the two
  loads issue together. Each byte is indexed in the header directly, which keeps the -Os gain of change 3 (no
  per-packet MESH_Y_SIZE add).
- **Prompt remainder** (v3v3). After a cut on a pass without progress (2D), the idle count is left one short of its
  limit, so the slots left behind go out on the next such pass instead of after 4. 1D code is identical to v3v2.

Per-path counts, instructions + estimated load-use stall (5-cycle loads). Completions as in §12.4 (`compu`: write and
a credit of 4 from one sender, aligned batch; `compm`: 4 credits written out).

N/S destination (the X byte decides):

| variant | level | write | atomic | fused | scatter | compu | compm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3t | -O3 | 66+13 | 82+18 | 105+21 | 100+17 | 110+20 | 136+26 |
| v3vc | -O3 | 63+11 | 82+18 | 102+21 | 94+17 | 107+18 | 133+24 |
| v3vd | -O3 | 62+16 | 80+23 | 101+26 | 93+22 | 107+23 | 132+29 |
| v3ve | -O3 | 63+11 | 81+18 | 102+21 | 94+17 | 108+18 | 133+24 |
| v3v | -O3 | 62+16 | 81+23 | 101+26 | 93+22 | 103+23 | 128+29 |
| **v3v2** | -O3 | 63+11 | 81+18 | 102+21 | 94+17 | 105+18 | 129+24 |
| v3v3 | -O3 | 64+14 | 82+21 | 102+24 | 95+20 | 111+19 | 133+22 |
| v3t | -O2 | 65+13 | 81+18 | 104+21 | 101+15 | 110+19 | 136+26 |
| v3vc | -O2 | 62+11 | 81+18 | 101+21 | 95+15 | 107+17 | 133+24 |
| v3vd | -O2 | 61+16 | 79+23 | 99+26 | 94+20 | 106+22 | 132+29 |
| v3ve | -O2 | 62+11 | 80+18 | 100+21 | 94+15 | 107+17 | 133+24 |
| v3v | -O2 | 61+16 | 80+23 | 100+26 | 93+20 | 104+22 | 131+28 |
| **v3v2** | -O2 | 62+11 | 81+18 | 101+21 | 94+15 | 105+17 | 132+23 |
| v3v3 | -O2 | 63+14 | 81+21 | 101+24 | 95+18 | 110+20 | 132+23 |
| v3t | -Os | 72+20 | 87+23 | 113+27 | 110+24 | 122+27 | 148+35 |
| v3vc | -Os | 69+20 | 87+23 | 111+28 | 104+24 | 119+27 | 145+35 |
| v3vd | -Os | 64+23 | 82+26 | 104+31 | 98+27 | 117+30 | 143+38 |
| v3ve | -Os | 66+19 | 84+22 | 106+27 | 100+23 | 119+26 | 145+34 |
| v3v | -Os | 64+23 | 82+26 | 104+31 | 98+27 | 115+31 | 140+40 |
| **v3v2** | -Os | 66+19 | 84+22 | 106+27 | 100+23 | 117+27 | 142+36 |
| v3v3 | -Os | 66+19 | 84+22 | 106+27 | 100+23 | 119+26 | 144+35 |

W (v3vd, v3ve, v3v and v3v2 have the same per-packet paths; they differ in the completion):

| variant | level | write | atomic | fused | scatter | compu | compm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3t | -O3 | 62+15 | 76+19 | 101+22 | 96+19 | 106+22 | 132+28 |
| v3vc | -O3 | 59+14 | 76+19 | 98+21 | 90+19 | 103+21 | 129+27 |
| v3vd, v3ve | -O3 | 59+12 | 76+17 | 98+19 | 90+17 | 103+19 | 129+25 |
| v3v, **v3v2** | -O3 | 59+12 | 76+17 | 98+19 | 90+17 | 101+19 | 125+25 |
| v3v3 | -O3 | 59+12 | 75+17 | 97+19 | 90+17 | 104+18 | 128+22 |
| v3t | -O2 | 63+15 | 77+17 | 101+21 | 99+16 | 109+21 | 134+28 |
| v3vc | -O2 | 60+14 | 77+17 | 98+21 | 93+16 | 106+20 | 131+27 |
| v3vd, v3ve | -O2 | 60+12 | 77+15 | 98+19 | 93+14 | 106+18 | 131+25 |
| v3v, **v3v2** | -O2 | 60+12 | 77+15 | 98+19 | 93+14 | 103+18 | 130+24 |
| v3v3 | -O2 | 60+12 | 77+15 | 98+19 | 92+15 | 106+17 | 130+21 |
| v3t | -Os | 61+17 | 74+20 | 99+23 | 97+21 | 114+24 | 140+32 |
| v3vc, v3vd, v3ve | -Os | 58+17 | 75+20 | 97+25 | 91+21 | 111+24 | 137+32 |
| v3v, **v3v2** | -Os | 58+17 | 75+20 | 97+25 | 91+21 | 108+25 | 134+33 |
| v3v3 | -Os | 58+17 | 75+20 | 97+25 | 91+21 | 110+24 | 135+33 |

1D (no 2D decode; v3vc = v3vd = v3ve, and v3v = v3v2 = v3v3):

| variant | level | write | atomic | fused | scatter | compu | compm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3t | -O3 | 58+11 | 72+17 | 97+23 | 92+14 | 107+18 | 130+24 |
| v3vc, v3vd, v3ve | -O3 | 55+9 | 72+17 | 94+23 | 86+13 | 104+16 | 127+22 |
| v3v, **v3v2**, v3v3 | -O3 | 54+9 | 70+17 | 92+23 | 84+13 | 94+18 | 119+23 |
| v3t | -O2 | 58+11 | 72+17 | 97+23 | 93+13 | 106+18 | 130+24 |
| v3vc, v3vd, v3ve | -O2 | 55+9 | 72+17 | 94+23 | 87+13 | 103+16 | 127+22 |
| v3v, **v3v2**, v3v3 | -O2 | 54+9 | 70+17 | 92+23 | 85+13 | 95+17 | 123+22 |
| v3t | -Os | 62+17 | 76+19 | 100+24 | 97+20 | 109+27 | 135+34 |
| v3vc, v3vd, v3ve | -Os | 58+17 | 77+19 | 96+25 | 91+20 | 105+23 | 131+30 |
| v3v, **v3v2**, v3v3 | -Os | 59+17 | 76+19 | 97+25 | 91+21 | 103+24 | 128+32 |

Largest VC1 E1, X branch (channel 0 packets that decide on the X byte; -Os built from the -O2 command):

| variant | level | write | atomic | fused | scatter |
| --- | --- | ---: | ---: | ---: | ---: |
| v3t | -O3 | 85+26 | 103+27 | 124+32 | 120+30 |
| v3vc | -O3 | 82+26 | 103+27 | 122+31 | 114+30 |
| v3vd | -O3 | 80+30 | 101+31 | 120+35 | 112+34 |
| v3ve | -O3 | 81+28 | 101+28 | 121+33 | 113+31 |
| v3v | -O3 | 79+31 | 98+31 | 118+36 | 110+34 |
| **v3v2** | -O3 | 80+29 | 99+30 | 119+35 | 111+33 |
| v3v3 | -O3 | 81+28 | 101+28 | 121+33 | 113+31 |
| v3t | -O2 | 85+23 | 102+27 | 124+32 | 121+25 |
| v3vc | -O2 | 82+20 | 102+27 | 121+32 | 115+24 |
| v3vd | -O2 | 80+24 | 100+31 | 119+36 | 113+28 |
| v3ve | -O2 | 81+21 | 100+29 | 120+33 | 114+25 |
| v3v | -O2 | 79+24 | 98+31 | 117+36 | 112+28 |
| **v3v2** | -O2 | 80+23 | 99+30 | 118+35 | 113+27 |
| v3v3 | -O2 | 81+21 | 100+29 | 120+33 | 114+25 |
| v3t | -Os | 94+35 | 112+37 | 133+42 | 131+39 |
| v3vc | -Os | 91+35 | 111+37 | 131+43 | 125+38 |
| v3vd | -Os | 88+36 | 108+38 | 128+44 | 122+39 |
| v3ve | -Os | 90+33 | 110+35 | 130+41 | 124+36 |
| v3v | -Os | 89+40 | 108+42 | 128+48 | 122+43 |
| **v3v2** | -Os | 91+37 | 110+39 | 130+45 | 124+40 |
| v3v3 | -Os | 91+37 | 110+39 | 130+45 | 124+40 |

Reading:
- **v3v2 on N/S**: the stall goes back to v3vc's level at -O3 and -O2 (write 63+11, atomic 81+18). At -Os it keeps
  most of change 3's gain (write 66+19 against v3t 72+20 and v3vc 69+20; atomic 84+22 against 87+23).
- **v3v2 on W and 1D**: v3v's paths exactly.
- **v3v3**: on N/S at -O3 and -O2, one coordinate is spilled (9 loads instead of 8), +1 instruction and +3 stall on
  every path (allocation). So compare it with v3v2 only on E/W rows; on 1D it is v3v2.
- **VC1 X branch**: v3v2 within ±1 instruction of v3v, and 1 to 3 less stall.

Checks (v3v, v3vc, v3vd, v3ve, v3v2, v3v3; scripts `fullcheck.sh`, `stabcheck.sh`, `loadcmp.py`, `sizes.py` in the
scratch dir):

| Check | Result |
| --- | --- |
| Offline builds, every logged config | each: -O3 536/536, -O2 536/536, -Os 534/536 (the 2 garbled VC1 commands, as for v3tk) + 96/96 VC1 at -Os from the -O2 commands |
| E0 against v3tk (PT_LOAD) | identical on every config for each: 208/208 at -O3 and -O2, 207/207 at -Os, 48/48 VC1 at -Os |
| E1 against v3tk | changed in exactly the 128 program configs (127 at -Os; 48/48 VC1); the 200 others identical |
| Ordinary edits (coldmul, init, templates, os; W, N/S, 1D, VC1 at -O3, -O2, -Os) | run_main_loop never DIFF, for each: coldmul moved; init moved (VC1 same); templates same; os moved at -O3 and -O2, same at -Os |
| VC1 code size, largest E0 + E1, -O3 (limit 26,624) | v3v 25,032; v3vc 25,064; v3vd 25,036; v3ve 25,072; v3v2 25,072; v3v3 25,032 (v3tk 25,100) |
| VC1 -O2 / -Os | v3v2 23,992 / 21,596; the others 23,900 to 23,944 / 21,572 to 21,632 |
| Comment reflow after the full checks (v3v, v3vc, v3vd: same line count) | hot E1 ELFs PT_LOAD-identical to the checked builds |
| Coverage (`coverage.py build`, every case, -O3; names v3vapi, v3v2c, v3v3c, v3vcc, v3vdc, v3vec) | each identical to v3tk case by case over 41 cases; the only failures are the inherited `*_nofuse` (4 cases) and `wh_vc1` (15/20). `bh_2d_vc2` 56/56 at -O3, -O2 and -Os for v3v2 |

### 12.12 Tried and not used, and what to run

Tried and not used (offline counts only):
- **Cut only while more than 4 slots wait (G), or only on passes with progress (E).** In the model (§12.10) both keep
  change 1's realigned batches and leave passes without progress as v3t's. Both source forms changed the per-packet
  code, though:
  - G: base and a constant were computed before the batch test, so W -O3 and -O2 grew by +2 to +3 on every path
    (write 61, atomic 79 against 59, 76).
  - E: 1D +3 at -O3 and -O2, W -Os +2 to +3.
  - Allocation and scheduling, so neither is a form to propose. If the hardware shows that idle-pass cuts cost,
    this is the policy to look for in another form.
- **Fixed decode with the E/W byte read through the codec** (a first v3v2 draft): W +2 stall at -O3 and -O2 against
  the plain field. Combined with the prompt remainder it also spilled the N/S coordinate.
- **`ncrisc_noc_nonposted_write_with_transaction_id_sent` in the completion check** (§12.9): changed the loop's code.

What to run (all three levels, every suite):
1. **v3vc, v3vd, v3ve, v3v2.** v3vd − v3vc is change 3 as run (N/S and E/W separately); v3ve − v3vc is the fixed
   decode; v3v2 − v3ve is change 1 with the fixed decode (1D gain, any E/W cost); v3v2 is the proposal.
2. **v3v** (API form): its per-path counts match v3v_raw's (§12.9), so it should reproduce p14v; v3v − v3vd is change
   1 with the plain decode. Run it if there is room, or a few rows against p14v to confirm the API port.
3. **v3v3**, only if v3v2 still shows the E/W write regressions against v3ve. Compare E/W rows only.

Rows to watch:
- NS_PureAtomic (-O3 and -O2), NS_MixSize4, NS_PureWrite4352, Mesh1HopNS, NS_Chunk8Fused_NoFlush (change 3).
- EW_PureWrite4352, Mesh1HopEW, Mesh3HopEW, EW_MixSize4, Linear3HopEW at -Os (change 1 or unexplained).
- The preset gains should hold in every dir: E/W and Linear1HopEW scatter, FWD/EW fused, N/S at -Os.
- 1D suite: v3vc / v3vd / v3ve against v3v / v3v2 gives change 1's 1D gain on the API form.
- The lead's derived dirs v3v_hook, v3v_strict and v3v_jit were built from the raw v3v (now v3v_raw). They need to
  be derived again for any of the dirs above.

### 12.13 Hardware round 2 (jobs 290–297), the 4352 B write rows, v3v4 and v3v5

Labels: p15v2 = v3v2, p15e = v3ve, p15c = v3vc, p15d = v3vd; v3t = p12t / p13k; p15k = v3tk re-run. RELAXED and STRICT
validation passed for all four. VC1 34/34 at every level for v3v2, v3ve and v3vd. v3vc hung once at -Os on VC1
(MultiMeshAllToAll_size_4096, unicast write). The capture showed all 48 routers alive and drained, with nothing
waiting, so it is not a router-held deadlock; the lead is handling it separately.

Confirmed from §12.10:
- **Change 3 as run causes the N/S regressions; the fixed decode removes them.** NS_PureAtomic, NS_MixSize4,
  NS_PureWrite16: v3vd 134.1 / 123.6 / 122.2, v3ve 127.3 / 120.1 / 116.5.
- **Change 1 (cut)** gains on 1D and E1-bound rows: Linear1HopEW scatter 148.7 → 142.5, NS fused 156.0 → 153.3. It
  costs EW_MixSize4 117.8 → 121.5, NS_MixSize4 120.1 → 122.6, Linear3HopEW write -Os 149.3 → 152.7 and Mesh3HopEW
  write +1.3.
- **v3v2 is the best on suite means at every level.** Against v3t (4352 / mix / flushmix / 1D):

| Level | 4352 | mix | flushmix | 1D |
| --- | ---: | ---: | ---: | ---: |
| -O3 | −2.08 | −1.15 | −0.25 | −3.68 |
| -O2 | −2.52 | −1.13 | −0.97 | −3.82 |
| -Os | −3.25 | −2.93 | −1.34 | −2.23 |

  v3v2 rows: FWD_PureFused4352 154.9 / 155.0 / 157.5 (-O3 / -O2 / -Os), NS fused 153.3 / 152.1 / 160.8, NS write16
  115.2 / 114.7 / 116.0.

**The regression in every variant: one-hop 4352 B writes, +2.4 to +3.0 at every level.** It shows in v3vc (preset
only) and v3vb (no preset) alike. 16 B writes, scatter and fused rows do not regress. At -O3:

| Row | v3t (p15k re-run) | variants |
| --- | ---: | --- |
| Mesh1HopEW write | 128.1 (128.1) | v3vc 130.9, v3vb 130.5, v3ve 131.0, v3v2 130.5 |
| Mesh1HopNS write | 128.3 (128.6) | 131.3 |
| EW_PureWrite4352 | 128.4 (128.5) | 131.1 |
| NS_PureWrite4352 | 128.5 (128.7) | 131.4 |
| Mesh3HopEW write | 144.0 | v3vc 145.1 |

**The per-packet path does not explain it** (W config, -O3, one-hop 4352 B write: receive, local write, completion;
`rxscen.py write -v`, `idlescen.py`):
- **v3vb's write path is v3tk's, instruction for instruction:** 62 instructions with the same register sequence.
  That is the command-buffer read and wait, then PACKET_TAG, NOC_CTRL, TARG_ADDR_LO, RET_ADDR_LO, RET_ADDR_COORDINATE,
  AT_LEN_BE and CMD_CTRL, then the two software write counters. Only the decode's two loads come in the other order
  (2 cycles less stall).
- **v3vc** waits for the command buffer at the same point, before the first register store. It does not write NOC_CTRL,
  and it updates the two counters between PACKET_TAG and CMD_CTRL instead of after (the API's order). 59
  instructions. The NoC command itself is the same: the same NOC_CTRL value, same VC, non-posted, with a transaction
  ID.
- **Idle passes are identical in all three:** 12 instructions (14 with 1 to 3 slots waiting), one packets-received
  register read and 3 taken branches. Nothing new sits between packet arrival and the write. Nothing changes when the
  write is issued, beyond a few cycles earlier in v3vc.
- **What every variant shares is a shorter E1 per packet in this stream.** Estimate: write path plus a quarter of a
  one-sender batch, instructions + stall at 5-cycle loads, -O3:

| | v3t | v3vb | v3vc | v3vd | v3ve | v3v | v3v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 89.75 | 87.25 | 85.75 | 83.75 | 83.75 | 83.25 | 83.25 |
| N/S | 91.75 | 95.00 | 86.75 | 91.00 | 87.00 | 90.00 | 86.25 |

  Variants with change 1 also cut a batch after each partial batch, which adds a credit frame.

**Correction (hardware, §12.15): this cause is rejected for the one-hop 4352 B write rows.** v3v4 left them exactly
where v3v2 had them, and v3vd's N/S rows were the worst of the four variants, not the closest to v3t. The hold did help
other rows (Mesh3HopEW, MixSize4, small writes), as the frame mechanism predicts for them. The original reasoning
follows.

**Cause (rejected for these rows): E0 pays per credit frame, so a faster E1 makes E0 slower.** This happened before,
in RESULTS §56 and §57.
- These rows are E0-bound. In §72, E0 took about 129 of a 132.9-cycle interval on one-hop writes and moved +27 under
  a 20-cycle E0 delay.
- Each E0 iteration that sees new completions makes one credit return to its upstream (`credit_worker` or
  `credit_upstream_router`, a NoC inline write). That costs about 35 cycles (§72: 7.6 cycles/packet at 0.22 frames
  per packet).
- In 2D, E1 credits a partial batch (1 to 3 slots) after 4 passes without progress. Between 4352 B packets, E1 idles
  for the packet interval minus its own per-packet work. E0's jitter stretches some gaps further: an iteration with a
  credit return sends ~35 cycles later.
- v3t's E1 sat just short of the point where a gap holds 4 idle passes. Every variant made the per-packet work 2 to 6
  cycles shorter, or added a cut frame after each partial batch, or both. So E1 sends more, smaller batches, and E0
  makes more credit returns.
- §56 was the same effect: an E1 speed-up took E/W 16 B writes +4.6, with completion frames going from 2.0 to 1.24
  packets per frame. §57's 4-pass hold fixed it then. v3t's E1 is faster again, enough to reach that hold on 4352 B
  streams.
- **It fits the row classes.**
  - It shows only where E0 binds and E1 has slack: one-hop 4352 B writes, and 3-hop a little.
  - Scatter, fused and atomics are E1-bound, so E1 has no idle passes and their batches stay whole.
  - 16 B writes leave E1 long gaps in v3t too, so both sides of the comparison are already past the hold.
  - Change 1's MixSize4 and Linear3Hop costs are the same mechanism through cut frames.
- **Model** (`sim/e0frames.py`: E0 122 cycles per packet + 35 per iteration with new completions; wire 118 per
  packet; E1 idle pass 15; 16 slots; uncalibrated):

| E1 per packet | 87 | 84 | 81 |
| --- | ---: | ---: | ---: |
| hold 4: cycles/packet, frames/packet | 132.56, 0.274 | 132.90, 0.284 | 134.30, 0.325 |
| hold 8: cycles/packet, frames/packet | 131.76, 0.250 | 131.76, 0.250 | 131.76, 0.250 |

- **Confidence: medium-high.** The mechanism was measured on hardware in §56, and it fits every variant and every row
  class. It has not been measured on these builds.
- **To confirm on hardware:**
  - EW_PureWrite4352 with `TT_FABRIC_TXQ_COUNTERS=1` (counts only; it slows runs), v3tk against v3vc. The receiving
    E1's TXQ1 transfers are one per completion frame (§56). Expect fewer packets per frame in v3vc than v3tk's ~3.9.
  - From the existing runs: v3vd has no cut, and its N/S per-packet estimate is about v3t's (91.0 against 91.75). So
    its NS_PureWrite4352 and Mesh1HopNS should regress least of the four variants, while its E/W rows regress like the
    others.

**v3v4: v3v2 with a 2D hold of 8 passes** (`partial_batch_idle_passes`, was 4).
- A partial batch goes out only after 8 passes without progress: at least 8 × 14 instructions with a register read
  each. That is longer than the gap between packets of a stream the link sustains (under ~125 cycles: a maximum-size
  packet's wire time minus E1's work for it).
  - So within a stream, E1's speed no longer changes how many frames E0 handles.
  - Partial batches still go out when a stream stops, so the hold is bounded and cannot hang.
- The code change is one named constant and its comment. Per-packet paths are identical to v3v2 in every config and
  level (`score.py`, 96 cells). The only code difference is one `li` on the idle path that tests the hold.
- 1D is unchanged: a partial batch goes out at the first pass without progress.
- **Risk:** a flow that stops with its window full and 1 to 3 slots uncredited waits about 4 more idle passes (~60 to
  80 cycles) per stop.
  - In RESULTS §28 a 16-step hold cost 5 to 65 cycles/packet on credit-loop-bound rows. But that hold also dropped
    the rule that a full batch goes out at once on an idle pass, so completions waited for a fifth packet. v3t keeps
    that rule.
  - Rows to watch: Mesh3HopEW, multicast, forwarded rows, flushmix, VC1.
  - If the hold costs there, the alternative is 6 passes.
- **Expected:** one-hop 4352 B writes back to v3t's 128 or below. Other rows as v3v2, or better where v3t's E1 already
  sent partial batches inside a stream (E0-bound 16 B and 1 KiB streams).

**v3v5: v3v4 with the cut in 1D only.**
- In 2D, completions are v3t's, and a batch may straddle two record words. With the 8-pass hold a mid-word start is
  rare in 2D, so the cut's 2D effect should be small either way. v3v5 against v3v4 measures it.
- Per-path counts: 2D equal to v3ve's (§12.11), 1D equal to v3v4's and v3v2's.
- Built on v3v4, as the expected better of v3v2 and v3v4. A v3v2-based form differs by the one constant.

Checks for v3v4 and v3v5: below (§12.14).

### 12.14 Checks for v3v4 and v3v5, and what to run

| Check | v3v4 | v3v5 |
| --- | --- | --- |
| Offline builds, every logged config | -O3 536/536, -O2 536/536, -Os 534/536 (the 2 garbled VC1 commands) + 96/96 VC1 at -Os from the -O2 commands | same |
| E0 against v3tk (PT_LOAD) | identical: 208/208, 208/208, 207/207, 48/48 VC1 | same |
| E1 against v3tk | the 128 program configs change (127 at -Os; 48 VC1); the 200 others identical | same |
| Against each other | | 1D E1 identical to v3v4 (all 40 1D program configs); the 88 2D program configs differ |
| Ordinary edits (coldmul, init, templates, os; W, N/S, 1D, VC1; -O3, -O2, -Os) | run_main_loop never DIFF: coldmul moved; init moved (VC1 same); templates same; os moved at -O3/-O2, same at -Os | same |
| VC1 code size, largest E0 + E1 | -O3 25,076; -O2 23,996; -Os 21,600 (limit 26,624) | -O3 25,076; -O2 23,944; -Os 21,628 |
| NoC API rule | no `NOC_CMD_BUF_WRITE_REG` / `NOC_CMD_BUF_READ_REG` in E1 code | same |
| Coverage, 41 cases at -O3 | identical to v3tk case by case (inherited `*_nofuse`, `wh_vc1` 15/20); `bh_2d_vc2` 56/56 at -O3, -O2, -Os | same |
| Per-path counts | identical to v3v2 (§12.11), all 96 cells | 2D identical to v3ve, 1D to v3v2 |

What to run (all three levels, every suite):
1. **v3v4.** Expected: one-hop 4352 B writes back to v3t's ~128 or below. Everything else as v3v2, or better on
   E0-bound small-write streams. Watch Mesh3HopEW, multicast, forwarded, flushmix and VC1 rows for the longer hold.
2. **v3v5.** v3v5 against v3v4 is the cut in 2D, now that 2D partial batches are rare. Expected: the MixSize4 rows
   back to v3ve's; any loss on E1-bound 2D rows whose bursts start mid-word.
3. **Optional confirmation of the cause:** EW_PureWrite4352 with `TT_FABRIC_TXQ_COUNTERS=1`, receiving E1's TXQ1
   transfers per packet (counts only), for v3tk, v3vc and v3v4. Expected: v3vc fewer packets per frame than v3tk;
   v3v4 back to ~4.
4. From the existing p15d data: v3vd's NS_PureWrite4352 and Mesh1HopNS should be the closest of the four variants
   to v3t, since its N/S per-packet estimate is v3t's and it has no cut.

### 12.15 Hardware round 3 (jobs 304–307), and the idle-polling diagnostics

**Hardware** (p16v4 = v3v4, p16v5 = v3v5): all validation passed; VC1 34/34 at every level for both.
- **v3v4's 8-pass hold leaves the one-hop 4352 B writes where v3v2 had them.** -O3: Mesh1HopEW 130.5 (v3v2 130.5, v3t
  128.1), EW_PureWrite4352 130.5, NS_PureWrite4352 130.8, Mesh1HopNS 130.7; the same at -O2 and -Os. So the
  frame-count cause of §12.13 is rejected for these rows.
- **The p15d prediction failed.** v3vd's N/S rows are the worst of the four variants, not the closest to v3t:

| Row | v3t | v3vc | v3vd | v3ve |
| --- | ---: | ---: | ---: | ---: |
| Mesh1HopNS write | 128.4 | 131.3 | 132.3 | 131.2 |
| NS_PureWrite4352 | 128.6 | 131.4 | 132.5 | 131.3 |

- **v3v4 helps elsewhere, and is the current best:** Mesh3HopEW write −0.8 to −1.8 against v3v2, MixSize4 −0.3 to
  −1.9, small writes −0.3 to −1.0. Its suite means are slightly better than v3v2's at every level.
- **v3v5:**
  - It restores MixSize4 (117.8 to 118.9) and keeps the 1D gain (Linear1HopEW scatter 142.5).
  - It loses the 2D cut gains: NS fused 156.1 against 153.5, EW scatter 146.9 against 144.5. So its 2D, mix and
    flushmix means are worse than v3v4's.
  - Linear3HopEW write -Os stays at +3.8 (152.6), so that cost comes from the 1D cut.

**What E1's idle pass touches, and what E0's hot path touches** (-O3 listings: E1 W, N/S, 1D, VC1 from v3vc; E0 from
a 2D v3tk build of the same router, main loop `sender_erisc::run_main_loop`; addresses resolved from the code).

E1, one pass without a packet (`idlescen.py`): 12 instructions, or 14 with 1 to 3 slots waiting, and 3 taken
branches.

| Access | Address | Notes |
| --- | --- | --- |
| packets-received count, receiver channel 0 | stream 4, `STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE` (reg 297): 0xFFB444A4 (2D); stream 2, 0xFFB424A4 (1D) | the only register access; VC1 configs also read receiver channel 1's stream |
| channel state, loop counter, idle count | registers | no load or store |
| L1 | none | the 15-instruction loop runs from the i-cache |
| NIU, TX queues | none | the completion block runs only when a batch is due |

Once per round (every `iterations_between_ctx_switch_and_teardown_checks` passes): termination test (3 L1 words: the
termination signal and the launch mailbox), the context-switch word (stream 30 scratch register, 0xFFB5E010, which E0
also reads every round), and a heartbeat store to L1 every 64 rounds.

E0 main loop, the accesses that hit memory-mapped blocks:

| Block | Addresses | Use |
| --- | --- | --- |
| stream registers 0 to 3 | 0xFFB4n4A4 reads (the grouped free-slot reads), 0xFFB4n438 writes (free-slot updates), 0xFFB4n028 for n = 1, 2 | worker and forwarding channels |
| stream 30 scratch | 0xFFB5E010 | context-switch handshake (per round) |
| TX queue 0 | 0xFFB90004 to 0xFFB90044 | sends, busy checks |
| NoC 0 command buffers 0 and 3 | 0xFFB20000 to 0xFFB20040, 0xFFB21800 to 0xFFB21840 | worker credit writes, atomics |
| debug registers | 0xFFB121F0 / 0xFFB121F8 (wall clock) | |
| L1 | completion counters, worker semaphore, connection info | |

- **E1's idle pass and E0's hot path share one block: the stream registers.** They use different streams: E1 stream 4
  (2D) or 2 (1D), E0 streams 0 to 3. They share no L1 word, no NIU and no TX queue.
- **The microbenchmarks (RESULTS §17) suggest that block is pipelined.** Four independent stream reads from one core
  cost about 4 to 5 cycles together, against 7 to 10 for one read used at once. E1 issues one read per ~20 to 25
  cycles of idle pass. If arbitration between the two cores works the same way, a collision costs an E0 access about
  a cycle. That is well under the 2.5 to 3 cycles/packet to explain, unless the two cores' accesses serialize for the
  full read latency, which no measurement here shows.
- **A closer candidate on the same block is the receiving tile.** Every packet's remote pointer update arrives over
  Ethernet as a write to the receiving tile's stream 4, the register E1 polls. If E1's reads delay that write, the
  receive side of the link slows, independent of E0.
- **Which E1 matters depends on the rows' direction.**
  - One-way rows: the binding E0 is on the sending tile, and that tile's E1 is idle in every variant, running the
    same 12-instruction idle pass. So a different polling rate next to E0 cannot explain them; the receiving-tile
    candidate above can.
  - Rows with traffic both ways: the receiving E1 shares a tile with a binding E0.
  - The lead knows which of these rows are bidirectional; that tells which mechanism to look at.
- **Caveat on "a faster E1":** v3vd's N/S E1 is no faster per packet than v3t's by the estimate in §12.13, yet its N/S
  rows are the worst. So neither the per-packet time nor the idle time per packet orders the variants.

**Diagnostic variants (DIAGNOSTIC ONLY: never for shipping):**

| Dir | Wait after a pass without progress |
| --- | --- |
| `v3vc_d8` | ~7 to 14 cycles (the cycle counter read twice): the control for the code change |
| `v3vc_d20` | ≥ 20 cycles |
| `v3vc_d50` | ≥ 50 cycles |
| `v3vc_d100` | ≥ 100 cycles |
| `v3tk_d50` | ≥ 50 cycles, on v3tk (optional control) |

- **What they change.** Each is v3vc (or v3tk) plus `diag_idle_wait()`. After a main-loop pass in which no receiver
  channel made progress, E1 spins on its own cycle counter (`csrr mcycle`, a core-local CSR) until N cycles have
  passed. The spin touches no stream register, L1 word, NIU or TX queue.
- **Per-packet code.** The four v3vc_dN have identical per-packet paths; only the constant differs. Against v3vc they
  carry +1 to +4 instructions per path from the bookkeeping (W +2 to +4, N/S +1 to +2).
  - So compare the dN with each other: v3vc_d8 is the zero point.
  - v3vc_d8 against v3vc shows the code change alone, plus a ~10-cycle wait.
- **Reading the results:**
  - **Contention.** If the one-hop 4352 B rows improve steadily from d8 to d100 while E1-bound rows (scatter, fused,
    atomic) stay or worsen, idle polling contention is the mechanism.
  - **v3tk_d50** then says whether v3t's own polling costs it too (contention in general), or only the variants'.
  - **No trend:** polling is ruled out.
- **Confounds:**
  - The wait also stretches the 4-pass hold (more passes' worth of cycles), so frame-sensitive rows (MixSize4, small
    writes, 3-hop) may improve for that reason. The v3v4 result above shows the 4352 B rows do not depend on it.
  - The wait adds up to N cycles to the detection of a packet that arrives during it: about N/2 more credit
    round-trip on average, and latency rows grow.
- **Checks (offline):**
  - Every logged config builds for each: -O3 536/536, -O2 536/536, -Os 534/536 (the 2 garbled VC1 commands), plus
    96/96 VC1 at -Os from the -O2 commands.
  - E0 is PT_LOAD-identical to v3vc's (v3tk_d50: to v3tk's) on every config.
  - E1 changes in the 128 program configs only.
  - VC1 -O3 code: 25,104 B (v3vc_dN), 25,148 B (v3tk_d50).

**A shipping form, if the diagnostics show contention** (sketch, not built):
- **Poll period with a floor.** The first pass after progress polls at once, so back-to-back packets are not delayed.
  Later passes without progress poll again only when at least W cycles have passed since the last poll of that
  channel. W is the smallest wait that recovers the rows in the diagnostics.
- **Cycles from a core-local counter.** The wall clock is not suitable: it is a register in the shared debug block,
  which E0 also reads. This needs a small API addition, a `get_core_cycles()` (mcycle CSR) in `risc_common.h`. The
  phase profiler has one (`fabric_core_cycles()`), and the rule against hand-rolled hardware access applies.
- **What it changes:**
  - E1-bound rows keep polling every pass, since they rarely have passes without progress.
  - Streams with idle gaps poll at most once per W cycles after the first idle pass.
  - A packet that lands during the gap waits up to W cycles more; latency rows would show that.
- **What it does not need:**
  - No branch hints or allocation steering.
  - Stream access stays through `read_stream_count`.
  - The partial-batch hold should count cycles too, not passes, so the two do not interact.
