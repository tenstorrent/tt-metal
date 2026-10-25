# Router v3e1: a whole-program redesign of E1 (the receiver ERISC)

v3e1 is v3a with ERISC 1's program rewritten. In two-ERISC mode ERISC 1 (`MY_ERISC_ID == 1`) now runs its own program,
`e1_kernel_main`: its part of kernel_main's init, its own downstream router connections, a new receive loop and
teardown. ERISC 0 and every build outside that scope run v3a's code, unchanged: **E0 is instruction-identical to v3a on
every logged config at -O3, -O2 and -Os** (0/208, 0/208, 0/160 + 0/48 VC1), and so are the 200 E1-processor builds the
program does not cover (ring, speedy, single ERISC).

**v3e1b** (after hardware jobs 224–226: local-delivery, VC1 and -Os fixes) is described in §9; **v3e1c / v3e1d**
(scatter, atomic and fused local delivery, on v3l2be1b) in §10; **v3t / v3tb** (a hot loop that cold edits do not
move, and paths less dependent on the opt level, on v3sb) in §11.

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

So insulation needs both a function boundary and no constant multiplications inside the function. With the boundary
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
