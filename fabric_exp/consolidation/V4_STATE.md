# v4 sender state: what is hot, where it lives, how to structure it (2026-10-30)

User direction: with placement now controllable (V4_PLACEMENT.md), analyse which state deserves registers and which can
live on the stack, and structure it so the source reads coherently. Base: `v4pl1`. Sender first (it binds 94 of 159
rows and carries the pressure); the receiver after, by the same method. Rules: no VC preference (both worker channels
are direct worker connections); placement primitives only push state out; the same rule for every config.

"Stack" here is the ERISC's private 8 KB local data memory (2 cycles a load used at once; a load issued while
stream/L1 reads are outstanding waits for them, +6 to +14; RESULTS §17-18).

## 1. Inventory (two-ERISC sender loop, v4pl1 source)

| state | words | read / written where | how often | v4pl1 placement (2D) |
| --- | ---: | --- | --- | --- |
| `pending` (VC0 + VC2 free-slots group result) | 1 | top of the pass; worker test; forwarding group tests | every pass (transient) | s0: lives across the worker visit's calls |
| `pending1` (VC1 group result) | 1 | VC1 group | every pass, VC1 routers | - |
| `in_flight` (forwarding channels with packets out) | 1 | group skip test; each forwarding visit | every pass | s11 |
| pass count (`n`) | 1 | loop | every pass | s6 |
| worker `sent` | 1 | `sent != completed` test; `++` per send | every pass | s1 |
| worker `completed` | 1 | same test; `+=` per completion batch | every pass | s9 |
| worker `acked` | 1 | first-level-ack VCs only (Y-torus VC0) | per ack | register |
| worker `expected_connection` | 1 | `semaphore != expected` | every pass | s8 |
| worker connection semaphore pointer | 1 | loaded through every pass | every pass; never changes | s2 (a constant in disguise: see 4) |
| the semaphore's value (transient) | 1 | read in the visit's group, compared after `credit_worker` | every pass | s7: lives across the credit call |
| worker `slot` | 1 | the send | per worker packet | s5 |
| far ring `free_slots` (per VC) | 1 | the send gate `free_slots >= needed`; `--` per send; `+=` per completion | every pass a channel holds a packet | s10 |
| far ring `next_slot` (per VC) | 1 | the send | per packet | s4 |
| forwarding records (slot, sent, completed, acked, credit address) | 6 each | the channel's visit | per visit | parked (`keep_in_memory`) |
| worker interfaces (credit address, connection info) | many | `credit_worker` (per completion batch), connection events | per batch / rare | frame object (escapes) |
| `round` (heartbeat, idle rounds, attempted, sent at round start) | 4 | per round; `attempted` on a busy TX queue | per round | frame object (escapes to `end_round`) |
| stream-register block base `0xFFB40000` | 1 | every stream register access | every pass | s3 (hoisted constant) |
| other constants | - | - | - | 78 built at their use, 5 hoisted and spilled (§V4_PLACEMENT) |

## 2. Where v4pl1 keeps them (DWARF location lists over the pass loop, `fieldloc.py`; registers by writer, `sregs_pass.py`)

- 2D, Y-torus, 1D: every per-pass field in a callee-saved register for 95-99% of the pass; nothing per-pass spills.
- VC1: VC1's far ring `next_slot` on the stack; the rest in registers.
- 2D + VC2: the every-pass values in registers (both workers' `sent`, `completed`, `expected`; both far rings' free
  counts; `in_flight`); the send-path values on the stack (both workers' `slot`, both far rings' `next_slot`) and VC0's
  semaphore pointer. The allocator already sorted by frequency there, by its own heuristics.

## 3. The budget

The pass loop contains calls (`credit_worker` per completion batch, `worker_connection_event`, and per-round calls in the
compiled loop). Values carried around it therefore need callee-saved registers (s0-s11, 12) or the stack. 2D uses all 12:
9 carried values (worker sent / completed / expected / slot / semaphore pointer, far ring free / next, `in_flight`, pass
count), 2 transients that only exist because a call sits inside their lifetime (`pending` across the worker visit, the
semaphore value across `credit_worker`), and 1 hoisted constant (the stream-register base). VC1 adds a far ring (2),
VC2 a worker and a far ring (8): they cannot fit, and the allocator chooses what spills.

Per config, carried values and transients against 12 (counted from the source, v4pl1; 2D checked against the binary):
2D 12; 1D 12; Y-torus 13 (the worker's `acked`); VC1 14 (+ VC1's far ring); VC2 19 (+ VC2's worker: 5, its far ring: 2).

## 4. Observations

1. Constants in disguise: each worker's connection semaphore pointer is a runtime value that never changes. The builder
   places the semaphore at a fixed offset from the channel's connection-info address (`erisc_datamover_builder.cpp:357-373`:
   info block, two 16-byte fields, the semaphore), and the info address is a compile-time constant in the kernel. Making
   it compile-time frees one register per worker channel on every config with a worker.
2. Transients across calls: the semaphore value and `pending` hold callee-saved registers only because a call is inside
   their lifetime. Ending a lifetime before the call frees the register; both need a careful look at order (the
   connection event must see this visit's credits; the forwarding groups use `pending` after the worker visit).
3. Send-path state (worker slot, far ring next slot) is touched once per packet on that path. Where the budget is short
   the allocator already spills exactly these first. Parking them explicitly would make VC1/VC2 deterministic and cost
   2D/1D/torus a few local-memory accesses per send; leaving them to the compiler keeps 2D as is.
4. The stream-register base in s3 is a good use of a register (every stream access); nothing to do.
5. Out-of-line calls in the pass are deliberate (they keep their constants out of the loop: 2D 1-hop 141.8 -> 130.5,
   RESULTS §98). They are also what makes the budget 12 instead of ~27.

## 5. Where the time goes (PC sampling, v4pl1, jobs 212-213; `pcs_row_lines.py`, results/prof_lines_pl1*.txt)

Busiest sending ERISC per row, cycles/packet by innermost source line (a stalled load shows at its first use):

| | 1-hop worker (127.2) | forwarded (125.7) | 3-hop (146.0) | multicast (141.8) | one ERISC 1-hop (200.3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| free-slots group read + tests (sender:189, 207-208, common:93) | ~36 | ~25 | ~31 | ~20 | ~44 |
| send: TX queue registers (tt_eth_api:27, 31) | ~19 | ~19 | ~20 | ~19 | ~23 |
| ring and slot advance after the send (sender:277-278) | ~15 | ~14 | ~14 | ~14 | ~20 |
| packet header read, L1 (packet_header:453) | 7 | 8.5 | 8 | 8.6 | 8 |
| connection semaphore value used (sender:456) | 4.8 | 2.0 | 7.0 | 4.1 | 5.2 |
| single-ERISC pass entries / exits (router_single.hpp) | - | - | - | - | ~22 |

No stack access is among the hot lines on these 2D rows: where the per-pass state fits (2D, §2), placement is not where
the time goes; the time is in hardware accesses (stream-register reads, TX queue, L1 header and counters). Placement
matters where the state does not fit (VC1, VC2: not sampled; no sampling yamls yet) and for stability. Not covered: VC1
(multi-mesh harness), VC2 (needs a VC2-traffic yaml, mock-vetted first), the receiver.

## 6. Proposal (for review)

Classes, one rule for every config and every worker channel:
- **Every pass** (registers; left to the compiler): `pending`, `in_flight`, the pass count; per worker channel `sent`,
  `completed`, `expected_connection` (+ `acked` where first-level acks are on); per far ring `free_slots`.
- **Per send on that path** (registers where they fit): the worker channel's `slot`, the far ring's `next_slot`.
- **Per visit** (parked): forwarding records (done in v4pl1).
- **Compile-time** (removed): the connection semaphore pointer.
- **Rare** (memory by construction): interfaces, round counters.

Steps, cheapest and most generic first (each offline: budget per config, stack maps, stability; then device):
1. Remove the semaphore pointer: derive each worker channel's semaphore address at compile time from its connection-info
   address (the builder's layout), with an init-time check against the runtime argument that fails loudly if the layout
   changes. -1 register per worker channel on every config (VC1 14 -> 13, VC2 19 -> 17).
2. End the two transients' lifetimes before the calls where order allows: the semaphore compare before `credit_worker`
   (the connection event must still see this visit's credits: needs a careful look), `pending` (the forwarding groups
   use it after the worker visit). Up to -2 on every config.
3. Only if VC1/VC2 still do not fit: park the send cursors (worker `slot`, far `next_slot`) explicitly. Not on 2D by
   default: the PC data shows the region right after the send (277-278) already costs ~15 cycles/packet, and parked
   cursors would add local-memory loads exactly there, where TX-queue accesses are outstanding.

Structure (names say role; objects are placement units):
- `WorkerChannel` {sent, completed, expected_connection, slot}: one per worker channel, loop locals; its interface in
  `WorkerConnections` (memory); no semaphore pointer.
- `FarRing` {free_slots, next_slot}: one per VC, loop locals.
- `ForwardingTable` {records}: parked; `in_flight` beside it as its own local.
- If step 3 is needed: the cursors move into one parked `SendCursors` {worker slots, far next slots} ("where the next
  packet comes from and goes to"), so each class stays one object.

## 7. Parking the send cursors: what grouped local-memory loads cost (job 216, `test_erisc_load_overlap`)

New patterns (RESULTS §17 kernel; the old patterns reproduce §17 exactly). Load cost in cycles, steady state (K >= 4):
4 stream reads 4; + 1 local-memory load (last or first) 5; + 3 local-memory loads last 7, first 7 (10 at K = 0); for
contrast 4 stream reads + 1 L1 word 14. Header read (L1, used at once) after the group: slot in a register 12 (4 + 8),
slot parked and loaded with the group 13 (+1), slot parked and loaded right before the header 10 against 8 (+2).

So local-memory loads cost about one cycle each in a group, in either order (stream reads first is better at small K),
and do not hit the four-loads-in-flight limit an L1 word does. Parked cursors (worker slot, far next slot, far free
count) loaded with a group cost ~1 cycle per load plus the stores (not measured) per send: cheap, not free. On 2D, where
they fit in registers, that is a net cost of a few cycles per packet (estimate); on VC1/VC2, where the allocator already
spills them, planned grouped loads replace its reloads.

## 8. Step 1: `v4pl2` (v4pl1 + compile-time worker connection semaphore address)

`router_common.hpp`: `sender_connection_semaphore_addr(ch)` = the channel's connection-info address + the info's size +
two 16-byte builder fields (the builder's L1 layout); `Channel<CH>::connection_semaphore`; `Worker` loses its pointer;
init checks the runtime argument against it (`ASSERT`, watcher builds). Built 1214/1214. Sender pass (instructions /
stack loads / stores) against v4pl1: 2D 502/48/15 (500/48/16: the address is now built per pass, `lui`), VC2 644/54/25
(636/61/31), VC1 890/79/35 (894/85/35), Y-torus 631/64/24, 1D 255/25/7, express 791/81/32. On VC2 the freed register
holds VC0's far `next_slot` (stack -> s7); VC0's slot and VC2's slot and next slot stay on the stack. On 2D the freed
register went to a value of the visit's completion read. Not run on device yet (needs validation: a wrong address
would break worker connections).

## 9. `v4pl3`: the class rule (user, 2026-10-30: no budget counting; uniform presence)

v4pl2 + (1) uniform presence: `Present<PRESENT, T, TAG>` / `Absent<TAG>` / `present<>()` (router_common.hpp) for every
VC's far ring (`sends_vc0/1/2`) and every worker channel (`has_worker`, `has_vc2_worker`), in `Workers`, `SendPath`,
`WorkerConnections`; no VC2-only special case; (2) the connection check picks its path before the credit call
(`credit_visit<CH>` in both paths; same order: credit, then the connection event); (3) the send path parked: `SendPath`
{`WorkerSlots slot`, `FarRings far`}, `keep_in_memory` on the visit's slot and far ring where its send-path work ends
(the worker visit: before the credit call; a forwarding visit: at its end, with its record); `Worker` keeps only the
every-pass counts; (4) the send gate tests `has_packet` before the far ring's free count. Built 1214/1214.
Sender pass (instructions / stack loads / stores; v4pl2 in brackets): 2D 534/58/27 (502/48/15), VC2 647/66/29
(644/54/25), VC1 932/100/48 (890/79/35), Y-torus 686/85/41 (631/64/24), 1D 273/31/13 (255/25/7), ring 341/40/17
(312/28/10), express 856/107/52 (791/81/32), one ERISC 2D 557/1/1 (558/5/5). Stack maps (2D, VC2, VC1): the every-pass
state (both workers' counts, `in_flight`) no longer touches the stack anywhere; the send path is loaded and stored where
used; a few ring-bound constants and the round state still spill. Device: jobs 217-228 (validation of v4pl3 and v4pl2;
census and VC2 A/B, v4pl1 / v4pl2 / v4pl3 in the same jobs, two passes).

## 10. Device: v4pl1 / v4pl2 / v4pl3 (jobs 217-228; two passes agreeing within 0.001)

Validation 13/13 for v4pl3 and v4pl2. Census suite means against v4pl1: v4pl2 2D +0.1%, mix +0.2%, VC1 split +0.8%
(1-hop and burst rows 140.9 -> 143.0; cause unknown), one ERISC -0.2% to -0.4%, VC2 on two ERISCs -0.3% to -0.9%, one
ERISC -0.5%; the rest flat. v4pl3: 2D +2.6%, mix +5.2%, Y-torus +1.7%, VC1 split +6.0%, trimmed +3.3%, one ERISC +0.3%
to +0.7%, VC2 two ERISCs +0.2% to +0.8%, one ERISC -1.2%. v4pl3's losses sit on the forwarding-heavy send-bound rows
(3-hop write 146.0 -> 159.4, multicast write 141.9 -> 152.0, VC1 1-hop and burst 141 -> 157, 1-hop write +2.7); it gains
on VC1 2-hop merge (-4.6) and VC1 multicast (-2.3). **v4pl3 rejected.** **WRONG (§7 and chat, 2026-10-30):** "parking
the cursors costs a few cycles per packet on 2D" (an estimate from grouped loads): +3.7 to +16 cycles per packet.
Hypothesis (not isolated): v4pl3 lets the compiler load the parked far ring and slot at their use, behind the send's
outstanding hardware reads (§17: +6 to +14), not in a group; and v4pl3 bundles three other changes (uniform presence,
the credit-path order, the send-gate order). Isolating A/B next: v4pl3 without the parking, against v4pl2.

## 11. Isolating v4pl3's loss (jobs 229-230; row-first A/B, user's direction)

Why parking cost so much (static, v4pl3 2D): the send groups its hardware reads between `std::atomic_signal_fence` pairs;
the fences are compiler barriers for everything in memory, and parked state is in memory, so the parked far ring and
slot are stored before and reloaded after each fence (the far ring's next slot is loaded twice per send; the free count
is reloaded right behind the visit's in-flight L1 reads). Variants: `v4pl3n` = v4pl3 without the parking (presence,
credit-path order, send-gate order kept); `v4pl4` = v4pl3 with the send path copied into locals where a visit first
needs it and written back once if it changed (the send works on locals, which the fences leave alone).

Rows (`row_ab_runs.sh`, `row_ab_report.py`; two repetitions agreeing exactly), cycles/packet, v4pl2 / v4pl3 / v4pl3n /
v4pl4: 1-hop write 127.2 / 129.8 / 127.1 / 128.8; 3-hop write 145.9 / 159.0 / 146.0 / 151.4; multicast write 142.3 /
152.0 / 142.7 / 145.4. VC1 split (`census_vc1_runs.sh pl4a`): suite 162.3 / 170.5 / 161.9 / 167.4; 1-hop and burst rows
143.0 / 157.0 / 142.3 / 151.0; inter-mesh flat. So the parking is v4pl3's whole cost (v4pl3 against v4pl3n changes
only that); copying in and out recovers about 60% of it and still costs 1.6-8 cycles per packet, VC1 included. **The
"send path parked" class is rejected**: the send path stays with the compiler (it spills only where the state does not
fit: VC2). v4pl3n's readability changes cost nothing on these rows. Open: VC1 1-hop 142.3 (v4pl3n) against 140.9
(v4pl1): the v4pl2 step (compile-time semaphore address) appears to cost VC1 ~1.4 cycles; cause unknown.

## 12. VC1 1-hop under v4pl2: the compile-time semaphore address (job 231)

Static (VC1 sender): the semaphore read has three sites; where it cannot share its address base with the completion
counter read (the worker idle, or just sent), the compile-time address costs a `lui` (v4pl1: `lw t3,0(s2)`; v4pl2:
`lui a5,0x16; lw a5,-736(a5)`). On VC1 1-hop rows VC0's worker on the sending router is idle, so that is every pass.
Isolating A/B: `v4pl3p` = v4pl3n with only the runtime pointer restored. VC1 1-hop and burst rows 142.1-142.4 (v4pl3n)
-> 139.9-140.2 (v4pl3p), other VC1 rows unchanged, suite 161.9 -> 160.5, inter-mesh 339.1 / 338.9, 2D 3-hop 146.0 /
145.9. So the compile-time address is what cost VC1 1-hop (~2.2 cycles); the extra `lui` per pass is the likely
mechanism (hypothesis). It had bought VC2 -0.3% to -0.9% and one ERISC -0.2% to -0.4% (v4pl2 against v4pl1). VC1 is a
production config and VC2 opt-in: keep the runtime pointer. **Re-checked 2026-11-01 (V4_RECEIVER_SHAPE.md §32, job 327): holds; v4rxc7c +2.1 to
+2.4 on VC1 1-hop.** v4pl3p = v4pl1 + uniform presence + the credit path decided
before the call + the send-gate order (+ `in_flight` separate): the candidate for the next full census.

## 13. `v4pl5`: the candidate (v4pl3p cleaned up)

v4pl1 + uniform presence for every VC and worker channel on the sender (`Present`/`Absent`/`present<>()`, router_common.hpp)
and the receiver (VC0's three objects, VC1's and VC2's records; `NoChannelState`/`NoVc2ChannelState` gone) + the
connection check deciding its path before the credit call + the send gate testing `has_packet` first + `in_flight` as
its own local. State classes: left to the compiler: `Workers` (slot, counts, connection semaphore pointer, expected
value), `FarRings`, `in_flight`; parked: the forwarding records; in memory: the interfaces. Gone: `SendPath` (the slot is
back in `Worker`; user, 2026-10-30), the compile-time semaphore address and its init check, the isolation notes. Built
1214/1214. Against v4pl3p: two-ERISC receivers byte-identical; senders the same instructions at the same addresses but
commuted compare operands and two init instructions reordered; one-ERISC passes the same instruction counts (state
record offsets changed); trimmed one-ERISC init 61 -> 50 instructions (absent VC0 state not initialized). Device:
validation and full census + VC2 A/B against v4pl1, two passes (jobs 232-242).

## 14. The receiver (started 2026-10-30, while v4pl5's census runs; the receiver's code is unchanged since v4pl1)

Inventory (receiver::run): VC0's `Position` {available, slot, index} and `CompletionCounts` {uncompleted, idle passes}
(per packet), VC0's `SlotRecords` (a byte per ring slot, indexed at run time), VC1's and VC2's `ChannelState` records
(the same three parts each), the credit sender (the credit counts and frame), `frame_due`, the downstream tables (passed
in from kernel_main: memory by design), `me`, the round and pass counts.

Placement (DWARF, v4pl5): VC0's position, counts and `frame_due` split (callee-saved registers where tracked, never the
stack); `records0`, the credit sender, `round` frame objects; VC1's and VC2's records frame objects (the runtime-indexed
records inside keep the whole record in memory: GCC does not split an aggregate with a variable-offset access, a
compiler behavior, not a guarantee); `me` and the pass count optimized out. Budget: the 2D receiver writes 5
callee-saved registers in its loop and holds 6 more invariant (constants 0xFFB91000, 0x16000, 0x01010101, 0xFFECC0A0
and two pointers): room to spare; the VC1 receiver uses all 12 and spills two constants. Calls in the pass:
`deliver_other` (out of line), `end_round` (per round).

Where the time goes (PC samples of v4pl1, busiest receiver ERISC, `pcs_row_lines.py --erisc 1 --pick receiver::`;
these rows are send-bound, so this is the receiver's own cost, not the bound side), cycles/packet, 1-hop write / 3-hop
write: packets-received register poll 12.7 / 4.5; a wait landing at the end of the receive step 11.7 / 12.5; local
write waiting for the NoC command buffer 9.0 / -; NoC command register writes 7.3 / 14; the NoC API's software write
counters (`noc_*_writes_num_issued`, `noc_nonposted_writes_acked`: a load and a store each per write) ~7 / ~7; the
completion check (writes left for a transaction ID) 9.7 / -; retire 7.2 / 4.8; the downstream cursors (next slot, free
count, from the table in memory) - / ~28 (where the waits land, not pure cursor traffic); the route lookup 3.2 / 3.7.
Receiver-bound rows (scatter, fused, atomic, ring, 1D half-ring and multicast) are being sampled (jobs 243-245).
Candidate levers (hypotheses): the NoC API's software counters, if the router's transaction-ID completion tracking makes
them unnecessary and the API has a variant without them; the downstream cursors' loads on the forward path.

Receiver-bound rows (v4pl5, jobs 243-245; `results/prof_lines_r5_rx.txt`), busiest receiver, cycles/packet:
scatter 142.0 / atomic 129.5 / ring 1-hop 150.7 / half-ring 182.4 / 1D multicast 142.8. NoC command buffer waits and
register writes 34 / 19 / 15 / 23 / 17; the NoC API's software counters 8.3 / 4.2 / 8.6 / 6.7 / 2.9; on atomic rows the
flushed-atomic wait on `noc_nonposted_writes_acked` (`ncrisc_noc_nonposted_writes_flushed`) and the flush test 13.6; the
credit sender's per-channel counters in memory (`ack_counters[src]`, `completion_counters[src]`) 12.1 on ring 1-hop, 9.3+
on half-ring; the admit test 10.3 / 10.0 on scatter / atomic; header decode, ack loop and slot records ~6 / ~6 / 26 /
24 / 10; downstream cursors ~22 on 1D multicast; completion checks 2.8-8.6. NoC counters (who reads them): the posted
counter is read only by the posted barrier (teardown, after `ncrisc_noc_counters_init()` resyncs from hardware), so its
increment on every forward is dead work; of the non-posted pair only `acked` is read during operation (the mesh/line
flush before a flushed atomic: hardware ack count == `acked`); `num_issued` only at teardown. Levers (hypotheses):
forwards with `update_counter=false`; local writes counting only `acked`; later, batching `acked` until something reads
it; the credit sender's counters.
**WRONG (2026-10-30, user's recollection, then the source):** the posted counter and `num_issued` are read during
operation: delivering a `NOC_UNICAST_INLINE_WRITE` packet uses Blackhole's emulated L1 inline write
(`noc_fast_spoof_write_dw_inline`, flush=true), which spins until the hardware sent counts equal the software issued
counts, posted and non-posted, on NOC1; an uncounted forward or local write there makes it spin forever (tt-metal
#28758, why the dataflow API forces `update_counter` on Blackhole). Two ERISCs: the receiver (ERISC1, NOC1) alone keeps
NOC1's counts; the sender is on ERISC0 / NOC0 (`worker_handshake_noc = noc_index`). One ERISC: the router runs on
ERISC1 with NOC_INDEX 1, so the sender's credit writes share NOC1's counts too. The "uncounted" variant is dropped.
Left (not pursued for now): batching the counts and publishing them before every reader (a missed reader hangs), or
giving inline-write delivery its own transaction-ID completion so nothing reads the global counts (changes main's
delivery helper).

## 15. v4pl5's acceptance run, and the one-ERISC isolation (jobs 232-242, 246)

v4pl5 against v4pl1 (two passes agreeing within 0.001; validation 13/13): two ERISCs flat (2D +0.1%, 1D, mix, burst,
ring 0.0%, Y-torus +0.1%, trimmed 0.0%, mux -0.2%), VC1 split -0.3%, inter-mesh +0.1%, VC2 on -0.3% to -1.0%; one ERISC
2D +1.0%, 1D +0.8%, ring +0.7% (multicast, bidirectional and 1D rows +1.0% to +1.5%): not accepted. Isolation (one
ERISC, `row_ab_runs.sh pl5i`, two repetitions agreeing exactly), cycles/packet, 1-hop write / multicast write: v4pl1
200.3 / 323.5; v4pl5 202.3 / 327.7; v4pl5c (the credit and the connection check back in v4pl1's sequence) 198.4 / 324.3;
v4pl5g (the send gate back in v4pl1's order) 204.2 / 326.8; v4pl5r (the receiver as in v4pl1) 202.3 / 327.7. So the
credit-path reorder (step 2.1) costs one ERISC (~4 cycles on both rows) and measured nothing on two ERISCs (v4pl3n against
v4pl2): dropped. The receiver presence cleanup is free; the send-gate order helps one-ERISC 1-hop by ~2 cycles and costs
multicast ~1. Why 2.1 costs one ERISC: hypothesis (duplicated credit code in the out-of-line, mostly idle, sender pass).
Next: v4pl6 = v4pl5 with v4pl1's credit sequence (v4pl5c, cleaned), full census against v4pl1.

## 16. v4pl6 accepted (jobs 247-257; two passes agreeing within 0.001)

v4pl6 = v4pl5 with v4pl1's credit and connection sequence (step 2.1 dropped). Validation 13/13. Against v4pl1: every
two-ERISC suite 0.0% (2D, 1D, mix, burst, ring, Y-torus, trimmed, mux; VC1 split and inter-mesh), one ERISC 2D -0.3%,
1D -0.2%, ring -0.3%, VC2 on two ERISCs 0.0% to -0.1%, one ERISC -0.9%; no row worse than 1% (MixCmd excluded, as in
every census), six one-ERISC rows better than 1%. Accepted: **lead `v4pl6`** (from v4sh2b, through v4pl1).

## 17. NoC command buffers: presets, and whether a second buffer overlaps (2026-10-30; job 263)

User's question: rotating a packet's back-to-back NoC commands (scatter chunks, multicast forwards) over several command
buffers, so one waits less; would reprogramming a buffer cost more than the wait? Counts from the NoC API that v4pl6
builds with, checked in the compiled v4pl6 2D receivers (E/W- and N/S-facing) and sender (scratchpad `noc_regs.py`).

Register writes per command (each command also reads its buffer's ready state first):

| Command | Buffer | Preset | Full | The preset holds |
| --- | --- | ---: | ---: | --- |
| local write (E1) | 0 | 6 | 7 | NOC_CTRL only |
| forward data, one live downstream (1D, E/W 2D) | 2 | 5 | 7 | NOC_CTRL, destination coordinate |
| forward credit, one live downstream | 1 | 1 (2 on rings) | 7 | all but the go register |
| forward data / credit, several downstreams | 2 / **3** | - | 7 / 7 | none |
| atomic (E1), upstream credit (E0) | 3 | - | 7 | none |
| worker credit (E0, out of line) | 0 | 3 | - | the connected worker's credit word |

Buffer use: E1 with one live downstream uses all four; E1 with several leaves buffer 1 unused (its forward credits go on
the atomics buffer); E0 on two ERISCs uses 0 and 3 and leaves 1 and 2 unused. The emulated inline-write delivery
(`noc_fast_spoof_write_dw_inline`) also writes buffer 0's NOC_CTRL; it writes the value the local-write preset holds,
so the preset survives, by coincidence of equal values. **WRONG (2026-11-01, V4_BOUNDS.md §B; same code in v4pl6 and
v4rxv2i3):** the inline delivery is posted (`noc_inline_dw_write<DEFAULT, true>`) and writes NOC_CTRL 0x4082, the
preset is non-posted 0x4092 (RESP_MARKED); disassembly of v4rxv2i3's 2D E/W E1 (cov_m bddlead_O3
bh_2d__11660148679966767161: d260-d268 vs d3a0-d3a8). Nothing restores the preset, so later local writes go out posted.

**Microbenchmark** (`EriscLoadOverlap.NocCmdBufOverlap`, kernel `erisc_load_ubench/kernels/noc_cmdbuf_overlap.cpp`;
`results/noc_cmdbuf_overlap.csv`): ERISC1 / NOC1 as the receiver, non-posted transaction-ID writes to a Tensix L1,
issued as the receiver's local writes (NOC_CTRL preset, then transaction ID with its ready wait, source, length,
destination, coordinate, go), rotating over 1 / 2 / 4 buffers; bursts of 4 from an idle interface (best of 64) and
streams of 64 (best of 8). Cycles per write:

| | burst issue | burst ready call | stream issue | stream ready call | stream all sent |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 B - 2 KiB, n = 1 / 2 / 4 | 40 / 37 / 38 | 15.3 / 14.5 / 14.3 | 42.7 / 46.1 / 42.5 | 14.1 / 15.2 / 14.0 | 43 / 46 / 43 |
| 4 KiB, n = 1 / 2 / 4 | 57 / 37 / 38 | 31.8 / 14.5 / 14.3 | 70.4 / 69.2 / 67.0 | 41.9 / 38.4 / 38.6 | 72.1 / 72.0 / 72.1 |

- Up to 2 KiB a write never waits for its buffer: the ready call costs ~14 with 1 buffer or 4 (the ready read
  itself plus the tag write), and the rest of the ~37-43 cycles is programming. More buffers have nothing to hide.
  The n = 2 / 1 differences at these sizes (burst -13, stream +3.4 per write, first write alone 38 / 35 / 36) appear
  at 64 B too, where nothing waits: code shape, not the NoC.
- At 4 KiB the writes leave at the NoC interface's rate (~72 cycles each, ~57 B/cycle, as RESULTS §17's 53-57):
  with 2 buffers a burst of four 4 KiB writes is issued ~80 cycles sooner (227 -> 148), but every write has left at
  the same time (342 -> 335). A second buffer frees the core for other work; it does not move data sooner.
- So the receiver's "command-buffer waits" (RESULTS §98 v4u: ~7 per NoC command) are the ready read, not a busy
  buffer, except for back-to-back large writes. Rotating buffers is not a lever for scatter (chunks are small) or
  for multicast forwards of small packets; for 4 KiB multicast it would only pay if the receiver had work to overlap.
  Earlier attempts at the ready reads: grouping them with the header (v4ub) lost; pairing a stateful forward's two
  (v4uf) gained on forwarded rows.
- Programming: the first write alone takes 35-38 cycles, ~14 of them the ready call, leaving ~21-24 for five register
  writes and their address arithmetic (estimate: ~3-4 cycles per back-to-back register write). **WRONG (job 266,
  table 4 below):** a fully programmed credit (7 register writes) against a preset one (1) costs 6.0 cycles more per
  pair, so ~1 cycle per register write; the rest of the 21-24 is address arithmetic, the loop and the timer reads.
- A write from buffer 1 or 3 needs this core as its source coordinate: `noc_init` sets NOC_TARG_ADDR_COORDINATE only on
  the two write buffers (0, 2), and buffers 1 and 3 keep their last read's or atomic's target. The first build wrote
  from them without it and hung ERISC1 (job 259; broker held the mesh degraded; reset through the broker). The
  benchmark presets the coordinate on every buffer and bounds its waits.

## 18. A ready mask over the command buffers, and forward-like pairs (2026-10-30; job 266)

User's idea: read the buffers' ready registers as a group (as the stream registers are), keep a mask of buffers known
ready (one seen ready stays ready until this core issues on it), take the next command's buffer from it (rotation, or
any ready buffer), reload when a needed bit is clear or opportunistically at each pass's end. Constraint found first
(§17): a buffer's leftover registers matter (an atomic or inline write leaves its destination as the source a later
write would read), so buffers are pooled by command class. The benchmark (`NocCmdBufOverlap`, now three kernel runs;
`results/noc_cmdbuf_mask.csv`) adds: table 2, the table-1 writes through a mask sampled before each repetition (the
previous pass's end, not timed in the issue) and after every 4 writes, the transaction ID then written without a ready
read; table 3, one grouped sample of 1-4 idle buffers; table 4, forward-like pairs (posted data write, then a posted
inline write to a stream register of the same core). Cycles; table 1 re-run in the same job.

| issue, 4-write burst (min) | per-command read: 1 / 2 / 4 buffers | mask: rotate 1 / 2 / 4, any of 2 / 4 |
| --- | --- | --- |
| 64 B - 2 KiB | 172 / 154 / 150 | 190 / 190 / 157 / 205 / 195 |
| 4 KiB | 232 / 154 / 150 | 255 / 211 / 157 / 232 / 195 |

| issue per write, stream of 64 (min) | per-command read: 1 / 2 / 4 | mask: rotate 1 / 2 / 4, any of 2 / 4 |
| --- | --- | --- |
| 64 B - 1 KiB | 44.0 / 43.9 / 47.0 | 47.6 / 45.4 / **40.3** / 53.0 / 51.0 |
| 4 KiB (all sent: 72 per write in every case) | 70.5 / 69.1 / 67.0 | 70.6 / 69.9 / 67.8 / 69.7 / 68.5 |

Table 3, one grouped sample: 1 buffer 11.0, 2: 15.1, 3: 18.1, 4: 22.1 (the first read ~10, each further one ~4).

| per pair (32 pairs, min) | shared buffer | data preset + credit full | both preset | preset + mask | preset + mask, data rotating |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 B: issue / all sent | 91.9 / 92.3 | 56.5 / 57.0 | **50.5** / 50.9 | 54.2 / 54.7 | 65.2 / 65.7 |
| 4 KiB: issue / all sent | 123.9 / 125.2 | 73.8 / 75.5 | 73.5 / 75.5 | 73.9 / 75.6 | 87.7 / 89.3 |

- The mask does not pay on its own. A per-command ready read costs only a few cycles beyond the mask's own
  bookkeeping (its latency overlaps the next command's address arithmetic), while the mask needs a sample per pass
  (11 + ~4 per buffer) and, picking any ready buffer, a runtime-to-buffer dispatch (+7 to +9 per write). Only fixed
  rotation over 4 buffers in a long stream wins (-3.7 per write); E1 has no four free buffers. Forward-like pairs
  with the mask: +3.7 per pair at 64 B, equal at 4 KiB; rotating the data over two buffers: +15.
- Separate buffers per command class matter: sharing one buffer between the data write and its credit costs +35
  per pair at 64 B (+50 at 4 KiB). The credit is kicked a few cycles after the data and waits for the data command
  to be accepted (~25-30 cycles at 64 B), and both are programmed in full. A data write takes ~35-40 cycles to
  program, so the previous write on the same buffer has been accepted by then (table 1: no wait below 4 KiB).
- Presets are worth ~1 cycle per register write avoided (credit full against preset: -6.0 per pair at 64 B; nothing at
  4 KiB, where the NoC's rate binds). §17's register counts at this cost: a forward on a several-downstream router
  ~8 more than a stateful one; E0's upstream credit ~5-6 per credit, once per completion group.
- What remains: grouping the ready reads a packet needs anyway, as v4uf does for a stateful forward's two (table 3:
  each further read in a group ~4 against ~10 alone), e.g. a fused packet's local-write and atomic buffers, or a
  write-and-forward packet's three. Hypothesis, a few cycles on those rows.

## 19. `v4hd`: header fields taken in once per packet into a per-slot ring (2026-10-30; jobs 267-270) - rejected

User's idea: load a packet's header fields once and keep them for later use (the receiver reads `src_ch_id` in the
first-level-ack loop and the same word again when processing; and a parse-ahead receiver would need it). v4hd = v4pl6 +
`SlotHeaders` per receiver channel (the size / send type / source word and the routing fields, per ring slot, in
memory), filled by a take-in step (in the ack loop under first-level acks, at the poll for every packet the
packets-received register shows otherwise) and read by processing. Coverage 1,610/1,610 (-O3); E1 code +56 to +260 B
(VC1 E0+E1 23,524, 3,100 left), E1 stack -128 B; validation 13/13 (job 267). Row-first A/B against v4pl6 in the same
jobs, 2 repetitions each, every pair within 0.1:

| row (cycles/packet) | v4pl6 | v4hd | |
| --- | ---: | ---: | ---: |
| ring1 / ringbidir / halfring | 150.7 / 150.6 / 182.4 | 153.5 / 153.5 / 185.2 | +2.8 / +2.9 / +2.8 |
| torusns | 158.8 | 171.0 | +12.2 |
| ew16 / ewscatter / hop3 / mcast | 115.8 / 141.9 / 145.9 / 141.8 | 126.0 / 159.0 / 152.8 / 145.6 | +10.2 / +17.1 / +6.9 / +3.8 |
| lin1 / linmcast | 124.4 / 142.8 | 124.4 / 150.2 | 0.0 / +7.4 |

Worse on every row but lin1. Rejected; deferred strict atomics need a pending record copied from the header, not this.
Hypothesis (static code only, cycles not split): the duplicate load it removes was already overlapped with the
others (the header word and route bytes issue back to back), so dropping it saves an issue slot; the ring adds, per
packet, two stores and two local-memory loads with index arithmetic for both arrays, and spills around the take-in
loop (torus E1: +19 loop instructions, +4 stack loads, +4 stack stores). Without first-level acks intake and
processing are the same moment, so the ring is pure cost there. Rows set by the worker's credit loop also slow down
when the receiver's per-packet time grows, even where the sender binds (a hypothesis for the size of the 2D losses).
