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
