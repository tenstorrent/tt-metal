# Controlling what the sender keeps on the stack (2026-10-29)

User direction: before more perf work, find a way to choose which router state lives in memory, so the rest is left to
the registers and placement stops shifting with unrelated edits; wrap it for reuse. DESIGN_BRIEF.md rule 2 was amended
for it (placement primitives with a documented contract). Stage 1: a compile-only standalone source pins down the
mechanism's contract. Stage 2: the router's offline builds. Device runs only after stage 2.

## Why (from the v4sh3 work, V4_SENDER_SHAPE.md 1c)

- Placement today is a side effect: moving VC2's interface out of the forwarding table let the compiler split the table
  on VC2 routers (DWARF `frame` -> `split`), and the VC2 sender pass grew 631 -> 730 instructions; VC2 1-hop writes
  +6 to +10% (job 195, one pass).
- "Split" is not "in registers": in the 2D sender pass every forwarding record's slot, sent, completed and credit
  address lives in a spill slot, loaded and stored at points the allocator picks; `in_flight`, the far rings' free counts
  and loop constants stay in registers.
- Earlier: a closure edit moved 2D 3-hop 162 -> 188 through placement; v3l's runtime-indexed records ("memory by
  construction") were split again by a later change (V3L_NOTES.md).

## Stage 1: the contract (`fabric_exp/placement/`)

`placement_test.cpp`: a sender-like loop compiled with the router's flags (`flags.txt`; the router's command minus LTO)
at -O3/-O2/-Os: a per-pass worker set (counts, semaphore pointer, far ring), N forwarding records visited when they have
work, the grouped stream-register reads between fences, and three out-of-line calls on rare paths (credits, connection
event). MODE: 0 the compiler's choice; 1 `asm volatile("" : "+m"(record))` at the end of each record's visit; 2 `volatile`
record fields; 3 the same asm without `volatile`; 4 the asm once per pass on the whole table; 5 once before the loop.
Tools: `analyze.py` (pass-loop instructions and stack accesses; `--slots` by source line, `--regs`, `--inv`), `loc.py`
(DWARF location kind of named locals), `head.py` (the grouped reads as emitted), `sweep.py`.

Findings:
1. The contract holds: with the asm operand (and with `volatile` fields) the table is a frame object at -O3, -O2 and -Os
   for N = 1, 3, 6 (DWARF `frame`; the compiler's choice: `split` everywhere). Each record field is loaded where the
   source loads it (the visit's `sent = r.sent`) and stored where the source stores it (on change); nothing else.
2. No cost seen: pass-loop instructions equal or fewer than the compiler's choice (-O3 N=3: 242 vs 251; N=6: 416 vs 427;
   -Os N=3: 218 vs 224); stack accesses about equal. `volatile` fields gave the same code as the asm in this test.
3. Scope is the whole enclosing object: `+m` on one record put the whole table in memory, `in_flight` with it (7 loads,
   3 stores in the pass loop); as a separate local, `in_flight` has no stack access. Rule: state meant for registers
   must not share an object with state pushed out.
4. Where the asm sits did not matter here (per visit, non-volatile, per pass, even once before the loop: the same pass
   loop), but only the in-loop forms carry the guarantee; before-the-loop works today only because GCC does not promote
   the table back into registers, which nothing requires.
5. The grouped register reads stay back to back in every mode (the load-group primitive is undisturbed).
6. What is not pushed out stays the compiler's (worker, far ring, `in_flight`: `split` in every mode), as intended.
7. Not answerable here: the router's pressure. In the toy the compiler already kept two of three records in memory, so
   there was little to gain; the router splits its table fully. Stage 2.
8. Side observation (toy; a hypothesis for the router): the out-of-line calls in the loop account for much of the stack
   traffic. With the three calls replaced by inline register writes, -O3 N=3 pass stack loads 45 -> 32, stores 19 -> 10.
   Values carried around the loop must survive those calls: callee-saved registers (s0-s11, 12) or the stack. In the
   toy those 12 held the worker's counts, the far ring, the loop counter, two hoisted constants and the semaphore
   pointer: the per-pass set alone fills them. If the router behaves the same, pushing records out is necessary but the
   budget for per-pass state is about 12 registers less hoisted constants.

Helper for stage 2 (router_common.hpp):

```cpp
// Keeps `obj` in memory here: its current value is stored before this point and read back after it (GCC's documented
// semantics of an asm memory operand; DESIGN_BRIEF.md rule 2). It applies to the whole object, so state meant to stay
// in registers must not share an object with it.
template <typename T>
FORCE_INLINE void keep_in_memory(T& obj) {
    asm volatile("" : "+m"(obj));
}
```

## Stage 2: the router, offline (`variants/v4pl1`)

`v4pl1` = v4sh3 (the role layout) + `keep_in_memory(record)` at the end of each forwarding visit + `in_flight` out of
the forwarding table into a loop local of its own (it is read every pass; finding 3). Nothing else is placed.

Built 1214/1214 at -O3 (cov_m, every case). DWARF: the forwarding table is a frame object on every sender (v4sh3:
split everywhere); `in_flight`, the workers and the far rings stay the compiler's (split).

Sender pass (instructions / stack loads / stack stores; `regpress.py`):

| config | v4sh2b | v4sh3 | v4pl1 |
| --- | --- | --- | --- |
| 2D | 588 / 58 / 19 | 588 / 58 / 19 | 500 / 48 / 16 |
| 2D + VC2 | 631 / 72 / 26 | 730 / 88 / 38 | 636 / 61 / 31 |
| Y-torus | 707 / 78 / 32 | 707 / 78 / 32 | 631 / 65 / 25 |
| VC1 | 1051 / 120 / 47 | 1051 / 120 / 47 | 894 / 85 / 35 |
| Galaxy express | 902 / 112 / 45 | 902 / 112 / 45 | 793 / 85 / 32 |
| 1D | 256 / 17 / 10 | 256 / 17 / 10 | 253 / 24 / 8 |
| ring | 313 / 23 / 13 | 313 / 23 / 13 | 312 / 29 / 11 |
| one ERISC 2D (single sender pass) | 549 / 5 / 5 | 551 / 5 / 5 | 559 / 5 / 5 |

(v3q's VC1 sender pass for reference: 914 / 68.) 2D stack map (`slots.py`): every record field is accessed exactly where
the source reads and writes it; the worker's counts, the far rings' free counts and `in_flight` no longer touch the
stack. What still does: loop-invariant values the compiler hoisted and spilled (used by the send and the credit
writes; 15 loads, no stores), `round.attempted` (one store per send site: `round` escapes to `end_round`), the per-round
count, and the worker interface (frame object). 1D and ring (one forwarding channel): the record's fields are now loaded
at each visit (+2 each for sent, completed, credit address) and a few spills go (net -2 / -1 instructions, +5 / +6 stack
loads). One ERISC: the record was in memory already (the state record); the barrier adds stores and reloads around it
(+8 instructions).

Stability (pass instructions / stack loads; benign edits at -O3: E1 the record's fields reordered, E2 the loop's locals
declared in another order, E3 the operands of the visit test's `||` swapped):

| config | v4sh3: O3, E1, E2, E3 | v4sh3 spread | v4pl1: O3, E1, E2, E3 | v4pl1 spread |
| --- | --- | --- | --- | --- |
| 2D | 588/58, 588/58, 582/58, 559/51 | 29 / 7 | 500/48, 500/48, 500/48, 512/48 | 12 / 0 |
| 2D + VC2 | 730/88, 730/88, 732/88, 695/80 | 37 / 8 | 636/61, 636/61, 637/61, 650/63 | 14 / 2 |
| Y-torus | 707/78, 706/74, 719/78, 727/80 | 21 / 6 | 631/65, 631/65, 631/65, 637/61 | 6 / 4 |
| VC1 | 1051/120, 1051/120, 1058/120, 1009/109 | 49 / 11 | 894/85, 894/85, 894/85, 918/85 | 24 / 0 |
| 1D | 256/17, 256/17, 256/17, 258/17 | 2 / 0 | 253/24, 253/24, 253/24, 256/23 | 3 / 1 |

E1 and E2 (layout and declaration order) no longer move v4pl1 (VC2 +1); they moved v4sh3 by up to 13 instructions.
E3 moves both, v4pl1 about half as much and without stack-load changes on 2D and VC1: the rest is control flow and
scheduling, not placement. -O2 equals -O3 for v4pl1 (v4sh3: VC1 +6). -Os (production uses it with VC1): v4pl1 against
v4sh3, 2D 407/35 vs 411/42, VC2 524/53 vs 521/63, Y-torus 526/53 vs 540/66, VC1 800/79 vs 825/112, 1D 179/17 vs
165/9. Placement kinds: identical across E1-E3, -O2 and -Os for v4pl1 (forwarding `frame` on every sender).

All static: whether v4pl1 is faster, and by how much on which rows, is for the device A/B.

## What "in memory" means here, and what a spill costs

The stack is each ERISC's private 8 KB local data memory at 0xFFB00000 (RESULTS §18), not L1: spills and the
frame-object forwarding records of v4pl1 live there. Measured (RESULTS §17, data cache off as in the router): a stack
load used at once costs 2 cycles, an L1 word 8; but a stack reload issued while stream-register or L1 reads are
outstanding waits for them (+6 to +14). So a spill is cheap in itself; what costs is where a load lands relative to the
hardware reads (the free-slots group, the send's reads), which the compiler's scheduler does not model. **WRONG
(said in chat 2026-10-29):** "reloading a spilled copy ... L1 load latency shows up as a stall"; spills are in local
memory.

Constants (`consts.py`, sender pass): 78-172 constants are built where they are used (lui/li in the pass), none is held
in a register across it, 3-5 are hoisted and spilled (2D v4pl1: two ring bounds 0x6D120 / 0x27920, the far router's
packets-received register address 0xFFB44438 written into the TX queue, a NoC command word 0x208A and a NoC address
mask 0x1000000F; v4sh3: 3; v4pl1 spills more because the freed registers let loop-invariant motion hoist more).
Rebuilding one at its use is lui + addi (2 single-cycle instructions) against a 2-cycle local-memory reload: about even
unless the reload waits behind outstanding reads. Not a lever on its own.

## Device results: v4pl1 (jobs 201-211, a09u32; two passes agreeing within 0.001 on every suite ratio)

Validation 13/13 (job 201). Census suite means, v4pl1 against v4sh2b (same jobs): 2D 144.5 -> 143.0 (-1.0%), mix 128.8
-> 126.7 (-1.6%), Y-torus -0.7%, 1D / ring / burst 0.0 / 0.0 / -0.1%, trimmed 2D 154.5 -> 150.7 (-2.4%), VC1 split 171.5
-> 160.9 (-6.2%; the 1-hop and burst rows 158 -> 141, v3q 137), VC1 inter-mesh -1.2%, one ERISC 2D 276.5 -> 270.5
(-2.2%; v4sh3 279.4), 1D -0.7%, ring -0.7%, mux +0.1%. VC2 on: two ERISCs +0.5% to +1.9% against v4sh2b (v4sh3 +0.9% to
+3.6%), one ERISC -2.1%. Gains sit on the send-bound forwarding rows (3-hop write 151.8 -> 146.0, multicast write 146.0
-> 141.9) and every one-ERISC 2D row (4-7 cycles). Rows worse by more than 1%: VC1 2-hop merge +2.6% / +1.5% (16 B /
4352 B), VC1 multicast +1.5%, one-ERISC NeighborExchange write +1.2%.

Attribution: on two-ERISC configs without VC2, v4sh3's images equal v4sh2b's (VC1: renamed registers), so these A/Bs
change only v4pl1's two edits (records kept in memory, `in_flight` out of the table); which of the two carries what is
not separated. On one ERISC, v4sh3 -> v4pl1 is the same pair. Why one ERISC gains while its static sender pass grew by 8
instructions is not known (hypothesis to check by profile). VC2's remaining +0.5% to +1.9%: only the 1-hop worker-write rows
(154-160 -> 164; forwarded, 3-hop, multicast and scatter rows unchanged). Hypothesis: two worker channels' per-pass state
does not fit the registers and the allocator picks victims (v4pl1 VC2 stack map: VC0's worker slot, both semaphore
pointers, a far ring's next slot spill; v4sh2b kept VC2's state in memory by accident, leaving VC0's in registers).
**Correction (user, 2026-10-30):** "parking VC2's state" would favour VC0's worker over VC2's with no basis (both are
direct worker connections), and the census measures VC0 traffic only, so v4pl1's VC2 result is one-sided: VC2's own
traffic may be faster in v4pl1. Needed first: VC2-traffic rows (VC2 alone, VC0 + VC2). Symmetric options: remove
per-worker state (the connection semaphore address is a fixed offset from the compile-time connection-info address in
the builder's layout, `erisc_datamover_builder.cpp:357-373`), or place by access frequency with one rule for every
worker.
