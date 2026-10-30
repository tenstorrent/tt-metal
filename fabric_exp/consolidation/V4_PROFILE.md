# v4 profile matrix: bound side, register pressure, phases (2026-10-29)

User (2026-10-29): before choosing between readability and perf, capture per config: register pressure (do we need to
cut state?), time in phases, and which ERISC bounds each row; side by side across configs; and say what the profiling
itself costs. Router: v4trm (the lead); v3qm alongside where the instrument allows. Gaps being explained:
V4_PERF_GAPS.md (census section).

## 1. Instruments and what they cost

| Instrument | What it gives | Cost to the router |
| --- | --- | --- |
| Delay sweep (`prof_delay_runs.sh`, `prof_report.py`; `variants/v4trmd`) | per row and side: the cost of a fixed spin after each send (TX) or each packet received (RX) | none at 0: v4trmd's images equal v4trm's, 600 of 600; with a spin on, its code shifts the loop slightly (2D sender +15 instructions, spills unchanged; VC1 sender +44, +3 stack loads), the same in the 32 and 96 builds |
| PC sampling (`prof_pcs_runs.sh` over `pcsv_runs.sh`, `pcs_profile.py`) | per ERISC, where cycles go by source line and section | none: read over the debug bus; rate unchanged within 0.01 cycles/packet (jobs 028-029) |
| Static scan (`regpress.py`) | per hot function: frame, callee-saved registers, instructions, stack loads/stores (spills), all loads/stores, constants and calls at each loop depth | none: offline, from the ELFs |
| Not used | main's on-device phase timers: perturbed rows 13-45% and add loop state | |

Reading the delay sweep: a spin of N costs about N+12 on its path. A side with no slack pays ~44 for a 32-cycle spin;
a side with slack S pays 44 - S. At 96 every row is past its slack on both sides, so the side is read at 32
(`prof_report.py`: "TX" or "RX" when only that side has under 8 cycles of slack, "both", or "-" when neither: the row is
bound elsewhere). On one ERISC both spins run on the same core; TX vs RX then says which pass (send or receive work) the
row is sensitive to on its critical router.

## 2. Bound side (a09u32 jobs 172-174, v4trmd; rows per side)

| suite | TX | RX | both | neither | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 2D 4352 B | 6 | 4 | | | writes TX (0 slack; RX 21-30); scatter RX (slack 4-7) |
| 1D | | 8 | 2 | 6 | 1-hop writes and NeighborExchange write: neither (TX 14, RX 28 slack): ~wire time (4352 B at 400G ~117 cycles) |
| mix | 21 | 8 | 1 | 3 | every write 16 B-4352 B TX (16 B: 119 cycles/packet, a 32-cycle spin costs 52); scatter, fused, atomics RX |
| ring | | 13 | | 1 | every single-flow ring row RX (0 slack), as the earlier ring A/B found |
| Y-torus | 5 | 5 | 2 | | E/W as 2D; N/S RX |
| one ERISC 2D | | 8 | 2 | | one-way rows RX (the receiving router's core); bidirectional both |
| one ERISC 1D | | 7 | 9 | | NeighborExchange write (the row below main) both |
| one ERISC ring | 2 | 6 | 8 | | multicast and bidirectional rows both |
| mux | | | 2 | 20 | 20 rows bound by the mux, not the router; full-ring multicast (below main) both |
| VC1 split | 8 | 1 | 2 | 2 | 1-hop and bursts TX (0 slack; RX 23-29): the VC1 sender |

On 3-hop and multicast writes (two ERISCs) a 32-cycle send spin costs 53-60, not 44: the send side's cost compounds along
the path there.

## 3. Register pressure (static, medians per config; `results/prof_regpress*.txt`)

Pass = loop depth >= 3 of the two-ERISC loops, or the whole body of the single-ERISC loop's out-of-line passes.

| config, function | v4trm: pass insns / stack ld / stack st / all ld / calls | v3qm |
| --- | --- | --- |
| 2D E0 sender | 582 / 58 / 19 / 131 / 6 | 515 / 43 / 16 / 101 / 3 |
| 2D E1 receiver | 439 / 4 / 4 / 73 / 2 | 865 / 2 / 2 / 129 / 3 |
| 1D E0 sender | 256 / 17 / 10 / 60 / 5 | 245 / 17 / 5 / 44 / 2 |
| ring E0 sender | 313 / 23 / 13 / 74 / 5 | 266 / 24 / 6 / 53 / 2 |
| Y-torus E0 sender | 709 / 78 / 32 / 164 / 6 | 579 / 56 / 18 / 118 / 3 |
| VC1 E0 sender | 1051 / 120 / 47 / 235 / 6 | 914 / 68 / 36 / 174 / 3 |
| VC1 E1 receiver | 1102 / 21 / 13 / 202 / 6 | 1341 / 25 / 7 / 202 / 10 |
| one ERISC 2D | receiver pass 571 / 2 / 2 / 100 / 1 + sender pass 549 / 5 / 5 / 126 / 2 (state in a record: 464 B frame in single::run) | one inlined loop: 1632 / 95 / 58 / 382 / 3 (1376 B frame) |

**This correction is itself WRONG (2026-10-29, checked in the DWARF of the v4trm_O3 images):** on two-ERISC senders without VC2 the compiler scalarizes the ForwardingTable (and `worker`, `far_vc0`, the semaphore pointer: location lists), so their sp-relative operations are register-allocator spills after all; only on VC2 routers is the table a frame object (`fbreg -272`: VC2's worker interface inside it escapes to out-of-line calls), and in the single-ERISC state record. The "64 table accesses" split by frame offsets below is meaningless. The original text follows. **Correction (2026-10-29, WRONG as first written):** the "stack ld / st" columns are not spills for the senders. The
sender's ForwardingTable (the forwarding channels' records) is a local of `sender::run` (and of `single::run`'s state),
memory-resident by design, so its field accesses are sp-relative too. On the 2D sender, about 64 of the 94 in-loop stack
operations are table accesses and about 30 are spills or reloads (split by the table's frame offsets; approximate). The
v4-vs-v3 comparison above is of stack traffic, not register pressure; a spill-only count needs the table's accesses
excluded per config.

Static symptoms only: v3's one-ERISC loop spills more than v4's passes and is faster (census: 245.5 vs 276.6), the
caveat that static counts do not rank changes.

## 4. Phases (PC sampling; a09u32 jobs 175-176; cycles per packet on each sampled ERISC; `results/prof_pcs_*.txt`)

Two ERISCs, v4trm against v3qm (the bound side from §2 in brackets):

| row | rate v4 / v3 | v4 sender ERISC | v3 sender ERISC | v4 receiver ERISC | v3 receiver ERISC |
| --- | --- | --- | --- | --- | --- |
| 2D 1-hop write [TX] | 127.5 / 126.9 | send 59.9, free-slots group read 36.5, worker checks 17.2, worker credit 6.0 | send 57.2, send check 15.2, connection poll 11.8, idle test 10.8, group read 10.1, credit 8.4, completion check 7.2 | delivery 34.1, completions 25.6, receive 16.6, retire 15.1, poll 13.3 | delivery 35.6, completions 23.0, decode 19.7, poll 14.3, retire 13.7 |
| 2D 3-hop write [TX] | 151.6 / 142.8 | send 67.2, group read 31.0, worker checks 13.1, upstream credit 12.6, forwarding checks 10.4, forwarding dispatch 9.9 | send 64.7, idle test 16.4, completions + bookkeeping 16.2, send check 12.1, upstream credit 9.9, connection poll 8.9, group read 7.5 | forward 50.5, receive 22.2, completions 19.9, retire 15.0, admission 14.4 | forward 52.2, decode 21.1, completions 20.2, retire 13.5, admission 12.7 |
| 2D multicast write [TX] | 145.9 / 141.1 | send 68.5, group read 20.6, forwarding checks 16.2, upstream credit 16.0, worker checks 10.0 | send 53.3 + 9.7, completions + bookkeeping 23.4, upstream credit 15.0, idle test 11.0 | forward 35.5, completions 24.3, delivery 21.9, receive 19.0, admission 15.3 | forward 35.3, delivery 24.6, completions 22.6, decode 18.5, admission 14.0 |
| 1D 1-hop write [wire] | 124.4 / 124.4 | send 56.1, group read 27.9, worker checks 24.7 | send 52.4, send check 17.4, connection poll 13.1, idle test 12.5 | delivery 31.7, completions 25.8, poll 14.6, retire 13.3, header 12.7 | delivery 33.8, completions 23.5, poll 16.1, retire 12.8 |
| ring 1-hop write [RX] | 150.7 / 176.7 | send 63.3, group read 30.5, worker checks 30.3, worker credit 19.8 | send 51.6, credit 34.8, completion check 28.5, send check 20.4 | delivery 31.9, acks 30.9, completions 26.6, pass call 16.0, header 11.6 | generic loop (not split) |
| half ring multicast [RX] | 182.2 / 229.7 | send 73.1, worker checks 26.6, forwarding checks 24.4, group read 21.1, upstream credit 20.4 | completions + bookkeeping 51.6, send 48.3, completion check 31.1 | forward 34.0, completions 27.2, acks 25.4, delivery 25.4 | generic loop (not split) |
| Y-torus N/S write [both] | 158.7 / 173.0 | send 59.6, group read 42.6, worker checks 28.9, worker credit 21.7 | send 51.4, credit 29.7, completion check 23.2, idle test 21.2 | acks 30.9, delivery 30.2, completions 27.0, pass call 19.0, decode 13.8 | generic loop (not split) |

One ERISC, v4trm (the one core of the critical router; v3's one-ERISC loop is all inlined in kernel_main, so its split is
not comparable: 64-78% "loop and round control"):

| row | rate v4 / v3 | v4 (one core) |
| --- | --- | --- |
| 2D 1-hop write [RX: receiving router] | 206.9 / 179.1 | delivery 36.1, retire 28.1, the idle sender pass (group read 28.7 + worker checks 18.6), receive 18.0, completions 17.0, out-of-line pass entries and calls (router_single.hpp:48-57, 101-102) ~39 |
| 1D 1-hop write [RX] | 185.3 / 159.1 | delivery 35.9, retire 23.8, idle sender pass 34.8, completions 17.4, header 12.6, pass entries and calls ~38 |
| NeighborExchange write [both; below main] | 244.9 / 231.5 (the census suite's row: 251.3) | send 69.4, delivery 37.5, retire 27.3, worker checks 23.7, completions 16.4, pass entries and calls ~26 |
| ring 1-hop bidirectional [both] | 379.3 / 343.6 | send 69.8, worker checks 42.0, acks 39.9, completions 35.0, delivery 33.8, worker credit 29.2, retire 22.3 |
| mux full-ring multicast [both; below main] | 436.6 / 398.3 | forward 73.1, send 68.1, worker checks 41.5, delivery 37.4, acks 36.4, completions 32.9, worker credit 30.3 |

## 5. What the matrix says (facts first; anything about what a change would gain is a hypothesis until an A/B)

Facts:
- Two-ERISC writes are send-side bound (2D, mix, torus E/W, VC1 1-hop, 3-hop, multicast); scatter, fused and atomic
  writes, and every single-flow ring row, are receive-side bound. 1D 1-hop writes run at the wire (~124 for 4352 B).
- On the bound sender, the send itself is 56-73 cycles per packet in both routers; the rest (50-75) is per-pass checks,
  the free-slots group read, credits and forwarding dispatch.
- The biggest static sender passes are on the bound side: 2D 582 instructions / 58 stack loads, Y-torus 709 / 78, VC1
  1051 / 120 (v3: 515 / 43, 579 / 56, 914 / 68). Receivers carry little spill (4-9; VC1 21).
  **WRONG as "spill":** those stack loads include the forwarding table's record accesses (§3 correction). **That marking is itself WRONG:** on non-VC2 senders the table is scalarized; these are spills (§3).
- On one ERISC, a one-way row is bound on the receiving router's core, which also runs an idle sender pass every pass
  (~35-47 cycles per packet) and the out-of-line pass structure (~26-39); retire costs 24-28 there against 13-15 on two
  ERISCs.
- Mux mode: 20 of 22 rows are bound by the mux, not the router.

Hypotheses for the gaps (each needs its isolating A/B before any variant):
- One ERISC vs v3 (+28 on the 1-hop rows): the idle sender pass and the out-of-line pass structure on the receiving
  router's core.
- VC1 vs v3 (+21, size-independent, TX-bound): the VC1 sender pass's size and spills.
- 3-hop and multicast vs v3 (+5 to +9, TX-bound): the sender's per-pass checks and forwarding dispatch around the send.
- Register pressure: cutting state is a candidate only for the two-ERISC senders (2D, torus, VC1); not for the receivers.

Not covered: VC1 and Galaxy were not PC-sampled; v3's one-ERISC loop needs a line-level classifier to compare; v3 has
no delay hooks (bound side measured on v4 only).

## 6. What a cycle of pass cost is worth (per-pass sweep; a09u32 jobs 177-179)

User (2026-10-29): optimize generically (the router in general, not one row); judge on the whole matrix. Instrument: a
fixed spin once per sender pass or once per receiver pass (`variants/v4trmd`; `TT_FABRIC_EXP_DRISC_DELAY`, reused: low
16 bits sender pass, high 16 bits receiver pass; at 0 every image equals v4trm's, 600 of 600). `prof_delay_runs.sh GROUP
pass`, `prof_report.py --pass`. Per row: the cost of 16 cycles per pass (the margin), and the 16 -> 48 step over 32.

Rows whose cost rises by at least 12 for a 16-cycle spin per pass (159 rows, MixCmd left out), and the mean cycles per
packet that one cycle of that pass costs on those rows:

| suite | rows | sender pass binds | per cycle | receiver pass binds | per cycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D 4352 B | 10 | 6 | 1.60 | 5 | 1.51 |
| 1D | 16 | 2 | 2.14 | 10 | 1.62 |
| mix | 30 | 21 | 1.56 | 22 | 1.27 |
| ring | 14 | 5 | 1.17 | 13 | 1.61 |
| Y-torus | 12 | 7 | 1.63 | 8 | 1.54 |
| one ERISC 2D | 10 | 10 | 1.58 | 10 | 1.68 |
| one ERISC 1D | 16 | 16 | 1.66 | 16 | 1.72 |
| one ERISC ring | 16 | 16 | 1.53 | 16 | 1.53 |
| mux | 22 | 3 | 1.41 | 3 | 1.41 |
| VC1 split | 13 | 8 | 1.92 | 5 | 1.68 |
| all | 159 | 94 | | 108 | (70 both, 27 neither: the mux and the wire) |

- At the margin a cycle of pass cost costs 1.2-2.1 cycles per packet where that pass binds: added pass time also
  lengthens the credit round trips. Past the slack (16 -> 48) it is 1.0-1.2 per cycle.
- The sender pass binds 94 of 159 rows: every two-ERISC write row (small writes ~2 per cycle), every one-ERISC row
  (there it runs on the bound core even when it has nothing to send), and the VC1 1-hop rows (~2 per cycle; the
  receiver pass ~0 there). The receiver pass binds 108: scatter / fused / atomic rows, rings, Y-torus N/S, every
  one-ERISC row.
- So both passes are generic levers. The sender pass is where v4 is heavier than v3 (§3) and where the one-ERISC, VC1,
  3-hop and small-write gaps sit; the receiver pass is where v4 already leads v3.

Where the 2D sender pass's stack traffic is (the -g twin of the sampled 1-hop sender; in-loop stack operations, 94 in 2D
against 49 in 1D): loop-invariant values reloaded at the loop's back edge (13), NoC command-buffer, TX-queue and header
field addresses, and the per-channel ring cursors (the channel's slot and the far ring's next slot) stored and reloaded
around each send; the forwarding completion check. **WRONG as first written:** most of those 94 (about 64) are the
forwarding table's record accesses (the table is a local of sender::run); about 30 are reloads of invariants and
cursors (§3 correction). **That marking is itself WRONG:** the 2D sender's table is scalarized (DWARF location
lists), so the 94 are spills and reloads of its fields and of other values (§3).

Inference, to be confirmed by the change itself: a cycle removed from the sender pass is worth ~1.2-2 cycles per packet
on the rows above (removal and addition need not be symmetric).
