# Design C: expectations, written before each build (2026-10-31)

Base for every prototype: the lead `v4rxdas2` (coverage build `v4rxdas2_O3`; debug twins of the profiled logs
`design_a/tw_s2_{tso,tv,hr}`). Each section was written before its prototype was compiled; the results and the
explanation of each mismatch are in RESULTS.md.

## C1 `v4rxc1`: the next packet's header read before the acks (Q1; written before the first build)

Change: in `step<CH>`, for a channel with first-level acks, when a packet is present (`available + arrived != 0`), the
packet's header word and route word are read first, then this pass's arrivals are acked (v4rx1's loop unchanged), then
the packet is handled. Channels without acks: the same flow as the lead (`packet_present` holds the lazy register
read). No new state, no kept headers, no fence.

Mechanism (the claim to check in the listings): the header loads are volatile (HeaderWord::bits; 2D Y-router route
bytes) or ordered before the acks' L1 stores by possible aliasing (1D's routing word), so they issue before the ack
loop, and their first use (the decode) comes after it. In the lead the order is count -> ack (mul, `src_ch_id` byte
load, its first use) -> header loads -> decode: two L1 latencies in series. In C1 the header's latency runs under the
ack loop.

State: unchanged (same loop-carried values). The header word and route word (on Y-torus N/S: word + two route bytes)
are live across the ack loop in caller-saved registers; the ack loop has no call.

Expected static results against the lead (rxcmp / loopstat, -O3):
- E0 images byte-identical everywhere.
- Builds without first-level acks (2D, 1D line, VC1, 2D VC2, mux, UDM, Galaxy 2D/express mesh): receiver loop identical
  or layout-only (the non-acking flow is the lead's, rewritten as `packet_present`); code-profiling builds (codeprof,
  allinstr) differ (the timer now opens only when a packet is present).
- First-level-ack receivers (bh_ring, bh_torus, glx_ring/torus, express): loop instructions 0 to +4 (the
  `available + arrived` test; the ack's own `arrived == 0` test stays), sp body accesses 0 (no new spill: the three
  values live only across the ack loop).
- Twins: tso (Y-torus N/S, VC2 off): `lw 40(slot)`, `lbu 48+y`, `lbu 50+x` issued right after the presence branch and
  before the ack loop's `mul`; no instruction using them before the loop ends; `me.y`/`me.x` stay in t-registers as in
  the lead. hr (ring): `lw 40`, `lw 44` before the loop. tv (VC2 on): the same, with `me.y`/`me.x` still reloaded from
  the stack per pass as in the lead (not made worse).
- No `sw` of the header word or route to the stack right after its load anywhere on the FLA path.

Expected on hardware (hypothesis, to be decided by A/B): first-level-ack rows -3 to -6 cycles/packet (one L1 latency,
~5-6 cycles, on passes where a packet is acked and handled in the same pass, i.e. most passes at one packet per pass),
2D rows unchanged. **WRONG (job 312, v4rxc1 vs lead, VC2 off: ring 1-hop +6.9, half-ring +5.1, TorusY1HopNS +3.7, its scatter +1.6 cycles/packet; 2D flat.)**

Would refute the mechanism: GCC sinking the header loads past the loop (impossible for the volatile loads), or spilling
them; or the ack's `src_ch_id` load and the header loads not overlapping at all because the loop's first load issues
late (then the gain is only the header latency, still expected -3 or more).

### C1, second form (after the first build; written before rebuilding)

The first build (`v4rxc1a_O3`: the step rewritten around a `packet_present` helper) changed non-acking images (2D
+1/+6 sp on 12 images, 1D +17 instructions): GCC re-laid out the lazy-read path. The second form keeps the lead's
`receive` and adds the ack inside it after the header read: non-acking images must be byte-identical.

### C1b: route fields loaded before the acks, decoded after them (written before its build)

Found in the C1 twins (RESULTS.md): on 2D Y-routers (tso, tv) `Routing2D::read` selects `y != 0 ? y : x` right after
the loads, in source before the acks, so the select's `zext.b` waits on the Y byte (~5 cycles) before the ack loop:
the stall moved, it was not removed. On the ring (1D, route word used only in decode) the mechanism works as designed.

Change: `Routing2D` and `Routing1D` split `read` into `load` (the header bytes the route word is made of, loaded and not
looked at: 2D the Y and X action bytes and, on edge routers, the destination mesh; 1D the routing word) and `word`
(the select / exit decision). `read` = `word(load(...))` for every other user. `receive`: header word + `load`, the
acks, then `word`, then handle.

Expected against the lead:
- Non-acking images byte-identical (load + word inlined back to back = read).
- tso/tv: after the presence branch `lw 40`, `lbu 48+y`, `lbu 50+x` back to back, then the ack loop, then the select
  (`zext.b`/`beqz` on the Y byte) after the loop; no instruction between the loads and the loop uses them. The three
  values are in caller-saved or free registers across the loop; no new sp access on the path (tv keeps its per-pass
  `me` reloads).
- hr: as C1 (unchanged by this form).
- Loop instruction delta vs the lead: -5 to +5 on FLA images (layout noise from GCC duplicating the read for the
  `available != 0` path, seen in C1); sp body 0 or fewer.
- Hardware (hypothesis): one L1 latency (~5 cycles) off every pass that acks and handles: ring/torus FLA rows -3 to -6,
  half-ring the same; one ERISC: the same mechanism inside receiver_pass; 2D rows identical images. **WRONG (job 312, v4rxc1 vs lead, VC2 off: ring 1-hop +6.9, half-ring +5.1, TorusY1HopNS +3.7, its scatter +1.6 cycles/packet; 2D flat.)**

### C1c: the header word and route fields as one fenced group of plain loads (written before its build)

Found in the C1b twins (RESULTS.md): on tso/tv the Y and X route bytes are volatile byte loads, and GCC follows each
volatile `lbu` with a `zext.b` of the loaded value right after the load (it does not fold an extension into a volatile
load), so the first use of the Y byte is still before the ack loop (~4-5 cycles); the select itself did move after the
loop. Non-acking images were byte-identical (load/word split with byte-typed fields).

Change: `receive` reads the header word and the route fields between two `std::atomic_signal_fence`s (the accepted
load-group primitive: nothing moves into or out of the group, the tests come after it), and `Routing2D::load` uses plain
loads (the group, not `volatile`, keeps both bytes unconditional and together; LEDGER E1-19's reason for the volatile).
Same in every build, with and without acks (one rule).

Expected:
- Without first-level acks: images change where the route bytes were volatile (2D N/S routers, Galaxy 2D): the
  `zext.b` after the X byte goes (-1 instruction per packet), the loads stay together, the select right after them
  (as the lead); E/W 2D routers (plain load already) and 1D: byte-identical or layout-only. sp body unchanged. If GCC
  reloads anything after the second fence that it kept in a register before (escaped memory: the downstream objects,
  noc counters), that shows as +1..+3 loads per packet: would be a regression to report.
- With first-level acks (tso, tv): `lw 40`, `lbu y`, `lbu x` back to back after the presence branch, no `zext.b`, no
  use before the ack loop; the select after the loop. hr unchanged from C1b.
- Hardware (hypothesis): FLA rows -3 to -6 (one L1 latency); 2D unchanged within noise (one instruction fewer on N/S). **WRONG (job 312, v4rxc1 vs lead, VC2 off: ring 1-hop +6.9, half-ring +5.1, TorusY1HopNS +3.7, its scatter +1.6 cycles/packet; 2D flat.)**

### C1d: the fenced group only on channels that route (written before its build)

C1c result (RESULTS.md): on tso the three loads issue together after the presence branch, no `zext.b`, first use after
the ack loop (loop -30 instructions, sp 22 -> 8 on the twin). But non-acking images changed: 2D VC2 +4 sp (VC2's
delivers-only step got a fence around its single header load, and its in-memory cursor is stored before / reloaded
after it), 2D N/S +1 sp (an allocation shuffle: a cold spill moved), VC1 mixed (-4 .. +0 sp).

Change: the fences only where there is a group (a channel that routes: word + route fields); a delivers-only channel
(VC2) reads its one word as the lead does. Expected: bh_2d_vc2's VC2 step back to the lead's (its +4 sp gone; VC0's
step there still changes like bh_2d's), FLA images as C1c, 2D as C1c (+0/+1 sp noise on 12 N/S images).

## C4 `v4rxc4`: the lead's memory placement of VC1's and VC2's cursors stated with keep_in_memory (Q2; before build)

Change: in `pass`, `keep_in_memory(tables.chN.cursor)` right before and right after channel N's step for N = 1, 2 (the
channels whose cursor the lead keeps in a record in `Tables`). Nothing else.

Mechanism: today the cursor is in memory because GCC does not scalarize `ChannelState` (an aggregate with a
runtime-indexed member, SRA's refusal: a heuristic). The asm "+m" operands make it a contract at the two points: the
cursor's value is in memory before the step's first access and after its last, so no part of it can be held in a
register across VC0's step or the frame.

Expected: every image byte-identical to the lead (GCC already loads the cursor at the step and stores it at its end).
If not identical: the differences are where GCC had kept a cursor field in a register across the step boundary (e.g.
VC1's `available` tested once per round), and the listing will show which; that would mean the lead's placement was
partly register-resident already.

## C2 `v4rxc2`: one register group for every router (Q4; written before its build)

Change (from the lead): the pass-start group reads every serviced channel's packets-received register (raw) and the TX
queue's status, on every router (today: only the acking channels' registers and the status, only where a channel
acks). A channel without acks takes its group value when its kept count is used up (`available == 0`), instead of a
read of its own at that point; `send_credit_frame` tests the group's status on every router (the `any_channel_acks`
condition goes). One rule, no special case for first-level acks in the group.

Mechanism and expected costs:
- 2D (VC0 only, no acks), every pass: +1 stream-register load and +2 TX loads (CMD, STATUS) issued back to back at the
  pass start (~3 issue cycles); the count's first use is `available == 0`'s path (as today's lazy read, the stall on it
  is the same when the kept count is used up; when `available > 0` the loaded value is dead, no stall). The status is
  live across the whole step to the frame test: one more long-lived value in a callee-saved register, or a spill
  around the out-of-line delivery call (cold). Saved: the frame's own two TX reads and their stall (~6) on passes that
  send a frame (about one pass in four on 2D, groups of 4: ~1.5/packet).
  Static: loop +3 to +8 instructions; sp body 0 to +2. Hardware (hypothesis): 2D write unchanged (sender-bound), 2D
  scatter 0 to +2.
- 2D VC2 / VC1 (two channels, no acks): as 2D plus the second channel's register in the group. VC1's/VC2's value is
  live across VC0's step: likely stored to the stack right after the group (a store right after the load stalls on
  it, the mechanism that cost v4rx2) or held in a register (GCC's choice). Static check: where its first use is.
- Y-torus VC2 on (tv): VC2's register joins VC0's in the group. Today VC2's idle step stalls ~6-8 cycles on its own
  register read every pass. If the group value survives VC0's step in a register (or a caller-saved one between the
  calls), VC2's idle step loses that stall: hypothesis -4 to -8 cycles/packet on TorusY1HopNS with VC2 on. **WRONG (job 314, VC2 on, vs lead: v4rxc2b TorusY1HopNS write -0.7, its scatter -4.2, 2D E/W scatter -7.0; v4rxc6 write -0.2, scatter -3.7, 2D E/W scatter -6.9. VC2-on builds run VC0 at 43/7/11 slots, V4_RECEIVER_SHAPE §21.)** If GCC spills
  it right after the read: no gain, possibly a loss (+2..+4).
- First-level-ack routers without VC2 (tso, hr): byte-identical to the lead (the group is the same reads).

## C3 `v4rxc3`: the release as one skeleton, the one-slot case in its named parts (Q3 readability; before build)

Change (router_completions.hpp only): `release_completions` is one body for every group size: due (how many of the
oldest uncredited slots go back: `due_slots<T>`), proof (`writes_leave<checks>` of the oldest slot's ID: one check for a
single slot, a bounded wait for a group), credit (`credit_slots<T>`: a single slot's source alone, a group's run as
today), the count. The `T == 1` differences live inside `due_slots` (one slot as soon as another than this pass's waits;
no idle count) and `credit_slots` (the slot's source byte), each an `if constexpr` of a few lines, instead of a second
copy of the whole release. `oldest_uncredited<N>` names the ring arithmetic. No behaviour change.

Expected: every image byte-identical to the lead, or layout/rename only (the same operations in the same order; the
`idle_passes = 0` store kept for groups only). If GCC folds differently (the `n` of a single slot is a constant 1 after
the due test; `first` computed once), at most +-2 instructions on VC1/VC2/2D images and nothing on the per-packet path.

## C5 `v4rxc5`: C3 + single-slot catch-up (Q3 (c); written before its build)

Change: for groups of one slot (first-level acks; rings under 8 slots), `release_completions` loops: every due slot
whose writes have left goes back this pass, oldest first, stopping at the first failed check. Groups of several slots:
unchanged (one per pass; `do ... while (every_proven_slot)` runs once).

Why: the lead credits at most one slot per pass and never the slot retired this pass. In steady state (one packet per
pass) a failed check therefore leaves one extra slot uncredited for every later pass until a pass without a packet
(a permanent one-pass completion lag; §9: a pass of lag costs 5.5-9.2 cycles/packet on these rows). Whether checks fail
in steady state is not known (no telemetry of failed checks): §10's numbers say a 4352 B write needs ~101-123 cycles to
leave and the check comes ~150+ cycles after the issue, so failures should be occasional (a hiccup), more likely on the
half-ring (forward + local write on one NIU).

Expected static: FLA images +2 to +6 instructions (the loop's back edge and second due test); 2D/VC1/VC2 groups: identical
to C3. Per pass in steady state (no lag): one more compare-and-branch after a credit (~1-2 cycles). On catch-up passes:
one more proof read (its stall, ~6-8) plus a credit (~10).
Hardware (hypothesis; this A/B decides whether lag accumulates in practice): if checks never fail in steady state,
FLA rows +0 to +2 (the extra test); **WRONG (job 313, v4rxc5 vs lead, VC2 off: ring 1-hop +11.7, half-ring +12.4, TorusY1HopNS +11.4, its scatter +8.9; one ERISC TorusY1HopNS +3.1; 2D flat.)** if they fail occasionally, a gain of up to the lag cost (several cycles/packet) on
the rows where they fail (half-ring first).

### C2b `v4rxc2b`: every channel's register in the group, the TX status only where a channel acks (before build)

C2 result (RESULTS.md): FLA routers without VC2 byte-identical; tv: VC2's count held in s3 across VC0's step (its idle
stall gone) but the scatter chunk encoding now spilled; 2D: E/W -10 sp, N/S +4..+12 sp (the group's TX status held in a
callee-saved register through the step pushes the scatter path's values to the stack).
Change from C2: the TX status joins the group only where a channel acks (as the lead); without acks the frame reads it
itself (as the lead). Expected: 2D N/S back near the lead (the count is dead after the `available == 0` test, short
lived); 2D E/W loses part of C2's gain; tv as C2.
