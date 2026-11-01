# Design C: static results (2026-10-31; offline only, nothing ran on a device)

Base: the lead `v4rxdas2` (coverage `v4rxdas2_O3`, now also `v4rxdas2_O2`, `v4rxdas2_Os`). Expectations written before
each build: `EXPECTATIONS.md`. Builds under `/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/<name>_O3` (2,086 of 2,216
configs build for every variant; the 130 `*_sc` configs fail by design, as the lead). Debug twins of the four profiled
logs: `/home/aagarwal/.tt-metal-v3r-builds/design_c/tw_<tag>_{tso,tv,hr,js1e}`; receiver listings with inline chains:
`.../design_c/listings/<cfg>_<tag>.lst` (tags: lead, c1, c1s, c1b, c1c, c2, c2b, c5, c6). Tools here: `twins.sh`,
`twlist.sh` (listings + summary), `placeagg.py` (placement.py's kinds aggregated over every image).

"Hypothesis" below means not decided on hardware. Static counts explain; hardware decides (standing rule 6).

## 0. Variants

| variant | question | change from the lead | static vs lead (-O3) | needs device? |
| --- | --- | --- | --- | --- |
| `v4rxc1` | Q1 | header word + route fields read as one fenced group before the acks, decoded after them (`Routing::load` / `Routing::word`) | FLA images change (torus loop -21..-29 insns, -7/-8 sp on 8 of 12); 2D N/S +0/+1 sp on 12 images (a cold reload in `credit_slots`); 1D and 2D E/W same counts (schedule/registers); neighbour identical; VC1 -2..+8 insns, -2..0 sp | yes |
| `v4rxc2b` | Q4 | every serviced channel's packets-received register in the pass-start group; TX status in the group only where a channel acks (as the lead) | FLA without VC2 byte-identical; 2D -1 insn, sp 0; 2D VC2 -14/+14, -1 sp; VC1 -6..+32, 0..+6 sp | yes |
| `v4rxc2` | Q4 (rejected form) | as c2b with the TX status in the group on every router | 2D N/S +4..+12 sp (scatter path), E/W -10 sp; tv scatter-encoding spill | no |
| `v4rxc6` | Q1+Q4 | c1 + c2b | as c1 plus c2b's VC2/VC1 effects; tv: no encoding spill | yes |
| `v4rxc3` | Q3 (a) | release as one skeleton (due / proof / credit), the one-slot case in named parts | **byte-identical, 2,086 of 2,086** | no |
| `v4rxc5` | Q3 (c) | c3 + single-slot catch-up (every due single slot whose check passes, oldest first) | FLA +7..+21 loop insns (+3 per pass on the path); T==1 VC1/VC2 rings change; 2D/1D identical | yes (+ perturbation sweep) |
| `v4rxc4` | Q2 | `keep_in_memory` on VC1's/VC2's cursor before and after its step | not identical: 2D VC2 N/S -31 insns +4 sp (cold), -1 per-packet `me.y` reload; E/W +2/+3, -7 sp; VC1 +8/+9 on 6 | optional |
| `v4rxcj1..3` | Q2 probes | benign edits of the lead (member order; spare member; cold init code) | loop unchanged (below) | no |

Size (`size_report.py`, -O3, worst over 33 cases): VC1 E0+E1 lead 23,248; c1 23,280; c2b 23,376; c3 23,248; c5 23,356;
c6 23,384 (limit 26,624). Least free stack E1 3,864 in all. E0 images identical to the lead in every variant (the
receiver program is not instantiated on E0).

## 1. Q1: the first-level-ack intake

### What the lead does per pass (tso twin, Y-torus N/S, VC2 off; one packet acked and handled per pass)
```
lw   a0,1188(s9)      count            (group)
lw   a4,4(a5); lw s8,8(a5)  TX CMD, STATUS
andi a0,a0,2047       <- stall on the count (~5-8)
... mul a4,a5,s1; add a4,a4,t1      <- ack's slot address (mul latency)
lbu  a3,43(a4)        src_ch_id of the acked packet
sh2add a1,a3,t3       <- stall on src (~5-7)
lw a4,0(a1); addi; sw (L1 ack count); sw (copy)
lw   a2,40(s2); lbu a3,48(a4); lbu a4,50(a5)   handled packet's word + route bytes (volatile)
zext.b a4,a4          <- stall on the route byte (~5-7)
```
Two L1 latencies in series (the acked packet's source byte, then the handled packet's header), although in steady state
they are the same packet and the same header word.

### Kept headers vs ordering
The kept-header intake (design A s3 / B) overlaps those two loads by reading the acked packet's header once and keeping
it; the cost is state (an ack pointer, a kept table, values live from the ack to the handling) on receivers already at
the register limit (design A §4, B §7: scatter encoding spilled, `me` reloaded per ack, status base reloaded).

The same overlap needs no state: read the *handled* packet's header before acking (the packet is known present:
`available + arrived != 0`), ack, then decode. The header loads then issue before the ack loop and are first used after
it. Nothing new is carried across passes.

Intermediate builds (cov_m/builds): `v4rxc1a_O3` (a `packet_present` rewrite: non-acking images changed, dropped),
`v4rxc1s_O3`, `v4rxc1b_O3`, `v4rxc1c_O3` (the forms below), `v4rxc1_O3` final. Getting GCC to keep the first use
after the acks took three forms (each mismatch explained in EXPECTATIONS.md):
- `c1s` (header read before the ack, `Routing2D::read` unchanged): on tso/tv GCC computes `y != 0 ? y : x` right after
  the loads, before the ack loop: the stall moved, it did not go. On the ring (1D, one route word) it worked.
- `c1b` (`read` split into `load` + `word`, the select after the acks): GCC still emits `zext.b` right after each
  volatile `lbu` (it does not fold an extension into a volatile byte load), so the Y byte is used before the acks.
- `c1c`/final (`v4rxc1`): the header word and the route fields as one group of plain loads between two
  `std::atomic_signal_fence`s (the accepted load-group primitive; it replaces the `volatile` that only served to keep
  both route bytes unconditional, LEDGER E1-19). Final form puts the fences only where there is a group (channels that
  route); `c1c` had them on VC2's single word too and cost 2D VC2 +4 sp (VC2's in-memory cursor stored/reloaded around
  the fence).

tso twin, `v4rxc1`:
```
lw   a7,1188(s8); lw a3,4(a5); lw s7,8(a5)    group
andi a7,a7,2047; or a5,a7,s3; beqz a5,...     <- count stall; one presence test (available + arrived)
lbu  a1,48(a3); lbu t3,50(a5); lw a2,40(s0)   route bytes + word: one group, no zext.b
beqz a7 / neg / slli / sw (register -arrived) / ... mul / add / lbu t5,43(a3) / sh2add (<- src stall) ... bne
beqz a1,...                                   select, after the loop: Y byte long arrived
```
hr twin: `lw s2,40(a4); lw t3,44(a4)` right after the presence branch, first used after the ack loop.

Static (twins, receiver loop: instructions / sp accesses in the loop): tso 473/22 -> 443/8; tv 833/33 -> 814/35 (the
per-packet `me.y` reload is gone; `me.x` still reloaded per packet, VC2's in-memory cursor as before); hr 447/9 ->
448/8; one ERISC (js1e) receiver_pass 622 -> 636 instructions, the group before the ack loop there too. Coverage: see §0.
At -Os: FLA receivers +3..+6 sp, all around the out-of-line delivery calls or per round (ring and torus checked); the
lead's -Os store of the TX status right after its group read is gone on torus.

What the listing does not settle: the select's position after the ack loop is GCC's choice (pure register code may move
across the fences); the loads' position is the fences' contract. The route fields travel from the group to the decode
in a small `Fields` struct (2D: two bytes and the destination mesh); GCC keeps it in registers in every listing checked
(-O3 and -Os, ring and torus), but a record kept in memory there would put a store right after each load (the
V4_RECEIVER_SHAPE §8 pattern); if that ever shows, pass the fields as scalars. The remaining chain per pass on tso is count (~5) ->
mul (~2) -> source byte (~5) -> stack count (~2): the source byte is the ack's own load and stays exposed.

**Answer (static; the gain is a hypothesis):** kept headers cannot add anything over ordering: in steady state the
acked and handled packet share one header word; ordering hides the handled packet's header latency under the acks with
no carried state, while the ack's own source-byte load stays exposed in both designs (it is the first use of a header
the ack must read). Expected on hardware: ring/torus first-level-ack rows -3 to -6 cycles/packet (one L1 latency per
pass), one ERISC the same mechanism, 2D unchanged within noise. **WRONG (job 312, v4rxc1 vs lead, VC2 off: ring 1-hop +6.9, half-ring +5.1, TorusY1HopNS +3.7, its scatter +1.6 cycles/packet; 2D flat.)** A further step, not built: when `available == 0` the
first arrival is the handled packet, so its source could come from the group's header word (removes the `mul` and the
source byte's latency, ~7 cycles, for one branch in the ack loop).

## 2. Q2: placement stability

### Which placements hold by contract, which by GCC's choice

| state | placement in the lead | why | contract or heuristic |
| --- | --- | --- | --- |
| slot records (`SlotRecords` words), credit counts (`completion_counters[]`, `ack_counters[]`) | memory | indexed by slot / sender channel at run time | contract (a runtime-indexed array cannot live in registers) |
| downstream connections (`Downstreams<0/1>`) | memory, reached through `run`'s reference parameters | kernel_main's objects, passed to the out-of-line loop; fields reloaded after calls | contract (escaped object) |
| one ERISC: every receiver object | memory (`ReceiverState` behind `receiver_pass`'s reference; noinline) | the pass is a call over a record | contract |
| VC1's / VC2's cursor (`ChannelState` in `Tables`) | memory | GCC's SRA does not scalarize an aggregate with a runtime-indexed member | **heuristic** (v4rx2 shows the price if it ever flips: +9.4 on torus VC2 scatter) |
| credit block addresses (`*_base_ptr` members) | constants (`lui`/`addi` at use) | GCC propagates the constructor's stores (the object does not escape) | **heuristic** (inside a memory aggregate they were loaded per use, V4_RECEIVER_SHAPE §5) |
| VC0's cursor, `frame_due`, loop counters | registers (callee-saved) | not address-taken, not in a runtime-indexed aggregate | heuristic by nature (register allocation; pinning is banned): the design only makes it possible |
| `me` (coordinates) | t-registers on tso/hr; reloaded from the stack per pass/packet on tv | allocator's choice at the register limit | heuristic |

### Evidence
- -O2: byte-identical to -O3 on every non-VC1 receiver loop (`loopstat.py v4rxdas2_O3 v4rxdas2_O2`: bh_2d, 1d, ring,
  torus, 2d_vc2, one ERISC, express all `ident`); VC1 differs (-42..+1 insns, 0..+3 sp; VC1's logged commands are -O2).
  The DWARF producer says -O2, so the rebuild did take the flag: GCC simply generates the same code at -O2 here.
- Placement kinds (`placeagg.py`, every image): O3 vs O2 and O3 vs Os identical for every receiver local (`cursor0`
  split, `tables` frame, `credits` frame, `frame_due` split) except debug-only kinds (`me` on VC1, the pass counter `n`).
- -Os spills pass-local values heavily: 2D N/S loop body 2 -> 36 sp (header word stored right after its load,
  `frame_due` in memory, `me` per packet, the NoC status base per check); VC1 +27..+61; torus +20 on 4 images. That is
  the -Os allocator, not the state design (the kinds above do not change).
- Benign edits of the lead (`v4rxcj1`: `Position` members reordered; `v4rxcj2`: a spare member in `Cursor`;
  `v4rxcj3`: a noinline init-time helper called before the loop), loopstat/rxcmp on bh_2d, 1d, ring, torus, vc1,
  2d_vc2, one ERISC, express, glx_2d: j1 every loop "regs" only (same mnemonics, offsets/registers renamed), counts
  equal; j2 identical where the cursor is scalarized (VC0), +1 instruction (one zero-init store) where it is in memory
  (VC1, VC2); j3 loop region +0/+0 everywhere (only the prologue grows). The lead's placement is stable under these.
- `v4rxc4` (state the VC1/VC2 placement with `keep_in_memory` before and after each step): VC2's cursor traffic is
  the same set of loads/stores (it already was in memory), but the two asm statements re-lay out the loop: 2D VC2 N/S
  -31 instructions, +4 sp (call-site saves, cold) and one per-packet `me.y` reload gone; E/W +2/+3, -7 sp; VC1 +8/+9 on
  6 images. So stating the contract is cheap in memory traffic but not codegen-neutral.

**Answer:** the memory placements the receiver relies on for correctness of cost (records, counts, downstreams, the
single-ERISC record) are by contract; two are GCC's choice and can be stated with accepted primitives: VC1's/VC2's
cursor (`keep_in_memory` at the step boundaries, `v4rxc4`, needs an A/B on VC2/VC1 rows because the codegen moves) and
the credit block addresses (compile-time constants, design B's `CreditCounts`, not rebuilt here). Register placements
(VC0's cursor, `frame_due`, `me`) cannot be guaranteed within the rules; the evidence says they are stable at -O3/-O2
and under benign edits, and degrade at -Os by allocation, not by placement class. Recommendation: adopt the constexpr
credit addresses (contract at no cost on two ERISCs) and either adopt `v4rxc4` after its A/B or document the SRA
reliance next to `ChannelState` (one comment line).

## 3. Q3: completions

### (a) Readability: `v4rxc3`, byte-identical
`release_completions` is one skeleton for every group size: `due<T>` (whether the oldest uncredited slots go back and
how many), the proof (`writes_leave<T == 1 ? 1 : completion_wait_checks>` of the oldest slot's ID), `credit_slots<T>`
(a single slot's source alone, or a group's run), the count. The one-slot differences are two `if constexpr` arms of a
few lines (`due`: `waiting > progress`, no idle count; `credit_slots`: the slot's source byte) instead of a second
copy of the whole release; `oldest_uncredited<N>` names the ring arithmetic. Getting byte identity needed the lead's
order of evaluation (due as a bool with `n` computed only when due; `idle_passes = 0` before the credit; `uncompleted =
waiting - n`): two earlier forms (`v4rxc3a`, a `due_slots` returning 0..T) changed 2D/VC1/VC2 code by +-14 instructions
and -6..+1 sp (`v4rxc3a_O3` keeps the first form). Final: 2,086 of 2,086 images byte-identical. No device run needed.

### (b) Cost per channel as VCs are added
Per pass, an idle channel's release is a test of its `uncompleted` count (VC2 on tv: `lw a4,84(sp); bgeu`, 2
instructions). The general form's per-packet cost on VC0 that grew with VC2 (v4rx1: `minu`, run check, `mul`) is what
the one-slot form removed (-20 measured). What still grows per channel is the receive's register poll of an idle
channel: VC2's idle step on tv is `lw 72(sp)`, `lw 80(sp)`, `lw 1188(stream)`, `andi` (~6-8 cycles of stall on the
register), `sw 72(sp)` every pass. That is Q4's subject (§4), not the completions'.

### (c) Holding on rings and tori
The one-slot release never holds a credit for a group: a slot goes back one pass after its retire if its writes have
left. But it credits at most one slot per pass and never this pass's, so in steady state (one packet per pass) a single
failed check leaves every later slot one pass late until a pass without a packet: a permanent one-pass lag (§9 of
V4_RECEIVER_SHAPE: 5.5-9.2 cycles/packet per pass of lag). Whether checks fail in steady state is unknown (no count of
failed checks exists); §10's latencies (4352 B write 101-123 cycles to leave; the check ~150+ cycles after the issue)
say occasionally, more on the half-ring (forward + local write on one NIU).

`v4rxc5` = c3 + catch-up for single-slot groups: `release_completions` repeats `release_group` while it releases
(every due slot whose check passes, oldest first, stopping at the first failure); groups of several slots unchanged.
Static: 2D, 1D byte-identical to the lead; FLA loops +7..+21 instructions; on tso the steady-state path is +3
instructions per pass (`li`, `mv`, `bltu` back-edge test); hr no new per-packet stack access; T==1 VC1/VC2 rings (rings
under 8 slots) change too. A cheaper proof was looked for and not found: the proof is one NIU status read whose stall
(~6-8) is exposed because nothing independent is left at that point; reading it earlier samples earlier (fails more for
4352 B writes, §10); proving the newest slot to credit older ones would assume NIU write ordering across command
buffers (not established).

**Answer:** adopt `v4rxc3` (readable, identical). The release does not hold credits on rings; its one weakness is the
lag after a failed check. `v4rxc5` removes it at ~+3 instructions per pass: hypothesis, decided by an A/B (gain only if
checks fail in steady state; otherwise +0 to +2 cycles/packet), with the perturbation sweep (it changes credit timing). **WRONG (job 313, v4rxc5 vs lead, VC2 off: ring 1-hop +11.7, half-ring +12.4, TorusY1HopNS +11.4, its scatter +8.9; one ERISC TorusY1HopNS +3.1; 2D flat.)**

## 4. Q4: arrival registers in one group

- `v4rxc2` (every register + TX status on every router): first-level-ack routers without VC2 byte-identical (the group
  is the lead's). 2D: the group's TX status held in a callee-saved register through the step: E/W -10 sp, N/S +4..+12
  sp on the scatter path (multi-chunk command-buffer waits). tv (Y-torus + VC2): VC2's count loaded in the group into
  `s3`, first used at VC2's step with no stall (its idle poll's ~6-8 stall gone), but the scatter chunk encoding is now
  stored right after its header load (`sw a2,8(sp)` at router_local_delivery.hpp:102), v4rx2's measured loss mechanism.
- `v4rxc2b` (every register; the TX status only where a channel acks): 2D -1 instruction, 0 sp on all 28 E1 images
  (E/W: `lw a5,1188(s6)` each pass at the group, then the lead's `available == 0` test; the lazy read's own `lui` gone);
  1D -1/0; 2D VC2 -14..+14, -1 sp; VC1 -6..+32, 0..+6 sp (8 images -3..-6); FLA without VC2 identical; tv identical to
  c2. Per pass on 2D: one more register load issued, dead when `available > 0` (~1 cycle), the same stall as today when
  the kept count is used up.
- -Os (`loopstat` against `v4rxdas2_Os`): c2b FLA images identical; 2D -1..-3 insns, 0/-4 sp; 2D VC2 -3..+2, -3..+2 sp;
  VC1 mostly +1 sp, two images +11 sp. c6 at -Os: c1's and c2b's effects added.
- `v4rxc6` (c1 + c2b): tv: VC2's count in `s6` across VC0's step, no scatter-encoding spill; a `sw t0,8(sp)` before and
  `lw t0,8(sp)` after the ack loop (me.x around the loop, not a store-after-load); `me.y` still per pass.

**Answer:** one rule for the arrival registers (`v4rxc2b`: every channel's register in the pass-start group, the TX
status only with first-level acks) reads better (no "FLA reads in the group, others lazily") and removes the idle poll's
stall of every additional channel (VC2, VC1). Costs, hypothesis: 2D write unchanged (sender-bound), 2D scatter +0 to
+1; **WRONG (job 312, v4rxc2b vs lead: 2D scatter -2.4, 142.0 -> 139.6; 2D write -0.3; FLA rows identical.)** Y-torus VC2-on write -4 to -8; VC2-on scatter about neutral alone (encoding spill) and better combined with c1
(`v4rxc6`). **WRONG (job 314, VC2 on, vs lead: v4rxc2b TorusY1HopNS write -0.7, its scatter -4.2, 2D E/W scatter -7.0; v4rxc6 write -0.2, scatter -3.7, 2D E/W scatter -6.9. VC2-on builds run VC0 at 43/7/11 slots, V4_RECEIVER_SHAPE §21.)** The TX status should not join the group on routers without acks (c2's 2D N/S spills).

## 5. Other readability findings

- The fenced load group (c1) replaces `volatile` route-byte reads whose only purpose was keeping both bytes
  unconditional; GCC never folds an extension into a volatile byte load, so each volatile `lbu` carried a `zext.b`
  right behind it (2D N/S: one instruction per packet, and it pins the first use next to the load).
- `Routing::load` / `Routing::word` name the two phases of the route decode (and the landing / exit steps sit in the
  phase they belong to).
- `release_completions` skeleton (c3) and `oldest_uncredited`.
- Credit block addresses as constants (design B's `CreditCounts`): removes the heuristic constant propagation and the
  dummy channel-index constructor of the shared credit sender type.
- C1 removes `Routing::read` (no user left); the 1D `Fields` is the routing word itself.

## 6. Device A/Bs proposed (none run)

Validation (13 tests) first for every variant that changes images (c1, c2b/c6, c5, c4).
1. `v4rxc1` vs lead, row-first, two repetitions: ring 1-hop, half-ring, TorusY1HopNS write and scatter (VC2 off and
   on), one-ERISC TorusY1HopNS write, 2D 1-hop write, 2D scatter. Expect -3 to -6 on the FLA rows, 2D flat (hypothesis). **WRONG (job 312, v4rxc1 vs lead, VC2 off: ring 1-hop +6.9, half-ring +5.1, TorusY1HopNS +3.7, its scatter +1.6 cycles/packet; 2D flat.)**
2. `v4rxc6` vs `v4rxc1` (isolates c2b on top of c1), same rows plus the VC1 split / inter-mesh rows and the VC2 census
   suite (`sh3_vc2_runs.sh`). If c1 loses, run `v4rxc2b` vs lead instead.
3. `v4rxc5` vs lead (c3 = lead images): ring 1-hop, half-ring, TorusY1HopNS write/scatter VC2 off/on; then the
   perturbation sweep and burst benchmark if it gains.
4. Optional: `v4rxc4` vs lead on TorusY1HopNS VC2 on, 2D VC2 census, VC1 split.
5. Profile proposal: PC samples of the lead and `v4rxc1` on TorusY1HopNS scatter (VC2 off) and the half-ring
   (`torusns_scatter.yaml`, `halfring_pcs_runs.sh`) to see the header's first-use stall move under the acks.
No run needed for `v4rxc3` (byte-identical).
