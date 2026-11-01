# Design B: expectations (written 2026-10-31, before any compile of the design; not edited afterwards)

Reference: v4rx1_O3 coverage images (1,666), read with `loopstat.py` (this directory) and `listing.py` before writing
this. Metric: the receiver loop function (`receiver::run` on E1, `single::receiver_pass` on one ERISC), its
instructions, and its sp-relative accesses split into callee-saved saves/restores, zero-init stores, and **body**
(everything else: spills, reloads, frame objects addressed directly off sp). Records and credit arrays reached as
`addi rX,sp,N` + indexed access are memory by design and are not counted as body accesses.

## v4rx1 baseline (from the existing build)

| class (case) | insns | body | s-regs | frame | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 2D mesh (bh_2d) | 568-676 | 2-12 | 13 | 160 | VC0 16 slots, groups of 4, 2 `mul` (credit_slots) |
| 1D line (bh_1d) | 573 | 22 | 13 | 160 | routing fields struct copied to the stack |
| ring (bh_ring) | 517 | 11 | 13 | 160 | first-level acks; 1 `mul` (ack loop's header address) |
| Y-torus (bh_torus) | 524 (E/W) - 642 (N/S) | 12-14 | 13 | 160-176 | first-level acks; me.x spilled, reloaded per packet; TX status kept raw in s9 to the frame |
| 2D + VC2 (bh_2d_vc2) | 886-1037 | 36-38 | 13 | 192-208 | VC0 11 slots T=2; VC2 4 slots T=1, its cursor in `Tables` (all its accesses sp-relative) |
| VC1 (bh_vc1) | 671-1410 | 8-38 | 13 | 176-224 | VC1's cursor in `Tables`; two Downstreams pointers |
| one ERISC 2D (bh_2d_1erisc) | 566-656 | 4 | 1-2 | 32 | state in `ReceiverState` (memory); `me` by reference (byte loads) |
| one ERISC torus (bh_torus_1erisc) | 521-622 | 6 | 2-3 | 32 | |

## What the two stages contain

- **Stage 1 (v4rxdbs1), shape only:** VC-indexed forwarding flags; `Channel<CH>` with every feature (vc, serviced,
  routes from the VC's flag, first-level acks per VC, downstream_vc, via_relay, Routing incl. a no-routing policy for
  channels that only deliver); one generic `step<CH>` / `receive<CH>` / `act<CH>` / `retire<CH>` over
  `std::index_sequence<0, 1, 2>`; `Coordinates` by value; `DownstreamSet`; the 1D routing word and the 2D edge
  route word (action | dst_mesh << 8); `CreditCounts` (constant block addresses) inside `Tables`; `try_send_frame`.
  Kept from v4rx1: the ack loop (every arrival acked each pass, header address by `mul`), the pass register group as
  v4rx1 reads it (count, TX status), the generic completion rule, VC0's cursor a loop local and VC1's/VC2's cursors
  in memory (members of `Tables` in stage 1).
- **Full design (v4rxdb):** stage 1 plus: one ack per pass from `ack_slot` (no `mul`), header word and route kept per
  slot (`Records<CH>::kept`, on run's frame), the `ready` word (arrival bit per acking channel + TXQ_FREE), the
  `T == 1` completion form, every channel's cursor a loop local, `keep_in_memory(cursor)` after each forward, the
  one-ERISC `ReceiverState {c0, c1, c2, tables, frame_due}`.

## State per config class (full design; stage 1 differences in brackets)

Placement classes: **R** = left to the compiler (register candidate), **M** = in memory by design, **K** = constant.

| object (members) | 2D mesh | 1D line | ring | Y-torus | 2D+VC2 | VC1 | one ERISC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `c0` Cursor<0> {available, slot, index, uncompleted} | R | R | R | R | R | R | M (ReceiverState) |
| `c0.idle_passes` (groups > 1 only) | R | R | - | - | R | R | M |
| `c0.ack_slot` (first-level acks only) | - | - | R | R | - | - | M (torus) |
| `c1` Cursor<1> (VC1 only, 5 values) | - | - | - | - | - | R [M] | M |
| `c2` Cursor<2> (VC2 only: 4 slots, T=1, 4 values) | - | - | - | - | R [M] | - | M |
| `t.records<CH>().source` (byte per slot) | M | M | M | M | M | M | M |
| `t.records<0>().kept[N]` {word, route} | - | - | M (128 B) | M (8 B x N) | - | - | M |
| `t.credits` {completions[], acks[]} | M | M | M | M | M | M | M |
| CreditCounts block addresses, ring bounds, stream ids | K | K | K | K | K | K | K |
| `frame_due` | R | R | R | R | R | R | M |
| `ready` (pass-local) | K=0 | K=0 | R | R | K=0 | K=0 | R (per call) |
| `me` (4 B) | R (bytes) | unused | unused | R (bytes) | R | R | R (argument) |
| `ds` DownstreamSet& (kernel_main's) | R ptr, M obj | same | same | same | same | R ptr (one, was two) | same |
| pass counter `n`, `round` | R, M | R, M | R, M | R, M | R, M | R, M | caller's |

Loop-carried values (variant, excluding invariants `ds`, `me`, constants) on E1, v4rx1 -> full design:

| | 2D mesh | 1D line | ring | Y-torus | 2D+VC2 | VC1 | Y-torus + VC2 (rx1tv) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4rx1 (in registers; + in memory) | 7 | 7 | 6 | 6 | 7 + 4 M | 7 + 5 M | 6 + 4 M |
| stage 1 | 7 | 7 | 6 | 6 | 7 + 4 M | 7 + 5 M | 6 + 4 M |
| full design | 7 | 7 | 7 (+ack_slot) | 7 | 11 | 12 | 11 |

(cursor fields + `frame_due` + `n`; with first-level acks `idle_passes` is absent/dead; v4rx1 also carried the raw TX
status from the group to the frame, a pass-local value like `ready`.)

Registers vs spills expected (full design):
- 2D mesh, 1D, ring, Y-torus without VC2: everything R stays in registers as in v4rx1 (13 s-regs used, as now); the
  extra `ack_slot` on ring/torus fits (it replaces the ack loop's temporaries and the TX status kept across the pass,
  and `kept` is addressed off sp, so no ring-base register as in v4rx1rg/r1). `me.x` stays spilled on Y-torus N/S as
  in v4rx1 (one reload per packet).
- 2D+VC2, VC1: VC1's/VC2's cursor values go to registers, as in v4rx2; GCC runs out of callee-saved registers, so some
  VC0 per-packet values spill (v4rx2 had +4..+8 body accesses on bh_2d_vc2). Expected the same as v4rx2's images
  on these cases within a few accesses; VC1 slightly better than v4rx2 (one `ds` pointer instead of two).
- Y-torus + VC2 (rx1tv): the step-2 placement risk. 11 variant values + `ready` + `me` bytes + `ds` + constants
  exceed the 12 callee-saved registers; expected: VC2's step writes 3-4 callee-saved registers and VC0's per-packet
  path spills 2-6 values (delivery NoC coordinates, scatter encoding, as v4rx2), body accesses above v4rx1's.
  `ready` itself: stored to the stack right after the group if GCC runs short (50/50), else kept in a register to the
  frame.
- One ERISC: state in memory by design; per call GCC loads the cursor fields once and stores them back; no spills
  beyond v4rx1's (body 4-6).

## Instruction and stack-access deltas against v4rx1 (receiver loop function; d_insns, d_body)

| case(s) | stage 1 | full design | why |
| --- | --- | --- | --- |
| every E0 image (two-ERISC cases) | .text identical | .text identical | the receiver program is not instantiated on E0 |
| bh_2d, glx_2d, glx_express_mesh, bh_2d_mux/_telemetry/_bwtelemetry/_trim/_codeprof/_allinstr | 0 +-2, 0; layout (credit arrays move into Tables: new sp offsets) | 0..+3, 0 | full: `keep_in_memory` after forwards changes scheduling only (ds is caller memory already) |
| bh_1d, glx_1d, bh_1d_* | -2, -2 (routing word, as v4rx1r1) | same as stage 1 (+0..+2) | |
| bh_ring, glx_ring | -2..+2, -2..0 | +5..+45, -3..+2; `mul` 1 -> 0 | one ack per pass + kept headers replace the ack loop; T==1 form; prologue zeroes `kept` (128 B: up to 32 zero stores, or a memset call) |
| bh_torus, glx_torus, glx_express, glx_express_line | 0 +-2, 0; layout/regs | +5..+45, -3..+3; `mul` 1 -> 0 | as ring; N/S more than E/W |
| bh_2d_vc2 | 0 +-2, 0 | like v4rx2 (-26/-13/+1 insns, +8/+4/0 body) minus 2..6 insns (VC2's T==1 form drops its dead `idle_passes` stores) | VC2 cursor scalarized |
| bh_vc1 (VC1 serviced) | -3..+3, -2..0 (one ds pointer; edge routers unpack the route word) | -38..-3, -9..0 (v4rx2-like plus one pointer) | VC1 cursor scalarized |
| bh_vc1 (the 8 images v4rx2 left unchanged) | -3..+3, -2..0 | as stage 1 | |
| bh_neighbor (speedy, delivers only) | 0, 0 | 0..+2, 0 | same code through the no-routing policy |
| one ERISC: 2D, 1D, ring, mux, neighbor, udm, trim | -4..+2, 0 | as stage 1 | `me` by value: byte extracts instead of byte loads; one ds pointer |
| bh_2d_vc2_1erisc | -4..+2, 0 | -8..0, 0 | VC2's T==1 form |
| bh_torus_1erisc, glx_express*_1erisc, bh_ring_1erisc (FLA) | -4..+2, 0 | -5..+20, -4..+2 | FLA rewrite on in-memory state (no kept zeroing here: that is in single::run) |

Byte classes: stage 1 makes almost every E1 image differ in bytes (Tables layout); "layout" or "regs" on 2D mesh,
torus, ring, VC2, neighbor; "differ" on 1D (routing word), VC1 edge routers, one-ERISC images.

## Placement (`placement.py`, DWARF kinds of run's locals)

- stage 1: same kinds as v4rx1 for every local that survives (`cursor0`->`c0` split, `tables`->`t` frame,
  `frame_due` split); `credits` disappears (inside `t`); `me` split.
- full: `c1`, `c2` become split (scalarized) where present; `t` frame; nothing else moves.

## Size (`size_report.py`)

- VC1 E0 + E1: E0 unchanged; E1 smaller (v4rx2-like code) -> at or below v4rx1's total, well under 26,624 B.
- E1 stack: first-level-ack routers grow run's frame by `kept` (8 B x VC0 slots, 128 B at 16) in the full design;
  one-ERISC FLA routers grow single::run's frame the same way. Least free E1 stack 3,864 -> about 3,700-3,760
  (v4rx1rg with the ring in .bss: 3,712).

## Hot listings (full design; debug twins of the profiled builds)

All four: no `mul` computing a slot or header address anywhere in the loop (the only `mul`s left are credit_slots'
each-slot identity on channels with groups > 1).

**Y-torus, VC2 off (rx1tso, E1 hash 16246661628456555625):**
- pass start: `lw` TX CMD, `lw` TX STATUS, `lw` VC0 packets-received (raw) issued back to back (address constants
  hoisted); then the reduction (count mask/test, status bit test, or) right after the fence: the TX status's first use
  is there, behind the count read issued after it; no store of either raw value.
- step 0: the two `kept` loads (sp-indexed) before the arrival test; take_in: `lw` header word (+40) and the two
  route-byte `lbu`s back to back (volatile), then the `ack_slot` advance (compare with the last slot, add the slot
  size: no `mul`), the ring index for `kept`, and only then the `kept` stores (no store right after its load);
  `src_ch_id` from `word >> 24` (no `lbu` of header+43); one stream-register decrement; the ack count update
  (local `lw`/`addi`/`sw` + L1 `sw`).
- receive: no header loads for the route (word/route come from `kept` or take_in); `me.x` reload from the stack as in
  v4rx1 (one per packet) is the only body access on the per-packet path.
- release (T==1): compare, `ring_add` (and/compare), one transaction-ID status read, `lbu` source byte, credit; no
  `minu`, no group run test.
- frame: `ready` bit test (register), TX busy read only when the group saw it busy, four TXQ writes.

**Y-torus, VC2 on (rx1tv, E1 hash 1960938547760765240):** as rx1tso for VC0, plus VC2's step with its cursor in
registers; expected spills on VC0's per-packet path (2-6 spill/reload pairs; delivery coordinates or the scatter
chunk encoding, as v4rx2) and `ready` possibly stored right after the group; VC0's T==1 release lean (v4rx1's VC2-on
build had the general `minu`/`mul` form: -10 or more instructions there).

**Ring (rx1hr, E1 hash 16383455491875178035):** as rx1tso with the 1D route: take_in loads the header word and the
routing word (plain load) back to back; the routing word stored to `kept` without a stack copy; `before_forward`
shifts the word and stores it to the header (no extension-word refill path on these builds).

**One ERISC Y-torus (rx1js1e):** `me` arrives in an argument register (no loads of `me`); the cursor fields and
`ack_slot` load from `ReceiverState` at the call's start; the header loads wait only on `ack_slot`'s local-memory
load; cursor fields stored back once per call.

## Hardware (for the user's runs; not measured here)

- Expected to win: VC2-on Y-torus rows' VC0 completion form (the T==1 lean release; §17's -9.6 came from it), the
  first-level-ack rows' ack side (v4rx1r1's ack/header saving, without the end block's cost).
- Expected risk: Y-torus + VC2 rows from VC2's scalarized cursor (v4rx2: +9.4 on scatter), possibly cancelling the
  gain above; 2D/1D/VC1 rows: within noise.
