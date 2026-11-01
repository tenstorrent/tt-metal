# Design A: expectations (written 2026-10-31, before any build of the design; not edited afterwards)

Base: v4rx1 (`v4rx1_O3` coverage build; twins of the profiled configs built from v4rx1 under
`/home/aagarwal/.tt-metal-v3r-builds/design_a/tw_v4rx1_*`). Counts below are for the receiver loop function
(`receiver::run`, two ERISCs; `single::receiver_pass`, one ERISC) as `design_a/rxcmp.py` counts them: all instructions,
and sp-relative loads/stores without the prologue's/epilogues' callee-saved saves ("fn"); the same inside the loop's
strongly connected region ("loop"). One-ERISC: fn only (the loop is in `single::run`; `receiver_pass` reaches its state
through a pointer, so its sp count is spills only).

Calibration used (earlier builds against v4rx1, rxcmp): 1D routing word (v4rx1rg -> v4rx1r1): line -2 insns / -2 sp,
ring -3 / -4. Splitting `handle` out of `receive` (v4rx1j): +5 insns on the 8 bh_2d N/S configs. v4rx2 (every cursor a
loop local): bh_2d_vc2 -26/+3, -13/-2, +1/-6, +1/-5; bh_vc1 -35/-8 to -6/-10.

v4rx1 baselines (fn insns, fn sp, loop sp): bh_2d 568-676, 22-28, 6-12; bh_1d 573, 38, 18; bh_ring 517, 28, 11;
bh_torus 524-642, 28-30, 11-12; bh_2d_vc2 886-1037, 61-63, 37-38; bh_vc1 671-1410, 31-72, 6-41; bh_2d_1erisc
566-656, 5-7; bh_torus_1erisc 521-622, 10-14. Profiled twins: rx1tv (Y-torus, VC2 on) 963 / 64 / 38; rx1tso (Y-torus,
VC2 off) 566 / 40 / 22; rx1hr (ring) 517 / 28 / 11.

## 1. Receiver state

### Stage 1 (shape only; placement as v4rx1)
| object | members | placement class (intended) |
| --- | --- | --- |
| `cursor0` (VC0's `Cursor`, loop local) | `position` {available, slot, index}, `counts` {uncompleted, idle_passes} | left to the compiler: registers |
| `frame_due` (loop local) | bool | registers |
| `tables` (`Tables`, frame object of `run`) | `records0` (VC0's `SlotRecords`), `vc1` / `vc2` (`ChannelState`: cursor + records, one record per channel) | memory (stack) |
| `credits` (`Credits`, frame object) | ack and completion counts (memory), block addresses (constants) | memory / constants |
| `me` | y, x, mesh_id | left to the compiler (v4rx1 keeps y/x on the stack on rx1tv, in caller-saved registers on rx1tso) |
| pass group (transient) | `arrived0..2`, `txq` raw | registers; `txq` held to the frame |
| `round`, `n` | round counters (memory, by reference to `end_round`), pass count | memory / register |

### Full design (S1-S4)
| object | members | placement class (intended) |
| --- | --- | --- |
| `c0`, `c1`, `c2` (`CursorOf<CH>`, one loop local per channel; `Absent` where the channel does not run) | `position` {available, slot, index, ack_slot}, `counts` {uncompleted, idle_passes} | left to the compiler, every channel |
| `frame_due` | bool | registers |
| `tables` (`Tables`, frame object) | `ch0..ch2` (`ChannelTable<CH>`): `sources` (`SlotRecords<N>`), and `kept[N]` (`KeptHeader` {word, route}) on channels with first-level acks | memory (stack, base sp) by design |
| `credits`, `me`, `round`, `n`, pass group | as stage 1 | as stage 1 |
| one ERISC: `single::ReceiverState` | {c0, c1, c2, tables, credits, frame_due} | memory, through `receiver_pass`'s pointer |

Dead members expected to vanish where the cursor is in registers: `ack_slot` on channels without acks; `idle_passes`
on channels with one-slot groups (T == 1: first-level-ack channels, VC1's and VC2's 4-slot rings). In memory (VC1/VC2
before S4, one ERISC) they stay as stores (v4rx1 VC2's two `sw zero, idle_passes`).

### Loop-carried values (two ERISCs; constants not counted)
| config | v4rx1 | stage 1 | full | expected placement (full) |
| --- | --- | --- | --- | --- |
| bh_2d, bh_1d (VC0, T = 4) | 5 cursor + frame_due + n = 7 | 7 | 7 (`ack_slot` dead) | all registers, as v4rx1 |
| bh_ring, bh_torus, rx1tso, rx1hr (VC0 acks, T = 1) | 4 cursor (idle dead) + frame_due + n = 6, `txq` pass-scoped | 6 | 7 (+ `ack_slot`) | registers; `txq` in a callee-saved register to the frame (as v4rx1's s9/s6/s8) |
| bh_2d_vc2 (VC0 T = 4, VC2 T = 1 delivers) | 7 (VC2's cursor in memory) | 7 | 11 (+ VC2's available, slot, index, uncompleted) | VC2 in callee-saved registers; spills elsewhere likely (v4rx2) |
| bh_vc1 (VC0 T = 4, VC1 T = 1 routes) | 7 (VC1 in memory) | 7 | 11 | as bh_2d_vc2; v4rx2 measured fewer stack accesses here |
| rx1tv (VC0 acks T = 1, VC2 T = 1) | 6 + `txq` (VC2 in memory) | 6 + `txq` | 11 + `txq` | at the register limit: a spill on VC0's path is likely (the step-2 risk) |

## 2. Per config class: image class and receiver-function deltas against v4rx1

E0 (sender) images: byte-identical in every two-ERISC config at both stages (nothing E0 compiles changes: the builder
arg array, `kernel_main`'s receiver branch and the receiver headers are not reached by the sender role).

| class | stage 1 | full design |
| --- | --- | --- |
| bh_2d | identical or layout on most; 0 to +5 insns on the N/S configs (the `handle` split, as v4rx1j); sp 0 | same as stage 1 (T = 4 untouched, no acks, `ack_slot` dead) |
| bh_1d | -2 insns / -2 sp (routing word; the struct copy kept on the stack for the sparse-multicast call goes), +-3 | same as stage 1 |
| bh_ring | -3 / -4 (routing word) +-3; ack loop unchanged | S2: -2 to -6 (byte load of the source, no `minu`/group arithmetic; GCC had folded most of it). S3: ack loop (with its `mul`) and the handle-time header loads go (~-36), one ack (+~30), kept loads/stores and the select (+~8): net -5 to +20 insns; sp +4 by design (kept: 2 loads per pass, 2 stores per ack) +-2. Total full: -10 to +15 insns, loop sp +2 to +6 |
| bh_torus | 0 to +5 (no routing-word change in 2D; `handle` split on N/S) | as bh_ring without the 1D part: -5 to +20 insns, loop sp +2 to +6 |
| bh_2d_vc2 | identical or layout (VC2's step: same operations, record in memory) +-5 | S2 on VC2: -3 to -6 insns, -2 to -3 sp (idle stores, word load + shift -> byte load). S4: VC2's cursor scalarized, about v4rx2's -26..+1 insns, -6..+3 sp. Total: -35 to 0 insns, fn sp -9 to +1; VC2's step memory accesses gone, a few new spills on VC0's path |
| bh_vc1 | intermesh (non-edge) configs identical or layout; edge configs (exits now resolved inside `read`): -8 to +5 insns, sp -2 to +2 | S2 on VC1 (4 slots): -3 to -6, -2 sp. S4: VC1 cursor scalarized, about v4rx2's -35 to -6 insns, -13 to -8 sp. Total: -40 to -5 insns, sp -15 to -6 |
| bh_2d_1erisc | identical or layout; 0 to +5 | `receiver_pass` same as stage 1 (+-3); `single::run` +1 store (`ack_slot` init) |
| bh_torus_1erisc | identical or layout +-5 | S2 -2 to -6; S3 as v4rx1rg/r1 minus their end block: -10 to +20 insns, sp -5 to +2 (state through the pointer; kept table inside the record, not sp) |

Profiled twins (full design, fn insns / loop sp against v4rx1):
- rx1tso (Y-torus, VC2 off): -5 to +20 insns; loop sp 22 -> 24-28 (kept +4; the scatter-path spills of v4rx1 stay).
- rx1hr (ring): -10 to +15 insns; loop sp 11 -> 13-17.
- rx1tv (Y-torus, VC2 on): S2 removes VC0's general-form release (~-20: `minu` x2, `credit_slots`'s run check and its
  `mul`), VC2's idle stores; S3 as rx1tso; S4 removes VC2's in-memory state traffic (~20 sp accesses in VC2's step) and
  adds spills elsewhere: fn -40 to -10 insns, loop sp 38 -> 20-34.
- rx1js1e (one ERISC, Y-torus N/S, receiver_pass ~622 insns): -10 to +20 insns.

## 3. What the hot listings should show (full design)

Two ERISCs, first-level acks (rx1tso, rx1hr, rx1tv), per pass:
1. Pass group: the packets-received register (raw) and the TX queue's CMD and STATUS reads issued back to back between
   the signal fences, no masking or test inside the group. The count's first use is its own branch
   (`andi 2047; beqz`); the status is first used at the frame test after the last step (`slli ..,15; bgez/bltz` on
   the raw word), held in a callee-saved register across the pass on rx1tso/rx1hr. On rx1tv (S4: four more
   loop-carried values) the status, or a cursor value, is likely stored to the stack right after its read: if so, a
   store right after a load in the group (v4rx2's mechanism) - the known step-2 risk.
2. VC0's step start: the two kept loads (`sh3add` index; two `lw` off sp) before the count branch, overlapping the
   group's latency (if GCC sinks them behind the branch, their latency shows at `handle`'s first use instead).
3. The ack (count non-zero): header word and route loads from `ack_slot` back to back (1D: `lw 40`, `lw 44`; 2D N/S:
   both route bytes as volatile `lbu`, then the select), no fence, no `mul` (the slot address is the `ack_slot`
   pointer; `ring_add` for the kept index is add + compare/sub, or `and` for 32 slots). Their first use is the kept
   stores (2 `sw` off sp) and `src_ch_id` (`srli 24`) for `add_ack`: one exposed L1 latency per acked packet (as
   v4rx1r1, 7.1 at its ring store). `ack_slot` advances with compare-to-last + add, the stream register gets -1.
4. The select of the handled packet's word/route (`available` was 0): a branch or two `mv`s, no stack traffic.
5. `handle`: route tests on the route word (`andi`), admission, forward/deliver, retire (`sb` of the source into the
   records, available-1, index/slot advance). No store right after a load on this path other than the stores whose
   value is a header field by necessity (the kept record, the NoC command registers).
6. Release (T == 1, written out): due test (`uncompleted > progress`), `ring_add` for the oldest slot, one
   transaction-ID status read, `lbu` of its source, the completion count (lw/addi/sw + the L1 block sw), uncompleted-1.
   No `minu`, no `mul`, no `of_run` shift/mask.
7. Frame: the raw status bit test; only when it reads busy, a fresh `eth_txq_is_busy` read; then the four TXQ writes.

No `mul` anywhere in the per-packet path of ring/torus receivers. On 2D VC0 (T = 4) `credit_slots`' run check keeps
its `mul` (unchanged from v4rx1).

One ERISC (rx1js1e): the state loads from the record (c0's fields, `me` through its pointer) precede the header loads at
the ack, as v4rx1rg (§15: route-byte loads wait behind the ack pointer and `me`); kept table accesses through the
record pointer, not sp.

## 4. Placement, size

- `placement.py` (DWARF kinds, by variable name): stage 1 same placement as v4rx1 on every image (cursor0 split,
  frame_due split, tables / credits / round frame). Full: `cursor0` replaced by `c0` (split); `c1`/`c2` split where
  the channel runs (bh_vc1, bh_2d_vc2, rx1tv), absent elsewhere; `tables`, `credits`, `round` frame.
- Size: E0 unchanged. VC1 E0 + E1 within +-200 B of v4rx1's 23,264 (well under 26,624). E1 stack: `run`'s frame grows
  by the kept table on first-level-ack routers (N x 8 B: 128 B torus, 256 B ring, 88 B rx1tv); least free E1 drops by
  up to ~256 B from 3,864 (still > 3,500).
- Coverage: every config v4rx1_O3 builds builds (1,666 / 1,666 of the shared configs; all non-`_sc` configs of the
  current case list); the `*_sc` configs fail on the counter-credit static_assert as in every v4 variant.
