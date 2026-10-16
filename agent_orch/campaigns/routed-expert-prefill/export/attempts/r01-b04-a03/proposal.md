# r01-b04-a03: 3-slot gate/up input rings (x, gate, up) and down rings (down weights, activated) when L1 has room, on top of the r01-b01-a02 code

## Motivation
This branch (b04) still holds root-level kernels. Its two nodes regressed: a01 read-ahead was 0.933,
and a02 widened K-blocks, which failed the accuracy gate. The best measured code is r01-b01-a02
(1.4716: expert-parallel row groups plus row-shared x tilize). This node starts by **porting
r01-b01-a02's op directory verbatim** (unified_routed_expert_ffn/*, the same five files). That
undoes the parent's widened K-block and the grandparent's read-ahead. It then adds one mechanism on
top.

Where the r01-b01-a02 time goes, from its run log (weights + x read + output write, per call):

| shape | experts/group | total DRAM GB/s |
|---|---|---|
| kimi-k2.7 | 3 | 434 |
| dsv3 | 2 | 416 |
| glm | 2 | 413 |
| dsv4-pro | 3 | 412 |
| dsv4-flash | 2 (K=128) | 378 |
| minimax-m3 | 1 | 378 |
| kimi-k3 | 7 (w=8) | 377 |
| mistral | 1 (K=128) | 360 |
| gpt-oss | 2 (bias) | 186 |

kimi-k2.7 proves that this access pattern can sustain >= 434 GB/s. The shapes at 360-380 GB/s have
fewer or shorter experts per group. On every shape the cores run at ~31-33 cycles per tile-mult,
about 2x the LoFi rate. r01-b01-a02's profile found that groups doing identical work finish 6-14%
apart at random, which points to contention and jitter rather than a schedule bug.

The pipeline is lock-step on two axes, with only 2 slots of slack on each:
- **x (in0)**: each gate/up K-block is tilized by one rotating owner per M-row and multicast to
  all 11 row cores. A row advances only as fast as its slowest core frees a `cb_in0_x` slot.
- **weights (in1)**: the column sender's two DM RISCs read gate (NoC 0) and up (NoC 1), then
  multicast to the group's other row. `cb_in1_gate/up` hold 2 blocks.
- **down**: `cb_in1_down` and `cb_in0_down_full` are both 2 blocks deep and reserved together per
  down K-block.

With a 2-slot ring, a stall on any one core (a DRAM read hit by bank contention, an owner core
doing its tilize, a sender doing its multicast) spreads to its row and column within a block. A
third slot lets producers run up to two blocks ahead. That absorbs this jitter and lets the
senders' DRAM reads for later blocks start while the slowest consumer is still behind. It
also lets the reader finish its gate/up loop a block earlier, so it reaches the down phase's first
weight read sooner, which overlaps the gate/up -> down transition.

## Mechanism
Host (`unified_routed_expert_ffn_program_factory.cpp`):
- The `cb_footprint_bytes(M, w)` fitter is unchanged, so (per_core_M_max, in0_block_w_gu) is
  exactly the baseline/r01-b01-a02 pick on every shape. The K-block width is an accuracy knob
  (r01-b04-a02) and stays untouched.
- After the fit, deepen the rings only if the extra slots fit the same L1 budget:
  1. `gu_slots = 3` (cb_in0_x, cb_in1_gate, cb_in1_up) if it fits;
  2. then `d_slots = 3` (cb_in1_down, cb_in0_down_full) if that still fits.
  Otherwise the rings keep 2 slots, which is today's layout byte for byte. `cb_footprint_bytes`
  takes the slot counts, and the `check_ring` invariants use them.
- Pass the slot counts to the writer as defines (`IN1_GU_SLOTS`, `IN1_D_SLOTS`). The writer never
  pushes cb_in1_up/down, so it indexes their slots off a counter modulo the ring depth (currently
  the constants `kUpNumSlots = 2` / `kDownNumSlots = 2`).

Kernels:
- writer: `kUpNumSlots` / `kDownNumSlots` come from those defines (default 2).
- reader: no logic change. The gate-slot reuse wait `mcast_done >= mc_seq - 2` is conservative and
  remains correct for depth >= 2. Block mc_seq reuses the slot of block mc_seq - S, and
  mc_seq - 2 >= mc_seq - S. Only the comment is updated.
- compute: no change. It waits for and pops one block at a time.

Every multicast still targets the sender's own slot address. Receivers hold the same slot because
every core of a group column (weights) or of an M-row (x, activated) pushes the identical block
sequence, and that holds for any ring depth.

Expected depths with my replica of the fitter (exact budget unknown): dsv3, kimi-k2.7, glm, mistral,
dsv4-flash and gpt-oss get gu_slots = 3 and probably d_slots = 3. kimi-k3 (M_max 8, w 8) probably
gets gu_slots = 3. dsv4-pro and minimax-m3 have too little L1 left and probably stay at 2/2
(minimax may get d_slots = 3).

## Why this is not a repeat
- r01-b03-a01 and r01-b04-a01 issued the next block's DRAM read BEFORE the current block's
  multicast on the same RISC. That delayed the critical multicast and lost 5-7%, and both ran on
  the root schedule. Here every RISC keeps its per-block order (read k -> multicast k -> read
  k+1). Only the slack between producer and consumer grows, and this runs on the row-group
  schedule, where the time is jitter- and DRAM-bound rather than bound by root's 11 serial senders.
- r01-b04-a02 halved the K-block count by widening blocks, which changed accuracy. This keeps the
  K-block geometry and every format exactly as before. The math is bit-identical.
- r01-b03-a02 batched x into bursts. That kept the same lead time and made arrival burstier.
  This increases lead time without changing burst size.
- Porting r01-b01-a02 is not the mechanism; it is the base. The delta to judge is against
  r01-b01-a02's 1.4716.

## Expected effect and risk
- Accuracy: bit-identical to baseline (same tiles, same K order, same formats). Any change at
  all means a slot-index bug.
- Speed vs r01-b01-a02: +1-4% on shapes that are jitter- or fill-bound (mistral, dsv4-flash,
  kimi-k3, glm, dsv3). About neutral on the DRAM-saturated kimi-k2.7, and on dsv4-pro and
  minimax-m3 if they stay at 2 slots. If every deepened shape is within noise of r01-b01-a02,
  jitter/coupling is not what limits them, and the remaining limit is DRAM plus per-expert
  fill/drain.
- Risks: a hang if the writer's slot count disagrees with the CB depth (eval `hang`). Wrong
  weights from a slot mismatch would show as `accuracy_fail`. A host TT_FATAL can't happen: the
  extra slots are only taken when they fit the fitter's own budget.
