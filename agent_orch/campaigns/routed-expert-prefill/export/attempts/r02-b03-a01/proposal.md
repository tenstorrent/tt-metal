# r02-b03-a01: two-phase expert schedule: full waves on the small row groups, then the leftover experts on larger row groups (5 or 10 rows), so the tail streams at lower per_core_M instead of leaving groups idle

## Motivation
r01-b04-a04 (1.5011, my code base) measured the group tail as the largest remaining waste. With 5
groups of 2 rows the expert counts rarely divide: kimi-k2.7 (12 experts: 3/3/2/2/2) has groups 2-4
idle for the last ~25% of the call, dsv4-pro (12) ~22%, dsv3/glm (8: 2/2/2/1/1) ~45% of two groups.

A simple time model fits the measured data well. Per expert, time is the max of
(a) per-core compute ~1.63 c per tile-row of per_core_M, and (b) its DRAM share, c x (number of
groups streaming at once), where c = one expert's weight bytes / ~443 GB/s. Measured µs per model
unit: dsv3 53.8, kimi-k2.7 56.8, glm 47.9, dsv4-pro 83.8, against c from the weight bytes of
56 / 56 / 48 / 84 µs. In the full waves the shapes are DRAM-bound (5 streams). In the tail,
1-3 groups each run an expert at per_core_M 2-3 and are compute-bound with DRAM half idle:
kimi-k2.7's last expert takes ~185 µs on 2 streams, which is 3.3 c, exactly 2 x 1.63 c.

The tail experts don't need fewer weight bytes. They need more rows, so per_core_M drops to 1 and
the remaining streams become DRAM-bound again. Splitting an expert by M across groups would re-read
its weights (bad, the op is DRAM-bound). Splitting it by N would need a cross-group reduction.
Moving the leftover experts onto bigger groups keeps bytes and numerics unchanged.

## Mechanism
A runtime two-phase plan (all in `kernels/adaptive_chunk.hpp`, computed identically by reader,
writer and compute from the device counts):
- Phase A: the current picker's group size rows_a (2 on these shapes). It runs the
  floor(E / G_a) x G_a LARGEST experts, dealt round-robin in index order. Every group gets the same
  count, and the plan requires every phase-A expert to have the same chunk count.
- Phase B: the leftover (smallest) experts run on groups of rows_b in {kGridY/2, kGridY} (5 or
  10 rows), list-scheduled as today.
- The plan is taken only when the model above predicts >= 5% over today's single-phase schedule.
  Otherwise the single-phase schedule runs unchanged. Predicted: dsv3 -10%, glm -10%, kimi-k2.7
  -9%, dsv4-pro -9% (all rows_b = 5). kimi-k3, dsv4-flash, mistral, minimax and gpt-oss keep
  today's plan.

What makes the phase switch safe without a barrier:
1. **CB ring alignment.** Weight multicasts write to the sender's own CB write pointer in every
   receiver. Phase-B column groups span rows from different phase-A groups. Because every phase-A
   group runs the same number of units (equal experts x equal chunks), every core has pushed the
   same number of gate/up and down blocks when it enters phase B, so the rings line up. x and
   activated multicasts are per M-row, and a group never splits a row.
2. **Semaphore isolation.** Phase A uses a NEW in1 ready/valid semaphore pair. Phase B (and the
   single-phase plan) uses the existing pair. A phase-B receiver that finishes phase A early
   cannot inc the ready count of a core that is still a phase-A sender. 13 -> 15 semaphores
   (limit 16).
3. **Writer slot counters.** The writer indexes its cb_in1_up / cb_in1_gate / cb_in1_down slots
   by a sender-only block counter (up_seq, down_blk). With roles changing between phases, that
   counter would drift from the reader's real CB pointer. Slots are now indexed by a counter of
   ALL blocks this core consumed (sender or not). It is identical to today's counter whenever the
   role is fixed. The go/done sequence counters stay sender-only and in lockstep with the reader.
4. Compute: UNPACK sends MATH/PACK each worked expert's (count_tiles, group rows) instead of a
   launch-wide group size.

Files: `kernels/adaptive_chunk.hpp` (plan + dealer), reader, writer, compute
(`fused_swiglu.cpp`), program factory (2 semaphores and their runtime args).

## Why this is not a repeat
- r01-b03-a04 / r01-b04-a04 added larger layouts, but one size for the whole launch. Here the
  size changes mid-launch, per expert.
- r01-b01-a03 balanced groups by pacing, which made every group slower. This plan adds no
  per-block cost. The only new work is at kernel start (plan) and one semaphore pair.
- r01-b01-a04 / r01-b03-a03 targeted the expert-boundary bubble. This targets idle groups in the
  tail.

## Expected effect and risk
Expected -7..-10% on dsv3, glm, kimi-k2.7 and dsv4-pro, and 0 on the rest: score ~1.55-1.57.
Accuracy is bit-identical (same per-core math, only which group computes an expert changes).
Risks:
- Hangs, if any kernel's deal or role counters disagree. The eval tool's timeout would show it.
- Wrong output, if ring alignment fails (accuracy fail).
- A phase-boundary bubble larger than modelled (a phase-B group waits for the slowest of the 2-3
  phase-A groups its rows came from). The profile's per-row end times would show it.
- The plan costs a few µs of scalar work at kernel start on every RISC.
