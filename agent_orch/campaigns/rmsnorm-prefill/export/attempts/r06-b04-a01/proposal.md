# r06-b04-a01: congestion-aware out-of-order dual-NoC drain

## Motivation
On the best node r05-b01-a01 (1.5696), once the output is ready the kernel is bound by the output drain.
The drain also gets slower from straggler cores in particular grid positions.

Evidence: `analysis/readpos.py` and `analysis/cores.py` on `reports/r05-b01-a01`, h7168, dev0. Values are per-core
medians over the 10 measured calls, in µs from the chip's first worker start.
- **Compute is not the limit.** C_POST takes a constant 1.97 µs on every core (28 tiles, output_cb holds the whole
  row, so pack never blocks). The drain then ends **0.7-2.3 µs after POST ends**.
- **The time from POST start to drain end depends on position.**
  - Wave A ranges from 2.6 µs at (13,3) and (14,2) to 3.96 at (12,4) and 4.20 at (14,4).
  - Wave B ranges from 2.7 at (2,5) and (3,4) to 3.75 at (13,4), with (7,4) at 3.61 and (7,2) at 3.72.
  - The kernel end is the last B core: median B drain end ≈ 14.7, max ≈ 15.4.
  - So ~0.6 µs of the tail is stragglers, not aggregate DRAM bandwidth.
- **The reads, by contrast, are uniform.** R_INPUT durations are 2.7-3.3 (A) and 3.1-3.6 (B) with no position
  gradient, so the input read is DRAM-shared, not link-bound.
- r02-b03-a01 already showed that the drain tail is set by **per-router arbitration on the shared NoC1 segment into
  a DRAM column** (through-traffic beats local injection), not by hop count. A core starves when its NoC1 injection
  can't get onto a hot row.

The current drain (r02-b02-a01's path-aware rule) sends tiles strictly **in column order**, with a **static** NoC
choice: NoC0 only for "short-eastward" destination banks, and only on every other visit, so ~25% of the tiles.
When a NoC1 tile is back-pressured, BRISC blocks on the NoC1 command buffer, while NoC0 sits idle. NoC0 has no input
reads during the drain: in the 2-wave pipeline the drains start ~3 µs after the last input read. Then the drain
flushes and pops every block (block_size tiles), so the slower NoC's last packet also stalls the other NoC at every
block boundary.

## Mechanism
Writer only: `dit_rmsnorm_fused_worker_writer.cpp`, the W_DRAIN block, column-split + dual-NoC path only. Every other
config keeps the old loop.
1. **Eligibility is unchanged (the path-aware rule).** A tile whose bank sits in the short-eastward DRAM column is
   *eligible* for NoC0 (E). Every other tile is NoC1-only (N). Long NoC0 wraps stay forbidden, which was
   r01-b02-a04's disaster.
2. **The NoC is picked by measured congestion, not by parity.** Before each issue, read each NoC's posted-write
   backlog: `issued (self + other RISC dynamic-NoC counters) − NIU_MST_POSTED_WR_REQ_SENT`. Then:
   - if N tiles are ready and NoC1's backlog ≤ NoC0's, send the next N tile on NoC1;
   - otherwise send the next E tile on the NoC with the smaller backlog (NoC0 on a tie);
   - if only N tiles remain, send on NoC1.

   A core whose NoC1 is starved moves all of its eligible tiles to NoC0. A core whose NoC0 is busy keeps them on
   NoC1.
3. **Tiles are issued out of order across the whole row.** output_cb holds 2 full rows and each worker has 1 row, so
   compute never waits on the drain. The writer extends its ready window block by block without blocking
   (`pages_available_at_front`), and two cursors (next N, next E) walk the ready tiles. It blocks on `wait_front`
   only when nothing is ready. One posted flush on each NoC and one pop at the end of the row replace the per-block
   flush+pop, so no block boundary waits on the slower NoC.

Bytes, addresses and destinations are identical. Only the NoC and the issue order change, so accuracy must be
bit-identical.

## Why this is not a repeat
- r01-b02-a04, r01-b03-a04, r02-b01-a01 and r02-b03-a01 used **static** splits (parity, position or destination).
  r02-b02-a01 is the static path-aware rule this node keeps for eligibility. None of them looked at congestion at
  run time, reordered tiles, or removed the per-block flush.
- r03-b02-a03 kept two writes in flight on the **same** NoC (two command buffers), which added head-of-line
  contention. Here there is still ≤1 command buffer per NoC. The change is which NoC, and which tile goes next.
- r03-b03-a03 (VC round robin) and r04-b03-a03 (bank de-phase) changed neither the NoC choice nor the issue order
  under congestion.
- Round-6 siblings (r06-b02/b03) change the wave split, which is orthogonal.

## Expected effect and risk
- **Expected:** the straggler cores finish closer to the median. The B tail should drop by ~0.3-0.6 µs at
  h6144/h7168 and by less on the narrow shapes, where the drain is 14 tiles per core. The A wave's straggler tail
  shrinks too, which also cuts the next call's start skew (r02-b02-a01 saw this coupling). Score target is
  +1.5-3%.
- **Risks:**
  - NoC0 might get crowded by eligible tiles from many cores at once. The backlog comparison self-limits this.
  - Removing the per-block flush could, in theory, let the NIU queues grow. Each NoC still has only one command
    buffer, so that can't happen.
  - Hangs are unlikely: the CB waits are cumulative and only extend.
- **How to tell from the eval:** chip-mean change on h6144/h7168. Analysis will rerun `readpos.py` and `chain.py`:
  - per-core POST start → drain end spread, which should shrink;
  - B drain end max − median;
  - kernel end.
