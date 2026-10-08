# r01-b03-a02: Column split with equal-width slices only (k divides num_tile_cols, k=4 -> 81 cores): one kernel group, no dispatch stall

## Motivation
The parent (r01-b03-a01, 0.987) column-split each tile-row across k=3 cores. Its reflection blamed
"cross-chip launch skew". The ops CSV shows something more specific. Per shape, median over measured
calls (reports/r01-b03-a01 ops_perf_results):

| shape | slices | kernel groups | OP TO OP LATENCY d0..d3 (ns) | kernel d0..d3 (µs) |
|---|---|---|---|---|
| h3584 (28) | 10/9/9 | 2 | 42390 / 44927 / 52061 / 49949 | 23.7 / 21.1 / 14.0 / 16.0 |
| h4096 (32) | 11/11/10 | 2 | 41616 / 44646 / 51442 / 49025 | 24.2 / 21.3 / 14.3 / 16.6 |
| h6144 (48) | 16/16/16 | **1** | **550 / 552 / 556 / 559** | 22.0 / 21.8 / 21.8 / 21.7 |
| h7168 (56) | 19/19/18 | 2 | 34766 / 37284 / 44894 / 42538 | 31.0 / 28.2 / 20.7 / 22.7 |

Baseline, b01, b02 and b04 all have about 540-620 ns op-to-op on every shape. Every shape with two kernel
groups (uneven slices; num_tile_cols is a CT arg) stalls 35-52 µs between ops on the device.
The host issues a call every ~15-18 µs, so the device falls behind and each chip's dispatch gap decides when
it starts. Each chip has a different gap, so the early chips sit in F_FABRIC waiting for the late one (the
d2 column). That wait is the "skew". The one equal-width shape (glm, k=3, 1 group) had no stall, balanced
chips, and gained 7.6%.

Zone timeline for glm in the parent (dev 0, per core): R_INPUT ends 3.6-5.9 µs (was 6.3-7.3 at baseline),
the AG ends ~12.7-13.3 µs, TRISC ends 16-18 µs, and W_DRAIN ends 17.3-23.4 µs. The drain end grows with
core x (x=2 ~17.5 µs, x=14 ~22.5 µs). So the remaining tail is NoC/DRAM write-path congestion. Spreading
the same bytes over more grid rows (more cores) should help it a bit, and halving per-core tiles shortens
read, PRE and POST.

## Mechanism
Factory only (`device/dit_fused_distributed_rmsnorm_program_factory.cpp`), in the col_splits choice:
- Only consider k that divide num_tile_cols exactly, so every slice has the same width. Then there is one
  reader/writer/compute kernel group (3 binaries, same as baseline), and the uneven 2-group path never runs.
- Raise the core budget from 64 to 80 workers (+1 forwarder = 81 of 110 cores). Pick the largest such k
  with k*rows <= 80 and slice >= 4 tiles. With 20 tile-rows this gives k=4 on all four shapes: slices of
  7/8/12/14 tiles, 80 workers.
Kernels, the leader-combine protocol and the go relay are unchanged from the parent. The protocol was already
validated (PCC 0.9999985, no hang) at k=3. With k=4 the leader sums 3 peer sticks and relays go to 3 followers.

## Why this is not a repeat
This repairs r01-b03-a01's execution, not its idea. The bug: uneven slices -> two kernel groups -> a ~40-50 µs
device-side dispatch gap per op on 3 of 4 shapes. The proof is in the table above: the one equal-width shape had
no gap. r01-b02-a01 never engaged (34-stick packet cap). No other node changes the decomposition.

## Expected effect and risk
- All shapes should show ~550 ns op-to-op and balanced F_FABRIC across chips.
- Per shape, assuming the parent's un-skewed critical path plus ~2 µs normal AG wait: h3584 ~14-15 µs (1.15),
  h4096 ~15-16 µs (1.15), h6144 ~20-21.5 µs (1.1-1.15), h7168 ~21-23 µs (1.15). That is roughly 1.1-1.15
  geomean. b01's gamma reorder is not on this lineage, so POST stays two passes.
- Risks:
  - 81 cores add host per-call cost (parent: 61 cores, 15.2 µs/call vs 13.5 baseline). If the device becomes
    faster than the host, chips start ~0.7 µs apart (host enqueue order) and the early ones wait. That is a
    small penalty, visible as o2o >> 0.5 µs with d0 waiting the most.
  - k=4 may not beat k=3 on glm if the drain is purely NoC-bound. Compare against the parent's 21.8 µs.
  - L1: the CBs shrink and peer_partials grows to 3 fp32 tiles, so no risk.
  - Accuracy: same partial-sum math (more partials) as the parent.
