# r02-b04-a03: single-row expert groups: a 10 x 1-row layout (8 x 1 on 8-row grids) with no weight multicast, so kimi-k3's 28 small experts run 2-3 per row at their own tile count instead of 5-6 per 2-row group at a padded per_core_M 2 (row-major x only; reverts the parent's writer-owned down stream)

## Motivation
- Parent r02-b04-a02 regressed (1.3612) and its reflection says to start from r02-b04-a01. This
  node restores r02-b04-a01's op code first (reader, writer, compute, factory).
- kimi-k3 is the large shape furthest from its DRAM floor: 557 MB in 1397 µs (r02-b04-a01) is
  ~400 GB/s, against the ~440-455 GB/s the other DRAM-bound shapes reach (kimi-k2.7 736 µs on
  r02-b03-a02, dsv4-pro 1083). That leaves ~9-13% on the table.
- The cause is the work partition. The test routing gives kimi-k3 28 experts of 3-4 tile-rows
  (11 x 4 + 17 x 3 = 95 tile-rows). On 2-row groups every expert costs per_core_M 2, so a 3-tile
  expert pays a phantom tile-row on one of its rows. The 5 groups get 6/6/6/5/5 experts. The
  critical rows then compute 12 tile-rows plus 6 per-expert fixed costs. Measured on
  r01-b04-a04, 6-expert rows end ~1340 µs and 5-expert rows ~1156 µs, i.e. ~92-100 µs per tile-row.
- With one row per group, each expert runs per_core_M = its own tile count (3 or 4), with no
  phantom rows. 28 experts over 10 rows balance to at most 10 tile-rows and 3 experts per row
  (e.g. {4,3,3} x 7, {3,3,3}, {4,4} x 2). The cost model rates this 13 units against today's 18.
- Earlier evidence: r01-b02-a01's 1-row groups on the 8-row grid tied 2-row groups on kimi-k3 at
  equal critical tile-rows (14 vs 14), even with 4 vs 7 experts per row and the old per-core x
  tilize. So per tile-row cost is the same in both, and 10 vs 12 tile-rows should show.

## Mechanism
1. Revert the parent's code change (git checkout of r02-b04-a01's op files).
2. `kernels/adaptive_chunk.hpp`:
   - kMinGroupRows 2 -> 1, kLayoutMaxGroups 5 -> 10.
   - Append a uniform 1-row layout to both tables ({10 x 1} and {8 x 1}). It is scheduled by the
     existing LPT path, so it replaces the winner only when STRICTLY better under the cost model.
   - Gate it with `UNIFIED_RE_SINGLE_ROW_GROUPS` (host-set, = x is row-major): plan_groups skips
     any layout with a 1-row group when it is 0.
   - By plan.py replay of the test routing, kimi-k3 is the only benchmark shape that switches.
     The other 8 keep their exact layouts and expert -> group maps.
3. Host factory:
   - Mirror kMinGroupRows = 1 in the num_chunks clamp bound. The cap stays >= M_tiles_full, so
     the clamp is unchanged in effect.
   - Skip the x/weight sender-collision TT_FATAL for 1-row groups. On the row-major path the x
     source is the rotating owner, which is already a weight sender on every first group row
     today. On the TILE path the kernels never pick the layout.
   - Pass UNIFIED_RE_SINGLE_ROW_GROUPS to reader, writer and compute.
4. Writer: guard the IN1_WRITER_MCAST gate/up multicast and valid-sem multicast with
   `in1_num_receivers > 0`, as the reader's paths already are. In a 1-row group every core is
   its own column's sender with 0 receivers.

## Why this is not a repeat
- Layout additions so far (r01-b03-a04, r01-b04-a04, r02-b02-a01, r02-b04-a01) all kept groups of
  >= 2 rows. 1-row groups were last run in round 1 on the 8-row grid (r01-b02-a01), before
  row-shared tilize and before the 10-row grid. There 28 experts on 8 rows meant 4 experts on
  half the rows.
- r01-b02-a02 dropped 1-row groups because the TILE-path fixed x sender collided with weight
  senders. The benchmark runs row-major x with rotating owners, and the TILE path keeps 2+ row
  groups here.
- r01-b01-a04 predicted "R=1 not promising" for kimi-k3 with 8 rows (4 vs 3 experts per row,
  equal critical tile-rows). With 10 rows the critical row drops from 12 to 10 tile-rows.

## Expected effect and risk
- kimi-k3: -6..-9% (toward the ~1266-1280 µs DRAM floor). Every other shape should be unchanged
  (same plan, same code path apart from the reverted parent change), so ~+0.7-1.0% geomean
  over r02-b04-a01.
- Risks:
  - 1-row groups may lose per-row efficiency: all 110 cores issue their own weight reads, and
    each x owner's reads share a RISC with its weight reads.
  - A 0-receiver multicast path I missed would hang. Accuracy is unaffected: same per-tile math
    and same K-block order, so it should be bit-identical.
- How to tell: kimi-k3 µs, and per-row TRISC1 ends from profile_log_device.csv (expect 10 rows
  ending within a few % of each other).
