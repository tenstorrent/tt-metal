# r01-b03-a04 result: 1.4856 (ok), new campaign best (prev. best r01-b01-a02 1.4716; parent 1.4574)

## What happened vs expected
All 9 shapes are valid and bit-identical to the baseline (PCC, max_abs and mean_abs equal to every
printed digit), including the three shapes whose layout changed. So the 5-row topology set, the
idle-row handling and the layout mailbox agree across all kernels. No hang, and the host built
first try.

Per shape vs the parent (µs, noise ±1%):

| shape | parent | this | change | layout used |
|---|---|---|---|---|
| kimi-k3 | 1489.1 | 1373.1 | **-7.8%** | 2 rows x 5 groups (was 2x4) |
| gpt-oss | 374.7 | 337.4 | **-9.9%** | 5 rows x 2 groups (was 4x2) |
| kimi-k2.7 | 767.5 | 762.2 | -0.7% (noise) | 2x5 (was 2x4) |
| dsv3 | 559.9 | 559.7 | 0 | 2x4, rows 8-9 idle |
| glm | 488.9 | 491.2 | +0.5% (noise) | 2x4 |
| mistral | 192.4 | 195.5 | +1.6% | 2x4 |
| dsv4-flash | 343.1 | 344.7 | +0.5% (noise) | 2x4 |
| dsv4-pro | 1151.5 | 1149.5 | -0.2% | 2x4 |
| minimax-m3 | 384.2 | 383.0 | -0.3% | 2x4 |

I predicted ~+15% on gpt-oss and up to ~+12% on kimi-k3. I got +10% and +8%, both far outside noise.
The other shapes tie in the cost model and run today's layout exactly. They stayed within noise.
mistral at +1.6% is in its known run-to-run range (186-200 µs across nodes with the same
schedule), so the 22 idle cores cost nothing measurable.

## Why (profiler evidence)
Per-row TRISC1 kernel-end, µs from op start, mean over the 10 measured calls
(reports/r01-b03-a04/.logs/profile_log_device.csv, /tmp/rep_b03a04/spread.py):
- kimi-k3: rows 0-5 (groups with 6 experts) end at 1307-1354; rows 6-9 (5 experts) end at 1146-1203.
  The critical path went from 7 to 6 experts per group (-14% of expert rounds) but gained only 8%.
  Per expert per group it is ~222 µs now against ~211 before, so the 5th concurrent group costs ~5%
  per expert. The total is now 557 MB in 1373 µs = 406 GB/s, close to the ~410-434 GB/s the
  DRAM-bound big shapes reach. kimi-k3 is now near the DRAM ceiling.
- gpt-oss: group 0 (rows 0-4) ends at 335, group 1 (rows 5-9) at 318. The M-work per group went
  from 3+2 to 2+2 tile-rows (-20%) for -10% time. The per-expert fixed part (activation SFPU
  pass, the phantom 11th down K-block, bias, x tilize/multicast fill) does not shrink with
  per_core_M.
- kimi-k2.7: rows 0-3 (3 experts per group) still end at ~745, and rows 4-9 (2 experts) at ~565-580.
  12 experts on 5 groups is still 3 rounds on the critical groups, so the layout only relieved
  the others. That fits the DRAM-bound reading: nothing changed on the critical path.

## Classification
win (+1.9% geomean vs parent, +1.0% vs the previous best). This is the first change to the op's
work partition since the row groups, and it is a pure scheduling change with bit-identical numerics.

## What a child of this node should try next
1. **Finer than whole-expert balancing on kimi-k3 / kimi-k2.7.** With 5 groups the critical groups
   (6 vs 5 experts on kimi-k3, 3 vs 2 on kimi-k2.7) end 150-180 µs after the others. Let the
   lightly loaded groups take part of the last-round experts: e.g. split the final expert of a
   heavy group by N (gate/up N and down N halves, or down K) across the idle group. Or use
   cost-aware layouts that mix group heights, such as 3 groups of 2 rows plus 1 group of 4 rows.
   That is worth up to ~10% on kimi-k3 if DRAM allows (it is now ~406 GB/s).
2. **gpt-oss is still the least efficient shape** (~200 GB/s total). Per expert it has a fixed
   cost that per_core_M doesn't touch: the SwiGLU-OAI SFPU pass (27 -> 18 tiles per core now), the
   fully phantom 11th down K-block (K 90-98, a full weight multicast of stale L1 plus an activated
   multicast per expert), single-buffered bias CBs, and 1x3 gate/up subblocks. Skipping the phantom
   down block (reader, writer and compute must agree on num_blocks_d = ceil(K_down / w_d)) is
   cheap and exact.
3. More layouts are now just table entries in `adaptive_chunk::kGroupLayouts` plus a host topology
   set: (10,1) for large token counts, or (5,2) variants. Add a layout only after you have
   simulated the cost model on the test's routing (/tmp/rep_b03a04/sim.py replicates it). A tie
   keeps the earlier layout.
4. Keep: the exported grid stays 11x8 (moe_fused_swiglu / the hybrid op read
   UNIFIED_ROUTED_EXPERT_CORE_GRID). Extra rows are an op-internal detail. GRID_ROWS falls back
   to 8 on devices with fewer than 10 rows.
