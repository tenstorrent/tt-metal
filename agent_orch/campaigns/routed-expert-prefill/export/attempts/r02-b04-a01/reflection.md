# r02-b04-a01 result: 1.5314 (ok), new campaign best (parent r01-b04-a04 1.5011, +2.0%)

## What happened vs expected
All 9 shapes are valid and bit-identical to the baseline and the parent (PCC, max_abs and mean_abs equal
to every printed digit), including the three shapes that switched to the mixed (3,3,2,2) layout. So the
plan agrees across reader, writer and the three compute threads, the 3-row multicast rectangles are
right, and there was no hang. The host built and the kernels JIT-compiled on the first try.

Per shape vs the parent (µs, noise ±1%; the cost model picked (3,3,2,2) for exactly the three shapes
predicted in proposal.md, and kept the uniform layout everywhere else):

| shape | parent | this | change | layout | predicted |
|---|---|---|---|---|---|
| gpt-oss | 338.8 | 277.1 | **-18.2%** | (3,3,2,2), one expert per group | -10..-25% |
| dsv4-flash | 331.2 | 322.4 | **-2.7%** | (3,3,2,2), two experts per group at pcM 2 | -5..-10% |
| dsv4-pro | 1128.0 | 1131.8 | +0.3% (noise) | (3,3,2,2) | at most -3..-7% |
| dsv3 | 547.5 | 555.7 | +1.5% | unchanged (2x5) | noise |
| kimi-k3 | 1378.8 | 1397.0 | +1.3% | unchanged | noise |
| minimax-m3 | 382.0 | 388.8 | +1.8% | unchanged | noise |
| kimi-k2.7 | 764.3 | 767.5 | +0.4% | unchanged | noise |
| glm | 487.6 | 486.3 | -0.3% | unchanged | noise |
| mistral | 193.0 | 192.6 | -0.2% | unchanged | noise |

The four shapes that look slower kept today's layout AND today's expert -> group map (checked against
the old schedule: `group_cost` == `expert_cost` for every t, rows, pcM_max, and the uniform path is the
same in-order list scheduling). Their MEDIAN core end time did not move (TRISC1 kernel end, median core,
mean of the 10 measured calls: kimi-k3 1307 -> 1306 µs, dsv3 500 -> 504, minimax 347 -> 348, mistral
162 -> 163; /tmp/r02b04a01/k3.py). Only the per-call max moved, and it scatters widely run to run
(kimi-k3 per-call max 1333-1415 before, 1359-1435 now). So that is the known tail lottery, not
planner cost in the prologue.

## Why (profiler evidence)
TRISC1 kernel end per logical row (rows 0..9), mean of the 10 measured calls, µs from op start
(/tmp/r02b04a01/rows.py on reports/<node>/.logs/profile_log_device.csv):

- gpt-oss: before, both 5-row groups (2 experts each, pcM 2) ended at 326-331. Now the 3-row groups
  (one 9-tile expert each, pcM 3) end at 236-237 and the 2-row groups (one 8-tile expert, pcM 4) end at
  263-269. Fitting time = F + U * pcM to those two group types gives U ~ 29 µs per tile-row and
  **F ~ 150 µs per expert**. The fixed per-expert cost is about 5 units, not the cost model's 1. That
  is why halving the experts per group (2 -> 1) bought 18% while the units went 4 -> 3-4.
- dsv4-flash: before, critical groups 312-324 and light groups 190-284. Now all four groups end at
  298-306, nearly perfectly balanced, as modelled. The op-level gain (-2.7%) is smaller than the
  critical-row gain (-6%) because the per-call max (tail lottery) adds back some of it.
- dsv4-pro: the 3-row groups (4 experts at pcM 1) end at 1113-1125, the 2-row groups (2 experts at
  pcM 2) at 748-750. So pcM 1 costs ~280 µs per expert and pcM 2 ~375: again F ~ 2U, and the 3-row
  groups became the critical path at the same time as before. The shape moves ~420 GB/s (474 MB in
  ~1130 µs), so it is DRAM-bound and no layout can help it much. Neutral, as the proposal allowed.

## Classification
win (+2.0% geomean over the best node; -18% on gpt-oss and -2.7% on dsv4-flash, both far outside noise;
bit-identical numerics; every other shape keeps exactly its old schedule).

## What a child of this node should try next
1. **Fix the cost model's fixed term.** Measured per-expert fixed cost F is ~2 units (dsv4-pro) to ~5
   units (gpt-oss), not 1. A better F changes which layouts win: `expert_cost` / `group_cost` in
   kernels/adaptive_chunk.hpp add `chunks` (1 per chunk); try `kExpertFixedUnits * chunks` with 2-3.
   Replay the test routing with /tmp/r02b04a01/plan.py first (add the F knob there; it matches the C++
   bit for bit, and /tmp/r02b04a01/cmp.py checks the C++ against it). Watch the shapes that keep a
   uniform layout today: a different F can change their uniform pick too (it decides 10 vs 5 vs 2 rows).
2. **Find what the ~150 µs per-expert fixed cost on gpt-oss is.** It does not scale with pcM, and
   gpt-oss (now ~245 GB/s, 4 groups) is not DRAM-bound. Suspects: the down phase (11 K-blocks of
   row-wide activated-multicast handshakes plus the fully phantom 11th block on gpt-oss, K 90-98), the
   SwiGLU-OAI + bias activation, and gate/up pipeline fill. Put DeviceZoneScopedN zones around
   gate/up, activation, down and the copy pass on TRISC1 in a measurement node (the profiler keeps 125
   zones per RISC per op, and op times come from the guaranteed markers, so a few zones per expert are safe).
3. **gpt-oss remaining imbalance:** the 2-row (pcM 4) groups finish ~30 µs after the 3-row (pcM 3)
   ones. With 4 experts and 10 rows, (3,3,2,2) is already the best whole-expert split. Only a
   sub-expert split (e.g. give one 2-row group's last tile-row to a 3-row group) or cutting F would
   close it.
4. Layout table entries are cheap: add one to kLayouts10 / kLayouts8 (the static_asserts check it tiles
   the grid). The host emits one topology set per layout automatically, and the sender-collision
   TT_FATAL runs for each.
5. Planner cost is not measurable today (median core end unchanged), but it walks counts through
   `tiles_of` ~400 times on kimi-k3. If more layouts are added, cache tiles in a local uint8/uint16
   array first.
