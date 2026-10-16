# r04-b03-a03 result: 1.6066 (ok). Neutral: -0.3% vs the parent r04-b03-a02 (1.6119), inside the ±1% noise band. The idle-row layout is a real win on minimax-m3 (-4.6% chip) and neutral on mistral. The size repair (planner at -Os on writer + UNPACK) delays the first gate/up by ~3 µs on every shape, which eats the gain

## What happened vs expected
- **Run 1 (not scored): runtime_error.** `Program size (70736..71648) too large for kernel config buffer (70656)` on 6 shapes (worst
  kimi-k2.7, +992 B). kimi-k3, dsv4-pro and minimax-m3 ran and were bit-identical. My margin estimate was wrong: the parent was only
  ~0.8 KB under the limit on k2.7, not ~4.5 KB, and the tie pass adds ~0.6 KB per planning RISC (reader, writer, UNPACK).
- **Repair:** the planner region is built at -Os on the writer and UNPACK, the reader keeps -O2 (local size: 4832 B per RISC at
  -O2 in the parent; now reader 5428 / writer 4024 / UNPACK 4024, about -1.0 KB per core). Same plan.
- **Run 2 (scored):** all 9 shapes valid, PCC / max_abs / mean_abs bit-identical to the parent and the baseline. The 3+3+2+2 plan
  for mistral and minimax-m3 (6-tile experts on 3 rows at pcM 2, 5-tile ones on y8-9 / y10-11 at pcM 3) delivers exactly the same
  outputs. No hang.
- Expected mistral -3..-6% and minimax-m3 -2..-4%, others unchanged. Got minimax-m3 -4.6%, mistral 0%, and +1..+3.6% on several
  shapes with identical plans.

Chip µs, parent -> this node; first GU start and mean core end from /tmp/r04b03a01/ends.py:

| shape | chip µs | first GU start µs | mean core end µs |
|---|---|---|---|
| dsv3 | 520.0 -> 519.5 | 6.7 -> 9.0 | 489.7 -> 488.1 |
| kimi-k2.7 | 744.8 -> 755.3 (+1.4%) | 8.1 -> 11.1 | 731.0 -> 727.6 |
| kimi-k3 | 1314.6 -> 1320.7 | **22.2 -> 29.7** | 1225.6 -> 1232.0 |
| glm | 452.1 -> 463.3 (+2.5%) | 6.7 -> 9.2 | 430.5 -> 433.5 |
| mistral | 183.0 -> 183.1 | 3.9 -> 7.0 | **160.9 -> 155.8** |
| dsv4-flash | 309.3 -> 320.6 (+3.6%) | 9.2 -> 12.1 | 284.7 -> 289.9 |
| dsv4-pro | 1087.8 -> 1080.6 | 7.9 -> 11.0 | 1068.3 -> 1064.9 |
| minimax-m3 | 368.8 -> **352.0 (-4.6%)** | 3.9 -> 7.0 | **341.6 -> 324.9** |
| gpt-oss | 264.4 -> 265.7 | 4.5 -> 5.4 | 237.7 -> 235.4 |

## Why (profiler evidence)
1. **The -Os planner is on the critical path.** First GU start moved +2.3..+3.1 µs on 8 shapes and +7.5 µs on kimi-k3 (28 experts,
   the heaviest plan). The compute (UNPACK) cannot wait on its first x / weight block until it has its own plan, and the -Os build
   plans slower than the reader's -O2 copy plus its first DRAM read. That is ~0.3-1.7% per shape, and with the usual lottery it
   explains the glm / dsv4-flash / k2.7 rises (their mean core end moved only -3..+5 µs; chip times moved +10..+11).
   - r04-b01-a02's "writer -Os is free" holds for the writer only; UNPACK is the one that matters.
2. **The idle-row layout works on minimax-m3.** Per-group ends (grp2.py, rows.py): all four groups now end at 329-333 µs
   (parent 334-358 mean per group, per-call max 368 -> 352).
   - The 3-row pcM-2 groups did NOT end earlier than the 2-row pcM-3 ones (329 vs 332). So mm3 is delivery-bound per group, not
     compute-bound, and lower pcM only shortens the GU of the 3-row groups (205-219 vs 217-237 before).
   - The whole shape got faster anyway (mean core -17 µs), i.e. spreading the same weight streams over all 10 rows (more x-owner
     cores, more L1 read/write ports per column multicast) helps a delivery-bound shape.
3. **mistral: the mean improved but the max did not.** 3-row groups end at 154 µs (pcM 3 -> 2: only -12..-15 µs, not the
   -25 µs U fit). The 5-tile group moved to rows y8-9 and became the slowest (172 µs vs 160 µs at the same expert in the parent).
   y10-11 ends at 163. The per-call max is 183 µs either way: the lottery just moved.

## Classification
neutral (within noise). The mechanism itself is a win on minimax-m3 (-4.6%) and neutral on mistral. The repair cost ~3 µs of start-up
on every shape, which cancels it. Without the -Os UNPACK penalty I would expect ~+0.5% over the parent.

## What a child of this node should try next
1. **Keep the tie pass, but find the size somewhere other than the UNPACK planner.**
   - (a) Best: plan once per core. r03-b01-a02 had the reader publish the `LaunchPlan` in a 1-page L1 CB that the writer and UNPACK
     read. That removes two planner copies (~8 KB per core) and the UNPACK planning time altogether, which should also take
     ~1-3 µs off first-GU on every shape. It is a host change (one more CB) plus reader/writer/compute edits.
   - (b) Cheaper: put the UNPACK planner back to -O2 and keep only the writer at -Os. That is ~+0.8 KB vs this node. Then cut the
     tie pass itself, e.g. hard-code it as one candidate (3+3+2+2 when exactly 4 experts and the uniform pick is 2x5) instead of
     looping over every mixed layout. Check with JIT ELF sizes: the k2.7 program must stay under 70656 B, and the local -O2 compile
     underestimates by up to ~0.5 KB.
2. **Port the tie pass onto r04-b02-a01 (best, 1.6275)** together with (1a) or (1b); the two touch different code
   (adaptive_chunk.hpp vs the reader's look-ahead). Expect mm3 ~-4%, and nothing measurable elsewhere.
3. mistral's slow row pair moved with its 5-tile expert (y2-3 -> y8-9). Before pushing more layout work on mistral, measure why one
   2-row group runs 10-15 µs slower than another with the same work (x-owner / DRAM-bank position). /tmp/w_r04b03a03c/outl.py and
   uwout.py split a slow group's time into x / activated / weight waits per call.
4. Tools: /tmp/w_r04b03a03c/sim (pt.cpp plan replay on the test routing; fz.cpp + fzmain.cpp old-vs-new plan fuzz, 400k cases per
   grid height; sz.cpp local planner size per RISC opt level), grp.py / grp2.py (per-call group ends for old / new layouts).
