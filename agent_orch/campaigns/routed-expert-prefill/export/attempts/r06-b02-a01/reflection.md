# r06-b02-a01 result: 1.6400 (ok). Negative: -1.6% vs the parent r05-b02-a01 (1.6667), outside the ±1% noise. The stagger moved the start-up advantage to the top group as intended, but the delayed groups below became the tail and the mean core end got worse. Bit-identical outputs

## What happened vs expected
- All 9 shapes are valid on the first and only run (kernel-only, no host build, no kernel-config overflow, no hang).
  PCC / max_abs / mean_abs match the parent on every printed digit.
- Expected: the top groups end 10-30 µs earlier and the lower groups absorb their delay inside their slack, about
  +2..+4% geomean. Got: the top group did gain, often by more than expected. The group just below it, which starts
  5-7.5 µs later, became the new slowest group. The bottom groups lost much more than their delay.

Per-row kernel end, parent -> this node (µs, mean of the 10 measured calls, `/tmp/r06b02/ends.py`). Delays are in brackets:

| shape | top group | 2nd group | 3rd group | bottom groups | chip µs |
|---|---|---|---|---|---|
| minimax-m3 (2x4) | y2-3 359 -> **319** [0] | y4-5 336 -> 354 [5] | y6-7 307 -> 335 [10] | y8-9 298 -> 323 [15] | 361.9 -> 368.3 |
| mistral (2x4) | y2-3 177 -> **166** [0] | y4-5 174 -> 180 [5] | y6-7 146 -> 184 [10] | y8-9 142 -> 170 [15] | 190.1 -> 197.8 |
| dsv3 (phase A 5x2) | y2-5 503 -> 492 | y6-8 450 -> 483 | | y9-11 409 -> 458 | 504.9 -> 510.9 |
| glm | y2-5 435 -> 435 | y6-8 394 -> 425 | | y9-11 352 -> 382 | 436.7 -> 444.1 |
| dsv4-flash (3+3+2+2) | y2-4 294 -> 301 [0] | y5-7 278 -> 300 [7.5] | y8-9 254 -> 285 [15] | y10-11 238 -> 273 [20] | 298.4 -> 315.7 |
| kimi-k3 | y2-7 1172-1216 -> 1174-1227 | | y8-9 950 -> 998 | y10-11 936 -> 962 | 1243.6 -> 1251.6 |
| dsv4-pro / kimi-k2.7 | top ~flat | | bottom +11 / +19 | | 993 -> 987 / 701 -> 702 |
| gpt-oss | ~flat (only y5-7 delayed) | | | | 261.0 -> 261.6 |

## Why (profiler evidence)
1. **Start order decides who wins, and small offsets grow.** minimax-m3's top group gate/up window fell
   259 -> 210 µs. The second group, started only 5 µs later, now takes 236-239 µs. That is a 26 µs gap from a 5 µs
   offset. mistral is the same picture: y2-3 GU 128 -> 112-116, y4-5 125 -> 123-127, y6-7 96 -> 114-116.
   So the row gradient in the parent is not a fixed geometric penalty on the top rows. It is a first-mover
   advantage in the shared DRAM/NoC stream. In the parent the bottom groups' first responses happen to win the
   synchronized burst, and the lead they take persists and grows over the whole gate/up phase. Shifting the start
   just hands that lead to the top group and makes the next group the loser.
2. **The stagger costs aggregate throughput.** The mean core end got worse on every staggered shape: dsv3 459 -> 479,
   minimax-m3 325 -> 333, dsv4-flash 270 -> 294. The bottom groups lost 25-50 µs against delays of 15-20 µs.
   During the staggered window fewer streams are active. The groups that start later then run in the slow
   "follower" mode for the whole phase, so total work does not finish sooner.
3. **dsv4-flash (-5.8%) is the worst case.** Its 3+3+2+2 layout has equal loads, so all three lower groups were held
   (7.5 / 15 / 20 µs). The 3-row y5-7 group became co-critical with the top group, and the top group did not
   speed up (294 -> 301). With 3-row groups on top, the top group did not get a head start large enough to matter.
4. gpt-oss is unchanged: its bottom groups were correctly not held (more work than the top group).

## Classification
flawed idea (as a static start-up offset). The measurement is the useful part. The per-group throughput is
path-dependent (a lead self-reinforces), not a per-row constant. A static stagger only rotates the loser and wastes
some aggregate rate.

## What a child of this node should try next
1. **Revert this node's reader change** (start from r05-b02-a01's code). Delete the `startup_stagger_cycles` call
   block in the reader's phase loop, or make `kStaggerCyclesPerRow` 0, which keeps the code inert.
2. **Attack the self-reinforcing lead, not the start.** The groups that get ahead keep winning because nothing
   bounds how far ahead a group's weight stream runs (2-3 ring slots plus the gate/up look-ahead per sender). Cheap
   tests, one per node:
   - cap in-flight weight reads per sender: issue the look-ahead block kb+1 only once block kb's multicast is done
     (`gu_lookahead` off, measured neutral before on another lineage), and check whether the per-row GU spread
     shrinks;
   - a row-dependent NoC split ("tilt"): top groups use less NoC 0 (s0 ~0.5) and bottom groups more (~0.7-0.75).
     The mean is kept at ~0.6. The parent's per-row data showed more NoC-0 share slowing the top rows and speeding
     the bottom ones (also r03-b03-a02). Unlike this node, it changes steady-state link sharing, not timing.
3. Planner-side alternative: since the bottom groups systematically win under the parent's start, give them the
   larger share of work. kimi-k3: deal the 6th experts to the bottom groups (they end ~270 µs early). dsv3 / glm: the
   phase-B 4-row group at the bottom. This uses the lead instead of fighting it.
4. Don't retry: start-up stagger / pacing in any static form (this node, r01-b01-a03).
5. Tools: `/tmp/r06b02/gu.py` (per-row GU/DN start and duration per unit), `ends.py` (per-row ends, works on every
   report), `x0.py` / `x0grid.py` (RE-X0 analysis on r05-b03-a01), `plan/drv2.cpp` (the kernel planner compiled
   natively on the test counts).
