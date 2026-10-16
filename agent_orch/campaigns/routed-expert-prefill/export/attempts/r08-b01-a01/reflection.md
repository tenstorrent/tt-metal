# r08-b01-a01 result: 1.7397 (ok). Neutral overall: +0.06% over the root r07-b02-a01 (1.7387), inside the ±1% noise. That net hides two parts going opposite ways: three clear wins (kimi-k2.7 -3.7%, dsv4-pro -4.5%, kimi-k3 -2.9%) and two clear losses (dsv3 +5.2%, glm +6.2%). Bit-identical outputs

## What happened vs expected
- All 9 shapes are valid on the first and only run. PCC, max_abs and mean_abs equal the root on every printed digit.
- No JIT error, no kernel-config overflow, and no hang. That includes the new case of a two-phase plan whose top
  phase-B group is idle.
- The host was rebuilt only because the eval checkout's last build differed. The change itself is kernel-only.
- The plans changed exactly as the replay (/tmp/r08b01/pt.cpp) predicted. Only the 5 targeted shapes moved.

| shape | root µs | this µs | Δ | plan change |
|---|---|---|---|---|
| dsv3 | 510.1 | 536.5 | **+5.2%** | two-phase 2x5 + 4+3+3 -> single-phase 2x5, 1/1/2/2/2 |
| glm | 441.0 | 468.5 | **+6.2%** | same as dsv3 |
| kimi-k2.7 | 700.7 | 674.8 | **-3.7%** | phase B 5+5 -> 4+3+3's lower two groups (top 4 rows idle) |
| dsv4-pro | 997.3 | 952.1 | **-4.5%** | same as k2.7 |
| kimi-k3 | 1212.3 | 1177.6 | **-2.9%** | 6/6/6/5/5 -> 5/5/6/6/6 |
| mistral / dsv4-flash / mm3 / gpt-oss | | | ±0.6% | unchanged |

Expected: dsv3 / glm -3..-6%, k2.7 -3..-5%, dsv4-pro ~0, k3 -3.4%.
- k2.7 and k3 came in as expected.
- dsv4-pro gained where I expected nothing. Its delivered weight rate rose to ~468 GB/s (446 MB in 952 µs), from ~448
  GB/s before. So the root's dsv4-pro was not at the DRAM ceiling; its tail was costing it.
- dsv3 and glm lost where I expected a gain.

## Why (profiler evidence; /tmp/r08b01/jobs.py and /tmp/r06b01a01/ends.py on reports/r08-b01-a01)
Per-job timelines are GU start | GU end | DN end in µs; the tables show row ends.

1. **dsv3 / glm: the middle group's second expert is the new tail.**
   - Per-row ends, dsv3: y2-3 438, y4-5 409, **y6-7 536**, y8-9 445, y10-11 412. glm: y6-7 467, the others
     345-396.
   - The top four rows did gain, as intended: dsv3's top group ended at 438 vs 509.
   - y6-7's second expert (2-row, per_core_M 3) ran 298 -> 535 = **237 µs**. The bottom groups' second experts ran
     197-230 µs. I had estimated ~190 µs, i.e. compute-bound.
   - So a per_core_M-3 expert on a 2-row group, starting while 4-5 groups still stream, is neither compute-bound
     nor fast. It gets a middle-rank DRAM share for its whole life.
   - The root's 4+3+3 phase B ran those leftovers at per_core_M 2 on 3-4 rows, in 139-188 µs. That beats the
     extra group-time the top rows lose by being in phase B.
   - So for dsv3 / glm, the top-free phase-B layouts, all of which need per_core_M 3 on 2-row groups, are worse
     than using the top group. The planner's model saw this (t_b 509 vs 346), but because the top-free rule left it
     no 4+3+3 option, it dropped to single-phase.
2. **kimi-k2.7 / dsv4-pro: keeping phase B off the top rows pays.**
   - k2.7: the top four rows end phase A at 649 / 619 and stop. Phase B runs on y9-11 from 352 (y8-11 idled to 514
     before) to 561, and on y6-8 from 544 to 672. Chip 673 is the y6-8 group.
   - dsv4-pro: the top rows end at 914 / 880. y9-11 runs 529 -> 819 and y6-8 758 -> 950.
   - The tail is now the middle phase-B group (y6-8), which starts when y6-7 ends phase A. It no longer waits for the
     top group. The root's 5+5 tail (one 5-row stream alone for the last 50-90 µs) is gone.
   - A 3-row per_core_M-1 phase-B expert still takes 190-290 µs while 4-5 groups stream. That is a share-bound
     rate, not the ~110 µs a near-lone group gets.
3. **kimi-k3: the conditional flip reproduces r07-b04-a01.**
   - Per-row ends: 1128 / 1093 / **1177** (y6-7) / 1126 / 1056. r07-b04-a01 measured 1117 / 1115 / 1192 / 1120 /
     1075.
   - mistral and minimax-m3 kept their plans: mistral 167.1 -> 166.9, mm3 333.7 -> 331.9. So the `active >= groups`
     guard removes r07-b04-a01's mistral / mm3 loss, as intended.
4. Size: the reader is +84 B and the writer +92 B in a local -O2 build, and it fit.

## Classification
Neutral within noise, as a sum of a win and a flawed sub-rule:
- **Win (keep):** the bottom tie-break when every group is busy, and the top-free phase B where it costs no more per
  expert. That is k2.7, dsv4-pro and k3, about -3.7% mean on 3 shapes, or +1.25% geomean.
- **Flawed sub-rule (drop):** forcing the top group idle in phase B even when the only top-free layouts raise the
  leftovers' per_core_M. On dsv3 / glm that pushed the plan to single-phase 1/1/2/2/2, +5..+6%.

## What a child of this node should try next
1. **Repair: make the top-free phase B a preference, not a requirement (expected ~1.76, +1.2% over the root).**
   - In `make_plan`'s phase-B loop, also accept layouts that use the top group: `l.groups >= n_left` with the old
     `rows[k]` (taller-first) assignment, as the root did.
   - A top-free candidate (`groups > n_left && rows[0] >= rows_a`, bottom-up assignment) replaces a top-using one
     whenever its t_b is <= the top-using one's. Otherwise the top-using one stays.
   - With the test counts:
     - dsv3 / glm: the top-free t_b is 509 vs 346, so 4+3+3 with every group used, exactly the root's plan.
     - k2.7 / dsv4-pro: 200 vs 200, so this node's top-free 4+3+3 is kept.
     - k3: unchanged.
   - Verify with /tmp/r08b01/pt.cpp before the eval: dsv3 / glm lines must match /tmp/r08b01/base.txt, and the
     other lines must match /tmp/r08b01/new.txt.
   - Then expect dsv3 ~510 and glm ~441 (root), with k2.7 ~675, dsv4-pro ~952 and k3 ~1178 from this node.
   - Keep `plan_launch`'s `last_b - rank` mapping only for top-free plans. The plan needs one flag, e.g. `top_free`
     in `Plan`, or derive it from `layout_b`'s group count vs n_left.
2. **The next tail on k2.7 / dsv4-pro is the y6-8 phase-B group.** It starts when the middle phase-A group (y6-7)
   ends, at 544 / 758 µs.
   - Try the 4+2+2+2 layout with leftovers on y8-9 and y10-11, which start at ~400 / 350 and ~597 / 533. The model
     currently rates their per_core_M 2 above 3-row per_core_M 1 (346 vs 200).
   - Given point 2 of "Why", the real cost difference is smaller than the model says. Worth one eval as a model
     tweak.
3. **dsv3 / glm** still have the root's phase-B tail on y2-5. The only cheaper option this run did not rule out is
   phase B 4+3+3 with the largest leftover on the BOTTOM 3-row group (bottom-up assignment while still using the
   top group). Its leftovers are all 5 tiles, so that alone changes nothing. Leave dsv3 / glm to the dynamic
   pacing direction (b04 in this round).
4. Don't retry:
   - single-phase bottom-heavy deals for dsv3 / glm (this node: +5..+6%);
   - an unconditional bottom tie-break (r07-b04-a01).
5. Tools:
   - /tmp/r08b01/jobs.py: per-row job intervals (GU start | GU end | DN end), for any report.
   - /tmp/r08b01/pt.cpp: planner replay. Compile with `g++ -std=c++17 -I <kernels dir>`.
   - /tmp/r08b01/tr/kc.sh: TRISC0 size; pass `-I/tmp/r08b01/basek` for the root's header.
   - /tmp/r07b02/sz/b.sh: reader / writer size.
