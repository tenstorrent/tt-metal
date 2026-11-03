# Slot study: how the VC0:VC1 weight and the slot split move perf with mesh size (2026-11-01)

Status: study notes (study agent). No builder or allocator change. Question (user, 2026-11-01): don't pick a VC0:VC1
weight yet; sweep the weights on the lead to see how mesh size and slot counts interact (P150 VC1 split, Galaxy 2x2
split, Galaxy single mesh; larger multi-mesh splits after the command-buffer-2 fix). A mesh-size-aware allocator may
come later. Background: ALLOCATION_NOTE.md (window model, 2026-10-28 study), V4_BOUNDS.md §1b (job 360).
Labels: measured (job), model (sls_model.py), hypothesis.

Router: `variants/v4rxv2i3` (the lead), -O3, flags 13129088, keepalive 1000, BW telemetry off, slots logged.

## Files

| file | what |
| --- | --- |
| `sls_model.py` | alloc_model.py's model and rule (imported unchanged) on other data: `v4gi` (the 2026-10-28 fit), `p150` (the lead, post-merge placement), `prov` (P0's provisional lead model), `glx` (Galaxy split), `2d` (VC0-only 2D on the lead, P150 or Galaxy) |
| `sls_predictions.txt` | P0 predictions, written before the first P1 run; `sls_predictions_check.txt`: the refit's predictions for its unrun picks (written before job 388) |
| `sls_p150_runs.sh` | P1: VC1 split runs (`fabric_exp/vc1/sls_vc1_all.yaml`: vc1_alloc + vc1_inter + post-merge mix rows, 82 finishes; run_vc1_cap; slots checked) |
| `sls_p150_vc0_runs.sh` | P3's P150 side: 2D / 1D splits on `fabric_exp/slotstudy/sls_{2d,1d}_p150.yaml` |
| `sls_report.py` | tables per allocation x row (p150, glx, p150vc0); `pred`: P0 predictions against P1 |
| `sls3k_model.py`, `sls3k_predictions.txt`, `sls3k_table.py` | the 3 / 3.5 KiB study: the interpolated model, its predictions (before job 167723), the per-path tables |
| `sls2k_report.py`, `sls2k_predictions*.txt` | the 2 KiB-slot study's tables (lr, arms, 2d, vc1, vc2) and its predictions |
| `V4_NS_FLOOR.md` | the Galaxy N/S floor: bound side (one-side A/B), mechanism (diagnostic variants v4bddsl / v4bddml), fix proposals |
| `sls_sl_runs.sh` | P150 A/B of the diagnostic variants (base / tx32 / rx32) |
| `fabric_exp/slotstudy/` | Galaxy: `mk_sls_tree.sh` (lean /data tree of the live tree + the lead), `sls_glx.sbatch` (vc1 / 2d / 1d steps), `sls_mock_vet.sh`, yamls, the 2x2 split MGD + rank bindings (copied from the VC2 line agent), P4's 4x4 / 4x2 MGDs + rank bindings |

## P0: allocation set and predictions (offline)

Allocations (worker, VC0 forwarding, VC0 receiver, VC1 forwarding, VC1 receiver; 81 slots of 4448 B):

| tag | allocation | from |
| --- | --- | --- |
| w12 | 20,6,16,7,6 | the 2026-10-28 fit (v4gi) at VC0:VC1 = 1:2 |
| j360 | 13,8,16,7,7 | job 360's deeper VC1 windows |
| w11 | 18,8,16,6,5 | v4gi fit, 1:1 |
| w32 | 18,9,16,5,5 | v4gi fit, 3:2 |
| today | 19,9,16,5,4 | the builder's study row since job 113 |
| w21 | 19,10,16,4,4 | v4gi fit, 2:1 |
| w31 | 20,11,16,3,3 | v4gi fit, 3:1 |
| main | 51,4,8,2,4 | main's tier table (reference) |

Calibration points (one window kind binding each, P1 step 1): worker 8 / 12 / 15 (cw8, cw12, cw15); VC0 receiver 8
(cr0_8 = 27,9,8,5,4); VC1 forwarding 3 / 4 / 5 with the VC1 ring at 7 (cf3, cf4, cf5); VC1 ring 3 / 4 / 5 with VC1
forwarding at 8 (cr3, cr4, cr5); both at 8 / 7 (cp = 13,7,16,8,7: is 7,7 at the VC1 path's bound?).

Placement changed with the merge (measured; affects every VC1 inter-mesh row): on a09u32 since the post-merge tree,
[0, 0] lands on [1, 0]: VC1Inter_0_0_to_1_0 is the direct row (job 360, census sdb: 208-209), 1_1 and 1_2 cross one
VC1 hop (346-397), 1_3 two (376-414). In jobs 110-112 (v4gi, pre-merge) [0, 0] landed on [1, 1] (direct 1_1: 206;
1_0, 1_3 one hop; 1_2 two). So alloc_model.py's row mapping (VC1Inter_0_0_to_1_1 = direct) and vc1_mix.yaml's flows
([0, 0] -> [1, 0] "then W on VC1") are pre-merge only: on today's tree vc1_mix.yaml's VC1 flow is the direct row and
crosses no VC1 hop. sls_model.py maps the post-merge rows; sls_vc1_all.yaml's mix rows are moved to the new placement
(VC1 flow [0, 0] -> [1, 1]; VC0 worker [1, 0] -> [1, 1]; forwarded VC0 [1, 2] -> [1, 1]).

Predictions (`sls_predictions.txt`, written 20:17-20:20 before job 376): two models.
- `v4gi` as is: every 1-hop row 158 (the lead runs 142); its VC1 rows extrapolate v4gi's loops.
- `prov`: v4gi's slopes with the intercepts of the kinds the lead already has points for moved to the lead (VC0
  forwarding at T=4 from the census's 2-hop 239.5 / 258.0 at 9 slots: RT0 1,679 / 1,913 vs v4gi's 1,911 / 2,131; the VC1
  ring from 19,9,16,5,4's 346.6 / 396.5 at 4 slots: rate x W 1,386 / 1,586 vs v4gi's 1,280 / 1,499, longer), and the
  lead's service bounds (1-hop 142, direct 208 / 209, the VC1 path 295 / 305 from 13,8,16,7,7, assumed at its bound).
- prov's own picks differ from v4gi's by a slot or two: 1:2 20,6,16,7,6; 1:1 20,7,16,6,6; 3:2 21,8,16,5,5;
  2:1 22,9,16,4,4; 3:1 20,11,16,3,3 (its worker step is 2,730 / 142 = 20).

## P1: P150 VC1 split, the lead (a09u32 jobs 376, 377, 379, 388; 2026-11-01 20:44-22:10)

Each allocation one run of sls_vc1_all.yaml (82/82 finishes, sanity data-checked, slots as requested every time);
sweep in two passes in reversed order (jobs 376 / 377), pass-to-pass <= 0.7 except the mix rows (<= 1.9). No hang.
Cycles/packet, mean of the passes:

| row | B | w12 20,6,16,7,6 | j360 13,8,16,7,7 | w11 18,8,16,6,5 | w32 18,9,16,5,5 | today 19,9,16,5,4 | w21 19,10,16,4,4 | w31 20,11,16,3,3 | main 51,4,8,2,4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 16 | 142.1 | 162.0 | 142.1 | 142.0 | 142.1 | 142.1 | 141.9 | 176.8 |
| 1-hop E/W | 4352 | 142.3 | 173.4 | 142.1 | 142.1 | 142.0 | 142.2 | 142.2 | 193.9 |
| bursts of 8 | 4352 | 142.2 | 192.8 | 150.9 | 150.9 | 142.3 | 142.2 | 142.1 | 193.9 |
| 2-hop | 16 | 333.0 | 259.4 | 260.2 | 239.4 | 239.4 | 224.1 | 220.3 | 431.0 |
| 2-hop | 4352 | 361.1 | 280.8 | 281.7 | 258.0 | 257.9 | 236.1 | 225.4 | 480.1 |
| 2-hop merged | 4352 | 190.7 | 153.1 | 152.9 | 144.2 | 144.3 | 132.8 | 134.6 | 257.5 |
| multicast | 4352 | 195.4 | 160.6 | 160.6 | 149.4 | 149.3 | 140.1 | 136.1 | 267.3 |
| inter direct (1_0) | 4352 | 209.1 | 212.8 | 209.2 | 209.2 | 209.1 | 209.1 | 209.1 | 236.1 |
| inter + 1 VC1 hop (1_1, 1_2) | 16 | 294.0 | 295.3 | 304.2 | 349.7 | 346.6 | 432.5 | 567.6 | 833.9 |
| inter + 1 VC1 hop (1_1, 1_2) | 4352 | 303.9 | 304.2 | 341.0 | 400.5 | 396.2 | 489.8 | 652.0 | 949.3 |
| inter + 2 VC1 hops (1_3) | 4352 | 315.6 | 315.8 | 353.5 | 416.0 | 413.3 | 503.0 | 666.1 | 979.5 |
| mix: VC0 worker / its VC1 flow | 4352 | 196 / 343 | 209 / 343 | 200 / 391 | 205 / 448 | 205 / 442 | 209 / 541 | 199 / 698 | 222 / 1015 |
| mix: VC0 forwarded / its VC1 flow | 4352 | 394 / 328 | 315 / 331 | 317 / 375 | 297 / 445 | 295 / 441 | 263 / 529 | 240 / 685 | 493 / 1005 |

(Solo VC0 flows in the mix yaml: worker [1, 0] -> [1, 1] as the 1-hop row; forwarded [1, 2] -> [1, 1] 218-337 at
4352 B, as the 2-hop row's f0 window. The VC1 flow alone is the 1_1 row. Every flow of a mix runs slower than alone, so
the post-merge mix rows do share [1, 0]'s E/W link: the VC0 worker 196-222 against 142, the VC1 flow +39-66.)

Calibration (one window kind shrunk per run; job 376 / 377 / 379), 4352 B (16 B in brackets):

| window | points (slots: cycles/packet) | rate x W = RT0 + b W |
| --- | --- | --- |
| worker (1-hop rows) | 8: 263.9, 12: 184.4, 13: 173.4, 15: 152.7, 16: 144.0, 18-19: 142 (bursts of 8: 18: 150.9, 19: 142.3) | 1,870 + 29.9 W (1,673 + 33.9 W) |
| VC0 receiver at 8 (T = 2; cr0_8, main) | 1-hop 194.4 (177.5), inter direct 236.2 (232.2) | 1,576 (1,437); behind the inter-mesh link 1,889 (1,857) |
| VC0 forwarding (T = 4), 2-hop row | 6: 361.1, 7: 322.0, 8: 281, 9: 258.0, 10: 236.1, 11: 225.4 | 1,801 + 58.1 W (1,483 + 76.7 W) |
| VC1 forwarding (T = 1; ring 7: cf3-cf5) | 3: 652, 4: 492, 5: 400.5 | 1,853 + 28.6 W (1,629 + 23.2 W) |
| VC1 ring (T = 1; forwarding 8: cr3-cr5) | 3: 486.4, 4: 364.6, 5: 302.9 (near the bound) | ~1,468 (1,224) |
| VC1 path, both deep (cp 13,7,16,8,7) | 1_1 / 1_2 294.0-294.6 (289.3), 1_3 295.0 (290.1) | service bound ~294 (v4gi: 293) |

Against v4gi (ALLOCATION_NOTE §2, 4352 B): the worker's loop is shorter (~2,400 at 18 slots against ~2,730), VC0
forwarding shorter (2,322 at 9 against 2,540), VC1 forwarding longer (2,000 at 5 against 1,864), the VC1 ring about the
same (1,468 against 1,499), the VC1 path's bound the same (294 / 293), 1-hop 142 against 158.

Which VC1 window binds (A/B: one window deepened at a time): today's 19,9,16,5,4 is bound by the VC1 forwarding window
(5), not the ring (4): forwarding 5 with the ring at 7 runs 400.5 (today 396.2), the ring at 4 with forwarding 8 runs
364.6. Same for 18,8,16,6,5 (ring 5 alone ~303, forwarding 6 alone ~335 by the fit, measured 341: both near, the slower
binds and costs ~7 more). On the lead the forwarding window binds while F1 < ~1.33 x R1 (the ratio of their loops);
alloc_model.py's rule ("forwarding binds while F1 <= R1") called 19,9,16,5,4 ring-bound. V4_BOUNDS §1b ("bound by VC1's
windows, 5 forwarding / 4 receiver") stands; it is the forwarding window.

Job 360's 13,8,16,7,7 on the VC0 rows (not run in job 360): the 13-slot worker binds every 1-hop row (+20 at 16 B, +31
at 4352 B: 142 -> 162 / 173), bursts of 8 +51, the 2-hop row +20 / +23 (forwarding 8). Its VC1 gain (-51 to -98) costs
VC0 1-hop traffic 14-22 %. The VC1 rows of 13,8,16,7,7 (304) are not at the VC1 path's bound: forwarding 8 (cp) gives
294 (-10).

Predictions (sls_predictions.txt, `sls_report.py pred`), sweep allocations without main: prov 70 points, mean |error|
2.6 %, VC0 rows within 1 % (2-hop: 333.0 measured / 332.8 predicted at 6 slots, 239.4 / 239.5 at 9) except where the
13-slot worker bound (j360 1-hop -13 / -18 %: no worker window in the P0 models) and 3:1's 2-hop at 16 B (-6.7 %);
every VC1 row 6-7 % optimistic (the lead's VC1 forwarding loop is longer than v4gi's), the two-hop VC1 row 8-13 %.
v4gi as is: 56 points, mean 6.6 % (1-hop 158 against 142, 2-hop +9-12 %, VC1 -5-8 %). Hand predictions for the worker
(v4gi's 2,730 round trip: 8 -> 341, 12 -> 228, 15 -> 182) were 30 % too slow (264, 184, 153).

Refit (`sls_model.py fit p150`: the earlier lead rows plus jobs 376-379, 190 points, mean |error| 1.1 %, max 15 %; the
misses are the T = 2 forwarding points (two sizes only), worker-8 rows other than 1-hop, and points within ~10 of the
VC1 path's bound, where two windows near each other run slower than either):

| weight VC0:VC1 | the rule's pick (lead fit) | model 2-hop / inter + 1 VC1 / + 2 VC1 (4352 B) | v4gi fit's pick |
| --- | --- | --- | --- |
| 1:2 | 18,7,16,7,5 | 315 / 294 (bound) / 323 (measured 323 / 308 / 325) | 20,6,16,7,6 |
| 1:1 | 18,7,16,7,5 | 315 / 294 (bound) / 323 (measured, job 388) | 18,8,16,6,5 |
| 3:2 | 19,9,16,5,4 (today's row) | 258 / 399 / 414 | 18,9,16,5,5 |
| 2:1 | 17,10,16,5,3 | 238 / 489 / 506 (measured 236 / 494 / 505) | 19,10,16,4,4 |
| 3:1 | 17,12,16,3,3 | 208 / 646 / 664 (measured 220 / 652 / 667) | 20,11,16,3,3 |

- 1:2 and 1:1 coincide: at 18,7,16,7,5 the VC1 path is at its service bound, so further VC1 weight buys nothing.
- The worker step gives 17 (steady 1-hop rows: 1,870 / (142 - 29.9)); bursts of 8 need 19 (18: +8.7 at 4352 B). The
  rule sizes the worker from steady rows only; a burst-aware worker step would give 19 (hypothesis: bursts lengthen the
  worker's loop by the idle gap's refill).

Check of the refit on its own unrun picks (job 388, 22:03-22:10, two passes d / e in reversed order; predictions in
`sls_predictions_check.txt`, written 21:09 before the job; a control run of 19,9,16,5,4 at both ends repeats jobs
376 / 377 within 0.9). 40 points, mean |error| 1.1 %, max 8.1 %. 4352 B (16 B), measured / predicted:

| allocation (pick) | 1-hop | bursts of 8 | 2-hop | inter + 1 VC1 | inter + 2 VC1 | multicast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 18,7,16,7,5 (1:2, 1:1) | 142.1 / 142.0 | 150.9 | 322.5 / 315.3 (296.1 / 288.5) | 307.6 / 294.3 (294.0 / 289.7) | 325.2 / 323.4 | 176.8 |
| 17,10,16,5,3 (2:1) | 142.2 / 142.0 | 152.6 | 236.2 / 238.2 (224.2 / 225.0) | 494.0 / 489.3 (409.3 / 407.9) | 504.5 / 505.9 | 140.1 |
| 17,12,16,3,3 (3:1) | 142.3 / 142.0 | 152.7 | 220.4 / 208.1 (217.8 / 200.2) | 652.1 / 646.4 | 667.1 / 663.8 | 136.1 |
| 19,7,16,7,4 (1:1 with a burst-sized worker) | 142.0 / 142.0 | 142.3 | 322.0 / 315.3 | 367.1 / 367.0 | 394.0 / 391.9 | 176.8 |

- The misses: the 1:1 pick's VC1 row sits 13 above the bound it was sized to reach (two windows near each other,
  pattern 5 below), and the 2-hop row stops gaining at ~218-220 (11 slots 220.3 / 225.4, 12 slots 217.8 / 220.4): the
  2-hop path has a bound of its own near 220, not the 1-hop's 142 the model extends to it (a lower bound, as
  ALLOCATION_NOTE §8 says), so on the lead VC0 forwarding beyond ~11 slots buys nothing.
- A worker of 17 costs bursts of 8 +10.5 (152.6); 19 does not (142.3): the steady-row worker step (17) is short for
  bursts.
- Host state: the main session's host rebuild (broker job 386) failed at once (exit 1 after 0.1 s), so every P150 run
  here (jobs 376-388) and the Galaxy tree used the same host build (libtt_metal.so identical in the live tree and in
  trees/sls_lead).

## P2: Galaxy 2x2 split, VC1 (Slurm job 167464, bh-glx-110-a07u08, 200G; 2026-11-01 20:52-21:39)

Tree sls_lead (the lead), sls_glx_vc1.yaml (98/98 finishes every step, sanity data-checked, slots as requested), the
sweep in two passes in reversed order (pass-to-pass <= 1.1), then the calibration points. No hang. Results kept in
`fabric_exp/results/glx167464/` (CSVs, logs without compile lines, gzipped); the job directory is deleted.
Rows: mesh 0's intra-mesh rows as on the P150; inter-mesh rows from the exit chip [0, 1] (direct to [1, 0], one VC1
hop E/W to [1, 1], one VC1 hop N/S to [1, 2] (it moves with the VC1 ring: it crosses at [1, 0] and goes S on VC1), two
to [1, 3]) and from [0, 0] (a VC0 hop into the exit chip first). Cycles/packet, mean of the passes:

| row | B | w12 20,6,16,7,6 | j360 13,8,16,7,7 | w11 18,8,16,6,5 | w32 18,9,16,5,5 | today 19,9,16,5,4 | w21 19,10,16,4,4 | w31 20,11,16,3,3 | main 51,4,8,2,4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 16 | 142.0 | 168.6 | 142.0 | 142.0 | 142.1 | 142.2 | 142.1 | 184.5 |
| 1-hop E/W | 4352 | 252.1 | 252.1 | 252.1 | 252.1 | 252.1 | 252.1 | 252.1 | 252.0 |
| bursts of 8 | 16 | 142.1 | 178.2 | 142.2 | 142.2 | 142.2 | 142.2 | 142.2 | 184.5 |
| 2-hop | 16 | 349.0 | 264.3 | 265.2 | 240.5 | 240.6 | 215.4 | 201.8 | 444.7 |
| 2-hop | 4352 | 376.1 | 301.7 | 301.5 | 276.7 | 276.4 | 261.6 | 252.9 | 521.3 |
| 2-hop merged | 4352 | 284.5 | 259.2 | 259.2 | 254.1 | 254.1 | 252.2 | 252.1 | 306.7 |
| multicast | 4352 | 298.6 | 268.0 | 267.9 | 260.0 | 260.1 | 255.9 | 255.1 | 338.6 |
| inter direct ([0,1] -> [1,0]) | 16 | 208.5 | 211.9 | 208.6 | 208.5 | 208.5 | 208.5 | 208.4 | 233.2 |
| inter direct ([0,1] -> [1,0]) | 4352 | 252.1 | 252.1 | 252.1 | 252.1 | 252.1 | 252.1 | 252.1 | 259.3 |
| +1 VC1 hop E/W ([0,1] -> [1,1]) | 16 | 295.0 | 296.3 | 317.3 | 369.2 | 368.2 | 448.3 | 603.1 | 898.5 |
| +1 VC1 hop E/W ([0,1] -> [1,1]) | 4352 | 323.2 | 323.9 | 365.0 | 436.3 | 433.9 | 535.6 | 716.8 | 1070.2 |
| +1 VC1 hop N/S ([0,1] -> [1,2]) | 4352 | 300.6 | 302.3 | 333.1 | 389.9 | 407.7 | 480.8 | 638.8 | 945.2 |
| +2 VC1 hops ([0,1] -> [1,3]) | 4352 | 316.3 | 316.8 | 357.6 | 416.4 | 421.9 | 511.5 | 675.2 | 989.4 |
| VC0 hop + direct ([0,0] -> [1,0]) | 4352 | 470.0 | 362.8 | 362.9 | 328.2 | 328.1 | 299.3 | 278.6 | 600.7 |
| VC0 hop + 1 VC1 hop ([0,0] -> [1,1]) | 4352 | 544.8 | 386.8 | 399.2 | 437.3 | 435.0 | 535.6 | 716.9 | 1066.2 |
| mix: VC0 worker / its VC1 flow | 4352 | 353 / 823 | 353 / 715 | 340 / 897 | 328 / 927 | 319 / 943 | 312 / 996 | 297 / 1138 | 307 / 1249 |
| mix: VC0 forwarded / its VC1 flow | 4352 | 479 / 465 | 435 / 534 | 427 / 583 | 399 / 654 | 394 / 658 | 358 / 735 | 326 / 891 | 572 / 1117 |

Calibration (job 167464, pass c): worker 8 / 12: 1-hop 254.4 / 179.0 at 16 B (4352 B: 296.6 / 252.1); VC0 ring 8: 1-hop
185.0 at 16 B, 252.1 at 4352 B (no cost at 200G and 4352 B); VC1 forwarding 3 / 4 at ring 7: E/W hop 716.6 / 538.6;
VC1 ring 3 / 4 at forwarding 8: 540.0 / 402.6; both deep (13,7,16,8,7): one hop E/W 298.6, N/S 295.6, two hops 295.9
(4352 B; 290.2 / 287.6 / 288.4 at 16 B).

Fit (`GLX_RES=fabric_exp/results/glx167464 sls_model.py fit glx`, 224 points, mean |error| 0.9 %, max 12.8 %), 4352 B,
against the P150 (P1):

| window | Galaxy (200G) | P150 (400G) |
| --- | --- | --- |
| service bounds | 1-hop 252 (wire), direct 252 (wire), VC1 path ~296-299 (E/W 298.6, N/S 295.6, 2 hops 295.9) | 142, 209, 294 |
| worker | 2,378 at 8 (16 B: 1,792 + 30.0 W) | 1,870 + 29.9 W (16 B: 1,673 + 33.9 W) |
| VC0 forwarding (T = 4) | 1,760 + 81.3 W (2,490 at 9) | 1,801 + 58.1 W (2,322 at 9) |
| VC0 forwarding into the exit chip (new kind) | 2,559 + 43.6 W (2,953 at 9) | not on the P150 split ([0, 0] is an exit chip) |
| VC1 forwarding E/W (T = 1) | 2,066 + 24.3 W (2,178 at 5) | 1,853 + 28.6 W (1,996 at 5) |
| VC1 forwarding N/S | 1,802 + 38.6 W (~2,000 at 5) | (same as E/W on the P150) |
| VC1 ring | 1,615 (N/S 1,627) | 1,468 |
| VC0 ring 8 (T = 2), 16 B | 1,470 (inter-mesh link 1,866) | 1,437 (1,857) |

- At 16 B the Galaxy's windows are the P150's within ~5 % (router-bound rates 142 / 208 identical); at 4352 B its
  loops are 7-10 % longer (each loop carries at least one 4352 B serialisation, 2x longer at 200G: hypothesis) while
  the VC0 rows' bound moves from the router (142) to the wire (252), so the VC0 windows need far fewer slots there:
  worker ~10 (2,378 / 252), VC0 ring 8 is enough, VC0 forwarding reaches the wire at 11 (2-hop 252.9).
- The VC1 path stays router-bound at ~296 on both, above the 200G wire: VC1 needs the same windows on the Galaxy as on
  the P150 (forwarding ~8, ring ~6-7 to reach its bound), a little more for E/W hops.
- Which VC1 window binds: the same ratio rule (forwarding 2,178 at 5 against the ring's 1,615: F1 < ~1.35 x R1); today's
  19,9,16,5,4 is forwarding-bound on the E/W hop (433.9; forwarding 5 at ring 7 not run on the Galaxy, cf4 at 4: 538.6,
  ring 4 at forwarding 8: 402.6) and ring-bound on the N/S hop (407.7 at ring 4; ring 5 (w32) 389.9; ring 4 at forwarding 8 (cr4) 406.3), with its
  forwarding window close behind (~400 at 5 by the fit).
- Mixed load at 200G: the shared link is the wire. The VC0 worker plus its VC1 flow add up to the wire (e.g. today
  319 and 943: 1 / (1/319 + 1/943) = 238 per packet on the link); the split moves the share (VC1 forwarding 7: 353 / 823,
  3: 297 / 1,138). On the P150 the sum sits at the router's ~140 per packet (205 and 442 -> 140).

The rule on the Galaxy fit (`sls_model.py alloc glx`; sized at the largest packet, 4352 B, as alloc_model.py does):

| weight | sized at 4352 B | sized at 16 B | P150 (4352 B) |
| --- | --- | --- | --- |
| 1:2 | 11,8,16,8,6 | 18,7,16,7,5 | 18,7,16,7,5 |
| 1:1 | 11,8,16,8,6 | 18,7,16,7,5 | 18,7,16,7,5 |
| 3:2 | 12,10,16,6,5 | 19,9,16,5,4 | 19,9,16,5,4 |
| 2:1 | 10,11,16,6,4 | 17,10,16,5,3 | 17,10,16,5,3 |
| 3:1 | 10,11,16,6,4 | 17,11,16,4,3 | 17,12,16,3,3 |

At 4352 B the Galaxy's worker needs only 10-12 slots (the wire hides its window), so 7-9 slots move to the forwarding
side; sized at 16 B the Galaxy's picks are the P150's. A worker of 11-13 costs the Galaxy's small-packet 1-hop rows
+26-37 (13: 168.6 at 16 B; 12: 179.0). None of the Galaxy picks was run.

## P3: one mesh, VC0 only: Galaxy 8x4 against the P150 2x4 (Galaxy job 167464; P150 jobs 379, 380)

Same 2D splits (worker, forwarding, receiver; 81 slots) on both, two passes each (P150 pass-to-pass <= 1.9 on merges,
<= 0.7 elsewhere; Galaxy <= 0.8, multicast 1.9), 1D splits (worker, forwarding, receiver; 83 slots) likewise. Every run clean (P150
14/14 and 6/6, Galaxy 22/22 and 10/10 tests per run), slots as requested. Rows (sls_2d_*.yaml): the P150 rows on both
(1-hop, bidirectional, 3-hop E/W, a turn [0,0] -> [0,7] = 1 N/S + 3 E/W hops, a 3-flow row merge, multicast all-to-all)
and, Galaxy only, 3-hop and 7-hop N/S down column 0, a 10-hop corner, a 7-flow column merge. Merge rows are per link
(the profiler sums the flows sharing the last link), multicast per sender (8 vs 32 chips: not comparable across systems).

Cycles/packet, P150 / Galaxy, 1 KiB:

| row | 17,16,16 (today) | 20,15,16 | 26,13,16 | 11,18,16 | 8,19,16 | 25,16,8 | 19,10,32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 116 / 123 | 113 / 113 | 113 / 113 | 173 / 183 | 232 / 244 | 158 / 177 | 113 / 114 |
| 3-hop E/W | 145 / 144 | 145 / 147 | 159 / 164 | 178 / 189 | 235 / 252 | 170 / 186 | 198 / 205 |
| turn (1 N/S + 3 E/W) | 145 / 178 | 148 / 179 | 167 / 182 | 180 / 218 | 235 / 272 | 171 / 208 | 207 / 209 |
| row merge (3 flows, link) | 124 / 123 | 124 / 124 | 128 / 129 | 125 / 124 | 135 / 136 | 163 / 179 | 141 / 143 |
| 3-hop N/S | - / 182 | - / 184 | - / 200 | - / 218 | - / 273 | - / 212 | - / 250 |
| 7-hop N/S | - / 186 | - / 197 | - / 221 | - / 218 | - / 273 | - / 250 | - / 280 |
| 10-hop corner | - / 187 | - / 196 | - / 222 | - / 218 | - / 273 | - / 250 | - / 280 |
| column merge (7 flows, link) | - / 166 | - / 169 | - / 176 | - / 165 | - / 166 | - / 216 | - / 212 |

4352 B:

| row | 17,16,16 | 20,15,16 | 26,13,16 | 11,18,16 | 8,19,16 | 25,16,8 | 19,10,32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W | 127 / 252 | 126 / 252 | 126 / 252 | 187 / 252 | 254 / 254 | 179 / 252 | 126 / 252 |
| 1-hop bidirectional | 135 / 259 | 127 / 259 | 127 / 259 | 200 / 259 | 257 / 295 | 190 / 259 | 127 / 259 |
| 3-hop E/W | 146 / 252 | 155 / 252 | 173 / 252 | 191 / 252 | 258 / 264 | 185 / 252 | 216 / 252 |
| turn | 148 / 252 | 160 / 252 | 178 / 252 | 193 / 252 | 257 / 298 | 185 / 252 | 225 / 252 |
| row merge | 132 / 252 | 136 / 252 | 141 / 252 | 132 / 252 | 139 / 253 | 180 / 252 | 169 / 252 |
| 3-hop / 7-hop N/S, corner | - / 252 | - / 252 | - / 252-253 | - / 252-253 | - / 299 | - / 253-271 | - / 276-302 |
| column merge | - / 252 | - / 252 | - / 252 | - / 252 | - / 254 | - / 260 | - / 271 |

1D lines (P150 / Galaxy, 1 KiB; at 4352 B every row is on the wire, 124.5 / 249.5, except forwarding 8: 3-hop 264 /
263, Galaxy 7-hop 312):

| row | 26,25,32 (builder) | 19,32,32 (rule) | 43,8,32 | 35,32,16 |
| --- | ---: | ---: | ---: | ---: |
| 1-hop | 95.5 / 95.5 | 100.9 / 107.2 | 95.3 / 95.3 | 98.6 / 101.4 |
| 3-hop | 118.0 / 118.6 | 118.7 / 119.8 | 234.2 / 242.7 | 118.9 / 120.1 |
| row merge | 108.1 / 108.1 | 107.9 / 107.9 | 163.2 / 167.2 | 111.3 / 112.6 |
| 7-hop N/S line (Galaxy) | - / 120.6 | - / 119.9 | - / 295.0 | - / 127.3 |
| column merge (Galaxy) | - / 117.2 | - / 116.5 | - / 222.1 | - / 121.1 |

Windows (`sls_model.py fit 2d`, SLS2D_SIZE / SLS2D_RES; rate x W = RT0 + b W, or the points), 1 KiB:

| window | P150 | Galaxy |
| --- | --- | --- |
| worker (1-hop) | 1,757 + 13.5 W | 1,853 + 14.3 W (+5 %) |
| VC0 receiver 8 (T = 2), 1-hop | 1,271 | 1,413 (+11 %) |
| E/W forwarding (3-hop E/W) | 1,978-2,315 (10-16 slots) | 2,052-2,132 (10-13) |
| turn forwarding | 2,075-2,318 | 2,086 at 10, then a floor ~178 (below) |
| N/S forwarding, 3 hops | (no N/S forwarding in a 2-row mesh) | 2,499-2,605 (10-13), floor ~182 |
| N/S forwarding, 7 hops / corner | - | 2,511 + 28.8 W (2,800-2,950 at 10-15) |
| 1D forwarding at 8 slots | 3-hop 1,874 | 3-hop 1,942, 7-hop 2,360 |

At 4352 B: P150 E/W forwarding 1,862 + 29.7 W, worker ~2,052; Galaxy N/S forwarding at 10 slots 2,755 (3 hops) and
3,012 (7 hops), the worker on multi-hop rows 2,392 at 8, everything else on the wire.

What moves with mesh size (measured unless marked):
- The worker and E/W forwarding windows cost about the same on both systems at 1 KiB (Galaxy +3-5 %): the 2x4 and the
  8x4 agree on the worker (~18-20) and on E/W forwarding.
- Longer N/S paths exist only in meshes with more than two rows, and their forwarding loop is longer: 3-hop N/S ~2,550,
  7-hop ~2,950 against ~2,100 for 3-hop E/W (1 KiB). The 7-hop N/S row and the corner are forwarding-window-bound below
  16 slots (A/B across splits: 16 -> 15 -> 13 -> 10 slots: 186 -> 197 -> 221 -> 280; rate x W 2,970 -> 2,950 -> 2,870 ->
  2,800). RESULTS §98's open "7-hop N/S runs ~181" was v3q's (another router, job 155240); on the lead the row is
  window-bound below 16 forwarding slots and sits on a non-window floor at 16 (next item); v3q's case is not re-run.
- Above that, the Galaxy's N/S-involving rows have a floor no split moves: 3-hop N/S 182-184 at 15-16 slots, turn
  178-182 at 13-16, the corner and 7-hop ~186 at 16 (P150's turn: 145-148). Not a window (flat across 13-16 forwarding
  and 17-26 worker slots); which ERISC binds it is unknown (needs a one-side delay A/B on the Galaxy). Hypothesis: the
  N/S routers' stateless forwards on a mesh with more than two rows (V4_BOUNDS §4: glx_2d has 20 stateless E1 configs,
  bh_2d 8); the 1D 7-hop N/S line (stateful forwards everywhere) runs 120. [Settled 2026-11-02, V4_NS_FLOOR.md: E1 binds
  (one-side A/B), ~172 with deep windows; the N/S routers' stateless forwards (~+37 E1) and multi-downstream bookkeeping
  (~+17-19) reproduce it on E/W forwarders (single-change A/Bs, P150 and Galaxy); ~8 left, hypothesis the decode. At 16
  forwarding slots the window adds ~10-13 to the long rows.]
- Merging: the 7-flow column merge and the 3-flow row merge are link-aggregate rows; they barely move with forwarding
  (Galaxy column merge 165-176 over 13-19 slots) and pay for a small receiver (receiver 8: 216; P150 row merge 163).
- 1D: the forwarding window's loop grows with the line's length (Galaxy, 8 slots: 3-hop 1,942, 7-hop 2,360); 32
  forwarding slots carry the 7-hop line at 120. The 1D worker at 19 costs 1-hop 5.4 (P150) and 11.7 (Galaxy) at 1 KiB.
- Link speed: at 4352 B the Galaxy's wire (252) hides every window above worker ~10, forwarding ~13 (N/S at 10: 276-302)
  and receiver 8 (except the 7-hop rows: 271): the 2D split matters at 200G only for packets well under ~2 KiB.

The 2D rule (ALLOCATION_NOTE §5: receiver 16, the worker that binds no 1-hop row, forwarding the rest), by hand from
these fits: P150 1 KiB 20,15,16 (worker 1,757 / (113 - 13.5) = 17.7 -> 18, +2 leftover); P150 4352 B 17,16,16 (worker
2,052 / 125.8 = 16.3); Galaxy 1 KiB 20,15,16 (1,853 / (113.1 - 14.3) = 18.8 -> 19, +1); Galaxy 4352 B 11,18,16 (worker
2,392 / 252 = 9.5 -> 10, +1). On the Galaxy at 1 KiB, today's 17,16,16 against 20,15,16 trades the 1-hop rows (123 vs
113) for the long N/S rows (7-hop 186 vs 197, corner 187 vs 196): the same trade the rule makes on the P150 (1-hop 116
vs 113, turn 145 vs 148), three times larger.

## Patterns (measured) and what a mesh-size-aware rule would key on (hypotheses)

Patterns, from P1-P3:
1. The VC0:VC1 weight trades the same currency on both systems: each VC1 forwarding slot per channel (3 slots) taken
   from VC0 forwarding costs the 2-hop row 20-30 % around 6-9 slots and buys the VC1 rows 15-25 %, until the VC1 path's
   own bound (~294 P150, ~296-299 Galaxy) at VC1 forwarding ~8 and ring ~6-7. Past that VC1 weight buys nothing: 1:2 and
   1:1 give the same allocation on both fits. The VC0 side saturates too: the P150's 2-hop row has a bound of its own
   near 220 (forwarding 11 / 12: 225 / 220 at 4352 B), the Galaxy's reaches the wire at 11 (252.9).
2. Link speed moves the binding windows, the 2x2 mesh size does not: at 16 B the Galaxy's windows are the P150's within
   ~5 % and the rule picks the same allocations; at 4352 B the Galaxy's VC0 rows sit on the 200G wire (252) for worker
   >= ~10, VC0 ring >= 8 and VC0 forwarding >= 11, while the VC1 path stays router-bound (~296) and keeps needing its
   windows. Sized at 4352 B the Galaxy rule moves 7-9 slots from the worker to forwarding (11,8,16,8,6 at 1:1).
3. Path shape moves them within a mesh: N/S forwarding (meshes with more than two rows) has a longer loop than E/W
   (1 KiB: 3-hop N/S ~2,550, 7-hop N/S ~2,950, 3-hop E/W ~2,100); a VC0 hop into an exit chip has a longer loop than a
   plain forwarding hop (Galaxy 4352 B, 9 slots: 2,953 against 2,490); in 1D the forwarding loop grows with the line
   (8 slots: 3-hop 1,942, 7-hop 2,360). One depth covers every forwarding channel of a VC, so the longest loop the
   traffic uses sets it.
4. Merging rows (link aggregates) move with the receiver (8 -> 16: Galaxy column merge 216 -> 166) more than with
   forwarding (13-19 slots: 165-176).
5. Two windows close to each other on one path run slower than either alone (P150 18,8,16,6,5: VC1 forwarding ~337
   and ring ~303 alone, 341 together; points within ~10 of the VC1 bound sit ~10 above it). Related: the worker's loop
   is longer on forwarded rows than on 1-hop rows (worker 11: Galaxy 2,400 against 2,020, every multi-hop row at 218;
   P150 1,960 against 1,905); hypothesis: a window's loop includes the slack of the windows downstream of it. The max()
   model misses both (its largest errors are such points).
6. Which VC1 window binds: forwarding while F1 < ~1.33 R1 (the loop ratio; P150 and Galaxy alike); today's 19,9,16,5,4
   is forwarding-bound on both systems' E/W VC1 hops, ring-bound on the Galaxy's N/S hop.
7. Small-packet VC0 1-hop rows need the worker at ~18-20 on both systems (16 B / 1 KiB); bursts of 8 need 19 on the
   P150 (steady rows 17). At 4352 B the P150 needs 17, the Galaxy ~10.

What a mesh-size-aware allocator would key on (hypotheses, none tested as a rule):
- H1, the design packet size and the link speed together: size each window against max(router bound, wire time at
  that size). On 400G the router binds at every size (same picks at 16 B and 4352 B); on 200G the wire hides the VC0
  1-hop windows above ~2 KiB, so a large-packet design point frees ~7-9 worker slots (and could take the VC0 ring to 8)
  that small packets would miss (+26-37 on 16 B 1-hop rows at worker 11-13).
- H2, the mesh's shape: a mesh with more than two rows has N/S forwarding with ~20-40 % longer loops (and, on the lead,
  a non-window floor), so its forwarding depth should come from the N/S loop; a 2-row mesh's from E/W. Per-direction
  depths (N/S channels deeper than E/W) would follow the loops more closely (a builder change; not proposed here).
- H3, where inter-mesh traffic enters: in meshes larger than 2x2 most inter-mesh flows forward on VC0 into an exit chip,
  whose loop is the longest VC0 forwarding loop measured; a multi-mesh rule would size VC0 forwarding from it.
- H4, the VC1 path: its bound (~295) is the router's, the same at 400G and 200G; VC1's windows (forwarding ~8, ring
  ~6-7 for one or two VC1 hops) look mesh-size independent at 2x2. Longer VC1 paths (4x4: up to 6 VC1 hops, the 4x2
  ring's pass-through) are untested (P4).
- H5, the weight itself: with VC1 saturating at ~31 slots, "VC1 to its bound, the rest to VC0" (1:1) or "VC0 first"
  (3:2 = today's row) are the two natural policies; between them the P150 trades 2-hop 258 -> 322 (+25 %) for VC1
  one-hop 396 -> 308 (-22 %; measured, 19,9,16,5,4 against 18,7,16,7,5), the Galaxy (4352 B sizing, model) 2-hop 257 ->
  301 for VC1 369 -> 299.

Open questions:
- O1: the Galaxy's N/S floor (~178-187 at 1 KiB on every N/S-involving 2D row; P150 turn 145): which ERISC binds it
  (one-side delay A/B on the Galaxy), and whether it is the stateless N/S forwards (hypothesis). [Answered 2026-11-02:
  V4_NS_FLOOR.md.]
- O2: a burst-aware worker step (19 instead of 17 on the P150)?
- O3: which packet size(s) a 200G system's allocation should be sized for (H1) - a policy input like the weight.
  [Measured 2026-11-02 for 2 KiB slots: section "2 KiB slots on a 200G Galaxy"; the decision stays the user's.]
- O4: the model's max() misses close windows (pattern 5); a smoother combination (e.g. adding the windows' occupancy)
  would change the picks near the VC1 bound.
- O5: VC1 on 4x4 / 4x2 meshes (P4, after the command-buffer-2 fix): longer VC1 paths, pass-through, exit-chip
  forwarding under VC1 load.
- O6: per-link loop differences on the Galaxy (E/W VC1 hop 2,178 against N/S ~2,000 at 5 slots on the 2x2 split):
  calibrate per link kind or per direction?
- O7: alloc_model.py's VC1 row map and vc1_mix.yaml describe the pre-merge placement (direct row 1_1, mix flow "then W
  on VC1"); today they mislabel the rows (their owner's call; sls_model.py / sls_vc1_all.yaml use the post-merge map).

## 2 KiB slots on a 200G Galaxy (user, 2026-11-02: "play around a 2K packet size on a galaxy")

Galaxy jobs 167648 (G1) and 167673 (G2), trees sls_bdd / sls_bddsl / sls_bddml (sls_lead + variants/v4bdd, the lead at
delay 0, byte-identical: 464 / 464 Galaxy coverage images, `bnd_loadcmp.py`); `fabric_exp/slotstudy/sls_glx2k.sbatch`;
predictions `sls2k_predictions.txt` (before G1), `sls2k_predictions_g2.txt` (before G2); tables
`sls2k_report.py fabric_exp/results/glx167648 lr|arms|2d|vc1|vc2`. Every step clean, slots logged as requested, no hang.

**What existed.** The "~2.5K" is RESULTS §98's Galaxy size sweep (job 155281, v3q, 4352 B slots): "v3q saturates the
link from ~2.3 KiB in 2D and ~2.5 KiB in 1D; at exactly 2 KiB it is 4.5-5 cycles short in 2D (96-96.5 %) and 9.5 short in
1D (93 %)", with the wire line 12.80 + P / 18.50 cycles/packet in 2D (+3.0 above ~4 KiB, a second Ethernet frame).
VC2_LINE_IDEA.md: "200G Galaxy use cases saturate at smaller packets, which give more slots than the ~81 per ERISC at
4352 B ... with ~2-2.3 KB packets (151-169 slots) ... Needs the packet-size-aware allocation rule (deferred, §25 of
V4_RECEIVER_SHAPE.md)"; V4_RECEIVER_SHAPE §25: the study rows apply when the row's design worker fits ("the default
4352 B packet: 81 slots; a larger packet keeps the tier table").

**Configuration.** The fabric's payload size is the router config's `max_packet_payload_size_bytes` (default 4352 B;
FabricContext::compute_max_payload_size_bytes), set from a test yaml by `fabric_setup: max_packet_size: N` (the fabric is
reopened when it changes); the slot is payload + header (2D / VC1 / VC2: +96 B; 1D: +48 B). Slots per ERISC, measured
(mock captures, TT_FABRIC_EXP_LOG_SLOTS) and on the Galaxy:

| fabric | 4352 B (4448 B slots) | 2560 B (2656 B) | 2048 B (2144 B / 1D 2096 B) | builder default at 2048 B |
| --- | ---: | ---: | ---: | --- |
| 2D (VC0 only) | 81: 17,16,16 | 137: 57,16,32 | 169: 41,32,32 | the tier table at 2 KiB (forwarding / receiver 32) + spare to the worker |
| 1D | 83: 26,25,32 | | 174: 71,71,32 | even split |
| VC0 + VC1 split | 81: 19,9,16,5,4 | 137: 75,9,16,5,4 | 169: 107,9,16,5,4 | the 4352 B study row, every extra slot to the worker |
| VC0 + VC2 | 81: 17,14,16 + 2,4 | 137: 73,14,16 + 2,4 | 169: 105,14,16 + 2,4 | likewise |

So at 2 KiB the builder keeps the 4352 B windows on VC1 / VC2 fabrics and gives 88 extra slots to a worker that needs
~20: the study rows must be re-derived per slot size (below).

**Line rate (G1, 1-hop single flows, 200,000 packets, default slots of each configuration).**

| payload | wire | 4352 B slots (worker 17) | 2560 B slots (worker 57) | 2048 B slots (worker 41) |
| ---: | ---: | --- | --- | --- |
| 1024 | 68.2 | 122.8 / 11.3 GB/s (worker window) | 111.0 / 12.5 | 109.8 / 12.6 (E0) |
| 1536 | 95.8 | 124.8 / 16.6 | 110.6 / 18.8 | 109.8 / 18.9 (E0) |
| 2048 | 123.5 | 127.9 / 21.6 (96.6 % of the wire) | 124.0 / 22.3 (99.6 %) | 124.0 / 22.3 (99.6 %) |
| 2304 | 137.3 | 138.0 / 22.6 (wire) | 137.9 / 22.6 | |
| 2560 | 151.2 | 151.8 / 22.8 | 151.8 / 22.8 | |
| 3072 / 3584 / 4096 | 178.9 / 206.5 / 237.2 | 179.6 / 207.4 / 238.2 (wire; 23.1 / 23.3 / 23.2 GB/s) | | |
| 4352 | 251.0 | 252.1 / 23.3 GB/s | | |

(E/W and N/S within 0.6. cycles/packet / payload GB/s at 1.35 GHz.)
- Confirmed for 2D: with 4352 B slots the lead reaches the wire at ~2.3 KiB (2304 B: 99.6 %), not at 2 KiB (96.6 %);
  RESULTS' ~2.5 KiB was 1D (v3q). Refined: what stops 2 KiB packets is the 17-slot worker window (at 2 KiB slots the
  same row with 17,16,16 runs 128.2, with 41 workers 124.0), so with 2 KiB (or 2.5 KiB) slots the lead reaches the wire
  at 2048 B.
- The wire carries 2048 B packets at 22.3 GB/s against 23.3 at 4352 B (-4.3 %: the ~237 B per-packet framing over
  twice the packets). Below ~1.9 KiB the router (E0, ~110) binds 1-hop rows, not the wire.

**2D rows (G1; P3's rows, cycles/packet, 1 KiB payload; 2 KiB alike within ~5 except where noted).**

| row | 4352 B slots 17,16,16 | 2 KiB 17,16,16 | 2 KiB default 41,32,32 | 2 KiB 34,32,32 | 2 KiB 48,24,32 | 2 KiB 64,16,32 | 2 KiB 16,40,32 | 2 KiB 64,32,8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W (2048 B) | 122.9 (127.9) | 122.5 (128.2) | 109.6 (124.0) | 110.0 (124.0) | 110.5 (124.0) | 109.3 (124.0) | 126.6 (133.3) | 176.1 (183.6) |
| 3-hop E/W | 143.4 | 140.8 | 140.7 | 139.5 | 140.9 | 140.3 | 140.6 | 184.7 |
| turn (1 N/S + 3 E/W) | 178.1 | 175.8 | 170.5 | 170.6 | 171.4 | 174.3 | 171.9 | 205.2 |
| 3-hop N/S | 182.3 | 180.6 | 172.5 | 172.5 | 174.4 | 179.5 | 173.2 | 208.7 |
| 7-hop N/S (2048 B) | 186.0 (190.1) | 184.6 (189.1) | 173.1 (173.3) | 173.2 | 175.2 | 184.4 (188.8) | 173.2 | 247.0 |
| 10-hop corner | 186.7 | 186.6 | 173.2 | 173.2 | 175.3 | 186.2 | 173.2 | 247.2 |
| row merge (link) | 123.0 | 120.7 | 119.9 | 119.7 | 120.3 | 120.2 | 120.3 | 179.3 |
| column merge (link) | 166.2 | 164.1 | 158.4 | 158.5 | 159.3 | 162.7 | 157.8 | 214.6 |
| multicast all-to-all | 567.3 | 561.1 | 554.4 | 554.5 | 556.7 | 570.5 | 546.5 | 731.1 |

- The same slot counts give the same rates at both slot sizes (17,16,16: within 3): a window is set by its slot count
  and the payload, not the slot size. 2 KiB slots buy their effect only through the doubled counts.
- With ~2x the slots no 2D window binds once the worker has ~20, the forwarding channels ~24-32 and the receiver 16-32:
  1-hop rows reach E0's floor (~110, 1 KiB) or the wire (2048 B); E/W multi-hop sits on E0 (~140); every N/S-involving
  row on E1 (~172; V4_NS_FLOOR.md: one-side A/B and the mechanism). Forwarding 16 still binds the long N/S rows
  (184-188); a receiver of 8 binds everything (+50-75).
- The 2D fit at 2 KiB slots (`SLS2D_RES=... SLS2D_KIND=2d2k SLS2D_CAP=169 sls_model.py fit 2d`, 1 KiB): worker 1,834 +
  13.5 W (4352 B slots: 1,853 + 14.3 W), N/S 7-hop forwarding ~2,950-2,990 at 16; the rule by hand: worker 20,
  receiver 32, forwarding the rest (39): every row at its bound (prediction 20,39,32: G2 below).

**VC1 split rows (G1 + G2; Galaxy 2x2 split; cycles/packet at 2048 B payload unless noted; mean of the passes, <= 1.1
apart).**

| row | 4352 B slots 19,9,16,5,4 | 2 KiB default 107,9,16,5,4 | 2 KiB 38,18,32,10,8 | 2 KiB 28,16,32,16,12 | rule @2048 B: 81,14,16,8,6 | rule @16 B: 88,13,16,7,5 | 2 KiB 32,18,32,12,10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W (16 B) | 142.0 (142.0) | 139.7 (139.5) | 139.5 (139.5) | 138.7 (138.7) | 138.6 (138.5) | 138.0 (138.0) | 138.6 (138.6) |
| 2-hop | 255.8 | 255.0 | 167.9 | 168.3 | 172.1 | 180.7 | 168.1 |
| 2-hop merged | 144.2 | 145.4 | 128.1 | 133.3 | 131.6 | 131.5 | 130.1 |
| multicast | 149.8 | 148.2 | 132.3 | 137.2 | 135.1 | 135.2 | 131.6 |
| inter direct | 208.9 | 207.6 | 206.7 | 206.8 | 207.7 | 207.7 | 206.7 |
| VC0 hop + inter direct ([0,0] -> [1,0]) | 302.5 | 301.8 | 234.4 | 237.0 | 238.5 | 239.5 | 236.2 |
| +1 VC1 hop E/W (16 B) | 390.0 (368.1) | 389.2 (366.7) | 282.6 (281.4) | 276.1 (275.8) | 290.2 (287.0) | 296.8 (292.9) | 279.0 (278.5) |
| +1 VC1 hop N/S | 362.4 | 361.2 | 282.3 | 276.8 | 286.6 | 292.8 | 279.2 |
| +2 VC1 hops | 394.4 | 394.8 | 280.2 | 274.0 | 289.3 | 335.8 | 276.7 |
| mix: VC0 worker / its VC1 flow | 204 / 430 | 202 / 434 | 184 / 289 | 182 / 277 | 189 / 305 | 193 / 346 | 182 / 281 |

- The builder's default at 2 KiB slots runs as the 4352 B row (same windows; the 88 extra worker slots buy 2-3 on 1-hop).
- With the slots spent on the windows, VC0 and VC1 reach their bounds together: VC0 2-hop 168 (its path's bound; 255 with
  9 forwarding slots), VC1 rows 274-283 (the VC1 path's bound at 2048 B, ~276; 361-395 with forwarding 5 / ring 4), and
  the shared-link mix rows improve on both flows. VC1's window limit goes away; no VC0:VC1 weight is needed at 2 KiB (the
  rule's picks are the same at every weight).
- The fit at 2 KiB slots (`GLX_KIND=vc12k GLX_CAP=169 sls_model.py fit glx`, G1's 7 allocations, 98 points, mean
  |error| 0.9 %): VC0 forwarding 2,295 at 9 (2048 B), VC1 forwarding 1,946-2,000 at 5-6, ring 1,412 at 4, worker 2,293 at
  13; bounds 1-hop 137.7, 2-hop 167.9, VC1 path 276.2, two VC1 hops 274.0. Predictions for G2's picks
  (`sls2k_predictions_g2.txt`): within 0.5-5 % (2-hop 167.9 / 176.5 predicted, 172.1 / 180.7 measured; VC1 one hop 276.2
  / 282.3 predicted, 290.2 / 296.8 measured: allocations sized exactly to the bounds land ~10-15 above them, as on the
  P150). The hand-made 32,18,32,12,10 (the leftovers to the receivers and VC1 instead of a worker of 81-88) beats the
  rule's picks by 4-13 on VC0 2-hop and the VC1 rows: at 2 KiB the leftover rule ("the rest to the worker") wastes
  slots.
- 2D at 2 KiB: the rule by hand, 20,39,32, measured within 0.7 of its prediction on every row (1-hop 110.1, 3-hop E/W
  139.6, N/S 172.3-172.4, turn 170.9).

**VC2-on rows (Galaxy single mesh, 2048 B payload; VC0 and VC2 from [0,0] to [0,1]; per-sender rates from the profiler's
log lines; VC0 alone with VC2 off: 127.9 at 4352 B slots, 124.0 at 2 KiB).**

| slots (VC0 w,f,r + VC2 w,r) | VC2 alone | VC0 alone, VC2 on (idle) | contention: VC0 / VC2 |
| --- | ---: | ---: | --- |
| 4352 B: 17,14,16 + 2,4 (study row) | 869.3 | 135.2 | 163.0 / 916.0 |
| 4352 B: 17,10,16 + 8,10 | 246.3 | 135.3 | 189.6 / 358.2 |
| 2 KiB: 105,14,16 + 2,4 (default) | 866.8 | 131.3 | 159.3 / 911.1 |
| 2 KiB: 34,28,32 + 4,8 | 473.2 | 131.0 | 164.1 / 536.2 |
| 2 KiB: 34,24,32 + 8,8 | 245.6 | 131.0 | 172.6 / 398.1 |
| 2 KiB: 34,20,32 + 16,16 (two passes) | 143.8 | 130.2 | 201.4 / 322.5 |

- VC2's 2-slot worker window binds VC2 at every slot size (~867-869 alone); its limit goes away at ~16 slots (143.8
  alone, near VC0's ~124-131), which 2 KiB slots make affordable. On a shared link the two VCs' rates add up to the wire
  (2 KiB, 16 / 16: 1 / (1/201.4 + 1/322.5) = 124.0 per packet; 4352 B slots, 8 / 10: 124.0).
- An idle VC2 costs VC0's 1-hop row ~7 at 2048 B on both slot sizes (135.2 vs 127.9; 131.3 vs 124.0), so with VC2 on, 2 KiB
  packets miss the wire (131 against 123.5).

**What 2 KiB slots cost: large transfers.** Per-packet router costs now apply to twice the packets per byte. Payload
GB/s at 1.35 GHz, a long stream of the largest packet each configuration carries, at its best measured allocation:

| path (Galaxy) | 4352 B slots, 4352 B packets | 2 KiB slots, 2048 B packets | change | bound at 2 KiB |
| --- | ---: | ---: | ---: | --- |
| 2D 1-hop (VC0 only) | 252.1 -> 23.3 GB/s | 124.0 -> 22.3 | -4.3 % | the wire (framing) |
| 2D 3-hop E/W | 252.1 -> 23.3 | 139.7 -> 19.8 | -15 % | E0 |
| 2D 3-hop / 7-hop N/S, corner | 252.1 -> 23.3 | 172.6-173.3 -> 16.0 | -31 % | E1 (V4_NS_FLOOR.md) |
| VC1-split 1-hop | 252.1 -> 23.3 | 138.6 -> 19.9 | -14 % | E0 (VC1 routers' pass) |
| VC1-split VC0 2-hop | 252.9 -> 23.2 (forwarding 11) | 168.1 -> 16.4 | -29 % | the 2-hop path |
| inter-mesh + VC1 hops | ~296-299 -> 19.7 (forwarding 8 / ring 7) | 276-279 -> 10.0 | -49 % | the VC1 path |
| VC2 alone (deep windows) | (not run deep at 4352 B) | 143.8 -> 19.2 | | |

- With 4352 B packets the 200G wire (251) hides every per-packet cost but the VC1 path's (~296); with 2048 B packets the
  wire (123.5) hides none but the 2D 1-hop row's. The N/S floor is per packet (1 KiB = 2 KiB within 0.4), so smaller
  packets multiply it: a clue for the open packet-size decision, measured.
- Small packets (<= 2 KiB) gain from 2 KiB slots only through the doubled slot counts: 1-hop at 1 KiB 109.6 vs 122.9
  (-11 %), VC1-split 2-hop 168 vs 256 (-34 %), VC1 rows 279 vs 390 (-28 %), VC2 alone 144 vs 869 (16 / 16 slots), N/S rows
  172 vs 182-186. The same counts at 4352 B slots do not fit 81 slots.

**Recommendation for 200G sizing (hypotheses where marked).**
- Measured: 4352 B slots for bandwidth on large transfers (every path but VC1 sits on the wire, 23.3 GB/s); 2048 B slots
  lose 4-49 % there and win 11-34 % on small packets and on window-bound VC1 / VC2 rows. Neither size is better for both.
- Whatever the slot size, allocate it deliberately: at 2 KiB the builder's 4352 B rows plus "the rest to the worker"
  waste ~88 slots (VC1 rows 389 vs 279). Rows that measured well at 2 KiB: VC1 split 32,18,32,12,10; 2D 20,39,32 (or
  34,32,32); VC0+VC2 34,20,32 + 16,16. At 4352 B on 200G: the rule sized at 4352 B (P2: worker ~10-12).
- Hypothesis (not run): a ~3-3.5 KiB slot (wire 179-207 cycles; ~99-110 slots per ERISC) keeps every VC0 path on the
  wire at large transfers (their per-packet costs are ~110-186) while giving ~25-35 % more slots than 4352 B, enough for
  VC1 forwarding ~8 / ring ~7 next to VC0 forwarding ~12-14. Next experiment: 3072 / 3584 B slots on the Galaxy with
  the N/S, VC1 and VC2 rows, then the same packet-size-aware rule.
- Cutting the N/S routers' per-packet E1 cost (V4_NS_FLOOR.md §3) would move the crossover toward smaller packets.

## 3-3.5 KiB slots on the 200G Galaxy (user, 2026-11-02: approved next experiment)

Galaxy job 167723 (bh-glx-110-a08u20; tree sls_bdd = the lead at delay 0; 26 steps, all clean: 33/33, 49/49 per rank,
8/8; slots as requested; no hang; results `fabric_exp/results/glx167723/`, job directory deleted). Predictions
`sls3k_predictions.txt` (written 05:05, before the job): VC1 from `sls3k_model.py` (each window kind's round trip
interpolated in payload between the Galaxy fits at 2048 B and 4352 B; bounds max(router, wire)); 2D and VC2 by hand.
Tables: `sls3k_table.py fabric_exp/results/glx167723` (one table per path and slot size).

**Slots per ERISC** (mock captures, confirmed on the Galaxy): 3072 B payload (3168 B slots) 115 for 2D, the VC1 split and
VC2-on (1D 117); 3584 B (3680 B slots) 99. Builder defaults: 2D 35,16,32 / 19,16,32; VC1 53,9,16,5,4 / 37,9,16,5,4 (the
4352 B study row, the rest to the worker); VC2-on 51,14,16 + 2,4 / 35,14,16 + 2,4.

**Allocations and the model.** The interpolated model's pick is the same at every VC0:VC1 weight at both sizes (24,15,16,
8,6 at 3072 B, 14,13,16,8,6 at 3584 B): with 99-115 slots both VCs reach their bounds, so the weight question vanishes
as at 2 KiB. Its predictions for the 10 VC1 allocations run: 70 points, mean |error| 1.5 %, max 6.1 %, 59 within 3 %
(misses: two-VC1-hop rows 5 % optimistic, VC0 hop + inter 3-4 %). Refits at each size (`GLX_KIND=vc13k GLX_CAP=115`,
`vc135k 99`): 70 points each, mean 0.9 % / 0.7 %; their picks put the VC1 ring at 1 because no run at these sizes had a
ring-bound point (a gap in the calibration, not a finding), so the interpolated picks stand. At 3584 B the remaining
trade is not VC0 against VC1 but the worker against VC0 forwarding: the rule sized at 3584 B gives a 14-slot worker,
which costs small packets (VC1-split 1-hop at 256 B 156.0 against 139.1 at 20 slots); 20,11,16,8,6 keeps small packets
and loses the VC0 2-hop row's wire (233.2 against 207.6). 2D by hand: worker 20, receiver 32 / 16, forwarding the rest:
20,21,32 (3072 B), 20,21,16 (3584 B).

**One table per path** (cycles/packet at the slot's largest payload -> payload GB/s at 1.35 GHz; "wire" within 1 % of
12.80 + P / 18.50; the best measured allocation per size, named; small payload = 1 KiB for 2D, 256 B or 16 B for VC1):

| path | 2048 B slots | 3072 B slots | 3584 B slots | 4352 B slots | bound below the wire (from the router per-packet costs) |
| --- | --- | --- | --- | --- | --- |
| wire | 123.5 -> 22.4 | 178.9 -> 23.2 | 206.5 -> 23.4 | 251.0 -> 23.4 | |
| 2D 1-hop | 124.0 -> 22.3 wire (41,32,32) | 179.6 -> 23.1 wire (20,21,32) | 207.4 -> 23.3 wire (20,21,16) | 252.1 -> 23.3 wire (17,16,16) | E0 ~110 |
| 2D 3-hop E/W | 139.7 -> 19.8 (E0) | 179.6 -> 23.1 wire | 207.4 -> 23.3 wire | 252.1 -> 23.3 wire | E0 ~140 |
| 2D 3-hop N/S | 172.6 -> 16.0 (E1) | 179.6 -> 23.1 wire (35,16,32: 182.7) | 207.4 -> 23.3 wire | 252.1 -> 23.3 wire | E1 ~172 + window at 16 forwarding |
| 2D 7-hop N/S, corner | 173.3 -> 16.0 (E1) | 180.0 -> 23.0 wire (35,16,32: 195.0 / 193.5, window) | 207.4 -> 23.3 wire (22,15,32: 211.9) | 252.1 -> 23.3 wire | E1 ~173; window ~3,000 / forwarding |
| 2D turn | 170.6 -> 16.2 (E1) | 179.6 -> 23.1 wire | 207.4 -> 23.3 wire | 252.1 -> 23.3 wire | E1 ~170 |
| 2D row merge (link) | 129.3 -> 21.4 | 179.6 -> 23.1 wire | 207.4 -> 23.3 wire | 252.1 -> 23.3 wire | E0 |
| 2D column merge (link) | 163.1 -> 17.0 (E1) | 180.8 -> 22.9 | 207.3 -> 23.3 wire | 252.1 -> 23.3 wire | E1 ~158-163 |
| 2D multicast all-to-all (per sender) | 553.5 -> 5.0 | 705.8 -> 5.9 | 817.5 -> 5.9 | 993 -> 5.9 | the links' sharing |
| VC1-split VC0 1-hop | 138.6 -> 19.9 (E0) | 179.6 -> 23.1 wire | 207.4 -> 23.3 wire | 252.1 -> 23.3 wire | VC1 routers' E0 ~139 |
| VC1-split VC0 2-hop | 168.1 -> 16.4 (path) | 179.7 -> 23.1 wire (20,15,16,9,7) | 207.6 -> 23.3 wire (14,13,16,8,6); 233.2 -> 20.7 (20,11,16,8,6) | 252.9 -> 23.2 wire (20,11,16,3,3: VC1 at 717) | the 2-hop path ~168 |
| VC1-split multicast | 131.6 -> 21.0 | 180.9 -> 22.9 | 209.5 -> 23.1 | 255.1 -> 23.0 | |
| inter-mesh direct | 206.7 -> 13.4 | 208.3 -> 19.9 | 208.6 -> 23.2 | 252.1 -> 23.3 wire | the crossover ~207 |
| VC0 hop + inter direct | 236.2 -> 11.7 | 238.7 -> 17.4 | 242.2 -> 20.0 | 278.6 -> 21.1 (forwarding 11) | ~236-242; window at 4352 B |
| inter + 1 VC1 hop E/W | 279.0 -> 9.9 | 286.5 -> 14.5 | 292.4 -> 16.5 | 298.6 -> 19.7 (13,7,16,8,7: VC0 2-hop 330) | the VC1 path ~276-299 |
| inter + 2 VC1 hops | 276.7 -> 10.0 | 285.0 -> 14.6 | 302.8 -> 16.0 | 295.9 -> 19.9 (13,7,16,8,7) | the VC1 path |
| VC2 alone (16 / 16 slots) | 143.8 -> 19.2 (router) | 179.6 -> 23.1 wire | 207.3 -> 23.3 wire | (2 / 4: 869 at 2048 B) | ~141-144 |
| VC0 + VC2, one link (16 / 16) | 201 / 323 (sum = the wire) | 348 / 371 (~half the wire each) | 400 / 431 (~half each) | | the wire |

Small payloads in the same allocations (measured; 2D at 256 B / 1 KiB): 2D 1-hop 110.0-110.3 (2D default 3584 B
19,16,32: 112.1 at 1 KiB, the worker), E/W 3-hop 141, N/S rows 175-178 (E1 ~172 + the 21-slot window), turn 173,
column merge 160-162, VC1-split 1-hop 138.6-139.1 (14 workers: 156.0), VC1 rows 282-288, VC2 alone 141 (16 / 16),
VC0 with VC2 idle 131.3-131.8 at 1 KiB (VC2 off 109.5-110: idle VC2 costs E0 ~21 at 1 KiB, hidden by the wire at
3072 / 3584 B: 179.6 / 207.3). With the builder defaults (VC2 2 / 4) VC2 alone stays at 833-944 at every size.

What it shows (measured):
- From 3072 B packets every VC0 path measured sits on the wire (23.1 GB/s at 3072 B, 23.3 at 3584 B; 4352 B 23.3) once
  its windows allow, because every VC0 router per-packet cost measured (E0 110-140, E1 ~170-173) is under 179 cycles. The
  2D defaults at 3072 B (forwarding 16) leave the long N/S rows 15 above the wire (window): forwarding 21 removes it.
- Paths with a router cost above the wire stay router-bound at 3072 / 3584 B: inter-mesh direct (~208: at the wire only
  from ~3.5 KiB), VC0 hop + inter (~240), and the VC1 path (~285-303). Their payload GB/s grows with the packet: the VC1
  rows 10.0 (2 KiB) -> 14.5 (3 KiB) -> 16.5 (3.5 KiB) -> 19.7-19.9 (4352 B, but only with VC1 windows that starve VC0
  at 81 slots).
- VC2's window limit goes away at 16 / 16 slots, which 3072 / 3584 B slots afford: VC2 at the wire alone, and VC0 + VC2
  share the link half and half.

Recommendation for 200G sizing (measured where stated; the default choice is the user's):
- **3072 B slots as the one default (115 slots)**: every VC0 path at the wire (-0.9 % against 4352 B), VC0 and VC1
  windows both at their bounds with no weight trade (20,15,16,9,7: VC0 2-hop 179.7 wire, VC1 one hop 286.5, small-packet
  1-hop 139.0), the 2D split 20,21,32 with every row at the wire and small packets at their router floors, VC2 at 16 / 16
  (VC2 at the wire). Its cost is the paths whose router cost is above 179: inter-mesh direct 19.9 GB/s (-15 % against
  4352 B), VC0 hop + inter 17.4 (-18 %), VC1 rows 14.5 (-26 % against the best 4352 B VC1 allocation, which costs VC0's
  2-hop 23 %: 330.4, 17.8 GB/s).
- 3584 B slots (99): inter-mesh direct at the wire too, VC1 rows 16.5 GB/s (-16 %), VC0 paths at the wire; but 99 slots
  force a worker-or-forwarding trade on the VC1 split (small packets 156 vs 139, or VC0 2-hop 233 vs 208).
- 4352 B slots stay best only for large VC1 transfers (19.9 GB/s), at the cost of the VC0:VC1 weight trade at 81 slots.
- 2 KiB slots: best small-packet window relief per slot, worst large-transfer bandwidth (every router-bound path).
- Hypothesis: the N/S floor fixes (V4_NS_FLOOR.md §3, about -37 and -17 E1 cycles) would not change large-transfer GB/s
  at 3072 B or more (those rows are already at the wire); they would move N/S rows at 2 KiB from 16.0 GB/s toward E0's
  bound (~140 -> 19.8 GB/s) and the small-packet N/S rows from ~172-178 toward ~140 at every size. Lowering the VC1 path's
  per-packet cost (~285) is what would move the VC1 rows at 3 KiB.

## Galaxy harness (P2-P4)

- Tree `/data/aagarwal/frs-glx/trees/sls_lead` (`mk_sls_tree.sh`, 2026-11-01 20:25, 734 MB, 17k files, 160 s): a lean
  copy of the live study tree including its build_Release (the /data `base` tree of 10-29 lacks the lead's host named
  args `SENDER_CH_<i>_CONNECTION_SEMAPHORE_ADDR`, so the lead does not build on it), the lead's 18 files swapped in; a
  swap was active in the live tree during the copy (same manifest before and after), its files were put back from the
  backup first. python_env = the VC2 line agent's vc2l_p4 venv (system python + tt-exalens, so hang_capture.py runs).
  Host state: the live tree's build (the main session's planned rebuild, job 386, failed and did not change it).
  Checked against git HEAD (22:12, main's hazard note): in tt_metal/fabric and tt_metal/hw/inc the tree differs from
  HEAD only in the lead's 18 files (byte-equal to variants/v4rxv2i3) and edm_fabric_worker_adapters.hpp (= the live
  tree's uncommitted study change); dataflow_api_raw.h and every other hw/inc header equal HEAD.
- `sls_glx.sbatch`: steps `vc1:` (two 2x2 meshes, ttrun.py, `--oversubscribe --mca btl self,sm`), `2d:` / `1d:` (the
  whole Galaxy as one 8x4 mesh / lines), `vc1x44:` / `vc1x42:` (P4, refused without P4_OK=1); slots logged and checked;
  a timeout kills only processes naming the job's tree, captures (hang_capture.py), `tt-smi -glx_reset`, stops. Dry-run
  with stubs: normal path, hang path (kill + capture + reset + stop), the P4 gate, P4 steps with 2 / 4 stub ranks.
- Mock vets (CPU only, `sls_mock_vet.sh`, the sls_lead tree's host and router on the mock Galaxy / a09u32
  descriptor; tests in sanity mode because the mock's 0 MHz clocks stop benchmark mode before traffic setup): every
  group of sls_2d_glx (11), sls_1d_glx (5), sls_glx_vc1 (21, both ranks), sls_2d_p150 (7) and sls_1d_p150 (3) reaches
  program launch; slots logged as requested.
- The Galaxy 8x4 grid in `single_bh_galaxy_clus_desc` (backtracking embedding of the grid graph in the descriptor's
  links): rows [0, 4, 12, 8] [1, 5, 13, 9] [2, 6, 14, 10] [3, 7, 15, 11] [27, 31, 23, 19] [26, 30, 22, 18] [25, 29, 21, 17]
  [24, 28, 20, 16]: chips 0-3 run down the long side, each tray of 8 chips is a 4x2 block. The VC2 agent's 2x2 split
  ({0,1,4,5} | {2,3,6,7}) is two 2x2 blocks of tray 0, joined by 1-2 and 5-6.
- 2026-11-02 (2 KiB study, N/S floor): trees `sls_bdd` (sls_lead + variants/v4bdd: 4 files), `sls_bddsl`, `sls_bddml`
  (+ the diagnostic variants' router_downstream.hpp), hard-linked, each checked to differ from its parent in its
  variant's files only; built from /data trees, so the live tree's swap state did not matter; the host is still
  sls_lead's (before the main session's 22:25 host rebuild; the lead reads none of its new args).
  `sls_glx2k.sbatch` (a copy of sls_glx.sbatch): kinds lr / 2d2k / 2dk / vc12k / vc1k / vc22k / vc2k, a TAG starting
  tx<N> / rx<N> / sp<N> / rp<N> arms v4bdd's delay hooks, KIND@TREE runs a step in another tree (XTREES); dry-run with
  stubs (normal, hang, multi-tree, arms). Jobs 167648 (33 steps) and 167673 (25 steps): all clean; results in
  `fabric_exp/results/glx167648/`, `glx167673/`; job directories deleted.
- Busy check: job 167463 stopped at once on bh-glx-120-c02u14 (a 7-day-old uid 1001 k8s shell matched the check, as for
  the VC2 agent's 167439); resubmitted as 167464 excluding c02u14 and c04u02.

## P4 (prepared, not run): VC1 on meshes larger than 2x2

Waiting for main's word that the command-buffer-2 fix (V4_BOUNDS.md §4, option (c), v4rxo* stage S2) is in.
- `fabric_exp/slotstudy/glx_4x4_split_{mgd.textproto,rank_bindings.yaml}`: chips 0-15 | 16-31 (two 4x4 blocks of the
  embedding above; 28 links between them incl. wraps and express, RELAXED count 8). `glx_4x2_split_*`: the four trays
  as four 4x2 meshes in a ring 0-1-2-3-0 (16 / 14 / 16 / 14 links between neighbours; no 0-2 or 1-3 links).
- Yamls `sls_glx_vc1_4x4.yaml` (48 tests: sanity, 3-hop E/W and N/S, a 6-hop corner, a row merge, inter-mesh rows
  corner to corner) and `sls_glx_vc1_4x2.yaml` (24 tests: sanity, rows along the 4-chip side, [0, 0] to meshes 1, 2, 3;
  mesh 2 is two inter-mesh links away, so its traffic passes through mesh 1 or 3 on VC1). Mock: every vetted group
  (sanity, a 3-hop N/S, two inter rows per split, the 0 -> 2 pass-through) reaches launch on all ranks (2 / 4).
- Router args on the mock (TT_FABRIC_EXP_DUMP_ARGS, 19,9,16,5,4): 4x4 split: MESH_X_SIZE 4, intermesh routers on mesh 0
  chips 0-3 (N) and mesh 1 chips 12-14 (S), 15 (E); 72 routers (144 ERISCs) are E/W, not intermesh / edge, and have
  RX_CH_0 and RX_CH_1 FWD_DATA_CMD_BUF_ID both 2 (the routers where VC0's stateful forwards and VC1's straight E/W
  forwards would share buffer 2: inner E/W routers on 3-hop rows). 4x2 split: MESH_X_SIZE 2, MESH_Y_SIZE 4: E/W legs
  are one hop, so by the VC2 agent's 2x2 argument VC1 is never forwarded straight on an E/W router there (hypothesis,
  code read; pass-through turns at the exit chip not checked) - the 4x2 split may not need the fix, the 4x4 does.
- Dry run (stubs): `vc1x44` / `vc1x42` steps refused without P4_OK=1; with it, 2 / 4 stub ranks run and the slot check
  passes.
