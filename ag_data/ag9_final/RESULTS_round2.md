# Round 2 results: numeric knobs around the round-1 best (measurement only)

Same harness and build as round 1: commit 99932452e22, build_Release unchanged, BH 1x8 mesh, 2 links,
packet 15232, AG_ITERS=20, AG_TOPO=both, tracy profiling, TT_METAL_OPERATION_TIMEOUT_SECONDS=45. Each
run is analysed within its own report with `ag_analyze.py <report> --vs-default --vs <ref> --min-delta 5 --csv`.

| run | knob file | shapes | cells | pytest | report dir | analysis files |
|---|---|---|---|---|---|---|
| R2a | round2_unicast.txt | 512K,5M,25M,p2K,p8K | 90 | **90 passed** (11:49) | `generated/profiler/reports/2026_10_16_00_58_15` | r2a.txt/.csv, r2a_vs.txt/.csv, tab_r2a.md, log r2a.log |
| R2b | round2_multicast.txt | 131K,512K,2M,p2K,p8K | 90 | **90 passed** (10:12) | `generated/profiler/reports/2026_10_16_01_12_19` | r2b.txt/.csv, r2b_vs.txt/.csv, tab_r2b.md, r2b_runcap0_compare.txt, r2b_kauto_vs_cap0_compare.txt, log r2b.log |

- Nothing TT_FATALed, so no line was dropped. Every config has n=20 on both sides of every A/B.
- **The reference string differs from the brief.** The analyzer prints `entry=one_packet` as
  `entry=scatter_aware`, so R2a's `--vs` had to be `factory=unicast,k=auto,entry=scatter_aware`.
  `--vs "factory=unicast,k=auto,entry=one_packet"` matches nothing and prints "nothing to compare".
- **One R2b trap: the analyzer hides `run_cap=0`.** `compact_tuning` treats `"0"` as a default value and
  drops it, so two lines get the wrong label:
  - `factory=multicast,k=auto,run_cap=0` is labelled `factory=multicast,k=auto`.
  - `factory=multicast,run_cap=0` is labelled `factory=multicast`, the same as the reference.
  - The analyzer therefore warned "10 configs in non-contiguous blocks" and named the `run_cap=0` cells
    `...#2`. `--vs` silently skips those cells, because their key has no reference.
  - I checked the raw ATTRIBUTES to confirm that block 1 is `run_cap_bytes=std::nullopt` and block 2 is
    `run_cap_bytes=0`. Run order follows the knob file.
  - To get those numbers, I split the raw ops CSV into per-tuning copies and ran
    `ag_analyze.py --compare A B --min-delta 5`. Same report, same verdict rule. The copies are in
    `scratchpad/r2b_split/`, and the results are in `r2b_runcap0_compare.txt` and
    `r2b_kauto_vs_cap0_compare.txt`.
  - Analyzer fix for later, not made here: don't drop `run_cap=0` when it is explicit (optional set to 0).
- Verdict rule as in round 1: **faster/slower** means |delta| >= 5 % and the bootstrap CIs don't overlap,
  and `·` means noise. For 131K and 512K, a non-noise result with |delta| < 10 % is downgraded to
  inconcl. The metric is the median over 20 iterations of the max over 8 devices.

## R2a unicast (reference `factory=unicast,k=auto,entry=one_packet`, i.e. the heuristic defaults for workers/signals/run_cap)

What the reference resolves to, read off all_gather_unicast_factory.cpp:184-261 for BH, 8 devices,
2 links, with `per_link_bytes = output / 2`:
- **Ring** keys on total output.
  - 512K (1 MB) gives **2**.
  - 5M (10.5 MB, stripe 20, long) gives **3**.
  - 25M (50 MB) gives **4**.
  - p2K and p8K (16,777,216 B = exactly 16 MiB, so `>= 16 MiB`) give **4**, assuming their stripe counts
    as long. Each of these matched its explicit workers=N row within noise.
- **Line**: per-link bytes are >= 640 KB / 8 for every shape here, so every line shape gets **2**.
- Defaults are signals = 2, and run_cap = 0 (cap only at >= 4 links).
**ring** (delta% vs `factory=unicast,k=auto,entry=one_packet`)

| tuning | 512K | 5M | 25M | p2K | p8K |
|---|---|---|---|---|---|
| *ref median us* | 27.5 | 93.7 | 376.9 | 142.3 | 142.4 |
| `run_cap=4096` | +21.5 **slower** | +109.0 **slower** | +129.0 **slower** | +111.6 **slower** | +30.2 **slower** |
| `run_cap=8192` | +2.4 · | +31.1 **slower** | +37.4 **slower** | +30.0 **slower** | +30.2 **slower** |
| `signals=1` | +8.7 inconcl. | +0.6 · | +0.0 · | -1.8 · | -0.4 · |
| `signals=4` | +11.4 **slower** | +7.7 **slower** | +2.6 · | +7.4 **slower** | +5.3 **slower** |
| `workers=1` | +19.8 **slower** | +93.8 **slower** | +130.6 **slower** | +82.4 **slower** | +83.1 **slower** |
| `workers=2` | +1.3 · | +4.2 · | +6.5 **slower** | -0.5 · | +0.0 · |
| `workers=3` | +10.8 **slower** | +0.2 · | +1.7 · | -2.9 · | -2.8 · |
| `workers=4` | +13.3 **slower** | +7.4 **slower** | +0.6 · | +1.4 · | +0.6 · |

**linear** (delta% vs `factory=unicast,k=auto,entry=one_packet`)

| tuning | 512K | 5M | 25M | p2K | p8K |
|---|---|---|---|---|---|
| *ref median us* | 43.0 | 175.1 | 745.0 | 252.2 | 251.8 |
| `run_cap=4096` | +24.7 **slower** | +126.0 **slower** | +144.9 **slower** | +144.1 **slower** | +43.7 **slower** |
| `run_cap=8192` | -1.1 · | +35.4 **slower** | +42.8 **slower** | +43.6 **slower** | +43.8 **slower** |
| `signals=1` | +19.7 **slower** | +13.1 **slower** | +8.0 **slower** | +10.8 **slower** | +11.2 **slower** |
| `signals=4` | +26.3 **slower** | +1.6 · | -0.1 · | +0.5 · | +1.0 · |
| `workers=1` | +5.4 inconcl. | +50.1 **slower** | +61.2 **slower** | +54.5 **slower** | +56.1 **slower** |
| `workers=2` | -1.7 · | +0.3 · | -0.1 · | -0.0 · | +0.0 · |
| `workers=3` | +11.4 **slower** | +3.8 · | +1.2 · | +3.4 · | +3.9 · |
| `workers=4` | +14.5 **slower** | +11.1 **slower** | +0.4 · | +7.4 **slower** | +7.8 **slower** |


**per-knob summary** (10 configs each; geomean_sig over non-noise rows, geomean_all over all 10)

| tuning | faster | slower | noise | geomean_sig% | geomean_all% | worst | best |
|---|---|---|---|---|---|---|---|
| `run_cap=4096` | 0 | 10 | 0 | +81.5 | +81.5 | +144.9 | +21.5 |
| `run_cap=8192` | 0 | 8 | 2 | +36.7 | +28.6 | +43.8 | -1.1 |
| `signals=1` | 0 | 6 | 4 | +11.9 | +6.8 | +19.7 | -1.8 |
| `signals=4` | 0 | 5 | 5 | +11.4 | +6.1 | +26.3 | -0.1 |
| `workers=1` | 0 | 10 | 0 | +60.0 | +60.0 | +130.6 | +5.4 |
| `workers=2` | 0 | 1 | 9 | +6.5 | +1.0 | +6.5 | -1.7 |
| `workers=3` | 0 | 2 | 8 | +11.1 | +3.0 | +11.4 | -2.9 |
| `workers=4` | 0 | 6 | 4 | +10.2 | +6.3 | +14.5 | +0.4 |

## R2b multicast (reference `factory=multicast`: BH defaults cb_depth 3, run_cap 8192 for page >= 1 KB, entry legacy, k=1)

Note on labels: the row labelled `k=auto` below is really **`k=auto,run_cap=0`** (see the trap above).
The `run_cap=0` row comes from `--compare`, not `--vs`; I spliced it in by hand with the same verdict rule.

**ring** (delta% vs `factory=multicast`)

| tuning | 131K | 512K | 2M | p2K | p8K |
|---|---|---|---|---|---|
| *ref median us* | 19.0 | 27.2 | 62.8 | 183.2 | 181.5 |
| `entry=one_packet,mcast_cb_depth=12` | +1.7 · | -0.3 · | -2.2 · | -2.3 · | -1.2 · |
| `entry=one_packet,mcast_cb_depth=6` | +7.7 inconcl. | -0.1 · | -2.8 · | -1.5 · | -1.4 · |
| `entry=one_packet,mcast_cb_depth=8` | +6.3 inconcl. | +1.7 · | -1.5 · | -2.6 · | -1.6 · |
| `k=auto,run_cap=0` | +5.8 inconcl. | +3.7 · | +0.2 · | -3.3 · | -2.8 · |
| `run_cap=0` (via --compare) | +1.5 · | +15.7 **slower**† | -1.1 · | -4.0 · | +0.1 · |
| `mcast_cb_depth=2` | +5.7 inconcl. | +1.8 · | -2.3 · | -0.3 · | -0.3 · |
| `mcast_cb_depth=4` | +2.5 · | +2.6 · | -2.1 · | -0.4 · | -0.3 · |
| `mcast_cb_depth=6` | +5.4 inconcl. | +5.7 inconcl. | -2.0 · | +0.6 · | -0.3 · |

**linear** (delta% vs `factory=multicast`)

| tuning | 131K | 512K | 2M | p2K | p8K |
|---|---|---|---|---|---|
| *ref median us* | 24.5 | 42.7 | 108.9 | 275.3 | 346.4 |
| `entry=one_packet,mcast_cb_depth=12` | -2.6 · | -4.0 · | -3.7 · | +25.5 **slower** | +0.3 · |
| `entry=one_packet,mcast_cb_depth=6` | -4.2 · | -1.7 · | +0.1 · | +25.9 **slower** | -0.2 · |
| `entry=one_packet,mcast_cb_depth=8` | -0.9 · | -4.7 · | -3.7 · | +25.4 **slower** | -0.1 · |
| `k=auto,run_cap=0` | +1.0 · | -1.6 · | -0.5 · | +2.1 · | -19.2 **faster** |
| `run_cap=0` (via --compare) | +2.4 · | -0.5 · | +0.2 · | +1.5 · | -0.2 · |
| `mcast_cb_depth=2` | +0.1 · | +0.1 · | -1.2 · | -0.7 · | -0.4 · |
| `mcast_cb_depth=4` | +0.3 · | +0.5 · | -0.7 · | -0.4 · | -0.3 · |
| `mcast_cb_depth=6` | -2.4 · | +1.0 · | -1.4 · | -0.6 · | -0.0 · |


**per-knob summary** (10 configs each; the run_cap=0 row is excluded from this summary because the analyzer could not see it)

| tuning | faster | slower | noise | geomean_sig% | geomean_all% | worst | best |
|---|---|---|---|---|---|---|---|
| `entry=one_packet,mcast_cb_depth=12` | 0 | 1 | 9 | +25.5 | +0.8 | +25.5 | -4.0 |
| `entry=one_packet,mcast_cb_depth=6` | 0 | 2 | 8 | +16.4 | +1.9 | +25.9 | -4.2 |
| `entry=one_packet,mcast_cb_depth=8` | 0 | 2 | 8 | +15.5 | +1.5 | +25.4 | -4.7 |
| `k=auto,run_cap=0` | 1 | 1 | 8 | -7.5 | -1.7 | +5.8 | -19.2 |
| `mcast_cb_depth=2` | 0 | 1 | 9 | +5.7 | +0.2 | +5.7 | -2.3 |
| `mcast_cb_depth=4` | 0 | 0 | 10 | +nan | +0.2 | +2.6 | -2.1 |
| `mcast_cb_depth=6` | 0 | 2 | 8 | +5.6 | +0.6 | +5.7 | -2.4 |

† 512K ring `run_cap=0` at +15.7 % clears the 10 % small-shape bar, but I read it as noise:
- `k=auto,run_cap=0` runs the same cap on the same shape and lands at +3.7 % (28.2 us vs 27.2 us).
- Two of the nine 512K-ring cells in this report have 37-41 % spread.

Extra A/B, isolating k with the cap off (`k=auto,run_cap=0` vs `run_cap=0`, from
r2b_kauto_vs_cap0_compare.txt):
- linear p8K: **-19.0 % faster** (345.6 to 279.9 us).
- 512K ring: -10.4 %, which is the mirror of the 15.7 % artefact above.
- Everything else is noise, including ring p8K at -3.0 %.

## Reading

### (1) Unicast

**workers_per_dir: the current BH heuristic picks the best value in every cell measured.** No explicit
workers=N is faster than the default anywhere. The table below gives each explicit N's delta from the
heuristic's own choice:

| topology | size class (shape) | heuristic | best measured | runner-up | others |
|---|---|---|---|---|---|
| ring | 1 MB total (512K, stripe 8) | 2 | **2** (=ref, +1.3 · for explicit 2) | — | 1: +19.8, 3: +10.8, 4: +13.3 (all slower) |
| ring | 10.5 MB (5M, stripe 20) | 3 | **3** (+0.2 ·) | 2: +4.2 (CI-separated, <5 %) | 4: +7.4 slower, 1: +94 |
| ring | 16 MiB RM (p2K, p8K) | 4 | **3 or 4**: 3 is -2.9 / -2.8 % vs 4, with non-overlapping CIs but under 5 % | 2: -0.5 / 0.0 | 1: +82 |
| ring | 50 MB (25M, stripe 32) | 4 | **4** (+0.6 ·) | 3: +1.7 · | 2: +6.5 slower, 1: +131 |
| line | 1 MB (512K) | 2 | **2** (-1.7 ·) | — | 3: +11.4, 4: +14.5, 1: +5.4 inconcl. |
| line | 10.5-50 MB (5M, 25M, p2K, p8K) | 2 | **2** (all within ±0.3 %) | 3: +1.2 to +3.9 (CI-separated at 5M, p2K, p8K) | 4: +0.4 to +11.1, 1: +50 to +61 |

- Ring is monotone in total bytes, as the tiers say: 2, then 3 at 10 MB, then 4 at 50 MB.
- The only soft spot is the ring 3/4 boundary at 16 MiB.
  - The RM shapes sit exactly on `>= 16 MiB`, so they get 4. Three is about 2.9 % faster there,
    consistently on both page sizes, with CIs that don't overlap.
  - That gap is below the 5 % bar, so I don't call it a result. It hints that the 3-to-4 step belongs
    somewhat above 16 MiB (between 16.8 and 50 MB).
- Line agrees with the code comment: 3 workers loses about 1-4 % from 5 MB up, and 4 workers loses more.

**signals_per_stripe: keep 2. Both alternatives only lose; nothing is faster.**
- signals=1 costs **+8 to +20 % on every line shape**. It is noise on ring, except 512K at +8.7 %
  (inconclusive).
- signals=4 costs **+5 to +11 % on ring** (512K, 5M, p2K, p8K) and +26 % on line 512K. It is noise on
  large line shapes.
- One-packet entries do not change the optimum; the "exactly 2" comment at unicast_factory.cpp:381 still
  holds.

**run_cap: any cap is a large loss at 2 links. Keep 0 (as the code does).**
- run_cap=4096: +21 to +145 %, and slower in every cell.
- run_cap=8192: +30 to +44 % on every shape >= 5 MB, and noise only at 512K.
- **The 8192 cap undoes k=auto on RM.**
  - With the cap, p8K lands at 362.1 us on line and 185.4 us on ring. Round 1's k=1 p8K baseline was
    362.9 / 185.0 us.
  - p2K with the cap lands at the same 362 / 185 us. The cap limits the run to 8 KB, which costs packet fill
    whatever the page size.
- So with one-packet entries, run_cap does not help on 2 links. The `num_links >= 4` gate is the only part
  of that rule this round can't speak to.

### (2) Multicast

**k=auto with run_cap=0: yes for linear p8K, at -19.2 % (346.4 to 279.9 us).**
- It is -19.0 % against `run_cap=0` alone, so the gain is k's, not the cap's.
- Ring p8K: -2.8 % vs the reference and -3.0 % vs cap 0. That is noise, though it leans the right way.
- p2K, the tile shapes and every ring shape: noise.
- This confirms round 1's suspicion: k=auto was a no-op on multicast only because the default
  8192 B cap forces k=1 for pages >= 1 KB.
- **p4K was not in R2b's shape list** (131K, 512K, 2M, p2K, p8K as briefed), so p4K is unmeasured. By
  analogy with unicast (p4K: -6 to -9 %), expect a smaller gain there. That is an expectation, not data.
- Even after the gain, multicast linear p8K (279.9 us) is about 11 % behind unicast k=auto linear p8K
  (251.8 us, from R2a). This is cross-report, so allow a few % of drift.

**run_cap=0 alone: changes nothing.**
- All ten cells are within ±4 %, except the 512K ring artefact above.
- Ring p2K is -4.0 % with CIs that don't overlap, but it is under the bar.
- So the 8192 cap neither helps nor hurts multicast by itself. Its only measured effect is blocking k.

**one_packet with a deeper ring (depth 6, 8, 12): not neutral on linear. The p2K regression is unchanged
by depth.**
- Linear p2K: +25.9 / +25.4 / +25.5 % at depth 6 / 8 / 12, against +26.4 % in round 1 at depth 3.
  - Every depth lands at **345-347 us**. That equals the multicast linear p8K baseline (346.4 us) and the
    multicast one_packet p2K/p4K plateau from round 1.
  - So the slowdown is a fixed per-transfer cadence, not trid-ring starvation. Depth does not touch it.
- Linear p8K: neutral, as in round 1.
- Short-stripe tile shapes on linear (512K, 2M) lean slightly positive at depth 8 and 12: -4.0 / -4.7 %
  at 512K and -3.7 / -3.7 % at 2M, with CIs that don't overlap. That is under 5 %, so noise by the rule.
  Depth 6 shows no lean.
- Ring: all noise (-2.8 to +1.7 %). 131K ring at depth 6 and 8 reads +7.7 / +6.3 %, which is
  inconclusive.
- 5M and 25M linear, where round 1 regressed by +10.7 / +28.0 %, were not in R2b's shape list. So this
  round can't say whether depth fixes those. Given p2K, it is unlikely.
- Verdict unchanged: keep one_packet off on multicast.

**Plain cb_depth: depth does not matter. The default 3 is fine; 2 is the cheapest at equal speed.**
- Every row with depth 2, 4 or 6 is within ±2.7 % on all shapes from 2M up and on RM, on both topologies.
- The only flags are at 131K and 512K on ring (+5.4 to +5.7 %), all inconclusive under the small-shape bar.
- No depth beats 3. If L1 matters, 2 costs nothing measurable. That matches the code comment that cb_depth
  "was already at its optimum".

### (3) Surprises
- `--vs "factory=unicast,k=auto,entry=one_packet"` silently matches nothing, because of the
  `scatter_aware` relabel. `run_cap=0` vanishes from labels entirely, so two different tunings merge
  under one label and only a "non-contiguous blocks" WARNING hints at it. Both are analyzer label issues
  that could mislead later rounds. Round 3 should avoid explicit `run_cap=0` lines, or fix
  `compact_tuning`.
- The unicast RM shapes collapse to one time, which is the k=auto effect working as intended.
  - With k=auto, ring is 142.3 / 142.4 us and line is 252.2 / 251.8 us for p2K / p8K.
  - The 8192 cap makes them collapse again, but to the slower k=1 time.
- Multicast's fixed ~346 us linear plateau now has three entry points: p8K by default, p2K/p4K with
  one_packet at any depth, and every RM shape under an 8 KB cap. It looks like one transfer-size ceiling
  on the linear multicast path. Worth a code look before round 3's crossover; I have not investigated it.
- The 512K ring multicast cells are bimodal in places: spreads of 24-41 % in 3 of 9 tunings. Any 512K
  ring multicast decision needs more iterations or repeats.
