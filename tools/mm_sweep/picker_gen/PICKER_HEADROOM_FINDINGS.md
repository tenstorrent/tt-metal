# Picker headroom: 12 confirmed mis-picks — APPLIED to kTable 2026-07-31

STATUS: all 12 are now `kTable` entries (7 updates to stale rows + 5 new keys). Verified that the picker
selects each one and delivers -3.9% to -19.4% vs the previous pick, PCC >= 0.99998. Suites after the
change: 111 correctness + 32 audit + 10 golden perf all pass. The `256x15360x768` golden threshold was
re-baselined 86.86 -> 81.30 us (its pick moved to 5,1,2,4,3 / 80 cores) so it still bounds the
implementation tightly.

Steps 2 and 3 below (fix `kNbandMax`, re-validate the remaining table rows) are still OPEN.

Exhaustive config sweeps on HEAD @ 7bdab431417 (BH p150b), driven by worst-utilisation ranking.
**3544 configs measured across 25 shapes.** Every win below re-confirmed with TWO fresh relaunches at
24 timed iterations; only deltas consistent at >1.5% on BOTH relaunches are listed.

| shape | Mt | deployed | NEW best | gain | %peak before | group |
|---|---|---|---|---|---|---|
| `512x6144x768` | 16 | `12,1,1,2,1` | **`6,1,2,2,3`** | **-19.5%** | 52% | Mt16 |
| `256x2048x2048` | 8 | `4,1,3,2,4` | **`4,1,2,2,8`** | **-14.1%** | 59% | shallowK |
| `256x2048x1536` | 8 | `4,1,3,2,3` | **`4,1,2,2,6`** | **-12.9%** | 57% | shallowK |
| `128x2048x1536` | 4 | `4,3,1,2,1` | **`4,2,1,2,3`** | **-12.6%** | 73% | shallowK |
| `512x3072x6144` | 16 | `6,2,1,2,1` | **`6,1,2,2,6`** | **-12.5%** | 62% | Mt16 |
| `512x6144x1536` | 16 | `12,1,1,2,1` | **`6,1,2,2,3`** | **-11.1%** | 60% | Mt16 |
| `256x2048x512` | 8 | `4,1,3,2,2` | **`4,1,2,2,2`** | **-10.3%** | 49% | shallowK |
| `512x2304x6144` | 16 | `3,4,1,1,1` | **`3,2,2,1,3`** | **-7.2%** | 61% | Mt16 |
| `64x2048x512` | 2 | `4,2,1,2,1` | **`4,1,1,2,2`** | **-6.0%** | 59% | shallowK |
| `256x15360x768` | 8 | `6,1,2,2,3` | **`5,1,2,4,3`** | **-5.3%** | 71% | deepK |
| `256x6144x1536` | 8 | `6,1,2,4,2` | **`12,1,1,2,1`** | **-5.0%** | 73% | deepK |
| `512x4096x5120` | 16 | `4,3,1,2,1` | **`4,3,1,4,1`** | **-4.8%** | 67% | Mt16 |

Median gain **10.7%**, range 4.8-19.5%. All PCC >= 0.99999 (scheduling changes only; bf16/HiFi2/fp32-acc
numerics untouched).

## Coverage: what was swept and found ALREADY OPTIMAL (do not re-sweep)

Mt=16 set (959 cfgs): 512x5120x2560 (138), 512x5120x5120 (152), 512x6144x2304 (66), 512x6144x4608 (101).
Worst-util set (2153 cfgs): 128x2048x512 (89), 256x2048x1024 (192), 32x2048x512 (23), 128x2048x1024 (159),
256x6144x768 (120), 64x2048x1024 (90), 32x256x512 (1 -- only one config exists at Kt=8).
Marginal / not confirmed: 128x6144x768 (-2.0%/-1.1%, washed out).

Cannot be fixed by configuration: 512x15360x768 -- only 5 feasible configs (Kt=480 in0-resident CB
dominates L1); stuck at ~54% of peak. Needs kernel work.

## Three INDEPENDENT picker weaknesses (they do not share one fix)

1. **M-split never considered outside narrow N.** `auto_select_config` bails with
   `if (Nband > kNbandMax || Mt < 2u) return anchor;` and `kNbandMax = 2`, so `Sm>1` is unreachable for
   `Nband>=3`. Cause of the four largest Mt=16 wins.
2. **Over-parallelisation on read-bound shapes.** The Mt=8/Kt=64 family picks `Sm=3`/96 cores where
   `Sm=2`/64 cores is 10-14% faster: extra cores add DRAM contention without adding bandwidth. Also
   `nsb` too narrow -- and `nsb=1` additionally BLOCKS reduce-scatter (its gate needs `N_sub>=2`).
3. **`kb` under-selection.** 512x4096x5120 (kb 2->4) and 256x15360x768 (Pk6/kb2 -> Pk5/kb4): deeper
   K-blocks cut the number of K blocks on deep-K shapes.

## WARNING: the errors are not one-directional

`256x6144x1536` goes the OPPOSITE way -- deployed `6,1,2,4,2` (M-split + reduce-scatter) loses 5% to
`12,1,1,2,1` (deep split-K + chain), which is the very pattern that was WRONG at Mt=16. A rule-based
patch generalised from the Mt=16 findings would have made this shape slower. Fixes must be
measurement-driven per shape.

## Stale lookup table

At least 5 of the 12 are `kTable` entries (measured winners from earlier campaigns) now beatable by
5-14%: 256x2048x{512,1536,2048}, 512x6144x1536, 128x2048x1536. The kernel stack moved under them
(reduce-scatter, in1-delivery, mesh placement). The table needs systematic re-validation, not just a
fallback fix.

## Suggested order

1. ~~Add all 12 to `kTable`~~ **DONE 2026-07-31.**
2. Fix `kNbandMax` / add a core-count-vs-read-bound term so the FALLBACK finds these structurally.
3. Re-validate every remaining `kTable` row on the current stack.

---

## Joint (reduction x placement) sweep — 2026-08-01. Gates are SOUND; mesh_gate "fix" REJECTED.

Reduction and placement are normally DERIVED from the config, so they were only ever toggled indirectly by
the config sweeps. Added temporary force hooks (since REMOVED) and measured the full 2x3 grid plus an
auto baseline on 15 shapes: the 12 newly-tabled winners + 3 confirmed-optimal controls.

**14 of 15 shapes: both gates already optimal.** Includes all 3 controls, so `512x6144x2304`,
`512x5120x5120` and `256x6144x768` are at their ceiling on config AND reduction AND placement -- their
61-70% of peak is a kernel limit, not a scheduling one.

**1 shape wants a different placement:** `512x2304x6144` (Pk3/Ns2/Sm2, Mt=16) is **-4.2%/-4.3%** on mesh
vs the in1-near the gate forces. All three clauses of the gate's first branch reject it: `Pk*Ns=6 < 10`,
`Sm != 1`, and `Ns != 1 && Pk < 4`. The second branch (`ring_bytes >= 2*in1_bytes`) also fails at Nt=192.

### The fix was implemented, measured, and REVERTED

Changed the "mesh fills the grid" test to PREADERS (`Pk*Ns*Sm >= 10`, which is what the comment always
claimed) and admitted `Sm>1` through it, structured so the `Sm==1` path stayed byte-identical. Corpus
re-run of every `Sm>1` shape it newly admitted:

| shape | Mt | mesh | in1-near | delta |
|---|---|---|---|---|
| 256x6144x4608 | 8 | 177.8 | 141.6 | **+25.6% REGRESSION** |
| 256x6144x6144 | 8 | 233.7 | 186.8 | **+25.1% REGRESSION** |
| 256x15360x1536 | 8 | 155.3 | 136.2 | **+14.1% REGRESSION** |
| 512x2304x6144 | 16 | 126.1 | 131.6 | -4.2% (intended win) |
| 512x3072x6144 | 16 | 136.6 | 137.8 | -1.0% neutral |

Net harmful: +4.2% on one shape against -14 to -26% on three. **The `Sm == 1` clause is LOAD-BEARING, not
a bug** -- it is what keeps Mt=8 M-split shapes on in1-near, where the slaves' short reader->slave hop
matters more than the mesh's ring saving. The original comment's "+8.7% to -22.2%" for `Sm>1` mesh was
accurate and if anything understated the downside. Reverted to the exact original expression; the three
regressors and the intended winner all measured back at baseline within 0.2%.

The winner/regressor split is Mt=16 vs Mt=8, but gating on `Mt >= 16` would be fitting a global heuristic
to n=1 in a space where being wrong costs 25%. Not worth it. If that 4.2% is wanted later, the right
mechanism is an explicit per-shape placement override (extend the lookup table to carry placement, which
it currently cannot express) -- not a loosened gate.

### Method note

Include a same-topology CONTROL cell in any such grid. On `512x4096x5120` the raw grid appeared to show
reduce-scatter beating chain by 3.2%, but that compared `rs+mesh` against a slow `auto` run; the
`chain+mesh` control cell -- the identical program to `auto` -- differed from it by 2.4%. Measured
like-for-like at fixed placement the reduction change was -1.0%/+0.1%, i.e. nothing. Without the control
cell that would have shipped as a finding.
