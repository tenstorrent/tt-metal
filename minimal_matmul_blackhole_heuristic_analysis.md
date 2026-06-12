# Why the auto block-sizer loses to swept-best-main on Blackhole — and how to fix it

Follow-up to `minimal_matmul_blackhole_results.md`. All 13 branch losses are **non-slicing** shapes where
the auto block sizer picks a worse block than main's best-swept block. This analyzes *why* and proposes a
BH-aware heuristic. Grid = 11×10, fp32 acc, bf16 io.

## Method
The auto path doesn't pin a config, so its block isn't in the profiler ATTRIBUTES. I (a) re-implemented the
factory's exact derivation in Python to recover the auto-chosen block, and (b) mined the baseline sweep
CSVs — which already measured *every* candidate block on-device — to get the full time-vs-block landscape.
So every number below is real silicon, not a model.

## Auto-chosen vs main-best (the losses)

| shape | main best | auto chose | Mpc/Npc | speedup |
|---|---|---|---|--:|
| 4096x6144x2304 | 6/4/4 | **10**/8/8 | 12/8 | 0.74x |
| 2048x6144x4608 | 7/8/7 | 7/8/**12** | 7/14 | 0.83x |
| 1024x2304x6144 | 4/4/9 | 4/8/**18** | 4/18 | 0.89x |
| 4864x4096x1024 | 7/4/4 | **14**/8/4 | 14/4 | 0.94x |
| 4864x2048x1024 | 7/4/4 | **14**/8/4 | 14/4 | 0.92x |
| 4096x6144x4608 | 7/4/14 | 8/8/**10** | 13/14 | 0.93x |
| 16384x6144x768 | 6/16/3 | **21**/8/3 | 47/3 | 0.93x |
| 2048x4608x6144 | 4/8/9 | 7/8/**12** | 7/18 | 0.94x |

(bold = the dim where auto diverges hardest from main)

## Root cause 1 — `choose()` is floored at 8 and only grows ⇒ uneven splits (the big one)

The block sizer clamps M/N block to `min(8, per_core)`, then `choose()` only searches blocks **larger than
8** toward *fewest blocks within the L1 budget*. Two failure modes follow on BH:

- **Best even divisor is < 8 and thus unreachable.** `2048x6144x4608` has Npc=14; the clean split is
  N=7 (14=7+7). But 7 < 8, so `choose` never considers it and instead grows to N=12 (14=12+2, an ugly
  2-tile tail). Landscape: every fast block uses N=7 (561us); auto's N=12 path is the loss.
- **Growing creates an uneven tail.** `4096x6144x2304` has Mpc=12. The L1 budget (sized for WH) rejects
  the clean M=12 (one block) at K=8, so `choose` falls to **M=10 (12=10+2)**. Measured cost of that one
  uneven dim: **10/8/8 = 716us vs even 12/8/8 = 548us = +31%.** Main's 6/4/4 (12=6+6) = 530us.

**Among blocks that evenly divide the per-core count, the runtime is nearly flat** (e.g. 4096x6144x2304:
6/12 × K4/8 × N4/8 all land 530–554us). The penalty is almost entirely the *tail*, not the block size or
K. The current "fewest blocks / biggest block that fits L1" objective — which won on WH — is the wrong
objective on BH: it trades a tiny reuse gain for a costly pipeline imbalance.

**Why WH didn't hit this:** on the square 8×8 grid, per-core tile counts land on clean multiples of 8 far
more often (M/N parallelized over 8 cores), so `choose` either early-returns 8 (`per_core % 8 == 0`) or
grows evenly. BH's 11×10 grid produces per-core counts like 12, 13, 14, 18, 47 — exactly the values 8
fragments. The bug was always there; the non-square grid exposes it.

## Root cause 2 — K_block hardcoded to 8 (secondary, but compounding)

The auto path fixes `K_block = min(8, K_tiles)` (only dropping to 4 *under slicing*, line ~296). Main's
best-swept K on these shapes is **4** in the majority of cases (and 16 once, for the Npc=3 skinny-N
`16384x6144x768`). Direct effect is small (~1–2% per shape: 4096x6144x2304 6/4/4=530 vs 6/8/4=536), but
K=8 **doubles the in0/in1/intermediate L1 footprint**, which is what pushes `choose` over the budget and
forces the uneven M/N growth in cause 1. Finer K both pipelines slightly better in BH's many-core /
small-per-core regime (same reason the slicing path already uses K=4) and *frees the L1 that lets the
clean even block fit*. The two causes are coupled.

## Why the philosophy inverts WH→BH
WH 8×8 = 64 cores ⇒ large per-core M/N ⇒ win comes from **reuse** (big blocks, K=8, fewest blocks). BH
11×10 = 110 cores ⇒ small per-core M/N ⇒ win comes from **pipelining** (even splits, finer K, more
smaller blocks). The auto sizer encodes the WH philosophy. This is exactly the per-arch difference
anticipated.

## Proposed heuristic changes (BH-aware)

1. **Rewrite the M/N block objective to minimize the tail, searching the full divisor range (not floored
   at 8).** For each dim, among subblock-multiple blocks `≤ per_core` that fit L1, rank by
   `(tail = per_core mod block ascending, then block_count ascending, then larger block)`. This makes
   `choose` reach the sub-8 even divisors (7, 6, …) it currently can't, and prefer 12=6+6 over 12=10+2.
   - Strictly non-worse where `per_core % 8 == 0` (still returns 8), so low WH-regression risk, but
     **re-sweep WH to confirm** (square grid rarely exercises the new branch, but verify).
   - Recovers the bulk of the losses on its own: the landscapes show an even-dividing block is at/near the
     optimum for every loss shape.

2. **Default K_block to 4 on BH (conditionally), generalizing the existing slicing-only refinement.**
   Gate on a small per-core compute footprint rather than `num_slices>1` — e.g. when
   `M_tiles_per_core * N_tiles_per_core` is below a threshold (the bulk-square winners like
   `8192x6144x4608`, Mpc·Npc=360, K=8 keep K=8; the losers at 56–182 move to K=4). Needs a short K∈{4,8,16}
   confirm-sweep to set the threshold; K=4 is the safe majority default for mid shapes. Apply on the
   *non-sliced* auto path too, not just sliced.

3. **Make `L1_CB_BUDGET` arch-aware.** It is hardcoded to ~1.25 MiB (WH). BH L1 is larger, so the budget
   should rise — BUT note (1)+(2) mean we mostly want *smaller* even blocks, so a bigger budget is not the
   fix by itself; it just removes an artificial constraint. Lower priority than 1 and 2.

## Suggested validation
Implement (1), rebuild, re-sweep big+ltx. Expected: the 13 losses collapse toward 1.0x (the even-split
block is on-silicon within ~1% of main best for every loss shape mined). Then layer (2) and re-sweep to
pick up the residual K and the few near-misses. Re-run the WH sweep to confirm (1) is neutral there.

## Outcome (implemented + re-swept on BH)

Implemented both changes in `minimal_matmul_program_factory.cpp`, arch-gated to Blackhole:
- **Change 1** — even-split block chooser, gated `is_blackhole && num_slices == 1`. Searches all
  subblock-multiple blocks in `[max(sb,3) .. per_core]` and ranks by `(waste, |b-8|, fewest-blocks)`.
  Waste-primary (not block-count) so a clean even split is never traded for an uneven one; the `b>=3`
  floor + L1 cap keep it prime-safe (47 -> 8, never the degenerate 47x1). Where `per_core % 8 == 0`
  returns 8 (matches WH). Sliced shapes keep the old chooser (they were already winning).
- **Change 2** — K_block=4 for non-sliced BH shapes with small per-core compute
  (`min(per_core)>2 && Mpc*Npc<=128`). The `min>2` guard excludes prefetch-gated shapes (mcast path
  prefers K=8; forcing K=4 there regressed 32x2048x32 1.48x->0.98x).

Result, 82 shapes, branch (auto) vs best-swept-block main (baseline unchanged — pinned-config path is
not affected by these `!config` changes, so only the branch side was re-run):

| heuristic | geomean | wins >1.05 | within ±5% | losses <0.95 |
|---|--:|--:|--:|--:|
| original branch | 1.150x | 38 | 30 | 14 |
| **refined (this change)** | **1.163x** | 38 | **39** | **5** |

(big 1.145->1.156, ltx 1.170->1.189.) The targeted loss cluster collapsed: 4096x6144x2304 0.74->0.97,
2048x6144x4608 0.83->1.00, 4096x6144x4608 0.93->1.02, 4224x6144x4608 0.94->1.03. The first refinement
pass over-reached (block-count-primary picked uneven blocks; K=4 hit gated shapes; the new chooser
perturbed sliced winners) — all three fixed by the gating + waste-primary ordering above.

The 5 remaining losses are not block-heuristic issues: `512x128x1536` 0.75 (9us overhead-bound, lost on
WH too), `512x6144x1536` 0.84 & `1216x4096x512` 0.92 (**sliced** — left untouched), `8192x128x1536` 0.94
(K=128, 4 tiles), `1024x6144x768` 0.93 (lone minor change-2 K=4 regression). Next lever if pursued: a
smarter K rule (4/8/16) and a BH-tuned sliced sub-grid chooser. Per-shape tables:
`minimal_matmul_blackhole_results_v3_{big,ltx}.md`.

## Follow-up: smarter K rule (and a proxy that lied)

Mined per-K best times from the baseline sweep (it swept K∈{4,8,16,32}) for all 82 shapes. Pattern:
non-sliced shapes with `min(per_core) >= 4` prefer K=4; `min(per_core) <= 3` (skinny, forwarding-bound)
prefer K=8. A proxy (assume the branch achieves the baseline's best block at each K) said dropping the
`Mpc*Npc<=128` area cap would also win K=4 on LARGE per-core shapes (scored 0.9986 of oracle vs 0.987).

**It regressed them on silicon** (16384x2304x6144 1.04x->0.92x, 8192x6144x4608 1.07x->0.99x). The proxy
was wrong: for large per-core the auto chooser's K=4 *blocking* is far worse than its K=8 blocking, so
forcing K=4 broke shapes the branch was already winning at K=8. **The area cap is load-bearing** — it
confines K=4 to the small/mid regime where the chooser tracks the optimum. Lesson: only trust the
per-K-best proxy where the auto chooser actually reaches that block (small search space).

Final K rule (shipped): `min(per_core) >= 4 && Mpc*Npc <= 128 -> K=4`. Raising the floor 2->4 (vs the
original change-2) routes the `min==3` shapes back to K=8, fixing `1024x6144x768` 0.93x->0.99x and
`4096x6144x768` 0.97x->1.00x while large shapes keep their K=8 wins. Net 82-shape: geomean 1.163x->1.164x,
losses 5->4. Remaining losses are all structural (512x128x1536 overhead-bound, 8192x128x1536 K=128, and
the two sliced shapes left for the deferred BH sliced-chooser).

## One-line summary
On Blackhole the auto sizer loses because it inherits Wormhole's "big-block / fewest-blocks / K=8" reuse
philosophy, but BH's larger grid → smaller per-core tiles → the win is in **even-dividing, smaller blocks
with finer K**. The single highest-value fix is letting the block chooser pick even divisors below 8
(it currently can't), which alone is worth up to the 31% seen on the worst shape.
