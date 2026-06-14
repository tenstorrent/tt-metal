# minimal_matmul WH sweep — underperformers vs same-FLOP family

Method: group the 2816-shape sweep by **tile-MACs** (`Mt·Nt·Kt`, ∝ FLOPs). Same FLOPs ⇒ same ideal
compute time, so within a family `util ∝ 1/time` and the underperformer is the shape that takes much
longer than the *fastest arrangement of the same work*. Two groupings used:
- **(A) same tile-MACs** — the literal "same FLOPs" ask.
- **(B) same `(out_tiles, Kt)`** — additionally holds output-tile count AND K-depth fixed, so compute
  *and* DRAM traffic are constant and the **only** variable is how `out_tiles` splits into `Mt×Nt`
  (aspect ratio + orientation). This isolates pure dataflow/heuristic inefficiency from the DRAM effect.

---

## Class A — shallow-K, huge-output shapes (DRAM / per-tile-overhead bound)

The worst "same-FLOP" offenders, up to **15× slower** than a same-tile-MAC sibling:

| shape | Mt×Nt×Kt | out tiles | Kt | µs | slow | same-FLOP best |
|---|---|---|---|---|---|---|
| 8192×32×6144 | 256·192·1 | 49152 | **1** | 922 | **15.2×** | 6144×4096×64 (192·2·128) 61µs |
| 8192×32×8192 | 256·256·1 | 65536 | **1** | 1163 | 14.2× | 8192×2048×128 81µs |
| 8192×64×6144 | 256·192·2 | 49152 | **2** | 935 | 8.8× | 8192×6144×64 (Kt192) 107µs |

**Hypothesis:** these are **DRAM-bandwidth / per-output-tile-overhead bound, not compute bound.** With
`Kt=1` the same MAC budget is spread over a *huge* number of output tiles (49152 vs the family-best's
384) — each tile still pays the full fixed cost (read an in0 row + in1 col, matmul setup, write the
output tile) but does only 1 K-tile of math. So runtime tracks `out_tiles` (data movement), not FLOPs.
`tile-MAC util` is misleading here because it treats a `Kt=1` tile-MAC as equal to a `Kt=128` one.
Corollary: K-par/slicing can't help (Kt=1 → nothing to split; output already over-fills the grid). The
real lever is **DRAM traffic** (narrower input dtype, dual-NoC output write), not occupancy.

---

## Class B — square / low-aspect shapes (poor reuse + heuristic under-serves them)

Controlling for `(out_tiles, Kt)` — same work, same DRAM — the **squarer** arrangement is consistently
**2–3.7× slower** than the **skinny** one. 91% of these underperformers are squarer than their best
same-work twin; the fastest arrangement almost always has one output dim ≈ 1–2 tiles.

| shape | Mt×Nt | out | Kt | (S,Pk) | µs | slow | best arrangement (same out,Kt) |
|---|---|---|---|---|---|---|---|
| 1024×6144×384 | 32×12 | 384 | 192 | S4,Pk2 | 304 | **3.7×** | 192×2 tiles, **S4,Pk2**, 83µs |
| 384×6144×1024 | 12×32 | 384 | 192 | S4,Pk2 | 264 | 3.2× | 192×2 tiles, 83µs |
| 192×6144×256 | 6×8 | 48 | 192 | S2,Pk4 | 73 | 3.1× | 48×1 tiles, 24µs |
| 256×8192×384 | 8×12 | 96 | 256 | **S1,Pk1** | 123 | 2.9× | 2×48 tiles, S2,Pk4, 43µs |

Two sub-causes (the data shows both):

**B1 — dataflow reuse favors skinny (the dominant effect).** The clearest case, `1024×6144×384`
(32×12) vs `6144×4096×64` (192×2): *identical* `out_tiles=384`, `Kt=192`, *and identical* `S4,Pk2` —
yet the square one is **3.7× slower**. Same work, same parallel config → the difference is pure
dataflow. minimal_matmul forwards the small input and reuses it across the big-dim cores; when one
output dim is tiny (Nt=2), the small input (in1) is fully reused across all M-cols (high arithmetic
intensity). For a square output neither input is small → far less reuse, more streaming per MAC. The op
was built/tuned for the skinny regime (LTX/FLUX: small batch × wide hidden) and is inefficient on square.

**B2 — the heuristic provides no help for square mid-size shapes (42% of class B).** N-slicing only
engages on *skew* and K-par only on *output-starvation*. A square, mid-size shape (e.g. `256×8192×384`,
8×12, out=96) is neither, so it gets `S1,Pk1` — no extra parallelism — while its skewed twin gets
`S2,Pk4`. So square shapes are doubly penalized: worse intrinsic reuse **and** no slicing/K-par.

---

## Joint (S, Pk, blocking) probe — is Class B a heuristic miss or intrinsic? (BOTH)

Swept blocking AND S/Pk *together* (the factory now allows explicit-env K-par with a pinned config) on
3 square underperformers + their skinny same-(out,Kt) twins. Best-achievable vs the heuristic default:

| square shape | Mt×Nt | default | best | best cfg | recoverable | best vs skinny-twin best |
|---|---|---|---|---|---|---|
| 1024×6144×384 | 32×12 | 304µs | 101µs | **S1,Pk2** | 3.01× | 1.66× |
| 256×8192×384 | 8×12 | 123µs | 66µs | **S1,Pk2** m2n2k8 | 1.88× | 1.53× |
| 192×6144×256 | 6×8 | 73µs | 39µs | **S1,Pk4** | 1.89× | 1.64× |

- **~1.9–3.0× is a recoverable heuristic miss (B2).** Every best config is **S=1**. The heuristic picks
  high S (e.g. S4,Pk2) because of its `S·Pk = grid.y` rule — but on a square shape `S=4 ⇒
  rows_per_group=1`, which **kills the small-output-dim (N) row-parallelism** (each core then does all N
  tiles serially). Fix: don't force `S·Pk=grid.y` on low-aspect shapes; keep S low and spend the row
  budget on `rows_per_group` (parallelize the small dim) + modest K-par.
- **~1.5–1.66× residual is intrinsic dataflow (B1).** Even the best square config can't reach the skinny
  twin's time for the same work — the tall-skinny reuse advantage is real; closing it needs a dataflow
  change (2D output tiling), not scheduling.

## Takeaways
- The single biggest "low util per FLOP" bucket is **shallow-K large-output** (Class A) — a DRAM bound,
  not a scheduling miss; util-per-FLOP overstates it. Address via input dtype / output-write NoC, not K-par.
- The **fixable** underperformers are **square / low-aspect** shapes (Class B): 2–3.7× off the skinny
  arrangement of the same work. The reuse gap (B1) is a dataflow property; the scheduling gap (B2) —
  square mid-size shapes getting `S1,Pk1` — is a heuristic gap that a square-aware slicing rule could
  close (e.g. 2D core-grid tiling of a square output, or engaging K-par on "grid-underfilled" rather
  than only "starved").
