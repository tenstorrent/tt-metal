# in0 ring delivery — cost model, device geometry, and candidate optimizations

Scope: `ttnn.experimental.regime_a_matmul` in0 delivery (ring all-gather) on the four golden configs.
Evidence base: `ABLATION_MATRIX.md` (22-mode critical-path matrix), `D1_RING_ATTRIBUTION.md` (distance vs
injection attribution), `DEDUP_BASELINE.md`, and the prior-art history in the project memory.

## 1. Device geometry (the fact that reframes the problem)

BH p150b, compute grid 11x10. Bank-adjacent workers, identical on both NoCs:

```
banks 0-3 -> logical (0,9) (0,0) (0,7) (0,3)   <- column x=0
banks 4-7 -> logical (6,9) (6,1) (6,6) (6,4)   <- column x=6
```

DRAM itself sits in two physical columns (x=0 and x=9 in the SOC descriptor). Consequence: the 8 cores of
every in0 ring are split 4/4 across a ~6-hop chip bisection, so **every ring cycle must cross that
bisection exactly twice**, and each crossing edge carries 7 shards.

## 2. Cost model (512x6144x2304, cfg Ns2/Pk6/Sm1/kb2/nsb1)

| quantity | value |
|---|---|
| shard (one ring hop payload) | W*M_block*kb = 2*16*2 = 64 tiles = **128 KB** |
| k-slice per ring == cb0 | 512 tiles = **1 MB** (of a 1440 KB L1 budget; total plan L1 ~1219 KB) |
| rings (Pk*Ns*Sm) x cores | 12 x 8 = 96 cores |
| ring NoC bytes | 12 rings * 8 edges * 7 shards * 128 KB = **86 MB** (in0 logical 6.3 MB -> **13.7x**) |
| crossing the x=0 <-> x=6 bisection | 24 crossing edges * 896 KB = **21.5 MB** on ~10 row-links |
| ideal 7-step serial forward | ~21 us; **measured exposure 52.7 us (2.5x)** |
| in0 own-shard DRAM read | 12 MB, theo 24.6 us, **measured 25.0 us -> 100% exposed** |
| in0 delivery total | **77.7 us of a 170 us wall (46%)**, additive with compute/output |

Two invariants worth keeping in mind:

- **in0 delivered bytes = `8 * Ns * |in0|`** — independent of Pk and Sm. Only ring size and Ns scale it.
- **Exchange rate: DRAM ~2.08 us/MB vs ring NoC ~0.62 us/MB** (from the two ablation gains). Any idea that
  buys NoC traffic with extra in0 DRAM reads is a guaranteed loss.

## 3. What D1 settled

The forward is **distance-bound**: same bytes at ~1 hop recovers 69-71% of the full forward cost; cost is
super-linear in bytes (queueing past the knee). The ~30% residual is the floor for topology-only fixes.
See `D1_RING_ATTRIBUTION.md`.

## 4. Candidate optimizations

### Host-only (no kernel change, correct for any permutation)

1. **~~Region-local ring re-partitioning (Ns>=2)~~ — MEASURED, REFUTED. See `IDEA1_RING_TOPOLOGY.md`.**
   Cost tracks PEAK LINK LOAD, not total hops: compact rings cut total hops 16% but raised the busiest link
   25% and measured -4..-9%. Compactness also *reduces* balancing headroom. Superseded by (2), which on the
   same shapes reaches +4.2% but is not yet shippable (the link model must cover in1/reduction/output too,
   and needs a hop budget for Sm>1).
   Original rationale, kept for context: ring membership is currently hardcoded as
   "8 banks x slice j", but the ring is purely an in0 concern: any 8 cores sharing the same `(kk, mm)` may
   form a ring, including cores with different `nn`. For Ns=2 the 16 cores of a `(kk,mm)` group split
   exactly 8 left / 8 right of the bisection -> build each ring inside one region. Bisection crossings 4
   per group -> 0; average edge ~3.6 -> ~2.0-2.5 hops. in0 DRAM read unchanged, bank adjacency untouched,
   in1 read untouched, kernel byte-identical. Ns=3 needs a balanced 8-clustering (12+12 doesn't divide by
   8) and still removes ~2/3 of crossings. Extra constraint: mm-siblings must keep equal `ring_pos` (the
   in1 forward order depends on it) -> build the partition for mm=0 and mirror it.
   Expected: -5% to -12% on the deep Ns>=2 shapes.
2. **Whole-op link-load-aware ring ordering — ✅ +6.1% on 512x6144x2304, neutral elsewhere (diag bit10).**
   See `RING_LINK_BALANCE.md`. Charging the FIXED traffic (in1 reads, in0 read, reduction, output) onto the
   link map first, plus worst-edge/hop budgets anchored on production and an adopt-only-if-better gate, turns
   bit9's mixed result into +6.1% (3 relaunches) with no regression on the four golden shapes. Recovers ~20%
   of the entire ring-forward cost from ordering alone. Needs the 60-shape corpus before promotion to mask 0.
   Superseded first attempt (bit9, in0-only, unbudgeted):
   `optimize_in0_ring_order` searches each ring's 5040 permutations *independently*, so all rings converge
   on the same corridors — self-inflicted contention. Replacing that with a joint objective minimizing peak
   link load (exact route model, `ring_topology_probe.py`) cuts the busiest link 28->21 and measures
   **+4.2% on 512x6144x2304** (stable both relaunch orders), +1-3% on 256x2048x6144, but **-1.6% on
   512x6144x4608 and -10% on 256x2048x2048 (Sm=3)**. The two 512x6144 shapes have IDENTICAL in0 ring
   problems (same placement, same 128 KB shard, same chosen order) and differ only in in1 volume, which
   localizes the remaining error: the model must cover **in1 reads + reduction + output**, not just in0, and
   needs a **hop budget** to stop it trading distance for a marginal peak gain under Sm>1. See
   `IDEA1_RING_TOPOLOGY.md`.
3. **Joint placement objective** (bytes x hops over in0 ring + in1 read + reduction + output) replacing the
   current two-phase bank-adjacent + IN1_NEAR heuristics. Hard constraint: keep bank adjacency (prior art:
   compact/rect placement collapsed the in1 read to 28%); optimize only the residual freedom of which core
   in a bank blob gets which slice index.

### Cheap kernel / CB changes

4. **in1 prefetch depth = fill leftover L1.** `cb1` is `4*kb*N_sub` = **4 blocks**, so the in1 reader (the
   dominant DRAM consumer) is paced by compute, which is paced by the gather: during the 77 us gather it
   runs at ~20% duty, leaving ~60 us of idle DRAM while ~40 us of in1 read still remains afterwards. Policy
   `cb1_depth = max(4, free_L1 / block_bytes)` consumes only leftover L1, so it can never make a config
   infeasible (~220 KB free today = ~55 extra blocks ~= 70% of the whole in1 stream for this shape).
   NOTE: a "CB1 depth" knob was rejected earlier as an *in1 backpressure* lever; this is a different
   mechanism (overlapping the in0 gather) and that rejection does not cover it.
5. **Dual-NoC ring forwarding** (split each shard across NOC_0/NOC_1, two semaphores, `DM_DYNAMIC_NOC`).
   De-prioritized by D1: it attacks injection, i.e. at best the ~30% residual.
6. **Multicast only the cross-Ns sibling copy.** Mcast is the one mechanism that cuts per-core L1 sourcing
   7x, but a rect destination set fights bank adjacency. Exception: the `nn`-siblings of one bank live in
   the same blob and can be forced into a `1xNs` column — a rect needing no placement compromise.

### Structural

7. **Two-phase bisection-minimal all-gather.** Region-local gather of 16 sub-shards + ONE cross-region
   half-slice exchange + local re-broadcast: in0 DRAM read **halves** (1 copy instead of Ns; the read is
   100% exposed, so ~12 us) and crossings drop ~4x, at unchanged per-core injection. The structural
   optimum; gate on (1) paying off.
8. **Wavefront (skewed KxN) traversal.** Today's N-outer + fully resident cb0 forces the whole gather into
   the first ~1/N_bpc of the kernel (hence the congestion) and burns 1 MB of 1.44 MB L1. A diagonal
   traversal spreads in0 delivery over the whole kernel, shrinks cb0 to ~N_bpc+1 blocks (frees ~600 KB,
   feeding (4) and larger kb/nsb) and keeps output/reduction staggered. Needs credit-based ring flow
   control and N_bpc accumulators. Biggest ceiling, biggest risk.
9. **Picker: penalize `8*Ns*|in0|`.** in0 delivery scales with Ns and is independent of Pk/Sm — A/B
   Ns1/Pk12 vs Ns2/Pk6 on the deep shapes with fixed configs (no code change).

## 5. Do not retry (prior art)

- Compact / rect placement to enable mcast — measured: collapses the in1 DRAM read to 28%.
- in0 ring chunk-streaming (finer forward granularity) — a promising single run collapsed under 2x10
  relaunch confirmation.
- in0 scatter / eager-exchange / round-robin / replicated-ring — all lost to the ring, but note they were
  tested on 256x2048x1024, where in0 delivery is *not* exposed (Z_C_IN0WAIT ~0.01 us); the deep golden
  shapes are a different regime (in0 delivery = 35-48% of the wall), which is the documented condition for
  reopening.
- Reduction tree (depth 3 -> 2) — refuted on the primary shape.
- Replacing the ring with replicated DRAM reads — wrong exchange rate (section 2).
