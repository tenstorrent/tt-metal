# Mt=8 diagnostic ablation report — component decomposition of the M-dependent excess

System under test: the shipping `ttnn.experimental.regime_a_matmul` kernels, driven through the internal
`ttnn::prim::regime_a_matmul_diag` entry with compile-gated ablation masks (RegimeADiag). The public path
is mask 0 and is byte-identical; raw data in `regime_a_diag_matrix.json` (per-shape, 3× relaunch) and
`regime_a_diag_mscale.json` (M-scaling). All numbers are median device-profiler kernel µs over 8 timed iters.

**Ablations are non-additive critical-path counterfactuals** (stages overlap): a large speedup under an
ablation means that stage is *exposed* on the critical path, not that its cost adds linearly.

## Zero-impact verification (Part 4)
- Full correctness suite: **20/20 pass** with the plumbing in (mask-0 public path).
- Mask-0 `full` vs the earlier characterization (median µs): 256x2048x1024 30.5 vs 30.1; 256x6144x768 53.9
  vs 54.3; 256x6144x2304 93.7 vs 94.1; 256x6144x4608 153.5 vs 154.6 — all within ~1% → perf-neutral.
- Six-shape perf-parity (op vs frozen oracle numbers): **worst |delta| 1.7% ≤ 5% bar → pass**.
- Mask-0 before/after deltas are within measurement spread except 256x2048x1024 (+1.3%, 30.1→30.5µs); that
  shape's per-relaunch spread is ~5%, so the +1.3% is noise, not a regression (the other three are <1%).

## Per-shape ablations (winner config; Δ vs full)
Δ = (ablation − full)/full. ideal = logical bytes / 512 GB/s.

| shape (N) | ideal | full | skipin1 | skipin0 | skipfwd | noreduce | skip in0+in1 | +noreduce (floor) |
|---|---|---|---|---|---|---|---|---|
| 256x2048x1024 (1024) KxM | 11.3 | 30.5 | 26.2 (-14%) | 27.9 (-9%) | **22.9 (-25%)** | 27.9 (-9%) | 24.3 | 20.4 (1.8x ideal) |
| 256x6144x768 (768) pure-K | 25.3 | 53.9 | 50.7 (-6%) | 50.4 (-6%) | **37.8 (-30%)** | 45.4 (-16%) | 45.5 | 33.9 (1.34x) |
| 256x6144x2304 (2304) pure-K | 63.7 | 93.7 | **76.2 (-19%)** | 88.7 (-5%) | 76.5 (-18%) | 87.6 (-7%) | 70.5 | 58.3 (0.92x) |
| 256x6144x4608 (4608) pure-K | 121.3 | 153.5 | **113.1 (-26%)** | 149.4 (-3%) | 137.9 (-10%) | 154.3 (~0%) | 107.8 | 95.0 (0.78x) |

**The dominant exposed stage shifts with N:**
- **Narrow-N (768, 1024):** the **in0 ring FORWARD** (`skipfwd`) is the single largest exposed cost
  (−25% to −30%). in1 read is barely exposed (`skipin1` −6 to −14%). Reduction matters only at deep Pk
  (768/Pk12: `noreduce` −16%). The compute+ring-sync floor (`skipin0+in1+noreduce`) is **above** the DRAM
  ideal (1.34–1.8×) → narrow-N is NOT bandwidth-bound; the pipeline (compute feed + ring semaphore schedule
  + in0 forwarding) sets the floor.
- **Wide-N (2304, 4608):** the **in1 READ** (`skipin1`) is dominant (−19% to −26%); reduction is hidden
  (`noreduce` ~0 at 4608). The compute-only floor is **below** the DRAM ideal (0.78–0.92×) → wide-N IS
  bandwidth-bound and healthy (~78% of 512 GB/s at N=4608). Its modest excess is in0-forward + tail.

## Excess-time table (measured − ideal_DRAM), M-scaling (Part 6)
Config = the picker's auto config at each M. Excess in µs.

**small-N (K=2048, N=1024):**
| ablation | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|
| full | 6.1 | 7.9 | 12.5 | 18.8 |
| skipin1 | 1.2 | 4.2 | 8.1 | 15.1 |
| skipfwd | 6.1 | 7.2 | 6.1 | 11.7 |
| noreduce | 2.0 | 3.0 | 10.4 | 16.6 |
| compute+sync floor (skip in0+in1+noreduce) | **-4.6** | **-2.9** | **4.1** | **9.9** |

**wide-N (K=6144, N=4608):**
| ablation | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|
| full | 6.7 | 8.9 | 15.0 | 32.9 |
| skipin1 | -78.7 | -64.5 | -53.4 | -7.9 |
| skipfwd | 6.3 | 6.1 | 7.8 | 16.2 |
| noreduce | 2.5 | 6.2 | 13.6 | 32.8 |
| compute+sync floor | -91.6 | -69.6 | -65.5 | -26.2 |

## Component-level explanation of the M-dependent excess
The full excess over the DRAM floor grows with M in BOTH regimes (small-N 6→19µs, wide-N 7→33µs). Its
composition:
- **Grows with M (NOT amortized by N):** the **compute-feed + ring/reduction synchronization schedule**
  and the **in0 ring forwarding**. In small-N this is the whole story — the compute+sync floor's excess
  climbs from −4.6µs (M32, compute under the roofline) to +9.9µs (M256, compute+sync now over the DRAM
  ideal), and in0-forward adds a further ~7–16µs; both scale with the M-block/core count, not with N.
- **Amortized by total in1 work (grows with N, not the problem):** the **in1 read**. It is negligible at
  narrow-N and only becomes dominant at wide-N — where it runs at near-optimal DRAM bandwidth (wide-N
  reaches ~78% of 512 GB/s; `skipin1` shows in1 is ~the entire DRAM floor there). So wide-N is healthy.

**Why narrow-N Mt=8 sits at 37–47%:** the M-dependent compute/ring-sync + in0-forward excess is a large
fraction of a tiny total workload; there isn't enough in1 (N) work to amortize it. This is consistent with
"efficiency tracks total DRAM work" — not a fixed per-invocation overhead, and not an in1-delivery problem.

## Decision gate (Part 7) — no production change made this phase
Candidate that appeared to have ≥8–10% *upper-bound* potential on two Mt=8 shapes:
- **in0 ring-forward reduction** — `skipfwd` frees −25% (256x2048x1024) and −30% (256x6144x768). But
  `skipfwd` only removes the payload write (keeping the per-step push pipeline and running compute on
  garbage), so it is an UPPER BOUND, not a realizable gain. The direct-scatter prototype below tests the
  realizable version and refutes it.

**Recommended experiment (RUN — see result below):** a test-only direct-scatter in0 delivery
(`DIAG_IN0_SCATTER`, mask 32): read own shard, scatter it to the G-1 cores ahead in ONE round, receive the
G-1 shards from the cores behind — replacing the G-1 serial ring rotations while reproducing the identical
cb0 layout (so compute/in1-pairing/reduction/output are unchanged). Note (per review): scatter does NOT
reduce total forwarded bytes (each shard still reaches 7 consumers); it removes serial rotations / critical
-path hops. `skipfwd`'s −25/−30% is an upper bound, not the expected scatter gain.

### RESULT — direct-scatter REFUTED (do NOT port)
Correct (PCC max_rel_err 0.0 on both targets). Benchmark (winner config, median of 3 relaunches; raw in
`regime_a_scatter_bench.json`):

| shape | group | ring µs | scatter µs | scatter vs ring |
|---|---|---|---|---|
| 256x2048x1024 | target | 30.2 | 33.1 | **+9.8% (slower)** |
| 256x6144x768 | target | 54.0 | 59.8 | **+10.8% (slower)** |
| 256x6144x2304 | control | 93.5 | 97.1 | +3.8% |
| 256x6144x4608 | control | 153.5 | 158.5 | +3.3% |

Direct-scatter **regresses on every shape**, worst on the two narrow-N targets — the opposite of the
`skipfwd` upper bound. Mechanism: the ring **overlaps forwarding with compute** — it `cb_push_back`s each
shard as it arrives, so compute starts consuming while later shards still rotate. The barrier-style scatter
must wait for all G-1 shards before pushing (single-count semaphore), losing that pipelining, and issues
G-1 concurrent NoC writes per core (8×7 in flight) that contend. So the ring's "serial" rotation is not
wasted latency — it is what hides the in0 delivery behind compute. `skipfwd` looked like headroom only
because it removed the payload write while keeping both the per-step push pipeline AND (garbage) compute.
**Per the ≥8–10% gate this does not clear — no production change made.**

**Next hypothesis (not run):** an *incremental-arrival* scatter (per-slot semaphores so each shard is
`cb_push_back`ed the moment it lands) could remove the serial-rotation dependency while keeping the
compute-overlap the ring relies on. This is the only remaining in0-delivery lever with a plausible path to
the `skipfwd` headroom; everything simpler (this barrier scatter) is refuted. If it too fails to clear
≥8–10%, narrow-N Mt=8 is at its practical floor for the current ring/compute architecture and the residual
is the compute+ring-sync schedule (a tiny-shape kernel-path question), not an in0-delivery one.

## Notes / limitations
- Ablations are counterfactual upper bounds, not additive; realizable gains are smaller.
- `NO_REDUCE` removes reduction communication AND the reduction-add compute together (combined).
- LOCAL_FEED (pure compute/pack floor, no ring/sync) is implemented as a define hook but not yet in the
  kernels; the `skipin0+in1+noreduce` floor still includes ring-gather + M-split semaphore schedule, so the
  reported compute floor is an over-estimate of pure compute. Add LOCAL_FEED if isolating compute from ring
  sync becomes necessary (the narrow-N floor being 1.3–1.8× ideal already suggests sync is material).
