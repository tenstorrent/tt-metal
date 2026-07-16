# Physical-topology-aware in0 ring ordering — report

Change: the in0 all-gather ring no longer visits the 8 bank-cores in bank-index order `0→1→…→7`. The factory
now picks, **per ring group**, a cyclic visiting order that minimizes the physical NoC route cost of the
forward edges, using the group's **writer NoC** and the authoritative directed-torus hop distance
(`tt::tt_metal::experimental::Device::get_worker_noc_hop_distance` — logical→physical + NOC0 `+x→+y` / NOC1
`−x→−y` routing with wraparound, harvesting-aware). Placement, work partitioning, compute, and reduction are
unchanged; only `ring_pos`/`ring_next_idx`/`ring_prev_idx` (hence which core seeds which in0 shard, the
forward route, and the in1 rotated read) change — the output is bit-identical for any permutation.

**Verdict: OPT (exhaustive) WINS — shipped as the production default.** −4 to −10% on the Mt=8 primaries and
wide-N controls, neutral on small-M (Mt≤1), no regression. Bank order (previous production) and greedy are
retained as `DIAG_RING_BANK` (`1<<12`) / `DIAG_RING_GREEDY` (`1<<13`) A/B diagnostics.

## Correction to prior documentation
Earlier notes described the production ring as nearest-neighbour. That was inaccurate: production hardcoded
`in.nn_chain = false` → **bank-index order**, and the planner's `nn_chain` path used *logical* Manhattan
distance (a proxy), not physical routing. This change replaces both with physical-NoC-aware ordering.

## Ordering modes (constructed per ring group; a ring is homogeneous in NoC)
Each ring = the 8 bank-cores of one `(kk, n-slice, m-block)` slice; all 8 share the slice's NoC, so the
group has a single writer NoC = opposite the reader's (`noc==0`→writer NOC1, `noc==1`→writer NOC0).
- **bank** (`DIAG_RING_BANK`): `[0..7]` — previous production baseline.
- **greedy** (`DIAG_RING_GREEDY`): greedy nearest-neighbour over the writer-NoC hop distance from bank 0.
- **opt** (DEFAULT): exhaustive over the 7! cycles through bank 0 (rotation-invariant), minimizing **max edge
  cost** then **total hops** (secondary). Directed edge costs on the writer NoC ⇒ a permutation and its
  reverse are both evaluated (both orientations). ~5040 cycles × preaders groups, each scored from a
  precomputed 8×8 hop matrix — a one-time host cost at program compile (<10 ms), negligible vs kernel build.
- **M-split (Sm>1):** the in1 *slaves* consume in1 in the mm==0 *reader's* shard order while their in0 rings
  are separate, so all `Sm` slices of a `(kk,nn)` group MUST use the SAME permutation (reader/slave
  `ring_pos` must agree per bank). The order is computed once from the mm==0 slice and applied to the group.
  (Getting this wrong corrupts Sm>1 output — caught in correctness during development.)

Congestion is not modeled explicitly (the hop helper returns distance, not the link path); **total hops** is
the congestion proxy, **max edge** is the primary (the longest single forward, i.e. the critical forward).

## Route cost (from the factory RINGCOST lines; aggregated across ring groups)
Opt roughly **halves** both metrics vs bank on the deep-Pk shapes:
| shape | bank max-edge | opt max-edge | bank total-hops | opt total-hops |
|---|---|---|---|---|
| 256×2048×1024 (Pk4) | 20 | 15 | 345 | 222 |
| 256×6144×768 (Pk12) | 25 | 14 | 1159 | 683 |
| 256×6144×{2304,4608}, Mt4 (Pk12) | 25 | 14 | 1159 | 683 |
| 64×6144×4608 (Pk6) | 22 | 16 | 480 | 323 |

## Correctness (gtest `RegimeADiagFixture.RingOrderCorrectness`)
bank / greedy / opt, random BF16 vs a CPU f32 golden, PCC ≥ 0.999, fresh AND cached-program, **all
BIT-IDENTICAL** across orders, covering: Pk=1 + split-K (Pk=2/4/12 give both reader-NoC orientations), Ns>1,
Sm>1, W=1/W>1, balanced K/N tails. Public 20/20 suite passes on the opt default. Watcher clean on the opt
default (Pk=12). (A pre-existing, unrelated `in1_reader` Sm>1 atomic-flush watcher warning fires identically
at bank baseline — orthogonal to ring ordering.)

## Performance (median device-profiler kernel µs, 3 interleaved relaunches, Δ vs bank)
Raw: `regime_a_ringorder_bench.json` (all relaunches, per-RISC, per-group RINGCOST, util%512, PCC). Two
independent runs agreed (opt −5.3/−10.3% and −4.4/−8.4% on the primaries — reproducible, relaunch variance ~1%).
| shape | group | cfg (Ns,Pk,Sm,kb,nsb) | bank µs | greedy Δ | **opt Δ** |
|---|---|---|---|---|---|
| 256×2048×1024 | target | 1,4,2,2,2 | 28.3 | −0.5% | **−4.4%** |
| 256×6144×768 | target | 1,12,1,2,1 | 51.6 | −2.3% | **−8.4%** |
| 256×6144×2304 | control | 1,12,1,2,1 | 90.5 | −2.9% | −4.8% |
| 256×6144×4608 | control | 1,12,1,2,1 | 151.2 | −1.8% | −3.4% |
| 32×6144×4608 (Mt1) | control | 1,12,1,2,1 | 116.8 | +0.1% | −0.0% |
| 64×6144×4608 (Mt2) | control | 1,6,1,4,2 | 118.6 | −0.0% | −0.1% |
| 128×6144×4608 (Mt4) | control | 1,12,1,2,1 | 128.5 | −1.5% | −2.5% |

**Opt beats greedy on every shape** (greedy is a heuristic — it can even lose to bank on individual groups),
so opt is the choice. The win is largest on the shallow-N deep-K primary 256×6144×768 (**−8.4%**), scales
with M-block/core count (Mt≥4), and is **neutral on small-M** (Mt≤2: 0 to −0.1% — the ring is fully hidden
behind compute there). No regression.

### Per-RISC (median µs) — shorter routes shift the whole pipeline earlier
| shape | order | wall | BRISC | NCRISC (writer) | TRISC |
|---|---|---|---|---|---|
| 256×6144×768 | bank | 51.6 | 43.4 | 43.4 | 43.5 |
| 256×6144×768 | opt | 47.2 | 38.5 | 38.7 | 39.1 |
| 256×2048×1024 | bank | 28.3 | 22.6 | 22.6 | 23.1 |
| 256×2048×1024 | opt | 27.1 | 21.5 | 21.7 | 22.8 |
| 32×6144×4608 (Mt1) | bank | 116.8 | 113.0 | 113.2 | 112.9 |
| 32×6144×4608 (Mt1) | opt | 116.8 | 113.0 | 113.1 | 112.8 |

On the winning shapes all three RISC spans drop ~4–5µs **in lockstep** — the shorter forward routes cut the
in0 ring-gather latency that gates the reader (BRISC), the writer's forwarding (NCRISC), and compute (TRISC)
alike, so the whole pipeline reaches steady state earlier. On Mt1 every span is flat (ring hidden). This is
why the effect showed up now: with the recently-landed pipelined phase-2 drain tightening the reduction
chain, the in0 forwarding **route** cost became exposed on the critical path — bank order scattered the ring
cores up to 25 NoC hops apart on a single forward, which opt cuts to 14.

## Decision & validation
Opt is the **production default** (public path, mask 0). Bank (`DIAG_RING_BANK`) and greedy
(`DIAG_RING_GREEDY`) are retained for A/B. Validation on the opt default:
- Public 20/20 correctness: pass.
- Six-shape parity vs the frozen (bank-order + barrier-era) C++ oracle: op faster on all six (−1.3% to
  −15.0%; the `|Δ|>5%` flags are the op *outrunning* the oracle — the cumulative pipelined-drain + opt-ring
  win — not a regression; no shape is slower).
- FLUX/LTX: ring ordering is independent of picker config selection and output is bit-identical, so picker
  choices/correctness are unchanged; the perf effect is a uniform Pareto improvement (Mt≥4 faster, Mt≤2
  neutral). A full FLUX/LTX perf re-sweep is optional follow-up.

The exhaustive 7!-per-group search runs host-side once per program compile; the RINGCOST diagnostic line is
gated behind `TT_MM_RINGCOST` so the production path is silent on compile.
