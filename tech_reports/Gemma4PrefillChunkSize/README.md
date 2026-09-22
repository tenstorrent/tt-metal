# Gemma4 prefill: why chunk 2048 is ~2x slower than 8192 at long context

> ## Navigation
> | doc | what it is |
> |---|---|
> | **[EXPERIMENTS.md](EXPERIMENTS.md)** | every experiment incl. negatives, keyed by env flag |
> | **[REPRODUCE.md](REPRODUCE.md)** | re-measure after a rebase: 6 runs + one drift check |
> | [scripts/](scripts/) | `run_e2e.sh`, `capture.sh`, `fit_chunks.py`, `diff_ops.py`, `check_baseline.py`, `BASELINE.json` |
> | [L1Activations/](L1Activations/) | 23 per-op tables, unpatched and patched |
> | [CHUNK_SIZE_ANATOMY.md](CHUNK_SIZE_ANATOMY.md) · [PER_OP_TABLES.md](PER_OP_TABLES.md) | earlier detail |



> **Base note.** Every number in this file is on the **pre-L1-activations** base.
> For the same measurements on `mmanzoor/svuckovic/gemma4-L1-activations` @ `9dc8e32a2`
> (Asif's branch, PR #56862) see [`L1Activations/`](L1Activations/). **Do not compare
> numbers across the two** — re-render both sides on one build instead.

Measured on a BH Galaxy, mesh 8x4 (CP8/TP4), branch `kmabee/gemma4-swa-multihop-halo`.
Gemma4-31B: 60 layers = 50 sliding + 10 full attention, hidden 5376, 16 KV heads, head_dim 256,
sliding window 1024.

---

## TL;DR

**Chunk 2048 is 2.09x slower than 8192 over a 256k prompt because of two independent effects,
each worth ~2x, living in different layers and needing different fixes. Neither is the
sliding-window halo, and neither is the fabric.**

| | 2048 vs 8192 | cause | whose |
|---|---|---|---|
| **per-chunk term** | **2.16x** | a **~94 ms cost** that doesn't shrink with the chunk, paid 4x more often (chunk is 4x smaller but only 1.85x cheaper) | ~84% the **50 sliding** layers, by count |
| **prefix term** | **1.99x** | the attention op leaves **71% of the core grid idle** at chunk 2048 (32 work units on ~110 cores) and pays 4x as many steps | **100%** the **10 full-attention** layers |
| | **= 2.09x** | | |

Prefill time splits exactly into two terms — a cost paid once per chunk, and a cost that grows
with how much context precedes each chunk:

`T = N·a(C) + slope(C)·N(N−1)/2`, with `N = ISL/C` chunks

| chunk | N | `a` = per-chunk (= TTFT) | per-chunk term | prefix term | **total at 256k** |
|---:|---:|---:|---:|---:|---:|
| 2048 | 128 | 131.3 ms | 16.81 s (**2.16x**) | 11.85 s (**1.99x**) | **28.66 s (2.09x)** |
| 4096 | 64 | 174.2 ms | 11.15 s (1.44x) | 6.04 s (**1.02x**) | 17.19 s (1.25x) |
| 8192 | 32 | 242.7 ms | 7.77 s | 5.95 s | 13.71 s |
| 16384 | 16 | 436.2 ms | 6.98 s | 4.84 s | 11.82 s * |
| 32768 | 8 | 917.9 ms | 7.34 s | 3.94 s | 11.28 s * |

> **\* The 16384 and 32768 rows were re-measured on 2026-09-18** and this table now carries
> the new values. The originally published slopes were **low by ~9–10%**: 37.10 → **40.33** at
> 16384 and 126.2 → **140.69** at 32768. That is a bad original fit, **not a regression** —
> refitting the surviving 2026-09-09 logs from the pre-halo branch `d3064a5fd6b` gives 40.91
> and 139.7, so two independent builds agree to 1.4% and 0.7% with the new numbers while the
> published ones are the outliers. The large-chunk slopes had been fitted from too few points
> (at chunk 32768 a `ctx_32k` run has N=1, from which no slope can be fitted at all). Both
> fresh runs are 8–16 points at ctx 256k with R² ≥ 0.9996. The chunk **2048 / 4096 / 8192**
> rows reproduce to **0.03%** and the 2.09x / 2.16x / 1.99x conclusions are unaffected; the
> best-throughput ranking is also unchanged (32768 fastest, then 16384). Found when the
> `prefill-perf-debug` skill refused to validate the 32768 row against any surviving log.

### Effect 1 — a fixed ~94 ms cost per chunk, paid 4x more often (2.16x)

Every chunk pays **~94 ms that does not shrink when the chunk shrinks**. A 4x smaller chunk is
only 1.85x cheaper, so 4x as many chunks costs 2.16x more overall. What it is, apportioned by
the per-op shares measured in [`PER_OP_TABLES.md`](PER_OP_TABLES.md):

| component | per chunk | why it does not scale with the chunk |
|---|---:|---|
| matmul weight reads | ~41 ms | the layer's **127 MB of weights** (per device, TP=4, bfp8) are read once per chunk whatever the token count. 2x the math from M=512→1024 costs only **1.09x** the time — saturated at the weight-read bound |
| `rms_norm` | ~24 ms | `ttnn.rms_norm` parallelises over **rows only**. A 256-row slab is 8 tiles → **8 of 120 cores**, each reducing a 168-tile-wide row-block. 4x the rows uses 4x the cores for **1.22x** the time |
| sliding-window SDPA halo | ~13 ms | the halo is a constant 1024 tokens regardless of chunk size |
| heads / rope / tilize / TP collectives | ~16 ms | TP collectives are only **4%** of the floor — they scale properly |

**83–85% of the floor is the 50 sliding layers** — not because each is expensive, but because
there are 50 of them. Their cost is chassis (weight reads, norms), not attention. Three
independent estimators give **82.8% / 84.8% / 85.0%**, and that agreement is the evidence —
see [`PER_OP_TABLES.md`](PER_OP_TABLES.md). (The **absolute** floor is far less
estimator-stable than its split: the same three give 79–113 ms, which is why the whole-model
affine fit below is the figure to quote.)

> **On the number itself.** An earlier revision quoted this floor as **70.4 ms**. That is the
> *excess over ideal token scaling* at chunk 2048, which is algebraically three quarters of the
> chunk-invariant cost — not the cost itself. The chunk-invariant cost is **~94 ms** (whole-model
> affine fit over 2048–8192: 94.2 ms; independent per-op fit: 105–113 ms). The `rms_norm` row
> above, 24 ms, is confirmed directly by the per-op fit at **24.3 ms**. The 2.16x / 1.99x / 2.09x
> results are measured from per-chunk device times and do not depend on either figure.

### Effect 2 — the prefix attention wastes 71% of the grid at chunk 2048 (1.99x)

Each chunk attends its tokens against all preceding tokens. **100% of that growth is the 10
full-attention ("global") layers' `RingJointSDPA`. The 50 sliding layers are exactly flat** —
slope −0.0001 ms per chunk index over a full 256k prefill.

The op splits work into units of `q_chunk_size` rows × one head, then `div_up`s them over ~110
cores — and **cores with no work are not skipped**; they run padded handshake iterations. With the
shipping `q_chunk_size=64`:

| chunk | work units | rounds over 110 cores | **grid occupancy** |
|---:|---:|---:|---:|
| 2048 | 32 | 1 | **29%** |
| 4096 | 64 | 1 | 58% |
| 8192 | 128 | 2 | 58% |
| 16384 | 256 | 3 | 78% |
| 32768 | 512 | 5 | 93% |

Chunk 2048 leaves **71% of the grid idle** *and* pays 4x as many prefix steps. A zero-parameter
model (`cost ∝ rounds/C`, where rounds is integer division) predicts the prefix term within
**3%** across chunk 2048 / 4096 / 8192 — the range this investigation is about. It degrades
outside that range: **−10.8%** at 16384 and **−14.8%** at 32768 (see the re-measurement note
below). So the mechanism is established where it was tested, and the model should not be
extrapolated to large chunks.

### The one number worth acting on

**Chunk 4096's prefix term is already within 2% of chunk 8192's.** All of 4096's 1.25x penalty is
the per-chunk floor — the prefix penalty appears only *below* 4096. So 4096 buys a **1.39x better
TTFT for 1.25x worse throughput**, whereas 2048 costs 2.09x and is not viable at long context.

### What the global attention op is actually bound by

Worth stating because it was previously reported as *fabric-bound*. It is not:

| we changed | effect on the prefix cost | verdict |
|---|---:|---|
| K/V **bytes** −47% (bfp8 → bfp4) | **0.998x** | **not** bandwidth-bound |
| softmax **exp** path (fast SFPU exp) | **1.007x** | softmax is negligible |
| **MAC passes** (HiFi2 → LoFi) | **0.786x** | **this is the cost** |
| MAC passes (HiFi2 → HiFi4) | **1.755x** | **this is the cost** |
| inner-loop blocking (`k_chunk` 256 → 128) | 1.156x worse | 256 is already optimal |

**It is MAC-throughput-bound on the QK^T and PV matmuls.** The earlier claim that ~53% of the op
was fabric movement does not hold — see [`CHUNK_SIZE_ANATOMY.md`](CHUNK_SIZE_ANATOMY.md) §4.

---

## How we measured it

Deliberately **not** mostly Tracy. Tracy told us *which ops*; it could not tell us *why*, because
its utilization columns (`ETH BW UTIL`, `NOC UTIL`, `DRAM BW UTIL`, `CB WAIT FRONT`) are **empty
on this path**. The "why" came from causal ablations.

| # | method | what it gave | notes |
|---|---|---|---|
| 1 | **Whole-model per-chunk device times** — the demo's own `[traced_perf]` lines, fitted to `t_i = a + slope·i` | the two-term split, and `a` / `slope` per chunk size. **The backbone of everything.** | R² ≥ 0.9996; extrapolates to 256k within 1% of independent 256k runs. No profiler |
| 2 | **Layer-count differencing** — whole model at 60 / 12 / 6 layers, then difference | the true per-layer cost **and** the non-layer cost (embedding + final norm + LM head + inter-layer CCL = **0.4% of TTFT**) | linear in layer count to 5–6%; no profiler. Valid because the HF layer pattern is a regular `[sliding×5, full]` repeat |
| 3 | **Per-layer-type depth curves** — the in-tree isolated-layer benchmark at every chunk index | separated **global vs sliding**: global slope 0.1455, sliding **−0.0001** | 256 measured replays; reconstructs the whole-model slope to 0.02–2.6% |
| 4 | **Tracy per-op profiling** (`python -m tracy -r -p -v`, chunk index 0) | the floor's **composition** — matmul 43%, `rms_norm` 26%, SDPA 14%, TP collectives 4% | the only place Tracy was used, and only for *which op*, never *why* |
| 5 | **Causal ablations** (temporary env-var diagnostics, all reverted) | **every mechanism finding**: K/V bytes, softmax, MAC fidelity, `q_chunk`, `k_chunk` | each varies one cost term and holds the rest fixed. No FLOP anchor, no spec-sheet bandwidth, no utilization column |
| 6 | **Off-model microbenchmarks** — standalone single-device scripts | that `rms_norm` is width-bound and precision-insensitive; that the in-model matmul config already beats a default | iterates in seconds instead of minutes |
| 7 | **Reading the op's source** | the work-unit math `div_up(B·NH·num_q_chunks, num_cores)`, and that idle cores are not skipped | `ring_joint_sdpa_program_factory.cpp` |

**Cross-validation was the point.** The floor was reached three independent ways
(whole-model fit, layer-count differencing, per-op summation) agreeing to **0.3%**. "The prefix
term is 100% global layers" was confirmed three ways (depth curves, per-op capture, layer-count
differencing) agreeing to **0.6–3.4%**.

**Two conclusions were retracted mid-investigation**, both from over-reading a model: a
"69% movement / 31% math" split that a second ablation contradicted, and an "exactly
chunk-invariant `rms_norm`" that came from comparing captures across two different builds. Both
are recorded in the detail doc rather than quietly fixed, because the lesson generalises: **never
compare per-op timings across builds, and never fit a two-term model to an ablation that changed
two things at once.**

---

## If you want to make it faster

Not what the investigation was for, but it fell out of it:

| option | effect at 256k | status |
|---|---|---|
| `q_chunk_size` per chunk size on global layers (32 / 64 / 128 for chunk 2048 / 4096 / 8192) | **−4.6%** at 8192, −8.0% at 2048, **TTFT unchanged** | **verified end-to-end; program-config change only** |
| SDPA math fidelity HiFi2 → LoFi | −9.9% | **blocked**: triples single-layer RMSE (0.0059 → 0.0178), compounding over 60 layers. Needs a real eval, not a PCC spot-check |
| Width-shard `rms_norm` | ~−3% throughput, −12.6 ms TTFT | ~2x measured (not the ~8x a width argument suggests); needs block-sharded activations plumbed through |
| **Variable chunk size per request** | collapses a 1.61–1.85x worst case toward ~1.0x | **largest available win**; separate effort |
| More fabric links / fabric mux / line-multicast halo | ~0 | previously measured to be worth nothing; the bytes ablation above explains why |

Also worth knowing: **HiFi4 is strictly dominated** for this op — no measurable accuracy gain over
HiFi2 (PCC 0.99967 vs 0.99965) for **1.755x** the prefix cost.

---

## More detail

| file | contents |
|---|---|
| [`PER_OP_TABLES.md`](PER_OP_TABLES.md) | **`tt-perf-report` per-op tables**, chunk 2048/4096/8192 side by side, at chunk index 0 and at a matched prior context of 49152 tokens. Confirms per-op that exactly one op grows with context and that the sliding layer is flat to 0.8%. Includes the corrected commands and two profiler-column traps |
| [`CHUNK_SIZE_ANATOMY.md`](CHUNK_SIZE_ANATOMY.md) | the full record: every measurement, per-op tables, the occupancy model, the ablations, both retractions, method traps, reproduction commands |
| [`NEXT_SESSION.md`](NEXT_SESSION.md) | handoff: what is solid, what is still open, the traps, the diagnostics to re-apply |
| analysis scripts — **not in-tree**, they live in `~/debug-docs/gemma4_chunk_size_anatomy-noissue/scripts/` (private repo `kmabeeTT/debug-docs`) | 20 scripts and runners — `why_2x.py` (the two-term split), `occupancy_model.py`, `parse_nlayers.py` (layer differencing), `floor_ops.py` (per-op), `fit_e2e.py`, `micro_*.py` |

Prior work this builds on and partly corrects: `gemma4_prefill_chunk_scaling-noissue/` (private repo `kmabeeTT/debug-docs`) (the
8k-vs-16k investigation) and `gemma4_swa_multihop_halo-noissue/` (same private repo) plus the sibling report
[`../SlidingWindowMultiHopHalo/`](../SlidingWindowMultiHopHalo/README.md) (the multi-hop halo that made chunk 2048 runnable).
