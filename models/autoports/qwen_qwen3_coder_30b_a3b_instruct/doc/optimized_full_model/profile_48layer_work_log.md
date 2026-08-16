# Stage 06 — where the time actually goes in a 48-layer TTNN decode step

> **This document profiles the tree as it was *before* the paged-SDPA program
> config and the live-row argmax slice were adopted.** Its op-level artifacts are
> therefore the `*_part1_preadoption.*` files in this directory, and its
> microsecond figures — `SdpaDecode` 20.704 us/layer, `ArgMax` 366.098 us,
> in-model per layer 396.904 us, iteration 19926.5 us — describe code that no
> longer ships. It is kept as written because it is the record of how the levers
> were found and sized. **The shipped profile is
> `ops_perf_full_model_48layer_decode.csv.gz` and the figures from it are in
> `README.md` and `work_log.md`.**

**A measurement and analysis pass. No model code was changed, nothing was
committed, nothing was pushed.** Every number below is from a run on this
machine on 2026-08-15, and every artifact it rests on is in this directory.

## Provenance

| what | how |
|---|---|
| tree | `8ea42a6b8ed` (stage 05) **plus the uncommitted stage-06 `tt/model.py`** (distributed argmax). `git status` at capture time: `M tt/model.py` and nothing else under `tt/`. |
| machine | 1x4 Blackhole P300_X2, `FABRIC_1D_RING`, 110 worker cores/die, `python_env` |
| capture | `python -m tracy -v -r -p --op-support-count 32000 -o /tmp/prof_fm48_dec probes/profile_full_model_48.py` → `logs/profile_full_model_48.log.gz` |
| window | `python probes/window_full_model_48.py <ops_perf_results_*.csv> --out … --layers 48` → `logs/window_full_model_48.log`, `ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz` |
| report | `tt-perf-report <window.csv> --no-color` → `tt_perf_report_full_model_48layer_decode_part1_preadoption.txt.gz` |
| split ranking | `python probes/rank_full_model_48.py <window.csv>` → `rank_full_model_48layer_decode_part1_preadoption.txt` |
| standalone probes | `probes/sdpa_depth_probe.py`, `sdpa_curpos_probe.py`, `sdpa_progcfg_probe.py`, `sdpa_crossover_probe.py`, `tile_1x32_blocker_probe.py`, `tile_1x32_in0_probe.py`, `lm_head_dram_sharded_probe.py`; logs of the same names under `logs/` |
| reference points, not re-derived | `probes/perf_full_model_part1_preadoption.csv` (token-out 21.4609 ms, model_trace 20.2222 ms, TTFT 126.161 ms — the part-1 run; it was the unsuffixed `perf_full_model.csv` when this log was written and was renamed at the end of the stage, so that the unsuffixed name holds the *shipped* run), `doc/optimized_multichip_decoder/window_decode.txt` (stage-04 single-layer window, 362.828 us) |

`doc/full_model/` was not written to and
`doc/full_model/probes/perf_full_model.py` was not run.

## 1. The 48-layer window is capturable, and it is verified

Stage 05 published a **2-layer** profile. That was the right window for the
terminal path and the wrong one for this question, so stage 06 captured all 48.
It worked; the only thing needed was `--op-support-count 32000`, because the
profiler's DRAM result buffer defaults to **1000 programs**
(`tt_metal/impl/profiler/profiler_state_manager.cpp:20`) and one 48-layer decode
iteration is **3519 programs per device** on its own. No buffer error was hit
and nothing was scaled or extrapolated: the published window is one real,
complete decode iteration.

`--sync-host-device` was **dropped**, unlike stage 05. Stage 05's own report
carries the warning that it "inflates every collective"; at 48 layers there are
96 collectives in the window and their absolute cost is the question.

### How the window was verified

Stage 03 published a window that straddled two decode iterations and invalidated
eight figures. The boundary here is checked, not eyeballed, on **ten independent
tallies per device, all four devices, every one exact**
(`logs/window_full_model_48.log`):

| count | op | why |
|---|---|---|
| 96 | `ReduceScatterMinimalAsync` | 2 all-reduces × 48 layers |
| 96 | `AllGatherAsync` | the same 2 all-reduces × 48 |
| 48 | `SdpaDecode` | 1 attention per layer |
| 96 | `SparseMatmul` | the expert pair × 48 |
| 96 | `PagedUpdateCache` | K and V × 48 |
| 3 | `Embeddings` | the boundary itself: token lookup + cos + sin |
| 1 | `ArgMax` | the sampler's per-die argmax |
| 2 / 2 / 1 | `AllBroadcast` / `Concat` / `Gather` | the distributed argmax's two composite 4-wide gathers, and its `ttnn.gather` |

Two of stage 05's constants were **stale and would have failed**: it expected
`2*layers + 1` all-gathers, from the old sampler's single full-vocabulary
`AllGatherAsync`. The stage-06 distributed argmax gathers **4-wide** tensors,
and at that width `ttnn.all_gather` takes its *composite* path — which is
`AllBroadcast` + `Concat`, not `AllGatherAsync` at all. That is itself a finding;
see lever 6.

### The independent cross-check

Summed `DEVICE KERNEL DURATION` over the window:

| device | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| ms | 19.9265 | 19.9210 | 19.9241 | 19.9199 |

against an independently measured `model_trace` of **20.2222 ms** and a
`token_out` of **21.4609 ms**. A window one iteration short or long would be off
by ~20 ms, not by 0.3. Device spread is 0.03%.

## 2. Where the time goes

Full split ranking: `rank_full_model_48layer_decode_part1_preadoption.txt`. Headline split of the
19.9265 ms iteration (device 0):

| region | ms | share |
|---|---|---|
| terminal-pre (token embedding + the two rope cos/sin gathers) | 0.0533 | 0.27% |
| **the 48-layer stack** | **19.0514** | **95.61%** |
| terminal-post (final norm, LM head, sampler, feedback) | 0.8217 | 4.12% |

**In-model per-layer cost: 396.90 us** (72 ops/layer). Top of the per-layer
ranking, as `us/layer`:

| us/layer | %iter | n | cores | op |
|---|---|---|---|---|
| 41.23 | 9.93% | 48 | 48 | `SparseMatmul 1x1x32x2048 @ 1x32x2048x1536` (gate/up) |
| 40.42 | 9.74% | 96 | 6 | `ReduceScatterMinimalAsync 1x1x32x2048` (×2) |
| 39.57 | 9.53% | 48 | 64 | `SparseMatmul 1x32x32x768 @ 1x32x768x2048` (down) |
| 26.36 | 6.35% | 48 | **1** | `TopK 1x1x32x128` (router top-8 over 128 experts) |
| 23.71 | 5.71% | 96 | 6 | `AllGatherAsync 1x1x32x512` (×2) |
| 20.70 | 4.99% | 48 | 110 | `SdpaDecode 1x1x32x128` |
| 16.48 + 16.28 + 9.20 | 10.55% | 144 | 48–110 | `ReshapeView` ×3 — expert-tile compaction |
| 13.96 | 3.36% | 96 | 110 | `Unary 1x32x32x1536` (silu, ×2) |
| 13.21 | 3.18% | 96 | 8 | `LayerNorm 1x1x32x2048` (residual norms, ×2) |
| 9.71 / 9.04 / 6.66 | 6.11% | 144 | 8–80 | QKV / o_proj / router matmuls |

Terminal-post, the whole of it:

| us | %iter | cores | op |
|---|---|---|---|
| 366.10 | 1.84% | 110 | `ArgMax 1x1x32x37984` — the distributed argmax's per-die reduction |
| 225.38 | 1.13% | 108 | `Matmul 32 x 2048 x 37984` — **the LM head**, `DRAM`-bound at 66.6%, 341 GB/s |
| ~141 | 0.71% | 1–4 | the two composite 4-wide gathers: `AllBroadcast` 21.7 + 9× `UntilizeWithUnpadding` 77.3 + 10× `Permute` 22.9 + `Concat` 3.6 + `TilizeWithValPadding` 15.6 |
| 16.93 / 14.64 | 0.16% | 108 / 32 | the sampler's `Untilize` and `Gather` over 37984 |
| 6.61 | 0.03% | 8 | `model.norm` |

## 3. The lower bound, re-derived

`48 × 0.4286 = 20.57` against a measured `model_trace` of 20.2222 — the model is
*under* the naive bound. Stage 05 attributed that to per-iteration host cost
being amortised once instead of 48 times, and that is right, but **0.4286 ms is
no longer the right figure to multiply**, for two reasons that cancel:

* stage-04's 0.4286 ms is a *wall* figure for a one-layer traced model, and it
  contains one iteration's worth of dispatch and host overhead. Its device-kernel
  content is **362.83 us** (`doc/optimized_multichip_decoder/window_decode.txt`);
* the **in-model** layer is *more* expensive in device time than the isolated
  one: **396.90 us**, +9.4%.

Reconstructing 20.2222 ms from the profile: 48 × 396.90 us of layer kernel =
19.051 ms, plus 0.053 ms terminal-pre, plus the non-sampler part of
terminal-post (0.822 − 0.581 = 0.241 ms) = 19.345 ms of kernel, leaving 0.877 ms
(4.3%) of dispatch and op-to-op gap for the whole iteration. **The right
per-layer multiplier is ~0.403 ms** (396.90 us of kernel plus its 5.9 us share
of the gap), not 0.4286. `48 × 0.403 = 19.34`, and the terminal path takes it to
20.22. The model is not under any bound; the old bound was 6% too high because it
charged 48 layers for an overhead paid once.

Where the +34.07 us/layer over stage 04 comes from: `SdpaDecode` +10.89,
`ReduceScatterMinimalAsync` +6.53, `LayerNorm` +1.64, `Matmul` (o_proj, router,
QKV) +1.5, and four extra ops per layer (72 vs 68) from stage 05's
`rotary_embedding_hf` rewiring and its resharding.

## 4. Ranks that changed since stage 04

| op | stage 04 (us/layer) | stage 06 in-model | change |
|---|---|---|---|
| `SdpaDecode` | 9.82 | **20.70** | **+111%, and it is a measurement-condition difference, not a regression — see lever 1** |
| `ReduceScatterMinimalAsync` ×2 | 33.89 | **40.42** | **+19%, and the increase is entirely in the MoE-side one — see lever 2** |
| `LayerNorm` (width-sharded) ×2 | 11.57 | 13.21 | +14% |
| `SparseMatmul` pair | 82.72 | 80.80 | −2% |
| `TopK` | 26.36 | 26.36 | 0% |
| `AllGatherAsync` ×2 | 24.22 | 23.71 | −2% |
| `ReshapeView` ×3 | 41.31 | 41.96 | +2% |

And in the terminal path the ranking is **completely new**, because stage 06's
own distributed argmax rewrote it: stage 05's report had `AllGatherAsync` at
31.69% and `ArgMax` at 29.01% of a 2-layer window, with the LM head third at
7.2%. Today the sampler's full-vocabulary gather is gone, and what is left is a
**366 us `ArgMax`** that is now the largest single op in the whole iteration
after the two `SparseMatmul`s — 1.6× the LM head it was supposed to be cheaper
than. The stage-05 audit could not have seen this; the op did not exist yet.

## 5. The eight levers

Ranked by expected gain on `token_out` (21.4609 ms today, 46.60 t/s/u).

### 1. `SdpaDecode`'s missing program config — 1.18x on the op at ctx128, **15–25x at real context**

`attention_decode_optimized` passes `program_config=_sdpa_program_config(...)`
on the *contiguous* KV path and `program_config=None` on the **paged** one
(`tt/multichip_decoder.py:1472`) — and the paged one is what the model runs.
The cap that config carries was added for a `TT_FATAL` on the contiguous path,
so the paged path was never given one.

Four standalone probes, all model-free (`logs/sdpa_*.log`):

* cost is **independent of allocated cache depth**: at `cur_pos=128`, 1024 /
  4096 / 16384 / 65536-deep caches read 28.86 / 30.17 / 28.60 / 28.70 us. The
  first hypothesis — that the stage-04→06 jump was cache-depth — is **refuted**;
* cost is **linear in `cur_pos`**: 17.68 us at 0, 27.02 at 128, 65.15 at 511,
  120.39 at 1023, 898 at 8192, 3554 at 32768. So the 9.82 → 20.70 us change is
  simply that stage 04's layer probe decoded at a much lower position than the
  full model does; nothing regressed;
* and at every position past ~256 the default is **far** off bandwidth. With
  `SDPAProgramConfig(q_chunk_size=32, k_chunk_size=512, max_cores_per_head_batch=32)`:

  | cur_pos | `None` (shipped) | `k512/c32` | speedup |
  |---|---|---|---|
  | 127 | 25.36 us | 24.49 us | 1.04x |
  | 131 | 28.36 | 24.07 | 1.18x |
  | 255 | 37.98 | 25.78 | 1.47x |
  | 511 | 65.63 | 22.99 | 2.85x |
  | 1023 | 120.90 | 23.43 | **5.16x** |
  | 4095 | 452.80 | 29.76 | **15.21x** |
  | 8191 | 899.24 | 35.45 | **25.4x** |

**Expected gain.** At the profiled ctx≈131, ~4 us/layer → 0.19 ms/iteration,
0.9% (46.60 → 47.0 t/s/u): small. At **ctx 4096 it is the whole model**: the
shipped path would spend 48 × 452.8 us = 21.7 ms/token in attention alone,
roughly doubling decode time, and the fix takes that to 1.4 ms. The advertised
contract is 262144 tokens of context, so this is not a corner case — it is the
difference between decode that holds up at length and decode that does not.

**Risk.** Not bit-identical: PCC 0.99981 / 0.99970 / 0.99853 at cur_pos
131 / 1023 / 8191, `max|diff|` ~5e-3 (flash-decode chunk order, same maths in
exact arithmetic). It must go through `run_prefill_check` /
`run_teacher_forcing` / the degenerate-output check, not be adopted on the op
number. It does **not** touch the dtype/fidelity/KV/CCL policy — it is a
program config on an op the model already calls.

**This is the one lever nobody has looked at, and it is much the largest.**

### 2. The MoE-side reduce-scatter is absorbing expert-routing skew — 0.80 ms/iteration, 3.7%

The two collectives per layer are not alike, and the profile separates them:

| | mean | median | min | max |
|---|---|---|---|---|
| attention-side `ReduceScatter` (in0 DRAM) | 12.54 us | 12.63 | 9.83 | 15.01 |
| **MoE-side `ReduceScatter` (in0 L1)** | **27.88 us** | 22.52 | **6.51** | **74.19** |
| both `AllGatherAsync` | 11.52 / 12.19 | 11.99 / 13.82 | 7.67 | 15.73 |

A 6.5-to-74 us spread on a fixed-size 128 KB collective is not work, it is
**waiting**. Per-layer expert work per die quantises in ~10 us steps (63.4 /
73.5 / 83.5 / 93.8 / 105.3 us) — the cost of one more active expert on that die
— because the global top-8 lands unevenly on the 4 dies. Summed over the
iteration the cross-die MoE work spread is **1.32 ms** and the MoE reduce-scatter
absorbs **0.80 ms** of it as idle. On the one layer where the dies happened to
balance exactly (layer 10, spread 0.4 us) all four reduce-scatters read
11.7–13.2 us, the same as the attention-side one.

**Expected gain.** Perfect balance would save ~10.5 us/layer = **0.50 ms**, 2.4%
(46.60 → 47.7 t/s/u). **Risk / caveat:** the imbalance is a property of the
router's output, so it cannot be removed without changing which experts fire,
which changes the model. What *is* addressable is the ~10 us slope per active
expert — i.e. lever 3 — and the expert-to-die assignment, which is a weight
layout choice rather than a routing change. Do not read this as a CCL lever; the
CCLs are innocent (see lever 6).

### 3. The expert `SparseMatmul` pair — 80.8 us/layer, 3.88 ms, 19.5% of the iteration

Still the largest single item, unchanged from stage 04 (82.72 → 80.80 us). With
`--active-experts 8` the report prices them at **53.1% DRAM / 12.1% FLOPs**
(gate/up, 48 cores) and **28.0% / 5.2%** (down, 64 cores), `LoFi BF16 x BFP4`.
The down projection at 28% of bandwidth on 64 cores is the weaker of the two.

**The `output_tile=Tile([1,32])` blocker still holds, and has hardened.**
Re-tested on today's TTNN (`logs/tile_1x32_blocker_probe.log`):

* with `in0` at the shipped 32×32 tile, `sparse_matmul` now **rejects the
  override outright** — `matmul_utilities.cpp:231: out_tile_shape[0] ==
  in0_tile_h`, *"the override output tile height (1) must equal to the in0 tile
  height (32)"*. Stage 02 could at least build the tensor;
* giving `in0` a 1×32 tile too makes the matmul run, and then `reshape`, `sum`
  and `untilize` all raise the same `mesh_buffer_->size() >=
  spec_.compute_pac…` family error stage 02 recorded. Only eltwise (`silu`)
  reads it; `to_torch` works. The one improvement is that `untilize` now
  **raises instead of silently returning wrong data**.

So the 1.07x that stage 02 measured is still unreachable, for the same reason,
and the item stays in the rejection ledger. **Expected gain if TTNN ever lands
non-32 tile heights outside matmul: ~1.07x on the layer plus the three
`ReshapeView`s below.** Nothing to do here now beyond keeping the upstream ask
alive.

### 4. Reshape / untilize / tilize layout churn around the experts — 41.96 us/layer, 2.01 ms, 10.1%

Three `ReshapeView`s per layer, all in the expert tail, costing 16.48 + 16.28 +
9.20 us — **more than the router's `TopK`, more than attention, and third in the
per-layer ranking**. They exist only to compact the 32×-row-padded `[B, E, 1, H]`
expert output back to `[B, E, H]`; they are the "31 + 33 + 46 us of reshapes"
the stage-02 ledger names as the cost of *not* having 1×32 tiles. Add the
per-layer `TilizeWithValPadding` (5.70) and `UntilizeWithUnpadding` (3.58 +
2.36) around the router and the pure-layout total is **~53.6 us/layer, 2.57 ms,
12.9% of the iteration** — larger than either `SparseMatmul` alone.

**Expected gain:** bounded by lever 3 and blocked on the same TTNN gap. A
partial, unblocked variant worth measuring: the second `ReshapeView`
(`1x32x32x2048`, 16.48 us, 64 cores) reshapes `down`'s output immediately before
a `mul` by the routing weight and a `ReduceDeviceOperation`; folding the routing
weight into `down`'s *input* was tried at stage 02 and measured a tie, but that
was before stage 04's tail. **Risk:** the ledger records that a plain reshape
instead of the permute here was faster and *silently wrong for every user but
the first*. Any work in this tail needs `test_optimized_decode_batch`.

### 5. The sampler's `ArgMax` — 366 us, 1.84%, and it is a brand-new rank

`ttnn.argmax` over the 37984-wide row-major per-die shard on 110 cores costs
**366 us** — the largest op in the iteration after the two `SparseMatmul`s and
**1.6× the LM head**. The stage-06 distributed argmax was measured against the
old full-vocabulary gather (1.1432 → 0.6275 ms, 1.82x) and it delivered, but it
moved the cost rather than removing it: the gather went away and the per-die
argmax is now the whole bill.

**What to try:** the reduction is a max over 37984 bf16 values per row, 32 rows
— ~2.4 MB per die, which at 512 GB/s is under 5 us. At 366 us it is running
~75× off bandwidth. A two-stage reduction (reshape to `[32, K, 37984/K]`, a
tile-layout `max` along the last axis, then a small argmax) is the obvious
shape, and the class docstring already records that `ttnn.max` over the full
shard costs 0.494 ms while `ttnn.gather` at a known index costs 0.059 — so the
cheap primitive exists. **Expected gain: up to ~0.30 ms, 1.4% (46.60 → 47.3
t/s/u)**, if a two-stage form gets within 5× of bandwidth. **Risk:** the
first-maximal tie rule is load-bearing and pinned by the existing probe; any
restructuring must keep it and re-run `probes/distributed_argmax_probe.py`.

### 6. The CCLs themselves — already right, ~0.4 ms of genuinely irreducible latency

`Ring` topology, `num_links=1` at decode, caller-owned persistent buffers: all
three were established by probe at stages 03/04 and the 48-layer profile does
not disturb them. The attention-side reduce-scatter (12.54 us) and both
all-gathers (11.52 / 12.19 us) sit in a tight 8–16 us band with no tail — that
is the latency floor of a 128 KB collective on this ring, and 96 of them is
1.15 ms. The *only* CCL anomaly is the MoE-side reduce-scatter, and lever 2
shows that is skew absorption, not the collective.

**One real, unexploited CCL finding:** the distributed argmax's two 4-wide
all-gathers do **not** take `AllGatherAsync`. At a gather dim of 4 (padded to a
32 tile) `ttnn.all_gather` falls to its composite path, which the profile shows
as `AllBroadcast` + 4× `UntilizeWithUnpadding` + 4× `Permute` + `Concat` +
`Permute` + `TilizeWithValPadding`, **twice: ~141 us**, against ~12 us for a real
`AllGatherAsync` at the layer's 512 width. That is 0.66% of the iteration spent
on the *layout* of gathering eight numbers. Stage 05 already hit the shape-8 end
of this — its `max_top_k=8` leg hung the mesh for twenty minutes on the same
composite gather — so this path is known-hostile. **Expected gain: ~0.12 ms,
0.55%**, by packing the value and the index into one 8-wide gather, or by
padding to a width the async path accepts. **Risk:** stage 05's mesh-hang;
anything here needs `../full_model/probes/ccl_watcher_ab.py` and a watcher run.

### 7. The LM head DRAM-sharded program config — **the recommendation is still printed, and it is not expressible**

`tt-perf-report` still flags it: `3474  DRAM  MatmulDeviceOperation 32 x 2048 x
37984 … 341 GB/s 66.6% … — Try a DRAM-sharded program config
(MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig)`. Essentially unchanged
from stage 05's 66.9%.

Two things kill it, and both are measured (`logs/lm_head_dram_sharded_probe.log`):

* **the headroom is tiny.** 225 us at 66.6% of bandwidth means a *perfect*
  DRAM-sharded rewrite saves at most 75 us — **0.35% of token-out**, 46.60 →
  46.76 t/s/u. Stage 05 saw it as 7.2% of a 2-layer window; at 48 layers it is
  1.13% of the iteration. The 2-layer window over-weighted it by 6×;
* **it does not fit.** `N = 37984` shards across 8 DRAM banks as 4748 columns per
  bank, and 4748 is not a multiple of 32. `ttnn` refuses:
  `tensor_layout.cpp:168 … Physical shard shape (2048, 4748) must be tile
  {32, 32} sized!` In tiles, `37984/32 = 1187`, and **1187 is prime** — there is
  no even split across 8 banks at any core count. Making it fit means padding
  the vocabulary shard, and this model's whole LM-head design is
  `vocab_padding: 0` (151936 = 4 × 37984 exactly).

**Verdict: close this lever.** It is 0.35% at best, blocked by arithmetic, and
unblocking it costs a documented policy property. The standalone shipped-config
leg reads 240.62 us, consistent with the 225–235 us in-model kernel.

### 8. Op count and dispatch — 3519 ops/iteration, ~0.88 ms of gap

The iteration's kernel time is 19.93 ms against a 20.22 ms `model_trace`, so
dispatch and op-to-op gap are **~0.88 ms, 4.3%** — already very tight for 3519
dispatches (~0.25 us each) and much better than the profiler's own inflated
3.45 ms. `tt-perf-report`'s only "High Op-to-Op Gap" entry is a single 50 us
stall before the sampler's `Untilize`, worth 44 us. The remaining sharded↔
interleaved traffic (1540 `ShardedToInterleaved` + 1356 `InterleavedToSharded`
per iteration = 0.73 ms of kernel) is the honest target here, but it is spread
across 30 call sites at 0.5–1.6 us each. **Expected gain: low single-digit
percent for a large amount of work.** Not a first lever.

## 6. Recommendation

**Pursue these three, in this order.**

1. **The paged `SdpaDecode` program config (lever 1).** It is one keyword
   argument on a call the model already makes; it is worth 1.18x on the op at
   the profiled context and **5x–25x at 1k–8k**; and it is the only lever whose
   size grows with the thing the contract advertises. Every previous stage
   optimised at a context where attention was nearly free, which is exactly the
   "the previous audit dismissed it as noise" pattern this project keeps hitting.
   Gate it on `run_prefill_check` / `run_teacher_forcing` / degenerate-output,
   because it is not bit-identical.
2. **The sampler's `ArgMax` (lever 5).** 366 us, ~75× off bandwidth, the single
   largest terminal op, and *created by the change that is currently
   uncommitted* — so it is both the newest item and the one with the clearest
   route (two-stage reduction, using the `ttnn.gather` primitive the class
   already relies on). Worth ~1.4%.
3. **The MoE expert tail (levers 2 + 4 together).** The `SparseMatmul` pair plus
   its three `ReshapeView`s plus the skew the MoE reduce-scatter absorbs is
   **6.7 ms, 33% of the iteration** — by far the biggest block on the machine.
   The clean 1.07x inside it stays blocked on TTNN, so this is exploratory rather
   than a known win, and it should be attacked third, after the two cheap
   certain ones.

**Explicitly close the LM head (lever 7)**: 0.35% at best and arithmetically
inexpressible at this vocabulary shard.

**And the honest summary of the layer itself:** stage 04 tuned it well. Nine of
the ten per-layer ops changed by less than 5% when embedded in 48 copies of
itself, the CCLs are at their latency floor, dispatch overhead is 4.3%, and the
in-model per-layer cost of 396.90 us is within 9% of what stage 04 measured in
isolation — with the whole of that 9% explained by one measurement-condition
difference (`SdpaDecode`'s decode position) and one skew effect. There is no
large, easy win left *inside the layer at ctx128*. The large win is that the
layer has never been measured at a context anyone would serve.

---

# Stage 06, part 2 — adopting the paged-SDPA program config

Part 1 above was a measurement pass and changed no model code. This part adopts
lever 1, and the adoption did not go the way part 1 predicted. **Nothing was
committed and nothing was pushed.** Provenance for every figure below: same
machine, same 1x4 P300_X2 / `FABRIC_1D_RING`, tree `8ea42a6b8ed` plus the
uncommitted stage-06 `tt/model.py`, plus the change described here; logs under
`logs/`, probes under `probes/`.

## What changed in `tt/`

| file | change |
|---|---|
| `tt/multichip_decoder.py` | paged SDPA-decode gets `SDPAProgramConfig(q=32, k=min(256, cache_depth), max_cores_per_head_batch=16)`; configs memoised; prefill config built and documented but **not** wired |
| `tt/functional_decoder.py` | `attention_prefill` gains an `sdpa_program_config` seam, defaulting to `None` (no behaviour change) |

dtype, fidelity, KV layout, CCL policy and the inter-layer residual layout are
untouched. `Topology::Linear` and `num_workers_per_link=1` were not introduced.

## 1. Part 1's recommended config was measured at the wrong dtype

Part 1 recommended `k_chunk_size=512, max_cores_per_head_batch=32` on the
strength of `sdpa_depth_probe` / `sdpa_curpos_probe` / `sdpa_progcfg_probe` /
`sdpa_crossover_probe`. **All four allocate the KV cache as `bfloat8_b`.**
`create_mesh_kv_cache` allocates `ttnn.bfloat16` (`tt/multichip_decoder.py`,
~line 1167). At the real dtype the ranking changes and `k1024/c64` stops
building at all (`program.cpp:1722`) while it builds fine at `bfloat8_b`.

Re-swept at bfloat16, 6 `k_chunk` x 4 `max_cores` at five positions
(`probes/sdpa_sweep_probe.py`, `logs/sdpa_sweep_probe_bf16.log`), finalists
re-timed at median of 5x50 over nine positions
(`probes/sdpa_sweep_confirm.py`, `logs/sdpa_sweep_confirm_bf16.log`). PCC is
against a float32 reference built from the same cache the kernel reads, not
against the `None` leg.

| cur_pos | None | k256/c16 | k256/c8 | k128/c16 | k128/c32 | k512/c16 | k64/c32 | k32/c32 |
|---|---|---|---|---|---|---|---|---|
| 127 | 23.72 | **19.00** | 17.86 | 22.52 | 24.96 | 22.48 | 22.52 | 23.91 |
| 255 | 37.54 | **17.48** | 17.23 | 19.85 | 22.43 | 19.75 | 20.85 | 23.15 |
| 511 | 65.21 | **18.78** | 18.68 | 20.30 | 20.49 | 19.71 | 23.47 | 27.31 |
| 1023 | 120.51 | **22.02** | 21.88 | 23.86 | 25.12 | 22.32 | 27.70 | 32.78 |
| 2047 | 231.02 | **25.61** | 25.57 | 28.16 | 28.34 | 25.44 | 33.21 | 36.35 |
| 4095 | 451.85 | **30.75** | 30.88 | 32.66 | 35.23 | 29.42 | 37.35 | 43.40 |
| 8191 | 893.14 | **38.15** | 41.24 | 40.83 | 42.34 | 37.09 | 44.60 | 57.10 |
| 16383 | 1777.60 | **49.83** | 62.10 | 57.21 | 53.24 | 50.36 | 59.18 | 84.72 |
| 32767 | 3545.05 | **74.13** | 104.62 | 90.09 | 75.69 | 73.90 | 88.29 | 139.95 |

`k512` is faster than `k256` by 0.3-4% at 4095 and above — and it is **wrong
in-model**, see §2 — so the choice is restricted to `k_chunk <= 256`.

**Chosen: `k_chunk_size=256, max_cores_per_head_batch=16`, fixed.** It is the
fastest safe config at every position from 4095 up, within 0.6% at 511-2047, and
its worst point is +6.4% at cur_pos 127 (19.00 vs 17.86 us for `k256/c8`) — 0.05
ms on a 20 ms iteration. **The best config is not meaningfully
context-dependent**, so there is no runtime selection; a captured decode trace
could not vary it per step in any case. The clamp in §2 is the one thing that
does vary, and it varies with the *allocated* cache, which is fixed for a
generator's lifetime.

## 2. The lever is a memory-safety hazard, and only the in-model gate saw it

`k_chunk_size` must not exceed the cache's **per-user allocated depth**.
Exceeding it does not raise — the op reads a full `k_chunk` past the end of the
cache buffer and returns whatever is there.

`test_multichip_decode_batch` allocates a 128-position paged cache and decodes at
`cur_pos=32`. At `k_chunk=256` it returns **PCC -0.10 to +0.06** against HF —
noise, not a degraded answer — but **only when another test has run before it in
the same process**. Run alone: 4/4 pass. Run after
`test_router_windows_partition_global_routing`: 4/4 fail, at every
`max_cores_per_head_batch` in {8, 16, 32, 64}. Sweeping that reproducer puts the
boundary exactly at the cache depth: `k_chunk` in {32, 64, 128} passes, 256
fails. The order-dependence is the tell — on a fresh device the memory past the
cache is zeros and the mask hides it; after another test has allocated and freed
tensors it is live garbage.

**No standalone probe reproduces this, and two were written specifically to try**
(`probes/sdpa_shallow_cache_probe.py`, `probes/sdpa_kchunk_rule_probe.py`).
Both match the shapes, the bfloat16 dtype, the 128-deep cache and the multi-user
paged page table exactly; both read PCC 0.9997 at `k512`, because in a probe the
cache is the only thing allocated. This is the same shape of miss as the stage-04
`rotary_embedding_llama` rejection, and it is the second time this stage that a
standalone probe recommended something the model cannot run.

Fix: `_sdpa_k_chunk` clamps to the cache's per-user depth (from
`page_table.shape[-1] * block_size`). At the shipped `max_context_len` (4096 and
up; the contract advertises 262144) it never binds.

## 3. Building the config per layer per call is not free

First adoption regressed `run_teacher_forcing` decode from **40.47 to 29.38
t/s/u** — 27% slower. Cause: `ttnn.SDPAProgramConfig` was built inside the layer,
48 times a token, and each build calls `device.compute_with_storage_grid_size()`,
a device query rather than a Python attribute. Free on the traced decode path
(capture-only), 96 device round-trips per token on the untraced ones.
Memoising on the grid size took it to **41.74 t/s/u**, above baseline.

## 4. Prefill has the same gap. It is larger, and it is not adopted

`attention_prefill` also called SDPA with no program config. Measured
(`probes/sdpa_prefill_confirm.py`, `logs/sdpa_prefill_confirm.log`):

| S | default | q128/k128 | q256/k256 |
|---|---|---|---|
| 128 | 23.92 us | 25.72 | 32.68 |
| 512 | 58.96 | **54.14** | 87.18 |
| 1024 | 230.58 | **88.36** | 127.54 |
| 2048 | 741.08 | 216.03 | **207.28** |
| 4096 | 2850.25 | 882.61 | **451.04** |
| 8192 | 10938.43 | 2907.58 | **1956.15** |
| 16384 | 44456.67 | 11364.22 | **6527.48** |

**Arbitrary S keeps working** — checked, not assumed: S in {1, 3, 31, 33, 100,
129, 255, 257, 1000, 1023, 1025, 2049, 4095, 4097, 5000} all build and run under
both chunkings with PCC identical to the default's to five decimals. Prefill is
not chunked in this model, so this property is load-bearing for the stage
contract. `q512/k512` is rejected: it fails to build at *every* length including
128, so it is a resource limit, not an alignment rule.

**Not adopted**, for two measured reasons:

1. it costs accuracy on the only gate that can see it — `run_teacher_forcing`
   top-1 **0.990 -> 0.980** (top-5 and top-100 stay 1.000), bisected to the
   prefill config specifically (`logs/run_teacher_forcing_leg_prefill.log` =
   0.980, `logs/run_teacher_forcing_leg_decode.log` = 0.990);
2. at the length being served it is a *loss*. The readiness reference prompt is
   **158 tokens**, below the S~384 crossover; measured TTFT 3448.79 ms baseline
   vs 3445.31 ms configured — noise.

It is left wired as a seam and fully documented. What it needs before adoption is
a readiness reference with a multi-thousand-token prompt, so the regime where it
pays is the regime the accuracy gate covers.

## 5. Results

**Token-out, before/after on the same tree** (`probes/perf_full_model.py
--layers 48 --gen-len 128 --context 8192 --tag ...`, 128 timed reps, median;
`logs/perf_p*_{before,after}.log`):

| prompt | token_out before | after | gain | t/s/u before -> after | model_trace before -> after | TTFT |
|---|---|---|---|---|---|---|
| 128 | 21.4776 ms | 20.1460 | **1.066x** | 46.56 -> **49.64** | 20.2319 -> 19.5604 | 125.88 -> 125.84 (unchanged) |
| 1024 | 26.1432 | 20.4268 | **1.280x** | 38.25 -> **48.96** | 24.8692 -> 19.8284 | 887.86 -> 887.68 |
| 4096 | 42.0623 | 20.9608 | **2.007x** | 23.77 -> **47.71** | 40.7693 -> 20.3417 | 3592.58 -> 3592.79 |

The row that matters is not any single gain, it is the shape: **after the change
token-out is flat in context** (20.15 / 20.43 / 20.96 ms) where before it grew
steeply (21.48 / 26.14 / 42.06). TTFT is unchanged at every length, which is the
expected signature of adopting decode and not prefill.

**PCC against the HF reference, in-model**, real multichip layer with a
prefill-primed paged cache (`probes/sdpa_hf_pcc_at_depth.py`):

| ctx / cur_pos | default | adopted | k_chunk |
|---|---|---|---|
| 128 / 127 | 0.999415 | 0.999431 | 128 (clamped) |
| 1024 / 1023 | 0.999356 | 0.999349 | 256 |
| 4096 / 4095 | 0.999534 | 0.999502 | 256 |
| 8192 / 8191 | 0.999405 | 0.999438 | 256 |
| 16384 / 16383 | 0.999294 | 0.999363 | 256 |

All far above the 0.995 layer bar; the two legs differ in the fifth decimal at
every depth. (32768 was attempted and the *host* was OOM-killed building the HF
reference, not the device.)

**Correctness and readiness gates:**

| gate | result |
|---|---|
| `pytest tests/ -m "not models_performance_bare_metal" -q` | **145 passed**, 16 deselected |
| the same under `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` | **145 passed, zero tripped asserts** (`logs/watcher_sdpa.log.gz`) |
| `run_prefill_check` | 0.980 / **1.000 / 1.000** — unchanged |
| `run_teacher_forcing` | 0.990 / **1.000 / 1.000** — unchanged; decode 40.47 -> 41.48 t/s/u |
| `run_autoregressive` + `check_degenerate_output --scope autoregressive` | **"No degenerate output detected"** |

## 6. Where part 1 was wrong, and by how much

| part 1 claim | what actually happened |
|---|---|
| `k512/c32` is the config | wrong dtype; at bfloat16 the winner is `k256/c16`, and `k512` is *unsafe in-model* |
| 1.18x @ ctx131, 15.2x @ 4095, 25.4x @ 8191 **on the op** | holds in direction; at bfloat16 and the safe config: 1.25x / 14.7x / 23.4x |
| "0.9% at the profiled ctx128" on token-out | **6.6%** measured (46.56 -> 49.64 t/s/u) — part 1 under-called it |
| "at ctx 4096 it is the whole model" | correct, and the strongest result here: **2.007x** on token-out |
| the default's PCC decays with depth (0.9932 @16k, 0.9897 @32k) | true of the **op on random caches**, and it **does not carry into the model**: in-model the default reads 0.9993 at cur_pos 16383. The accuracy argument for this lever does not survive contact with a real prefill-primed cache; the speed argument does. |

The last row is the honest summary of the gap between probe and model: on speed
the standalone probes were directionally right and slightly conservative, and on
accuracy they were measuring their own random inputs rather than the model.

---

# Stage 06, part 3 — the last two levers: the sampler `ArgMax`, and the MoE skew

Parts 1 and 2 left two items open. This part closes both: **one is adopted and
measured, the other is closed with evidence rather than with an attempt.**
Nothing was committed and nothing was pushed. Same machine, same 1x4 P300_X2 /
`FABRIC_1D_RING`, tree `8ea42a6b8ed` plus the uncommitted stage-06 `tt/model.py`
(distributed argmax) and `tt/multichip_decoder.py` (paged SDPA program config),
plus the change described here. Probes and logs are in `probes/` and `logs/`;
`doc/full_model/` was not written to and `doc/full_model/probes/perf_full_model.py`
was not run.

## Lever A — the sampler `ArgMax`, 366 us

### A1. The 75x is scalar compare throughput, not bandwidth and not the barrier

`ttnn.argmax`'s multicore path is **one reader kernel and no compute kernel**.
`argmax_multi_core_program_factory.cpp` places a single
`reader_argmax_interleaved_multicore.cpp` on `RISCV_1` — a *data-movement* RISC —
and the reduction itself is a C++ `for` loop calling `bfloat16_greater` one
element at a time. Nothing touches the FPU or the SFPU. At `[1,1,32,37984]` that
is 1.22M scalar comparisons split over 110 cores, ~11k each, and 366 us is simply
what that costs. The op is not 75x off *bandwidth*; it never had a bandwidth
problem to be off. This is why part 1's "two-stage reduction to get within 5x of
bandwidth" framing did not survive: there is no stage of it that is not the same
scalar loop.

Two structural predictions from the same source, both tested:

* the kernel's **outer** loop (`outer_dim_units`) carries a full 110-core
  semaphore barrier per iteration; its **inner** loop (`inner_dim_units`) carries
  none. With `keepdim=True` the 32 rows land on `outer` (32 barriers); with
  `keepdim=False` they land on `inner` (one). Measured: **371.1 -> 309.3 us**, so
  the 31 saved barriers are worth 62 us — real, and only 17% of the bill;
* if the cost is scalar work, it must scale as `1/cores`. `sub_core_grids` at
  8 / 16 / 32 cores reads 2735.5 / 1429.5 / 792.2 us; times core count that is
  21.9 / 22.9 / 25.4 / 40.8 core-ms, i.e. flat-ish work with a rising
  per-core-count overhead. Fewer cores is monotonically worse. Confirmed.

### A2. What was adopted: reduce the live user rows, not the padding

`decode_terminal` hands the sampler a **logically 32-row** logit tile because
`ttnn.sampling` addresses 32 fixed user slots. At batch 1, thirty-one of those
rows are the zero rows `ttnn.pad(..., value=0.0)` puts on the pre-head hidden,
and `lm_head` has no bias, so their logits are exactly zero. The reduction was
spending 31/32 of its time on padding.

`_sample_argmax` now slices the ROW_MAJOR tensor to the model's `max_batch_size`
rows before the argmax, runs the whole reduction at that width, and pads the
token vector back to 32 slots with `value=0` at the end.

**The substitution is exact, and it is checked on the device rather than
argued.** `argmax_outer_dim_probe.py`'s `padding_rows_produce_token_zero` leg
runs the *shipped* 32-row reduction on logits whose rows 1..31 are zero and reads
back token **0** in every one of them — all four dies tie at 0.0, so the masked
`min` keeps global index 0 — which is precisely the value the pad writes. Slot
for slot the buffer is unchanged. That also keeps every slot a valid token id,
which matters because `embed_decode` runs `ttnn.embedding` over all 32 before
slicing to `batch`, and an out-of-vocabulary id there is an out-of-bounds table
read.

Standalone, at the shipped shape, trace-captured, median of 60
(`probes/argmax_outer_dim_probe.py`, `logs/argmax_outer_dim_probe_b.log`). **The
harness floor is ~58 us** — `pad_token_1_to_32`, which is a 1-element pad, reads
57.9 — so subtract that before reading anything as an op cost:

| leg | wall | minus floor |
|---|---|---|
| `untilize` 37984 | 75.1 us | 17.2 (matches the 16.93 in-model profile row) |
| `argmax`, 32 rows, `keepdim=True` (shipped) | 371.1 | 313.2 |
| `argmax`, 32 rows, `keepdim=False` | 309.3 | 251.4 |
| `argmax`, 32 rows, `keepdim=False` + reshape | 315.4 | 257.5 |
| ROW_MAJOR slice to 1 row, alone | 58.4 | 0.5 |
| ROW_MAJOR slice to 1 row + `argmax` | **58.0** | **~0** |
| same at 2 / 4 / 8 rows | 58.2 / 65.5 / 122.8 | ~0 / 7.6 / 64.9 |
| **whole reduction, 32 rows (shipped)** | **631.6** | 573.7 |
| **whole reduction, 1 row (adopted)** | **250.8** | 192.9 |

**2.52x on the whole sampler**, 2.97x floor-corrected. All three crafted-tie
cases (cross-die, within-die, triple) still return the first-maximal index, at
both `keepdim` settings.

### A3. In-model, before and after

`probes/perf_full_model.py --layers 48 --gen-len 128 --context 8192`, 128 timed
reps, median (`logs/perf_p*_argmaxrows.log`, `probes/perf_full_model_p*_argmaxrows.json`).
"Before" is part 2's adopted tree.

| prompt | token_out before | after | gain | t/s/u | model_trace | TTFT |
|---|---|---|---|---|---|---|
| 128 | 20.1460 ms | **19.6925** | 0.4535 ms | 49.64 -> **50.78** | 19.5604 -> 19.5667 | 125.84 -> 125.43 |
| 1024 | 20.4268 | **19.9787** | 0.4480 | 48.96 -> **50.05** | 19.8284 -> 19.8338 | 887.68 -> 887.99 |
| 4096 | 20.9608 | **20.5050** | 0.4557 | 47.71 -> **48.77** | 20.3417 -> 20.3374 | 3592.79 -> 3593.05 |

The signature is exactly right for a change confined to the sampling trace:
**a flat 0.45 ms at every depth**, `model_trace` unmoved to within noise at all
three, and TTFT unmoved. Nothing regressed at any depth. Part 2's "token-out is
flat in context" property is preserved and improved: 19.69 / 19.98 / 20.51.

### A4. The rejection ledger for this lever

| candidate | measured | why not |
|---|---|---|
| `keepdim=False` (32 barriers -> 1) | 309.3 vs 371.1 us; **251.0 vs 250.8 on the full reduction once rows are sliced** | Real on its own and worth 62 us, but it buys **nothing** on top of the row slice, and it costs a `[1,1,B] -> [1,1,B,1]` reshape the rest of the reduction needs. Not adopted; the row slice subsumes it. It is the right lever for a future `max_batch_size=32` deployment, where the row slice is a no-op — recorded here for that case. |
| `sub_core_grids` at 8 / 16 / 32 cores | 2735.5 / 1429.5 / 792.2 us | Monotonically worse than the default 110. The cost is scalar work, so fewer cores is strictly more time. |
| `sub_core_grids` at 64 cores | **hung the device** | Not a perf result, an upstream bug — see A5. |
| `ttnn.topk(k=1)` on the ROW_MAJOR tensor | does not build | `topk_device_operation.cpp:166` requires TILE layout, and the multicore argmax requires ROW_MAJOR. The two are mutually exclusive at this shape. |
| `ttnn.topk(k=32)` on the TILE tensor | 6047.3 us | 16x the shipped argmax. |
| `argmax` on the TILE tensor (skip the untilize) | 23252 us, from part 1's probe | Single-core path. Unchanged. |
| TILE-layout slice to 1 row, then untilize | 71.5 us vs 75.1 for the full untilize | 3.6 us: a TILE slice on a non-tile-aligned height still touches all 1187 tiles. The untilize stays where it is and the slice happens in ROW_MAJOR after it, at ~0.5 us. |
| two-stage reduction via a tile-aligned reshape (part 1's recommendation) | not attempted | 37984 = 2^5 x 1187 with 1187 prime, so the only tile-aligned reshape is `[1,1,37984,32]`, whose 32-wide reduction dimension gives the multicore factory 2 cores. And the premise was wrong anyway: the op is scalar-compare-bound, not bandwidth-bound, so a "get within 5x of bandwidth" target does not exist. |

### A5. An upstream bug found on the way

`sub_core_grids` with **two** core ranges whose total padded work exceeds the
tensor by more than one core's share **hangs the device**.
`argmax_multi_core_program_factory.cpp` computes

    red_dim_units_last1 = red_dim_units1 - (ideal_red_dim_units - red_dim_units)

in `uint32_t`. At 64 cores on this 13-wide grid that is `608 - 928`, which wraps
to 4294966976; the reader then issues an unbounded NOC read and never returns.
8 / 16 / 32 cores are safe only because their remainder happens to be smaller
than one core's share. Reproduced once (board reset required) and then removed
from the sweep rather than re-run; the guard is documented in the probe. This is
not on the model's path — the model does not pass `sub_core_grids` — but it is a
real defect and should be filed.

## Lever B — the MoE reduce-scatter skew: closed, with the arithmetic

**No change was made and none should be.** The analysis is
`probes/moe_skew_analysis.py` / `.json` / `logs/moe_skew_analysis.log`; it is
pure analysis of the archived 48-layer profile, opens no device, and re-runs in
under a second.

### B1. The per-die active-expert count is recoverable, and the recovery validates itself

The gate/up `SparseMatmul` duration quantises exactly:

    t = 29.38 us + 6.85 us * k

`k` is the number of this die's 32 local experts that the global top-8 selected.
The 29.38 us floor is the `nnz=None` dynamic-sparsity scan that EP forces (32
slots x 0.79 us; `multichip_decoder.py`'s "``nnz`` contract" section), paid
whether or not anything fires. **Rounding `(t - 29.38)/6.85` and summing over the
four dies gives exactly 8 in all 48 layers** — the router's top-8. A wrong step
would not do that, so the recovered counts are not a fit, they are a measurement.

### B2. The MoE reduce-scatter is not a collective, it is a queue

Regress each die's MoE-side reduce-scatter on how far behind the slowest die it
finished its experts in that layer:

| | mean | min | max | corr with lag | slope |
|---|---|---|---|---|---|
| attention-side `ReduceScatter` | 12.49 us | 9.83 | 15.276 | **0.092** | — |
| MoE-side `ReduceScatter` | 24.32 | 6.39 | 74.19 | **0.989** | **1.051 us per us** |
| both `AllGatherAsync` | 11.67 / 12.45 | | | | |

0.989 with a slope of 1.05 is not a correlation, it is an identity: every
microsecond a die finishes early is a microsecond it stands at the collective.
The attention-side reduce-scatter, at the same 128 KB on the same ring, does not
see it at all. And on layer 10, where the four dies happened to balance to within
0.41 us of work, the MoE reduce-scatter reads **11.72 / 11.82 / 13.06 / 13.21 us**
on the four dies — the attention-side figure. **The collective is already at its
floor.** Topology, link count and persistent buffers are not the lever, which is
what part 1's lever 6 said and this confirms independently.

### B3. The skew is combinatorial, and the permutation lever looked like zero here — it is not zero, see the correction below

Under EP=4 the top-8 lands in four windows of 32. If the router selected
uniformly across dies, the per-die counts would be `multinomial(8, 1/4)`:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| measured (192 die-layers) | 18 | 45 | 70 | 45 | 9 | 2 | 3 |
| uniform expectation | 19.2 | 51.3 | 59.8 | 39.9 | 16.6 | 4.4 | 0.7 |

chi-square **8.06** on 6 bins — the routing is statistically indistinguishable
from uniform across the shipped contiguous windows.

That settles the permutation question without needing to build one. A collective
waits for the **maximum**, not the mean, and for `multinomial(8, 1/4)` the
expected maximum is **3.538** against a mean of 2 — 1.538 experts, ~10.5 us a
layer, of imbalance that exists no matter how the experts are labelled. The
measured mean maximum is **3.417**.

> **Correction, written after the stage-06 review.** The paragraph that stood
> here read 3.417 against 3.538 as "the shipped contiguous assignment is already
> 0.83 us/layer ahead of an arbitrarily chosen partition, so a permutation is a
> loss in expectation". That is noise: the standard error of that mean over 48
> layers is ~0.12, so the gap is under one standard error (z ≈ −0.8). The
> chi-square above also dropped its sparsest bin rather than pooling it, and the
> bin it dropped (k=6, expected 0.74) had **3** observations in it. And the whole
> sample is 48 layers of one decode token. The conclusion drawn here is
> withdrawn; see the README's "The MoE-skew rejection is withdrawn" and
> `probes/moe_routing_across_tokens_probe.py`, which measures 128 tokens and
> finds a permutation worth 0.173 ms/iteration on held-out tokens. What survives
> from this section is the *arithmetic*: a collective waits for the maximum, the
> unavoidable floor under uniform routing is 1.538 experts of imbalance, and a
> permutation would carry a numerical-identity obligation (permute the expert
> weights *and* `_expert_window_matrix` together, and prove it).

### B4. The budget, and what a real fix would have to cost

| | ms/iteration |
|---|---|
| MoE-side reduce-scatter, total | 1.168 |
| attention-side reduce-scatter, total | 0.600 |
| MoE-side excess over the attention-side baseline | **0.568** |
| idle attributable to measured skew (`(mean max_k - 2) x 6.85 us x 48`) | **0.466** |
| the same under *perfectly uniform* routing — the floor | 0.506 |

So of the ~0.57 ms the MoE collective spends above the attention one, 0.466 ms is
skew idle and the residual is the collective's own variance. **The achievable
saving is not 0.47 ms, it is 0 ms**, because the 0.466 measured is already below
the 0.506 that uniform routing implies.

Three things were considered and none of them is available:

* **overlap the imbalance.** The MoE reduce-scatter is a data dependency on the
  reduced expert output; it cannot start before the slowest die's experts finish.
  Moving the wait somewhere else does not remove it — the dies must rendezvous
  once per layer to keep the residual replicated, and the residual layout is a
  contract this stage must preserve;
* **merge the two collectives.** They are separated by the router and both
  RMSNorms; merging them would mean one all-reduce per layer instead of two,
  which changes what the second RMSNorm reads. Not a scheduling change, a
  different model;
* **make per-die cost independent of `k`.** This is the only thing that actually
  removes the skew, and under the SPMD `nnz=None` contract it means computing all
  32 local experts every layer: **16x the expert FLOPs (128 expert-slots instead
  of 8) to recover 0.47 ms of a 20 ms iteration.**

The one genuinely addressable number in the neighbourhood is the **6.85 us per
active expert** slope and the **29.38 us** dynamic-`nnz` floor — that is part 1's
lever 3, and it is still blocked on TTNN's `Tile([1,32])` gap.

## Results

**Token-out, part 2 -> part 3, same tree:**

| prompt | part 2 | part 3 | t/s/u |
|---|---|---|---|
| 128 | 20.1460 ms | **19.6925** | 49.64 -> **50.78** |
| 1024 | 20.4268 | **19.9787** | 48.96 -> **50.05** |
| 4096 | 20.9608 | **20.5050** | 47.71 -> **48.77** |

**Correctness and readiness gates**, all re-run on this tree:

| gate | result |
|---|---|
| `pytest tests/ -m "not models_performance_bare_metal" -q` | **145 passed**, 16 deselected (`logs/pytest_argmax_rows.log`) |
| the same under `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` | **145 passed, zero tripped asserts** (`logs/watcher_argmaxrows.log.gz`) |
| `run_prefill_check` | **0.980 / 1.000 / 1.000** — unchanged (`logs/run_prefill_check_argmaxrows.log`) |
| `run_teacher_forcing` | **0.990 / 1.000 / 1.000** — unchanged; decode 41.48 -> **42.25 t/s/u** (`logs/run_teacher_forcing_argmaxrows.log`) |
| `run_autoregressive` + `check_degenerate_output --scope autoregressive` | **"No degenerate output detected"** (`logs/check_degenerate_argmaxrows.log`) |

Policy preserved: dtype, fidelity, KV-cache layout, activation memory configs,
CCL policy (`Topology.Ring`, no `num_workers_per_link` pinned) and the
inter-layer residual layout are untouched, non-aligned prompt lengths still work
(the 145-test suite covers them), and `Topology::Linear` + `num_workers_per_link=1`
was not introduced. The one behavioural seam added is
`_WatcherCleanSampling1D._dist_active_rows`, which defaults to `None` and in that
state reproduces the previous code exactly.

## What is left

The sampler is now **250.8 us standalone**, and ~141 us of that is part 1's
**lever 6**: the two 4-wide all-gathers that fall off `AllGatherAsync` onto a
composite `AllBroadcast` + `UntilizeWithUnpadding` x9 + `Permute` x10 + `Concat`
+ `TilizeWithValPadding` path. That is now the largest item in the terminal
block and the obvious next target, at ~0.55% of token-out — but it is the path
that hung the mesh for twenty minutes at stage 05, so it needs the watcher A/B
first. It was out of scope here.

Inside the layer, nothing changed and nothing should: part 1's conclusion stands
that stage 04 tuned it well, and lever B above now shows the remaining collective
cost is queueing on a combinatorial imbalance rather than anything the collective
or the layout can fix.
