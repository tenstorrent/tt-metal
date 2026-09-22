# Per-op tables: Gemma4 prefill at chunk 2048 / 4096 / 8192

> **Base note.** Every number in this file is on the **pre-L1-activations** base.
> For the same measurements on `mmanzoor/svuckovic/gemma4-L1-activations` @ `9dc8e32a2`
> (Asif's branch, PR #56862) see [`L1Activations/`](L1Activations/). **Do not compare
> numbers across the two** — re-render both sides on one build instead.

Generated with `tt-perf-report` (v1.2.9) over Tracy captures of the in-tree isolated-layer
benchmark. This is the per-op, per-core view of the two effects described in
[`README.md`](README.md), with one op per row and chunk sizes side by side.

BH Galaxy, mesh 8x4 (CP8/TP4), branch `kmabee/gemma4-swa-multihop-halo`.
Gemma4-31B: 60 layers = 50 sliding (`local`) + 10 full attention (`global`).

---

## TL;DR

1. **Exactly one op in the whole model grows with context: the global layers'
   `RingJointSDPA`.** It accounts for **99.8-100.0%** of the measured growth of a global
   layer between 0 and 49152 tokens of prior KV. Nothing else in either layer type moves.

2. **The sliding layer is context-invariant to within 0.8%.** Same layer, prior context 0 vs
   49152 tokens: **+0.2% / +0.8% / -0.1%** at chunk 2048 / 4096 / 8192. Its SDPA is
   405 / 454 / 461 us and does not move with depth. This is the direct per-op confirmation
   of the SWA question.

3. **Chunk 2048 does a quarter of the prefix work but spends 44.3% of the time** -- a
   **1.77x** efficiency loss. Pre-registered prediction was 0.50x for the work-unit
   occupancy model and 0.25x for an efficiency-neutral op; measured **0.443x**.
   Doubling 2048 -> 4096 costs only **4.6%** more SDPA time, because both are depth-1.

4. **`Cores` does not reveal this.** The global SDPA reports **114 cores at every chunk
   size**, including chunk 2048 where only 32 of ~110 cores have a work unit -- the rest run
   padded handshake iterations rather than being dropped from the grid. Occupancy has to be
   computed from the op's own work-unit math (table below), not read off the report.

5. **No regression from the multi-hop halo at chunk 8192** (the 1-hop path that predates it):
   every op within 2% of the pre-halo branch, sliding layer ~4% faster. Checked by
   re-rendering both branches' captures with one tool -- see
   [Regression check](#regression-check-the-multi-hop-halo-vs-the-pre-halo-branch). Chunk
   4096 (2 hops) has no pre-halo baseline and is the remaining gap.

6. **`Total %` and `Op-to-Op Gap` are unusable on these captures.** Each replay's first op
   carries a host-side gap measured from before the signpost (1 121 464 us in one case),
   which swamps the percentage column -- the 8.27 ms SDPA is shown as `0.7 %`. Every other
   gap is ~1 us. Use `Device Time`.

---

## How to generate these

Two corrections to the commands that were circulating: the demo moved to `gemma4_d_p`, and
the sliding layer's signpost is named **`local`**, not `sliding`. The chunk index is a test
parameter now, not the `GEMMA4_PERF_CHUNK_IDX` env var. `layer_type=both` measures one
global and one local layer in a single device session.

```bash
# Capture. ~12 min per run; writes ~5 GB of raw profiler logs under $OUT/profiler/.logs/.
# Do NOT add --device-trace-profiler: it profiles only trace regions and breaks attribution.
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
python -m tracy -r -p -v -o $OUT/profiler -m pytest \
  "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk6-both-sz8192-ctx_256k-8x4]" -sv

# Render, per layer type. --csv diverts the table out of stdout, so the human-readable
# table needs its own invocation.
tt-perf-report --no-color --no-summary \
  --start-signpost gemma4-layer-global-chunk6-start \
  --end-signpost   gemma4-layer-global-chunk6-stop \
  $OUT/profiler/reports/*/ops_perf_results_*.csv
```

**Hold the prior context fixed, not the chunk index.** Comparing chunk index 7 across chunk
sizes compares different amounts of prior KV, which confounds the thing being measured. All
tables below use a prior context of **49152 tokens**: `sz8192` idx 6, `sz4096` idx 12,
`sz2048` idx 24, each confirmed by the harness' own `kv_actual_global=49152`.

---

## 1. Chunk index 0 -- the per-chunk floor, no prefix

Sum of device time per op, one layer. A 4x smaller chunk should cost 0.25x if everything
scaled with the token count. The `2048/8192` column is how far each op misses that.

```
=== global layer ===
op                                        n     2048 us    4096 us    8192 us   cores   2048/8192
-----------------------------------------------------------------------------------------------------
RingJointSDPADeviceOperation              1       189.0      357.3     1316.2     114       0.14x
MatmulDeviceOperation _x5376x5376         3       582.3      885.8      862.1   96,84       0.68x
LayerNormDeviceOperation                  6       469.1      484.2      610.0  120..8       0.77x
AllGatherDeviceOperation                  3       159.9      314.3      563.7 34,10,2       0.28x
ReduceScatterDeviceOperation              2       144.7      258.2      495.6   34,18       0.29x
GatherCodegenDeviceOperation              5       176.9      213.6      367.5 120..32       0.48x
BinaryNgDeviceOperation                   6        99.0      189.2      361.4     120       0.27x
MatmulDeviceOperation _x5376x4608         1       185.5      283.0      248.1   96,72       0.75x
MatmulDeviceOperation _x4096x5376         1       152.4      233.4      279.6   96,84       0.55x
SliceDeviceOperation                      5       220.4      236.7      263.6 120..32       0.84x
TilizeDeviceOperation                     1        56.7       90.2      122.0      84       0.46x
NlpCreateHeadsDeviceOperation             1        70.6       72.7       77.1  32,16,8      0.92x
UnaryDeviceOperation                      1        20.9       37.0       70.6     120       0.30x
NLPConcatHeadsDeviceOperation             1        57.3       59.2       63.8  32,16,8      0.90x
-----------------------------------------------------------------------------------------------------
TOTAL                                           2669.1     3822.2     5857.6               0.46x
  per 1k tokens                                 1334.6      955.5      732.2

=== local (sliding) layer ===
MatmulDeviceOperation _x5376x5376         3       582.0      885.8      862.9   96,84       0.67x
LayerNormDeviceOperation                  7       467.9      481.3      612.0  120..8       0.76x
AllGatherDeviceOperation                  3       158.3      312.0      562.8 34,10,2       0.28x
ReduceScatterDeviceOperation              2       144.0      257.3      489.7   34,18       0.29x
RingJointSDPADeviceOperation              1       404.0      421.3      455.1 114,112       0.89x
BinaryNgDeviceOperation                   5        96.2      186.1      352.1     120       0.27x
-----------------------------------------------------------------------------------------------------
TOTAL                                           2430.3     3241.9     4362.3               0.56x
  per 1k tokens                                 1215.2      810.5      545.3
```

**What to read.** The collectives and the elementwise ops scale essentially perfectly
(0.27-0.29x against an ideal 0.25x). The floor is:

| op | 2048/8192 | why |
|---|---|---|
| `LayerNormDeviceOperation` | **0.77x** | runs on **8-32 cores**, visible in the `Cores` column. `ttnn.rms_norm` parallelises over rows only, so a 256-row slab is 8 tiles = 8 of 120 cores |
| `MatmulDeviceOperation _x5376x5376` | **0.68x** | weight-read bound at small M: the layer's weights are read once per chunk whatever the token count |
| `NlpCreateHeads` / `NLPConcatHeads` | 0.90-0.92x | 32 cores, essentially fixed cost |
| sliding `RingJointSDPA` | **0.89x** | the halo is a constant 1024 tokens regardless of chunk size |

Fitting `cost = F + k*tokens` per op over the three chunk sizes gives a fixed part of
**1.94 ms** per global layer and **1.87 ms** per local layer; 10x + 50x = **112.9 ms**, of
which the **50 local layers are 82.8%**.

> **Correction (2026-09-18): that 1.94 ms carries a clamp, and the absolute total is
> estimator-dependent.** The fit clamps each op's intercept at zero (`max(0.0, F)` in
> `cmp_ops.py`). It fires on exactly **one** op -- the global layer's `RingJointSDPA`, whose
> least-squares intercept is **-290.4 us** -- and inflates the global per-layer fixed cost
> from 1651 to 1942 us (**+17.6%**). The negative intercept is not noise: at chunk index 0
> that op's cost is roughly **quadratic** in the chunk (causal self-attention over the chunk
> itself, ~C^2/CP), so an affine-in-C model is misspecified for it and the clamp hides that
> rather than fixing it. Unclamped, the global layer's fixed cost is 1.65 ms and the local
> share rises to **85.0%**.
>
> Three defensible estimators of the **floor share** give **82.8% / 84.8% / 85.0%** -- the
> share is robust, and its robustness is the actual evidence. The same three estimators give
> **112.9 / 79.0 / 110 ms** for the **absolute** floor, a 43% spread, so they do *not* agree
> there and should not be quoted as if they do. **Prefer the whole-model affine fit (94.2 ms)
> for the absolute figure**: it is anchored on end-to-end per-chunk device times, whereas
> every per-op sum inherits the isolated-layer harness's missing inter-layer overlap and its
> staging ops. Found by the `prefill-perf-debug` skill's validation pass.

## 2. Prior context 49152 tokens -- the prefix cost, active

Same layers, same chunk sizes, now with real prior KV.

```
=== global layer ===
op                                        n     2048 us    4096 us    8192 us   cores   2048/8192
-----------------------------------------------------------------------------------------------------
RingJointSDPADeviceOperation              1      3661.9     3831.2     8266.0     114       0.44x
MatmulDeviceOperation _x5376x5376         3       581.7      885.9      863.9   96,84       0.67x
LayerNormDeviceOperation                  6       467.9      483.9      608.3  120..8       0.77x
AllGatherDeviceOperation                  3       160.7      313.7      564.8 34,10,2       0.28x
ReduceScatterDeviceOperation              2       146.9      262.8      515.8   34,18       0.28x
-----------------------------------------------------------------------------------------------------
TOTAL                                           6141.9     7297.4    12822.1               0.48x

=== local (sliding) layer ===
MatmulDeviceOperation _x5376x5376         3       582.5      886.0      862.0   96,84       0.68x
LayerNormDeviceOperation                  7       467.6      481.8      610.2  120..8       0.77x
RingJointSDPADeviceOperation              1       404.7      453.6      460.6 114,112       0.88x
-----------------------------------------------------------------------------------------------------
TOTAL                                           2435.4     3269.1     4357.7               0.56x
```

### Only one op moves with context

| | chunk | layer total at ctx 0 | at ctx 49152 | delta |
|---|---:|---:|---:|---:|
| **local** | 2048 | 2430 us | 2435 us | **+0.2%** |
| **local** | 4096 | 3242 us | 3269 us | **+0.8%** |
| **local** | 8192 | 4362 us | 4358 us | **-0.1%** |
| global | 2048 | 2669 us | 6142 us | +130% |
| global | 4096 | 3822 us | 7297 us | +91% |
| global | 8192 | 5858 us | 12822 us | +119% |

And all of the global growth is one op:

| chunk | global layer delta | `RingJointSDPA` delta | share |
|---:|---:|---:|---:|
| 2048 | 3473 us | 3473 us | **100.0%** |
| 4096 | 3475 us | 3474 us | **100.0%** |
| 8192 | 6964 us | 6950 us | **99.8%** |

### The efficiency loss at small chunks

At a fixed prior context the prefix work is proportional to the chunk's token count, so
chunk 2048 does exactly **0.25x** the work of chunk 8192. It spends **0.443x** the time.

| chunk | prefix work | SDPA time | vs 8192 | efficiency vs 8192 |
|---:|---:|---:|---:|---:|
| 2048 | 0.25x | 3662 us | **0.443x** | **1.77x worse** |
| 4096 | 0.50x | 3831 us | 0.463x | 0.93x (better) |
| 8192 | 1.00x | 8266 us | 1.0 | 1.0 |

2048 and 4096 differ by only **4.6%** despite 4096 doing twice the prefix work. Both are
depth-1 in the op's work-unit math; 8192 is depth-2.

`ring_joint_sdpa_program_factory.cpp` splits work into `q_chunk_size`-row x one-head units,
`div_up`s them over the grid, and **does not skip cores without a unit** -- they run padded
handshake iterations, so every core loops `max_q_per_core` times:

| chunk | q chunks | work units | depth | slots | useful | **`Cores` reports** |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 4 | 32 | 1 | 110 | **29%** | 114 |
| 4096 | 8 | 64 | 1 | 110 | 58% | 114 |
| 8192 | 16 | 128 | 2 | 220 | 58% | 114 |
| 16384 | 32 | 256 | 3 | 330 | 78% | 114 |
| 32768 | 64 | 512 | 5 | 550 | 93% | 114 |

(`NH` = 8 local q heads = 32/TP4, `q_chunk_size` = 64, SDPA grid ~110 cores.)

The pre-registered discriminating test: the occupancy model predicts cost proportional to
depth, so **0.50x** for 2048-vs-8192; an efficiency-neutral op predicts **0.25x**. Measured
**0.443x**. The 2048-vs-4096 pair cannot discriminate -- both models predict 0.5 there --
which is why the comparison has to include 2048.

### Reconstructing the whole-model prefix slope from this one op

Taking the global SDPA as linear in prior tokens and multiplying by 10 global layers:

| chunk | SDPA @49152 | implied slope, ms/chunk-index | measured whole-model slope | ratio |
|---:|---:|---:|---:|---:|
| 2048 | 3662 us | 1.526 | 1.458 | 1.05x |
| 4096 | 3831 us | 3.193 | 2.995 | 1.07x |
| 8192 | 8266 us | 13.777 | 11.990 | 1.15x |

One op in 10 of 60 layers accounts for 87-95% of the whole-model prefix term. The residual
is the isolated-layer benchmark's lack of inter-layer overlap.

---

## Artifacts

On `bh-glx-120-b03u02`, under `/data/kmabee/gemma4_runs/`:

| path | contents |
|---|---|
| `perf_reports/COMPARE_chunk0.txt` | the chunk-0 comparison above, full op list + `F`/`k` fit |
| `perf_reports/COMPARE_deep.txt` | the prior-context-49152 comparison, full op list |
| `perf_reports/{floor_c*,deep_c*}_{global,local}.txt` | the 12 raw `tt-perf-report` tables |
| `perf_reports/*.summary.png` | `tt-perf-report` stacked summary plots, grouped by category |
| `floor_c*/`, `deep_c*/` | the Tracy captures (`ops_perf_results_*.csv`, ~37 MB each) |
| `cmp_ops.py`, `mk_reports.sh`, `run_deepidx.sh` | the comparison and render scripts |
| `PREDICTION.md` | the prediction recorded before the deep captures completed |

---

## Corrections to [`README.md`](README.md)

- The "**~70 ms floor**" is the per-layer *excess over ideal token scaling* at chunk 2048
  (`2.181 - 4.029/4 = 1.174 ms`, x60). That equals **three quarters** of the chunk-invariant
  cost, not the cost itself. The chunk-invariant cost is **~94 ms** (whole-model affine fit
  over 2048-8192: 94.2 ms; per-op fit: 105-113 ms). The 2.16x / 1.99x / 2.09x conclusions
  are measured directly from the per-chunk device times and do not depend on this number.
- The README attributes the prefix term to the global layers from curve fits. It is now
  confirmed per-op: 99.8-100.0% of a global layer's context growth is `RingJointSDPA`, and
  the sliding layer is flat to within 0.8%.

---

## Regression check: the multi-hop halo vs the pre-halo branch

The multi-hop sliding halo on `kmabee/gemma4-swa-multihop-halo` changed the sliding path in
order to make small chunks runnable. That path is also used by the chunk sizes that already
worked, so it needs a standing check that it did not cost them anything. **Checked
2026-09-17 at chunk 8192: no regression.**

The 2026-09-10 captures on `svuckovic/gemma4-prefill-model @ d3064a5fd6b` are still on
`bh-glx-120-b03u02` at `/data/kmabee/gemma4_runs/attn_op_captures/{isl0k,isl48k,...}/`, so
both branches can be re-rendered with one tool. Identical tooling, `tt-perf-report` 1.2.9,
sum of `Device Time`, cross-device merge:

| layer | idx | pre-halo layer | multi-hop layer | pre-halo SDPA | multi-hop SDPA | SDPA delta |
|---|---:|---:|---:|---:|---:|---:|
| local | 0 | 4.549 ms | **4.362 ms** | 0.447 ms | 0.455 ms | +1.9% |
| local | 6 | 4.555 ms | **4.358 ms** | 0.459 ms | 0.461 ms | +0.3% |
| global | 0 | 5.944 ms | 5.858 ms | 1.313 ms | 1.316 ms | +0.2% |
| global | 6 | 12.916 ms | 12.822 ms | 8.273 ms | 8.266 ms | -0.1% |

Every op is within 2%, and the sliding **layer** is ~4% *faster* on the multi-hop branch.

### Why this is the right configuration to check

`sliding_halo_hop_count` (`models/demos/gemma4_d_p/tt/model.py:152`) is
`ceil(window / (chunk/CP))`, window 1024, CP 8:

| chunk | tokens/rank | halo hops | status |
|---:|---:|---:|---|
| 2048 | 256 | 4 | new capability — nothing to regress |
| 4096 | 512 | 2 | **not verified** |
| 8192 | 1024 | **1** | **verified, idx 0 and 6** |
| 16384 | 2048 | 1 | same 1-hop path as 8192 |
| 32768 | 4096 | 1 | same 1-hop path as 8192 |

Chunk 8192 takes the **single-hop** path, which is the code the pre-halo branch also ran, so
this is the configuration where a regression could actually appear. 16384 and 32768 take the
same 1-hop path. The multi-hop chunk sizes are new capability. **Chunk 4096 (2 hops) is the
gap**: if it ran before the change, it is unverified — there is no pre-halo capture of it.

### A retracted observation, and the trap behind it

An earlier reading of this comparison reported the sliding SDPA moving **0.38 -> 0.46 ms
(+21%)** at chunk 8192 idx 0 and guessed the multi-hop halo had cost the sliding path. That
was wrong: it compared the gist's published table value against a `tt-perf-report` number.
Re-deriving the *same capture* with `tt-perf-report` gives **0.447 ms**, not 0.38 — a +1.9%
delta, not +21%.

The per-device spread is what makes this easy to get wrong. For the sliding SDPA in
`isl0k`, across the 32 devices:

```
min 382.9 us | median 434.5 us | device 0 394.0 us | max 446.8 us      spread 17%
```

`tt-perf-report` merges devices and reports the **slowest**. The gist's 0.38 is close to the
minimum. Its deeper values cannot be reproduced from its own CSV by any per-device
aggregation of `DEVICE KERNEL DURATION` at all -- its `isl48k` figure of 480 us exceeds the
**max** across all 32 devices (459.4 us) -- so that column's aggregation differs from this
one in a way that was not pinned down.

**The rule: a cross-branch per-op comparison is only valid if both sides are re-rendered by
the same tool from the raw captures.** The ~17% device spread here is the same size as the
effects being chased, so a figure lifted from someone's summary table can manufacture or
hide a regression on its own. This is the same trap as the earlier `rms_norm` retraction
(§4 of [`CHUNK_SIZE_ANATOMY.md`](CHUNK_SIZE_ANATOMY.md)), which compared per-op timings
across two builds -- except that here, re-rendering showed the hardware had not changed at
all.

### To re-run this check

```bash
# Render both branches' captures with the same tool, then diff.
for pair in isl0k:0 isl48k:6; do d=${pair%%:*}; i=${pair##*:}
  for lt in global local; do
    tt-perf-report --no-color --no-summary --csv OLD_${d}_${lt}.csv \
      --start-signpost gemma4-layer-${lt}-chunk${i}-start \
      --end-signpost   gemma4-layer-${lt}-chunk${i}-stop \
      /data/kmabee/gemma4_runs/attn_op_captures/$d/ops_perf_results_*.csv
  done
done
# Compare against the current branch's floor_c8192 / deep_c8192_i6 captures the same way.
```

Worth redoing whenever the sliding halo, `ring_joint_sdpa`, or the sliding program config
changes. The pre-halo captures are the only baseline for it, so they should not be deleted.

---

## Opening these captures yourself

Everything is on `bh-glx-120-b03u02` under `/data/kmabee/gemma4_runs/`. `tt-perf-report` is
already installed there (`pip install tt-perf-report`; v1.2.9 at `~/.local/bin`).

| tag | chunk | chunk_idx | prior KV | pytest node id |
|---|---:|---:|---:|---|
| `floor_c2048` | 2048 | 0 | 0 | `[blackhole-chunk0-both-sz2048-ctx_256k-8x4]` |
| `floor_c4096` | 4096 | 0 | 0 | `[blackhole-chunk0-both-sz4096-ctx_256k-8x4]` |
| `floor_c8192` | 8192 | 0 | 0 | `[blackhole-chunk0-both-sz8192-ctx_256k-8x4]` |
| `deep_c2048_i24` | 2048 | 24 | 49152 | `[blackhole-chunk24-both-sz2048-ctx_256k-8x4]` |
| `deep_c4096_i12` | 4096 | 12 | 49152 | `[blackhole-chunk12-both-sz4096-ctx_256k-8x4]` |
| `deep_c8192_i6` | 8192 | 6 | 49152 | `[blackhole-chunk6-both-sz8192-ctx_256k-8x4]` |

Each `<tag>/profiler/reports/<timestamp>/` holds:

| file | size | what |
|---|---:|---|
| `ops_perf_results_<ts>.csv` | ~37 MB | **what `tt-perf-report` reads** |
| `tracy_profile_log_host.tracy` | ~420 MB | open in the Tracy GUI |
| `profile_log_device.csv` | ~4.8 GB | raw device rows; the ops report already distils it, safe to delete |

### One-liner

`view.sh <tag> <global|local>` resolves the report path and the signposts for you:

```bash
cd /data/kmabee/gemma4_runs
./view.sh                              # list the captures
./view.sh deep_c2048_i24 global        # the decisive one: 3662 us SDPA, 114 cores
./view.sh deep_c8192_i6  global        # compare: 8266 us for 4x the prefix work
./view.sh floor_c2048    local         # the floor: LayerNorm on 8-32 cores
./view.sh deep_c8192_i6  local --no-advice
```

Extra arguments pass through to `tt-perf-report`, e.g. `--no-summary`, `--no-advice`,
`--csv out.csv`, `--group-by category`.

### Or by hand

```bash
tt-perf-report \
  --start-signpost gemma4-layer-global-chunk24-start \
  --end-signpost   gemma4-layer-global-chunk24-stop \
  /data/kmabee/gemma4_runs/deep_c2048_i24/profiler/reports/*/ops_perf_results_*.csv
```

The signpost name is `gemma4-layer-{global|local}-chunk{chunk_idx}-{start|stop}`, where
`chunk_idx` is the one in the table above — **not** the chunk size, and the sliding layer is
`local`.

### Regenerating a capture from scratch

`run_deepidx.sh` (in `/data/kmabee/gemma4_runs/`) is the driver that produced the three deep
captures; `mk_reports.sh` renders a capture's two layer types to files, and `cmp_ops.py`
builds the side-by-side tables in this document. A single capture directly:

```bash
source /data/kmabee/gemma4_runs/env.sh && cd $TT_METAL_HOME
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
python_env/bin/python3 -m tracy -r -p -v -o /tmp/mycapture/profiler -m pytest \
  "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk24-both-sz2048-ctx_256k-8x4]" -sv
```

~12 min per capture. `python -m tracy` starts a WASM web-UI server that looks like it has
hung before it writes the report — let it run. A killed run can be recovered with
`--process-logs-only`.
