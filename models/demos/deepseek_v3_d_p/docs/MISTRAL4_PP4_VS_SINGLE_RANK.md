# Mistral Small 4 prefill: PP=4 x (8,1) vs single-rank

Reproduce with `models/demos/deepseek_v3_d_p/tests/perf/pp4/` (see its README for prerequisites --
the weight caches and checkpoint are not in the repo).

**Measured 2026-08-31 on `bh-glx-110-a04u02` (32-chip Blackhole galaxy), branch
`kmabee/mistral4-prefill-full-rebased.aug27` @ `779a4af546b`.**

Two configurations of the same 36-layer model, both driven through the **same** runner + producer
(`prefill_runner` + `prefill_producer` under `tt-run`), so **topology is the only variable**:

- **single-rank** — SP=8 x TP=4 on one 8x4 mesh, `2d_torus_xy`
- **PP=4** — four `[8,1]` column sub-meshes, hidden state handed stage-to-stage over a real
  device-to-device `ttnn` MeshSocket on fabric, `2d_torus_y`

Chunk size is the production 5,120 throughout; ISL is varied by chunk count, because that is what a
long prompt actually is (100K = 20 chunks, not one 100K forward).

---

# 1. End-to-end results

## 1.1 Single-request latency (one request, start to finish)

| ISL | 1-rank | PP=4 | winner |
|---:|---:|---:|---|
| 5,120 | **0.704 s** (7,277 tok/s) | 0.947 s (5,406) | **1-rank by 1.35x** |
| 25,600 | 1.375 s (18,612) | **1.209 s** (21,177) | PP=4 by 1.14x |
| 102,400 | 5.745 s (17,824) | **4.130 s** (24,792) | PP=4 by 1.39x |
| 261,120 | 21.558 s (12,112) | **15.146 s** (17,240) | PP=4 by 1.42x |

**The crossover is between 5,120 and 25,600.** At 5,120 the request is a single chunk, so there is
nothing to pipeline: it traverses all four stages serially and PP=4 pays pipeline depth for no
overlap. From 25,600 up there are enough chunks in flight to fill the pipeline and PP=4 wins, by more
as context grows.

## 1.2 Throughput (many requests, steady state)

Last rank's median chunk-to-chunk interval, first 8 intervals discarded.

| ISL | 1-rank ms/chunk | 1-rank tok/s | PP=4 ms/chunk | PP=4 tok/s | ratio |
|---:|---:|---:|---:|---:|---:|
| 5,120 | 143.8 (143–147) | 35,597 | 108.5 (106–110) | **47,178** | **1.33x** |
| 25,600 | 166.8 (145–185) | 30,689 | 119.2 (106–134) | **42,940** | **1.40x** |
| 102,400 | 315.3 (150–380) | 16,238 | 199.6 (91–248) | **25,653** | **1.58x** |
| 261,120 | 437.3 (155–619) | 11,708 | 299.9 (89–446) | **17,073** | **1.46x** |

**PP=4 wins throughput at every ISL.** The wide min–max at long context is expected, not noise: a
multi-request run interleaves shallow and deep chunks, so the median mixes both. Use the
single-request latency table when you want a clean long-context number.

## 1.3 Rerun

```bash
S=models/demos/deepseek_v3_d_p/tests/perf/pp4
$S/preflight.sh          # validates chips/build/caches; flags per-machine items
$S/run_matrix.sh         # all 16 cells; completed cells are skipped (FORCE=1 to redo)
```
Selective: `CONFIGS=pp4 ISLS="102400 261120" MODES=ttft $S/run_matrix.sh`

**Logs**: written to `<repo>/mistral4_perf_<hostname>/<cell>/{runner,producer}.log` (not committed -- multi-GB)
**Analysis**: `analyze_pp.py <runner.log> 8` (throughput) · `analyze_ttft.py <runner.log>` (latency)

---

# 2. Single-layer profiling

One layer, captured with the device profiler + Tracy, driven through the **real chunked runner** so
the KV cache actually deepens (8 chunks x 5,120 = 40,960 context). Both configs, same harness.

```bash
S=models/demos/deepseek_v3_d_p/tests/perf/pp4
DEEP_CHUNKS=8 $S/run_single_layer_profile.sh 1rank_deep
DEEP_CHUNKS=8 $S/run_single_layer_profile.sh pp4_deep
```

## 2.1 Where the time goes, per layer per chunk

| | 1-rank (32 chips, TP=4) | PP=4 stage (8 chips, TP=1) |
|---|---:|---:|
| **compute / layer** | **5.59 ms** | **11.89 ms** |
| MLA / attention | 1.029 ms (18.4%) | 2.953 ms (24.8%) |
| MoE | 1.898 ms (33.9%) | 6.323 ms (53.2%) |
| norm | 0.024 ms (0.4%) | 0.155 ms (1.3%) |
| other (incl. TP collectives) | 2.643 ms (47.2%) | 2.461 ms (20.7%) |

**The identity that explains the whole result:**

```
1-rank  :  5.59 ms x 32 chips = 178.9 chip-ms per layer
PP stage: 11.89 ms x  8 chips =  95.1 chip-ms per layer
-> a PP layer is 2.13x SLOWER in wall time but uses 1.88x LESS silicon time
```

PP=4 does not make a layer faster. It makes a layer *cheaper*, and pipelining is what converts that
into throughput by keeping all 32 chips busy.

**Where the saving comes from — TP=1 deletes the tensor-parallel collectives.** Collective ops present
in each capture (instance counts):

| | 1-rank | PP=4 stage |
|---|---|---|
| `HighBwAllGather` | 1440 | – |
| `ReduceScatterMinimalAsync` | 1152 | – |
| `LayerNormPre/PostAllGather` | 864 each | – (uses the fused single-op `LayerNorm`) |
| `AllGatherAsync` / `AllGather` | 576 / 576 | 72 (SP-axis MoE routing only) |
| `ReduceScatter` | 288 | – |

Seven collective op types at TP=4; **one** at TP=1. That is the mechanism, visible directly.

## 2.2 The cost of context depth is entirely MLA/SDPA

Per-chunk op duration as the KV cache grows (PP=4 stage 0, chunks 1→8, 5K→41K context):

| op | μs across the ramp | growth |
|---|---|---:|
| **RingJointSDPA** | 961 → 1422 → 2034 → 2678 → 3320 → 3960 → 4599 → **5242** | **5.46x** |
| UnifiedRoutedExpertFfn | 1892 → … → 1879 | 0.99x |
| Combine | 2272 → … → 2132 | 0.94x |
| Dispatch | 1753 → … → 1939 | 1.11x |
| Matmul (x10) | 2089 → … → 2088 | **1.00x** |

Single-rank shows the same shape: SDPA 355 → 1857 μs (**5.23x**), everything else flat.

Growth is **linear at ~640 μs per additional 5,120 tokens of KV**. This extrapolates and
**cross-checks against the end-to-end table**: at 102,400 (20 chunks) it predicts
SDPA 13.1 + MoE 6.3 + other 2.6 = **22.0 ms/layer**, and the measured PP=4 throughput cell is
199.6 ms/chunk / 9 layers = **22.2 ms/layer**. Two independent measurements agreeing to 1%.

**Actionable conclusion: at short context MoE dominates the layer (53% of a PP stage), but it is
FIXED. Every additional token of context lands entirely in MLA/SDPA. Long-context optimisation effort
belongs in MLA/SDPA, not MoE.**

## 2.3 Stage asymmetry (PP=4)

| stage | compute/layer | note |
|---|---:|---|
| 0 | 11.89 ms | + embedding; outbound D2D only |
| 1 | 14.63 ms | MoE 9.12 ms — `Combine` 3.94 vs 2.16 elsewhere |
| 2 | 11.97 ms | |
| 3 | 16.17 ms | + final norm and LM head (`Matmul` x11, 4.76 ms vs 2.09) |

The pipeline period is set by the **slowest** stage, so stage 3 (and stage 1's MoE outlier) are what
would bound throughput. Stage 1's `Combine` is worth a look — possible expert-routing imbalance — but
it is a single layer, so do not over-read it.

## 2.4 The captures are committed

The data behind this section is in `models/demos/deepseek_v3_d_p/tests/perf/pp4/captures/`, so it
can be read and re-analysed **without a galaxy**: `report_<n>.txt` is the rendered `tt-perf-report`
op table plus both analyzer summaries, and `ops_<n>.csv.gz` is the row data (all devices, reduced
columns) that reproduces the numbers above exactly. See that directory's README for what was
dropped and how to read it. The full 13 GB captures are not committed.

## 2.5 Analysing a capture — `tt-perf-report` first

`tt-perf-report` is the first-class tool for Tracy op reports; start there. It is a pip package and
installs into `~/.local/bin`, which is **local disk per machine**, so install it wherever you work:

```bash
python3 -m pip install --user tt-perf-report
export PATH="$HOME/.local/bin:$PATH"
P=<repo>/mistral4_perf_profile

# single-rank
tt-perf-report $(ls $P/1rank_deep/rank0/reports/*/*/ops_perf_results*.csv | tail -1)

# each PP=4 stage (0 = first layer + embedding, 3 = last layer + LM head)
for r in 0 1 2 3; do echo "===== stage $r ====="
  tt-perf-report $(ls $P/pp4_deep/rank$r/reports/*/*/ops_perf_results*.csv | tail -1)
done

# bound to one leg of the layer (the model emits these signposts itself)
tt-perf-report --start-signpost MLA_START --end-signpost MLA_END <csv>
tt-perf-report --start-signpost MoE_START --end-signpost MoE_END <csv>

# other flags worth knowing
tt-perf-report --print-signposts <csv>          # what boundaries this capture has
tt-perf-report --id-range A-B <csv>             # isolate one chunk from a multi-chunk capture
tt-perf-report --group-by memory <csv>          # or: op, category
tt-perf-report --no-color --no-advice <csv>     # for piping / diffing
```

**The one thing it cannot tell you**, and the only reason the two scripts below exist:
`tt-perf-report` does not know that `InboundSocketServiceSyncOperation` is a *blocking wait*, so in a
PP stage capture that op is ~99% of device time and its "Total %" column is mostly idle rather than
work. These exclude the socket waits and normalise per device, which is what makes the per-layer
numbers comparable between configurations:

```bash
S=models/demos/deepseek_v3_d_p/tests/perf/pp4
python3 $S/analyze_layer_budget.py <csv> "label"   # per-layer budget, socket waits excluded
python3 $S/analyze_kv_ramp.py      <csv> "label"   # per-chunk ramp -> cost of KV depth
```

Reading a committed capture instead of running one: `report_<n>.txt` in `tests/perf/pp4/captures/`
**is** rendered `tt-perf-report` output, so it needs no tooling. `ops_<n>.csv.gz` re-runs the two
scripts above but **cannot** be fed to `tt-perf-report` — columns were dropped to fit it in git.

---

# 3. Notes, traps and caveats

**Read these before quoting any number.**

- **`InboundSocketServiceSyncOperation` is ~99% of device time in a PP stage capture and is NOT
  transport cost.** It is the receiver *blocking* until upstream data arrives — pure idle. Leaving it
  in makes every real op look like rounding error. `analyze_layer_budget.py` excludes it and reports
  it separately.
- **One CSV row per device.** A stage spans 8 chips and its ops run concurrently, so an op's cost is
  the max across devices, not the sum. Summing inflates a 12 ms layer to 96 ms.
- **Never sum the four PP stage tables.** The sum is roughly one chunk's *latency* through the
  pipeline; throughput is `1 / max(stage)`.
- **The captures are eager and instrumented; the throughput numbers are traced.** Read op *durations*
  and *ratios*, not op-to-op gaps (host dispatch, absent under trace) and not absolute wall time —
  profiler instrumentation inflates kernel durations somewhat.
- **The single-layer capture amortises transport over 1 layer instead of 9**, so do not read a
  transport *fraction* off it. One D2D hop measures **~11 ms end-to-end** for the 42 MB activation
  (0.47–0.59 ms host push, 2.25–2.42 ms enqueue+grant+sync), from a weightless 4-rank microbenchmark.
- **The latency column is prefill-completion latency, not literally TTFT — no token is emitted** —
  but **it is a good TTFT proxy: the missing tail is ~2.79 ms, i.e. 0.3% at 5,120 and 0.02% at
  261,120.** Treat the latency table as TTFT.
  Why no token: with the LM head enabled the last rank does an event sync and
  `TT_FATAL: Event Synchronization is not supported during trace capture`
  (`fd_mesh_command_queue.cpp:932`), so traced + token is impossible; eager costs ~1.35x and breaks
  comparability with the traced cells.
  The 2.79 ms is measured, not assumed: in the `pp4_deep` capture (run with the LM head ON) stage 3's
  matmul durations are identical to stage 2's **plus one extra matmul at 2,720 us**, plus one extra
  norm (66 us). An end-to-end A/B (`MODES=tokentail notail`, 4 runs each, warm) **could not resolve
  it**: means 0.6935 s vs 0.6562 s, but the per-run differences are +42, +61, -4, +50 ms — the sign
  flips, and run-to-run sd (~18 ms/arm) exceeds the effect. Even taking that unresolvable 37 ms as a
  pessimistic bound, the error is <4% at 5,120 and <0.3% at 261,120.
- **A single-request latency measurement of a NEW code path includes cold kernel JIT.** The first
  `tokentail` run measured 11.775 s vs 0.655 s — all compilation (85.8% JIT cache hits vs 100%), not
  LM-head cost. Always run such a cell twice and use the warm one.
- **PP=4 has no correctness gate.** `prefill_runner` rejects `PREFILL_MOCK_MIGRATION` for
  `num_ranks>1`. PP correctness rests on the single-rank KV-PCC run: 36 layers bottoms at **0.9034**,
  which an 8-layer control proved is depth accumulation (layers 0–7 reproduce to five digits and pass
  at 0.9931) rather than miswiring. The in-tree floor for this quantity is 0.85
  (`test_prefill_transformer_chunked.py:161`); **use `PREFILL_STANDALONE_CHUNKED_PCC=0.88`** for
  Mistral4, not the Kimi-calibrated 0.93 default.
- **Prefill only. No decode data of any kind.**
- **Do not compare against numbers from a different harness.** Single-rank measured here is
  **35,597 tok/s** at 5,120; the older pytest-harness figure is 29,767 — 20% apart. Using the latter
  would claim 1.58x for PP=4 instead of the real 1.33x.

**Machine-specific gotchas**

- **The `[8,1]` column -> physical device map is per-galaxy.** `bh-glx-b03u02` and `bh-glx-110-a04u02`
  enumerate differently, and a wrong map does **not** error — it builds stages that are not columns.
  Regenerate on every new machine: `python3 $S/gen_pp4_binding.py [--profile]`. The runners prefer a
  `<binding>.<hostname>.yaml` automatically.
- **`tt-smi -r` does not recover this box** (CPLD < v1.16; the tool says so and points at
  `-glx_reset`). Use `tt-smi -glx_reset` (~90 s). The matrix does this automatically on failure and
  aborts rather than cascading if the fabric is still unmappable.
- **`$HOME` is local disk per machine; `/data` is shared NFS.** `~/debug-docs` and `~/.local/bin`
  (where `tt-perf-report` lives) do not travel between boxes.
- **Never edit a script while a run is executing it.** Bash reads scripts by byte offset; a rewrite
  mid-run makes it resume at the wrong place and produces baffling syntax errors in untouched lines.
  This cost three runs during this work. For a long unattended sweep, copy the scripts to a scratch
  directory and launch from that immutable snapshot rather than from the tree you are editing.
