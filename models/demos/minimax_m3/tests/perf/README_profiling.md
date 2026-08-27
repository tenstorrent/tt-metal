# MiniMax-M3 prefill zone profiling

Per-zone device-kernel time for one prefill chunk attending an existing KV cache — the
"5k attended to 25k / 55k" case — split into the parts we care about: `ring_joint_sdpa` and the dense
MLP on the dense layers (0-2), and the full MSA + MoE breakdown on the sparse layers (3-59).

## Before the first run

```bash
cd $TT_METAL_HOME
```

Needs: the tilized weight cache at `$HF_MODEL/tensor_cache_bfp8_MeshShape([8, 4])` (without it the run
falls back to the ~869 GB bf16 source read), a golden trace to tile tokens from (defaults to
`$GOLDEN_DIR/longbook_qa_eng_prefill_56320_nopad`), ~50 GB free disk and ~150 GB free RAM.

## Two commands

**1. Capture.** Prints the CSV path when it finishes.

```bash
LEVEL=2 LAYERS=6 CACHE=25600 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh
```

**2. View.** Renders the report and serves it — `--open` prints a URL you can click.

```bash
python3 models/demos/minimax_m3/tests/perf/visualize_zones.py \
    "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" --open
```

```
==============================================================================
  REPORT: http://localhost:8090/zone_report.html
==============================================================================
```

In VS Code / Cursor over SSH a notification offers to open the forwarded port — accept it. Otherwise
tunnel it yourself: `ssh -NL 8090:127.0.0.1:8090 <you>@<host>`.

The `ls -t | head -1` picks the newest capture, so there is no path to copy by hand. Pass the CSV
explicitly if you want an older one.

Rendering is a separate command because the capture is the expensive part (~20 min at 6 layers) and
you will want to look at it more than once.

### Sharing it

The report is a single self-contained HTML file — no external assets — so any of these work:

```bash
# let colleagues open it directly off the lab box
python3 .../visualize_zones.py <csv> --open --bind 0.0.0.0     # then send http://<host>:8090/...

# or just hand them the file
scp <you>@<host>:<path>/zone_report.html .
```

Without `--open` the report is written next to the CSV as `zone_report.html` (or wherever `-o` points).
Opening that path in an editor shows you HTML source, not the report — it needs a browser.

### Capture flags

| flag | meaning | default |
|---|---|---|
| `LEVEL=1\|2\|3` | zone detail — see below | 2 |
| `LAYERS=N` | build only the first N layers. 0-2 are dense, 3+ sparse, so N≥4 covers both; N=6 gives 3+3, N=8 gives 5 sparse samples for the per-chip view | all 60 |
| `LAYER_IDS=a,b` | explicit global layer indices instead of the first N. `LAYER_IDS=0,3` is the fastest useful run: one dense + one sparse, ~10 min | — |
| `CACHE=N` | tokens already cached before the profiled chunk | runs both 25600 and 56320 |
| `CHUNK=N` | tokens in the profiled chunk | 5120 |
| `EXPERT_DTYPE=bf4\|bf8` | MoE routed-expert weight dtype | bf4 |
| `NOC_TRACES=1` | + DRAM/NOC utilization per op. Requires tt-npe installed separately (see *Reading the report*) | off |
| `SKIP_PREFIX=1` | skip the prefill, attend a zeroed cache — fast but MoE routing is unrepresentative | off |

### Detail levels

| level | zones/layer | what you get |
|---|---|---|
| **1** coarse | ~3 | `attn` vs `mlp` per layer. Start here — it answers "which block". |
| **2** medium | ~20 | every block that costs real time: sdpa, the CCLs, `cache_read`, `indexer`, and the MoE stages (`dispatch` / `experts_mm` / `combine` / `moe_reduce`). The default. |
| **3** fine | ~35 | + norms, residuals, rope, head splits, and sub-splits (`deshard` vs `slice`). |

Suppressing a zone never loses time — its ops are charged to the nearest enclosing zone, so every
level accounts for 100% of the chunk, just in fewer buckets. Levels also buy headroom against Tracy's
32K source-location cap on long captures.

The wrapper follows `scripts/run_prefill_perf.sh` conventions: venv activate, `tt-smi -glx_reset` per run, real
tokens tiled from a long golden trace, `LOGURU_LEVEL=INFO` + DEBUG filter, logs in
`prefill_profile_logs/`.

### How long, and which to run

| purpose | command | time |
|---|---|---|
| smoke test the pipeline | `LEVEL=1 LAYER_IDS=0,3 CACHE=5120 SKIP_PREFIX=1 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh` | ~4 min |
| quick iteration on compute | `LEVEL=2 LAYER_IDS=0,3 CACHE=25600 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh` | ~10 min |
| **standard: 3 dense + 3 sparse** | `LEVEL=2 LAYERS=6 CACHE=25600 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh` | **~20 min** |
| collectives + per-chip imbalance | `LEVEL=2 LAYERS=8 CACHE=25600 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh` | ~33 min |

Compute zones (`ring_joint_sdpa`, `sparse_sdpa`, the matmuls) reproduce to within a few percent at any
layer count. The collectives (`moe_reduce`, `combine`, `dispatch`) swing a lot between individual
layers, so a 1-sparse-layer run gives you one draw from that distribution rather than a typical value —
use 6 or 8 layers when the answer depends on them.

Do not scale past ~8 layers — see below.

## Memory and disk

**This is the one way to break the machine, so it is worth understanding.** Capture volume scales with
`layers x chunks x 32 devices`. The dangerous step is not the run on the device — it is tracy's
post-processing, which `tracy-csvexport`s the trace into an intermediate `tracy_ops_times.csv` and then
loads that file into pandas in one go.

Measured on this box:

| layers | `tracy_ops_times.csv` | post-process peak RSS | ops CSV (kept) | transient disk |
|---|---|---|---|---|
| 2 (`LAYER_IDS=0,3`) | ~5 GB | ~30 GiB | 52 MB | ~6 GB |
| 6 | 14 GB | ~65 GiB | 148 MB | ~16 GB |
| 8 | 19 GB | ~110 GiB | 215 MB | ~23 GB |
| **60** | **129 GB** | **388 GiB → OOM** | never produced | ~150 GB |

At 60 layers the post-process ran for ~50 minutes, exhausted 566 GB of RAM, and the OOM killer took out
the harness mid-chunk — losing the capture *and* the hour spent on it. Everything else on the box
suffers while that happens.

So:

- **Stay at or below 8 layers.** 6 is the standard run and leaves plenty of headroom.
- **Watch it if you go higher.** `watch -n5 free -g` during post-processing; if available RAM drops
  under ~90 GB, kill the `python3 -m tracy` process — it cannot finish and the capture is lost anyway.
- The long silent stretch at the end of a run *is* the post-process. It is normal: ~6 min at 2 layers,
  ~10 at 6, ~20 at 8.
- Budget ~50 GB free disk per run before cleanup, and clean up afterwards (below) — the intermediates
  are worthless once the ops CSV exists.

## How it works

| piece | what it does |
|---|---|
| [utils/profiler_utils.py](../../utils/profiler_utils.py) | `zone(name, level)` context manager: emits `M3_ZONE_START/END <name>` Tracy signposts (+ a host Tracy zone). No-op unless `M3_PROFILE_ZONES=1` and `level <= M3_PROFILE_LEVEL`. |
| [profile_prefill.py](profile_prefill.py) | warmup → fill cache to N tokens (un-profiled) → run ONE chunk inside a `profiled_chunk` zone, with the profiler drained per layer BEFORE the chunk and flushed once after it. |
| [parse_zone_perf.py](parse_zone_perf.py) | streams the ops CSV, rebuilds the zone hierarchy from the signpost rows, rolls up ns / ops / bytes / GB/s per zone per device. Also a library. |
| [visualize_zones.py](visualize_zones.py) | the render step: text table + standalone HTML with the per-layer breakdown, per-chip spread, op-level detail and device-busy accounting. |

Attribution: CSV rows are in host-enqueue order, so the ops between a zone's START and END signposts
are exactly the ops that zone enqueued. Each op is charged to the innermost open zone and every
enclosing one, so a parent's total always covers its children. Only zones under `profiled_chunk` are
reported — that is what excludes warmup and the cache-prefix chunks, whose ops share the same CSV.

Same mechanism deepseek_v3_d_p uses (`forward_layer_{i}_start` in `tt/tt_prefill_transformer.py`,
`MLA_START`/`MLA_END` in `tt/mla/mla.py`), extended to a nested hierarchy.

## Reading the report

- **`ms` is the worst device's sum.** With 32 chips the mesh waits for the slowest, so the max is the
  wall-clock-relevant number. `skew ms` (max − min) is what separates a genuinely slow CCL from one
  that is merely waiting on a peer.
- **`GB/s` is bytes-moved ÷ that zone's device time**, with bytes computed from each op's input+output
  shapes and dtypes (block-float formats include their block scales). Compare against the chip's DRAM
  ceiling to judge whether a zone is bandwidth-bound.
- **`DRAM%` / `NOC%`** are the answer to "is this zone bandwidth-bound or just waiting". They only
  appear with `NOC_TRACES=1`, which needs **tt-npe installed separately** — it is not vendored in
  tt-metal. Clone and build https://github.com/tenstorrent/tt-npe, then `source tt-npe/ENV_SETUP` so
  `npe_analyze_noc_trace_dir` is importable. Without it the capture still collects NoC traces (and pays
  for them) but the analysis is skipped and the columns read `-`; the report says so explicitly rather
  than showing zeros.
- `ops/layer` on a parent zone counts its children's ops too.

## Optional: DRAM / NOC utilization (tt-npe)

**Not set up yet — this is a candidate next step, not something that works today.**

The profiler tells you how long a zone took and how many bytes it moved, so it can compute achieved
GB/s. What it cannot tell you is whether that number is a *ceiling* or a *symptom*: `combine` at
48 GB/s and `moe_reduce` at 45 GB/s might be saturating DRAM, or might be mostly one chip waiting for
another. Those call for completely different fixes.

`NOC_TRACES=1` answers it by adding per-op `DRAM BW UTIL (%)` and `NOC UTIL (%)`, which the report
already knows how to display — the columns appear automatically once the data exists, no code change.

The catch is that the analysis needs **tt-npe**, which is a separate repo, not vendored in tt-metal and
not currently installed on the lab box:

```bash
git clone https://github.com/tenstorrent/tt-npe && cd tt-npe   # see its quick-start for the build
source tt-npe/ENV_SETUP                                        # puts npe_analyze_noc_trace_dir on PYTHONPATH
NOC_TRACES=1 LEVEL=2 LAYERS=6 CACHE=25600 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh
```

Without it, `NOC_TRACES=1` still collects NoC traces during the run — paying the overhead and the disk
— and then silently skips the analysis (`tools/tracy/process_ops_logs.py` logs *"Could not import
tt-npe module"*). The report will show `-` in those columns and say they were not measured. So there is
no point passing the flag until tt-npe is built.

Worth doing if the MoE collectives become the thing you want to optimise; not worth it otherwise.

## Gotchas that will bite

**The device profiler buffer.** It holds `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` programs (default
**1000**, which the runner raises to 20000) and one M3 chunk enqueues ~72 ops × num_layers. The harness
drains it via `ttnn.ReadDeviceProfiler` after every layer during warmup and the prefix fill, but goes
**silent during the profiled chunk** and flushes once at the end — a drain is a blocking sync that
lands in the trace as a multi-second `OP TO OP LATENCY` on the next op, which would destroy the gap
measurement. So the chunk's ops must all fit in the buffer at once; that is what the raised count buys.

**Wall-clock here is meaningless.** Even with no drains inside the chunk, tracy's per-op host work
means the device idles waiting for dispatch — an 8-layer chunk measured 5 061 ms against ~180 ms
unprofiled. `DEVICE KERNEL DURATION` and `DEVICE FW DURATION` are on-device and unaffected;
`OP TO OP LATENCY` is not, and the report excludes it. Latency numbers come from
`scripts/run_prefill_perf.sh`.

**Tracy caps a trace at 32K source locations.** Each zone entry allocates one, as does each ttnn op.
A long capture will hit it and silently start dropping zones — use a lower `LEVEL`, fewer `LAYERS`, or
`M3_PROFILE_HOST_ZONES=0` (which drops the host-side Tracy zones; signposts, which the parser reads,
cost no source locations).

**`PROFILE_SKIP_PREFIX=1` is approximate.** It skips the prefix fill and attends a zeroed cache. Op
shapes and therefore costs are identical, but the attention outputs are garbage, so the hidden states
reaching the MoE router are unrealistic and the expert load imbalance (`dispatch`, `experts_mm`,
`combine`) is not representative. Bring-up only. For the same reason the harness uses real tiled tokens
rather than random ids.

## Clean up afterwards

Each run leaves ~6-25 GB of intermediates (see the table above). The ops CSV is the only thing worth
keeping — it is what `visualize_zones.py` reads, and it is 50-215 MB. Re-rendering an old capture needs
nothing else.

```bash
cd $TT_METAL_HOME
rm -rf generated/profiler/.logs/*
rm -f  generated/profiler/reports/*/profile_log_device.csv
rm -f  generated/profiler/reports/*/tracy_profile_log_host.tracy
rm -f  build/profiler/build_wasm/traces/*.tracy
pkill -f tools/tracy/serve_wasm.py     # tracy leaves a WASM server on :8080
```

The `pkill` line is only needed if the web viewer was started at all. Profiling runs that just want
the numbers can skip it by capturing with `python -m tracy --no-web-server ...` (or exporting
`TRACY_NO_WEB_SERVER=1` for the whole session), which never starts the server in the first place.

## Reference numbers

Sanity references for "does my capture look right", **not** CI targets — they are per-zone kernel
times from a deliberately partial 6-layer build, so they do not belong in `models/model_targets.yaml`
(which holds CI-enforced end-to-end model metrics). Nothing validates these; they are here to catch a
broken capture. A healthy `LEVEL=2 LAYERS=6 CACHE=25600` run, real weights, bf4 experts, measured three
times across two days:

| | expected |
|---|---|
| dense layer | 4.33 - 4.35 ms |
| sparse layer | 9.9 - 10.3 ms |
| `attn/ring_joint_sdpa` | 1.952 ms |
| `attn/sparse_sdpa` | 1.717 ms |
| firmware multiplier | ~1.35x |
| 60-layer projection | 860 - 875 ms |

Compute zones land within ~1%. The collectives (`combine`, `dispatch`, `moe_reduce`) move by tens of
percent between runs — that variance is real cross-chip skew, not a broken capture.

## The `cache_read/deshard` hypothesis

The packed KV cache is one tensor per K/V/index_k of shape
`[num_users*num_layers, 1, seq_local, head_dim]` ([attention/kv_cache.py](../../tt/attention/kv_cache.py)).
The MSA cache-read path converts the **whole** tensor from NdShard to DRAM-interleaved on **every**
sparse layer — the round-robin bank mapping is only intact for the full tensor, so it cannot slice one
layer's slot first ([attention/prefill.py](../../tt/attention/prefill.py)). At 61440 tokens that is
~63 MiB per tensor, read+write, ×3 tensors, ×57 layers ≈ 20+ GiB of DRAM traffic per chunk — plausibly
more than every expert weight read combined.

`profile_prefill.py` logs the expected byte count at startup, and the report separates
`attn/cache_read/deshard` from `attn/cache_read/slice`, so the measured cost and GB/s land right next
to the prediction.
