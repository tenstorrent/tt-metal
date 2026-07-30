# MiniMax-M3 prefill zone profiling

Per-zone device-kernel time for one prefill chunk attending an existing KV cache — the
"5k attended to 25k / 55k" case — split into the parts we care about: `ring_joint_sdpa` and the dense
MLP on the dense layers (0-2), and the full MSA + MoE breakdown on the sparse layers (3-59).

## Two commands

**1. Capture.** Prints the CSV path when it finishes.

```bash
LEVEL=2 LAYERS=6 CACHE=25600 ./run_prefill_profile.sh
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
| `NOC_TRACES=1` | + tt-npe DRAM/NOC utilization per op | off |
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

The wrapper follows `run_prefill_perf.sh` conventions: venv activate, `tt-smi -glx_reset` per run, real
tokens tiled from a long golden trace, `LOGURU_LEVEL=INFO` + DEBUG filter, logs in
`prefill_profile_logs/`.

### How long, and which to run

| purpose | command | time |
|---|---|---|
| smoke test the pipeline | `LEVEL=1 LAYER_IDS=0,3 CACHE=5120 SKIP_PREFIX=1 ./run_prefill_profile.sh` | ~4 min |
| quick iteration on compute | `LEVEL=2 LAYER_IDS=0,3 CACHE=25600 ./run_prefill_profile.sh` | ~10 min |
| **standard: 3 dense + 3 sparse** | `LEVEL=2 LAYERS=6 CACHE=25600 ./run_prefill_profile.sh` | **~20 min** |
| collectives + per-chip imbalance | `LEVEL=2 LAYERS=8 CACHE=25600 ./run_prefill_profile.sh` | ~33 min |

Compute zones (`ring_joint_sdpa`, `sparse_sdpa`, the matmuls) reproduce to within a few percent at any
layer count. The collectives (`moe_reduce`, `combine`, `dispatch`) swing a lot between individual
layers, so a 1-sparse-layer run gives you one draw from that distribution rather than a typical value —
use 6 or 8 layers when the answer depends on them.

Do not scale past ~8 layers: capture volume grows with layers x chunks, and at 60 the intermediate CSV
reached 129 GB and OOM-killed the run.

## How it works

| piece | what it does |
|---|---|
| [utils/profiler_utils.py](../../utils/profiler_utils.py) | `zone(name)` context manager: emits `M3_ZONE_START/END <name>` Tracy signposts (+ a host Tracy zone). No-op unless `M3_PROFILE_ZONES=1`. |
| [profile_prefill.py](profile_prefill.py) | warmup → fill cache to N tokens (un-profiled) → run ONE chunk inside a `profiled_chunk` zone, reading the device profiler after every layer. |
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
- **`DRAM%` / NOC util** only appear with `NOC_TRACES=1` (tt-npe simulates the traffic and the profiler
  fills `DRAM BW UTIL (%)` / `NOC UTIL (%)` per op).
- `ops/layer` on a parent zone counts its children's ops too.

## Two gotchas that will bite

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
`run_prefill_perf.sh`.

**Tracy caps a trace at 32K source locations.** Each zone entry allocates one, as does each ttnn op.
A long capture will hit it and silently start dropping zones — use a lower `LEVEL`, fewer `LAYERS`, or
`M3_PROFILE_HOST_ZONES=0` (which drops the host-side Tracy zones; signposts, which the parser reads,
cost no source locations).

**`PROFILE_SKIP_PREFIX=1` is approximate.** It skips the prefix fill and attends a zeroed cache. Op
shapes and therefore costs are identical, but the attention outputs are garbage, so the hidden states
reaching the MoE router are unrealistic and the expert load imbalance (`dispatch`, `experts_mm`,
`combine`) is not representative. Bring-up only. For the same reason the harness uses real tiled tokens
rather than random ids.

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
