# GPT-OSS prefill zone profiling

Per-zone device-kernel time for one prefill chunk, split into the parts we care about: where the
time goes between **compute** (SDPA, the matmuls, the expert FFNs), **communication** (the TP/SP
collectives, MoE dispatch/combine/reduce) and **KV-cache memory traffic** — per layer class
(sliding-window vs full attention) and per chip.

Mirror of `minimax_m3/tests/perf/` — same mechanism, same workflow, same report.

## The two cases it measures

| config | path exercised | why you care |
|---|---|---|
| `CACHE=0` | **one-shot**: all-gather Q/K/V → SDPA → reduce-scatter | attention comm is separate ops here, so this capture gives the clean comm/compute reference |
| `CACHE=24576` (default) | **chunked ring**: cache-backed RingJointSDPA reads the accumulated prefix | the production chunked path — the one that is ~16× slower than one-shot (#52000) |

The default run does both, back to back. Comparing the two reports is the whole point: it localizes
what the chunked ring path costs relative to one-shot, zone by zone.

## Before the first run

Needs: gpt-oss-120b weights (defaults to `/data/jmalone/.cache/huggingface/hub/models--openai--gpt-oss-120b/gpt-oss-120b`,
which has the tilized cache + MoE bias sidecars populated — so the default `FROM_CACHE=1` load is
fast), a golden trace to tile tokens from (defaults to `/data/jmalone/gpt_oss_golden/full_context`),
the (4,8) Blackhole galaxy, ~50 GB free disk and ~100 GB free RAM.

## Two commands

**1. Capture.** Prints the CSV path(s) when it finishes.

```bash
cd $TT_METAL_HOME
LEVEL=2 LAYERS=4 ./models/demos/gpt_oss_d_p/scripts/run_prefill_profile.sh
```

**2. View.** Renders the report and serves it — `--open` prints a URL you can click.

```bash
python3 models/demos/gpt_oss_d_p/tests/perf/visualize_zones.py \
    "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" --open
```

In VS Code / Cursor over SSH a notification offers to open the forwarded port — accept it. Otherwise
tunnel it yourself: `ssh -NL 8090:127.0.0.1:8090 <you>@<host>`. The report is a single
self-contained HTML file; `--bind 0.0.0.0` serves it to colleagues directly, or just `scp` it.

Rendering is a separate command because the capture is the expensive part and you will want to look
at it more than once. The `ls -t | head -1` picks the newest capture.

### Capture flags

| flag | meaning | default |
|---|---|---|
| `LEVEL=1\|2\|3` | zone detail — see below | 2 |
| `LAYERS=N` | build only the first N layers. Layers alternate sliding (even) / full (odd), so N≥2 covers both classes; 4 gives 2 samples of each | 4 |
| `CACHE=N` | tokens already cached before the profiled chunk (rounded down to whole chunks). `CACHE=0` = one-shot | runs both 0 and 24576 |
| `CHUNK=N` | tokens in the profiled chunk (multiple of 256) | 8192 |
| `EXPERT_DTYPE=bf4\|bf8` | MoE routed-expert weight dtype | bf4 |
| `FROM_CACHE=0` | load real safetensors instead of the tilized TTNN cache (needed once to populate the cache + bias sidecars) | 1 |
| `NOC_TRACES=1` | + DRAM/NOC utilization per op. Requires tt-npe installed separately | off |
| `SKIP_PREFIX=1` | skip the prefill, attend a zeroed cache — fast but MoE routing is unrepresentative | off |

### Detail levels

| level | zones/layer | what you get |
|---|---|---|
| **1** coarse | ~3 | `attn` vs `mlp` per layer. Start here — it answers "which block". |
| **2** medium | ~15 | every block that costs real time: the SDPAs, the CCLs, and the MoE stages (`dispatch` / `experts_mm` / `combine` / `moe_reduce`). The default. |
| **3** fine | ~25 | + norms, residuals, rope, head splits, and the small glue ops. |

Suppressing a zone never loses time — its ops are charged to the nearest enclosing zone, so every
level accounts for 100% of the chunk, just in fewer buckets. Levels also buy headroom against
Tracy's 32K source-location cap on long captures.

## Memory and disk — do not scale past ~8 layers

Capture volume scales with `layers × chunks × 32 devices`, and the dangerous step is not the device
run but tracy's post-processing, which loads the exported ops CSV into pandas in one go. On the M3
sibling (more ops/layer than gpt-oss, so a conservative upper bound): 2 layers → ~30 GiB peak RSS,
6 → ~65 GiB, 8 → ~110 GiB, and a full-model run → **OOM-killed after ~50 min, losing the capture**.
The long silent stretch at the end of a run *is* the post-process; it is normal. Stay at or below
8 layers, and clean up the intermediates afterwards (below).

## How it works

| piece | what it does |
|---|---|
| [utils/profiler_utils.py](../../utils/profiler_utils.py) | `zone(name, level)` context manager: emits `GPTOSS_ZONE_START/END <name>` Tracy signposts (+ a host Tracy zone). No-op unless `GPTOSS_PROFILE_ZONES=1` and `level <= GPTOSS_PROFILE_LEVEL`. |
| [profile_prefill.py](profile_prefill.py) | warmup → fill cache to N tokens (un-profiled) → run ONE chunk inside a `profiled_chunk` zone, with the profiler drained per layer BEFORE the chunk and flushed once after it. |
| [parse_zone_perf.py](parse_zone_perf.py) | streams the ops CSV, rebuilds the zone hierarchy from the signpost rows, rolls up ns / ops / bytes / GB/s per zone per device. Also a library. |
| [visualize_zones.py](visualize_zones.py) | the render step: text table + standalone HTML with the per-class breakdown, compute/comm/memory split, per-chip spread, op-level detail and device-busy accounting. |
| [test_zone_profiler.py](test_zone_profiler.py) | device-free pytest pinning the contracts that would otherwise fail silently (attribution, leaf detection, categorization, chunk plan). |

Attribution: CSV rows are in host-enqueue order, so the ops between a zone's START and END signposts
are exactly the ops that zone enqueued. Each op is charged to the innermost open zone and every
enclosing one, so a parent's total always covers its children. Only zones under `profiled_chunk` are
reported — that is what excludes warmup and the cache-prefix chunks, whose ops share the same CSV.

Zone tree (LEVEL=2; FINE-only zones in parentheses):

```
profiled_chunk
└─ layerNN_{sliding|full}
   ├─ (input_norm)
   ├─ attn
   │  ├─ qkv_proj, (split_heads), (rope), kv_write
   │  ├─ ring_joint_sdpa            ← chunked ring path (fused compute + ring CCL)
   │  ├─ ag_qkv, sdpa, sdpa_reduce_scatter   ← one-shot path
   │  ├─ (concat_heads)
   │  └─ o_proj + ccl_out_allreduce  (or o_proj_fused_rs + ccl_out_allgather on WH)
   ├─ (residual_attn), (post_attn_norm)
   ├─ mlp
   │  ├─ router_topk, (routing_setup)
   │  └─ dispatch, experts_mm, combine, moe_reduce, tp_allgather
   └─ (residual_mlp)
```

## Reading the report

- **`ms` is the worst device's sum.** With 32 chips the mesh waits for the slowest, so the max is
  the wall-clock-relevant number. `skew ms` (max − min) is what separates a genuinely slow CCL from
  one that is merely waiting on a peer.
- **`ring_joint_sdpa` is categorized as compute** even though it embodies the SP ring communication:
  the ring rotation and the attention math are fused into one device op and cannot be split. Use the
  one-shot capture (`CACHE=0`), where `ag_qkv` / `sdpa` / `sdpa_reduce_scatter` are separate ops, as
  the reference for attention's comm/compute ratio.
- **`GB/s` is bytes-moved ÷ that zone's device time**, from each op's input+output shapes and dtypes.
  Compare against the chip's DRAM ceiling to judge whether a zone is bandwidth-bound.
- **`DRAM%` / `NOC%`** only appear with `NOC_TRACES=1`, which needs tt-npe built separately
  (https://github.com/tenstorrent/tt-npe, then `source tt-npe/ENV_SETUP`). Without it the capture
  still pays for the NoC traces but the columns read `-`.
- The MoE collectives (`dispatch`, `combine`, `moe_reduce`) swing between layers with the expert
  routing — that variance is real. Use 4+ layers when the answer depends on them; compute zones
  reproduce to within a few percent at any layer count.

## Gotchas that will bite

**The device profiler buffer.** It holds `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` programs (default
**1000**, which the runner raises to 20000). The harness drains it after every layer during warmup
and the prefix fill, but goes **silent during the profiled chunk** and flushes once at the end — a
drain is a blocking sync that would land in the trace as a multi-second gap. So the chunk's ops must
all fit in the buffer at once; that is what the raised count buys.

**Wall-clock here is meaningless.** Under tracy the host cannot dispatch fast enough, so the device
idles between ops. `DEVICE KERNEL DURATION` and `DEVICE FW DURATION` are on-device and unaffected;
`OP TO OP LATENCY` is not, and the report excludes it. Throughput numbers come from
`tests/galaxy_prefill_kv_pcc.py` (`PREFILL_TPS_ITERS`).

**Tracy caps a trace at 32K source locations.** A long capture will silently start dropping zones —
use a lower `LEVEL`, fewer `LAYERS`, or `GPTOSS_PROFILE_HOST_ZONES=0` (signposts, which the parser
reads, cost no source locations).

**`SKIP_PREFIX=1` is approximate.** Op shapes and costs are identical, but the attention outputs are
garbage, so the hidden states reaching the MoE router are unrealistic and the expert load imbalance
(`dispatch`, `experts_mm`, `combine`) is not representative. Bring-up only. For the same reason the
harness tiles real tokens rather than generating random ids.

## Clean up afterwards

The ops CSV is the only thing worth keeping — it is what `visualize_zones.py` reads.

```bash
cd $TT_METAL_HOME
rm -rf generated/profiler/.logs/*
rm -f  generated/profiler/reports/*/profile_log_device.csv
rm -f  generated/profiler/reports/*/tracy_profile_log_host.tracy
rm -f  build/profiler/build_wasm/traces/*.tracy
pkill -f tools/tracy/serve_wasm.py     # tracy leaves a WASM server on :8080
```
