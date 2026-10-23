# GPT-OSS prefill zone profiling

Per-zone device-kernel time for one prefill chunk, split into the parts we care about: where the
time goes between **compute** (SDPA, the matmuls, the expert FFNs), **communication** (the TP/SP
collectives, MoE dispatch/combine/reduce) and **KV-cache memory traffic** — per layer class
(sliding-window vs full attention) and per chip.

The mechanism, parser and report are shared with MiniMax-M3 and live in
[`models/demos/common/prefill/profiling/`](../../../common/prefill/profiling/); this directory holds the
GPT-OSS harness, the wrapper script and two one-line CLI shims. What is GPT-OSS-specific (signpost
prefix, env vars, layer classes, which zones are communication) is declared once in
[`utils/profiler_utils.py`](../../utils/profiler_utils.py).

## The two cases it measures

| config | path exercised | why you care |
|---|---|---|
| `CACHE=0` | **one-shot**: all-gather Q/K/V → SDPA → reduce-scatter | attention comm is separate ops here, so this capture gives the clean comm/compute reference |
| `CACHE=24576` (default) | **chunked ring**: cache-backed RingJointSDPA reads the accumulated prefix | the production chunked path — the one with the genuine ~2× warm overhead over one-shot that [#52000](https://github.com/tenstorrent/tt-metal/issues/52000) tracks (its original ~16× was a first-compile artifact) |

The default run does both, back to back. Comparing the two reports is the whole point: it localizes
what the chunked ring path costs relative to one-shot, zone by zone.

## Before the first run

Needs: the (4,8) Blackhole galaxy on a **fresh allocation** (see `RESET` below), ~50 GB free disk and
~100 GB free RAM, and:

- gpt-oss-120b weights with the tilized cache next to them. The default `HF_MODEL` is
  `/mnt/models/blaze/openai/gpt-oss-120b`, the copy the CI stages use; its
  `tensor_cache_bfp8_MeshShape([4, 8])` has the MoE bias sidecars, so the default `FROM_CACHE=1` load
  is fast. On a fresh `HF_MODEL`, run once with `FROM_CACHE=0` to populate the cache.
- a golden trace to tile tokens from: `$HF_MODEL/golden/longbook_qa_eng_prefill_5000/metadata.json` by
  default; any dir with a `metadata.json` carrying `token_ids` works (`GOLDEN_DIR` / `SRC_TRACE`).

## Two commands

**1. Capture.** Prints the CSV path(s) when it finishes.

```bash
cd $TT_METAL_HOME
LEVEL=2 LAYERS=4 ./models/demos/gpt_oss_d_p/scripts/run_prefill_profile.sh
```

Each finished capture is moved out of `generated/profiler/` into
`prefill_profile_results/<stamp>_gptoss_layers<N>_cache<N>_<dtype>/` (the ops CSV plus a copy of the
log), so the tracy intermediates can be wiped between experiments and every capture stays
re-renderable. `RESULTS_DIR` overrides the root.

**2. View.** Renders the report and serves it — `--open` prints a URL you can click.

```bash
python3 models/demos/gpt_oss_d_p/tests/perf/visualize_zones.py \
    "$(ls -t prefill_profile_results/*/ops_perf_results_*.csv | head -1)" --open
```

In VS Code / Cursor over SSH a notification offers to open the forwarded port — accept it. Otherwise
tunnel it yourself: `ssh -NL 8090:127.0.0.1:8090 <you>@<host>`. The report is a single
self-contained HTML file; `--bind 0.0.0.0` serves it to colleagues directly, or just `scp` it.

Rendering is a separate command because the capture is the expensive part (~10 min for both configs at
4 layers) and you will want to look at it more than once. The `ls -t | head -1` picks the newest capture.

### Capture flags

| flag | meaning | default |
|---|---|---|
| `HF_MODEL` | weights dir with the tilized cache next to it | `/mnt/models/blaze/openai/gpt-oss-120b` |
| `GOLDEN_DIR` / `SRC_TRACE` | golden traces dir / the `metadata.json` to tile tokens from | `$HF_MODEL/golden/longbook_qa_eng_prefill_5000` |
| `RESULTS_DIR` | where finished captures are moved | `$TT_METAL_HOME/prefill_profile_results` |
| `LEVEL=1\|2\|3` | zone detail — see below | 2 |
| `LAYERS=N` | build only the first N layers. Layers alternate sliding (even) / full (odd), so N≥2 covers both classes; 4 gives 2 samples of each | 4 |
| `CACHE=N` | tokens already cached before the profiled chunk (rounded down to whole chunks). `CACHE=0` = one-shot | runs both 0 and 24576 |
| `CHUNK=N` | tokens in the profiled chunk (multiple of 256) | 8192 |
| `EXPERT_DTYPE=bf4\|bf8` | MoE routed-expert weight dtype | bf4 |
| `FROM_CACHE=0` | load real safetensors instead of the tilized TTNN cache (needed once to populate the cache + bias sidecars) | 1 |
| `NOC_TRACES=1` | + DRAM/NOC utilization per op. Requires tt-npe installed separately | off |
| `SKIP_PREFIX=1` | skip the prefill, attend a zeroed cache — fast but MoE routing is unrepresentative | off |
| `READ_IN_CHUNK=1` | also drain the device profiler per layer inside the profiled chunk — an experiment to see whether drains change CCL times, not a mode | off |
| `RESET=1` | `tt-smi -glx_reset` before each capture. Off because on exabox galaxies a reset tears down the torus wraparound links and they do not retrain (the next ring open fails with "Graph specified in MGD could not fit"), and `-glx_reset` is an IPMI tray reset that can wedge the node. Profile on a fresh allocation instead | off |

### Detail levels

| level | zones/layer | what you get |
|---|---|---|
| **1** coarse | ~3 | `attn` vs `mlp` per layer. Start here — it answers "which block". |
| **2** medium | ~15 | every block that costs real time: the SDPAs, the CCLs, and the MoE stages (`dispatch` / `experts_mm` / `combine` / `moe_reduce`). The default. |
| **3** fine | ~25 | + norms, residuals, rope, head splits, and the small glue ops. |

Suppressing a zone never loses time: its ops are charged to the nearest enclosing zone and the report
shows them as that zone's **`(self)` bucket** (`(self)` for the layer itself, `mlp/(self)` for the glue
between the MoE stages). So at every level the leaves — real leaf zones plus `(self)` buckets — sum
exactly to the layer total, in fewer buckets at a coarser level. Levels also buy headroom against
Tracy's 32K source-location cap on long captures.

## How it works

| piece | what it does |
|---|---|
| [common/prefill/profiling/zones.py](../../../common/prefill/profiling/zones.py) | `ZoneProfiler.zone(name, level)` context manager: emits `GPTOSS_ZONE_START/END <name>` Tracy signposts (+ a host Tracy zone). No-op unless `GPTOSS_PROFILE_ZONES=1` and `level <= GPTOSS_PROFILE_LEVEL`. |
| [utils/profiler_utils.py](../../utils/profiler_utils.py) | the GPT-OSS `ZoneSpec` (prefix, env vars, `sliding`/`full` classes, comm/memory keys) and the `zone` / `read_profiler` the model code imports. |
| [profile_prefill.py](profile_prefill.py) | warmup → fill cache to N tokens (un-profiled, profiler drained after every layer and chunk) → run ONE chunk inside a `profiled_chunk` zone with no drains, flushed once after it. |
| [common/prefill/profiling/parse_zone_perf.py](../../../common/prefill/profiling/parse_zone_perf.py) | streams the ops CSV once, rebuilds the zone hierarchy from the signpost rows, rolls up ns / ops / bytes / GB/s per zone per device and per layer class. [parse_zone_perf.py](parse_zone_perf.py) here is the GPT-OSS shim. |
| [common/prefill/profiling/visualize_zones.py](../../../common/prefill/profiling/visualize_zones.py) | the render step: text table + standalone HTML with the per-class breakdown, compute/comm/memory split, per-chip spread, op-level detail, device-busy accounting and capture warnings. [visualize_zones.py](visualize_zones.py) here is the shim. |
| [test_zone_profiler.py](test_zone_profiler.py) | the GPT-OSS contract: layer tags, comm/memory keys, env-var names, chunk plan. The mechanism is tested in [common/prefill/tests/test_zone_profiling.py](../../../common/prefill/tests/test_zone_profiling.py). |

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

- **Each layer is read on one chip.** For every sampled layer the report picks the chip with the
  largest layer total and reads all of that layer's zones on that same chip, so the zones of a layer
  add up to its total (summing each zone's own worst chip would mix chips and exceed any real layer).
  The per-chip spread view and the "first layer" table still show each zone's own max / min /
  `skew ms` (max − min): with 32 chips the mesh waits for the slowest, and the skew is what separates
  a genuinely slow CCL from one that is merely waiting on a peer.
- **The compute/communication split undercounts attention's ring traffic on the chunked path.**
  `ring_joint_sdpa` is categorized as compute even though it embodies the SP ring communication: the
  ring rotation and the attention math are fused into one device op and cannot be split. So a headline
  like "sliding layer 69% comm" excludes attention's own ring traffic. Use the one-shot capture
  (`CACHE=0`), where `ag_qkv` / `sdpa` / `sdpa_reduce_scatter` are separate ops, as the reference for
  attention's comm/compute ratio.
- **The full-model number is a projection.** It scales each class's measured per-layer time by its
  full-model layer count (18 sliding + 18 full). The first and last layers carry work no other layer
  does and are not sampled by a short build; the report labels it as a projection for that reason.
- **`GB/s` is bytes-moved ÷ that zone's device time**, from each op's input+output shapes and dtypes.
  Compare against the chip's DRAM ceiling to judge whether a zone is bandwidth-bound.
- **`DRAM%` / `NOC%`** only appear with `NOC_TRACES=1`, which needs tt-npe built separately
  (https://github.com/tenstorrent/tt-npe, then `source tt-npe/ENV_SETUP`). Without it the capture
  still pays for the NoC traces but the columns read `-`.
- The MoE collectives (`dispatch`, `combine`, `moe_reduce`) swing between layers with the expert
  routing — that variance is real. Use 4+ layers when the answer depends on them; compute zones
  reproduce to within a few percent at any layer count.
- **Capture warnings** at the top of the report mean the numbers below are not trustworthy as-is: a
  truncated capture (a zone never closed — the harness died mid-chunk), device ops with no kernel time
  (the device profiler buffer overflowed), host ops or host↔device transfers inside the chunk.
- **"transfers not measured"** is the expected state with the current runtime. The wrapper asks tracy
  for the buffer-copy / `CompileProgram` child calls (`--child-functions`), but no such host zone is
  nested under an op zone any more (the only child recorded is `TT_DNN_DEVICE_OP`), so the CSV carries
  no `*_TT_HOST_FUNC` columns and the report says so instead of claiming a clean capture. The host-op
  check (`OP TYPE != tt_dnn_device`) does not depend on this and still catches CPU fallbacks.

## Gotchas that will bite

**The device profiler buffer.** It holds `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` programs (default
**1000**, which the runner raises to 20000). The harness drains it after compile, after every prefix
layer (through the runtime's layer-completion sink) and after every prefix chunk, but goes **silent
during the profiled chunk** and flushes once after its closing marker — a drain is a blocking sync
that would land in the trace as a multi-second gap. So the chunk's ops must all fit in the buffer at
once; that is what the raised count buys. If they did not, the report says how many ops lost their
kernel time.

**Wall-clock here is meaningless.** Under tracy the host cannot dispatch fast enough, so the device
idles between ops. `DEVICE KERNEL DURATION` and `DEVICE FW DURATION` are on-device and unaffected;
`OP TO OP LATENCY` is not, and the report excludes it. Throughput numbers come from
`tests/galaxy_prefill_kv_pcc.py` (`PREFILL_TPS_ITERS`). The device-busy accounting (kernel + per-op
firmware vs. the chunk's wall-clock) is what quantifies how much of a chunk is dispatch gap — the part
metal trace removes.

**A crashed capture is not a report.** `python3 -m tracy` runs with `--check-exit-code`, so a harness
that dies mid-run fails the wrapper instead of leaving a partial CSV that looks complete; the parser
independently flags zones that never closed.

**Tracy caps a trace at 32K source locations.** A long capture will silently start dropping zones —
use a lower `LEVEL`, fewer `LAYERS`, or `GPTOSS_PROFILE_HOST_ZONES=0` (signposts, which the parser
reads, cost no source locations).

**`SKIP_PREFIX=1` is approximate.** Op shapes and costs are identical, but the attention outputs are
garbage, so the hidden states reaching the MoE router are unrealistic and the expert load imbalance
(`dispatch`, `experts_mm`, `combine`) is not representative. Bring-up only. For the same reason the
harness tiles real tokens rather than generating random ids.

**Zones cost nothing when off, little when on.** With `GPTOSS_PROFILE_ZONES` unset every `zone()` call
returns one shared no-op context manager: ~0.2 ms of host time per 36-layer chunk (~900 enters),
measured on the galaxy host. Armed at LEVEL=2 the signposts cost ~2.2 ms per chunk (~2.7 ms with the
host Tracy zones), all host-side; device timings are unaffected either way.

## Memory and disk — do not scale past ~8 layers

Capture volume scales with `layers × chunks × 32 devices`, and the dangerous step is not the device
run but tracy's post-processing, which loads the exported ops CSV into pandas in one go. On the M3
sibling (more ops/layer than gpt-oss, so a conservative upper bound): 2 layers → ~30 GiB peak RSS,
6 → ~65 GiB, 8 → ~110 GiB, and a full-model run → **OOM-killed after ~50 min, losing the capture**.
The long silent stretch at the end of a run *is* the post-process; it is normal. Stay at or below
8 layers, and clean up the intermediates afterwards (below).

## Clean up afterwards

The ops CSV is the only thing worth keeping — it is what `visualize_zones.py` reads, and the wrapper
has already moved it to `RESULTS_DIR`.

```bash
cd $TT_METAL_HOME
rm -rf generated/profiler/.logs/*
rm -f  generated/profiler/reports/*/profile_log_device.csv
rm -f  generated/profiler/reports/*/tracy_profile_log_host.tracy
rm -f  build/profiler/build_wasm/traces/*.tracy
pkill -f tools/tracy/serve_wasm.py     # tracy leaves a WASM server on :8080
```
