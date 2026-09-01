# Mistral Small 4 — PP=4 x (8,1) vs single-rank prefill perf harness

Measures the same 36-layer model two ways **through the same runner and producer**, so topology is the
only variable:

* **single-rank** — SP=8 x TP=4 on one 8x4 mesh (`2d_torus_xy`)
* **PP=4** — four `[8,1]` column sub-meshes, hidden state handed stage-to-stage over a real
  device-to-device `ttnn` MeshSocket on fabric (`2d_torus_y`)

Results and analysis: **`models/demos/deepseek_v3_d_p/docs/MISTRAL4_PP4_VS_SINGLE_RANK.md`**.

## Prerequisites

A 32-chip Blackhole galaxy, a profiler-enabled build (`ENABLE_TRACY=ON`), and three artifacts that are
NOT in the repo. Point `env.sh` at your copies (every path there is overridable):

| artifact | what it is |
|---|---|
| `MISTRAL4_HF_MODEL` | the HF safetensors checkpoint |
| `M4_CACHE_8x4` / `M4_CACHE_8x1` | TTNN weight caches, `{name}_{arch}_{Ndev}/{sp}x{tp}` |
| `GOLDEN_5120` | golden KV trace (only for the correctness run) |

**The weight caches are the expensive prerequisite** (~65 GB each). Two things save a rebuild:
the device-count part of the cache path is *namespacing only* — `32dev/8x1` and `8dev/8x1` files are
byte-identical, so `cp -al` an existing 8x1 cache into an `8dev` tree — and cache keys are **global**
layer indices, so all four PP ranks share one directory that must hold layers 0..35.

## Quick start

```bash
S=models/demos/deepseek_v3_d_p/tests/perf/pp4
$S/preflight.sh                 # chips, build, tools, caches; warns on per-machine items

# REQUIRED on any new galaxy: the [8,1] column -> device map is per-machine
$PY $S/gen_pp4_binding.py                 # -> ..._torus_y.<hostname>.yaml
$PY $S/gen_pp4_binding.py --profile       # -> ..._torus_y_profile.<hostname>.yaml

$S/run_pp4_probe.sh             # ~2 min, no weights: topology + D2D sockets + shutdown
$S/run_matrix.sh                # the 16-cell table (~50 min); completed cells are skipped
DEEP_CHUNKS=8 $S/run_single_layer_profile.sh pp4_deep    # per-stage Tracy capture
DEEP_CHUNKS=8 $S/run_single_layer_profile.sh 1rank_deep
```

**Regenerating the binding is not optional.** The checked-in `TT_VISIBLE_DEVICES` lists were read off
one specific galaxy, and two 32-chip galaxies enumerate differently (verified). A wrong map does
**not** error — it builds a pipeline whose stages are not columns and quietly reports wrong numbers.
The runners prefer `<binding>.<hostname>.yaml` automatically when present.

## Scripts

| file | purpose |
|---|---|
| `env.sh` | all paths; source it or override the vars |
| `preflight.sh` | environment validation; run first on any new machine |
| `gen_pp4_binding.py` | derive the per-galaxy `[8,1]` column map and emit a host-specific binding |
| `probe_columns.py` | print the raw column -> device map from a live 8x4 mesh |
| `probe_pp4_d2d.py` / `run_pp4_probe.sh` | weightless 4-rank topology + D2D + shutdown check |
| `run_pp4_model.sh` | one run: launch N ranks under `tt-run`, drive with the producer |
| `run_matrix.sh` | the {1rank,pp4} x {4 ISLs} x {ttft,thru} sweep |
| `run_single_layer_profile.sh` | Tracy/device-profiler captures, incl. chunked-at-depth `_deep` modes |
| `tracy_rank_wrapper.sh` | per-MPI-rank Tracy wrapper (port + output dir from the rank) |
| `run_1rank_smoke.sh` | single-rank KV-PCC correctness gate against a golden trace |
| `check_board.sh` | is the fabric still mappable? used by `run_matrix.sh` to abort vs cascade |
| `analyze_pp.py` | steady-state throughput from the last rank's chunk intervals |
| `analyze_ttft.py` | single-request latency; refuses multi-request logs |
| `analyze_longctx2.py` | chunked long-context `total` + `steady` metrics |
| `analyze_layer_budget.py` | per-layer device-time budget from an `ops_perf_results` CSV |
| `analyze_kv_ramp.py` | per-chunk op durations -> the cost of KV depth |

## Gotchas that cost real time

* **`InboundSocketServiceSyncOperation` is ~99% of device time in a PP stage capture and is not
  transport** — it is the receiver blocking on upstream. `analyze_layer_budget.py` excludes it.
* **One CSV row per device.** A stage spans 8 chips running concurrently; an op costs the max across
  devices, not the sum.
* **A single-request latency run of a NEW code path includes cold kernel JIT** (seen: 11.8 s vs
  0.66 s, 86% vs 100% cache hits). Run such a cell twice and use the warm one.
* **`tt-smi -r` does not recover every galaxy** (CPLD < v1.16 -> use `tt-smi -glx_reset`, ~90 s).
  `run_matrix.sh` does this automatically and aborts rather than cascading if still unmappable.
* **Never edit a script while a run is executing it** — bash reads by byte offset and a rewrite
  mid-run makes it resume at the wrong place, with baffling syntax errors on untouched lines.
* Outputs are large: the matrix ~5 MB, but a `_deep` Tracy capture is **several GB per rank**. They
  land under the repo root (`mistral4_perf_<hostname>/`, `mistral4_perf_profile/`) — keep them out of
  git.
