# Qwen3.8-27B campaign benchmark harness

Goal metric: **decode t/s/u at concurrency 8, ISL 10,240, OSL ~1024 on P150x8 / P150x4 — target 100 t/s/u.**
Secondary: TTFT / prefill ms/token at ISL {2048, 10240, 65536}, per-op device-time attribution.

Everything here is additive to `models/demos/blackhole/qwen36/` and reuses its serving
path (per-user chunked prefill + B-wide traced decode). The model under test is selected
purely by `HF_MODEL` (point it at the Qwen3.8-27B weights dir).

## One optimization iteration

```
# on your machine
git push origin my-optimization-branch

# on exabox login node (submission only — never build there)
sbatch --export=ALL,REF=my-optimization-branch \
    models/demos/blackhole/qwen36/campaign/run_bench.sbatch

# when it finishes
tail -5 /data/ayerofieiev/qwen38/results.jsonl        # or:
grep BENCH_JSON /data/ayerofieiev/qwen38/logs/bench-<jobid>.log
```

`run_bench.sbatch` (job name `qwen38-meas`, `--dependency=singleton` → all measurement
and profile jobs serialize on one device owner per user, `--nice=10000`, hard time
limit): checks out `REF` detached, incrementally rebuilds (`SKIP_BUILD=1` for
kernel-only changes — JIT recompiles those), runs the fast PCC gate
(`tests/test_mlp_tp.py::test_mlp_tp`, one layer's weights only), runs `bench_decode.py`
(+ `bench_prefill.py` with `RUN_PREFILL=1`), parses the log, appends to
`/data/ayerofieiev/qwen38/results.jsonl`, and sweeps node-local caches on exit.

Cluster specifics (workspace path, venv python, weights, mesh) come from
`/data/ayerofieiev/qwen38/env.sh` (override with `QWEN38_ENV`). Contract — it must
export: `QWEN38_WORKSPACE`, `QWEN38_PYTHON`, `HF_MODEL`, `MESH_DEVICE`
(optionally `QWEN38_TRACY_BUILD_DIR`, `HF_HOME` on NFS, mesh descriptor vars).
Pick partition/node at submit time: `sbatch -p <part> -w <node> ...`.

## Benchmarks

- **`bench_decode.py`** — the goal metric. Prefills B users to ISL depth once, then
  times each full decode step (input update + traced replay + folded on-device argmax
  readback). Reports min/median/p90 step latency, `tsu_median` (= headline t/s/u),
  aggregate tok/s, TTFT, and the pipelined replay-only device ceiling.
  Knobs: `QWEN38_BENCH_ISL` (10240), `QWEN38_BENCH_BATCH` (8), `QWEN38_BENCH_STEPS`
  (256; use 1024 for the OSL-matched headline), `QWEN38_BENCH_WARMUP` (16),
  `QWEN38_BENCH_SYNTH_STATE=1` (skip prefill — fast decode-only iteration),
  `QWEN38_BENCH_MODE=eager` (per-op profiling), `QWEN38_BENCH_REAL_PROMPT=1`.
- **`bench_prefill.py`** — prefill ms/token + TTFT decomposition (one-time trace
  capture vs device exec vs logits readback) on the single-user chunked path.
  Knobs: `QWEN38_PREFILL_ISLS` ("2048,10240,65536"), `QWEN38_PREFILL_REPEATS`
  (auto: 3 below 16k, 1 above).

Both emit greppable `BENCH_JSON {...}` lines. `parse_bench.py LOG [--append results.jsonl]`
prints one-line summaries and appends records `{kind, ts, ref, node, mesh, hf_model,
config, metrics}`. A `SYNTH` / `EAGER` tag in the summary means the number is an
iteration signal, not a headline; `ROWS_DIVERGED` means identical prompts decoded
differently — do not trust that run.

## Profiling (per-op decode-step attribution)

```
sbatch models/demos/blackhole/qwen36/campaign/profile_decode.sbatch   # eager decode steps
sbatch models/demos/blackhole/qwen36/campaign/profile_prefill.sbatch  # one 2048-token chunk
```

Both need a Tracy build. **Gotcha:** `ENABLE_TRACY` is sticky-OFF in `CMakeCache.txt`
after any `--disable-profiler` build — build a separate tree once
(`./build_metal.sh --enable-profiler --build-dir build_Release_tracy`) and export
`QWEN38_TRACY_BUILD_DIR=build_Release_tracy` in env.sh. The scripts swap the `build`
symlink to it, set `TT_METAL_DEVICE_PROFILER=1`, run a short bench, tar
`generated/profiler` into `/data/ayerofieiev/qwen38/profiles/<timestamp>/`, and restore
the measurement build on exit. Analyze offline with tt-perf-report on the extracted
`ops_perf_results_*.csv` — the question to answer: which of GDN decode / full-attn
decode / MLP / CCL / sampling dominates the decode step.

## Headline run

```
sbatch --export=ALL,REF=<ref>,QWEN38_BENCH_STEPS=1024,RUN_PREFILL=1 \
    models/demos/blackhole/qwen36/campaign/run_bench.sbatch
```
