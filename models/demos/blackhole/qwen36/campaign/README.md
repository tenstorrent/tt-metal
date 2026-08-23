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

## Accuracy gate (precision ladder)

`eval_gate.py` + `run_eval.sbatch` (job name `qwen38-eval`, own singleton) — the
go/no-go for every precision rung (bfp8 KV, bfp4 down-proj, ...). Self-contained
(no eval service); runs the same serving path the benches measure. Two stages,
each emitting an `EVAL_JSON {...}` line (appended to
`/data/ayerofieiev/qwen38/eval_results.jsonl`):

- **`test_gpqa_diamond_10`** — GPQA-diamond 10-doc subset, free generation +
  letter extraction. HARD-FAILS on empty responses (serving collapse must not
  masquerade as a low score). Needs the one-time data fetch below; skips
  (loudly) without it.
- **`test_top1_agreement`** — the cheap rung gate: `QWEN38_EVAL_TF_STEPS` (200)
  teacher-forced decode steps over a fixed corpus, greedy top-1 recorded per
  step. Dump the reference config once (`QWEN38_EVAL_DUMP_REF=/data/.../tf_ref.json`),
  then each rung reports agreement % against it (`QWEN38_EVAL_REF=...`); also
  asserts step determinism across identical batch rows.
- **`test_humaneval_subset`** — generation-quality gate on a PUBLIC dataset
  (GPQA is HF-gated and blocked on the cluster; agentic coding is the target
  workload, so HumanEval is the right proxy). Greedy pass@1 over the first 24
  `openai_humaneval` problems (3 full B=8 rounds); each candidate runs the
  official `check()` in a subprocess. Dump the bf16-stack baseline once
  (`QWEN38_HE_DUMP=/data/.../he_ref.json`), then each arm compares on the SAME
  problems (`QWEN38_HE_REF=...`) and HARD-FAILS if pass@1 drops by more than
  `QWEN38_HE_MAX_DROP` (default 1) problems. Same empty-response collapse guard
  as GPQA.

GPQA data is HF-gated and cluster egress is slow — fetch the 10-doc subset once
on a workstation (exact recipe in the `eval_gate.py` docstring) and rsync it to
`/data/ayerofieiev/qwen38/eval_data/gpqa_diamond_10.json` (`QWEN38_GPQA_PATH`
overrides). HumanEval is public (no login): same-docstring recipe writes
`/data/ayerofieiev/qwen38/eval_data/humaneval_24.json` (`QWEN38_HUMANEVAL_PATH`).
`EVAL_K=<pytest -k expr>` narrows a job to one stage.

```
# reference (once per model/mesh)
sbatch --export=ALL,REF=<ref>,QWEN38_EVAL_DUMP_REF=/data/ayerofieiev/qwen38/eval_data/tf_ref.json,QWEN38_HE_DUMP=/data/ayerofieiev/qwen38/eval_data/he_ref.json \
    models/demos/blackhole/qwen36/campaign/run_eval.sbatch
# each precision rung
sbatch --export=ALL,REF=<rung-branch>,<ARM_FLAG>=1,QWEN38_EVAL_REF=/data/ayerofieiev/qwen38/eval_data/tf_ref.json,QWEN38_HE_REF=/data/ayerofieiev/qwen38/eval_data/he_ref.json \
    models/demos/blackhole/qwen36/campaign/run_eval.sbatch
```

## Precision ladder (decode dtype arms)

Four independent env flags, each default OFF; all-off is byte-identical to the
base stack (pinned by `tests/test_precision_flags.py`, host-only — flag plumbing
+ weight-cache-key uniqueness through the real loaders). Every dtype change
writes a DISTINCT weight-cache key (`.bfp4` suffix), so bfp4 arms can never
silently reload bfp8 NFS caches; first flag-on run per mesh pays one-time cache
generation.

| Arm | Flag | What changes | Cache key |
|---|---|---|---|
| KV bfp8 | `QWEN_SDPA_BF8=1` | paged KV cache + Q/KV into SDPA -> bfloat8_b (also required for the 262K-ISL memory budget) | none (runtime alloc) |
| MLP w2 bfp4 | `QWEN36_MLP_DOWN_BF4=1` | down-proj weights -> bfloat4_b (w1/w3 already bfp4) | `mlp.down_proj.weight.bfp4.tp` |
| GDN proj bfp4 | `QWEN36_GDN_BF4=1` | qkvzab in-proj + out-proj weights -> bfloat4_b | `qkvzab.*.bfp4`, `out*.bfp4` |
| LM head bfp4 | `QWEN36_LM_HEAD_BF4=1` | lm_head weights -> bfloat4_b | `output.weight.vshard.bfp4` |

Projected decode-step savings from the Tracy attribution of the 25.99 ms
goal-config step (marginal-wall method; weights-BW-bound ops scale ~with bytes):

| Arm | Attribution today | Mechanism | Projected saving |
|---|---|---|---|
| KV bfp8 | SdpaDecode 3.88 ms (KV-read-BW floor @ ISL 10240 bf16) | KV bytes -47% | −1.5 .. −1.9 ms |
| MLP w2 bfp4 | w2 = 64 of the 192 MLP matmuls @ ~30.8 µs (~24 µs DRAM floor) | w2 weight bytes -50% | −0.7 .. −1.0 ms |
| GDN proj bfp4 | GDN in/out-proj matmuls 2.38 ms @ ~24.8 µs, BW-bound | proj weight bytes -50% | −0.9 .. −1.2 ms |
| LM head bfp4 | lm_head matmul 0.45 ms (~159 MB bfp8, ~345 µs floor) | weight bytes -50% | −0.15 .. −0.2 ms |

All four green: ~ −3.3 .. −4.3 ms -> ~21.7-22.7 ms step (~44-46 t/s/u before MTP
multiplication). Acceptance per arm (user mandate — precision REQUIRES accuracy
evals): top1-agreement gate green AND HumanEval pass@1 drop <= 1 problem vs the
bf16-stack baseline on the same subset; otherwise the arm is rejected.

Per-arm lane (fleet): same-node A/B decode bench at the goal config, then the
eval pair. `<STACK>` = the stack-v2 flags
`QWEN36_GDN_FUSED_DECODE=1,QWEN36_ASYNC_DECODE_STEP=1,QWEN36_ROPE_PERMUTE=1`.

```
# A (flag OFF, control):
sbatch --export=ALL,REF=ayerofieiev/qwen38/precision-ladder,<STACK>,QWEN38_BENCH_SYNTH_STATE=1 \
    models/demos/blackhole/qwen36/campaign/run_bench.sbatch
# B (flag ON, same node):
sbatch --export=ALL,REF=ayerofieiev/qwen38/precision-ladder,<STACK>,<ARM_FLAG>=1,QWEN38_BENCH_SYNTH_STATE=1 \
    models/demos/blackhole/qwen36/campaign/run_bench.sbatch
# eval baseline once (all arms OFF), then per-arm eval with <ARM_FLAG>=1 (commands above)
```

Synth state stays a valid A/B medium for every arm (the KV cache is allocated
in the flagged dtype either way; step timing is state-content-independent —
validated at 25.885 synth vs 25.993 real). Accuracy lives only in the eval
pair, never in the bench.
