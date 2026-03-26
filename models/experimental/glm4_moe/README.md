# GLM4.7 REAP 218B MoE (REAP) Notes

This note documents the run commands/results for the GLM-4.7 REAP bring-up.

## Key Commands

Set the repo root once (example):

```bash
export TT_METAL_HOME=/path/to/tt-metal   # e.g. /home/cust-team/sdawle/mike_glm4.7_reap_268b_a32b/tt-metal
```

### 1) Greedy run (trace + sampling + CCL ring topology + 4 links)

```bash
cd "$TT_METAL_HOME"
export PYTHONPATH="$TT_METAL_HOME"
export GLM4_MOE_REDUCE_IMPL=native
export GLM4_MOE_EP_REDUCE_DEVICE=1
export GLM4_MOE_CCL_NUM_LINKS=4
export GLM4_MOE_CCL_TOPOLOGY=ring
"$TT_METAL_HOME/python_env/bin/python3" \
  "$TT_METAL_HOME/models/experimental/glm4_moe/scripts/debug_run_full_tt_greedy.py" \
  --model-id cerebras/GLM-4.7-REAP-218B-A32B \
  --prompt "Summarize the following document. " \
  --simulate-context-len 128 \
  --min-cache-tokens 256 \
  --max-new-tokens 128 \
  --batch-size 1 \
  --max-batch-size 32 \
  --mesh-rows 8 \
  --mesh-cols 4 \
  --kv-cache-dtype bf8 \
  --enable-trace \
  --trace-mode sampling
```

### 2) Sweep run

```bash
cd "$TT_METAL_HOME"
export PYTHONPATH="$TT_METAL_HOME"
python3 models/experimental/glm4_moe/scripts/run_sweep_isl_batch.py \
  --out-dir models/experimental/glm4_moe/experiments/g1_multilink_4_ring_isl_sweep \
  --timeout 1200 \
  --mesh-rows 8 --mesh-cols 4 \
  --model-id cerebras/GLM-4.7-REAP-218B-A32B \
  --isl 128 512 1024 2048 4096 8192 16384 32768 65536 131072 \
  --batch 1 \
  --verbose-child-output
```

## Environment Variables


| Env var                             | Meaning                                                                | Enabled in latest ring+4 run |
| ----------------------------------- | ---------------------------------------------------------------------- | ---------------------------- |
| `GLM4_MOE_REDUCE_IMPL=native`       | Use on-device all-reduce implementation (trace-safe vs host fallback). | Yes                          |
| `GLM4_MOE_EP_REDUCE_DEVICE=1`       | Keep EP reduce on device to avoid host reads during trace.             | Yes                          |
| `GLM4_MOE_CCL_NUM_LINKS=4`          | Number of CCL links for gather/scatter paths.                          | Yes (ring run)               |
| `GLM4_MOE_CCL_TOPOLOGY=ring`        | CCL topology (`linear` default, `ring` optional).                      | Yes (ring run)               |
| `GLM4_MOE_EP_L1=1`                  | Use L1 memory mode for EP path (set by sweep defaults).                | Sweep default                |
| `GLM4_MOE_PREFILL_CHUNK_SIZE=32768` | Prefill chunk size (helps long-context memory behavior).               | Sweep default                |
| `GLM4_MOE_EXPERTS_TT_DTYPE=bf4`     | Expert weight dtype (memory/perf tradeoff).                            | Sweep default                |


## Performance Snapshot (from your terminal logs)

Command setup: `simulate-context-len=128`, `min-cache-tokens=256`, `max-new-tokens=128`, `batch=1`, `trace-sampling`.


| Config                          | Prefill (s) | Decode total (s) | First token (ms) | Subsequent mean (ms) | Steady state (tok/s) | TTFT (ms) |
| ------------------------------- | ----------- | ---------------- | ---------------- | --------------------- | -------------------- | --------- |
| Default CCL (`linear`, links=1) | 12.779      | 33.313           | 14050.1          | 152.8                 | 6.54                 | 26829.5   |
| Ring + 4 links                  | 3.825       | 24.919           | 5552.3           | 153.7                 | 6.51                 | 9377.5    |


Notes:

- **Decode total (s)** is **decode only**: wall time for the greedy decode loop after prefill (`decode_s` in `debug_run_full_tt_greedy.py`). It **does not** include prefill. **Prefill (s)** is reported separately.
- **First token (ms)** is the first **decode** step only (often trace-heavy). **TTFT (ms)** is **prefill + that first decode step** (`prefill_s * 1000 + first_decode_step_ms`), i.e. true time-to-first-token from the start of the run.
- **Steady state (tok/s)** uses only steps after the first decode step: `1000 / subsequent_mean_ms`. It intentionally **excludes** the first decode latency and prefill; use **First token** and **TTFT** for those.
- Biggest gain is in prefill and first token (trace capture heavy path).
- Steady-state subsequent decode latency stayed roughly similar (see Subsequent mean).
- `nanobind` leak warnings appeared at shutdown in both runs; run completed and device closed.
