# GLM4 MoE (REAP) Notes

This note documents the current local move and the run commands/results for the GLM-4.7 REAP bring-up.

## Path Move Impact

You moved files from `models/demos/glm4_moe/...` to `models/experimental/glm4_moe/...`.

| Change | Functional impact |
|---|---|
| Move Python modules from `models/demos/glm4_moe` to `models/experimental/glm4_moe` | Breaks imports that still reference `models.demos.glm4_moe.*` (current scripts and TT registration still do). |
| Keep old import paths unchanged | `ModuleNotFoundError` / import failures in vLLM loader and debug scripts. |
| Keep both trees temporarily (or add import aliasing) | Safest transition while updating all callers. |

Current code under `models/experimental/glm4_moe/scripts/debug_run_full_tt_greedy.py` still imports:
- `from models.demos.glm4_moe.tt...`

So unless you update imports and all references, functionality will be affected.

## Key Commands

### 1) Greedy debug run (trace + sampling)

```bash
cd /home/cust-team/sdawle/mike_glm4.7_reap_268b_a32b/tt-metal
export PYTHONPATH=$(pwd)
export GLM4_MOE_REDUCE_IMPL=native
export GLM4_MOE_EP_REDUCE_DEVICE=1
/home/cust-team/sdawle/mike_glm4.7_reap_268b_a32b/tt-metal/python_env/bin/python3 \
  /home/cust-team/sdawle/mike_glm4.7_reap_268b_a32b/tt-metal/models/demos/glm4_moe/scripts/debug_run_full_tt_greedy.py \
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

### 2) Greedy debug run with CCL ring + 4 links

```bash
export GLM4_MOE_CCL_NUM_LINKS=4
export GLM4_MOE_CCL_TOPOLOGY=ring
# (same python command as above)
```

### 3) Sweep run

```bash
python3 models/demos/glm4_moe/scripts/run_sweep_isl_batch.py \
  --out-dir models/demos/glm4_moe/experiments/g1_multilink_4_ring_isl_sweep \
  --timeout 1200 \
  --mesh-rows 8 --mesh-cols 4 \
  --model-id cerebras/GLM-4.7-REAP-218B-A32B \
  --isl 128 512 1024 2048 4096 8192 16384 32768 65536 131072 \
  --batch 1 \
  --verbose-child-output
```

## Environment Variables

| Env var | Meaning | Enabled in latest ring+4 run |
|---|---|---|
| `GLM4_MOE_REDUCE_IMPL=native` | Use on-device all-reduce implementation (trace-safe vs host fallback). | Yes |
| `GLM4_MOE_EP_REDUCE_DEVICE=1` | Keep EP reduce on device to avoid host reads during trace. | Yes |
| `GLM4_MOE_CCL_NUM_LINKS=4` | Number of CCL links for gather/scatter paths. | Yes (ring run) |
| `GLM4_MOE_CCL_TOPOLOGY=ring` | CCL topology (`linear` default, `ring` optional). | Yes (ring run) |
| `GLM4_MOE_EP_L1=1` | Use L1 memory mode for EP path (set by sweep defaults). | Sweep default |
| `GLM4_MOE_PREFILL_CHUNK_SIZE=32768` | Prefill chunk size (helps long-context memory behavior). | Sweep default |
| `GLM4_MOE_EXPERTS_TT_DTYPE=bf4` | Expert weight dtype (memory/perf tradeoff). | Sweep default |

## Performance Snapshot (from your terminal logs)

Command setup: `simulate-context-len=128`, `min-cache-tokens=256`, `max-new-tokens=128`, `batch=1`, `trace-sampling`.

| Config | Prefill (s) | Decode total (s) | Throughput (tok/s) | First token (ms) | Subsequent mean (ms) | TTFT (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Default CCL (`linear`, links=1) | 12.779 | 33.313 | 3.81 | 14050.1 | 152.8 | 26829.5 |
| Ring + 4 links | 3.825 | 24.919 | 5.10 | 5552.3 | 153.7 | 9377.5 |

Notes:
- Biggest gain is in prefill and first token (trace capture heavy path).
- Steady-state subsequent decode latency stayed roughly similar.
- `nanobind` leak warnings appeared at shutdown in both runs; run completed and device closed.
