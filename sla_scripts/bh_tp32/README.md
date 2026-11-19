# Llama-3.3-70B as one TP32 model on a Blackhole Galaxy

Scripts for the vLLM SLA runs and prefill profiling of `models/demos/llama3_70b_galaxy` on one BH Galaxy (8x4 mesh).
No paths are baked in: set these for your host first.

```bash
export TT_METAL_DIR=/path/to/tt-metal          # this repo, built, python_env with vLLM installed
export HF_MODEL=/path/to/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/path/to/weight-cache       # created on first run if empty
export PLUGIN_DIR=/path/to/vllm-tt-plugin
export OUT=/path/to/results/run1
source sla_scripts/bh_tp32/env.sh                 # best known model switches
```

| Script | What it does |
|---|---|
| `run_sla.sh` | Starts the server, runs every SLA line, prints the table, stops the server with SIGINT. |
| `run_server.sh` / `bench.sh` / `summary.py $OUT` | The same steps, one at a time. |
| `sanity_check.py [url]` | Known-answer check (short questions one at a time and batched, a ~9K-token continuation); `bench.sh` runs it first and stops if it fails. |
| `prefill_profile.sh "tag:ENV=1" ...` | Per-op device time of one prefill chunk of a 1-layer model, per variant. |

Tests in `models/demos/llama3_70b_galaxy/tests/` (run with `MESH_DEVICE=TG pytest <file> -s`):
- `test_prefill_switch_pcc.py`: full 80-layer model; last-token logits with the prefill switches vs all off.
- `test_rs_tune.py` / `test_mm_tune.py`: isolated reduce-scatter and minimal-matmul sweeps (`MM_M=2048|4096`).
- `test_prefill_layer_prof.py`: the 1-layer prefill that `prefill_profile.sh` profiles. `test_mesh_health.py`: quick all-gather check.
- `test_agmm_tune.py`: fused all-gather + minimal_matmul (`all_gather_minimal_matmul_async`) vs the unfused pair at the
  FF2 (and FF1/FF3+SwiGLU) prefill shapes on the 8x4 mesh.

Reset the Galaxy (`tt-smi -glx_reset_auto`) before an accuracy or SLA run whenever the previous process hung, was
hard-killed, failed to start, or ran an experimental op: the chips can keep producing garbage in the next process otherwise.

Decode timing: `LLAMA_DECODE_TIMING=1` logs host time per decode step every 200 steps (`DECODE_TIMING` lines in the server log).

The env switches are described in `env.sh` and at their use sites. All of them are off by default.
