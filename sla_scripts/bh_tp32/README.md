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
| `prefill_profile.sh "tag:ENV=1" ...` | Per-op device time of one prefill chunk of a 1-layer model, per variant. |

Tests in `models/demos/llama3_70b_galaxy/tests/` (run with `MESH_DEVICE=TG pytest <file> -s`):
- `test_prefill_switch_pcc.py`: full 80-layer model; last-token logits with the prefill switches vs all off.
- `test_rs_tune.py` / `test_mm_tune.py`: isolated reduce-scatter and minimal-matmul sweeps (`MM_M=2048|4096`).
- `test_prefill_layer_prof.py`: the 1-layer prefill that `prefill_profile.sh` profiles. `test_mesh_health.py`: quick all-gather check.

The env switches are described in `env.sh` and at their use sites. All of them are off by default.
