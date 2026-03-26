# GLM4.7 REAP 218B MoE (REAP) Notes

This note documents the run commands/results for the GLM-4.7 REAP bring-up.

## Key Commands

Set the repo root once (example):

```bash
export TT_METAL_HOME=/path/to/tt-metal   # e.g. ~/sdawle/main/tt-metal
```

### 1) Greedy run (trace + sampling + CCL ring topology + 4 links) - For Single Test Run

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

### 2) Sweep run - Sweep test across Batch and ISL

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

### 3) vLLM (OpenAI-compatible server)

Use the **Tenstorrent vLLM fork** under the tt-metal tree and a TT-capable environment (see [`vllm/tt_metal/README.md`](../../../vllm/tt_metal/README.md): `install-vllm-tt.sh`, `VLLM_TARGET_DEVICE=tt`, `PYTHONPATH` including the tt-metal repo root).

The TT backend turns the Hugging Face architecture **`Glm4MoeForCausalLM`** into **`TTGlm4MoeForCausalLM`**, which loads [`tt/generator_vllm.py`](tt/generator_vllm.py). You must pass the **HF model id** to vLLM and set **`HF_MODEL`** or **`GLM4_MOE_HF_MODEL`** to the same id (required by `initialize_vllm_model`).

**Offline inference test** (from the vLLM tree, with the same TT env as the server example):

```bash
cd "$TT_METAL_HOME/vllm"
export VLLM_TARGET_DEVICE=tt
export PYTHONPATH="$TT_METAL_HOME"
export HF_MODEL=cerebras/GLM-4.7-REAP-218B-A32B
export MESH_DEVICE=TG
export GLM4_MOE_REDUCE_IMPL=native
export GLM4_MOE_EP_REDUCE_DEVICE=1
export GLM4_MOE_CCL_NUM_LINKS=4
export GLM4_MOE_CCL_TOPOLOGY=ring
export GLM4_MOE_KV_CACHE_TT_DTYPE=bf8
python examples/offline_inference_tt.py --model "$HF_MODEL"
```

**Download model weights** (Hugging Face CLI `hf`; run from any directory):

```bash
hf download cerebras/GLM-4.7-REAP-218B-A32B --resume-download
```

**Galaxy TG (32 devices, mesh 8×4)** — example server:

```bash
cd "$TT_METAL_HOME/vllm"
# Use the same Python env where tt-metal + vLLM are installed (see vllm/tt_metal/README.md).
export VLLM_TARGET_DEVICE=tt
export PYTHONPATH="$TT_METAL_HOME"
export HF_MODEL=cerebras/GLM-4.7-REAP-218B-A32B
export MESH_DEVICE=TG
# Optional: match greedy-debug CCL / reduce behavior (see table below).
export GLM4_MOE_REDUCE_IMPL=native
export GLM4_MOE_EP_REDUCE_DEVICE=1
export GLM4_MOE_CCL_NUM_LINKS=4
export GLM4_MOE_CCL_TOPOLOGY=ring
export GLM4_MOE_KV_CACHE_TT_DTYPE=bf8

VLLM_RPC_TIMEOUT=100000 python examples/server_example_tt.py \
  --model cerebras/GLM-4.7-REAP-218B-A32B \
  --max_num_seqs 1 \
  --block_size 64
```

**Smoke request** (after the server prints it is listening, default port **8000**):

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cerebras/GLM-4.7-REAP-218B-A32B",
    "prompt": "Hello",
    "max_tokens": 32,
    "temperature": 0
  }'
```

Notes:

- Weights must be available locally or via Hugging Face (same as the greedy script). Optional: `GLM4_MOE_SNAPSHOT_DIR` for an explicit checkout path; `GLM4_MOE_CACHE_DIR` for TT cache (defaults under `~/.cache/ttnn/models/glm4_moe/vllm`).
- For fabric / L1 / trace region sizing, use `--override_tt_config '{...}'` as documented in [`vllm/tt_metal/README.md`](../../../vllm/tt_metal/README.md) if you hit device init or OOM issues.
- Once vLLM is initialized we need to register the model if not already done so, by modifying the `vllm/platforms/tt.py` and adding the following lines:

```python
# GLM-4.7 MoE (REAP / full) — tt-metal models/experimental/glm4_moe
    _register_model_if_missing(
        ModelRegistry,
        "TTGlm4MoeForCausalLM",
        "models.experimental.glm4_moe.tt.generator_vllm:Glm4MoeForCausalLM",
    )
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


## Performance Snapshot

Command setup: `simulate-context-len=128`, `min-cache-tokens=256`, `max-new-tokens=128`, `batch=1`, `trace-sampling`.


| Config                          | Prefill (s) | Decode total (s) | First token (ms) | Subsequent mean (ms) | Steady state (tok/s) | TTFT (ms) |
| ------------------------------- | ----------- | ---------------- | ---------------- | --------------------- | -------------------- | --------- |
| trace + sampling+ Ring + 4 links| 3.825       | 24.919           | 5552.3           | 153.7                 | 6.51                 | 9377.5    |


Notes:

- **Decode total (s)** is **decode only**: wall time for the greedy decode loop after prefill. It **does not** include prefill. **Prefill (s)** is reported separately.
- **First token (ms)** is the first **decode** step only (often trace-heavy). **TTFT (ms)** is **prefill + that first decode step** (`prefill_s * 1000 + first_decode_step_ms`), i.e. true time-to-first-token from the start of the run.
- **Steady state (tok/s)** uses only steps after the first decode step: `1000 / subsequent_mean_ms`. It intentionally **excludes** the first decode latency and prefill; use **First token** and **TTFT** for those.
- `nanobind` leak warnings appeared at shutdown in both runs; run completed and device closed.
