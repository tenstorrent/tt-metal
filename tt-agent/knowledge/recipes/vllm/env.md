# vLLM Environment Variables

Set these when submitting device jobs.
For developer setup (HF_TOKEN, cache paths), see `recipes/developer-setup.md`.

## Required for every vLLM job

| Variable | Source | Example |
|---|---|---|
| `TT_METAL_HOME` | Workspace-detect | `/localdev/user/workspaces/feat/tt-metal` |
| `PYTHONPATH` | TT_METAL_HOME + vllm path | `/localdev/user/workspaces/feat/tt-metal:/localdev/user/workspaces/feat/vllm` |
| `VLLM_TARGET_DEVICE` | Always `tt` | `tt` |
| `HF_MODEL` | User request — **always required by tt model loader** | `meta-llama/Llama-3.1-8B-Instruct` |
| `HF_HOME` | Workspace-detect | `/localdev/user/hf_data` |
| `HF_TOKEN` | Workspace-detect (when model download needed) | Developer's token |

## Optional / situational

| Variable | Source | Example |
|---|---|---|
| `MESH_DEVICE` | User request or test requirements | `T3K`, `TG`, `N300`, `(4, 8)` |
| `VLLM_RPC_TIMEOUT` | Recipe default, increase for large models | `300000`–`900000` |

## TT config overrides (CLI flags, not env vars)

Pass via `--override_tt_config '{...}'` on the server command:

| Key | Purpose | Values |
|---|---|---|
| `fabric_config` | Network fabric type | `FABRIC_1D`, `FABRIC_1D_RING` |
| `sample_on_device_mode` | Sampling location | `all`, `decode_only` |
| `worker_l1_size` | Per-worker L1 allocation | integer (bytes) |
| `trace_region_size` | Trace buffer size | integer (bytes) |
