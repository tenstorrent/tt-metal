# vLLM Environment Variables

Variables needed when building and running vLLM with TT backend. For global
developer setup (HF_TOKEN, cache paths), see `recipes/developer-setup.md`.

## Critical (must set before install)

| Variable | Purpose | Value |
|---|---|---|
| `VLLM_TARGET_DEVICE` | Selects TT backend | `tt` |

## Runtime

| Variable | Purpose | Example |
|---|---|---|
| `VLLM_RPC_TIMEOUT` | RPC timeout in ms (increase for large models) | `300000`–`900000` |
| `TT_METAL_HOME` | tt-metal repo root (set by workspace activation) | `/path/to/tt-metal` |
| `PYTHONPATH` | Must include tt-metal and vllm | `$TT_METAL_HOME:$WORKSPACE/vllm` |

## Device / topology

| Variable | Purpose | Example |
|---|---|---|
| `MESH_DEVICE` | Device mesh topology | `T3K`, `TG`, `N300`, `(4, 8)` |

## TT config keys (via --override_tt_config)

| Key | Purpose | Values |
|---|---|---|
| `fabric_config` | Network fabric type | `FABRIC_1D`, `FABRIC_1D_RING` |
| `sample_on_device_mode` | Sampling location | `all`, `decode_only` |
| `worker_l1_size` | Per-worker L1 allocation | integer (bytes) |
| `trace_region_size` | Trace buffer size | integer (bytes) |
| `dispatch_core_axis` | Dispatch core layout | `col`, `row` |
