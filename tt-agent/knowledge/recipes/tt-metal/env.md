# tt-metal Environment Variables

Variables needed when running tt-metal commands. For global developer setup
(HF_TOKEN, GH_TOKEN, cache paths), see `recipes/developer-setup.md`.

## Device selection

| Variable | Purpose | Values |
|---|---|---|
| `MESH_DEVICE` | Device topology | `N150`, `N300`, `T3K`, `TG`, `DUAL`, `QUAD`, `AUTO` |

Architecture is auto-detected from hardware.

## Model weights

| Variable | Purpose | Example |
|---|---|---|
| `HF_MODEL` | HuggingFace model ID (per-test) | `meta-llama/Llama-3.1-8B-Instruct` |
| `TT_CACHE_PATH` | Cached TT weight tensors | `/mnt/MLPerf/huggingface/tt_cache/...` |

## Build / runtime

| Variable | Purpose | Example |
|---|---|---|
| `TT_METAL_HOME` | Repo root (set by workspace activation) | `/path/to/tt-metal` |
| `PYTHONPATH` | Must include TT_METAL_HOME | `$TT_METAL_HOME` |
