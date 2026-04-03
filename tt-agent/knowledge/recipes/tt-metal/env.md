# tt-metal Environment Variables

## Device selection

| Variable | Purpose | Values |
|---|---|---|
| `MESH_DEVICE` | Device topology | `N150`, `N300`, `T3K`, `TG`, `DUAL`, `QUAD`, `AUTO` |
| `--tt-arch` | Architecture (pytest CLI) | `wormhole_b0`, `blackhole`, `grayskull` (auto-detected) |
| `FAKE_DEVICE` | Run without hardware | `1` (limited tests only) |

## Model weights

| Variable | Purpose | Example |
|---|---|---|
| `HF_MODEL` | HuggingFace model ID | `meta-llama/Llama-3.1-8B-Instruct` |
| `HF_TOKEN` | HuggingFace auth token | `hf_...` |
| `HF_HOME` | HuggingFace cache dir | `/mnt/MLPerf/huggingface` |
| `TT_CACHE_PATH` | Cached TT weight tensors | `/mnt/MLPerf/huggingface/tt_cache/...` |

## Build / runtime

| Variable | Purpose | Example |
|---|---|---|
| `TT_METAL_HOME` | Repo root (usually auto-detected) | `/path/to/tt-metal` |
| `ARCH_NAME` | Silicon architecture | `wormhole_b0`, `blackhole` |
