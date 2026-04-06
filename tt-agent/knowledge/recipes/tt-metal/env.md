# tt-metal Environment Variables

Set these when submitting device jobs.
For developer setup (HF_TOKEN, cache paths), see `recipes/developer-setup.md`.

| Variable | Source | Example |
|---|---|---|
| `TT_METAL_HOME` | Workspace-detect | `/localdev/user/workspaces/feat/tt-metal` |
| `PYTHONPATH` | Same as TT_METAL_HOME | `/localdev/user/workspaces/feat/tt-metal` |
| `MESH_DEVICE` | User request or test requirements | `N150`, `N300`, `T3K`, `TG`, `AUTO` |
| `HF_MODEL` | User request (when test needs a model) | `meta-llama/Llama-3.1-8B-Instruct` |
| `TT_CACHE_PATH` | Test config (if cached weights exist) | `/mnt/MLPerf/huggingface/tt_cache/...` |
| `HF_HOME` | Workspace-detect | `/localdev/user/hf_data` |
| `HF_TOKEN` | Workspace-detect (when model download needed) | Developer's token |
