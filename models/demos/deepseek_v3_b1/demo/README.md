# DeepSeek V3 B1 Demo CLI

This folder contains a CLI for running the current DeepSeek V3 B1 demo.

The demo runs prefill + decode over `DeepSeekV3` sockets and streams decoded text to stdout.

## Requirements

- **Process count:** The pod pipeline expects **4**, **16**, or **64** distributed ranks (single galaxy vs scaleout). Startup fails if the count does not match.
- Slow dispatch mode must be enabled (see below).

## Weight loading (`--weights`)

| Mode | Flag | Extra args | Behavior |
|------|------|------------|----------|
| **Cached tensorbin** | `--weights real` (default) | `--cache-path` (required) | Loads pre-generated `.tensorbin` + `manifest.json` from the weight cache (same layout as `scripts/generate_cache.py`). |
| **HF safetensors + prepare** | `--weights state_dict` | `--model-path` (required) | Uses `LazyStateDict` ([`models/demos/deepseek_v3/utils/lazy_state_dict.py`](../../deepseek_v3/utils/lazy_state_dict.py)) over a local HuggingFace checkpoint (`model.safetensors.index.json` + shards) and runs the same `prepare_*` path as synthetic weights (`StateDictWeightProvider` in [`weight_provider.py`](weight_provider.py)) — no tensorbin cache. |
| **Synthetic** | `--weights synthetic` | — | Random HF-shaped tensors through `prepare_*` for bring-up (no disk weights). |

Optional overrides (all modes): `--dense-layer-id-override`, `--moe-layer-id-override` (see `python -m models.demos.deepseek_v3_b1.demo.cli --help`).

Other common flags: `--prompt`, `--max-new-tokens`, `--tokenizer`, `--fp32` / `--no-fp32`, `--persistent-mode` / `--no-persistent-mode`.

## Running the demo on single galaxy

From the repo root (`tt-metal/`), run the following. Do this after every `tt-smi -glx_reset`:

```bash
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_HOME=$PWD
tt-smi -glx_reset
python tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py
tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml  \
    python -m models.demos.deepseek_v3_b1.demo.cli \
    --weights real \
    --cache-path /mnt/models/deepseek-ai/deepseek_v3_b1_cache
```

**From HuggingFace weights on disk (no cache):**

```bash
tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml  \
    python -m models.demos.deepseek_v3_b1.demo.cli \
    --weights state_dict \
    --model-path /path/to/DeepSeek-V3
```

Adjust `--cache-path` or `--model-path` to your machine. You can add `--prompt`, `--max-new-tokens`, etc.

## Running the demo on single-pod

```bash
./tools/scaleout/exabox/recover.sh --hosts bh-glx-d06u08,bh-glx-d06u02,bh-glx-d05u08,bh-glx-d05u02 # Modify with your 4 hosts
python3 models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py models/demos/deepseek_v3_b1/scaleout_configs/blitz_pipeline_config_single_pod.yaml # Generate pipeline config for 1 pod
tt-run --mpi-args "--map-by rankfile:file=blitz_decode_pipeline_rank_file_single_pod --bind-to hwt:overload-allowed --host bh-glx-d05u02:4,bh-glx-d05u08:4,bh-glx-d06u02:4,bh-glx-d06u08:4 --tag-output" --rank-binding blitz_decode_pipeline_rank_binding_single_pod.yaml \
    python -m models.demos.deepseek_v3_b1.demo.cli \
    --weights real \
    --cache-path /mnt/models/deepseek-ai/deepseek_v3_b1_cache
```

Use `--weights state_dict --model-path ...` similarly if you want runtime preparation from safetensors instead of a cache.
