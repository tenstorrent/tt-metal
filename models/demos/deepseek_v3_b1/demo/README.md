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

## Host setup

Before running any of the commands below, set `$HOSTS` and `$HOSTSP` for the hosts you are using. `$HOSTS` is a comma-separated list of hostnames; `$HOSTSP` appends `:4` to each hostname (slots per host for MPI). For example, for a single pod using four galaxies:

```bash
export HOSTS=bh-glx-d05u08,bh-glx-d05u02,bh-glx-d06u02,bh-glx-d06u08
export HOSTSP=bh-glx-d05u08:4,bh-glx-d05u02:4,bh-glx-d06u02:4,bh-glx-d06u08:4
```

For a full SP4 (16 galaxies):

```bash
export HOSTS=bh-glx-d03u02,bh-glx-d03u08,bh-glx-d04u02,bh-glx-d04u08,bh-glx-d06u08,bh-glx-d06u02,bh-glx-d05u08,bh-glx-d05u02,bh-glx-d07u02,bh-glx-d07u08,bh-glx-d08u02,bh-glx-d08u08,bh-glx-d09u02,bh-glx-d09u08,bh-glx-d10u02,bh-glx-d10u08
export HOSTSP=bh-glx-d03u08:4,bh-glx-d03u02:4,bh-glx-d04u02:4,bh-glx-d04u08:4,bh-glx-d06u08:4,bh-glx-d06u02:4,bh-glx-d05u08:4,bh-glx-d05u02:4,bh-glx-d07u02:4,bh-glx-d07u08:4,bh-glx-d08u02:4,bh-glx-d08u08:4,bh-glx-d09u02:4,bh-glx-d09u08:4,bh-glx-d10u02:4,bh-glx-d10u08:4
```

See `source.sh` in the repo root for more host configurations.

## Single galaxy demo

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml \
    python -m models.demos.deepseek_v3_b1.demo.cli \
    --max-new-tokens 2048 \
    --weights real \
    --cache-path /mnt/models/deepseek-ai/cache-2026-03-22 \
    --prompt "Solve this step by step: Design a cache for sharded LLM weights across multiple hosts with local NVMe and shared NFS. Minimize startup latency, avoid duplicate reads, support cache invalidation, and explain tradeoffs between per-host caches, content-addressable storage, and pre-sharded artifacts." \
    --model-path /mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized
```

## Pod demo

First generate the pipeline config for a single pod:

```bash
python3 models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py \
    models/demos/deepseek_v3_b1/scaleout_configs/blitz_pipeline_config_single_pod_3_4.yaml
```

Then run the demo:

```bash
TT_METAL_DPRINT_CORES=all TT_METAL_SLOW_DISPATCH_MODE=1 tt-run \
    --mpi-args "--map-by rankfile:file=blitz_decode_pipeline_rank_file_single_pod --bind-to hwt:overload-allowed --host $HOSTSP --tag-output" \
    --rank-binding blitz_decode_pipeline_rank_binding_single_pod.yaml \
    python -m models.demos.deepseek_v3_b1.demo.cli \
    --max-new-tokens 2048 \
    --weights real \
    --cache-path /mnt/models/deepseek-ai/cache-2026-03-22 \
    --prompt "Solve this step by step: Design a cache for sharded LLM weights across multiple hosts with local NVMe and shared NFS. Minimize startup latency, avoid duplicate reads, support cache invalidation, and explain tradeoffs between per-host caches, content-addressable storage, and pre-sharded artifacts." \
    --model-path /mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized
```

## Full SP4

Generate the superpod pipeline config:

```bash
python3 models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py \
    models/demos/deepseek_v3_b1/scaleout_configs/blitz_pipeline_config_superpod.yaml
```

Then run the demo across all 16 galaxies:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run \
    --mpi-args "--map-by rankfile:file=blitz_decode_pipeline_rank_file_superpod --bind-to hwt:overload-allowed --host $HOSTSP --tag-output" \
    --rank-binding blitz_decode_pipeline_rank_binding_superpod.yaml \
    python -m models.demos.deepseek_v3_b1.demo.cli \
    --max-new-tokens 2048 \
    --weights real \
    --cache-path /mnt/models/deepseek-ai/cache-2026-03-22 \
    --prompt "Solve this step by step: Design a cache for sharded LLM weights across multiple hosts with local NVMe and shared NFS. Minimize startup latency, avoid duplicate reads, support cache invalidation, and explain tradeoffs between per-host caches, content-addressable storage, and pre-sharded artifacts." \
    --model-path /mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized
```
