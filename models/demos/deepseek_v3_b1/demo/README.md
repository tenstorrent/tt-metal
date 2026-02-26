# DeepSeek V3 B1 Demo CLI

This folder contains a CLI for running the current DeepSeek V3 B1 demo.

The demo runs prefill + decode over `DeepSeekV3` sockets and streams decoded text to stdout.

## Requirements

- **Mesh shape:** The demo requires a **4×2** mesh. The CLI will fail at startup if the mesh shape does not match.
- Slow dispatch mode must be enabled (see below).

## Running the demo on single galaxy

From the repo root (`tt-metal/`), run the following. Do this after every `tt-smi -glx_reset`:

```bash
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_HOME=$PWD
tt-smi -glx_reset
python tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py
tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml  \
    python -m models.demos.deepseek_v3_b1.demo.cli --cache-path /mnt/models/deepseek-ai/deepseek_v3_b1_cache --layer-id-offset 4
```

Adjust `--cache-path` to your prepared weight cache location and `--layer-id-offset` as needed. You can also pass `--prompt` and `--max-new-tokens`, for example:

## Running the demo on single-pod:

```bash
./tools/scaleout/exabox/recover_4x32.sh bh-glx-d06u08,bh-glx-d06u02,bh-glx-d05u08,bh-glx-d05u02 # Modify with your 4 hosts
python3 models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py models/demos/deepseek_v3_b1/scaleout_configs/blitz_pipeline_config_single_pod.yaml # Generate pipeline config for 1 pod
tt-run --mpi-args "--map-by rankfile:file=blitz_decode_pipeline_rank_file_single_pod --bind-to hwt:overload-allowed --host bh-glx-d05u02:4,bh-glx-d05u08:4,bh-glx-d06u02:4,bh-glx-d06u08:4 --tag-output" --rank-binding blitz_decode_pipeline_rank_binding_single_pod.yaml \
    python -m models.demos.deepseek_v3_b1.demo.cli --cache-path /mnt/models/deepseek-ai/deepseek_v3_b1_cache --layer-id-offset 4
```
