# LoRA Fine-Tuning for Llama

Fine-tune a Llama model on a toy Shakespeare dataset using LoRA adapters on Tenstorrent hardware.

## Setup

```bash
export TT_METAL_HOME=/path/to/tt-metal
```

## Usage

### Single device (default)

```bash
python3 ./sources/examples/lora_llama/train_lora_llama.py --batch 8 --steps 200
```

### With pretrained weights

```bash
python3 ./sources/examples/lora_llama/train_lora_llama.py \
    -m ./configs/model_configs/tinyllama.yaml \
    --pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batch 8 --steps 200
```

### Multi-device DDP (8 devices)

DDP requires a mesh graph descriptor (MGD) file with ring topology on the DDP axis.
The script validates that the MGD topology matches the requested mesh shape `[1, ddp]`
and that the DDP axis uses `RING`. You can adjust the validation in the script if
a different topology is needed.

For a Wormhole Loud Box (T3K, 8 devices):

```bash
export TT_MESH_GRAPH_DESC_PATH=/path/to/tt-metal/tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto

python3 ./sources/examples/lora_llama/train_lora_llama.py \
    -m ./configs/model_configs/tinyllama.yaml \
    --pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --ddp 8 --batch 8 --steps 200
```

## Model config example

`configs/model_configs/tinyllama.yaml`:

```yaml
transformer_config:
  model_type: "llama"
  num_heads: 32
  num_groups: 4
  embedding_dim: 2048
  dropout_prob: 0.0
  num_blocks: 22
  vocab_size: 32000
  max_sequence_length: 2048
  runner_type: memory_efficient
  theta: 10000.0
  weight_tying: "disabled"
```

Without `-m`, a small 6-layer model is used by default.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-m`, `--model_config` | None | Path to model config YAML |
| `--pretrained` | None | HuggingFace repo ID or local path to `.safetensors` weights |
| `--batch` | 1 | Global batch size (must be divisible by `--ddp`) |
| `--steps` | 500 | Number of training steps |
| `--ddp` | 1 | Number of devices for distributed data parallel |
| `--resume` | None | Path to a LoRA checkpoint `.safetensors` to resume from |
| `--save_every` | 0 | Save LoRA checkpoint every N steps (0 = disabled) |
| `--save_dir` | `checkpoints/` | Directory for LoRA checkpoints |
| `--track_memory` | off | Print memory usage stats after the first iteration |
