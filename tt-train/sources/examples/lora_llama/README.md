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
    -m ./configs/model_configs/tinyllama_bpe.yaml \
    --pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batch 8 --steps 200
```

The architecture and vocabulary come from the checkpoint's `config.json`. The YAML then only
sets training-time knobs: `max_sequence_length`, `runner_type` and `dropout_prob`.

Loading fails, rather than silently cropping or padding a weight, if a tensor in the checkpoint
does not have the shape its `config.json` implies. It also fails if the tokenizer has more tokens
than the checkpoint has embedding rows.

Under tensor parallelism the token embedding is vocab-sharded, pretrained or not: it is the only
layout tied embeddings (e.g. Llama 3.2 1B) can load into.

### Multiple devices

`--mesh_shape` sets the device mesh, and `--dp_axis` / `--tp_axis` pick which of its axes
data parallelism and tensor parallelism run over. For example, DDP over the 8 devices of a
Wormhole Loud Box (T3K):

```bash
export TT_MESH_GRAPH_DESC_PATH=/path/to/tt-metal/tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto

python3 ./sources/examples/lora_llama/train_lora_llama.py \
    -m ./configs/model_configs/tinyllama_bpe.yaml \
    --pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --mesh_shape 1,8 --dp_axis 1 --batch 8 --steps 200
```

Tensor parallelism over two devices is `--mesh_shape 1,2 --tp_axis 1`.

## Model config example

`configs/model_configs/tinyllama_bpe.yaml`:

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
  runner_type: default
  theta: 10000.0
  weight_tying: "disabled"
```

Without `-m`, a small 6-layer model is used by default. With `--pretrained`, the architecture
fields above (including `vocab_size`) are ignored in favour of the checkpoint's `config.json`.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-m`, `--model_config` | None | Path to model config YAML |
| `--pretrained` | None | HuggingFace repo ID, or a local directory with its `.safetensors` weights, `config.json` and tokenizer files; decides the architecture |
| `--batch` | 1 | Global batch size (must be divisible by the DP axis size) |
| `--steps` | 500 | Number of training steps |
| `--mesh_shape` | `1,1` | Device mesh shape, as comma-separated integers |
| `--dp_axis` | -1 | Index of the data-parallel axis in `--mesh_shape` (-1: no DP) |
| `--tp_axis` | -1 | Index of the tensor-parallel axis in `--mesh_shape` (-1: no TP) |
| `--resume` | None | Path to a LoRA checkpoint `.safetensors` to resume from |
| `--save_every` | 0 | Save LoRA checkpoint every N steps (0 = disabled) |
| `--save_dir` | `checkpoints/` | Directory for LoRA checkpoints |
| `--track_memory` | off | Print memory usage stats after the first iteration |
