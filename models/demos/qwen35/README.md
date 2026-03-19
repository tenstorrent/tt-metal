# Qwen3.5-27B

First Qwen3.5 implementation on Tenstorrent hardware. Generates coherent text
on a single P100A Blackhole card at ~2.9 tokens/sec.

Qwen3.5-27B is a hybrid architecture: 48 Gated DeltaNet (linear attention)
layers + 16 standard GQA attention layers with output gate, interleaved every
4th layer.

## Setup

Download the model:
```bash
huggingface-cli download Qwen/Qwen3.5-27B
```

Set the environment variable:
```bash
export HF_MODEL=/path/to/Qwen3.5-27B
```

## Run

```bash
python models/demos/qwen35/demo/demo.py
python models/demos/qwen35/demo/demo.py --prompt "Explain quantum computing"
python models/demos/qwen35/demo/demo.py --max_tokens 500
```

## Run tests

```bash
pytest models/demos/qwen35/tests/ -v
```

## Architecture

| Parameter | Value |
|---|---|
| Layers | 64 (48 DeltaNet + 16 full attention) |
| Hidden dim | 5120 |
| DeltaNet heads | Q/K: 16, V: 48, head_dim: 128 |
| Full attention heads | Q: 24, KV: 4, head_dim: 256 |
| Partial RoPE | 64/256 dims |
| MLP | SwiGLU, intermediate: 17408 |
| Vocab | 248,320 |

## Implementation Notes

The DeltaNet recurrent state is extremely sensitive to compound quantization
error. The recurrence (state decay, retrieve, delta, write, read) runs on
host in float32 while all other operations (projections, conv1d, norms, MLP,
attention) run on device. See [gated_deltanet.py](../../tt_transformers/tt/gated_deltanet.py)
for details.
