# Qwen3.5-27B on tt-metal

First working Qwen3.5 implementation on Tenstorrent hardware.
Generates coherent text on a single P100A Blackhole card.

## Architecture

Qwen3.5-27B is a hybrid linear-attention + softmax-attention model.
It has 64 decoder layers total:

- **48 DeltaNet layers** (linear attention with gated delta rule recurrence)
- **16 GatedAttention layers** (standard grouped-query attention with output gate)

Full attention layers are placed at every 4th layer (layers 3, 7, 11, ...).

Key dimensions:

| Parameter | Value |
|---|---|
| hidden_size | 5120 |
| num_attention_heads (full attn) | 24 |
| num_kv_heads (full attn) | 4 |
| head_dim (full attn) | 256 |
| partial_rotary_factor | 0.25 (64/256 dims rotated) |
| linear_num_value_heads (DeltaNet) | 48 |
| linear_num_key_heads (DeltaNet) | 16 |
| linear_key_head_dim | 128 |
| linear_value_head_dim | 128 |
| intermediate_size (MLP) | 17408 |
| vocab_size | 248320 |
| rope_theta | 1000000 |
| conv_kernel_size (DeltaNet) | 4 |

## How to run

```bash
export HF_MODEL=/path/to/Qwen3.5-27B
python test_qwen35_generate.py
```

The script runs token-by-token decode on a single P100A device.

## DRAM budget (P100A = 28 GB)

| Component | dtype | Size |
|---|---|---|
| DeltaNet projections (48 layers) | bfp8 | 5.4 GB |
| MLP w1+w2+w3 (64 layers) | bfp8 | 17.1 GB |
| Attention QKV+WO+gate (16 layers) | bf16 | 2.2 GB |
| Other (norms, KV cache, conv, RoPE) | bf16 | ~0.3 GB |
| **Total** | | **~25 GB** |

Host embedding + CPU LM head save ~2.4 GB DRAM.

## DeltaNet host recurrence

The DeltaNet recurrent state is extremely sensitive to compound quantization
error. With bfp8 projection weights, upstream layers produce slightly wrong
Q/K/V values. The persistent recurrent state amplifies these errors across
tokens, causing state norms to explode (e.g. 9 -> 205,000 in 12 steps).

The fix: run the recurrence (decay, retrieve, delta, write, read) on host
in float32 while keeping projections, conv1d, norms, and output projection
on device. The host roundtrip is ~50 KB per layer per token.

Future optimization: move recurrence back to device once tt-metal supports
fp32 accumulation in matmul for small tensors (128x128), or once the
upstream layers can run with sufficient precision (bf16 DeltaNet weights
require bfp4 MLP to fit in DRAM, but bfp4 MLP is currently too lossy).

## Performance

Current: **~2.9 tok/s** decode on single P100A.

Bottleneck is host roundtrips:
- DeltaNet recurrence: 5 tensor transfers per layer (Q, K, V, gates, output)
- GatedAttention: custom RoPE + host KV cache update
- Embedding: host lookup, LM head: CPU matmul

## PCC validation results

| Layer type | PCC | Notes |
|---|---|---|
| DeltaNet (layer 0) | 0.999 | Single-token decode, zero initial state |
| GatedAttention (layer 3) | 0.999991 | Single-token decode |
| MLP | 0.999955 | Single forward pass |

Run PCC tests:

```bash
pytest models/demos/qwen35/tests/test_pcc.py -v
```

## Modified / added files

```
models/tt_transformers/tt/gated_deltanet.py       # DeltaNet linear attention (host recurrence)
models/tt_transformers/tt/gated_attention.py       # GatedAttention with output gate + partial RoPE
models/tt_transformers/tt/qwen35_decoder.py        # Decoder block wrapping DeltaNet/GatedAttention + MLP
models/tt_transformers/tt/model_config.py          # Qwen3.5 config params, program_config overrides
models/tt_transformers/tt/load_checkpoints.py      # HF->meta key conversion for Qwen3.5
models/tt_transformers/tt/QWEN35_README.md         # This file
models/demos/qwen35/demo/demo.py                   # End-to-end generation demo
models/demos/qwen35/tests/test_pcc.py              # PCC validation tests
```
