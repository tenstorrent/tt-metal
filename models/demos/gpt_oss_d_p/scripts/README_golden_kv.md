# Generating Golden KV Cache for GPT-OSS

This guide explains how to generate golden reference KV cache traces for validating the GPT-OSS prefill implementation.

## Prerequisites

- **Machine**: ~100GB+ free RAM recommended for weights+activations; 55k one-shot needs tiled attention (below), not 500GB+ for score matrices
- **Model**: GPT-OSS checkpoint (mxfp4-quantized weights on disk, bf16 compute for attention)
  - Official: `https://huggingface.co/openai/gpt-oss-120b` or `gpt-oss-20b`
- **Prompt**: JSON file or text
- **Time**: Expect several minutes to hours depending on model size and prompt length

### Why 55k used to OOM

GPT-OSS sets `_supports_sdpa=False` (attention sinks need concat-then-softmax). Stock eager
attention allocates full `[Hq, S, S]` scores — at S=55k / Hq=64 / bf16 that alone is ~387 GB.

**Fix (MiniMax-style true one-shot):** query-row tiling (`REF_ATTN_Q_CHUNK`, default 256) +
MoE token tiling (`REF_FFN_TOKEN_CHUNK`, default 4096). Same causal/sliding/sinks/MoE math as
one big forward; scores peak at `[H, chunk, S]` and are deleted each chunk.

## Quick Start

```bash
cd /path/to/tt-metal

export HF_MODEL=/path/to/gpt-oss-120b

# Option 1: Plain tokenization (default — matches GPT-OSS demos)
python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \
    --prompt-json prompt.json \
    --out /mnt/models/gpt-oss-cache/golden/longbook_full \
    --max-tokens 56320

# Option 2: Quick test
python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \
    --prompt "What are the prime factors of 1?" \
    --out /tmp/gpt_oss_test \
    --max-tokens 1024
```

## GPT-OSS Specifics

### Separate K/V format (GQA)

Unlike DeepSeek MLA (`kv_post_transform`), GPT-OSS uses separate K and V caches:

- `key_cache_layer_{N}`: post-RoPE K
- `value_cache_layer_{N}`: raw V
- Shape: `[1, num_kv_heads, seq_len, head_dim]` (8 heads × 64 dim for 120B)

### Sliding-window layers

GPT-OSS alternates `sliding_attention` (window=128) and `full_attention` layers. The reference uses `FullKVCapture` to snapshot the **full** sequence K/V before sliding-window truncation — required for long-prefill golden traces.

Use `--disable-sliding-window` only for diagnostics (changes the math vs production).

## Output Format

```
{out_dir}/
    metadata.json
    kv_cache/
        layer_0.safetensors
        layer_1.safetensors
        ...
```

## Usage with Prefill PCC

```bash
export PREFILL_TRACE_DIR=/mnt/models/gpt-oss-cache/golden/longbook_full
export HF_MODEL=/path/to/gpt-oss-120b

python3 models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py
```

The safetensors golden trace is the MiniMax-style format consumed via `PREFILL_TRACE_DIR`.

## Options

| Flag / env | Description |
|------------|-------------|
| `--prompt-json` / `--prompt` | Input prompt |
| `--max-tokens N` | Truncate to N tokens |
| `--chat-template` | Apply chat template (off by default) |
| `--dtype` | Stored dtype only (default: bfloat16); compute is always bfloat16 |
| `--num-layers N` | Capture first N layers only |
| `--zero-sinks` | Zero attention sinks (diagnostic) |
| `--disable-sliding-window` | Force full attention everywhere (diagnostic) |
| `--model-path` | HF dir, or `$HF_MODEL` |
| `REF_ATTN_Q_CHUNK` | Query-row tile size (default 256); lower if still OOM |
| `REF_FFN_TOKEN_CHUNK` | MoE token tile size (default 4096) |

## Tokenization Consistency

| Your Test Uses | Generate Golden With |
|----------------|----------------------|
| Plain `tok(prompt)["input_ids"]` | Default (no `--chat-template`) |
| Chat template | `--chat-template` |

## Storage Estimate

At 56K tokens, 36 layers, 8 KV heads, head_dim=64:

- Per layer: ~2 × (1 × 8 × 56320 × 64 × 2 bytes) ≈ 115 MB
- Total: ~4 GB

## Example Workflow

```bash
export HF_MODEL=/data/models/gpt-oss-120b
export OUT_DIR=/data/golden_traces/gpt_oss

# Short test first
python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \
    --prompt "Hello world" \
    --out $OUT_DIR/test_1k \
    --max-tokens 1024

# Verify
python3 models/demos/gpt_oss_d_p/scripts/verify_golden_kv.py $OUT_DIR/test_1k

# Full 55k one-shot (tiled attn/MoE — no prompt chunking)
nohup env REF_ATTN_Q_CHUNK=256 REF_FFN_TOKEN_CHUNK=4096 \
    python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \
    --prompt-json prompt.json \
    --out $OUT_DIR/longbook_full \
    --max-tokens 56320 \
    > generate_golden.log 2>&1 &
```
