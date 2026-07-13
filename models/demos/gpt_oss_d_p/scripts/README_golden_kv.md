# Generating Golden KV Cache for GPT-OSS

This guide explains how to generate golden reference KV cache traces for validating the GPT-OSS prefill implementation.

## Prerequisites

- **Machine**: 100GB+ RAM recommended (120B model mmap'd via `low_cpu_mem_usage`)
- **Model**: GPT-OSS checkpoint (mxfp4-quantized weights on disk, bf16 compute for attention)
  - Official: `https://huggingface.co/openai/gpt-oss-120b` or `gpt-oss-20b`
- **Prompt**: JSON file or text
- **Time**: Expect several minutes to hours depending on model size and prompt length

## Quick Start

```bash
cd /path/to/tt-metal

export HF_MODEL=/path/to/gpt-oss-120b

# Option 1: Plain tokenization (default — matches GPT-OSS demos and hf_reference_oracle.py)
python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \
    --prompt-json prompt.json \
    --out /mnt/models/gpt-oss-cache/golden/longbook_full \
    --max-tokens 56320

# Option 2: With chat template
python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \
    --prompt-json prompt.json \
    --out /mnt/models/gpt-oss-cache/golden/longbook_full \
    --max-tokens 56320 \
    --chat-template

# Option 3: Quick test
python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \
    --prompt "What are the prime factors of 1?" \
    --out /tmp/gpt_oss_test \
    --max-tokens 1024
```

Wrapper with logging + auto-verify:

```bash
export HF_MODEL=/path/to/gpt-oss-120b
models/demos/gpt_oss_d_p/scripts/run_golden_generation.sh --test
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

## Usage with Prefill Runner

```bash
export PREFILL_TRACE_DIR=/mnt/models/gpt-oss-cache/golden/longbook_full
export DEEPSEEK_V3_HF_MODEL=/path/to/gpt-oss-120b   # prefill_runner env var name

python3 models/demos/gpt_oss_d_p/tt/runners/prefill_runner.py
```

For per-layer KV PCC against the HF oracle `.npy` format, see also:

```bash
python3 models/demos/gpt_oss_d_p/tests/accuracy/hf_reference_oracle.py --save-kv ...
python3 models/demos/gpt_oss_d_p/tests/accuracy/kv_cache_prefill.py --oracle-dir ...
```

The safetensors golden trace here is the portable, MiniMax-style format for `PREFILL_TRACE_DIR` integration.

## Options

| Flag | Description |
|------|-------------|
| `--prompt-json` / `--prompt` | Input prompt |
| `--max-tokens N` | Truncate to N tokens |
| `--chat-template` | Apply chat template (off by default) |
| `--dtype` | Stored dtype (default: bfloat16) |
| `--num-layers N` | Capture first N layers only |
| `--zero-sinks` | Zero attention sinks (diagnostic) |
| `--disable-sliding-window` | Force full attention everywhere (diagnostic) |
| `--model-path` | HF dir, or `$HF_MODEL` / `$DEEPSEEK_V3_HF_MODEL` |

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

# Full run (overnight)
nohup python3 models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py \
    --prompt-json prompt.json \
    --out $OUT_DIR/longbook_full \
    --max-tokens 56320 \
    > generate_golden.log 2>&1 &
```
