# Generating Golden KV Cache for MiniMax M3

This guide explains how to generate golden reference KV cache traces for validating the MiniMax M3 prefill implementation.

## Prerequisites

- **Machine**: 500GB+ RAM recommended (model uses ~426GB via mmap)
- **Model**: MiniMax M3 checkpoint (bf16)
  - Official weights: `https://huggingface.co/MiniMaxAI/MiniMax-M3` ✅
  - Any HuggingFace-compatible directory with `config.json` + `*.safetensors`
- **Prompt**: Your test prompt (JSON file or text)
- **Time**: Expect 10-30+ minutes for 60K+ token prompts on CPU

## Quick Start

```bash
# From tt-metal root
cd /path/to/tt-metal

# Set model path (works with official MiniMaxAI/MiniMax-M3 weights)
export HF_MODEL=/path/to/MiniMax-M3

# Option 1: Use your prompt.json (with chat template - default)
python3 models/demos/minimax_m3/scripts/generate_golden_kv_cache.py \
    --prompt-json prompt.json \
    --out /mnt/models/minimax-m3-cache/golden/longbook_full \
    --max-tokens 65536

# Option 2: Raw prompt without chat template (if your tests don't use chat template)
python3 models/demos/minimax_m3/scripts/generate_golden_kv_cache.py \
    --prompt-json prompt.json \
    --out /mnt/models/minimax-m3-cache/golden/longbook_full \
    --max-tokens 65536 \
    --no-chat-template

# Option 3: Quick test with short prompt
python3 models/demos/minimax_m3/scripts/generate_golden_kv_cache.py \
    --prompt "The capital of France is" \
    --out /tmp/minimax_m3_test \
    --max-tokens 1024
```

## Memory Management Strategy

The script uses **mmap** (memory-mapped files) to keep RAM usage manageable:

1. **Model weights**: ~426GB via `low_cpu_mem_usage=True` (mmap)
2. **Activations**: ~10-20GB during forward pass
3. **KV cache**: Saved layer-by-layer (no full accumulation in RAM)

**Peak RAM**: ~450-480GB (should fit in 500GB)

## Output Format

The script generates a trace directory with:

```
{out_dir}/
    metadata.json           # Prompt, token IDs, model info
    kv_cache/
        layer_0.safetensors # Contains separate K and V caches
        layer_1.safetensors
        ...
        layer_N.safetensors
```

Each `layer_X.safetensors` contains **separate K and V caches** (MiniMax M3 uses GQA, not MLA):
- Keys: `key_cache_layer_{X}` and `value_cache_layer_{X}`
- Shape: `[1, num_kv_heads, seq_len, head_dim]`
  - num_kv_heads = 4 (GQA with 4 KV heads)
  - head_dim = 128
  - Example: `[1, 4, 65536, 128]` for 65K tokens
- Dtype: bfloat16

**Note**: This is different from DeepSeek V3's MLA format which uses a single `kv_post_transform` tensor.

## Usage with Prefill Runner

Once generated, use the trace in your prefill tests:

```bash
export PREFILL_TRACE_DIR=/mnt/models/minimax-m3-cache/golden/longbook_full
export PREFILL_STANDALONE_PCC=1

# Run prefill runner with PCC check
python3 models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py
```

The runner's `kv_cache_pcc_check()` will validate your device KV cache against this golden reference.

## Options

### Prompt Input
- `--prompt-json FILE`: JSON file with `{"prompt": "..."}`
- `--prompt TEXT`: Direct prompt text

### Processing
- `--max-tokens N`: Truncate to N tokens (useful for testing)
- `--no-chat-template`: Skip chat template wrapping
- `--dtype {bfloat16,float32,float16}`: Model dtype (default: bfloat16)

### Model
- `--model-path PATH`: HF model directory (or use `$HF_MODEL` env var)
  - Works with official `MiniMaxAI/MiniMax-M3` weights from HuggingFace

## Important: Tokenization Consistency

⚠️ **The KV cache PCC check loads raw token IDs from the trace - it doesn't re-tokenize.**

This means you must generate the golden trace with the **same tokenization** as your tests:

| Your Test Uses | Generate Golden With |
|----------------|----------------------|
| Chat template (default) | Default (no flags) |
| Raw prompts | `--no-chat-template` |

**Example**:
- If your test does `tok.apply_chat_template(...)` → generate golden **without** `--no-chat-template`
- If your test does `tok(prompt)["input_ids"]` → generate golden **with** `--no-chat-template`

The golden trace stores the final token IDs, so whatever tokenization you use is baked in. The runner just loads those IDs and runs them through the model.

## Troubleshooting

### Out of Memory (OOM)

If you hit OOM even with 500GB RAM:

1. **Reduce prompt length**: Use `--max-tokens 32768` or lower
2. **Use swap**: Ensure you have swap space enabled (slow but works)
3. **Split into chunks**: Run multiple passes with different prompt chunks

### Slow Performance

CPU inference is inherently slow:
- **Expected**: 1-5 tokens/sec for 230B+ model
- **60K tokens**: 3-10 hours is normal
- **Recommendation**: Run overnight or use a smaller test prompt first

### Model Loading Fails

Check your model path has:
- `config.json`
- `model.safetensors.index.json`
- `model-xxxxx-of-yyyyy.safetensors` shards

### Wrong KV Shape

If you see shape warnings, the model may use different KV dimensions:
- Check `model.config.kv_lora_rank` and `qk_rope_head_dim`
- Update the `expected_shape` in the script if needed

## Performance Estimates

Based on typical 230B MoE models on CPU:

| Prompt Tokens | CPU Cores | Estimated Time |
|--------------|-----------|----------------|
| 1K           | 32        | 5-10 min       |
| 10K          | 32        | 30-60 min      |
| 60K          | 32        | 3-6 hours      |
| 100K         | 64        | 4-10 hours     |

**Tip**: Start with a small test (`--max-tokens 1024`) to verify everything works before running the full 60K+ token prompt.

## Example: Full Workflow

```bash
# 1. Set up environment
export HF_MODEL=/data/models/MiniMax-M3-dequantized
export OUT_DIR=/data/golden_traces/minimax_m3

# 2. Test with short prompt first
python3 models/demos/minimax_m3/scripts/generate_golden_kv_cache.py \
    --prompt "Hello world" \
    --out $OUT_DIR/test_1k \
    --max-tokens 1024

# 3. If test works, run full prompt (overnight recommended)
nohup python3 models/demos/minimax_m3/scripts/generate_golden_kv_cache.py \
    --prompt-json prompt.json \
    --out $OUT_DIR/longbook_full \
    --max-tokens 65536 \
    > generate_golden.log 2>&1 &

# 4. Monitor progress
tail -f generate_golden.log

# 5. Use in tests
export PREFILL_TRACE_DIR=$OUT_DIR/longbook_full
export PREFILL_STANDALONE_PCC=1
python3 models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py
```

## Notes

- **One-time generation**: Run this once, reuse the trace for all future tests
- **Deterministic**: Same prompt → same KV cache (for debugging)
- **Storage**: Expect ~5-10GB for 60K token trace (61 layers × ~150MB/layer)
