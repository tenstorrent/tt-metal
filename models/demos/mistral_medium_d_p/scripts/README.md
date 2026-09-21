# Mistral-Medium-3.5-128B golden KV cache

`generate_golden_kv_cache.py` produces the golden per-layer KV cache that the prefill PCC checks
read from `$PREFILL_TRACE_DIR` (`metadata.json` + `kv_cache/layer_<i>.safetensors`, MiniMax-style
layout: post-RoPE K in the HF half-split convention and raw V, `[1, num_kv_heads, seq, head_dim]`).

The model implementation is **not** in this directory. The reference is Hugging Face
`transformers`' own `ministral3` decoder (`transformers.models.ministral3.modeling_ministral3`),
run one layer at a time from the mmapped fp8 checkpoint, so nothing here re-implements the math.

## Dependencies

| Package | Version used | Why |
| --- | --- | --- |
| `torch` | 2.11 (>= 2.1 needed for `float8_e4m3fn`) | compute, fp8 dequant |
| `safetensors` | any recent | mmapped shard reads, output files |
| `transformers` | 5.12.1 (>= 5 for `ministral3`) | `Ministral3DecoderLayer`, `Ministral3RotaryEmbedding` (YaRN), tokenizer |

A CUDA GPU is used automatically when available (`--device`). One layer is ~2.8 GB of bf16 weights;
attention goes through `torch.nn.functional.scaled_dot_product_attention`, so the `[96, S, S]`
score matrix is never materialised. Defaults: bf16 compute, bf16 storage, sdpa attention, and a
layer-0 causality self-check.

## Generating the 5k and 10k goldens

```bash
export HF_MODEL=/path/to/Mistral-Medium-3.5-128B
S=models/demos/mistral_medium_d_p/scripts

python3 $S/generate_golden_kv_cache.py --prompt-file $S/prompts/bringup_prompt.txt \
    --repeat-prompt-to 5120  --out $HOME/mistral_medium_golden/s5120

python3 $S/generate_golden_kv_cache.py --prompt-file $S/prompts/bringup_prompt.txt \
    --repeat-prompt-to 10240 --out $HOME/mistral_medium_golden/s10240
```

`--repeat-prompt-to N` tiles the prompt until it tokenizes to at least N tokens and cuts it to
exactly N real tokens (the chat template is applied by default; add `--no-chat-template` if the
device test feeds raw tokens). Use `--token-ids-json ids.json` to reproduce a device test's exact
ids, `--num-layers 4` for a smoke run, `--compute-dtype float32` for a higher-precision golden.

Output size: 88 layers x 2 x 8 heads x 128 x 2 bytes = 352 KiB per token, i.e. ~1.7 GiB at 5120
tokens and ~3.4 GiB at 10240.
