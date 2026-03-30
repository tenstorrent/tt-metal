# EAGLE3 draft and base layer sizes

## Base model (LLaMA from HF)

The **base** is typically **DeepSeek-R1-Distill-LLaMA-8B** (full LLaMA from HuggingFace). Standard LLaMA-8B:

| Component | Shape / size |
|-----------|--------------|
| `hidden_size` | 4096 |
| `num_hidden_layers` | 32 |
| `intermediate_size` (MLP) | 11008 (often 14336 in some configs) |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| Per-layer input/output | `(batch, seq, 4096)` → `(batch, seq, 4096)` |
| `output_hidden_states=True` | Returns tuple of 33 tensors: embed output + 32 layer outputs; each `(batch, seq, 4096)` |

For EAGLE3 we take 3 layer hidden states (e.g. layers 1, 29, 57 from config or default 7, 16, 27), last position only: `hidden_states[i][:, -1, :]` each `(batch, 4096)`, concatenated → **multi_layer_hidden** `(batch, 12288)` = 3×4096.

## EAGLE3 draft head (not a full LLaMA)

The **draft** is the EAGLE3 head checkpoint (e.g. **yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B**). It is **not** a full LLaMA; it is a small head with:

| Weight / component | Shape | Description |
|--------------------|--------|-------------|
| `fc.weight` | `(hidden_size, fc_input_size)` | FC **input** size = `fc_input_size`, **output** size = `hidden_size`. So the FC **receives** vectors of size `fc_input_size` (e.g. 12288 = 3×4096) and **outputs** size `hidden_size` (e.g. 4096). |
| `norm.weight` | `(hidden_size,)` | RMSNorm, typically hidden_size=4096. |
| `lm_head.weight` | `(draft_vocab_size, hidden_size)` | Output logits in draft vocab (e.g. 32000); then `d2t` maps to base vocab. |
| `d2t` | `(draft_vocab_size,)` | Offset per draft token to base vocab. |
| **Midlayer** (optional) | | Small transformer block: input `hidden_size`, output `hidden_size`; has self_attn (q/k/v/o), MLP (gate/up/down), layernorms. Uses base’s `embed_tokens` for token embeddings. |

Sizes are read from the checkpoint at load time:

- `hidden_size = norm_weight.shape[0]` → FC **output** dimension
- `fc_input_size = fc_weight.shape[1]` → FC **input** dimension (must be 3×hidden_size for 3-layer fusion)

So for a checkpoint trained with LLaMA-8B base:

- **FC**: **receives** 12288 (`fc_input_size`), **outputs** 4096 (`hidden_size`)  
- **norm**: 4096  
- **lm_head**: (draft_vocab_size, 4096)  
- **midlayer**: 4096 → 4096 (with internal heads, MLP intermediate size, etc.)

## FC usage by depth

- **Depth 0** (first speculative token): input is base’s 3-layer concat `(12288)` or 1-layer fallback `(4096)`. We **always** apply the FC (if 1-layer, we do `FC(cat([h,h,h]))` so first-token logits do not depend on `use_fc_at_every_depth`).
- **Depth ≥ 1**: input is previous step’s output `(4096)`. If `use_fc_at_every_depth` we do `FC(cat([h,h,h]))`; else (paper) we skip FC and use `h` as-is.
