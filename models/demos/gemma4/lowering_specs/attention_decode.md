# Lowering Spec: Attention Decode

Source: `Gemma4TextAttention.forward`, Transformers `v5.5.0`, `modeling_gemma4.py` lines 1125-1240.

## Inputs

| Name | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `hidden_states` | `[1, 1, 1, 2816]` | BF16 | Post-input-layernorm token state. |
| `cos_cache`, `sin_cache` | `[max_seq_len, head_dim]` | BF16 | Layer-type-specific RoPE cache for on-device position lookup. |
| `position_idx` | `[1, 32]` uint32 | UINT32 | Padded decode position for `ttnn.embedding` RoPE lookup. |
| `position_idx_cache` | `[1]` int32 | INT32 | Decode position for paged cache update and SDPA cur-pos. |
| `page_table` | `[1, max_blocks]` | INT32 | Identity page table for batch=1. |
| `kv_cache` | per layer K/V tensors | BF16 | Persistent paged KV cache. |

## Shape Rules

Sliding layers:

| Tensor | Shape |
| --- | --- |
| Q projection | `16 * 256 = 4096` |
| K/V projection | `8 * 256 = 2048` each |
| Q heads | `[1, local_q_heads, 1, 256]` |
| KV heads | `[1, local_kv_heads, cache_len, 256]` |

Full layers:

| Tensor | Shape |
| --- | --- |
| Q projection | `16 * 512 = 8192` |
| K projection | `2 * 512 = 1024` |
| V projection | absent; V reuses K before scale-less V norm |
| Q heads | `[1, local_q_heads, 1, 512]` |
| KV heads | `[1, local_kv_heads, cache_len, 512]` |

On TP=8, sliding KV heads are one per device. Full KV heads are fewer than TP, so the current lowering replicates/assigns the KV head needed by the local Q-head group.

## Pseudocode

```python
layer_type = config.layer_types[layer_idx]
is_sliding = layer_type == "sliding_attention"
head_dim = 256 if is_sliding else 512
num_kv_heads = 8 if is_sliding else 2

q = linear(x, q_proj).view(B, S, 16, head_dim)
k = linear(x, k_proj).view(B, S, num_kv_heads, head_dim)
v = linear(x, v_proj).view(B, S, num_kv_heads, head_dim) if is_sliding else k

q = rms_norm(q, q_norm_weight)
k = rms_norm(k, k_norm_weight)
v = rms_norm(v, weight=None)

q = rope(q, cos[layer_type], sin[layer_type])
k = rope(k, cos[layer_type], sin[layer_type])

cache.update(layer_idx, k, v, position_idx_cache)

k = repeat_kv(k_cache, 16 // num_kv_heads)
v = repeat_kv(v_cache, 16 // num_kv_heads)
scores = matmul(q, transpose(k)) * 1.0
scores += causal_or_sliding_mask(position_idx_cache, sliding_window=1024 if is_sliding else None)
weights = softmax(scores, fp32=True).to(q.dtype)
out = matmul(weights, v)
out = linear(concat_heads(out), o_proj)
out = tensor_parallel_allreduce(out)
```

## TTNN Decomposition

1. Fuse Q/K/V weights per TP shard with K duplicated as V for full attention.
2. `ttnn.linear` for fused QKV.
3. `ttnn.experimental.nlp_create_qkv_heads_decode`.
4. Per-head `ttnn.rms_norm` by flattening `[heads, batch]` into the row dimension.
5. `ttnn.embedding` gathers cos/sin for the current position inside the trace.
6. `ttnn.experimental.rotary_embedding` applies HF-format RoPE.
7. `ttnn.experimental.paged_update_cache` updates persistent K/V cache.
8. `ttnn.transformer.paged_scaled_dot_product_attention_decode` with `scale=1.0` and `sliding_window_size=1024` for sliding layers.
9. `ttnn.experimental.nlp_concat_heads`, output projection, TP allreduce.

## Edge Cases

Full attention head_dim 512 needs a smaller SDPA grid on Blackhole to avoid L1 pressure. Sliding layers can use the device compute grid. RoPE cache generation must not import `transformers.models.gemma4` from the installed environment because current local env may not have Transformers 5.5.
