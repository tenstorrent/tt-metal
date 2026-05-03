# Lowering Spec: Router And MoE

Sources: `Gemma4TextRouter.forward` and `Gemma4TextExperts.forward`, Transformers `v5.5.0`, `modeling_gemma4.py` lines 1243-1316.

## Inputs

| Name | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `router_input` | `[1, 1, S, 2816]` | BF16 | Pre-feedforward residual, before pre-FF norm. |
| `expert_input` | `[1, 1, S, 2816]` | BF16 | `pre_feedforward_layernorm_2(router_input)`. |
| `router.scale` | `[2816]` | BF16 | Learned per-hidden-channel router scale. |
| `router.proj.weight` | `[128, 2816]` | BF16 | Expert score projection. |
| `per_expert_scale` | `[128]` | BF16 | Post-renormalization expert scale. |
| `gate_up_proj` | `[128, 1408, 2816]` | BF16 | First 704 rows gate, second 704 rows up. |
| `down_proj` | `[128, 2816, 704]` | BF16 | Expert down projection. |

## Router Pseudocode

```python
x = rms_norm(router_input, weight=None)
x = x * router.scale * (2816 ** -0.5)
scores = linear(x, router.proj.weight)       # [tokens, 128]
router_probs = softmax(scores, dim=-1)
top_k_weights, top_k_index = topk(router_probs, k=8, dim=-1)
top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
top_k_weights = top_k_weights * per_expert_scale[top_k_index]
dense_routing = zeros_like(router_probs).scatter(-1, top_k_index, top_k_weights)
```

Important: the top-k values are already probabilities from the full 128-way softmax. They must be divided by their sum. Applying another softmax to the selected values is not equivalent.

## Expert Pseudocode

```python
final = zeros_like(expert_input_flat)
for expert in active_experts(top_k_index):
    token_idx, top_k_pos = positions_for_expert(expert)
    current = expert_input_flat[token_idx]
    gate, up = linear(current, gate_up_proj[expert]).chunk(2, dim=-1)
    y = gelu_pytorch_tanh(gate) * up
    y = linear(y, down_proj[expert])
    y = y * top_k_weights[token_idx, top_k_pos, None]
    final.index_add_(0, token_idx, y)
```

## TTNN Decomposition

Router:

1. Local RMSNorm without scale.
2. Broadcast multiply by learned scale and scalar root size.
3. `ttnn.linear`.
4. `ttnn.softmax` over all experts.
5. `ttnn.topk(k=8)`.
6. `ttnn.sum(..., dim=-1, keepdim=True)` plus reciprocal/multiply for HF renormalization.
7. `ttnn.scatter` into dense `[1,1,S,128]`.
8. Broadcast multiply by `per_expert_scale`.

Experts:

1. Split source `gate_up_proj` into contiguous gate/up halves.
2. Transpose to sparse-matmul layouts.
3. Decode path uses `ttnn.sparse_matmul` for gate, up, and down with dense routing weights.
4. Prefill path may chunk sequence to keep sparse matmul core allocation bounded.

## Mesh And Memory

Correctness-first path uses BF16 logical weights. The existing code can store MLP and expert weights in lower precision through the `dtype` argument, but BF16 should remain available until sample quality and logits are source-verified.

For batch=1 decode, active-expert execution is required. Dense routing tensors are acceptable as a decomposed reference, but a compact active-expert metadata contract is the likely custom-op target if dense scatter/sparse_matmul overhead dominates.
