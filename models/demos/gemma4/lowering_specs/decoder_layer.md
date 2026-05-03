# Lowering Spec: Decoder Layer

Source: `Gemma4TextDecoderLayer.forward`, Transformers `v5.5.0`, `modeling_gemma4.py` lines 1319-1403.

## Inputs And Outputs

Input and output tensor shape is `[1, 1, S, 2816]` in TTNN demo layout.

For decode, `S=1`. For prefill, `S` is padded to a tile-aligned length.

## Pseudocode

```python
residual = x
x = input_layernorm(x)
x = self_attn(x)
x = post_attention_layernorm(x)
x = residual + x

residual = x
x = pre_feedforward_layernorm(x)
x = shared_mlp(x)

if enable_moe_block:
    x_1 = post_feedforward_layernorm_1(x)

    router_input = residual.reshape(-1, hidden_size)
    _, top_k_weights, top_k_index = router(router_input)

    x_2 = pre_feedforward_layernorm_2(router_input)
    x_2 = experts(x_2, top_k_index, top_k_weights)
    x_2 = x_2.reshape(residual.shape)
    x_2 = post_feedforward_layernorm_2(x_2)

    x = x_1 + x_2

x = post_feedforward_layernorm(x)
x = residual + x
x = x * layer_scalar
```

## TTNN Decomposition

1. Use local `RMSNorm` wrappers for each norm.
2. Call attention lowering with layer-type-specific RoPE and cache.
3. Shared MLP: `linear(gate) -> gelu_pytorch_tanh -> linear(up) -> mul -> linear(down) -> TP allreduce`.
4. MoE: route from the pre-FF residual, not from the pre-FF-normalized tensor.
5. Expert input is separately pre-normalized by `pre_feedforward_layernorm_2`.
6. Apply `post_feedforward_layernorm_1` to dense MLP output and `post_feedforward_layernorm_2` to expert output before summing.
7. Apply `post_feedforward_layernorm`, residual add, and scalar.

## Tests

Reference tests must checkpoint these boundaries: input norm output, attention output, post-attention residual, shared MLP output, router top-k indices/weights, expert output, post-FF residual, and final layer scalar output.
