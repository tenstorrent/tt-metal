# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Torch -> TTNN weight preparation for ModernBERT.

Hand-rolled rather than preprocess_model_parameters, which dereferences
`model.bias` on every LayerNorm and so raises on this bias-free model.

  * ttnn.linear wants [in, out]; torch stores [out, in], so linear weights are
    transposed here at load time.
  * Layer 0 has no attn_norm (nn.Identity in HF), so its entry is None.
  * Matmul weights are bfloat8_b, embeddings and norms bfloat16.
"""

import ttnn
from models.experimental.modernbert.tt.model_config import LINEAR_WEIGHTS_DTYPE, WEIGHTS_DTYPE

EXPECTED_TENSOR_COUNT = 134
# encoder (134) + head.dense.weight + head.norm.weight + decoder.weight + decoder.bias
EXPECTED_MLM_TENSOR_COUNT = 138


def _linear(weight, device, dtype):
    """torch nn.Linear stores [out, in]; ttnn.linear wants [in, out]."""
    return ttnn.from_torch(
        weight.transpose(-1, -2).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def _qkv_linear(weight, device, dtype, head_dim):
    """Wqkv with the attention scaling folded into the Q third.

    sdpa.cpp rescales attn_mask by 1/scale on every call unless scale is exactly
    1.0. Folding 1/sqrt(head_dim) into Wq lets attention pass scale=1.0 and skip
    that rescale in every layer carrying a mask.

    Exact, not an approximation: head_dim 64 gives 1/8, a power of two. RoPE is
    linear in Q, so scaling before it is equivalent to scaling after. Q is the
    first contiguous third of rows (Wqkv reshapes to (B, S, 3, n_heads, head_dim)).
    """
    scaled = weight.clone()
    q_rows = scaled.shape[0] // 3
    scaled[:q_rows] *= head_dim**-0.5
    return _linear(scaled, device, dtype)


def _norm(weight, device, dtype):
    return ttnn.from_torch(weight, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def head_dim(torch_model):
    """head_dim, read from the model's own config.

    Both the reference ModernBertModel and HF's expose .config, and
    prepare_weights accepts either.
    """
    cfg = torch_model.config
    return cfg.hidden_size // cfg.num_attention_heads


def config_intermediate(layer):
    """intermediate_size, read from the layer's own Wi shape.

    Wi emits 2 * intermediate_size, so the split point is half its output.
    """
    return layer.mlp.Wi.weight.shape[0] // 2


def _embedding(weight, device, dtype):
    """Embedding tables are stored row-major.

    ttnn.embedding consumes a row-major table. A tiled one is converted on every
    call, which for this 50368x768 table showed up in the profile as a 752 us
    untilize immediately before the embedding op -- 5.5% of the forward pass.
    """
    return ttnn.from_torch(weight, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def prepare_weights(torch_model, device, dtype=WEIGHTS_DTYPE, linear_dtype=LINEAR_WEIGHTS_DTYPE):
    """Convert a torch ModernBertModel (ours or HF) into device tensors.

    Returns a nested dict mirroring the module tree:
        {"embeddings": {...}, "layers": [ {...} x22 ], "final_norm": t}

    `linear_dtype` covers the matmul weight matrices, `dtype` the embedding table and
    the norm weights. They differ because bfloat8_b is only representable for the
    former: it is a tiled block format, so it cannot hold the row-major embedding
    table, and a 1-D norm weight would share its exponent with 31 rows of tile
    padding. Both are exposed as arguments so the choice stays measurable -- see
    model_config for what the measurement showed.
    """
    sd_count = len(torch_model.state_dict())
    if sd_count != EXPECTED_TENSOR_COUNT:
        raise ValueError(f"expected {EXPECTED_TENSOR_COUNT} tensors, got {sd_count}")

    params = {
        "embeddings": {
            "tok_embeddings": _embedding(torch_model.embeddings.tok_embeddings.weight, device, dtype),
            "norm": _norm(torch_model.embeddings.norm.weight, device, dtype),
        },
        "layers": [],
        "final_norm": _norm(torch_model.final_norm.weight, device, dtype),
    }

    qkv_head_dim = head_dim(torch_model)
    for idx, layer in enumerate(torch_model.layers):
        entry = {
            # None for layer 0: HF uses nn.Identity there because the embedding
            # LayerNorm has already normalised the input.
            "attn_norm": None if idx == 0 else _norm(layer.attn_norm.weight, device, dtype),
            "attn": {
                # Kept fused. Wqkv output reshapes to (B, S, 3, n_heads, head_dim),
                # i.e. the 3 is the OUTER dim, so Q/K/V split contiguously.
                "Wqkv": _qkv_linear(layer.attn.Wqkv.weight, device, linear_dtype, qkv_head_dim),
                "Wo": _linear(layer.attn.Wo.weight, device, linear_dtype),
            },
            "mlp_norm": _norm(layer.mlp_norm.weight, device, dtype),
            "mlp": {
                # Wi produces 2 * intermediate_size, which HF chunks into
                # (input, gate). Splitting it here into two weights lets the MLP
                # run two matmuls instead of one wide matmul plus two runtime
                # slices, measured faster at every validated shape:
                #   seq 256   189.7 us -> 151.1 us
                #   seq 512   218.0 us -> 203.8 us
                # torch stores [out, in]; after transposing to [in, out] the
                # activated half is the first intermediate_size columns.
                "Wi_act": _linear(layer.mlp.Wi.weight[: config_intermediate(layer), :], device, linear_dtype),
                "Wi_gate": _linear(layer.mlp.Wi.weight[config_intermediate(layer) :, :], device, linear_dtype),
                "Wo": _linear(layer.mlp.Wo.weight, device, linear_dtype),
            },
        }
        params["layers"].append(entry)

    return params


def deallocate_weights(params):
    """Release every device tensor held by prepare_weights()."""

    def _walk(node):
        if isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)
        elif isinstance(node, ttnn.Tensor):
            ttnn.deallocate(node)

    _walk(params)


def prepare_mlm_weights(torch_mlm, device, dtype=WEIGHTS_DTYPE, linear_dtype=LINEAR_WEIGHTS_DTYPE):
    """Convert a torch ModernBertForMaskedLM into device tensors.

    Returns the encoder params under "model", plus the MLM head:
        {"model": {...}, "head": {...}, "decoder": {"weight": t, "bias": t}}

    The decoder is the only biased layer in the model (decoder_bias=True) and its
    weight is tied to embeddings.tok_embeddings.weight (tie_word_embeddings=True),
    so it is uploaded a second time here in transposed [in, out] form for use as a
    linear rather than shared with the embedding lookup.
    """
    sd_count = len(torch_mlm.state_dict())
    if sd_count != EXPECTED_MLM_TENSOR_COUNT:
        raise ValueError(f"expected {EXPECTED_MLM_TENSOR_COUNT} tensors, got {sd_count}")

    return {
        "model": prepare_weights(torch_mlm.model, device, dtype, linear_dtype),
        "head": {
            "dense": _linear(torch_mlm.head.dense.weight, device, linear_dtype),
            "norm": _norm(torch_mlm.head.norm.weight, device, dtype),
        },
        "decoder": {
            "weight": _linear(torch_mlm.decoder.weight, device, linear_dtype),
            "bias": _norm(torch_mlm.decoder.bias, device, dtype),
        },
    }
