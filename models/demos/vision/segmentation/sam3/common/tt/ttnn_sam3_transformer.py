# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SAM3 Transformer Encoder and Decoder implementation on ttnn.

Hybrid approach:
- ttnn for linear projections (Q, K, V, output, FFN)
- torch fallback for attention computation (reshape, permute, scaled_dot_product_attention)
- torch fallback for layer_norm (3D support)
"""

import torch
import torch.nn.functional as F
import ttnn

from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_common import (
    preprocess_linear_bias,
    preprocess_linear_weight,
)


# ---------------------------------------------------------------------------
# Multi-head attention (hybrid: ttnn linear projections, torch attention)
# ---------------------------------------------------------------------------


def tt_multihead_attention(query, key, value, qw, qb, kw, kb, vw, vb, ow, ob, num_heads, device):
    """Multi-head attention with ttnn linear projections and torch attention.

    All inputs/outputs are ttnn tensors on device.
    Internally converts to torch for the attention computation.

    Supports both 2D (seq, d_model) and 3D (batch, seq, d_model) or (seq, batch, d_model) inputs.
    The batch dimension handling follows the input shape.

    Args:
        query: ttnn tensor - query input.
        key: ttnn tensor - key input.
        value: ttnn tensor - value input.
        qw, qb: Preprocessed Q projection weight/bias (ttnn, on device).
        kw, kb: Preprocessed K projection weight/bias (ttnn, on device).
        vw, vb: Preprocessed V projection weight/bias (ttnn, on device).
        ow, ob: Preprocessed output projection weight/bias (ttnn, on device).
        num_heads: Number of attention heads.
        device: ttnn device.

    Returns:
        ttnn tensor with same leading dims as query, last dim = d_model.
    """
    # Q, K, V linear projections on device
    q = ttnn.linear(query, qw, bias=qb)
    k = ttnn.linear(key, kw, bias=kb)
    v = ttnn.linear(value, vw, bias=vb)

    # Move to torch for attention computation
    q_torch = ttnn.to_torch(q).float()
    k_torch = ttnn.to_torch(k).float()
    v_torch = ttnn.to_torch(v).float()

    orig_shape = q_torch.shape
    d_model = orig_shape[-1]
    head_dim = d_model // num_heads

    if q_torch.dim() == 3:
        # Detect if batch-first or seq-first by checking if Q and K have same dim0
        # For cross-attention with different seq lengths, dim0 must be seq (seq-first format)
        q_dim0, q_dim1 = orig_shape[0], orig_shape[1]
        k_dim0 = k_torch.shape[0]

        if q_dim0 != k_dim0:
            # Different dim0 -> seq-first: (seq, batch, d_model)
            # batch is dim1, seq varies between Q and K
            batch_size = q_dim1
            q_seq = q_dim0
            k_seq = k_dim0
            # Transpose to batch-first: (batch, seq, d_model)
            q_torch = q_torch.transpose(0, 1)
            k_torch = k_torch.transpose(0, 1)
            v_torch = v_torch.transpose(0, 1)
            seq_first = True
        else:
            # Same dim0 -> treat as batch-first: (batch, seq, d_model)
            batch_size = q_dim0
            q_seq = q_dim1
            seq_first = False

        # Now all are batch-first: (batch, seq, d_model) -> (batch, num_heads, seq, head_dim)
        q_torch = q_torch.reshape(batch_size, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        k_torch = k_torch.reshape(batch_size, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        v_torch = v_torch.reshape(batch_size, -1, num_heads, head_dim).permute(0, 2, 1, 3)
    elif q_torch.dim() == 2:
        q_torch = q_torch.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        k_torch = k_torch.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        v_torch = v_torch.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        seq_first = False
    else:
        raise ValueError(f"Unsupported query dim: {q_torch.dim()}")

    # Scaled dot-product attention: (batch, heads, q_seq, head_dim) @ (batch, heads, kv_seq, head_dim)
    attn_output = F.scaled_dot_product_attention(q_torch, k_torch, v_torch)

    # Reshape back: (batch, heads, q_seq, head_dim) -> (batch, q_seq, d_model)
    attn_output = attn_output.permute(0, 2, 1, 3).reshape(attn_output.shape[0], -1, d_model)

    if len(orig_shape) == 3 and seq_first:
        # Convert back to seq-first: (batch, seq, d_model) -> (seq, batch, d_model)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(orig_shape[0], orig_shape[1], d_model)
    elif len(orig_shape) == 3:
        attn_output = attn_output.reshape(orig_shape[0], orig_shape[1], d_model)
    else:
        attn_output = attn_output.squeeze(0)

    # Convert back to ttnn for output projection
    attn_tt = ttnn.from_torch(attn_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Output projection on device
    output = ttnn.linear(attn_tt, ow, bias=ob)
    return output


# ---------------------------------------------------------------------------
# Encoder layer (pre-norm, batch-first)
# ---------------------------------------------------------------------------


def tt_encoder_layer(src, pos, text_features, layer_params, device):
    """Single transformer encoder layer with pre-norm.

    Matches TransformerEncoderLayer.forward_pre from encoder.py.
    SAM3 config: pos_enc_at_attn=True, pos_enc_at_cross_attn_keys=False,
    pos_enc_at_cross_attn_queries=False, batch_first=True.

    Args:
        src: ttnn tensor (batch, seq_len, d_model) - input features (batch-first).
        pos: ttnn tensor (batch, seq_len, d_model) - position embeddings for src.
        text_features: ttnn tensor (batch, text_len, d_model) or None - text for cross-attention.
        layer_params: dict with preprocessed weights for this layer.
        device: ttnn device.

    Returns:
        ttnn tensor (batch, seq_len, d_model).
    """
    p = layer_params

    # --- Self-attention (pre-norm) ---
    # norm1(src)
    src_torch = ttnn.to_torch(src).float()
    pos_torch = ttnn.to_torch(pos).float()
    d = src_torch.shape[-1]
    tgt2 = F.layer_norm(src_torch, [d], weight=p["norm1_weight"], bias=p["norm1_bias"])

    # q = k = norm1(src) + pos  (pos_enc_at_attn=True)
    q_input = tgt2 + pos_torch
    q_tt = ttnn.from_torch(q_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tgt2_tt = ttnn.from_torch(tgt2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    sa_out = tt_multihead_attention(
        q_tt, q_tt, tgt2_tt,
        p["sa_qw"], p["sa_qb"], p["sa_kw"], p["sa_kb"],
        p["sa_vw"], p["sa_vb"], p["sa_ow"], p["sa_ob"],
        p["num_heads"], device,
    )

    # Residual
    src = ttnn.add(src, sa_out)

    # --- Cross-attention to text (if provided) ---
    if text_features is not None:
        # norm2(src)
        src_torch = ttnn.to_torch(src).float()
        tgt2 = F.layer_norm(src_torch, [d], weight=p["norm2_weight"], bias=p["norm2_bias"])
        # pos_enc_at_cross_attn_queries=False, pos_enc_at_cross_attn_keys=False
        # query = norm2(src), key = value = text_features
        tgt2_tt = ttnn.from_torch(tgt2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ca_out = tt_multihead_attention(
            tgt2_tt, text_features, text_features,
            p["ca_qw"], p["ca_qb"], p["ca_kw"], p["ca_kb"],
            p["ca_vw"], p["ca_vb"], p["ca_ow"], p["ca_ob"],
            p["num_heads"], device,
        )

        # Residual
        src = ttnn.add(src, ca_out)

    # --- FFN (pre-norm) ---
    src_torch = ttnn.to_torch(src).float()
    tgt2 = F.layer_norm(src_torch, [d], weight=p["norm3_weight"], bias=p["norm3_bias"])
    tgt2_tt = ttnn.from_torch(tgt2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # FFN: Linear -> ReLU -> Linear
    ffn_hidden = ttnn.linear(tgt2_tt, p["ffn_w1"], bias=p["ffn_b1"])
    ffn_hidden = ttnn.relu(ffn_hidden)
    ffn_out = ttnn.linear(ffn_hidden, p["ffn_w2"], bias=p["ffn_b2"])

    # Residual
    src = ttnn.add(src, ffn_out)
    return src


def tt_transformer_encoder(features, pos_embeds, text_features, encoder_params, device):
    """Run the full transformer encoder (6 layers).

    Args:
        features: ttnn tensor (batch, seq_len, d_model) - flattened image features.
        pos_embeds: ttnn tensor (batch, seq_len, d_model) - position embeddings.
        text_features: ttnn tensor (batch, text_len, d_model) or None - text features.
        encoder_params: dict with 'layers' list of per-layer params.
        device: ttnn device.

    Returns:
        ttnn tensor (batch, seq_len, d_model).
    """
    output = features
    for layer_params in encoder_params["layers"]:
        output = tt_encoder_layer(output, pos_embeds, text_features, layer_params, device)
    return output


# ---------------------------------------------------------------------------
# Decoder layer (post-norm, seq-first)
# ---------------------------------------------------------------------------


def tt_decoder_layer(queries, memory, query_pos, memory_pos, layer_params, device):
    """Single transformer decoder layer.

    Matches TransformerDecoderLayer.forward from decoder.py (simplified, no DAC/box-refinement).
    Order: self-attention -> norm2 -> cross-attention -> norm1 -> FFN -> norm3.

    Args:
        queries: ttnn tensor (num_queries, batch, d_model) - decoder queries (seq-first).
        memory: ttnn tensor (hw, batch, d_model) - encoder memory (seq-first).
        query_pos: ttnn tensor (num_queries, batch, d_model) - position for queries.
        memory_pos: ttnn tensor (hw, batch, d_model) - position for memory.
        layer_params: dict with preprocessed weights.
        device: ttnn device.

    Returns:
        ttnn tensor (num_queries, batch, d_model).
    """
    p = layer_params

    # --- Self-attention ---
    # q = k = queries + query_pos (with_pos_embed)
    q_torch = ttnn.to_torch(queries).float()
    qpos_torch = ttnn.to_torch(query_pos).float()
    qk_input = q_torch + qpos_torch
    qk_tt = ttnn.from_torch(qk_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    sa_out = tt_multihead_attention(
        qk_tt, qk_tt, queries,
        p["sa_qw"], p["sa_qb"], p["sa_kw"], p["sa_kb"],
        p["sa_vw"], p["sa_vb"], p["sa_ow"], p["sa_ob"],
        p["num_heads"], device,
    )

    # Residual + norm2 (post-norm)
    tgt = ttnn.add(queries, sa_out)
    tgt_torch = ttnn.to_torch(tgt).float()
    d = tgt_torch.shape[-1]
    tgt_normed = F.layer_norm(tgt_torch, [d], weight=p["norm2_weight"], bias=p["norm2_bias"])
    tgt = ttnn.from_torch(tgt_normed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # --- Cross-attention to memory ---
    # query = tgt + query_pos, key = memory + memory_pos, value = memory
    tgt_torch = ttnn.to_torch(tgt).float()
    q_input = tgt_torch + qpos_torch
    q_tt = ttnn.from_torch(q_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    mem_torch = ttnn.to_torch(memory).float()
    mpos_torch = ttnn.to_torch(memory_pos).float()
    k_input = mem_torch + mpos_torch
    k_tt = ttnn.from_torch(k_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ca_out = tt_multihead_attention(
        q_tt, k_tt, memory,
        p["ca_qw"], p["ca_qb"], p["ca_kw"], p["ca_kb"],
        p["ca_vw"], p["ca_vb"], p["ca_ow"], p["ca_ob"],
        p["num_heads"], device,
    )

    # Residual + norm1 (post-norm)
    tgt = ttnn.add(tgt, ca_out)
    tgt_torch = ttnn.to_torch(tgt).float()
    tgt_normed = F.layer_norm(tgt_torch, [d], weight=p["norm1_weight"], bias=p["norm1_bias"])
    tgt = ttnn.from_torch(tgt_normed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # --- FFN + norm3 (post-norm) ---
    ffn_hidden = ttnn.linear(tgt, p["ffn_w1"], bias=p["ffn_b1"])
    ffn_hidden = ttnn.relu(ffn_hidden)
    ffn_out = ttnn.linear(ffn_hidden, p["ffn_w2"], bias=p["ffn_b2"])

    tgt = ttnn.add(tgt, ffn_out)
    tgt_torch = ttnn.to_torch(tgt).float()
    tgt_normed = F.layer_norm(tgt_torch, [d], weight=p["norm3_weight"], bias=p["norm3_bias"])
    tgt = ttnn.from_torch(tgt_normed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return tgt


def tt_transformer_decoder(queries, memory, memory_pos, decoder_params, device):
    """Run the full transformer decoder (6 layers).

    Simplified version: runs decoder layers without box refinement or DAC.
    Uses zeros as query_pos (in the full model, query_pos derives from reference boxes).

    Args:
        queries: ttnn tensor (num_queries, batch, d_model) - initial object queries (seq-first).
        memory: ttnn tensor (hw, batch, d_model) - encoder output (seq-first).
        memory_pos: ttnn tensor (hw, batch, d_model) - position embeddings for memory.
        decoder_params: dict with 'layers' list of per-layer params.
        device: ttnn device.

    Returns:
        ttnn tensor (num_queries, batch, d_model) - decoder output after all layers.
    """
    output = queries

    # Zero query_pos as placeholder (full model uses ref_point_head(sine_embed(boxes)))
    q_torch = ttnn.to_torch(queries).float()
    query_pos_torch = torch.zeros_like(q_torch)
    query_pos = ttnn.from_torch(query_pos_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    for layer_params in decoder_params["layers"]:
        output = tt_decoder_layer(output, memory, query_pos, memory_pos, layer_params, device)

    return output


# ---------------------------------------------------------------------------
# Weight preprocessing
# ---------------------------------------------------------------------------


def _preprocess_mha_weights(mha_module, device):
    """Preprocess nn.MultiheadAttention weights for tt_multihead_attention.

    nn.MultiheadAttention stores Q/K/V as a single in_proj_weight (3*d, d)
    or separate q/k/v proj weights depending on configuration.

    Args:
        mha_module: nn.MultiheadAttention module.
        device: ttnn device.

    Returns:
        dict with qw, qb, kw, kb, vw, vb, ow, ob as ttnn tensors on device.
    """
    d_model = mha_module.embed_dim

    if mha_module.in_proj_weight is not None:
        w = mha_module.in_proj_weight.data
        q_w, k_w, v_w = w[:d_model], w[d_model:2*d_model], w[2*d_model:]
    else:
        q_w = mha_module.q_proj_weight.data
        k_w = mha_module.k_proj_weight.data
        v_w = mha_module.v_proj_weight.data

    if mha_module.in_proj_bias is not None:
        b = mha_module.in_proj_bias.data
        q_b, k_b, v_b = b[:d_model], b[d_model:2*d_model], b[2*d_model:]
    else:
        q_b = torch.zeros(d_model)
        k_b = torch.zeros(d_model)
        v_b = torch.zeros(d_model)

    o_w = mha_module.out_proj.weight.data
    o_b = mha_module.out_proj.bias.data if mha_module.out_proj.bias is not None else torch.zeros(d_model)

    return {
        "qw": ttnn.to_device(preprocess_linear_weight(q_w), device),
        "qb": ttnn.to_device(preprocess_linear_bias(q_b), device),
        "kw": ttnn.to_device(preprocess_linear_weight(k_w), device),
        "kb": ttnn.to_device(preprocess_linear_bias(k_b), device),
        "vw": ttnn.to_device(preprocess_linear_weight(v_w), device),
        "vb": ttnn.to_device(preprocess_linear_bias(v_b), device),
        "ow": ttnn.to_device(preprocess_linear_weight(o_w), device),
        "ob": ttnn.to_device(preprocess_linear_bias(o_b), device),
    }


def preprocess_encoder_layer_weights(encoder_layer, num_heads, device):
    """Preprocess weights for a single encoder layer.

    Args:
        encoder_layer: TransformerEncoderLayer module.
        num_heads: Number of attention heads.
        device: ttnn device.

    Returns:
        dict with all preprocessed weights for tt_encoder_layer.
    """
    params = {"num_heads": num_heads}

    # Self-attention weights
    sa = _preprocess_mha_weights(encoder_layer.self_attn, device)
    for k, v in sa.items():
        params[f"sa_{k}"] = v

    # Cross-attention weights
    ca = _preprocess_mha_weights(encoder_layer.cross_attn_image, device)
    for k, v in ca.items():
        params[f"ca_{k}"] = v

    # Layer norms (kept as torch tensors for torch fallback)
    params["norm1_weight"] = encoder_layer.norm1.weight.data.clone()
    params["norm1_bias"] = encoder_layer.norm1.bias.data.clone()
    params["norm2_weight"] = encoder_layer.norm2.weight.data.clone()
    params["norm2_bias"] = encoder_layer.norm2.bias.data.clone()
    params["norm3_weight"] = encoder_layer.norm3.weight.data.clone()
    params["norm3_bias"] = encoder_layer.norm3.bias.data.clone()

    # FFN weights
    params["ffn_w1"] = ttnn.to_device(preprocess_linear_weight(encoder_layer.linear1.weight.data), device)
    params["ffn_b1"] = ttnn.to_device(preprocess_linear_bias(encoder_layer.linear1.bias.data), device)
    params["ffn_w2"] = ttnn.to_device(preprocess_linear_weight(encoder_layer.linear2.weight.data), device)
    params["ffn_b2"] = ttnn.to_device(preprocess_linear_bias(encoder_layer.linear2.bias.data), device)

    return params


def preprocess_encoder_weights(encoder, num_heads, device):
    """Preprocess weights for the full transformer encoder.

    Args:
        encoder: TransformerEncoderFusion or TransformerEncoder module.
        num_heads: Number of attention heads.
        device: ttnn device.

    Returns:
        dict with 'layers' list of per-layer params.
    """
    params = {"layers": []}
    for layer in encoder.layers:
        params["layers"].append(preprocess_encoder_layer_weights(layer, num_heads, device))
    return params


def preprocess_decoder_layer_weights(decoder_layer, num_heads, device):
    """Preprocess weights for a single decoder layer.

    Args:
        decoder_layer: TransformerDecoderLayer module.
        num_heads: Number of attention heads.
        device: ttnn device.

    Returns:
        dict with all preprocessed weights for tt_decoder_layer.
    """
    params = {"num_heads": num_heads}

    # Self-attention weights
    sa = _preprocess_mha_weights(decoder_layer.self_attn, device)
    for k, v in sa.items():
        params[f"sa_{k}"] = v

    # Cross-attention weights
    ca = _preprocess_mha_weights(decoder_layer.cross_attn, device)
    for k, v in ca.items():
        params[f"ca_{k}"] = v

    # Layer norms (kept as torch tensors for torch fallback)
    params["norm1_weight"] = decoder_layer.norm1.weight.data.clone()
    params["norm1_bias"] = decoder_layer.norm1.bias.data.clone()
    params["norm2_weight"] = decoder_layer.norm2.weight.data.clone()
    params["norm2_bias"] = decoder_layer.norm2.bias.data.clone()
    params["norm3_weight"] = decoder_layer.norm3.weight.data.clone()
    params["norm3_bias"] = decoder_layer.norm3.bias.data.clone()

    # FFN weights
    params["ffn_w1"] = ttnn.to_device(preprocess_linear_weight(decoder_layer.linear1.weight.data), device)
    params["ffn_b1"] = ttnn.to_device(preprocess_linear_bias(decoder_layer.linear1.bias.data), device)
    params["ffn_w2"] = ttnn.to_device(preprocess_linear_weight(decoder_layer.linear2.weight.data), device)
    params["ffn_b2"] = ttnn.to_device(preprocess_linear_bias(decoder_layer.linear2.bias.data), device)

    return params


def preprocess_decoder_weights(decoder, num_heads, device):
    """Preprocess weights for the full transformer decoder.

    Args:
        decoder: TransformerDecoder module.
        num_heads: Number of attention heads.
        device: ttnn device.

    Returns:
        dict with 'layers' list of per-layer params.
    """
    params = {"layers": []}
    for layer in decoder.layers:
        params["layers"].append(preprocess_decoder_layer_weights(layer, num_heads, device))
    return params
