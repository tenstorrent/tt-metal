# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Optional

import ttnn
import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight, make_parameter_dict

from models.experimental.seamless_m4t_v2_large.reference.torch_text_decoder import embed_scale_for_config
from models.experimental.seamless_m4t_v2_large.tt.common import (
    TILE,
    create_dram_sharded_mem_config,
    dram_shard_core_count,
)


def _conv1d_weight(conv: torch.nn.Conv1d, *, device: ttnn.Device) -> ttnn.Tensor:
    """Host ROW_MAJOR PyTorch-shaped weights (``[out, in/groups, K]``).

    ``ttnn.conv1d`` / conv2d expect either host tensors (prepared + uploaded per call) or
    device tensors that already pass ``is_valid_device_conv_weights`` (TILE, padded layout).
    Uploading raw ROW_MAJOR weights to device triggers a host round-trip and warnings in
    ``conv2d.cpp``; keeping weights on host avoids that.
    """
    _ = device  # kept for call-site symmetry with other preprocess helpers
    w = conv.weight.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _conv1d_bias(conv: torch.nn.Conv1d, *, device: ttnn.Device) -> Optional[ttnn.Tensor]:
    if conv.bias is None:
        return None
    _ = device
    b = conv.bias.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        b.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _conv_transpose1d_weight(conv: torch.nn.ConvTranspose1d, *, device: ttnn.Device) -> ttnn.Tensor:
    """PyTorch ``[in_c, out_c, K]`` -> TTNN conv_transpose2d-style ``[in_c, out_c, K, 1]``."""
    w = conv.weight.detach().to(torch.bfloat16).contiguous()
    w2 = w.unsqueeze(-1).contiguous()
    return ttnn.from_torch(
        w2,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _conv_transpose1d_bias_host(conv: torch.nn.ConvTranspose1d) -> Optional[ttnn.Tensor]:
    if conv.bias is None:
        return None
    b = conv.bias.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        b.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _embedding_weight(emb: torch.nn.Embedding, *, device: ttnn.Device) -> ttnn.Tensor:
    w = emb.weight.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _vocoder_embedding_weight_row_major(emb: torch.nn.Embedding, *, device: ttnn.Device) -> ttnn.Tensor:
    """ROW_MAJOR embedding table for [`TTSeamlessM4Tv2CodeHifiGan`] (matches text decoder / T2U decoder).

    ``ttnn.embedding`` still returns TILE_LAYOUT activations when ``layout=ttnn.TILE_LAYOUT`` is passed;
    TILE-stored tables can add a trailing untilize on lookup. Call sites unchanged in ``tt_code_hifigan``.
    """
    w = emb.weight.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _ln_to_device(param: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    x = param.detach().reshape(1, 1, -1).contiguous()
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _linear_pair(
    linear: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    bias_dtype: Optional[ttnn.DataType] = None,
) -> dict:
    b_dtype = bias_dtype if bias_dtype is not None else weight_dtype
    w = preprocess_linear_weight(linear.weight.detach(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    b = preprocess_linear_bias(linear.bias.detach(), dtype=b_dtype, layout=ttnn.TILE_LAYOUT)
    return {
        "weight": ttnn.to_device(w, device),
        "bias": ttnn.to_device(b, device),
    }


def _linear_bias_dram_sharded(
    bias: torch.Tensor,
    *,
    device: ttnn.Device,
    m: int,
    n: int,
) -> ttnn.Tensor:
    """DRAM width-sharded bias tile for ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``."""
    dram_cores = dram_shard_core_count(device, n)
    _, padded_n = create_dram_sharded_mem_config(device, TILE, n)
    bias_4d = bias.detach().reshape(1, 1, 1, int(bias.numel())).to(torch.bfloat16)
    bias_4d = torch.nn.functional.pad(bias_4d, (0, padded_n - int(bias.numel())))
    bias_4d = bias_4d.expand(1, 1, m, padded_n).contiguous()
    dram_grid = device.dram_grid_size()
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, dram_grid.y - 1))}
    )
    bias_shard_shape = [TILE, padded_n // dram_cores]
    bias_shard_spec = ttnn.ShardSpec(shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    bias_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec)
    return ttnn.from_torch(
        bias_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bias_mem_config,
    )


def _linear_pair_dram_sharded(
    linear: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    m: int = 32,
) -> dict:
    w_torch = linear.weight.detach().T.contiguous()
    k, n = int(w_torch.shape[0]), int(w_torch.shape[1])
    mem_config, padded_n = create_dram_sharded_mem_config(device, k, n)
    if padded_n > n:
        w_torch = torch.nn.functional.pad(w_torch, (0, padded_n - n))
    weight = ttnn.from_torch(
        w_torch,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    bias = _linear_bias_dram_sharded(linear.bias.detach(), device=device, m=m, n=padded_n)
    return {"weight": weight, "bias": bias}


def _fused_linear_weight_dram_sharded(
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType,
    m: int = 32,
) -> dict:
    w_torch = weight.T.contiguous()
    k, n = int(w_torch.shape[0]), int(w_torch.shape[1])
    mem_config, padded_n = create_dram_sharded_mem_config(device, k, n)
    if padded_n > n:
        w_torch = torch.nn.functional.pad(w_torch, (0, padded_n - n))
    w = ttnn.from_torch(
        w_torch,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    b = _linear_bias_dram_sharded(bias, device=device, m=m, n=padded_n)
    return {"weight": w, "bias": b}


def _fused_qkv_pair(
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    q_scale: float = 1.0,
) -> dict:
    """Concatenate Q/K/V projection weights into a single fused QKV linear.

    Produced tensor pairs feed ``ttnn.linear`` followed by
    ``ttnn.experimental.nlp_create_qkv_heads`` (fused QKV head path).
    Concatenation is along the output dimension so the fused matmul
    output is laid out as ``[..., 3 * hidden]`` (Q | K | V).

    Stage 8: when ``q_scale != 1.0``, the Q rows of the concatenated weight
    and bias are pre-multiplied by ``q_scale`` (typically ``1 / sqrt(head_dim)``).
    This folds the attention scale factor into the weights at preprocessing time,
    eliminating a runtime ``multiply`` per conformer self-attention layer.
    """
    q_w = q_proj.weight.detach()
    q_b = q_proj.bias.detach()
    if q_scale != 1.0:
        q_w = (q_w * q_scale).to(q_w.dtype)
        q_b = (q_b * q_scale).to(q_b.dtype)
    qkv_weight = torch.cat([q_w, k_proj.weight.detach(), v_proj.weight.detach()], dim=0).contiguous()
    qkv_bias = torch.cat([q_b, k_proj.bias.detach(), v_proj.bias.detach()], dim=0).contiguous()
    w = preprocess_linear_weight(qkv_weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    b = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return {
        "weight": ttnn.to_device(w, device),
        "bias": ttnn.to_device(b, device),
    }


def _fused_qkv_pair_dram_sharded(
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    q_scale: float = 1.0,
    m: int = 32,
) -> dict:
    q_w = q_proj.weight.detach()
    q_b = q_proj.bias.detach()
    if q_scale != 1.0:
        q_w = (q_w * q_scale).to(q_w.dtype)
        q_b = (q_b * q_scale).to(q_b.dtype)
    qkv_weight = torch.cat([q_w, k_proj.weight.detach(), v_proj.weight.detach()], dim=0).contiguous()
    qkv_bias = torch.cat([q_b, k_proj.bias.detach(), v_proj.bias.detach()], dim=0).contiguous()
    return _fused_linear_weight_dram_sharded(qkv_weight, qkv_bias, device=device, weight_dtype=weight_dtype, m=m)


def _fused_kv_pair(
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    """Concatenate K/V projection weights for one matmul over shared activations (cross-attn).

    Output layout is ``[..., 2 * hidden]`` (K | V) on the last dim; the decoder splits before
    ``_heads``.
    """
    kv_weight = torch.cat([k_proj.weight.detach(), v_proj.weight.detach()], dim=0).contiguous()
    kv_bias = torch.cat([k_proj.bias.detach(), v_proj.bias.detach()], dim=0).contiguous()
    w = preprocess_linear_weight(kv_weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    b = preprocess_linear_bias(kv_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return {
        "weight": ttnn.to_device(w, device),
        "bias": ttnn.to_device(b, device),
    }


def create_text_decoder_parameters(decoder, *, device: ttnn.Device) -> dict:
    """
    Convert [`SeamlessM4Tv2Decoder`] weights to TTNN tensors on ``device``.

    Token embeddings include ``embed_scale`` (see [`SeamlessM4Tv2ScaledWordEmbedding`]).
    """
    cfg = decoder.config
    scale = embed_scale_for_config(cfg)

    # ROW_MAJOR embedding tables (matches text encoder / T2U decoder).
    # ``ttnn.embedding`` emits TILE_LAYOUT activations regardless; TILE-stored weights
    # can force a trailing ``UntilizeWithUnpaddingDeviceOperation`` per table lookup.
    scaled_emb = (decoder.embed_tokens.weight.detach() * scale).contiguous()
    embed_tokens_weight = ttnn.from_torch(
        scaled_emb,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    pos_w = decoder.embed_positions.weights.detach()
    if pos_w.dtype != torch.bfloat16:
        pos_w = pos_w.to(dtype=torch.bfloat16)
    embed_positions_weight = ttnn.from_torch(
        pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    layers = []
    for layer in decoder.layers:
        layer_dict = {
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            # Fused self-attn Q|K|V; cross-attn K|V fused (see ``cross_attention``).
            # Attention linear weights in bfloat8_b (bandwidth; biases stay bf16) — encoder pattern.
            "self_attn": {
                # Prefill: interleaved L1 matmul + ``MatmulMultiCoreReuseMultiCast1DProgramConfig`` (text encoder).
                # Decode: separate ``qkv_decode`` weights for KV-cache PCC.
                "qkv": _fused_qkv_pair(
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    device=device,
                    weight_dtype=ttnn.bfloat8_b,
                ),
                "qkv_decode": _fused_qkv_pair(
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    device=device,
                    weight_dtype=ttnn.bfloat8_b,
                ),
                "out_proj": _linear_pair(layer.self_attn.out_proj, device=device, weight_dtype=ttnn.bfloat8_b),
            },
            "cross_attention_layer_norm": {
                "weight": _ln_to_device(layer.cross_attention_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.cross_attention_layer_norm.bias, device=device),
            },
            # Fused K|V over encoder hidden states (one matmul vs two; Q stays separate).
            "cross_attention": {
                "q_proj": _linear_pair(layer.cross_attention.q_proj, device=device, weight_dtype=ttnn.bfloat8_b),
                "kv": _fused_kv_pair(
                    layer.cross_attention.k_proj,
                    layer.cross_attention.v_proj,
                    device=device,
                    weight_dtype=ttnn.bfloat8_b,
                ),
                "out_proj": _linear_pair(layer.cross_attention.out_proj, device=device, weight_dtype=ttnn.bfloat8_b),
            },
            "ffn_layer_norm": {
                "weight": _ln_to_device(layer.ffn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn_layer_norm.bias, device=device),
            },
            "ffn": {
                "fc1": _linear_pair(layer.ffn.fc1, device=device, weight_dtype=ttnn.bfloat8_b),
                "fc2": _linear_pair(layer.ffn.fc2, device=device, weight_dtype=ttnn.bfloat8_b),
            },
        }
        layers.append(make_parameter_dict(layer_dict))

    out = {
        "embed_tokens": make_parameter_dict({"weight": embed_tokens_weight}),
        "embed_positions": make_parameter_dict({"weight": embed_positions_weight}),
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(decoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(decoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def _m4t_encoder_self_attn_ffn_layers(
    encoder,
    *,
    device: ttnn.Device,
    ffn_weight_dtype: ttnn.DataType = ttnn.bfloat16,
    attn_weight_dtype: ttnn.DataType = ttnn.bfloat16,
    fuse_qkv: bool = False,
    dram_width_sharded_weights: bool = False,
    prefill_token_rows: int = 32,
) -> list:
    """Layer parameter dicts shared by text encoder and text-to-unit encoder stacks.

    ``ffn_weight_dtype`` controls the storage dtype of ``fc1.weight`` /
    ``fc2.weight``. ``attn_weight_dtype`` controls the storage dtype of
    ``q_proj``/``k_proj``/``v_proj``/``out_proj`` weights. Pass
    ``ttnn.bfloat8_b`` for memory-bound matmuls (bandwidth optimization);
    biases and LayerNorm parameters always stay at ``bfloat16``.

    When ``fuse_qkv=True`` the per-layer ``self_attn`` dict exposes a single
    ``qkv`` entry with concatenated Q|K|V weights/biases (consumed by
    ``ttnn.experimental.nlp_create_qkv_heads`` in the encoder forward) instead
    of separate ``q_proj``/``k_proj``/``v_proj`` entries. ``out_proj`` is
    always exposed as a separate linear pair.

    ``dram_width_sharded_weights=True`` stores linear weights as DRAM width-sharded
    BFP8 tiles with DRAM-sharded bias for ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``.
    ``prefill_token_rows`` is the M dimension baked into DRAM-sharded bias tiles.
    """
    linear_pair = _linear_pair_dram_sharded if dram_width_sharded_weights else _linear_pair
    fused_qkv_pair = _fused_qkv_pair_dram_sharded if dram_width_sharded_weights else _fused_qkv_pair
    linear_kwargs = {"m": prefill_token_rows} if dram_width_sharded_weights else {}

    layers = []
    for layer in encoder.layers:
        if fuse_qkv:
            self_attn = {
                "qkv": fused_qkv_pair(
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    device=device,
                    weight_dtype=attn_weight_dtype,
                    **linear_kwargs,
                ),
                "out_proj": linear_pair(
                    layer.self_attn.out_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
            }
        else:
            self_attn = {
                "q_proj": linear_pair(
                    layer.self_attn.q_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
                "k_proj": linear_pair(
                    layer.self_attn.k_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
                "v_proj": linear_pair(
                    layer.self_attn.v_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
                "out_proj": linear_pair(
                    layer.self_attn.out_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
            }

        layer_dict = {
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            "self_attn": self_attn,
            "ffn_layer_norm": {
                "weight": _ln_to_device(layer.ffn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn_layer_norm.bias, device=device),
            },
            "ffn": {
                "fc1": linear_pair(layer.ffn.fc1, device=device, weight_dtype=ffn_weight_dtype, **linear_kwargs),
                "fc2": linear_pair(layer.ffn.fc2, device=device, weight_dtype=ffn_weight_dtype, **linear_kwargs),
            },
        }
        layers.append(make_parameter_dict(layer_dict))
    return layers


def create_text_encoder_parameters(encoder, *, device: ttnn.Device, prefill_token_rows: int = 32) -> dict:
    """
    Convert [`SeamlessM4Tv2Encoder`] weights to TTNN tensors on ``device``.

    Token embeddings include ``embed_scale`` (see [`SeamlessM4Tv2ScaledWordEmbedding`]).
    Linear weights (QKV, out_proj, FFN) use L1 width-sharded activations + DRAM width-sharded
    BFP8 weights for ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``.
    """
    cfg = encoder.config
    scale = embed_scale_for_config(cfg)

    scaled_emb = (encoder.embed_tokens.weight.detach() * scale).contiguous()
    embed_tokens_weight = ttnn.from_torch(
        scaled_emb,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    pos_w = encoder.embed_positions.weights.detach()
    if pos_w.dtype != torch.bfloat16:
        pos_w = pos_w.to(dtype=torch.bfloat16)
    embed_positions_weight = ttnn.from_torch(
        pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    layers = _m4t_encoder_self_attn_ffn_layers(
        encoder,
        device=device,
        ffn_weight_dtype=ttnn.bfloat8_b,
        attn_weight_dtype=ttnn.bfloat8_b,
        fuse_qkv=True,
        dram_width_sharded_weights=True,
        prefill_token_rows=prefill_token_rows,
    )

    out = {
        "embed_tokens": make_parameter_dict({"weight": embed_tokens_weight}),
        "embed_positions": make_parameter_dict({"weight": embed_positions_weight}),
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(encoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(encoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def _conv_padding_int(conv: torch.nn.Module) -> int:
    p = conv.padding
    if isinstance(p, int):
        return int(p)
    return int(p[0])


def _conv1d_like_padding_int(conv: torch.nn.Module) -> int:
    """
    TTNN paths need an integral symmetric padding. HF may use ``padding='same'`` on
    [`torch.nn.Conv1d`] (string); map that to PyTorch's stride-1 effective padding.
    """
    p = conv.padding
    if isinstance(p, str):
        if p == "same":
            k = int(conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size)
            d = int(conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation)
            s = int(conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride)
            if s != 1:
                raise ValueError("padding='same' with stride != 1 is not supported for TT export")
            return (k - 1) * d // 2
        if p in ("valid", "zeros"):
            return 0
        raise ValueError(f"Unsupported Conv1d padding mode {p!r}")
    if isinstance(p, (tuple, list)):
        return int(p[0])
    return int(p)


def _conformer_feed_forward_params(
    ffn: torch.nn.Module,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    out_scale: float = 1.0,
) -> dict:
    """Preprocess conformer FFN weights.

    Stage 13a: when ``out_scale != 1.0`` (e.g. 0.5 for Macaron-style half-step residual),
    the ``output_dense`` weight and bias are pre-multiplied at load time.  This folds the
    constant runtime ``multiply(ff, 0.5)`` that follows every conformer FFN call into the
    weights, eliminating 49 scalar-multiply ops per forward pass.
    """
    od_w = ffn.output_dense.weight.detach()
    od_b = ffn.output_dense.bias.detach()
    if out_scale != 1.0:
        od_w = (od_w * out_scale).to(od_w.dtype)
        od_b = (od_b * out_scale).to(od_b.dtype)
    od_w_tt = preprocess_linear_weight(od_w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    od_b_tt = preprocess_linear_bias(od_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return make_parameter_dict(
        {
            "intermediate_dense": _linear_pair(ffn.intermediate_dense, device=device, weight_dtype=weight_dtype),
            "output_dense": {
                "weight": ttnn.to_device(od_w_tt, device),
                "bias": ttnn.to_device(od_b_tt, device),
            },
        }
    )


def _conformer_conv_module_params(conv_module: torch.nn.Module, *, device: ttnn.Device) -> dict:
    return make_parameter_dict(
        {
            "layer_norm": {
                "weight": _ln_to_device(conv_module.layer_norm.weight, device=device),
                "bias": _ln_to_device(conv_module.layer_norm.bias, device=device),
                "eps": float(conv_module.layer_norm.eps),
            },
            "pointwise_conv1": {
                "weight": _conv1d_weight(conv_module.pointwise_conv1, device=device),
                "bias": _conv1d_bias(conv_module.pointwise_conv1, device=device),
                "in_channels": conv_module.pointwise_conv1.in_channels,
                "out_channels": conv_module.pointwise_conv1.out_channels,
                "kernel_size": int(conv_module.pointwise_conv1.kernel_size[0]),
                "padding": _conv_padding_int(conv_module.pointwise_conv1),
                "stride": int(conv_module.pointwise_conv1.stride[0]),
                "groups": conv_module.pointwise_conv1.groups,
            },
            "depthwise_conv": {
                "weight": _conv1d_weight(conv_module.depthwise_conv, device=device),
                "bias": _conv1d_bias(conv_module.depthwise_conv, device=device),
                "in_channels": conv_module.depthwise_conv.in_channels,
                "out_channels": conv_module.depthwise_conv.out_channels,
                "kernel_size": int(conv_module.depthwise_conv.kernel_size[0]),
                "padding": _conv_padding_int(conv_module.depthwise_conv),
                "stride": int(conv_module.depthwise_conv.stride[0]),
                "groups": conv_module.depthwise_conv.groups,
                "left_pad": int(conv_module.depthwise_conv.kernel_size[0]) - 1,
            },
            "depthwise_layer_norm": {
                "weight": _ln_to_device(conv_module.depthwise_layer_norm.weight, device=device),
                "bias": _ln_to_device(conv_module.depthwise_layer_norm.bias, device=device),
                "eps": float(conv_module.depthwise_layer_norm.eps),
            },
            "pointwise_conv2": {
                "weight": _conv1d_weight(conv_module.pointwise_conv2, device=device),
                "bias": _conv1d_bias(conv_module.pointwise_conv2, device=device),
                "in_channels": conv_module.pointwise_conv2.in_channels,
                "out_channels": conv_module.pointwise_conv2.out_channels,
                "kernel_size": int(conv_module.pointwise_conv2.kernel_size[0]),
                "padding": _conv_padding_int(conv_module.pointwise_conv2),
                "stride": int(conv_module.pointwise_conv2.stride[0]),
                "groups": conv_module.pointwise_conv2.groups,
            },
        }
    )


def _conformer_self_attn_params(
    attn: torch.nn.Module,
    *,
    device: ttnn.Device,
    with_relative: bool,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    # Fused Q|K|V uses ``weight_dtype`` (bf8 on speech encoder for DRAM bandwidth).
    # Stage 8: for relative-position self-attention, pre-fold 1/√head_dim into Q weights
    # and bias, eliminating a runtime multiply per conformer layer.  Adapter self-attn
    # (with_relative=False) uses SDPA which applies the scale internally at runtime.
    if with_relative:
        q_scale = 1.0 / math.sqrt(int(attn.head_size))
    else:
        q_scale = 1.0
    out = {
        "qkv": _fused_qkv_pair(
            attn.linear_q,
            attn.linear_k,
            attn.linear_v,
            device=device,
            weight_dtype=weight_dtype,
            q_scale=q_scale,
        ),
        "linear_out": _linear_pair(attn.linear_out, device=device, weight_dtype=weight_dtype),
    }
    if with_relative and getattr(attn, "distance_embedding", None) is not None:
        out["distance_embedding"] = make_parameter_dict(
            {"weight": _embedding_weight(attn.distance_embedding, device=device)}
        )
        out["left_max_position_embeddings"] = int(attn.left_max_position_embeddings)
        out["right_max_position_embeddings"] = int(attn.right_max_position_embeddings)
    return make_parameter_dict(out)


def _conformer_encoder_layer_params(
    layer: torch.nn.Module,
    *,
    device: ttnn.Device,
    matmul_weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    return make_parameter_dict(
        {
            "ffn1_layer_norm": {
                "weight": _ln_to_device(layer.ffn1_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn1_layer_norm.bias, device=device),
            },
            "ffn1": _conformer_feed_forward_params(
                layer.ffn1, device=device, weight_dtype=matmul_weight_dtype, out_scale=0.5
            ),
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            "self_attn": _conformer_self_attn_params(
                layer.self_attn, device=device, with_relative=True, weight_dtype=matmul_weight_dtype
            ),
            "conv_module": _conformer_conv_module_params(layer.conv_module, device=device),
            "ffn2_layer_norm": {
                "weight": _ln_to_device(layer.ffn2_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn2_layer_norm.bias, device=device),
            },
            "ffn2": _conformer_feed_forward_params(
                layer.ffn2, device=device, weight_dtype=matmul_weight_dtype, out_scale=0.5
            ),
            "final_layer_norm": {
                "weight": _ln_to_device(layer.final_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.final_layer_norm.bias, device=device),
            },
        }
    )


def _speech_adapter_layer_params(
    layer: torch.nn.Module, *, device: ttnn.Device, matmul_weight_dtype: ttnn.DataType = ttnn.bfloat16
) -> dict:
    return make_parameter_dict(
        {
            "kernel_size": int(layer.kernel_size),
            "stride": int(layer.stride),
            "residual_layer_norm": {
                "weight": _ln_to_device(layer.residual_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.residual_layer_norm.bias, device=device),
            },
            "residual_conv": {
                "weight": _conv1d_weight(layer.residual_conv, device=device),
                "bias": _conv1d_bias(layer.residual_conv, device=device),
                "in_channels": layer.residual_conv.in_channels,
                "out_channels": layer.residual_conv.out_channels,
                "kernel_size": int(layer.residual_conv.kernel_size[0]),
                "padding": _conv1d_like_padding_int(layer.residual_conv),
                "stride": int(layer.residual_conv.stride[0]),
                "groups": layer.residual_conv.groups,
            },
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            "self_attn_conv": {
                "weight": _conv1d_weight(layer.self_attn_conv, device=device),
                "bias": _conv1d_bias(layer.self_attn_conv, device=device),
                "in_channels": layer.self_attn_conv.in_channels,
                "out_channels": layer.self_attn_conv.out_channels,
                "kernel_size": int(layer.self_attn_conv.kernel_size[0]),
                "padding": _conv1d_like_padding_int(layer.self_attn_conv),
                "stride": int(layer.self_attn_conv.stride[0]),
                "groups": layer.self_attn_conv.groups,
            },
            "self_attn": _conformer_self_attn_params(
                layer.self_attn, device=device, with_relative=False, weight_dtype=matmul_weight_dtype
            ),
            "ffn_layer_norm": {
                "weight": _ln_to_device(layer.ffn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn_layer_norm.bias, device=device),
            },
            "ffn": _conformer_feed_forward_params(layer.ffn, device=device, weight_dtype=matmul_weight_dtype),
        }
    )


def create_speech_encoder_parameters(speech_encoder, *, device: ttnn.Device) -> dict:
    """
    Convert [`SeamlessM4Tv2SpeechEncoder`] weights to TTNN tensors for [`TTSeamlessM4Tv2SpeechEncoder`].

    FFN, fused QKV, and attention out projections use ``bfloat8_b`` weights (DRAM bandwidth).
    Biases stay ``bfloat16``; activations and matmul accumulate in bf16/fp32 at runtime.
    """
    matmul_bf8 = ttnn.bfloat8_b
    fp = speech_encoder.feature_projection
    feature_projection = {
        "layer_norm": {
            "weight": _ln_to_device(fp.layer_norm.weight, device=device),
            "bias": _ln_to_device(fp.layer_norm.bias, device=device),
            "eps": float(fp.layer_norm.eps),
        },
        "projection": _linear_pair(fp.projection, device=device, weight_dtype=matmul_bf8),
    }
    enc = speech_encoder.encoder
    enc_layers = [
        _conformer_encoder_layer_params(layer, device=device, matmul_weight_dtype=matmul_bf8) for layer in enc.layers
    ]
    encoder = {
        "layers": enc_layers,
        "layer_norm": {
            "weight": _ln_to_device(enc.layer_norm.weight, device=device),
            "bias": _ln_to_device(enc.layer_norm.bias, device=device),
        },
    }
    im = speech_encoder.intermediate_ffn
    intermediate_ffn = _conformer_feed_forward_params(im, device=device, weight_dtype=matmul_bf8, out_scale=0.5)
    inner_layer_norm = {
        "weight": _ln_to_device(speech_encoder.inner_layer_norm.weight, device=device),
        "bias": _ln_to_device(speech_encoder.inner_layer_norm.bias, device=device),
    }
    out = {
        "feature_projection": make_parameter_dict(feature_projection),
        "encoder": make_parameter_dict(encoder),
        "intermediate_ffn": intermediate_ffn,
        "inner_layer_norm": make_parameter_dict(inner_layer_norm),
    }
    if speech_encoder.adapter is not None:
        adapter_layers = [
            _speech_adapter_layer_params(layer, device=device, matmul_weight_dtype=matmul_bf8)
            for layer in speech_encoder.adapter.layers
        ]
        out["adapter"] = make_parameter_dict({"layers": adapter_layers})
    return make_parameter_dict(out)


def create_text_to_unit_parameters(encoder, *, device: ttnn.Device) -> dict:
    """
    Convert the encoder submodule of Transformers
    [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] — ``model.encoder``, i.e.
    [`SeamlessM4Tv2Encoder`] with ``is_t2u_encoder=True`` — to TTNN tensors for
    [`TTSeamlessM4Tv2TextToUnitEncoder`].

    Expects ``encoder.layers`` as [`SeamlessM4Tv2EncoderLayer`] (self-attention + FFN only).
    Weights cover the transformer stack only (``inputs_embeds`` path; no token or position
    embeddings on this submodule).

    Store FFN ``fc1``/``fc2`` weights as ``bfloat8_b`` (block-float8).
    Halves DRAM bandwidth on the two largest matmuls per layer; the multiplier still
    runs at bf16 fidelity (``HiFi2`` + ``fp32_dest_acc_en``), so PCC is preserved.

    Extend the same ``bfloat8_b`` storage to the attention projections
    (fused ``qkv`` + ``out_proj`` via ``fuse_qkv=True``).  The underlying matmul
    still accumulates in fp32 with HiFi math fidelity, so the bf8 weight
    quantization is well below PCC headroom and we pick up an extra
    DRAM-bandwidth saving on the fused QKV and out projections per layer.
    """
    layers = _m4t_encoder_self_attn_ffn_layers(
        encoder,
        device=device,
        ffn_weight_dtype=ttnn.bfloat8_b,
        attn_weight_dtype=ttnn.bfloat8_b,
        fuse_qkv=True,
    )
    out = {
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(encoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(encoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def _t2u_variance_predictor_parameters(var_pred: torch.nn.Module, *, device: ttnn.Device) -> dict:
    """[`SeamlessM4Tv2VariancePredictor`] weights for TTNN text-to-unit duration path."""
    out = {
        "conv1": {
            "weight": _conv1d_weight(var_pred.conv1, device=device),
            "bias": _conv1d_bias(var_pred.conv1, device=device),
        },
        "ln1": {
            "weight": _ln_to_device(var_pred.ln1.weight, device=device),
            "bias": _ln_to_device(var_pred.ln1.bias, device=device),
        },
        "conv2": {
            "weight": _conv1d_weight(var_pred.conv2, device=device),
            "bias": _conv1d_bias(var_pred.conv2, device=device),
        },
        "ln2": {
            "weight": _ln_to_device(var_pred.ln2.weight, device=device),
            "bias": _ln_to_device(var_pred.ln2.bias, device=device),
        },
        # fp32 proj: ``log_dur`` rounding boundary must match HF; upload once here so
        # ``TTSeamlessM4Tv2TextToUnitForConditionalGeneration`` does not ``typecast`` at init.
        "proj": _linear_pair(
            var_pred.proj,
            device=device,
            weight_dtype=ttnn.float32,
            bias_dtype=ttnn.float32,
        ),
    }
    return make_parameter_dict(out)


def _t2u_decoder_layer_parameters(layer: torch.nn.Module, *, device: ttnn.Device) -> dict:
    """[`SeamlessM4Tv2TextToUnitDecoderLayer`] weights."""
    layer_dict = {
        "self_attn": {
            "qkv": _fused_qkv_pair(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                device=device,
                weight_dtype=ttnn.bfloat8_b,
            ),
            "out_proj": _linear_pair(layer.self_attn.out_proj, device=device),
        },
        "self_attn_layer_norm": {
            "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
            "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
        },
        "conv1": {
            "weight": _conv1d_weight(layer.conv1, device=device),
            "bias": _conv1d_bias(layer.conv1, device=device),
        },
        "conv2": {
            "weight": _conv1d_weight(layer.conv2, device=device),
            "bias": _conv1d_bias(layer.conv2, device=device),
        },
        "conv_layer_norm": {
            "weight": _ln_to_device(layer.conv_layer_norm.weight, device=device),
            "bias": _ln_to_device(layer.conv_layer_norm.bias, device=device),
        },
    }
    return make_parameter_dict(layer_dict)


def _t2u_decoder_parameters(decoder: torch.nn.Module, *, device: ttnn.Device) -> dict:
    """[`SeamlessM4Tv2TextToUnitDecoder`] weights (character + duration + conv decoder stack)."""
    cfg = decoder.config
    scale = embed_scale_for_config(cfg)
    # Upload embedding tables ROW-MAJOR (matches text encoder recipe).
    # ``ttnn.embedding`` produces a TILE_LAYOUT output regardless of how its weight is
    # stored; uploading the weight in ROW_MAJOR_LAYOUT avoids the trailing
    # ``UntilizeWithUnpaddingDeviceOperation`` that the embedding kernel emits when the
    # weight is already tile-padded.  Numerically identical, ~3 ops cheaper per forward.
    scaled_char = (decoder.embed_char.weight.detach() * scale).contiguous()
    embed_char_weight = ttnn.from_torch(
        scaled_char,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    char_pos_w = decoder.embed_char_positions.weights.detach()
    if char_pos_w.dtype != torch.bfloat16:
        char_pos_w = char_pos_w.to(dtype=torch.bfloat16)
    embed_char_positions_weight = ttnn.from_torch(
        char_pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_w = decoder.embed_positions.weights.detach()
    if pos_w.dtype != torch.bfloat16:
        pos_w = pos_w.to(dtype=torch.bfloat16)
    embed_positions_weight = ttnn.from_torch(
        pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_emb_alpha_char = ttnn.from_torch(
        decoder.pos_emb_alpha_char.detach().reshape(1, 1, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_emb_alpha = ttnn.from_torch(
        decoder.pos_emb_alpha.detach().reshape(1, 1, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    layers = [_t2u_decoder_layer_parameters(layer, device=device) for layer in decoder.layers]
    out = {
        "embed_char": make_parameter_dict({"weight": embed_char_weight}),
        "embed_char_positions": make_parameter_dict({"weight": embed_char_positions_weight}),
        "embed_positions": make_parameter_dict({"weight": embed_positions_weight}),
        "pos_emb_alpha_char": pos_emb_alpha_char,
        "pos_emb_alpha": pos_emb_alpha,
        "duration_predictor": _t2u_variance_predictor_parameters(decoder.duration_predictor, device=device),
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(decoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(decoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def create_text_to_unit_condgen_parameters(
    t2u: torch.nn.Module,
    *,
    device: ttnn.Device,
) -> dict:
    """
    Full [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] weights for TTNN:
    ``model.encoder``, ``model.decoder``, and ``lm_head``.
    """
    w_lm = preprocess_linear_weight(
        t2u.lm_head.weight.detach(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    out = {
        "encoder": create_text_to_unit_parameters(t2u.model.encoder, device=device),
        "decoder": _t2u_decoder_parameters(t2u.model.decoder, device=device),
        "lm_head": make_parameter_dict({"weight": ttnn.to_device(w_lm, device)}),
    }
    return make_parameter_dict(out)


def create_code_hifigan_parameters(vocoder, *, device: ttnn.Device) -> dict:
    """
    Convert [`SeamlessM4Tv2CodeHifiGan`] (``model.vocoder``) to TTNN tensors for
    [`TTSeamlessM4Tv2CodeHifiGan`].
    """
    dp = vocoder.dur_predictor
    dur_predictor = {
        "conv1": {
            "weight": _conv1d_weight(dp.conv1, device=device),
            "bias": _conv1d_bias(dp.conv1, device=device),
            "in_channels": dp.conv1.in_channels,
            "out_channels": dp.conv1.out_channels,
            "kernel_size": int(dp.conv1.kernel_size[0]),
            "padding": _conv1d_like_padding_int(dp.conv1),
        },
        "ln1": {
            "weight": _ln_to_device(dp.ln1.weight, device=device),
            "bias": _ln_to_device(dp.ln1.bias, device=device),
            "eps": float(dp.ln1.eps),
        },
        "conv2": {
            "weight": _conv1d_weight(dp.conv2, device=device),
            "bias": _conv1d_bias(dp.conv2, device=device),
            "in_channels": dp.conv2.in_channels,
            "out_channels": dp.conv2.out_channels,
            "kernel_size": int(dp.conv2.kernel_size[0]),
            "padding": _conv1d_like_padding_int(dp.conv2),
        },
        "ln2": {
            "weight": _ln_to_device(dp.ln2.weight, device=device),
            "bias": _ln_to_device(dp.ln2.bias, device=device),
            "eps": float(dp.ln2.eps),
        },
        "proj": _linear_pair(dp.proj, device=device),
    }

    hg = vocoder.hifi_gan
    upsampler_layers = []
    for layer in hg.upsampler:
        assert isinstance(layer, torch.nn.ConvTranspose1d)
        k = int(layer.kernel_size[0])
        s = int(layer.stride[0])
        p = _conv1d_like_padding_int(layer)
        upsampler_layers.append(
            {
                "weight": _conv_transpose1d_weight(layer, device=device),
                "bias": _conv_transpose1d_bias_host(layer),
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
            }
        )

    resblock_layers = []
    for rb in hg.resblocks:
        c1 = []
        c2 = []
        for c in rb.convs1:
            c1.append(
                {
                    "weight": _conv1d_weight(c, device=device),
                    "bias": _conv1d_bias(c, device=device),
                    "kernel_size": int(c.kernel_size[0]),
                    "dilation": int(c.dilation[0]),
                    "padding": _conv1d_like_padding_int(c),
                    "in_channels": c.in_channels,
                    "out_channels": c.out_channels,
                }
            )
        for c in rb.convs2:
            c2.append(
                {
                    "weight": _conv1d_weight(c, device=device),
                    "bias": _conv1d_bias(c, device=device),
                    "kernel_size": int(c.kernel_size[0]),
                    "dilation": int(c.dilation[0]),
                    "padding": _conv1d_like_padding_int(c),
                    "in_channels": c.in_channels,
                    "out_channels": c.out_channels,
                }
            )
        resblock_layers.append(make_parameter_dict({"convs1": c1, "convs2": c2}))

    out = {
        "unit_embedding": make_parameter_dict(
            {"weight": _vocoder_embedding_weight_row_major(vocoder.unit_embedding, device=device)}
        ),
        "speaker_embedding": make_parameter_dict(
            {"weight": _vocoder_embedding_weight_row_major(vocoder.speaker_embedding, device=device)}
        ),
        "language_embedding": make_parameter_dict(
            {"weight": _vocoder_embedding_weight_row_major(vocoder.language_embedding, device=device)}
        ),
        "dur_predictor": make_parameter_dict(dur_predictor),
        "hifi_gan": make_parameter_dict(
            {
                "conv_pre": {
                    "weight": _conv1d_weight(hg.conv_pre, device=device),
                    "bias": _conv1d_bias(hg.conv_pre, device=device),
                    "kernel_size": int(hg.conv_pre.kernel_size[0]),
                    "padding": _conv1d_like_padding_int(hg.conv_pre),
                    "in_channels": hg.conv_pre.in_channels,
                    "out_channels": hg.conv_pre.out_channels,
                },
                "upsampler": upsampler_layers,
                "resblocks": resblock_layers,
                "conv_post": {
                    "weight": _conv1d_weight(hg.conv_post, device=device),
                    "bias": _conv1d_bias(hg.conv_post, device=device),
                    "kernel_size": int(hg.conv_post.kernel_size[0]),
                    "padding": _conv1d_like_padding_int(hg.conv_post),
                    "in_channels": hg.conv_post.in_channels,
                    "out_channels": hg.conv_post.out_channels,
                },
            }
        ),
    }
    return make_parameter_dict(out)


def create_seamless_m4t_v2_model_parameters(model: torch.nn.Module, *, device: ttnn.Device) -> dict:
    """
    Full [`SeamlessM4Tv2Model`] weights for TTNN: ``text_encoder``, ``text_decoder``, ``speech_encoder``,
    main ``lm_head``, ``t2u_model``, and ``vocoder`` (same submodules as Hugging Face).
    """
    w_lm = preprocess_linear_weight(
        model.lm_head.weight.detach(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )
    out = {
        "text_encoder": create_text_encoder_parameters(model.text_encoder, device=device),
        "text_decoder": create_text_decoder_parameters(model.text_decoder, device=device),
        "speech_encoder": create_speech_encoder_parameters(model.speech_encoder, device=device),
        "lm_head": make_parameter_dict({"weight": ttnn.to_device(w_lm, device)}),
        "t2u": create_text_to_unit_condgen_parameters(model.t2u_model, device=device),
        "vocoder": create_code_hifigan_parameters(model.vocoder, device=device),
    }
    return make_parameter_dict(out)
