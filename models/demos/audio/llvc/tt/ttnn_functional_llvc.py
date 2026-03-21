# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of LLVC (Low-Latency Low-Resource Voice Conversion).

This module implements the LLVC model using TTNN operations for execution
on Tenstorrent hardware. The implementation follows the reference PyTorch model
from KoeAI/LLVC and maps operations to efficient TTNN equivalents.

Stage 3 optimization strategy:
  - Input Conv1d runs on device (BLOCK_SHARDED fallback for L1 overflow)
  - Transformer decoder (attention, FFN, LN) on device with pre-uploaded weights
  - Label embedding (linear, LN, ReLU) on device with pre-uploaded weights
  - All decoder/label weights pre-uploaded during preprocessing (zero per-inference transfers)
  - L1 memory for attention intermediates, DRAM for long-lived tensors
  - Encoder (dilated causal convolutions, all layers) stays on CPU
    (hybrid approach tested but reverted — transfer overhead kills streaming)
  - MaskNet projections (grouped 1x1 convs) stay on CPU
  - Output ConvTranspose1d stays on CPU (no TTNN equivalent)

F0 Mode Support:
  - F0-free mode (default): Uses label/speaker embedding for voice identity.
  - F0-based mode: Not implemented (requires additional model components).

Vocoder Integration:
  - LLVC integrates the synthesis step directly via ConvTranspose1d
    in the output stage.
"""

import math

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

# Memory configs
LLVC_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
LLVC_L1_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG
LLVC_L1_SMALL_SIZE = 16384


def _get_compute_kernel_config(device):
    """Get compute kernel config for linear/layernorm ops on device."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _get_core_grid(device):
    """Get the compute grid from the device for ttnn.linear core_grid param."""
    compute_grid_size = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_grid_size.y, x=compute_grid_size.x)


def _is_mesh_device(device):
    """Check if device is a multi-device mesh (e.g., N300s with 2 chips)."""
    if device is None:
        return False
    try:
        return device.get_num_devices() > 1
    except (AttributeError, Exception):
        return False


def _to_device(tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Convert a torch tensor to TTNN tensor on device, handling mesh devices."""
    if _is_mesh_device(device):
        return ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)


def _from_device(tensor, device, batch_size=1):
    """Convert a TTNN tensor to torch, handling mesh devices."""
    if _is_mesh_device(device):
        t = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        return t[:batch_size] if t.shape[0] > batch_size else t
    return ttnn.to_torch(tensor)


def _get_param(params, *keys):
    """Safely navigate nested parameter dict."""
    val = params
    for k in keys:
        val = val[k]
    return val


def _depthwise_separable_conv_torch(x, params, dilation=1):
    """
    Depthwise separable convolution entirely in torch (CPU).

    The hybrid CPU/device approach was tested in Stage 3 but reverted
    because the per-layer host↔device transfer overhead (3 transfers
    × 8 layers × N chunks) kills streaming performance for small tensors.

    Sequential layers:
      0: Conv1d (depthwise, groups=channels)
      1: LayerNormPermuted
      2: ReLU
      3: Conv1d (pointwise, 1x1)
      4: LayerNormPermuted
      5: ReLU
    """
    layers = params["layers"]

    # Depthwise conv (groups=in_channels)
    dw_weight = layers["0"]["weight"]
    dw_bias = layers["0"].get("bias", None)
    channels = dw_weight.shape[0]
    x = F.conv1d(x, dw_weight, bias=dw_bias, stride=1, padding=0, dilation=dilation, groups=channels)

    # LayerNorm (permuted: apply on channel dim)
    x = x.permute(0, 2, 1)
    x = F.layer_norm(x, [channels], layers["1"]["weight"], layers["1"]["bias"])
    x = x.permute(0, 2, 1)
    x = torch.relu(x)

    # Pointwise conv (1x1)
    pw_weight = layers["3"]["weight"]
    pw_bias = layers["3"].get("bias", None)
    out_channels = pw_weight.shape[0]
    x = F.conv1d(x, pw_weight, bias=pw_bias, stride=1, padding=0)

    # LayerNorm (permuted)
    x = x.permute(0, 2, 1)
    x = F.layer_norm(x, [out_channels], layers["4"]["weight"], layers["4"]["bias"])
    x = x.permute(0, 2, 1)
    x = torch.relu(x)

    return x


def _dilated_causal_conv_encoder_torch(x, ctx_buf, params, config, device=None):
    """
    Dilated causal convolution encoder — runs entirely on CPU.

    Uses grouped convolutions (groups=channels) which are not supported
    by ttnn.conv1d. The hybrid CPU/device approach was tested but reverted
    because the per-layer transfer overhead kills streaming performance.

    Args:
        x: [B, channels, T] tensor
        ctx_buf: [B, channels, total_buf_len] context buffer
        params: encoder parameters dict
        config: dict with num_layers, buf_lengths, buf_indices
        device: unused (kept for API compatibility)
    Returns:
        x, ctx_buf (both updated)
    """
    num_layers = config["num_layers"]
    buf_lengths = config["buf_lengths"]
    buf_indices = config["buf_indices"]

    for i in range(num_layers):
        buf_start = buf_indices[i]
        buf_end = buf_start + buf_lengths[i]

        # Prepend context buffer slice
        dcc_in = torch.cat((ctx_buf[..., buf_start:buf_end], x), dim=-1)
        # Update context buffer
        ctx_buf[..., buf_start:buf_end] = dcc_in[..., -buf_lengths[i] :]

        # Apply depthwise separable conv with dilation=2^i (CPU only)
        dcc_out = _depthwise_separable_conv_torch(dcc_in, params["dcc_layers"][f"dcc_{i}"], dilation=2**i)

        # Residual connection
        x = x + dcc_out

    return x, ctx_buf


def _causal_unfold(x, unfold, ctx_len, chunk_size):
    """Causal unfold: reshape for chunk-wise processing."""
    B, T, C = x.shape
    x = x.permute(0, 2, 1)
    x = unfold(x.unsqueeze(-1))
    x = x.permute(0, 2, 1)
    x = x.reshape(B, -1, C, ctx_len + chunk_size)
    x = x.reshape(-1, C, ctx_len + chunk_size)
    x = x.permute(0, 2, 1)
    return x


def _get_pos_enc(seq_len, d_model, max_len=200):
    """Generate sinusoidal positional encoding."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe[:seq_len].unsqueeze(0)


def _transformer_decoder_layer_torch(tgt, memory, params, chunk_size):
    """
    Single causal transformer decoder layer in torch.

    Performs self-attention on last chunk_size tokens, cross-attention with memory,
    and feed-forward network.
    """
    nhead = params["nhead"]
    tgt_last = tgt[:, -chunk_size:, :]

    # Self attention
    sa_params = params["self_attn"]
    embed_dim = sa_params["in_proj_weight"].shape[1]
    tmp, _ = F.multi_head_attention_forward(
        tgt_last.transpose(0, 1),
        tgt.transpose(0, 1),
        tgt.transpose(0, 1),
        embed_dim,
        nhead,
        sa_params["in_proj_weight"],
        sa_params["in_proj_bias"],
        None,
        None,
        False,
        0.0,
        sa_params["out_proj"]["weight"],
        sa_params["out_proj"]["bias"],
        training=False,
    )
    tgt_last = tgt_last + tmp.transpose(0, 1)

    # Norm1
    tgt_last = F.layer_norm(
        tgt_last,
        [tgt_last.shape[-1]],
        params["norm1"]["weight"],
        params["norm1"]["bias"],
    )

    # Cross attention (encoder-decoder)
    if memory is not None:
        ca_params = params["multihead_attn"]
        tmp, _ = F.multi_head_attention_forward(
            tgt_last.transpose(0, 1),
            memory.transpose(0, 1),
            memory.transpose(0, 1),
            ca_params["in_proj_weight"].shape[1],
            nhead,
            ca_params["in_proj_weight"],
            ca_params["in_proj_bias"],
            None,
            None,
            False,
            0.0,
            ca_params["out_proj"]["weight"],
            ca_params["out_proj"]["bias"],
            training=False,
        )
        tgt_last = tgt_last + tmp.transpose(0, 1)

        # Norm2
        tgt_last = F.layer_norm(
            tgt_last,
            [tgt_last.shape[-1]],
            params["norm2"]["weight"],
            params["norm2"]["bias"],
        )

    # Feed-forward
    ff = F.linear(tgt_last, params["linear1"]["weight"], params["linear1"]["bias"])
    ff = torch.relu(ff)
    ff = F.linear(ff, params["linear2"]["weight"], params["linear2"]["bias"])
    tgt_last = tgt_last + ff

    # Norm3
    tgt_last = F.layer_norm(
        tgt_last,
        [tgt_last.shape[-1]],
        params["norm3"]["weight"],
        params["norm3"]["bias"],
    )

    return tgt_last


def _transformer_decoder_layer_ttnn(tgt, memory, params, chunk_size, device):
    """
    Single causal transformer decoder layer on TT device via TTNN ops.

    Stage 3: Uses pre-uploaded weights from params["_ttnn"] to eliminate
    host→device weight transfer overhead. Only activation tensors are transferred.
    Uses L1 memory for attention intermediates.
    """
    nhead = params["nhead"]
    tw = params.get("_ttnn")
    if tw is None:
        # Fallback: no pre-uploaded weights, use Stage 2 path
        return _transformer_decoder_layer_ttnn_fallback(tgt, memory, params, chunk_size, device)

    embed_dim = params["self_attn"]["in_proj_weight"].shape[1]
    head_dim = embed_dim // nhead
    scale = head_dim**-0.5
    compute_config = _get_compute_kernel_config(device)

    tgt_last = tgt[:, -chunk_size:, :]
    B = tgt.shape[0]
    S_q = chunk_size
    S_kv = tgt.shape[1]

    # --- Self attention with pre-uploaded weights ---
    tgt_last_tt = _to_device(tgt_last, device)
    tgt_tt = _to_device(tgt, device)

    q_tt = ttnn.linear(tgt_last_tt, tw["sa_wq"], bias=tw["sa_bq"], memory_config=LLVC_L1_MEMORY_CONFIG)
    k_tt = ttnn.linear(tgt_tt, tw["sa_wk"], bias=tw["sa_bk"], memory_config=LLVC_L1_MEMORY_CONFIG)
    v_tt = ttnn.linear(tgt_tt, tw["sa_wv"], bias=tw["sa_bv"], memory_config=LLVC_L1_MEMORY_CONFIG)
    ttnn.deallocate(tgt_last_tt)
    ttnn.deallocate(tgt_tt)

    # Reshape to [B, nhead, S, head_dim]
    q_tt = ttnn.reshape(q_tt, (B, S_q, nhead, head_dim))
    q_tt = ttnn.transpose(q_tt, 1, 2)
    k_tt = ttnn.reshape(k_tt, (B, S_kv, nhead, head_dim))
    k_tt = ttnn.transpose(k_tt, 1, 2)
    v_tt = ttnn.reshape(v_tt, (B, S_kv, nhead, head_dim))
    v_tt = ttnn.transpose(v_tt, 1, 2)

    # Attention: Q @ K^T * scale
    k_t = ttnn.transpose(k_tt, -2, -1)
    attn_weights = ttnn.matmul(q_tt, k_t, memory_config=LLVC_L1_MEMORY_CONFIG)
    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_t)
    attn_weights = ttnn.multiply(attn_weights, scale)
    attn_weights = ttnn.softmax(attn_weights, dim=-1)

    attn_out = ttnn.matmul(attn_weights, v_tt, memory_config=LLVC_L1_MEMORY_CONFIG)
    ttnn.deallocate(attn_weights)
    ttnn.deallocate(v_tt)

    attn_out = ttnn.transpose(attn_out, 1, 2)
    attn_out = ttnn.reshape(attn_out, (B, S_q, embed_dim))

    # Output projection
    attn_out = ttnn.linear(attn_out, tw["sa_out_w"], bias=tw["sa_out_b"], memory_config=LLVC_MEMORY_CONFIG)

    # Residual + Norm1
    tgt_last_res = _to_device(tgt_last, device)
    tgt_last_tt = ttnn.add(tgt_last_res, attn_out)
    ttnn.deallocate(attn_out)
    ttnn.deallocate(tgt_last_res)

    tgt_last_tt = ttnn.layer_norm(
        tgt_last_tt,
        weight=tw["norm1"]["weight"],
        bias=tw["norm1"]["bias"],
        memory_config=LLVC_MEMORY_CONFIG,
        compute_kernel_config=compute_config,
    )

    # --- Cross attention with pre-uploaded weights ---
    if memory is not None and "ca_wq" in tw:
        mem_tt = _to_device(memory, device)
        S_mem = memory.shape[1]

        ca_q = ttnn.linear(tgt_last_tt, tw["ca_wq"], bias=tw["ca_bq"], memory_config=LLVC_L1_MEMORY_CONFIG)
        ca_k = ttnn.linear(mem_tt, tw["ca_wk"], bias=tw["ca_bk"], memory_config=LLVC_L1_MEMORY_CONFIG)
        ca_v = ttnn.linear(mem_tt, tw["ca_wv"], bias=tw["ca_bv"], memory_config=LLVC_L1_MEMORY_CONFIG)
        ttnn.deallocate(mem_tt)

        ca_q = ttnn.reshape(ca_q, (B, S_q, nhead, head_dim))
        ca_q = ttnn.transpose(ca_q, 1, 2)
        ca_k = ttnn.reshape(ca_k, (B, S_mem, nhead, head_dim))
        ca_k = ttnn.transpose(ca_k, 1, 2)
        ca_v = ttnn.reshape(ca_v, (B, S_mem, nhead, head_dim))
        ca_v = ttnn.transpose(ca_v, 1, 2)

        ca_k_t = ttnn.transpose(ca_k, -2, -1)
        ca_attn = ttnn.matmul(ca_q, ca_k_t, memory_config=LLVC_L1_MEMORY_CONFIG)
        ttnn.deallocate(ca_q)
        ttnn.deallocate(ca_k_t)
        ca_attn = ttnn.multiply(ca_attn, scale)
        ca_attn = ttnn.softmax(ca_attn, dim=-1)
        ca_out = ttnn.matmul(ca_attn, ca_v, memory_config=LLVC_L1_MEMORY_CONFIG)
        ttnn.deallocate(ca_attn)
        ttnn.deallocate(ca_v)

        ca_out = ttnn.transpose(ca_out, 1, 2)
        ca_out = ttnn.reshape(ca_out, (B, S_q, embed_dim))

        ca_out = ttnn.linear(ca_out, tw["ca_out_w"], bias=tw["ca_out_b"], memory_config=LLVC_MEMORY_CONFIG)

        tgt_last_tt = ttnn.add(tgt_last_tt, ca_out)
        ttnn.deallocate(ca_out)

        tgt_last_tt = ttnn.layer_norm(
            tgt_last_tt,
            weight=tw["norm2"]["weight"],
            bias=tw["norm2"]["bias"],
            memory_config=LLVC_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )

    # --- FFN with pre-uploaded weights ---
    ff_out = ttnn.linear(tgt_last_tt, tw["ff_w1"], bias=tw["ff_b1"], memory_config=LLVC_L1_MEMORY_CONFIG)
    ff_out = ttnn.relu(ff_out)
    ff_out = ttnn.linear(ff_out, tw["ff_w2"], bias=tw["ff_b2"], memory_config=LLVC_MEMORY_CONFIG)

    tgt_last_tt = ttnn.add(tgt_last_tt, ff_out)
    ttnn.deallocate(ff_out)

    # Norm3
    tgt_last_tt = ttnn.layer_norm(
        tgt_last_tt,
        weight=tw["norm3"]["weight"],
        bias=tw["norm3"]["bias"],
        memory_config=LLVC_MEMORY_CONFIG,
        compute_kernel_config=compute_config,
    )

    # Back to torch
    result = _from_device(tgt_last_tt, device, batch_size=B).float()
    ttnn.deallocate(tgt_last_tt)
    return result


def _transformer_decoder_layer_ttnn_fallback(tgt, memory, params, chunk_size, device):
    """Fallback: run decoder layer via torch when no pre-uploaded weights."""
    return _transformer_decoder_layer_torch(tgt, memory, params, chunk_size)


def _causal_transformer_decoder_torch(tgt, mem, ctx_buf, params, config):
    """
    Full causal transformer decoder in torch.

    Args:
        tgt: [B, C, T] target
        mem: [B, C, T] memory (encoder output)
        ctx_buf: [B, num_layers+1, ctx_len, model_dim]
        params: decoder parameters
        config: dict with ctx_len, chunk_size, num_layers, use_pos_enc
    Returns:
        tgt, ctx_buf (both updated)
    """
    ctx_len = config["ctx_len"]
    chunk_size = config["chunk_size"]
    num_layers = config["num_layers"]
    use_pos_enc = config["use_pos_enc"]

    # Mod pad
    mem, _ = _mod_pad(mem, chunk_size, (0, 0))
    tgt, mod = _mod_pad(tgt, chunk_size, (0, 0))

    B, C, T = tgt.shape
    tgt = tgt.permute(0, 2, 1)  # [B, T, C]
    mem = mem.permute(0, 2, 1)  # [B, T, C]

    # Prepend context for memory
    mem = torch.cat((ctx_buf[:, 0, :, :], mem), dim=1)
    ctx_buf[:, 0, :, :] = mem[:, -ctx_len:, :]

    unfold = torch.nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)
    mem_ctx = _causal_unfold(mem, unfold, ctx_len, chunk_size)

    if use_pos_enc:
        pe = _get_pos_enc(mem_ctx.shape[1], C)
        mem_ctx = mem_ctx + pe

    K = 1000  # Batch processing chunk size
    for i in range(num_layers):
        tgt = torch.cat((ctx_buf[:, i + 1, :, :], tgt), dim=1)
        ctx_buf[:, i + 1, :, :] = tgt[:, -ctx_len:, :]

        tgt_ctx = _causal_unfold(tgt, unfold, ctx_len, chunk_size)
        if use_pos_enc and i == 0:
            pe = _get_pos_enc(tgt_ctx.shape[1], C)
            tgt_ctx = tgt_ctx + pe

        tgt_out = torch.zeros_like(tgt_ctx)[:, -chunk_size:, :]

        layer_params = params["tf_dec_layers"][str(i)]

        for j in range(int(math.ceil(tgt_out.shape[0] / K))):
            s = slice(j * K, (j + 1) * K)
            tgt_out[s] = _transformer_decoder_layer_torch(tgt_ctx[s], mem_ctx[s], layer_params, chunk_size)

        tgt = tgt_out.reshape(B, T, C)

    tgt = tgt.permute(0, 2, 1)  # [B, C, T]
    if mod != 0:
        tgt = tgt[..., :-mod]

    return tgt, ctx_buf


def _causal_transformer_decoder_device(tgt, mem, ctx_buf, params, config, device):
    """
    Causal transformer decoder with TTNN device acceleration.

    Context buffer management (cat, unfold, reshape) stays in torch.
    The compute-heavy transformer layer (attention + FFN) runs on TT device.
    """
    ctx_len = config["ctx_len"]
    chunk_size = config["chunk_size"]
    num_layers = config["num_layers"]
    use_pos_enc = config["use_pos_enc"]

    mem, _ = _mod_pad(mem, chunk_size, (0, 0))
    tgt, mod = _mod_pad(tgt, chunk_size, (0, 0))

    B, C, T = tgt.shape
    tgt = tgt.permute(0, 2, 1)
    mem = mem.permute(0, 2, 1)

    mem = torch.cat((ctx_buf[:, 0, :, :], mem), dim=1)
    ctx_buf[:, 0, :, :] = mem[:, -ctx_len:, :]

    unfold = torch.nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)
    mem_ctx = _causal_unfold(mem, unfold, ctx_len, chunk_size)

    if use_pos_enc:
        pe = _get_pos_enc(mem_ctx.shape[1], C)
        mem_ctx = mem_ctx + pe

    K = 1000
    for i in range(num_layers):
        tgt = torch.cat((ctx_buf[:, i + 1, :, :], tgt), dim=1)
        ctx_buf[:, i + 1, :, :] = tgt[:, -ctx_len:, :]

        tgt_ctx = _causal_unfold(tgt, unfold, ctx_len, chunk_size)
        if use_pos_enc and i == 0:
            pe = _get_pos_enc(tgt_ctx.shape[1], C)
            tgt_ctx = tgt_ctx + pe

        tgt_out = torch.zeros_like(tgt_ctx)[:, -chunk_size:, :]
        layer_params = params["tf_dec_layers"][str(i)]

        # Use TTNN device for the transformer layer compute
        for j in range(int(math.ceil(tgt_out.shape[0] / K))):
            s = slice(j * K, (j + 1) * K)
            tgt_out[s] = _transformer_decoder_layer_ttnn(tgt_ctx[s], mem_ctx[s], layer_params, chunk_size, device)

        tgt = tgt_out.reshape(B, T, C)

    tgt = tgt.permute(0, 2, 1)
    if mod != 0:
        tgt = tgt[..., :-mod]

    return tgt, ctx_buf


def _label_embedding_device(label, params, enc_dim, device):
    """
    Label embedding on TT device using ttnn.linear + ttnn.layer_norm.

    Linear(1,512) -> LN(512) -> ReLU -> Linear(512,enc_dim) -> LN(enc_dim) -> ReLU
    """
    compute_config = _get_compute_kernel_config(device)

    tw = params.get("_ttnn")
    l_tt = _to_device(label, device)

    if tw is not None:
        # Stage 3: use pre-uploaded weights
        l_tt = ttnn.linear(l_tt, tw["w0"], bias=tw["b0"], memory_config=LLVC_MEMORY_CONFIG)
        l_tt = ttnn.layer_norm(
            l_tt,
            weight=tw["ln1"]["weight"],
            bias=tw["ln1"]["bias"],
            memory_config=LLVC_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        l_tt = ttnn.relu(l_tt)

        l_tt = ttnn.linear(l_tt, tw["w3"], bias=tw["b3"], memory_config=LLVC_MEMORY_CONFIG)
        l_tt = ttnn.layer_norm(
            l_tt,
            weight=tw["ln4"]["weight"],
            bias=tw["ln4"]["bias"],
            memory_config=LLVC_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        l_tt = ttnn.relu(l_tt)
    else:
        # Fallback: upload weights per call
        w0 = _to_device(params["0"]["weight"].T.contiguous(), device)
        b0 = _to_device(params["0"]["bias"].unsqueeze(0), device)
        l_tt = ttnn.linear(l_tt, w0, bias=b0, memory_config=LLVC_MEMORY_CONFIG)
        ln1_w = _to_device(params["1"]["weight"].unsqueeze(0), device)
        ln1_b = _to_device(params["1"]["bias"].unsqueeze(0), device)
        l_tt = ttnn.layer_norm(
            l_tt,
            weight=ln1_w,
            bias=ln1_b,
            memory_config=LLVC_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        l_tt = ttnn.relu(l_tt)

        w3 = _to_device(params["3"]["weight"].T.contiguous(), device)
        b3 = _to_device(params["3"]["bias"].unsqueeze(0), device)
        l_tt = ttnn.linear(l_tt, w3, bias=b3, memory_config=LLVC_MEMORY_CONFIG)
        ln4_w = _to_device(params["4"]["weight"].unsqueeze(0), device)
        ln4_b = _to_device(params["4"]["bias"].unsqueeze(0), device)
        l_tt = ttnn.layer_norm(
            l_tt,
            weight=ln4_w,
            bias=ln4_b,
            memory_config=LLVC_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        l_tt = ttnn.relu(l_tt)

    # Back to torch
    result = _from_device(l_tt, device, batch_size=label.shape[0]).float()
    ttnn.deallocate(l_tt)
    return result


def _mask_net(
    x, label_emb, enc_buf, dec_buf, params, encoder_config, decoder_config, skip_connection, proj, device=None
):
    """
    MaskNet forward pass, with optional TT device acceleration.

    The encoder (dilated causal convolutions with groups>1) always runs on CPU.
    The decoder (transformer attention) runs on TT device when available.
    Projection layers (grouped 1x1 convs) stay on CPU.

    Args:
        x: [B, enc_dim, T]
        label_emb: [B, enc_dim] label embedding
        enc_buf, dec_buf: context buffers
        params: MaskNet parameters
        device: TTNN device or None for CPU-only
    Returns:
        m, enc_buf, dec_buf
    """
    # Encoder (dilated causal convolutions — depthwise on CPU, pointwise on device)
    e, enc_buf = _dilated_causal_conv_encoder_torch(x, enc_buf, params["encoder"], encoder_config, device=device)

    # Label integration: broadcast multiply
    l_expanded = label_emb.unsqueeze(2) * e  # [B, enc_dim, T]

    if proj:
        # Project encoder output: enc_dim -> dec_dim
        w_e2d_e = params["proj_e2d_e"]["0"]["weight"]
        b_e2d_e = params["proj_e2d_e"]["0"].get("bias", None)
        e_proj = F.conv1d(e, w_e2d_e, bias=b_e2d_e, groups=e.shape[1] // w_e2d_e.shape[1])
        e_proj = torch.relu(e_proj)

        # Project label: enc_dim -> dec_dim
        w_e2d_l = params["proj_e2d_l"]["0"]["weight"]
        b_e2d_l = params["proj_e2d_l"]["0"].get("bias", None)
        m = F.conv1d(l_expanded, w_e2d_l, bias=b_e2d_l, groups=l_expanded.shape[1] // w_e2d_l.shape[1])
        m = torch.relu(m)

        # Decoder — route to device when available
        if device is not None:
            m, dec_buf = _causal_transformer_decoder_device(
                m, e_proj, dec_buf, params["decoder"], decoder_config, device
            )
        else:
            m, dec_buf = _causal_transformer_decoder_torch(m, e_proj, dec_buf, params["decoder"], decoder_config)

        # Project back: dec_dim -> enc_dim
        w_d2e = params["proj_d2e"]["0"]["weight"]
        b_d2e = params["proj_d2e"]["0"].get("bias", None)
        m = F.conv1d(m, w_d2e, bias=b_d2e, groups=m.shape[1] // w_d2e.shape[1])
        m = torch.relu(m)
    else:
        if device is not None:
            m, dec_buf = _causal_transformer_decoder_device(
                l_expanded, e, dec_buf, params["decoder"], decoder_config, device
            )
        else:
            m, dec_buf = _causal_transformer_decoder_torch(l_expanded, e, dec_buf, params["decoder"], decoder_config)

    if skip_connection:
        m = l_expanded + m

    return m, enc_buf, dec_buf


def _mod_pad(x, chunk_size, pad):
    """Mod pad to ensure integer number of chunks."""
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)
    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)
    return x, mod


def _input_conv_device(x_torch, conv_params, device, L, config):
    """
    Run input Conv1d on TT device using ttnn.conv1d.

    This is the primary TTNN device operation in Stage 1.
    Input waveform [B, 1, T] is encoded into latent space [B, enc_dim, T//L].

    Args:
        x_torch: [B, 1, T_padded] input (already padded, float32)
        conv_params: dict with "weight" key (TTNN tensor, ROW_MAJOR)
        device: TTNN device or MeshDevice
        L: hop length (stride)
        config: model config dict
    Returns:
        x_torch: [B, enc_dim, T_out] (float32 on CPU)
    """
    B = x_torch.shape[0]
    T = x_torch.shape[-1]
    enc_dim = config["enc_dim"]
    lookahead = config.get("lookahead", True)
    kernel_size = 3 * L if lookahead else L

    # Prepare input: [B, 1, T] → [B, T, 1] (channels-last for TTNN conv1d)
    x_cl = x_torch.permute(0, 2, 1).contiguous()

    # Stage 3: Try BLOCK_SHARDED first (handles longer sequences),
    # fall back to HEIGHT_SHARDED, then CPU if both fail.
    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Try BLOCK_SHARDED, if that fails try HEIGHT_SHARDED
    x_tt = _to_device(x_cl, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    try:
        conv_out, [weights_device, _] = ttnn.conv1d(
            input_tensor=x_tt,
            weight_tensor=conv_params["weight"],
            device=device,
            in_channels=1,
            out_channels=enc_dim,
            batch_size=B,
            input_length=T,
            kernel_size=kernel_size,
            stride=L,
            padding=0,
            dilation=1,
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=conv_config,
            compute_config=compute_config,
            return_weights_and_bias=True,
        )
    except Exception:
        # Fall back to HEIGHT_SHARDED
        conv_config_hs = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        x_tt = _to_device(x_cl, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        conv_out, [weights_device, _] = ttnn.conv1d(
            input_tensor=x_tt,
            weight_tensor=conv_params["weight"],
            device=device,
            in_channels=1,
            out_channels=enc_dim,
            batch_size=B,
            input_length=T,
            kernel_size=kernel_size,
            stride=L,
            padding=0,
            dilation=1,
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=conv_config_hs,
            compute_config=compute_config,
            return_weights_and_bias=True,
        )

    # Store device weights for potential reuse
    conv_params["weight"] = weights_device

    # ReLU on TT device
    conv_out = ttnn.relu(conv_out)

    # Convert from sharded to interleaved memory for to_torch
    conv_out = ttnn.sharded_to_interleaved(conv_out)

    # To torch: output is [B, T_out, C_out] in channels-last format
    x_out = _from_device(conv_out, device, batch_size=B).float()

    # Channels-last → channels-first: [B, T_out, C_out] → [B, C_out, T_out]
    if x_out.dim() == 4:
        x_out = x_out.squeeze(1)
    x_out = x_out.permute(0, 2, 1)

    return x_out


def ttnn_llvc_forward(
    x,
    *,
    parameters,
    config,
    device,
    init_enc_buf=None,
    init_dec_buf=None,
    init_out_buf=None,
    convnet_pre_ctx=None,
    pad=True,
    use_f0=False,
):
    """
    Full LLVC forward pass with TTNN device acceleration (Stage 2).

    On TT device: input Conv1d, transformer decoder (attention + FFN + LN),
    and label embedding. On CPU: dilated causal conv encoder (grouped convs),
    MaskNet projections (grouped 1x1 convs), output ConvTranspose1d.

    Args:
        x: input audio as TTNN tensor [B, 1, T]
        parameters: preprocessed model parameters
        config: model configuration dict
        device: TTNN device
        init_enc_buf/init_dec_buf/init_out_buf: streaming buffers (optional)
        convnet_pre_ctx: convnet pre-processing context (optional)
        pad: whether to apply padding
        use_f0: whether to use F0-based mode (default: False = F0-free mode)
    Returns:
        output audio as TTNN tensor (and buffers if streaming)
    """
    L = config["L"]
    enc_dim = config["enc_dim"]
    out_buf_len = config["out_buf_len"]
    lookahead = config.get("lookahead", True)

    # Convert input TTNN -> torch (or use directly if already torch)
    if isinstance(x, torch.Tensor):
        x_torch = x
    else:
        x_torch = _from_device(x, device, batch_size=1)
        # TTNN tensors are bfloat16 but model weights are float32 — cast to match
        x_torch = x_torch.float()

    # F0 mode: label is zeros for F0-free (speaker embedding only).
    # F0-based mode would require pitch extraction here.
    if use_f0:
        raise NotImplementedError("F0-based mode not yet implemented for TTNN.")
    label = torch.zeros(x_torch.shape[0], 1)

    # Padding
    mod = 0
    if pad:
        pad_size = (L, L) if lookahead else (0, 0)
        x_torch, mod = _mod_pad(x_torch, chunk_size=L, pad=pad_size)

    if "convnet_pre" in parameters:
        raise NotImplementedError("convnet_pre is not supported in TTNN yet. Please disable it via config.")

    # ========================================================================
    # Input Conv1d encoder — runs on TT DEVICE when available
    # Conv1d(1, enc_dim, kernel_size=3*L, stride=L) + ReLU
    # ========================================================================
    in_conv_weight = parameters["in_conv"]["0"]["weight"]

    if device is not None and "in_conv_ttnn" in parameters:
        x_torch = _input_conv_device(x_torch, parameters["in_conv_ttnn"], device, L, config)
        logger.debug("Input Conv1d executed on TT device via ttnn.conv1d")
    else:
        x_torch = F.conv1d(x_torch, in_conv_weight, stride=L, padding=0)
        x_torch = torch.relu(x_torch)

    # ========================================================================
    # Label embedding — runs on TT DEVICE when available
    # Linear(1,512) -> LN(512) -> ReLU -> Linear(512,enc_dim) -> LN(enc_dim) -> ReLU
    # ========================================================================
    lp = parameters["label_embedding"]
    if device is not None:
        label_emb = _label_embedding_device(label, lp, enc_dim, device)
        logger.debug("Label embedding executed on TT device")
    else:
        label_emb = F.linear(label, lp["0"]["weight"], lp["0"]["bias"])
        label_emb = F.layer_norm(label_emb, [512], lp["1"]["weight"], lp["1"]["bias"])
        label_emb = torch.relu(label_emb)
        label_emb = F.linear(label_emb, lp["3"]["weight"], lp["3"]["bias"])
        label_emb = F.layer_norm(label_emb, [enc_dim], lp["4"]["weight"], lp["4"]["bias"])
        label_emb = torch.relu(label_emb)

    # ========================================================================
    # Encoder + Decoder + Mask
    # Encoder: grouped depthwise convolutions (always CPU)
    # Decoder: transformer attention (TT device when available)
    # ========================================================================
    encoder_config = _get_encoder_config(config)
    decoder_config = _get_decoder_config(config)

    if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:
        enc_buf = torch.zeros(
            x_torch.shape[0],
            enc_dim,
            (encoder_config["kernel_size"] - 1) * (2 ** encoder_config["num_layers"] - 1),
        )
        dec_buf = torch.zeros(
            x_torch.shape[0],
            decoder_config["num_layers"] + 1,
            decoder_config["ctx_len"],
            decoder_config["model_dim"],
        )
        out_buf = torch.zeros(x_torch.shape[0], enc_dim, out_buf_len)
    else:
        enc_buf = init_enc_buf if isinstance(init_enc_buf, torch.Tensor) else ttnn.to_torch(init_enc_buf)
        dec_buf = init_dec_buf if isinstance(init_dec_buf, torch.Tensor) else ttnn.to_torch(init_dec_buf)
        out_buf = init_out_buf if isinstance(init_out_buf, torch.Tensor) else ttnn.to_torch(init_out_buf)

    # Mask generation (core of LLVC) — decoder runs on TT device when available
    m, enc_buf, dec_buf = _mask_net(
        x_torch,
        label_emb,
        enc_buf,
        dec_buf,
        parameters["mask_gen"],
        encoder_config,
        decoder_config,
        skip_connection=config.get("skip_connection", True),
        proj=config.get("proj", True),
        device=device,
    )

    # Apply mask
    x_torch = x_torch * m

    # ========================================================================
    # Output buffer + ConvTranspose1d vocoder — runs on CPU
    # The ConvTranspose1d IS the vocoder/synthesis stage in LLVC.
    # It converts the masked latent representation back to waveform.
    # ========================================================================
    x_torch = torch.cat((out_buf, x_torch), dim=-1)
    out_buf = x_torch[..., -out_buf_len:]

    out_conv_weight = parameters["out_conv"]["0"]["weight"]
    x_torch = F.conv_transpose1d(x_torch, out_conv_weight, stride=L, padding=out_buf_len * L)
    x_torch = torch.tanh(x_torch)

    # Remove padding
    if mod != 0:
        x_torch = x_torch[:, :, :-mod]

    # Convert output to TTNN (or return torch tensor if no device)
    if device is not None:
        output = _to_device(x_torch, device)
    else:
        output = x_torch

    if init_enc_buf is None:
        return output
    else:
        return output, enc_buf, dec_buf, out_buf, convnet_pre_ctx


def preprocess_model_parameters(model, *, device=None):
    """
    Preprocess PyTorch model parameters for TTNN execution.

    Stage 3: When device is available, pre-uploads ALL weights to device
    (transformer decoder Q/K/V/out/FFN, label embedding, encoder pointwise convs)
    to eliminate per-inference host→device weight transfer overhead.
    """

    def _process_module(module):
        result = {}
        for name, child in module.named_children():
            result[name] = _process_module(child)

        for name, param in module.named_parameters(recurse=False):
            result[name] = param.data.clone().detach().float()

        for name, buf in module.named_buffers(recurse=False):
            result[name] = buf.data.clone().detach().float()

        return result

    parameters = _process_module(model)

    # Add nhead metadata for transformer decoder layers
    if "mask_gen" in parameters and "decoder" in parameters["mask_gen"]:
        dec_layers = parameters["mask_gen"]["decoder"].get("tf_dec_layers", {})
        for layer_key in dec_layers:
            dec_layers[layer_key]["nhead"] = 8  # LLVC default

    # Prepare TTNN device tensors when device is available
    if device is not None:
        # --- Input Conv1d weights ---
        conv_weight = parameters["in_conv"]["0"]["weight"]
        if _is_mesh_device(device):
            conv_weight_tt = ttnn.from_torch(
                conv_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        else:
            conv_weight_tt = ttnn.from_torch(
                conv_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        parameters["in_conv_ttnn"] = {"weight": conv_weight_tt}
        logger.info("Prepared input Conv1d weights for TT device (ttnn.conv1d)")

        # --- Pre-upload transformer decoder weights ---
        _preupload_transformer_weights(parameters, device)
        logger.info("Pre-uploaded transformer decoder weights to TT device")

        # --- Pre-upload label embedding weights ---
        _preupload_label_weights(parameters, device)
        logger.info("Pre-uploaded label embedding weights to TT device")

        # Note: Encoder pointwise conv pre-upload removed — encoder stays CPU-only
        # because per-layer host↔device overhead kills streaming performance.

    return parameters


def _upload_weight(w, device, transpose=True):
    """Upload a weight tensor to device in TILE_LAYOUT for ttnn.linear."""
    if transpose:
        w = w.T.contiguous()
    return _to_device(w, device)


def _upload_bias(b, device):
    """Upload a bias tensor to device, unsqueezed for ttnn.linear."""
    return _to_device(b.unsqueeze(0), device)


def _upload_ln(params, device):
    """Upload LayerNorm weight and bias to device."""
    return {
        "weight": _to_device(params["weight"].unsqueeze(0), device),
        "bias": _to_device(params["bias"].unsqueeze(0), device),
    }


def _preupload_transformer_weights(parameters, device):
    """Pre-upload all transformer decoder weights to device."""
    dec_layers = parameters["mask_gen"]["decoder"].get("tf_dec_layers", {})

    for layer_key, layer_params in dec_layers.items():
        embed_dim = layer_params["self_attn"]["in_proj_weight"].shape[1]
        ttnn_w = {}

        # Self-attention: split in_proj_weight into Q, K, V
        sa = layer_params["self_attn"]
        ipw = sa["in_proj_weight"]
        ipb = sa["in_proj_bias"]

        ttnn_w["sa_wq"] = _upload_weight(ipw[:embed_dim, :], device)
        ttnn_w["sa_bq"] = _upload_bias(ipb[:embed_dim], device)
        ttnn_w["sa_wk"] = _upload_weight(ipw[embed_dim : 2 * embed_dim, :], device)
        ttnn_w["sa_bk"] = _upload_bias(ipb[embed_dim : 2 * embed_dim], device)
        ttnn_w["sa_wv"] = _upload_weight(ipw[2 * embed_dim :, :], device)
        ttnn_w["sa_bv"] = _upload_bias(ipb[2 * embed_dim :], device)

        # Self-attention output projection
        ttnn_w["sa_out_w"] = _upload_weight(sa["out_proj"]["weight"], device)
        ttnn_w["sa_out_b"] = _upload_bias(sa["out_proj"]["bias"], device)

        # Layer norms
        ttnn_w["norm1"] = _upload_ln(layer_params["norm1"], device)
        ttnn_w["norm3"] = _upload_ln(layer_params["norm3"], device)

        # Cross-attention (if present)
        if "multihead_attn" in layer_params:
            ca = layer_params["multihead_attn"]
            ca_ipw = ca["in_proj_weight"]
            ca_ipb = ca["in_proj_bias"]

            ttnn_w["ca_wq"] = _upload_weight(ca_ipw[:embed_dim, :], device)
            ttnn_w["ca_bq"] = _upload_bias(ca_ipb[:embed_dim], device)
            ttnn_w["ca_wk"] = _upload_weight(ca_ipw[embed_dim : 2 * embed_dim, :], device)
            ttnn_w["ca_bk"] = _upload_bias(ca_ipb[embed_dim : 2 * embed_dim], device)
            ttnn_w["ca_wv"] = _upload_weight(ca_ipw[2 * embed_dim :, :], device)
            ttnn_w["ca_bv"] = _upload_bias(ca_ipb[2 * embed_dim :], device)
            ttnn_w["ca_out_w"] = _upload_weight(ca["out_proj"]["weight"], device)
            ttnn_w["ca_out_b"] = _upload_bias(ca["out_proj"]["bias"], device)
            ttnn_w["norm2"] = _upload_ln(layer_params["norm2"], device)

        # FFN
        ttnn_w["ff_w1"] = _upload_weight(layer_params["linear1"]["weight"], device)
        ttnn_w["ff_b1"] = _upload_bias(layer_params["linear1"]["bias"], device)
        ttnn_w["ff_w2"] = _upload_weight(layer_params["linear2"]["weight"], device)
        ttnn_w["ff_b2"] = _upload_bias(layer_params["linear2"]["bias"], device)

        layer_params["_ttnn"] = ttnn_w


def _preupload_label_weights(parameters, device):
    """Pre-upload label embedding weights to device."""
    lp = parameters["label_embedding"]
    lp["_ttnn"] = {
        "w0": _upload_weight(lp["0"]["weight"], device),
        "b0": _upload_bias(lp["0"]["bias"], device),
        "ln1": _upload_ln(lp["1"], device),
        "w3": _upload_weight(lp["3"]["weight"], device),
        "b3": _upload_bias(lp["3"]["bias"], device),
        "ln4": _upload_ln(lp["4"], device),
    }


def _preupload_encoder_pointwise_weights(parameters, device):
    """Pre-upload encoder pointwise (1x1) conv weights to device."""
    enc_params = parameters["mask_gen"]["encoder"]
    for i in range(8):  # 8 encoder layers
        key = f"dcc_{i}"
        if key not in enc_params.get("dcc_layers", {}):
            continue
        layers = enc_params["dcc_layers"][key]["layers"]
        # Pointwise conv weight: [out_ch, in_ch, 1] → squeeze to [out_ch, in_ch]
        pw_w = layers["3"]["weight"].squeeze(-1)  # Remove kernel dim
        pw_b = layers["3"].get("bias", None)
        ttnn_w = {
            "pw_w": _upload_weight(pw_w, device),
            "ln_pw": _upload_ln(layers["4"], device),
        }
        if pw_b is not None:
            ttnn_w["pw_b"] = _upload_bias(pw_b, device)
        # Also upload the depthwise LayerNorm
        ttnn_w["ln_dw"] = _upload_ln(layers["1"], device)
        layers["_ttnn"] = ttnn_w


def _get_encoder_config(config):
    """Build encoder config from model config."""
    num_layers = config.get("num_enc_layers", 8)
    kernel_size = 3
    buf_lengths = [(kernel_size - 1) * 2**i for i in range(num_layers)]
    buf_indices = [0]
    for i in range(num_layers - 1):
        buf_indices.append(buf_indices[-1] + buf_lengths[i])

    return {
        "num_layers": num_layers,
        "kernel_size": kernel_size,
        "buf_lengths": buf_lengths,
        "buf_indices": buf_indices,
    }


def _get_decoder_config(config):
    """Build decoder config from model config."""
    return {
        "ctx_len": config.get("dec_buf_len", 13),
        "chunk_size": config.get("dec_chunk_size", 13),
        "num_layers": config.get("num_dec_layers", 1),
        "model_dim": config.get("dec_dim", 256),
        "use_pos_enc": config.get("use_pos_enc", True),
    }


def init_buffers(batch_size, config):
    """
    Initialize streaming buffers for LLVC inference.

    Used when calling ttnn_llvc_forward in streaming (chunked) mode.
    Pass the returned buffers as init_enc_buf, init_dec_buf, init_out_buf.

    Args:
        batch_size: batch size
        config: TTNN config dict (from _get_ttnn_config or equivalent)
    Returns:
        (enc_buf, dec_buf, out_buf) - torch tensors
    """
    enc_dim = config["enc_dim"]
    num_enc_layers = config.get("num_enc_layers", 8)
    dec_dim = config.get("dec_dim", 256)
    dec_buf_len = config.get("dec_buf_len", 13)
    num_dec_layers = config.get("num_dec_layers", 1)
    out_buf_len = config["out_buf_len"]
    kernel_size = 3

    enc_buf = torch.zeros(
        batch_size,
        enc_dim,
        (kernel_size - 1) * (2**num_enc_layers - 1),
    )
    dec_buf = torch.zeros(
        batch_size,
        num_dec_layers + 1,
        dec_buf_len,
        dec_dim,
    )
    out_buf = torch.zeros(batch_size, enc_dim, out_buf_len)

    return enc_buf, dec_buf, out_buf
