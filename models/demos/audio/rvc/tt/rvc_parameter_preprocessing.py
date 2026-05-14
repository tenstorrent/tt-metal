# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Parameter preprocessing for RVC model on TTNN.

Handles conversion of PyTorch parameters to TTNN format,
device placement, and memory configuration.
"""

from typing import Optional

import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn


class RVCParameters:
    """Container for TTNN-processed RVC model parameters."""

    def __init__(self):
        self.device = None
        self.encoder = None
        self.flow = None
        self.vocoder = None
        self.rmvpe = None


class EncoderParameters:
    def __init__(self):
        self.pre_conv_weight = None
        self.pre_conv_bias = None
        self.enc_layers = []
        self.proj_m_weight = None
        self.proj_m_bias = None
        self.proj_logs_weight = None
        self.proj_logs_bias = None


class EncoderLayerParameters:
    def __init__(self):
        self.norm1 = None
        self.attn = None
        self.norm2 = None
        self.ffn = None


class AttentionParameters:
    def __init__(self):
        self.in_proj_weight = None
        self.in_proj_bias = None
        self.out_proj = None


class FFNParameters:
    def __init__(self):
        self.fc1 = None
        self.fc2 = None


class LinearParameters:
    def __init__(self):
        self.weight = None
        self.bias = None


class LayerNormParameters:
    def __init__(self):
        self.weight = None
        self.bias = None


class FlowParameters:
    def __init__(self):
        self.flows = []


class FlowLayerParameters:
    def __init__(self):
        self.weight = None
        self.bias = None


class VocoderParameters:
    def __init__(self):
        self.upsample_rates = []
        self.upsample_kernel_sizes = []
        self.conv_pre_weight = None
        self.conv_pre_bias = None
        self.ups = []
        self.resblocks = []
        self.conv_post_weight = None
        self.conv_post_bias = None


class RMVPEParameters:
    def __init__(self):
        self.convs = []
        self.f0_weight = None
        self.f0_bias = None


def preprocess_conv1d_params(conv, device=None, dtype=ttnn.bfloat16, to_device=False):
    """Convert a Conv1d module's parameters."""
    weight = conv.weight.data.clone()
    bias = conv.bias.data.clone() if conv.bias is not None else None
    result = type("ConvParams", (), {
        "weight": weight,
        "bias": bias,
        "kernel_size": conv.kernel_size[0],
        "padding": conv.padding[0],
    })()
    return result


def preprocess_linear_params(linear, device, dtype=ttnn.bfloat16):
    """Convert a Linear module's parameters to TTNN."""
    weight = preprocess_linear_weight(linear.weight, dtype=dtype)
    bias = preprocess_linear_bias(linear.bias, dtype=dtype) if linear.bias is not None else None

    weight = ttnn.from_torch(weight, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if bias is not None:
        bias = ttnn.from_torch(bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    params = LinearParameters()
    params.weight = weight
    params.bias = bias
    return params


def preprocess_rvc_model(
    model,
    device,
    mesh_mapper=None,
    dtype=ttnn.bfloat16,
):
    """
    Preprocess the full RVC PyTorch model into TTNN-compatible parameters.

    Args:
        model: RVCModel (PyTorch)
        device: TTNN device
        mesh_mapper: Optional mesh mapper for multi-device
        dtype: Target dtype

    Returns:
        RVCParameters object
    """
    logger.info("RVC: Preprocessing model parameters")

    params = RVCParameters()
    params.device = device

    # Encoder parameters
    params.encoder = _preprocess_encoder(model.encoder, device, dtype)
    logger.info("RVC: Encoder parameters preprocessed")

    # Flow decoder parameters (keep on CPU for torch conv1d)
    params.flow = _preprocess_flow(model.flow, dtype)
    logger.info("RVC: Flow decoder parameters preprocessed")

    # Vocoder parameters (keep on CPU for torch conv1d)
    params.vocoder = _preprocess_vocoder(model.vocoder, dtype)
    logger.info("RVC: Vocoder parameters preprocessed")

    # RMVPE parameters (keep on CPU)
    params.rmvpe = _preprocess_rmvpe(model.rmvpe, dtype)
    logger.info("RVC: RMVPE parameters preprocessed")

    return params


def _preprocess_encoder(encoder, device, dtype):
    """Preprocess posterior encoder parameters."""
    params = EncoderParameters()

    # Pre-conv (CPU for torch conv1d)
    params.pre_convs = [preprocess_conv1d_params(encoder.pre_conv)]
    params.pre_conv_weight = encoder.pre_conv.weight.data.clone()
    params.pre_conv_bias = encoder.pre_conv.bias.data.clone() if encoder.pre_conv.bias is not None else None

    # Encoder layers
    params.enc_layers = []
    for layer in encoder.enc_layers:
        layer_params = EncoderLayerParameters()

        # Norm1
        layer_params.norm1 = LayerNormParameters()
        layer_params.norm1.weight = layer.norm1.gamma.data.clone()
        layer_params.norm1.bias = layer.norm1.beta.data.clone()

        # Attention (to device for TTNN)
        layer_params.attn = AttentionParameters()
        # Combine q, k, v conv weights into single in_proj
        q_w = layer.attn.conv_q.weight.data.clone()
        k_w = layer.attn.conv_k.weight.data.clone()
        v_w = layer.attn.conv_v.weight.data.clone()
        in_proj_weight = torch.cat([q_w, k_w, v_w], dim=0)  # [3C, C, 1] → squeeze to [3C, C]

        q_b = layer.attn.conv_q.bias.data.clone()
        k_b = layer.attn.conv_k.bias.data.clone()
        v_b = layer.attn.conv_v.bias.data.clone()
        in_proj_bias = torch.cat([q_b, k_b, v_b], dim=0)

        layer_params.attn.in_proj_weight = torch.cat([
            q_w.squeeze(-1), k_w.squeeze(-1), v_w.squeeze(-1)
        ], dim=0)
        layer_params.attn.in_proj_bias = torch.cat([q_b, k_b, v_b], dim=0)

        layer_params.attn.out_proj = type("OutProjParams", (), {
            "weight": layer.attn.conv_o.weight.data.squeeze(-1).clone(),
            "bias": layer.attn.conv_o.bias.data.clone(),
        })()

        # Norm2
        layer_params.norm2 = LayerNormParameters()
        layer_params.norm2.weight = layer.norm2.gamma.data.clone()
        layer_params.norm2.bias = layer.norm2.beta.data.clone()

        # FFN
        layer_params.ffn = FFNParameters()
        layer_params.ffn.fc1 = type("FCParams", (), {
            "weight": layer.ffn.conv1.weight.data.squeeze(-1).clone(),
            "bias": layer.ffn.conv1.bias.data.clone(),
        })()
        layer_params.ffn.fc2 = type("FCParams", (), {
            "weight": layer.ffn.conv2.weight.data.squeeze(-1).clone(),
            "bias": layer.ffn.conv2.bias.data.clone(),
        })()

        params.enc_layers.append(layer_params)

    # Projection layers
    params.proj_m_weight = encoder.proj_m.weight.data.clone()
    params.proj_m_bias = encoder.proj_m.bias.data.clone()
    params.proj_logs_weight = encoder.proj_logs.weight.data.clone()
    params.proj_logs_bias = encoder.proj_logs.bias.data.clone()

    return params


def _preprocess_flow(flow, dtype):
    """Preprocess flow decoder parameters."""
    params = FlowParameters()
    params.flows = []

    for coupling_layer in flow.flows:
        flow_params = FlowLayerParameters()
        # Combine conv weights for the affine coupling
        w1 = coupling_layer.conv1.weight.data.clone()
        w2 = coupling_layer.conv2.weight.data.clone()
        w3 = coupling_layer.conv3.weight.data.clone()
        flow_params.weight = w3.data.clone()

        b1 = coupling_layer.conv1.bias.data.clone() if coupling_layer.conv1.bias is not None else None
        b2 = coupling_layer.conv2.bias.data.clone() if coupling_layer.conv2.bias is not None else None
        b3 = coupling_layer.conv3.bias.data.clone() if coupling_layer.conv3.bias is not None else None
        flow_params.bias = b3

        params.flows.append(flow_params)

    return params


def _preprocess_vocoder(vocoder, dtype):
    """Preprocess HiFi-GAN vocoder parameters."""
    params = VocoderParameters()
    params.upsample_rates = vocoder.upsample_rates
    params.upsample_kernel_sizes = []  # Will extract from actual conv params
    params.upsample_initial_channel = 512

    params.conv_pre_weight = vocoder.conv_pre.weight.data.clone()
    params.conv_pre_bias = vocoder.conv_pre.bias.data.clone() if vocoder.conv_pre.bias is not None else None

    params.ups = []
    for i, up in enumerate(vocoder.ups):
        up_params = type("UpParams", (), {
            "weight": up.weight.data.clone(),
            "bias": up.bias.data.clone() if up.bias is not None else None,
        })()
        params.ups.append(up_params)
        params.upsample_kernel_sizes.append(up.kernel_size[0])

    params.resblocks = []
    for i, resblock in enumerate(vocoder.resblocks):
        res_params = type("ResParams", (), {
            "kernel_sizes": [3, 7, 11],
            "dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "convs": resblock.convs,
        })()
        params.resblocks.append(res_params)

    params.conv_post_weight = vocoder.conv_post.weight.data.clone()
    params.conv_post_bias = vocoder.conv_post.bias.data.clone() if vocoder.conv_post.bias is not None else None

    return params


def _preprocess_rmvpe(rmvpe, dtype):
    """Preprocess RMVPE pitch extraction parameters."""
    params = RMVPEParameters()

    params.convs = []
    for conv in [rmvpe.conv1, rmvpe.conv2, rmvpe.conv3]:
        params.convs.append(type("ConvParams", (), {
            "weight": conv.weight.data.clone(),
            "bias": conv.bias.data.clone() if conv.bias is not None else None,
            "padding": conv.padding[0],
        })())

    params.f0_weight = rmvpe.f0_head.weight.data.clone()
    params.f0_bias = rmvpe.f0_head.bias.data.clone() if rmvpe.f0_head.bias is not None else None

    return params
