# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN BasicBlock implementation for DiffusionDrive ResNet-34 backbone.

Implements a single ResNet-34 BasicBlock (conv3x3 + BN + ReLU + conv3x3 + BN +
optional 1x1 downsample + residual add + ReLU) using TTNN conv2d with BN-folded
weights.

Public classes:
    TtnnBasicBlock — drop-in TTNN replacement for a single nn.Sequential BasicBlock.

Helper functions:
    prepare_basic_block_params(conv1, bn1, conv2, bn2, downsample=None)
        → dict suitable for TtnnBasicBlock constructor

Notes:
    - BN is folded into Conv at preprocessor time (fp32 fold, cast to bfloat16).
    - Weights are passed as raw bfloat16 torch tensors; ttnn.conv2d handles layout.
    - conv2d uses the auto shard layout; a following ReLU is fused into the conv
      (see ``_RELU_ACT``) rather than run as a separate ttnn.relu op.
    - This helper module backs every conv in the model (backbone BasicBlocks,
      stems, FPN, perception 1x1, grid-sample value_proj); each of those runs on
      TTNN — nothing here remains a PyTorch fallback.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import ttnn
from models.demos.diffusion_drive.tt.common import fold_bn

# Fused conv activation for the Stage 2 "relu with conv" optimisation: a single
# shared, read-only descriptor passed into Conv2dConfig so a conv writes its
# ReLU'd output directly, removing a standalone ttnn.relu op per conv+relu pair.
# Mirrors the in-tree ufld_v2 ResNet-34 (models/demos/vision/segmentation/ufld_v2).
_RELU_ACT = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------


def prepare_basic_block_params(
    conv1: nn.Conv2d,
    bn1: nn.BatchNorm2d,
    conv2: nn.Conv2d,
    bn2: nn.BatchNorm2d,
    downsample: Optional[nn.Sequential] = None,
) -> Dict[str, torch.Tensor]:
    """Fold BN into each Conv2d and return a parameter dict.

    Returns a dict with keys "conv1", "conv2", and optionally "downsample",
    each containing {"weight": bfloat16 torch tensor, "bias": bfloat16 torch tensor}.
    """
    w1, b1 = fold_bn(conv1, bn1)
    w2, b2 = fold_bn(conv2, bn2)

    params: Dict = {
        "conv1": {"weight": w1, "bias": b1},
        "conv2": {"weight": w2, "bias": b2},
    }

    if downsample is not None:
        # timm ResNet downsample is nn.Sequential([Conv2d, BatchNorm2d])
        ds_conv, ds_bn = downsample[0], downsample[1]
        w_ds, b_ds = fold_bn(ds_conv, ds_bn)
        params["downsample"] = {"weight": w_ds, "bias": b_ds}

    return params


def prepare_resnet34_stage_params(layer: nn.Sequential) -> list:
    """Fold BN for every BasicBlock in one timm ResNet-34 stage.

    Args:
        layer: nn.Sequential of timm BasicBlock objects (e.g. model.layer1).

    Returns:
        List of (stride: int, params: dict) tuples, one per block.
        ``stride`` is the block's stride (1 or 2); ``params`` is suitable
        for TtnnBasicBlock's constructor.
    """
    result = []
    for block in layer:
        params = prepare_basic_block_params(
            block.conv1,
            block.bn1,
            block.conv2,
            block.bn2,
            downsample=block.downsample,
        )
        result.append((int(block.stride), params))
    return result


# ---------------------------------------------------------------------------
# TTNN BasicBlock
# ---------------------------------------------------------------------------


def _make_conv_config(activation=None) -> ttnn.Conv2dConfig:
    """Conv2d config: bfloat16 weights, auto shard layout.

    ``activation`` is a ``ttnn.UnaryWithParam`` (e.g. ``_RELU_ACT``) or ``None``.
    When set, the conv fuses that activation into its output writeback, removing a
    standalone elementwise op for every conv that feeds a ReLU.

    Note: older binaries do not accept a 'dtype' kwarg in Conv2dConfig; the
    output dtype is controlled via the 'dtype' kwarg of ttnn.conv2d itself.
    """
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        deallocate_activation=False,
        reallocate_halo_output=True,
        reshard_if_not_optimal=False,
        shard_layout=None,  # auto-selected (interleaved output at these sizes)
        activation=activation,
    )


def prep_conv_weights(weight: torch.Tensor, bias: torch.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Pre-convert a torch weight/bias pair to TTNN host tensors (do once at build).

    Previously this conversion happened inside every ``ttnn.conv2d`` call, so the
    weights were re-tilized on the host on every forward pass.  Hoisting it into
    the block/FPN constructors removes that per-forward cost.
    """
    w_ttnn = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
    b_ttnn = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)
    return w_ttnn, b_ttnn


def _ttnn_conv2d(
    device: ttnn.Device,
    x: ttnn.Tensor,
    w_ttnn: ttnn.Tensor,
    b_ttnn: ttnn.Tensor,
    B: int,
    H: int,
    W: int,
    C_in: int,
    C_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
    activation=None,
) -> Tuple[ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor]:
    """Run ttnn.conv2d and return (output, out_H, out_W, prepared_w, prepared_b).

    ``w_ttnn`` / ``b_ttnn`` may be host tensors (first call) or the device-prepared
    tensors returned by a previous call. Returning the prepared (device-resident)
    weights lets the caller cache them and skip the per-forward host upload — which
    is both a small perf win and **required for trace capture** (host->device writes
    are illegal between begin/end_trace_capture).

    ``activation`` (a ``ttnn.UnaryWithParam`` such as ``_RELU_ACT``, or ``None``)
    is fused into the conv, so a caller that passes it must not run a separate
    ttnn.relu on the result.
    """
    conv_config = _make_conv_config(activation)
    [out, [out_H, out_W], [prep_w, prep_b]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=w_ttnn,
        bias_tensor=b_ttnn,
        in_channels=C_in,
        out_channels=C_out,
        device=device,
        kernel_size=[kernel_size, kernel_size],
        stride=[stride, stride],
        padding=[padding, padding, padding, padding],
        dilation=[1, 1],
        batch_size=B,
        input_height=H,
        input_width=W,
        conv_config=conv_config,
        return_weights_and_bias=True,
        return_output_dim=True,
    )
    return out, out_H, out_W, prep_w, prep_b


class TtnnBasicBlock:
    """
    TTNN implementation of a single ResNet-34 BasicBlock.

    conv3×3 + BN-fold → ReLU
    conv3×3 + BN-fold
    optional shortcut (1×1 stride-2 + BN-fold)
    residual add → ReLU

    Usage::

        params = prepare_basic_block_params(conv1, bn1, conv2, bn2, downsample)
        block = TtnnBasicBlock(params, stride=block.stride[0], device=device)
        x_ttnn, (B, H_out, W_out, C_out) = block(x_ttnn, (B, H, W, C_in))
    """

    def __init__(
        self,
        params: Dict,
        stride: int,
        device: ttnn.Device,
    ) -> None:
        self._stride = stride
        self._device = device
        self._has_downsample = "downsample" in params

        # Pre-convert weights to TTNN host tensors once (was per-forward before).
        self._C_mid = int(params["conv1"]["weight"].shape[0])
        self._C_out = int(params["conv2"]["weight"].shape[0])
        self._w1, self._b1 = prep_conv_weights(params["conv1"]["weight"], params["conv1"]["bias"])
        self._w2, self._b2 = prep_conv_weights(params["conv2"]["weight"], params["conv2"]["bias"])
        if self._has_downsample:
            self._w_ds, self._b_ds = prep_conv_weights(params["downsample"]["weight"], params["downsample"]["bias"])

    def __call__(
        self,
        x: ttnn.Tensor,
        shape: Tuple[int, int, int, int],  # (B, H, W, C_in)
    ) -> Tuple[ttnn.Tensor, Tuple[int, int, int, int]]:
        B, H, W, C_in = shape

        # Conv1: 3×3, stride=self._stride, BN-folded + ReLU
        C_mid = self._C_mid
        identity = x

        # Reassign self._w1/_b1 to the device-prepared weights conv2d returns, so the
        # next forward reuses them instead of re-uploading host weights (trace-safe).
        out, H1, W1, self._w1, self._b1 = _ttnn_conv2d(
            self._device,
            x,
            self._w1,
            self._b1,
            B,
            H,
            W,
            C_in,
            C_mid,
            kernel_size=3,
            stride=self._stride,
            padding=1,
            activation=_RELU_ACT,  # ReLU fused into conv1
        )
        # Move to interleaved DRAM + TILE for reliable add (ReLU already applied by the conv).
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG) if out.is_sharded() else out

        # Conv2: 3×3, stride=1, BN-folded (no activation)
        C_out = self._C_out

        out, H2, W2, self._w2, self._b2 = _ttnn_conv2d(
            self._device, out, self._w2, self._b2, B, H1, W1, C_mid, C_out, kernel_size=3, stride=1, padding=1
        )
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG) if out.is_sharded() else out

        # Shortcut
        if self._has_downsample:
            identity, _, _, self._w_ds, self._b_ds = _ttnn_conv2d(
                self._device,
                identity,
                self._w_ds,
                self._b_ds,
                B,
                H,
                W,
                C_in,
                C_out,
                kernel_size=1,
                stride=self._stride,
                padding=0,
            )
            identity = (
                ttnn.sharded_to_interleaved(identity, ttnn.DRAM_MEMORY_CONFIG) if identity.is_sharded() else identity
            )

        # Ensure both tensors are in TILE_LAYOUT for add
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        if identity.layout != ttnn.TILE_LAYOUT:
            identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)

        out = ttnn.add(out, identity)
        out = ttnn.relu(out)

        return out, (B, H2, W2, C_out)
