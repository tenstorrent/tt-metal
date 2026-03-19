# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FPN neck (Sam3DualViTDetNeck) for SAM3 on ttnn.

All convolution operations run on CPU via PyTorch for correctness.
Results are converted to ttnn tensors at the end.

# TODO: Replace CPU conv/deconv ops with ttnn equivalents for on-device execution.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import ttnn


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------


def preprocess_neck_weights(neck_module) -> Dict:
    """Extract conv weights and biases from a Sam3DualViTDetNeck module.

    Args:
        neck_module: Sam3DualViTDetNeck PyTorch module.

    Returns:
        dict with key 'convs', a list of per-scale dicts.  Each per-scale dict
        contains the named layer weights extracted from the nn.Sequential conv
        pipeline for that scale.
    """
    params = {"convs": []}
    for seq in neck_module.convs:
        scale_params = {}
        for name, module in seq.named_modules():
            if name == "":
                # Skip the Sequential container itself
                continue
            mtype = type(module).__name__
            if mtype in ("ConvTranspose2d", "Conv2d"):
                scale_params[name] = {
                    "weight": module.weight.data.clone(),
                    "bias": module.bias.data.clone() if module.bias is not None else None,
                    "type": mtype,
                    "stride": module.stride,
                    "padding": module.padding,
                    "kernel_size": module.kernel_size,
                }
            elif mtype == "MaxPool2d":
                scale_params[name] = {
                    "type": mtype,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                }
            elif mtype == "GELU":
                scale_params[name] = {"type": "GELU"}
        params["convs"].append(scale_params)
    return params


# ---------------------------------------------------------------------------
# Position encoding
# ---------------------------------------------------------------------------


def generate_position_encodings(
    feature_shapes: List[Tuple[int, int]],
    num_pos_feats: int = 256,
    temperature: int = 10000,
    normalize: bool = True,
    scale: Optional[float] = None,
) -> List[torch.Tensor]:
    """Generate sinusoidal 2-D position encodings for a list of spatial sizes.

    Implements the same algorithm as sam3.model.position_encoding.PositionEmbeddingSine.

    Args:
        feature_shapes: list of (H, W) tuples, one per FPN scale.
        num_pos_feats: total number of positional feature channels (must be even).
        temperature: temperature for sinusoidal encoding.
        normalize: whether to normalize the position grid to [0, 2*pi].
        scale: scale factor applied after normalization; defaults to 2*pi.

    Returns:
        List of torch.Tensor, each with shape (1, num_pos_feats, H, W).
    """
    assert num_pos_feats % 2 == 0, "num_pos_feats must be even"
    half_feats = num_pos_feats // 2
    if scale is None:
        scale = 2 * math.pi

    encodings = []
    for h, w in feature_shapes:
        # Build y and x grids of shape (1, H, W)
        y_embed = (
            torch.arange(1, h + 1, dtype=torch.float32)
            .view(1, -1, 1)
            .repeat(1, 1, w)
        )
        x_embed = (
            torch.arange(1, w + 1, dtype=torch.float32)
            .view(1, 1, -1)
            .repeat(1, h, 1)
        )

        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(half_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / half_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (1, H, W, half_feats)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)  # (1, H, W, half_feats)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)
        encodings.append(pos)

    return encodings


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------


def _apply_conv_sequential(x: torch.Tensor, scale_params: Dict) -> torch.Tensor:
    """Apply all operations in a single FPN scale's conv sequential on CPU.

    Args:
        x: (B, C, H, W) input tensor.
        scale_params: dict of named layers extracted by preprocess_neck_weights.

    Returns:
        Output tensor after applying all layers.
    """
    # Process layers in the order they appear in the dict (insertion-ordered, Python 3.7+)
    for name, lp in scale_params.items():
        ltype = lp["type"]
        if ltype == "ConvTranspose2d":
            x = F.conv_transpose2d(
                x,
                lp["weight"],
                bias=lp["bias"],
                stride=lp["stride"],
                padding=lp["padding"],
            )
        elif ltype == "Conv2d":
            x = F.conv2d(
                x,
                lp["weight"],
                bias=lp["bias"],
                stride=lp["stride"],
                padding=lp["padding"],
            )
        elif ltype == "MaxPool2d":
            x = F.max_pool2d(x, kernel_size=lp["kernel_size"], stride=lp["stride"])
        elif ltype == "GELU":
            x = F.gelu(x)
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tt_fpn_neck(
    vit_features: torch.Tensor,
    neck_params: Dict,
    device,
) -> Dict:
    """Run the SAM3 FPN neck on pre-computed ViT backbone features.

    All convolution operations run on CPU via PyTorch.  The output feature maps
    and position encodings are returned as lists of ttnn tensors.

    Args:
        vit_features: (B, 1024, 72, 72) torch tensor in NCHW format – the last
            output of the ViT backbone (xs[-1] in the reference forward pass).
        neck_params: dict returned by preprocess_neck_weights().
        device: ttnn device.

    Returns:
        dict with:
            'backbone_fpn'  – list of ttnn tensors, one per FPN scale,
                              each (B, 256, H_i, W_i) in TILE_LAYOUT.
            'vision_pos_enc' – list of ttnn tensors, one per FPN scale,
                               each (B, 256, H_i, W_i) in TILE_LAYOUT.
    """
    x = vit_features  # (B, 1024, 72, 72) on CPU

    fpn_out = []
    feature_shapes = []

    with torch.no_grad():
        for scale_params in neck_params["convs"]:
            feat = _apply_conv_sequential(x, scale_params)
            fpn_out.append(feat)
            feature_shapes.append((feat.shape[2], feat.shape[3]))

    # Generate position encodings matching each output spatial size
    pos_encs = generate_position_encodings(feature_shapes)

    # Convert everything to ttnn tensors on device
    tt_fpn = []
    for feat in fpn_out:
        # feat is (B, C, H, W); convert to ttnn in ROW_MAJOR first, then TILE
        tt_feat = ttnn.from_torch(
            feat.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        tt_fpn.append(tt_feat)

    tt_pos = []
    for pos in pos_encs:
        tt_p = ttnn.from_torch(
            pos.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        tt_pos.append(tt_p)

    return {
        "backbone_fpn": tt_fpn,
        "vision_pos_enc": tt_pos,
    }
