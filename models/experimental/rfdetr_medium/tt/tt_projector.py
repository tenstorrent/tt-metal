# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN MultiScaleProjector for RF-DETR Medium.
Pure TTNN — all conv2d, layer_norm, activation on device.

Input:  4 × [B, 384, 36, 36] NCHW (from backbone)
Output: 1 × [B, 256, 36, 36] NCHW (P4)

Architecture for scale_factors=[1.0], in_channels=[384]*4:
  1. scale=1.0 → no sampling (pass-through)
  2. Concat all 4 features along channel: [B, 1536, 36, 36]
  3. C2f(1536, 256, n=3, layer_norm=True):
     - cv1: Conv1x1(1536 → 256) + LN + SiLU
     - Split → 2 × [B, 128, 36, 36]
     - 3 × Bottleneck(128 → 128, shortcut=False): Conv3x3+LN+SiLU → Conv3x3+LN+SiLU (no residual)
     - Concat all → [B, 640, 36, 36]  (128*5 = 640)
     - cv2: Conv1x1(640 → 256) + LN + SiLU
  4. LayerNorm(256)

Conv2d patterns from: models/experimental/vadv2/tt/common.py (TtConv2D)
"""

import ttnn

from models.experimental.rfdetr_medium.common import (
    HIDDEN_DIM,
    VIT_HIDDEN_SIZE,
    NUM_PATCHES_PER_SIDE,
)


def _make_conv_config(device, weights_dtype=ttnn.bfloat16, activation=None):
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=False,
        reshard_if_not_optimal=True,
        activation=activation,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        math_approx_mode=False,
    )
    return conv_config, compute_config


def conv2d_ln_silu(
    x,
    weight,
    bias,
    ln_weight,
    ln_bias,
    in_channels,
    out_channels,
    kernel_size,
    batch_size,
    input_height,
    input_width,
    device,
    conv_config,
    compute_config,
    stride=1,
    padding=0,
    groups=1,
):
    """
    ConvX block: Conv2d → LayerNorm → SiLU. All on device.

    Conv2d uses NHWC format.
    LayerNorm applied in NHWC (channel-wise).
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    [x, [out_h, out_w], [weight, bias]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )

    # Conv2d output is flattened NHWC [1, 1, B*H*W, C], may be sharded
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, (batch_size, out_h, out_w, out_channels))

    # LayerNorm in NHWC (last dim = channels)
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.layer_norm(x, weight=ln_weight, bias=ln_bias, epsilon=1e-6)

    # SiLU activation
    x = ttnn.silu(x)

    return x, out_h, out_w, weight, bias


def projector_forward(feature_maps, projector_params, batch_size, device):
    """
    MultiScaleProjector forward. Pure TTNN.

    Args:
        feature_maps: list of 4 TTNN tensors [B, 384, 36, 36] (NCHW, on device)
        projector_params: dict from load_projector_weights()
        batch_size: batch size
        device: TTNN device

    Returns:
        list of 1 TTNN tensor [B, 256, 36, 36] (NCHW, on device)
    """
    conv_config, compute_config = _make_conv_config(device)

    # scale=1.0 → pass-through (no sampling needed)
    # Concat along channels: 4 × [B, 384, 36, 36] → [B, 1536, 36, 36]
    fused = ttnn.concat(feature_maps, dim=1)

    # Convert NCHW → NHWC for conv2d: [B, 1536, 36, 36] → [B, 36, 36, 1536]
    fused = ttnn.to_layout(fused, layout=ttnn.ROW_MAJOR_LAYOUT)
    fused = ttnn.permute(fused, (0, 2, 3, 1))

    h, w = NUM_PATCHES_PER_SIDE, NUM_PATCHES_PER_SIDE
    in_channels = VIT_HIDDEN_SIZE * 4  # 1536

    # ---- C2f block ----
    p = projector_params

    # cv1: Conv1x1(1536 → 256) + LN + SiLU
    # Output 2*c = 256, where c = 128
    c2f_c = HIDDEN_DIM // 2  # 128
    cv1_out_ch = 2 * c2f_c  # 256

    x, h, w, p["cv1_weight"], p["cv1_bias_conv"] = conv2d_ln_silu(
        fused,
        p["cv1_weight"],
        p["cv1_bias_conv"],
        p["cv1_ln_weight"],
        p["cv1_ln_bias"],
        in_channels,
        cv1_out_ch,
        kernel_size=1,
        batch_size=batch_size,
        input_height=h,
        input_width=w,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
    )
    ttnn.deallocate(fused)

    # Split into 2 chunks of 128: y[0], y[1]
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.permute(x, (0, 3, 1, 2))  # NHWC → NCHW for split
    y0, y1 = ttnn.split(x, c2f_c, dim=1)  # split_size=128 → 2 chunks of [B, 128, 36, 36]
    ttnn.deallocate(x)

    chunks = [y0, y1]  # ROW_MAJOR NCHW; concat in ROW_MAJOR to avoid tile-padding corruption

    # 3 × Bottleneck(128 → 128, shortcut=False, no residual)
    last = y1  # ROW_MAJOR NCHW
    for bn_idx in range(3):
        bn = p["bottlenecks"][bn_idx]

        # Convert to NHWC for conv2d
        inp = ttnn.to_layout(last, layout=ttnn.ROW_MAJOR_LAYOUT)
        inp = ttnn.permute(inp, (0, 2, 3, 1))  # [B, H, W, 128]

        # Bottleneck cv1: Conv3x3(128 → 128) + LN + SiLU
        out, bh, bw, bn["cv1_weight"], bn["cv1_bias_conv"] = conv2d_ln_silu(
            inp,
            bn["cv1_weight"],
            bn["cv1_bias_conv"],
            bn["cv1_ln_weight"],
            bn["cv1_ln_bias"],
            c2f_c,
            c2f_c,
            kernel_size=3,
            padding=1,
            batch_size=batch_size,
            input_height=h,
            input_width=w,
            device=device,
            conv_config=conv_config,
            compute_config=compute_config,
        )

        # Bottleneck cv2: Conv3x3(128 → 128) + LN + SiLU
        out, bh, bw, bn["cv2_weight"], bn["cv2_bias_conv"] = conv2d_ln_silu(
            out,
            bn["cv2_weight"],
            bn["cv2_bias_conv"],
            bn["cv2_ln_weight"],
            bn["cv2_ln_bias"],
            c2f_c,
            c2f_c,
            kernel_size=3,
            padding=1,
            batch_size=batch_size,
            input_height=bh,
            input_width=bw,
            device=device,
            conv_config=conv_config,
            compute_config=compute_config,
        )

        # No shortcut (C2f uses shortcut=False): permute back to NCHW
        out = ttnn.to_layout(out, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.permute(out, (0, 3, 1, 2))  # NHWC → NCHW
        chunks.append(out)
        last = out

    # Concat all chunks in ROW_MAJOR: (2 + 3) * 128 = 640 channels
    cat = ttnn.concat(chunks, dim=1)  # [B, 640, 36, 36] NCHW ROW_MAJOR

    # Convert to NHWC for cv2 conv
    cat_nhwc = ttnn.permute(cat, (0, 2, 3, 1))  # NHWC
    ttnn.deallocate(cat)

    cv2_in_ch = (2 + 3) * c2f_c  # 640
    result, rh, rw, p["cv2_weight"], p["cv2_bias_conv"] = conv2d_ln_silu(
        cat_nhwc,
        p["cv2_weight"],
        p["cv2_bias_conv"],
        p["cv2_ln_weight"],
        p["cv2_ln_bias"],
        cv2_in_ch,
        HIDDEN_DIM,
        kernel_size=1,
        batch_size=batch_size,
        input_height=h,
        input_width=w,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
    )
    ttnn.deallocate(cat_nhwc)

    # Final LayerNorm
    result = ttnn.layer_norm(result, weight=p["final_ln_weight"], bias=p["final_ln_bias"], epsilon=1e-6)

    # Convert NHWC → NCHW: [B, 36, 36, 256] → [B, 256, 36, 36]
    result = ttnn.to_layout(result, layout=ttnn.ROW_MAJOR_LAYOUT)
    result = ttnn.permute(result, (0, 3, 1, 2))

    return [result]
