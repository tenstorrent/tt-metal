# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# Block C — #1 critical conv2d (pos=305 / SSA %275, IR line 1313 of
# ttnn_block_C_cylinder_backbone.mlir).  Standalone reconstruction of just this
# conv: prepare_conv2d_weights + prepare_conv2d_bias + ttnn.conv2d.
#
#   IC=192 -> OC=192, H=160 x W=288, k=3x3, s=1, pad=1, groups=1
#   weights bfp_bf8 | output bf16 | HiFi3 | fp32_dest_acc | relu6
#   config_tensors_in_dram=true | act_block_h_override=384 (-> 2 tiles)
#   enable_act_double_buffer=true | enable_weights_double_buffer=true
#   input: BLOCK_SHARDED L1, shard [6592, 32] on cores (0,0)-(5,6) = 42 cores
#
# Reference bottleneck row (block_C_conv2d_bottleneck_analysis.md, pos=305):
#   DEVICE KERNEL DURATION = 2394.3 us   PM IDEAL = 349.9 us
#   Actual / PM Ideal      = 6.84x       PM FPU UTIL = 14.61%   -> HIGH (compute
#   underutilized: abh=2 tiles too small over 206 tiles/core -> 103 CB-stall blocks)
#
# Profile:
#   TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' \
#     python -m tracy -r -p -m pytest \
#     tests/ttnn/unit_tests/operations/conv/test_block_C_conv2d_pos305.py
# then read DEVICE KERNEL DURATION / PM IDEAL / PM FPU UTIL from the ops CSV.
# =============================================================================

import pytest
import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT

# Op geometry (matches pos=305 / %275).
N, C_IN, H, W = 1, 192, 160, 288
C_OUT, KH, KW = 192, 3, 3
NHW = N * H * W  # 46080


def _block_sharded_input_cfg():
    # IR: #ttnn.memory_config<#l1, <block_sharded>,
    #       shard_spec<[core_range<(0,0),(5,6)>], <6592x32>, row_major>>
    # 6 cols x 7 rows = 42 cores; per-core shard [6592, 32] (206 out-tiles high x 1 tile wide).
    shard = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 6))}),
        [6592, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)


def _conv_config():
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,  # bfp_bf8
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        deallocate_activation=True,
        act_block_h_override=384,
        config_tensors_in_dram=True,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        enable_kernel_stride_folding=False,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        output_layout=TILE,
    )


def test_block_C_conv2d_pos305(device):
    """Reproduce Block C's #1 bottleneck conv (pos=305, %275) standalone."""
    torch.manual_seed(0)
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    conv_cfg = _conv_config()
    in_mem = _block_sharded_input_cfg()

    w = torch.randn(C_OUT, C_IN, KH, KW, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 1, C_OUT, dtype=torch.bfloat16)

    common = dict(
        input_memory_config=in_mem,
        input_layout=TILE,
        in_channels=C_IN,
        out_channels=C_OUT,
        batch_size=N,
        input_height=H,
        input_width=W,
        kernel_size=(KH, KW),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # prepare_conv2d_weights (bfp_bf8) + prepare_conv2d_bias — both on the block-sharded input.
    tt_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=RM),
        weights_format="OIHW",
        has_bias=True,
        **common,
    )
    tt_b = ttnn.prepare_conv_bias(bias_tensor=ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=RM), **common)

    # activation [1,1,46080,192]; deallocate_activation=True frees it each call, so
    # re-upload + reshard to the block-sharded L1 input per iteration.
    x = torch.randn(1, 1, NHW, C_IN, dtype=torch.bfloat16)
    out = None
    for _ in range(5):
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=DRAM)
        tt_x = ttnn.to_memory_config(tt_x, in_mem)
        out = ttnn.conv2d(
            input_tensor=tt_x,
            weight_tensor=tt_w,
            bias_tensor=tt_b,
            device=device,
            in_channels=C_IN,
            out_channels=C_OUT,
            batch_size=N,
            input_height=H,
            input_width=W,
            kernel_size=(KH, KW),
            stride=(1, 1),
            padding=(1, 1, 1, 1),
            dilation=(1, 1),
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=conv_cfg,
            compute_config=compute,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
    ttnn.synchronize_device(device)
    assert list(out.shape) == [1, 1, NHW, C_OUT], f"unexpected output shape {list(out.shape)}"
