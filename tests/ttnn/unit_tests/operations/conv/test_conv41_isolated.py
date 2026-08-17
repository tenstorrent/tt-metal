# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Isolated reproduction of the %41 1x1 conv2d (in=24, out=64, HxW=320x576)
# with NO upstream ops — fresh device, clean allocator.
# Determines whether the conv clashes intrinsically or only due to history.

import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT


def test_conv41_isolated(device):
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    cfg = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        deallocate_activation=True,
        act_block_h_override=0,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
        output_layout=TILE,
    )

    w = torch.randn(64, 24, 1, 1, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 1, 64, dtype=torch.bfloat16)

    tt_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=24,
        out_channels=64,
        batch_size=1,
        input_height=320,
        input_width=576,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_b = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        in_channels=24,
        out_channels=64,
        batch_size=1,
        input_height=320,
        input_width=576,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # input [1,1,184320,24] TILE DRAM interleaved
    x = ttnn.from_torch(
        torch.randn(1, 1, 184320, 24, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=TILE,
        device=device,
        memory_config=DRAM,
    )

    print("=== L1 STATE (isolated, fresh device) ===")
    ttnn.dump_device_memory_state(device, prefix="isolated_")

    out = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=tt_w,
        bias_tensor=tt_b,
        in_channels=24,
        out_channels=64,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=1,
        input_height=320,
        input_width=576,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    print(f"isolated conv OK: {out.shape}  {out.memory_config()}")
    ttnn.deallocate(out)
