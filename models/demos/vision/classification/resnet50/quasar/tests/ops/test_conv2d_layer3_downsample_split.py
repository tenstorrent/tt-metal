# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Validate the BLOCK_SHARDED 1x1 stride-2 downsample on the Quasar SPLIT plain-matmul route.

WHY THIS SHAPE
--------------
The resnet50 layer3_module1 downsample (in=512, out=1024, 1x1, stride 2, input 28x28) is the ONE downsample
the model forces BLOCK_SHARDED: HEIGHT_SHARDED N-halves the stride-2 1x1 channel-expansion (device out = 512
not 1024), and @28 it cannot go height-sharded. On the FUSED conv_bmm path a block-sharded 1x1 splits K
across grid columns (in0_num_blocks_w>1 -> nbw2) and does the multi-K-block matmul-partials accumulate, which
DEADLOCKS in the kb0->kb1 transition / kb1 accumulate-pack (tt-metal #48679 / tt-llk #48504). It is the last
host-bypassed conv in the e2e (a single global _CONV_ON_DEVICE["downsample"] flag).

WHAT THIS VALIDATES
-------------------
The [#48552 Stage2] extension of `force_1x1_nonmm_split` (conv2d.cpp) to BLOCK_SHARDED 1x1: forcing
act_block_w = full_K makes it a SINGLE K-block (in0_num_blocks_w == 1), so the conv takes the SPLIT path
(Program A gather+tilize -> Program B quasar matmul::linear, a single-K-block 2D-mcast GEMM with the N-split
in its 2D config). That dodges BOTH the fused multi-K accumulate hang AND the HEIGHT_SHARDED N-halving. If
this passes (all 1024 out channels, PCC ~1.0), the layer3_module1 downsample has a working device path and
the last host bypass can be removed (per-downsample or global flip).

RUN (emulator, forced JIT; SPLIT env set inside the test):
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer3_downsample_split.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_layer3_module1_downsample_block_split(mesh_device, monkeypatch):
    """layer3_module1 downsample: 1x1, in=512, out=1024, stride 2, input 28x28, BLOCK_SHARDED via the SPLIT
    plain-matmul route. Must keep all 1024 out channels and match a torch golden (PCC ~0.99)."""
    if is_wormhole_b0():
        pytest.skip("Quasar-only: exercises the force_1x1_nonmm_split BLOCK_SHARDED extension (arch_is_quasar).")

    device = mesh_device
    torch.manual_seed(0)

    # Required: force_1x1_nonmm_split only fires under the split env.
    monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")  # monkeypatch auto-restores after the test

    batch = 1
    in_ch = 512
    out_ch = 1024
    input_height = input_width = 28  # layer3_module1 downsample input (layer2 output spatial)
    kernel_size = (1, 1)
    stride = (2, 2)
    padding = (0, 0)

    torch_input = torch.randn((batch, in_ch, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_ch, in_ch, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((out_ch,), dtype=torch.bfloat16).float()
    torch_golden = torch.nn.functional.conv2d(
        torch_input, torch_weight, bias=torch_bias, stride=stride, padding=padding
    )  # no activation on the downsample (RELU is fused in the residual add_)

    nhw = batch * input_height * input_width
    flat = torch.permute(torch_input, (0, 2, 3, 1)).reshape(1, 1, nhw, in_ch).contiguous()
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = max(c for c in range(1, max_cores + 1) if nhw % c == 0)
    shard_h = nhw // num_cores
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, in_ch),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(device, in_mem)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias.reshape(1, 1, 1, out_ch), dtype=ttnn.bfloat16)

    # Verbatim from run_downsample_if_req for input_height == 28 (the BLOCK_SHARDED branch).
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        deallocate_activation=True,
        reallocate_halo_output=True,
        act_block_h_override=32,
        reshard_if_not_optimal=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    out, [oh, ow], _wb = ttnn.experimental.quasar.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=in_ch,
        out_channels=out_ch,
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1),
        groups=1,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    # The split route must keep all 1024 output channels (the N-halving bug drops half).
    assert tt_out.shape[-1] >= out_ch, f"downsample dropped channels: got last-dim {tt_out.shape[-1]}, want >= {out_ch}"
    tt_out = tt_out.reshape(batch, oh, ow, tt_out.shape[-1])[:, :, :, :out_ch]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))  # NHWC -> NCHW
    assert_with_pcc(torch_golden, tt_out.float(), 0.99)
