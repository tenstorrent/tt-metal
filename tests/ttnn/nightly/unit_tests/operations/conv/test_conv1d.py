# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
@pytest.mark.parametrize(
    "channels, path",
    (
        # 512*2*7 stays under the NoC burst limit -> coalesced read path.
        (512, "coalesced"),
        # 1280*2*7 exceeds the WH (8192B) and BH (16384B) burst limits -> non-coalesced read path.
        (1280, "non-coalesced"),
    ),
)
def test_conv1d_depthwise_multi_height_block(device, channels, path):
    """Exercises depthwise conv1d with more than one output height block per core
    (in0_num_blocks_h > 1) on both the coalesced and non-coalesced read paths."""
    torch.manual_seed(0)
    C, L, k, pad = channels, 512, 7, 3  # out_length == L
    groups = C
    x_ncl = torch.randn(1, C, L, dtype=torch.bfloat16).float()
    w = torch.randn(C, 1, k, dtype=torch.bfloat16).float()
    golden = torch.nn.functional.conv1d(x_ncl, w, bias=None, stride=1, padding=pad, groups=groups)

    x_tt = ttnn.from_torch(x_ncl.permute(0, 2, 1), dtype=ttnn.bfloat16)
    w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=False,
    )
    # Pin 8 cores in a row: 512 rows / 8 = 64 rows (2 tiles) per core.
    conv_config.override_sharding_config = True
    conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 0))})
    # One tile-row per height block -> in0_num_blocks_h == 2 per core.
    conv_config.act_block_h_override = 32

    tt_out, out_length = ttnn.conv1d(
        input_tensor=x_tt,
        weight_tensor=w_tt,
        device=device,
        in_channels=C,
        out_channels=C,
        batch_size=1,
        input_length=L,
        kernel_size=k,
        stride=1,
        padding=pad,
        groups=groups,
        conv_config=conv_config,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )

    out = ttnn.to_torch(tt_out).reshape(1, out_length, C).permute(0, 2, 1)
    passing, pcc_msg = check_with_pcc_without_tensor_printout(out, golden, pcc=0.998)
    logger.info(f"[{path}] {pcc_msg}")
    assert passing, pcc_msg


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 16}], indirect=True)
@pytest.mark.parametrize(
    "C, k, stride, path",
    (
        # fp32: 64*4*12 = 3072 B fits every arch's NoC burst -> coalesced.
        (64, 12, 1, "coalesced"),
        # fp32: 512*4*12 = 24576 B exceeds WH (8192B) and BH (16384B) bursts -> non-coalesced.
        (512, 12, 2, "non-coalesced"),
    ),
)
def test_conv1d_depthwise_fp32_exact(device, C, k, stride, path):
    """fp32 depthwise conv1d precision gate for both SFPU accumulation layouts.

    Inputs are TRUE fp32 (not bf16-widened, which is exact under any path and cannot see an
    unpack-mode regression). The SFPU path measures ~7e-8 rel RMSE vs a float64 golden; the
    FPU path it replaced measures ~1.5e-3, so the 5e-6 bound fails a regression by ~300x.
    """
    torch.manual_seed(0)
    L = 4096
    x = torch.randn(1, C, L, dtype=torch.float32)
    w = torch.randn(C, 1, k, dtype=torch.float32)
    golden = torch.nn.functional.conv1d(x.double(), w.double(), stride=stride, padding=0, groups=C)

    x_tt = ttnn.from_torch(x.permute(0, 2, 1), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    w_tt = ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_out, out_length = ttnn.conv1d(
        input_tensor=x_tt,
        weight_tensor=w_tt,
        device=device,
        in_channels=C,
        out_channels=C,
        batch_size=1,
        input_length=L,
        kernel_size=k,
        stride=stride,
        padding=0,
        groups=C,
        conv_config=ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        compute_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        ),
        dtype=ttnn.float32,
        return_output_dim=True,
    )
    out = ttnn.to_torch(tt_out).reshape(1, out_length, C).permute(0, 2, 1).double()
    rel_rmse = ((out - golden).pow(2).mean().sqrt() / golden.std()).item()
    logger.info(f"[{path}] fp32 depthwise rel RMSE vs float64: {rel_rmse:.3e}")
    assert rel_rmse <= 5e-6, f"[{path}] fp32 depthwise conv1d rel RMSE {rel_rmse:.3e} > 5e-6 vs float64 golden"
