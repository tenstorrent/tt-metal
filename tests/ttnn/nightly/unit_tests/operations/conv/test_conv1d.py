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
