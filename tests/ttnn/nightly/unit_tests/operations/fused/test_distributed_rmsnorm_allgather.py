# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.fused.utility_functions import (
    ttnn_rms_norm_pre_all_gather,
    ttnn_rms_norm_post_all_gather,
)

# Module-scoped device: every test here shares one device configuration
pytestmark = pytest.mark.use_module_device


def _assert_multi_row_2d_layout(device, seq_len, hidden_per_device):
    """Verify that the test shape assigns multiple rows to each core."""
    tile_size = 32
    num_tile_rows = seq_len // tile_size
    width_tiles = hidden_per_device // tile_size
    max_cores_y = device.compute_with_storage_grid_size().y

    cores_x = min(max_cores_y, num_tile_rows)
    while num_tile_rows % cores_x != 0 and cores_x > 1:
        cores_x -= 1
    tiles_per_core_x = num_tile_rows // cores_x

    cores_y = min(max_cores_y, width_tiles)
    while width_tiles % cores_y != 0 and cores_y > 1:
        cores_y -= 1
    tiles_per_core_y = width_tiles // cores_y

    assert cores_x > 1
    assert cores_y > 1
    assert tiles_per_core_x > 1
    assert tiles_per_core_y < width_tiles


def _run_distributed_rmsnorm_single_device(
    device,
    seq_len,
    hidden_dim_total,
    num_simulated_devices,
    use_2d_core_grid,
    eps=1e-5,
    pcc_threshold=0.99,
):
    assert hidden_dim_total % num_simulated_devices == 0
    hidden_per_dev = hidden_dim_total // num_simulated_devices

    torch.manual_seed(1234)

    inp_shape = (1, 1, seq_len, hidden_dim_total)
    torch_input = torch.randn(inp_shape, dtype=torch.bfloat16)
    torch_weight = torch.randn(hidden_dim_total, dtype=torch.bfloat16)

    # Reference output
    ref = torch.nn.RMSNorm(normalized_shape=hidden_dim_total, eps=eps)
    ref.weight.data = torch_weight.clone().float()
    with torch.no_grad():
        torch_output = ref(torch_input.float()).to(torch.bfloat16)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # Chunk input and weight along hidden dim to simulate per-device shards.
    input_chunks = torch.chunk(torch_input, num_simulated_devices, dim=-1)
    weight_chunks = torch.chunk(torch_weight, num_simulated_devices, dim=-1)

    tt_inputs = [
        ttnn.from_torch(
            chunk,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for chunk in input_chunks
    ]
    tt_weights = [
        ttnn.from_torch(
            w.reshape(1, 1, 1, hidden_per_dev),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in weight_chunks
    ]

    # Step 1: per-shard pre-all-gather stats.
    tt_stats = [
        ttnn_rms_norm_pre_all_gather(
            t,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            use_2d_core_grid=use_2d_core_grid,
        )
        for t in tt_inputs
    ]

    # Step 2: simulate the all-gather by concatenating along stats dim (=3).
    tt_stats_gathered = ttnn.concat(tt_stats, dim=3)

    # Step 3: per-shard post-all-gather norm using the gathered stats.
    tt_outputs = [
        ttnn_rms_norm_post_all_gather(
            tt_inputs[i],
            tt_stats_gathered,
            epsilon=eps,
            weight=tt_weights[i],
            compute_kernel_config=compute_kernel_config,
            use_2d_core_grid=use_2d_core_grid,
        )
        for i in range(num_simulated_devices)
    ]

    tt_out_concat = ttnn.concat(tt_outputs, dim=-1)
    tt_output_torch = ttnn.to_torch(tt_out_concat).to(torch.bfloat16)

    passing, pcc_msg = comp_pcc(torch_output, tt_output_torch, pcc=pcc_threshold)
    logger.info(f"use_2d_core_grid={use_2d_core_grid} | {pcc_msg}")

    return passing, pcc_msg


@pytest.mark.parametrize(
    "seq_len, hidden_dim_total, num_simulated_devices, requires_multi_row_2d",
    [
        # LLaMA 70B Galaxy decode shape: hidden=8192, cluster_axis=1 with 4 devices.
        # The 2D-grid path activates when shape[-2] == 128.
        (128, 8192, 4, False),
        # Regression: cores_y > tiles_per_core_y (Wt=32 -> cores_y=8, tiles_per_core_y=4) exercises
        # the c_15 merge-gather CB OOB; fails unless c_15 is sized by cores_y.
        (128, 4096, 4, False),
        # On an 8-row grid: Ht=12 -> 6 cores x 2 rows, Wt=20 -> 5 cores x 4 columns.
        (384, 2560, 4, True),
    ],
)
@pytest.mark.parametrize("use_2d_core_grid", [False, True])
def test_rmsnorm_2d_core_grid_single_device(
    device, seq_len, hidden_dim_total, num_simulated_devices, requires_multi_row_2d, use_2d_core_grid
):
    if use_2d_core_grid and requires_multi_row_2d:
        _assert_multi_row_2d_layout(device, seq_len, hidden_dim_total // num_simulated_devices)
    passing, pcc_msg = _run_distributed_rmsnorm_single_device(
        device=device,
        seq_len=seq_len,
        hidden_dim_total=hidden_dim_total,
        num_simulated_devices=num_simulated_devices,
        use_2d_core_grid=use_2d_core_grid,
    )
    assert passing, f"PCC check failed (use_2d_core_grid={use_2d_core_grid}): {pcc_msg}"
