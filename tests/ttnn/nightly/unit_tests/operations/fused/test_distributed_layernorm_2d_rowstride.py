# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Regression test for https://github.com/tenstorrent/tt-metal/issues/56908
# (distributed LayerNorm/RMSNorm 2D-core-grid row-stride corruption).
#
# Exercises the post-all-gather Welford path with use_2d_core_grid=True on
# multi-row shapes (tiles_per_core_x > 1), where each core owns several
# row-tiles and the reader/writer must stride by (Wt - tiles_per_core_y)
# between local rows. Compares end-to-end against torch.nn.LayerNorm.

import pytest
import torch
import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.fused.utility_functions import (
    ttnn_layer_norm_pre_all_gather,
    ttnn_layer_norm_post_all_gather,
)

# Module-scoped device: every test here shares one device configuration
pytestmark = pytest.mark.use_module_device


def _run_distributed_layernorm_single_device(
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
    torch_bias = torch.randn(hidden_dim_total, dtype=torch.bfloat16)

    # Reference output
    ref = torch.nn.LayerNorm(normalized_shape=hidden_dim_total, eps=eps)
    ref.weight.data = torch_weight.clone().float()
    ref.bias.data = torch_bias.clone().float()
    with torch.no_grad():
        torch_output = ref(torch_input.float()).to(torch.bfloat16)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # Chunk input / weight / bias along hidden dim to simulate per-device shards.
    input_chunks = torch.chunk(torch_input, num_simulated_devices, dim=-1)
    weight_chunks = torch.chunk(torch_weight, num_simulated_devices, dim=-1)
    bias_chunks = torch.chunk(torch_bias, num_simulated_devices, dim=-1)

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
    tt_biases = [
        ttnn.from_torch(
            b.reshape(1, 1, 1, hidden_per_dev),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for b in bias_chunks
    ]

    # Step 1: per-shard pre-all-gather stats (Welford partials).
    tt_stats = [
        ttnn_layer_norm_pre_all_gather(
            t,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            use_2d_core_grid=use_2d_core_grid,
        )
        for t in tt_inputs
    ]

    # Step 2: simulate the all-gather by concatenating along stats dim (=3).
    tt_stats_gathered = ttnn.concat(tt_stats, dim=3)

    # Step 3: per-shard post-all-gather norm (Welford path) using the gathered stats.
    tt_outputs = [
        ttnn_layer_norm_post_all_gather(
            tt_inputs[i],
            tt_stats_gathered,
            epsilon=eps,
            weight=tt_weights[i],
            bias=tt_biases[i],
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
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
    "seq_len, hidden_dim_total, num_simulated_devices",
    [
        # Regression for Issue #56908: multi-row per core (tiles_per_core_x > 1).
        # Ht=8 (seq_len=256) -> tiles_per_core_x=2; catches row-stride flat-increment
        # corruption on both the read and write paths of the Welford kernels.
        (256, 8192, 4),
        # Ht=16 (seq_len=512) -> tiles_per_core_x=4.
        (512, 4096, 4),
    ],
)
@pytest.mark.parametrize("use_2d_core_grid", [True])
def test_layernorm_2d_core_grid_rowstride_single_device(
    device, seq_len, hidden_dim_total, num_simulated_devices, use_2d_core_grid
):
    passing, pcc_msg = _run_distributed_layernorm_single_device(
        device=device,
        seq_len=seq_len,
        hidden_dim_total=hidden_dim_total,
        num_simulated_devices=num_simulated_devices,
        use_2d_core_grid=use_2d_core_grid,
    )
    assert passing, f"PCC check failed (use_2d_core_grid={use_2d_core_grid}): {pcc_msg}"
