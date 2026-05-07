# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Single-device reproducer for the bad-PCC bug triggered by use_2d_core_grid=True
in ttnn.rms_norm_pre_all_gather / ttnn.rms_norm_post_all_gather.

Reference: https://github.com/tenstorrent/tt-metal/issues/35435

Originally observed in the LLaMA 70B Galaxy model (6u, 8x4 mesh) via
models/demos/llama3_70b_galaxy/tt/llama_ccl.py:tt_distributed_rmsnorm, where
use_2d_grid is enabled when inp.shape[-2] == 128. With the flag enabled,
rms_norm_post_all_gather emits garbage; with it disabled, output is correct.

This test simulates the multi-device flow on a single device by chunking the
input across the hidden-dim axis, running rms_norm_pre_all_gather per chunk,
concatenating the per-chunk stats (acting as the all-gather), and running
rms_norm_post_all_gather per chunk. The flag is parametrized so the failing
configuration can be compared directly against the passing one.
"""

import pytest
import torch
import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


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
        ttnn.rms_norm_pre_all_gather(
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
        ttnn.rms_norm_post_all_gather(
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
    "seq_len, hidden_dim_total, num_simulated_devices",
    [
        # LLaMA 70B Galaxy decode shape: hidden=8192, cluster_axis=1 with 4 devices.
        # The 2D-grid path activates when shape[-2] == 128.
        (128, 8192, 4),
    ],
)
@pytest.mark.parametrize(
    "use_2d_core_grid",
    [
        pytest.param(False, id="use_2d_core_grid_False"),
        pytest.param(
            True,
            id="use_2d_core_grid_True",
            marks=pytest.mark.xfail(
                reason="Reproduces issue #35435: bad PCC when use_2d_core_grid=True",
                strict=False,
            ),
        ),
    ],
)
def test_rmsnorm_2d_core_grid_single_device(device, seq_len, hidden_dim_total, num_simulated_devices, use_2d_core_grid):
    passing, pcc_msg = _run_distributed_rmsnorm_single_device(
        device=device,
        seq_len=seq_len,
        hidden_dim_total=hidden_dim_total,
        num_simulated_devices=num_simulated_devices,
        use_2d_core_grid=use_2d_core_grid,
    )
    assert passing, f"PCC check failed (use_2d_core_grid={use_2d_core_grid}): {pcc_msg}"
