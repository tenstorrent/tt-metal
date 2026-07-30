# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Validator tests for mcast reuse matmul program configs.

The 2D reuse-mcast sender-writer path (used by every core in the M-column
when ``num_blocks_y == 1``) always emits ``per_core_M`` tile-rows and does
not clamp its H-loop to ``Mt``. When ``per_core_M > Mt`` those extra rows
land past the end of the output DRAM buffer and silently corrupt whichever
allocations follow it in the interleaved DRAM page stream. The 1D
``mcast_in1`` path has the analogous bug at ``start_core``.

These tests exercise the validator asserts added in
``matmul_device_operation.cpp`` that reject those configs up front.
"""

import pytest
import torch

import ttnn


M = 128  # Mt = M / TILE = 4
K = 32
N = 32
TILE = 32


def _from_torch(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_inputs(device):
    torch.manual_seed(0)
    a = torch.zeros((1, 1, M, K), dtype=torch.bfloat16)
    b = torch.zeros((1, 1, K, N), dtype=torch.bfloat16)
    return _from_torch(a, device), _from_torch(b, device)


@pytest.mark.parametrize(
    "per_core_M",
    [
        pytest.param(7, id="per_core_M=7"),
        pytest.param(8, id="per_core_M=8"),
    ],
)
def test_mcast_2d_rejects_per_core_M_gt_Mt(device, expect_error, per_core_M):
    """2D reuse-mcast must reject ``per_core_M > Mt`` at validate time.

    Otherwise the sender-writer's H-loop overruns the output DRAM buffer by
    ``(per_core_M - Mt)`` tile-rows per core.
    """
    activation_tt, weight_tt = _make_inputs(device)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )

    with expect_error(RuntimeError, "per_core_M .* must be <= Mt"):
        ttnn.matmul(
            activation_tt,
            weight_tt,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )


@pytest.mark.parametrize(
    "per_core_M",
    [
        pytest.param(7, id="per_core_M=7"),
        pytest.param(8, id="per_core_M=8"),
    ],
)
def test_mcast_1d_mcast_in1_rejects_per_core_M_gt_Mt(device, expect_error, per_core_M):
    """1D reuse-mcast with ``mcast_in0=False`` (mcast_in1) must reject
    ``per_core_M > Mt``. ``mcast_in0`` is unaffected — its factory clamps
    ``in0_last_per_core_M = min(M, per_core_M)`` and pushes the clamped H
    count into the sender-writer runtime args, which is why the existing
    ``num_blocks_y == 1`` check for that mode is safe.
    """
    activation_tt, weight_tt = _make_inputs(device)
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=False,
    )

    with expect_error(RuntimeError, "mcast_in1 requires per_core_M .* <= Mt"):
        ttnn.matmul(
            activation_tt,
            weight_tt,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
