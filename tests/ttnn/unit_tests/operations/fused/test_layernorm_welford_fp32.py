# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Welford LayerNorm FP32 coverage (issue #44650).

The legacy (non-Welford) FP32 path is already covered by ``test_layernorm_mix_precision``
(interleaved) and ``test_layernorm_sharded_*_mix_precision_rm`` (sharded); the distributed
post-all-gather FP32 path by ``test_layernorm_part_2_fp32``. This file fills the remaining
gap: the **Welford** compute path with FLOAT32 input/output, on both the interleaved and
block-sharded program factories, with FP32 and bf16 gamma/beta.

FP32 on Welford requires ``fp32_dest_acc_en=True`` (the unpacker must preserve the full
mantissa into DEST; the sharded factory also enables its ``welford_fp32_alias`` only under
that flag). bf16 + legacy are parametrized as passing controls. Welford requires TILE input
(row-major input is unsupported and hangs, independent of dtype) and does not support RMSNorm.
"""

import math

import pytest
import torch

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _compute_kernel_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,  # required for FP32 on the Welford path
        packer_l1_acc=False,
    )


def _recip_interleaved(device, width):
    grid = device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.create_layer_norm_reciprocals(device, crs, width)


@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32], ids=["gb_bf16", "gb_fp32"])
@pytest.mark.parametrize("use_welford", [True, False], ids=["welford", "legacy"])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_layernorm_interleaved_welford_fp32(device, dtype, use_welford, gamma_dtype):
    torch.manual_seed(0)
    M, K = 256, 1024
    x = torch.rand((1, 1, M, K), dtype=torch.float32)
    w = torch.rand((K,), dtype=torch.float32)
    b = torch.rand((K,), dtype=torch.float32)
    ref = torch.nn.functional.layer_norm(x, (K,), weight=w, bias=b, eps=1e-12)

    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    wt = ttnn.from_torch(w.reshape(1, 1, 1, K), dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    bt = ttnn.from_torch(b.reshape(1, 1, 1, K), dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip = _recip_interleaved(device, K) if use_welford else None

    out = ttnn.layer_norm(
        xt,
        epsilon=1e-12,
        weight=wt,
        bias=bt,
        program_config=cfg,
        compute_kernel_config=_compute_kernel_config(device),
        recip_tensor=recip,
    )
    ot = ttnn.to_torch(ttnn.from_device(out)).float().reshape(ref.shape)

    passing, pcc = comp_pcc(ref, ot, pcc=0.999)
    assert (
        passing
    ), f"interleaved {'welford' if use_welford else 'legacy'} {dtype} gamma={gamma_dtype} PCC failed: {pcc}"


@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32], ids=["gb_bf16", "gb_fp32"])
@pytest.mark.parametrize("use_welford", [True, False], ids=["welford", "legacy"])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_layernorm_block_sharded_welford_fp32(device, dtype, use_welford, gamma_dtype):
    torch.manual_seed(1234)
    g = device.compute_with_storage_grid_size()
    grid_size = [g.x, min(g.y, 8)]
    batch = grid_size[1]
    width = 128 * grid_size[1]
    in0_shape = (batch, 1, 32 * grid_size[0], width)
    M, K = in0_shape[2] * batch, in0_shape[3]

    x = torch.rand(in0_shape, dtype=torch.float32) * 2 - 0.95
    w = torch.rand(K, dtype=torch.float32) * 2 - 1
    b = torch.rand(K, dtype=torch.float32) * 2 - 1.1
    ref = torch.nn.functional.layer_norm(x, (K,), weight=w, bias=b, eps=1e-2)

    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_shape = [M // grid_size[0], math.ceil(K / grid_size[1] / 32) * 32]
    x_shard = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_shape, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.ShardOrientation.COL_MAJOR
    )

    wt = ttnn.from_torch(w.reshape(1, 1, -1, 32), dtype=gamma_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    bt = ttnn.from_torch(b.reshape(1, 1, -1, 32), dtype=gamma_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        inplace=False,
        use_welford=use_welford,
    )
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, x_shard.memory_config().shard_spec
    )

    recip = None
    if use_welford:
        sspec = x_shard.memory_config().shard_spec
        recip = ttnn.create_layer_norm_reciprocals(device, sspec.grid, sspec.shape[1])

    out = ttnn.layer_norm(
        x_shard,
        epsilon=1e-2,
        weight=wt,
        bias=bt,
        memory_config=out_mem,
        program_config=cfg,
        compute_kernel_config=_compute_kernel_config(device),
        recip_tensor=recip,
    )
    ot = (
        ttnn.to_torch(ttnn.from_device(ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)))
        .float()
        .reshape(ref.shape)
    )

    passing, pcc = comp_pcc(ref, ot, pcc=0.999)
    assert (
        passing
    ), f"block-sharded {'welford' if use_welford else 'legacy'} {dtype} gamma={gamma_dtype} PCC failed: {pcc}"
