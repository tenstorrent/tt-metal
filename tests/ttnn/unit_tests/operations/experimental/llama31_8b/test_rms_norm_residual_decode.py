# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.fixture(autouse=True)
def p100_only(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Llama-3.1-8B P100 configs")


DIM = 4096


def _width_sharded(x, y, w):
    return ttnn.create_sharded_memory_config(
        shape=(32, w),
        core_grid=ttnn.CoreGrid(y=y, x=x),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _program_config(gx, gy):
    block_w = DIM // (gx * gy) // 32
    subblock_w = 4
    while block_w % subblock_w:
        subblock_w -= 1
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[gx, gy], subblock_w=subblock_w, block_h=1, block_w=block_w, inplace=False
    )


def _as_int16(t):
    return ttnn.to_torch(t).contiguous().view(torch.int16)


@pytest.mark.parametrize("norm_grid", [(8, 8, 64), (8, 4, 128)], ids=["ff", "attn"])
@pytest.mark.parametrize("input_layout", ["residual", "matmul_64", "norm"])
def test_rms_norm_residual_decode_matches_add_and_rms_norm(device, norm_grid, input_layout):
    torch.manual_seed(0)
    gx, gy, shard_w = norm_grid
    residual_mc = _width_sharded(8, 4, 128)
    norm_mc = _width_sharded(gx, gy, shard_w)
    input_mc = {"residual": residual_mc, "matmul_64": _width_sharded(8, 8, 64), "norm": norm_mc}[input_layout]
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    weight = ttnn.as_tensor(
        (torch.rand(DIM) * 2).reshape(1, 1, DIM // 32, 32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    entries = None
    for scale_r, scale_y in ((1.0, 1.0), (50.0, 0.01)):
        r0 = torch.randn(1, 1, 32, DIM) * scale_r
        y0 = torch.randn(1, 1, 32, DIM) * scale_y
        r = ttnn.from_torch(r0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=residual_mc)
        y = ttnn.from_torch(y0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=residual_mc)
        h_ref = ttnn.add(r, y, memory_config=residual_mc)
        out_ref = ttnn.rms_norm(
            ttnn.to_memory_config(h_ref, norm_mc),
            epsilon=1e-5,
            weight=weight,
            program_config=_program_config(gx, gy),
            memory_config=norm_mc,
            compute_kernel_config=ckc,
        )
        residual = ttnn.from_torch(
            r0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=residual_mc
        )
        y_in = ttnn.from_torch(y0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mc)
        call = lambda: ttnn.experimental.llama31_8b.rms_norm_residual_decode(
            y_in,
            residual,
            weight,
            1e-5,
            program_config=_program_config(gx, gy),
            memory_config=norm_mc,
            compute_kernel_config=ckc,
        )
        assert ttnn.experimental.llama31_8b.rms_norm_residual_decode_supported(
            y_in, residual, weight, program_config=_program_config(gx, gy), memory_config=norm_mc
        )
        out = call()
        if entries is None:
            entries = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == entries
        assert torch.equal(_as_int16(out), _as_int16(out_ref))
        assert torch.equal(_as_int16(residual), _as_int16(h_ref))


@pytest.mark.parametrize("case", ["width_2048", "residual_8x8", "input_4x8", "grid_4x8", "subblock_1", "weight_tile"])
def test_rms_norm_residual_decode_rejects_calls_outside_its_table(device, case, expect_error):
    width = 2048 if case == "width_2048" else DIM
    residual_mc = _width_sharded(8, 4, width // 32)
    norm_mc = _width_sharded(8, 4, width // 32)
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4], subblock_w=4, block_h=1, block_w=width // 32 // 32, inplace=False
    )
    input_mc = residual_mc
    if case == "residual_8x8":
        residual_mc = _width_sharded(8, 8, 64)
    elif case == "input_4x8":
        input_mc = _width_sharded(4, 8, 128)
    elif case == "grid_4x8":
        norm_mc = _width_sharded(4, 8, 128)
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[4, 8], subblock_w=4, block_h=1, block_w=4, inplace=False
        )
    elif case == "subblock_1":
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4], subblock_w=1, block_h=1, block_w=4, inplace=False
        )
    weight = ttnn.as_tensor(
        torch.ones(1, 1, width // 32, 32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT if case == "weight_tile" else ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    r0 = torch.randn(1, 1, 32, width)
    residual = ttnn.from_torch(
        r0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=residual_mc
    )
    y_in = ttnn.from_torch(
        torch.randn(1, 1, 32, width),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mc,
    )
    assert not ttnn.experimental.llama31_8b.rms_norm_residual_decode_supported(
        y_in, residual, weight, program_config=program_config, memory_config=norm_mc
    )
    with expect_error(RuntimeError, "rms_norm_residual_decode"):
        ttnn.experimental.llama31_8b.rms_norm_residual_decode(
            y_in,
            residual,
            weight,
            1e-5,
            program_config=program_config,
            memory_config=norm_mc,
            compute_kernel_config=ckc,
        )
    assert torch.equal(_as_int16(residual), r0.bfloat16().view(torch.int16))
