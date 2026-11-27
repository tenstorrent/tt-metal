# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The block-sharded layer/RMS norm program over interleaved (DRAM or L1) input and output tensors: each core reads its
block of the input's tile grid by tile id and writes the output the same way."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

MEMS = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
EPS = 1e-6


def largest_divisor(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def block_program_config(device, rows, width):
    mt, wt = rows // 32, width // 32
    grid = device.compute_with_storage_grid_size()
    gy = largest_divisor(mt, grid.y)
    gx = largest_divisor(wt, min(grid.x, 8))
    block_w = wt // gx
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        subblock_w=largest_divisor(block_w, 4),
        block_h=mt // gy,
        block_w=block_w,
        inplace=False,
    )


@pytest.mark.parametrize("rows, width", [(160, 2048), (640, 512), (32, 1536), (640, 128)])
@pytest.mark.parametrize("norm", ["rms", "layer"])
@pytest.mark.parametrize("in_mem, out_mem", [("dram", "dram"), ("l1", "l1"), ("l1", "dram")])
@pytest.mark.parametrize("affine", ["row_major", "tile", None], ids=["affine_rm", "affine_tile", "no_affine"])
def test_sharded_program_interleaved_io(device, rows, width, norm, in_mem, out_mem, affine):
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, width) * 3 + 1
    gamma = torch.randn(width) if affine else None
    beta = torch.randn(width) if affine and norm == "layer" else None

    def reference(x):
        if norm == "rms":
            y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
        else:
            y = torch.nn.functional.layer_norm(x, (width,), eps=EPS)
        if gamma is not None:
            y = y * gamma
        if beta is not None:
            y = y + beta
        return y

    def to_device(x):
        return ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=MEMS[in_mem]
        )

    tt_x = to_device(x)
    # Row-major gamma/beta (as the models pass them, [W/32, 32] rows) or a [1, W] TILE row.
    affine_args = {
        name: ttnn.from_torch(
            t.reshape(1, 1, -1, 32) if affine == "row_major" else t.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if affine == "row_major" else ttnn.TILE_LAYOUT,
            device=device,
        )
        for name, t in (("weight", gamma), ("bias", beta))
        if t is not None
    }
    op = ttnn.rms_norm if norm == "rms" else ttnn.layer_norm
    program_config = block_program_config(device, rows, width)

    out = op(tt_x, epsilon=EPS, program_config=program_config, memory_config=MEMS[out_mem], **affine_args)
    assert out.memory_config().buffer_type == MEMS[out_mem].buffer_type and not out.is_sharded()
    interleaved = op(tt_x, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, **affine_args)
    got = ttnn.to_torch(out).float()
    assert_with_pcc(reference(x), got, 0.999)
    assert_with_pcc(ttnn.to_torch(interleaved).float(), got, 0.9995)

    # A program-cache hit with different data at a fresh input address (the first input stays live), and a fresh
    # output, reads and writes the new buffers.
    x2 = torch.randn(1, 1, rows, width) * 2 - 1
    tt_x2 = to_device(x2)
    entries = device.num_program_cache_entries()
    again = op(tt_x2, epsilon=EPS, program_config=program_config, memory_config=MEMS[out_mem], **affine_args)
    assert device.num_program_cache_entries() == entries, "the second call must hit the program cache"
    assert tt_x2.buffer_address() != tt_x.buffer_address()
    assert_with_pcc(reference(x2), ttnn.to_torch(again).float(), 0.999)
