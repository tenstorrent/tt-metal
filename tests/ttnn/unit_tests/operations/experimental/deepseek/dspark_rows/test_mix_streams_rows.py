# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek.mix_streams`` (the unfused hyper-connection stream mix) on a DSpark block:
streams [1, 1, hc*h, D] in (h x 32) tiles (row s*h + j = stream s of block row j), sublayer_out [1, 1, h, D] in the
same tiles, per-row post [1, n, hc, 1] / comb [1, n, hc, hc] (TILE 32x32, block row j at dim 1).

    out[i, j] = sum_s comb_j[s, i] * streams[s, j] + post_j[i] * sublayer_out[j]

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/dspark_rows/test_mix_streams_rows.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config

HC = 4
D = 5120
STREAMS_CORES, SUB_CORES = 8, 80
HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _reference(post, comb, sub, streams):
    """post [n, hc], comb [n, hc, hc], sub [n, D], streams [n, hc, D] -> [n, hc, D]."""
    return torch.einsum("jsi,jsd->jid", comb, streams) + post[..., None] * sub[:, None, :]


def _inputs(device, rows, h, seed=0):
    torch.manual_seed(seed)
    post = torch.randn(rows, HC).bfloat16().float()
    comb = torch.rand(rows, HC, HC).bfloat16().float()
    sub = torch.randn(rows, D).bfloat16().float()
    streams = torch.randn(rows, HC, D).bfloat16().float()
    dram_tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if h == 0:
        post_tt = ttnn.from_torch(post.reshape(1, rows, HC, 1), **dram_tile)
        comb_tt = ttnn.from_torch(comb.reshape(1, rows, HC, HC), **dram_tile)
        sub_tt = ttnn.from_torch(
            sub.reshape(1, rows, 1, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=width_sharded_l1_config(rows, D, device, num_cores=SUB_CORES, tile_height=1),
        )
        streams_tt = ttnn.from_torch(
            streams.reshape(1, rows, HC, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=width_sharded_l1_config(rows * HC, D, device, num_cores=STREAMS_CORES, tile_height=1),
        )
    else:
        tile = ttnn.Tile((h, 32))
        post_tt = ttnn.from_torch(post.reshape(1, rows, HC, 1), **dram_tile)
        comb_tt = ttnn.from_torch(comb.reshape(1, rows, HC, HC), **dram_tile)
        sub_tt = ttnn.from_torch(
            sub.reshape(1, 1, rows, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=tile,
            device=device,
            memory_config=width_sharded_l1_config(h, D, device, num_cores=SUB_CORES, tile_height=h),
        )
        xs = torch.zeros(HC, h, D)
        xs[:, :rows] = streams.transpose(0, 1)
        streams_tt = ttnn.from_torch(
            xs.reshape(1, 1, HC * h, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=tile,
            device=device,
            memory_config=width_sharded_l1_config(HC * h, D, device, num_cores=STREAMS_CORES, tile_height=h),
        )
    return (post, comb, sub, streams), (post_tt, comb_tt, sub_tt, streams_tt)


@pytest.mark.parametrize("rows, h", [(4, 4), (1, 1), (2, 2), (5, 8), (8, 8)], ids=str)
def test_mix_streams_rows(device, rows, h):
    ref_in, tt_in = _inputs(device, rows, h)
    expected = _reference(*ref_in)
    for _ in range(2):
        out_tt = ttnn.experimental.deepseek.mix_streams(*tt_in, compute_kernel_config=HIFI4)
    streams_tt = tt_in[3]
    assert out_tt.memory_config() == streams_tt.memory_config()
    assert out_tt.layout == ttnn.TILE_LAYOUT and tuple(out_tt.tile.tile_shape) == (h, 32)
    out = ttnn.to_torch(out_tt).float().reshape(HC, h, D)
    for j in range(rows):
        for i in range(HC):
            passing, pcc = comp_pcc(expected[j, i], out[i, j], 0.9999)
            logger.info(f"row {j} stream {i}: PCC {pcc}")
            assert passing, f"row {j} stream {i}: PCC {pcc}"
    # Padding rows of a block (j >= rows) mix to zero.
    assert torch.all(out[:, rows:] == 0)


def test_mix_streams_rows_perf(device):
    from tracy import signpost

    _, s1 = _inputs(device, 1, 0)
    _, s4 = _inputs(device, 4, 4)
    for name, args in {"s1": s1, "s4": s4}.items():
        ttnn.deallocate(ttnn.experimental.deepseek.mix_streams(*args, compute_kernel_config=HIFI4))
        ttnn.synchronize_device(device)
        signpost(name)
        for _ in range(20):
            ttnn.deallocate(ttnn.experimental.deepseek.mix_streams(*args, compute_kernel_config=HIFI4))
        ttnn.synchronize_device(device)
