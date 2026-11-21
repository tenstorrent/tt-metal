# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the thread plan of rotary_embedding_llama requires (op_design.md, Work Distribution):

(a) every NEO of the plan computed its own head_dim tiles (a correct result alone does not show it);
(b) the compute-thread clamp, one case per effective C, plus a plan where a NEO holds two tiles, so the
    matmul's strided tile index (k * stride) is exercised;
(c) the op in one device session after other programs and before one of tt-metal's own.
"""

import pytest
import torch

import ttnn
from ttnn.operations.rotary_embedding_llama import default_compute_kernel_config, rotary_embedding_llama
from ttnn.operations.rotary_embedding_llama.rotary_embedding_llama import _dispatch
from ttnn.operations.rotary_embedding_llama.rotary_embedding_llama_program_artifacts import (
    THREAD_PLAN,
    TILE,
    effective_compute_threads,
)

PCC = 0.9997  # the first phase's numeric entry (bfloat16, 32-bit accumulator)


def _shard(device, torch_tensor, rows):
    """Height-sharded in L1 on core (0, 0), one tile row per shard, as the callers place them."""
    config = ttnn.create_sharded_memory_config(
        shape=(rows, torch_tensor.shape[-1]),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)


def _inputs(device, heads, head_dim, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, 1, heads, head_dim).bfloat16()
    cos = torch.randn(1, 1, 1, head_dim).bfloat16()
    sin = torch.randn(1, 1, 1, head_dim).bfloat16()
    t = torch.randn(1, 1, TILE, TILE).bfloat16()
    host = (x, cos, sin, t)
    return host, tuple(_shard(device, h, TILE) for h in host)


def _terms(x, cos, sin, t):
    """(rotated * sin, x * cos) in float64."""
    x64 = x.double()
    rotated = (x64.reshape(*x.shape[:-1], -1, TILE) @ t.double().reshape(TILE, TILE)).reshape(x.shape)
    return rotated * sin.double(), x64 * cos.double()


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _check(got, expected):
    pcc = _pcc(got, expected)
    assert pcc >= PCC, f"PCC {pcc} < {PCC}"


@pytest.mark.parametrize(
    "head_dim, threads, expected_c",
    [
        (32, THREAD_PLAN, 1),
        (64, THREAD_PLAN, 2),
        (96, THREAD_PLAN, 1),
        (128, THREAD_PLAN, 4),
        (128, (1, 2, 1), 2),  # a NEO holds two tiles: the matmul's tile index is k * stride for k = 1
        (128, (1, 1, 1), 1),  # one NEO, four tiles in one DEST section
    ],
)
@pytest.mark.parametrize("heads", [32, 8])
def test_thread_clamp(device, heads, head_dim, threads, expected_c):
    assert effective_compute_threads(head_dim // TILE, threads) == expected_c
    host, dev = _inputs(device, heads, head_dim)
    out = _dispatch(*dev, default_compute_kernel_config(), threads=threads)
    a, b = _terms(*host)
    _check(ttnn.to_torch(out).float(), (a + b).float())


@pytest.mark.parametrize("threads", [THREAD_PLAN, (1, 2, 1)], ids=lambda p: "R%dC%dW%d" % p)
def test_each_neo_computes_its_own_tiles(device, threads):
    """With tag_threads, odd NEOs subtract: head_dim tile j must come from NEO j mod C."""
    heads, head_dim = 32, 128
    C = effective_compute_threads(head_dim // TILE, threads)
    assert C == threads[1]
    host, dev = _inputs(device, heads, head_dim, seed=1)
    got = ttnn.to_torch(_dispatch(*dev, default_compute_kernel_config(), threads=threads, tag_threads=True)).float()
    a, b = _terms(*host)
    subtracted = 0
    for j in range(head_dim // TILE):
        cols = slice(j * TILE, (j + 1) * TILE)
        odd_neo = (j % C) & 1
        subtracted += odd_neo
        expected = (a - b if odd_neo else a + b)[..., cols].float()
        assert _pcc(got[..., cols], expected) >= PCC, (j, j % C)
    assert subtracted == (head_dim // TILE) // 2  # half the tiles came from odd NEOs


def test_session_order(device):
    """In one device session: the op after another call of itself and after tt-metal's own Quasar add (implicit
    sync, tagged NoC transfers), and before a program of tt-metal's own (to_layout to row-major, an untilize with
    untagged NoC transfers)."""
    host, dev = _inputs(device, 32, 128, seed=2)
    a, b = _terms(*host)
    expected = (a + b).float()
    first = rotary_embedding_llama(*dev, is_decode_mode=True)
    _check(ttnn.to_torch(first).float(), expected)

    tx = torch.randn(1, 1, 64, 64).bfloat16()
    plain = ttnn.from_torch(tx, layout=ttnn.TILE_LAYOUT, device=device)
    summed = ttnn.experimental.quasar.add(plain, plain)
    assert torch.allclose(ttnn.to_torch(summed).float(), (tx + tx).float(), atol=0.05)

    second = rotary_embedding_llama(*dev, is_decode_mode=True)
    _check(ttnn.to_torch(second).float(), expected)
    row_major = ttnn.to_layout(second, ttnn.ROW_MAJOR_LAYOUT)
    _check(ttnn.to_torch(ttnn.from_device(row_major)).float(), expected)
    assert second.memory_config() == dev[0].memory_config()
