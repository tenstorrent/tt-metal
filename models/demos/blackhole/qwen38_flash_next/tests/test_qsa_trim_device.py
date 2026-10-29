# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_TRIM cuts on the TP=8 mesh, one-row decode operands: bit-for-bit against the programs they replace, and the
traced per-call time of both.  Run: QWEN38_FUSED_DEVICE_TEST=1 pytest <file> -s -p no:cacheprovider -o addopts=
(the MESH_DEVICE / TT_VISIBLE_DEVICES environment of the model's TP=8 runs)."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block, qsa_rows, qsa_trim

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="needs the TP=8 mesh")
REPLAYS = int(os.environ.get("X45_REPLAYS", "200"))
OUT = os.environ.get("X45_OUT")
RESULTS = {}


def _up(mesh, t, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _bits_equal(a, b):
    return a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _traced_us(mesh, call):
    def capture(n):
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(n):
            call()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        return trace

    def wall(trace):
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = 1e30
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, time.perf_counter() - t0)
        return best / REPLAYS * 1e6

    short, long_ = capture(4), capture(12)
    try:
        ts, tl = wall(short), wall(long_)
    finally:
        ttnn.release_trace(mesh, short)
        ttnn.release_trace(mesh, long_)
    return round((tl - ts) / 8, 2)


def _chain(local, blocks):
    """The decode chain: slice row 0 to the resident blocks, reshape to the gather's 2 KB pages."""

    row = ttnn.slice(local, (0, 0, 0, 0), (1, 1, 1, blocks), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    paged = ttnn.reshape(row, (1, 1, blocks // 1024, 1024))
    if paged.buffer_address() != row.buffer_address():
        ttnn.deallocate(row)
    return paged


def _free(t):
    for x in t if isinstance(t, (tuple, list)) else (t,):
        ttnn.deallocate(x)


def _dump():
    print("X45RESULT", json.dumps(RESULTS), flush=True)
    if OUT:
        with open(OUT, "w") as f:
            json.dump(RESULTS, f, indent=1)


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 8 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("blocks", [8192])
def test_score_pages_is_slice_plus_reshape(mesh_device, blocks):
    mesh = mesh_device
    g = torch.Generator().manual_seed(45)
    equal = 0
    for i in range(8):
        scores = (torch.randn(1, 1, 1, blocks + 32, generator=g) * 4).to(torch.bfloat16)
        pad = torch.zeros(1, 1, 31, blocks + 32, dtype=torch.bfloat16)
        local = _up(mesh, torch.cat([scores, pad], dim=2), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        want = _chain(local, blocks)
        got = qsa_trim.score_pages_row0(local, blocks)
        equal += int(_bits_equal(_dev0(want), _dev0(got)))
        _free([want, got])
        ttnn.deallocate(local)
    local = _up(mesh, torch.randn(1, 1, 32, blocks + 32).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    us_chain = _traced_us(mesh, lambda: _free(_chain(local, blocks)))
    us_new = _traced_us(mesh, lambda: _free(qsa_trim.score_pages_row0(local, blocks)))
    RESULTS["score_pages"] = {"bitwise_equal": f"{equal}/8", "us_chain": us_chain, "us_program": us_new}
    _dump()
    assert equal == 8


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_post_attention_rows_is_decode_post_attention(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(46)
    equal_decode = equal_chain = 0
    qg_tiles = qsa_block.QG_WIDTH // 32
    first = 7  # a window inside a wider shard, as the merged projection shard
    for i in range(8):
        attention = _up(
            mesh, torch.randn(1, 32, 1, 256, generator=g).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
        )
        qg_full = torch.randn(1, 1, 1, (first + qg_tiles + 3) * 32, generator=g) * 2
        qg = _up(mesh, qg_full, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        today = qsa_block.post_attention(attention, qg, qg_first=first)
        rows = qsa_rows.post_attention_rows(attention, qg, qg_first=first)
        equal_decode += int(_bits_equal(_dev0(today)[..., :1, :], _dev0(rows)[..., :1, :]))
        if first == 0:
            chain = qsa_rows.post_attention_rows_composed(attention, qg)
            equal_chain += int(_bits_equal(_dev0(today)[..., :1, :], _dev0(chain)[..., :1, :]))
            _free(chain)
        _free([today, rows, attention, qg])
    attention = _up(mesh, torch.randn(1, 32, 1, 256).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    qg = _up(
        mesh, torch.randn(1, 1, 1, (first + qg_tiles + 3) * 32).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
    )
    us_decode = _traced_us(mesh, lambda: _free(qsa_block.post_attention(attention, qg, qg_first=first)))
    us_rows = _traced_us(mesh, lambda: _free(qsa_rows.post_attention_rows(attention, qg, qg_first=first)))
    RESULTS["post_attention"] = {
        "bitwise_equal": f"{equal_decode}/8",
        "us_decode_program": us_decode,
        "us_rows_program": us_rows,
    }
    _dump()
    assert equal_decode == 8


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_reduce_w2n_is_widen_reduce_typecast(mesh_device):
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_reduce as rr

    mesh = mesh_device
    rr.prepare(mesh)
    shard = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]),
            [32, 160],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    g = torch.Generator().manual_seed(47)

    def partial(scale):
        host = (torch.randn(1, 1, 1, 8 * 2560, generator=g) * scale).to(torch.bfloat16)
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=shard,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )

    def today(p):
        wide = qsa_block.widen_partial(p)
        reduced = rr.row_reduce_or_reduce_scatter(wide, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.typecast(reduced, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free([wide, reduced])
        return out

    equal = total = 0
    for scale in (0.01, 1.0, 1.0, 30.0, 1.0, 1000.0, 1.0, 0.3):
        p = partial(scale)
        a = today(p)
        b = qsa_trim.reduce_w2n(p, reference=p)
        for da, db in zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)):
            total += 1
            equal += int(_bits_equal(ttnn.to_torch(da)[..., :1, :], ttnn.to_torch(db)[..., :1, :]))
        _free([a, b, p])
    p = partial(1.0)
    us_today = _traced_us(mesh, lambda: _free(today(p)))
    us_new = _traced_us(mesh, lambda: _free(qsa_trim.reduce_w2n(p, reference=p)))
    RESULTS["reduce_w2n"] = {
        "bitwise_equal": f"{equal}/{total}",
        "us_widen_reduce_typecast": us_today,
        "us_program": us_new,
    }
    _dump()
    assert equal == total
