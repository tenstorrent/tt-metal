# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_EDGE_TRIM cuts on the TP=8 mesh, decode operands: bit-for-bit against the chains they replace, and the traced per-call time of both.  Run: QWEN38_FUSED_DEVICE_TEST=1 pytest <file> -s -p no:cacheprovider -o addopts=
(the MESH_DEVICE / TT_VISIBLE_DEVICES environment of the model's TP=8 runs)."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import edge_trim, row_gather, row_reduce

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="needs the TP=8 mesh")
REPLAYS = int(os.environ.get("X48_REPLAYS", "200"))
OUT = os.environ.get("X48_OUT")
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
    print("X48RESULT", json.dumps(RESULTS), flush=True)
    if OUT:
        with open(OUT, "w") as f:
            json.dump(RESULTS, f, indent=1)


PARAMS = [
    pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True),
    pytest.mark.parametrize(
        "device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 8 * 1024 * 1024}], indirect=True
    ),
]


def _mesh_test(fn):
    for mark in reversed(PARAMS):
        fn = mark(fn)
    return fn


def _shard3(mesh, host, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=demo.MESH_SHAPE, dims=(None, 3)),
    )


def _all_devices_equal(a, b, rows):
    ok = 0
    for da, db in zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)):
        ok += int(_bits_equal(ttnn.to_torch(da)[..., :rows, :], ttnn.to_torch(db)[..., :rows, :]))
    return ok, len(ttnn.get_device_tensors(a))


@_mesh_test
def test_embed_reduce_is_all_reduce_plus_partition(mesh_device):
    mesh = mesh_device
    row_reduce.prepare(mesh)
    g = torch.Generator().manual_seed(48)
    equal = total = 0
    for owner in (0, 3, 7, 5):
        host = torch.zeros(1, 1, 1, 8 * 2560)
        host[..., owner * 2560 : (owner + 1) * 2560] = torch.randn(1, 1, 1, 2560, generator=g) * 3
        host[..., owner * 2560] = -0.0  # a negative zero in the owner's row
        partial = _shard3(mesh, host)
        reduced = ttnn.all_reduce(
            partial, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Linear
        )
        want = ttnn.mesh_partition(reduced, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        got = row_reduce.row_reduce(partial, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ok, n = _all_devices_equal(want, got, 1)
        equal += ok
        total += n
        _free([partial, reduced, want, got])
    partial = _shard3(mesh, torch.randn(1, 1, 1, 8 * 2560))

    def chain():
        r = ttnn.all_reduce(
            partial, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Linear
        )
        _free(ttnn.mesh_partition(r, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        _free(r)

    us_chain = _traced_us(mesh, chain)
    us_new = _traced_us(
        mesh, lambda: _free(row_reduce.row_reduce(partial, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    )
    RESULTS["embed_reduce"] = {"bitwise_equal": f"{equal}/{total}", "us_chain": us_chain, "us_program": us_new}
    _dump()
    assert equal == total


@_mesh_test
@pytest.mark.parametrize("rows", [1])
def test_hidden_gather_is_all_gather(mesh_device, rows, monkeypatch):
    mesh = mesh_device
    row_gather.prepare(mesh, 10)
    monkeypatch.setenv(edge_trim.ENV, "1")
    monkeypatch.setenv(edge_trim.OPTS_ENV, "ple_gather")
    g = torch.Generator().manual_seed(49 + rows)
    equal = total = 0
    for _ in range(6):
        local = _shard3(mesh, torch.randn(1, 1, rows, 8 * 320, generator=g) * 2)
        want = ttnn.all_gather(local, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        got = edge_trim.hidden_gather(local, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ok, n = _all_devices_equal(want, got, rows)
        equal += ok
        total += n
        _free([local, want, got])
    local = _shard3(mesh, torch.randn(1, 1, rows, 8 * 320))
    us_chain = _traced_us(
        mesh, lambda: _free(ttnn.all_gather(local, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    )
    us_new = _traced_us(mesh, lambda: _free(edge_trim.hidden_gather(local, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
    RESULTS[f"hidden_gather_rows{rows}"] = {
        "bitwise_equal": f"{equal}/{total}",
        "us_all_gather": us_chain,
        "us_row_gather": us_new,
    }
    _dump()
    assert equal == total


@_mesh_test
def test_ple_kv_is_the_two_linears(mesh_device, monkeypatch):
    import types

    mesh = mesh_device
    monkeypatch.setenv(edge_trim.ENV, "1")
    monkeypatch.setenv(edge_trim.OPTS_ENV, "ple_kv")
    g = torch.Generator().manual_seed(50)
    key = _shard3(mesh, torch.randn(1, 1, 2560, 8 * 1280, generator=g) * 0.05)
    value = _shard3(mesh, torch.randn(1, 1, 2560, 8 * 320, generator=g) * 0.05)
    config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    module = types.SimpleNamespace(
        mesh_device=mesh, weights=types.SimpleNamespace(key=key, value=value), compute_config=config
    )
    edge_trim.build_kv(module)
    equal = total = 0
    for scale in (1.0, 0.1, 10.0, 1.0):
        emb = ttnn.from_torch(
            (torch.randn(1, 1, 1, 2560, generator=g) * scale).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        want_k = ttnn.linear(emb, key, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=config)
        want_v = ttnn.linear(emb, value, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=config)
        got_k, got_v = edge_trim.ple_kv(module, emb)
        for w, gt in ((want_k, got_k), (want_v, got_v)):
            ok, n = _all_devices_equal(w, gt, 1)
            equal += ok
            total += n
        _free([emb, want_k, want_v, got_k, got_v])
    emb = ttnn.from_torch(
        torch.randn(1, 1, 1, 2560).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def chain():
        _free(
            [
                ttnn.linear(emb, key, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=config),
                ttnn.linear(emb, value, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=config),
            ]
        )

    us_chain = _traced_us(mesh, chain)
    us_new = _traced_us(mesh, lambda: _free(list(edge_trim.ple_kv(module, emb))))
    RESULTS["ple_kv"] = {"bitwise_equal": f"{equal}/{total}", "us_two_linears": us_chain, "us_merged": us_new}
    _dump()
    assert equal == total
