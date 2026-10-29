# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_ROUTER_L1 (the router linear's bf16 L1 shard straight into the top-k) on the TP=8 mesh, decode operands: bit-for-bit against the chains they replace, and the traced per-call time of both.  Run: QWEN38_FUSED_DEVICE_TEST=1 pytest <file> -s -p no:cacheprovider -o addopts=
(the MESH_DEVICE / TT_VISIBLE_DEVICES environment of the model's TP=8 runs)."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="needs the TP=8 mesh")
REPLAYS = int(os.environ.get("X54_REPLAYS", "200"))
OUT = os.environ.get("X54_OUT")
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
    print("X54RESULT", json.dumps(RESULTS), flush=True)
    if OUT:
        with open(OUT, "w") as f:
            json.dump(RESULTS, f, indent=1)


SENTINEL_SCORE, SENTINEL_INDEX = 12345.0, 0xFFFF


def _sentinel_outputs(mesh):
    scores = ttnn.from_torch(
        torch.full((1, 1, 1, 10), SENTINEL_SCORE).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    indices = ttnn.from_torch(
        torch.full((1, 1, 1, 10), SENTINEL_INDEX, dtype=torch.int32),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return scores, indices


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 8 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("form", ["sort_core", "fast"])
def test_router_tail_on_the_l1_shard_is_the_drained_fp32(mesh_device, form, monkeypatch):
    mesh = mesh_device
    monkeypatch.setenv("QWEN38_ROUTER_L1", "1")
    monkeypatch.setenv(router_tail_sc.FAST_ENV, "1" if form == "fast" else "0")
    router_tail_sc.prepare(mesh)
    g = torch.Generator().manual_seed(54)
    act_config, program_config = dm.dram_sharded_matmul_configs(mesh, 2560, 512, num_cores=5)
    compute = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    # small-integer operands: the logits are integers, so ties are everywhere (also among the top 10)
    weight_host = torch.randint(-2, 3, (1, 1, 2560, 512), generator=g).float()
    weight = ttnn.from_torch(
        weight_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dm.dram_sharded_weight_memory_config(mesh, 2560, 512),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    equal = total = 0
    per_row_unwritten = 0
    for i in range(48):
        scale = (1.0, 0.05, 0.002)[i % 3]
        hidden = (torch.randint(-3, 4, (1, 1, 1, 2560), generator=g).float() * scale).to(torch.bfloat16)
        act = ttnn.from_torch(
            hidden,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        act = ttnn.to_memory_config(act, act_config)
        shard = ttnn.linear(
            act,
            weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute,
        )
        drained = ttnn.to_memory_config(shard, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
        want = _sentinel_outputs(mesh)
        got = _sentinel_outputs(mesh)
        router_tail_sc.router_tail_sc_into(drained, *want)
        router_tail_sc.router_tail_sc_into(shard, *got)
        for w, gt in zip(want, got):
            for dw, dg in zip(ttnn.get_device_tensors(w), ttnn.get_device_tensors(gt)):
                a, b = ttnn.to_torch(dw), ttnn.to_torch(dg)
                total += 1
                equal += int(_bits_equal(a, b))
                per_row_unwritten += int(
                    (b.float() == (SENTINEL_SCORE if b.dtype == torch.bfloat16 else float(SENTINEL_INDEX))).any()
                )
        _free([act, shard, drained, *want, *got])
    # timing: S2I + router tail on the fp32 drain vs the router tail on the shard
    act = ttnn.to_memory_config(
        ttnn.from_torch(
            torch.randint(-3, 4, (1, 1, 1, 2560)).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        ),
        act_config,
    )
    shard = ttnn.linear(
        act,
        weight,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=program_config,
        compute_kernel_config=compute,
    )
    outs = _sentinel_outputs(mesh)

    def with_drain():
        d = ttnn.to_memory_config(shard, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
        router_tail_sc.router_tail_sc_into(d, *outs)
        ttnn.deallocate(d)

    us_drain = _traced_us(mesh, with_drain)
    us_shard = _traced_us(mesh, lambda: router_tail_sc.router_tail_sc_into(shard, *outs))
    RESULTS[form] = {
        "bitwise_equal": f"{equal}/{total}",
        "unwritten_outputs": per_row_unwritten,
        "us_drain_plus_tail": us_drain,
        "us_tail_on_shard": us_shard,
    }
    _dump()
    assert equal == total and per_row_unwritten == 0
