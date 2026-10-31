# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_MT_ROW=1 (main_tail's one-row staging core) against the default main_tail, one lane, on the eight-die mesh.

 * test_mt_row_bitwise_steps: from one initial state, both programs run decode positions 120..170 (crossing the block
   starts 128 and 160) on random inputs that include -0.0; after every step the sparse query, the KV cache rows 0..P and
   the staging row P % 32 are bit-identical.
 * test_mt_row_value_table: the default program's output bits for a new row of special bf16 values (-0.0, denormals,
   Inf, NaNs, ...), and the one-row program's bits for the same row (the table is printed as MT_ROW_TABLE lines).
 * test_mt_row_timing: traced slope per call of both programs.
Run (group 2): MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=16,17,18,19,23,22,21,20 QWEN38_FUSED_DEVICE_TEST=1 flock
/home/ttuser/atupe/device.lock.g2 python_env/bin/pytest <this file> -s -p no:cacheprovider -o addopts=
"""

import json
import os
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the held mesh")
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 20_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
CACHE_ROWS = 512
TRACE_SHORT, TRACE_LONG, REPLAYS, REPEATS = 4, 12, 20, 3
SPECIAL = [
    0x8000,  # -0.0
    0x0000,  # +0.0
    0x0001,
    0x8001,
    0x007F,
    0x807F,  # denormals
    0x0080,
    0x8080,  # smallest normals
    0x7F80,
    0xFF80,  # +Inf, -Inf
    0x7FC0,  # the quiet NaN
    0xFFC0,  # a NaN with the sign bit
    0x7F81,  # a signalling NaN pattern
    0x7FC1,  # a NaN with payload
    0x7FFF,
    0xFFFF,
    0x3F80,
    0xBF80,
    0x7F7F,
    0xFF7F,  # +-1.0, +-max finite
]


def bf16_from_bits(bits):
    return torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)


def up(mesh, tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    if dtype == ttnn.uint32:
        tensor = tensor.to(torch.int64).to(torch.uint32)
    # tilized on the host and then copied: a device-side tilize flushes denormals and turns NaN into Inf
    on_host = ttnn.from_torch(tensor, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    return ttnn.to_device(on_host, mesh, memory_config=DRAM)


def host(tensor):
    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0].view(torch.int16), other.view(torch.int16)) for other in replicas[1:])
    return replicas[0]


def bits(t):
    return t.view(torch.int16).to(torch.int32) & 0xFFFF


@contextmanager
def mt_row(mode):
    old = os.environ.get(qsa_block.MT_ROW_ENV)
    os.environ[qsa_block.MT_ROW_ENV] = mode
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(qsa_block.MT_ROW_ENV, None)
        else:
            os.environ[qsa_block.MT_ROW_ENV] = old


class Inputs:
    """Device inputs of one decode step on one lane (position P)."""

    def __init__(self, mesh, generator, position, v_host=None, zero_sprinkle=True):
        heads, hd = qsa_block.LOCAL_HEADS, qsa_block.HEAD_DIM
        bf = lambda *shape: torch.randn(*shape, generator=generator).to(torch.bfloat16)
        tile = lambda t: up(mesh, t, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        def sprinkle(t, n):
            flat = t.view(-1)
            idx = torch.randint(0, flat.numel(), (n,), generator=generator)
            flat[idx] = bf16_from_bits([0x8000] * n)
            return t

        v = bf(1, 1, 1, hd) if v_host is None else v_host
        k = bf(1, 1, 1, hd)
        if zero_sprinkle and v_host is None:
            sprinkle(v, 12)
            sprinkle(k, 12)
        hit = torch.zeros(1, 1, 32, 1, dtype=torch.bfloat16)
        hit[0, 0, position % 32, 0] = 1.0
        self.position = SimpleNamespace(
            kv_block_start=up(mesh, torch.tensor([position & ~31]).reshape(1, 1, 1, 1), ttnn.uint32),
            kv_row_hit=tile(hit),
        )
        self.kwargs = dict(
            qg_ws=tile(bf(1, 1, 1, 2 * heads * hd)),
            k_ws=tile(k),
            v_ws=tile(v),
            position=self.position,
            q_norm=tile(bf(1, 1, 1, hd).abs() + 0.5),
            k_norm=tile(bf(1, 1, 1, hd).abs() + 0.5),
            cos=tile(torch.cos(bf(1, 1, 1, 64).float()).to(torch.bfloat16)),
            sin=tile(torch.sin(bf(1, 1, 1, 64).float()).to(torch.bfloat16)),
        )
        self.v_host = v


class State:
    def __init__(self, mesh, staging_host, cache_host):
        self.staging = up(mesh, staging_host, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.cache = up(mesh, cache_host, ttnn.bfloat16)

    def run(self, inputs):
        return qsa_block.main_tail(**inputs.kwargs, staging=self.staging, kv_cache=self.cache)


def initial_state(generator, block_start):
    staging = torch.randn(1, 1, 32, 512, generator=generator).to(torch.bfloat16)
    cache = torch.randn(1, 1, CACHE_ROWS, 512, generator=generator).to(torch.bfloat16)
    cache[0, 0, block_start : block_start + 32] = staging[0, 0]  # the cache block == the staging block, as in decode
    return staging, cache


@MESH
@PARAMS
def test_mt_row_bitwise_steps(mesh_device):
    mesh = mesh_device
    generator = torch.Generator().manual_seed(2026)
    staging_h, cache_h = initial_state(generator, 96)
    old, new = State(mesh, staging_h, cache_h), State(mesh, staging_h, cache_h)
    for p in range(120, 171):
        inputs = Inputs(mesh, generator, p)
        with mt_row("0"):
            q_old = host(old.run(inputs))
        with mt_row("1"):
            q_new = host(new.run(inputs))
        assert torch.equal(bits(q_old), bits(q_new)), f"query differs at P={p}"
        c_old, c_new = host(old.cache)[0, 0, : p + 1], host(new.cache)[0, 0, : p + 1]
        assert torch.equal(bits(c_old), bits(c_new)), f"cache rows 0..P differ at P={p}"
        s_old, s_new = host(old.staging)[0, 0, p % 32], host(new.staging)[0, 0, p % 32]
        assert torch.equal(bits(s_old), bits(s_new)), f"staging row differs at P={p}"
        assert not (bits(c_new[p]) == 0x8000).any()
    print("MT_ROW_BITWISE_OK positions 120..170")


@MESH
@PARAMS
def test_mt_row_value_table(mesh_device):
    mesh = mesh_device
    generator = torch.Generator().manual_seed(77)
    p = 10
    cols = [SPECIAL[i % len(SPECIAL)] for i in range(256)]
    v = bf16_from_bits(cols).reshape(1, 1, 1, 256)
    assert torch.equal(bits(v.reshape(-1)), torch.tensor(cols, dtype=torch.int32)), "host bf16 view changed bits"
    inputs = Inputs(mesh, generator, p, v_host=v, zero_sprinkle=False)
    zeros_s, zeros_c = torch.zeros(1, 1, 32, 512, dtype=torch.bfloat16), torch.zeros(
        1, 1, CACHE_ROWS, 512, dtype=torch.bfloat16
    )
    old, new = State(mesh, zeros_s, zeros_c), State(mesh, zeros_s, zeros_c)
    assert torch.equal(bits(host(inputs.kwargs["v_ws"])[0, 0, 0, :256]), torch.tensor(cols, dtype=torch.int32))
    with mt_row("0"):
        old.run(inputs)
    with mt_row("1"):
        new.run(inputs)
    row_old, row_new = bits(host(old.cache)[0, 0, p]), bits(host(new.cache)[0, 0, p])
    stg_old, stg_new = bits(host(old.staging)[0, 0, p % 32]), bits(host(new.staging)[0, 0, p % 32])
    table = {}
    for i, c in enumerate(cols):
        table.setdefault(c, set()).add((int(row_old[i]), int(row_new[i]), int(stg_old[i]), int(stg_new[i])))
    for c in SPECIAL:
        print(
            "MT_ROW_TABLE in=0x%04X (old cache, new cache, old staging, new staging) = %s"
            % (c, sorted("0x%04X 0x%04X 0x%04X 0x%04X" % t for t in table[c]))
        )
    assert torch.equal(row_old, row_new), "cache row differs (see MT_ROW_TABLE)"
    assert torch.equal(stg_old, stg_new), "staging row differs (see MT_ROW_TABLE)"


@MESH
@PARAMS
def test_mt_row_all_bf16_values(mesh_device):
    """Every one of the 65536 bf16 bit patterns as a value of the new row (256 per call, in the value half): the one-row
    program's cache and staging bits equal the default program's.  MT_ROW_SWEEP_DUMP=<file> saves the default
    program's input -> output map (int32 pairs)."""

    mesh = mesh_device
    generator = torch.Generator().manual_seed(88)
    p = 10
    zeros_s = torch.zeros(1, 1, 32, 512, dtype=torch.bfloat16)
    zeros_c = torch.zeros(1, 1, CACHE_ROWS, 512, dtype=torch.bfloat16)
    old, new = State(mesh, zeros_s, zeros_c), State(mesh, zeros_s, zeros_c)
    out_old = torch.zeros(65536, dtype=torch.int32)
    bad = 0
    for chunk in range(256):
        values = list(range(chunk * 256, chunk * 256 + 256))
        inputs = Inputs(mesh, generator, p, v_host=bf16_from_bits(values).reshape(1, 1, 1, 256), zero_sprinkle=False)
        with mt_row("0"):
            old.run(inputs)
        with mt_row("1"):
            new.run(inputs)
        row_old, row_new = bits(host(old.cache)[0, 0, p]), bits(host(new.cache)[0, 0, p])
        stg_old, stg_new = bits(host(old.staging)[0, 0, p % 32]), bits(host(new.staging)[0, 0, p % 32])
        out_old[chunk * 256 : chunk * 256 + 256] = row_old[:256]
        diff = (row_old != row_new) | (stg_old != stg_new)
        if diff.any():
            bad += int(diff.sum())
            if bad <= 40:
                for i in diff.nonzero().reshape(-1).tolist():
                    print(
                        "MT_ROW_SWEEP_DIFF in=0x%04X old=0x%04X new=0x%04X"
                        % (values[i] if i < 256 else -1, row_old[i], row_new[i])
                    )
    if os.environ.get("MT_ROW_SWEEP_DUMP"):
        torch.save(out_old, os.environ["MT_ROW_SWEEP_DUMP"])
    print("MT_ROW_SWEEP_MISMATCHES", bad)
    assert bad == 0


def traced_us(mesh, call):
    def capture(ops):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(ops):
            ttnn.deallocate(call())
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        return tid

    def wall(tid):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = 1e30
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, time.perf_counter() - t0)
        return best / REPLAYS * 1e6

    ttnn.deallocate(call())
    short, long_ = capture(TRACE_SHORT), capture(TRACE_LONG)
    try:
        ts, tl = wall(short), wall(long_)
    finally:
        ttnn.release_trace(mesh, short)
        ttnn.release_trace(mesh, long_)
    return round((tl - ts) / (TRACE_LONG - TRACE_SHORT), 2), round(tl / TRACE_LONG, 2)


@MESH
@PARAMS
def test_mt_row_timing(mesh_device):
    mesh = mesh_device
    generator = torch.Generator().manual_seed(5)
    report = {}
    for p in (130, 250):
        staging_h, cache_h = initial_state(generator, p & ~31)
        state = State(mesh, staging_h, cache_h)
        inputs = Inputs(mesh, generator, p)
        for mode in ("0", "1", "0", "1"):
            with mt_row(mode):
                report.setdefault(f"P{p}_mode{mode}", []).append(traced_us(mesh, lambda: state.run(inputs)))
    print("MT_ROW_TIMING " + json.dumps(report))
