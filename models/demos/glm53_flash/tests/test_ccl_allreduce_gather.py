# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode ledger B7: ``GLMMesh.all_reduce_cols`` as one all-gather of the column partials on a singleton outer dim +
``fast_reduce_nc`` over it (``GLM53_TP_ALLREDUCE=gather``) vs the generic ``ttnn.all_reduce`` composite (``ccl``,
the default) on a chip block, after tests/test_mesh_ccl.py: random bf16 partials per chip at the decode shapes
(4 and 32 users per row = 32 and 256 users on the 8x4; DRAM and L1 inputs), the fp32 torch sum as the reference,
both modes bit-close to it and to each other, the row's columns byte-identical, shape / dtype / memory config
unchanged, the gather path captured in a trace and replayed on new inputs, and the traced per-call time of both
modes (logged: the ledger expects ~23 us against the composite's 51).

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 pytest models/demos/glm53_flash/tests/test_ccl_allreduce_gather.py -s
"""

from __future__ import annotations

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

ROWS, COLS = (int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
H = 4096
N_REP = 50
MEM = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}


def _mesh(mesh_device) -> GLMMesh:
    m = GLMMesh.from_device(mesh_device)
    assert (m.rows, m.cols) == (ROWS, COLS), (m.rows, m.cols)
    return m


def _partials(B: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[ROWS, COLS, B, H] bf16 partials (chip (r, c) holds [r, c]) and the fp32 row sums [ROWS, B, H]."""
    torch.manual_seed(seed)
    parts = torch.randn(ROWS, COLS, B, H, dtype=torch.bfloat16)
    return parts, parts.float().sum(1)


def _upload(mesh: GLMMesh, parts: torch.Tensor, memory_config) -> ttnn.Tensor:
    return ttnn.from_torch(
        parts,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh.device,
        memory_config=memory_config,
        mesh_mapper=mesh.shard_2d(0, 1),
    )  # chip (r, c): [1, 1, B, H] = parts[r, c]


def _ulp(ref: torch.Tensor) -> float:
    """One bf16 ulp at the reference's largest magnitude (8 significant bits)."""
    return 2.0 ** (math.floor(math.log2(float(ref.abs().max()))) - 7)


def _check(mesh: GLMMesh, y: ttnn.Tensor, ref: torch.Tensor, x: ttnn.Tensor, label: str, max_ulps: float):
    """Every chip of row r holds the row's sum with ``x``'s shape / dtype / memory config, the columns byte-identical,
    within ``max_ulps`` bf16 ulps (of the reference's max magnitude) of the fp32 reference; the fp32 shards."""
    B = ref.shape[-2]
    assert tuple(y.shape) == tuple(x.shape), (label, tuple(y.shape), tuple(x.shape))
    assert y.dtype == x.dtype and y.layout == ttnn.TILE_LAYOUT, (label, y.dtype, y.layout)
    assert y.memory_config() == x.memory_config(), (label, y.memory_config(), x.memory_config())
    shards = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y)]
    ulp = _ulp(ref)
    worst_err, worst_pcc = 0.0, 1.0
    for r in range(ROWS):
        for c in range(COLS):
            s = shards[r * COLS + c]
            assert tuple(s.shape) == (1, 1, B, H), (label, r, c, tuple(s.shape))
            if c:
                assert torch.equal(s, shards[r * COLS]), (label, r, c)
            worst_err = max(worst_err, float((s[0, 0] - ref[r]).abs().max()) / ulp)
            worst_pcc = min(worst_pcc, pcc(ref[r], s[0, 0]))
    logger.info(f"{label}: worst |err| {worst_err:.2f} bf16 ulps vs the fp32 sum, worst pcc {worst_pcc:.7f}")
    assert worst_err <= max_ulps, (label, worst_err, max_ulps)
    assert worst_pcc > 0.99999, (label, worst_pcc)
    return shards


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=50_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("mem", ["dram", "l1"])
@pytest.mark.parametrize("B", [4, 32])
def test_all_reduce_cols_gather_vs_ccl(mesh_device, B, mem, monkeypatch):
    """Both modes on the same input: the composite within 4 ulps of the fp32 sum (bf16 ring hops), the gather path
    within 1 ulp (one fp32 sum rounded once), the two within 4 ulps of each other; an explicit ``memory_config``
    still lands where asked."""
    m = _mesh(mesh_device)
    parts, ref = _partials(B, seed=B)
    tt_x = _upload(m, parts, MEM[mem])
    monkeypatch.setenv("GLM53_TP_ALLREDUCE", "ccl")
    assert not m.tp_allreduce_gather
    y_ccl = m.all_reduce_cols(tt_x)
    ccl_sh = _check(m, y_ccl, ref, tt_x, f"B={B} {mem} ccl (reduce-scatter + all-gather composite)", max_ulps=4.0)
    monkeypatch.setenv("GLM53_TP_ALLREDUCE", "gather")
    assert m.tp_allreduce_gather and GLMMesh._gather_reduce_dim(tt_x) == 1
    y_g = m.all_reduce_cols(tt_x)
    g_sh = _check(m, y_g, ref, tt_x, f"B={B} {mem} gather (all_gather_cols dim 1 + fast_reduce_nc)", max_ulps=1.0)
    ulp = _ulp(ref)
    diff = max(float((a - b).abs().max()) for a, b in zip(ccl_sh, g_sh)) / ulp
    same = all(torch.equal(a, b) for a, b in zip(ccl_sh, g_sh))
    logger.info(f"B={B} {mem}: gather vs ccl: max |diff| {diff:.2f} bf16 ulps, byte-identical {same}")
    assert diff <= 4.0, diff
    y_d = m.all_reduce_cols(tt_x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert y_d.memory_config().buffer_type == ttnn.BufferType.DRAM and tuple(y_d.shape) == tuple(tt_x.shape)
    for t in (y_ccl, y_g, y_d):
        ttnn.deallocate(t)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=50_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
def test_all_reduce_cols_gather_in_trace(mesh_device, B, monkeypatch):
    """After test_mesh_ccl.py::test_collectives_in_trace for the gather path: one eager warm-up (compiles the gather
    and the reduce), then captured and replayed on new inputs through a persistent input tensor; at B=4 the
    preallocated reduce output (the logical [1,1,4,H] shape) is allocated inside the capture."""
    monkeypatch.setenv("GLM53_TP_ALLREDUCE", "gather")
    m = _mesh(mesh_device)
    parts, ref = _partials(B, seed=100 + B)
    tt_x = _upload(m, parts, ttnn.DRAM_MEMORY_CONFIG)  # persistent input
    y = m.all_reduce_cols(tt_x)
    _check(m, y, ref, tt_x, f"B={B} gather eager", max_ulps=1.0)
    ttnn.deallocate(y)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y_tr = m.all_reduce_cols(tt_x)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for step in range(3):
        parts_s, ref_s = _partials(B, seed=1000 + 10 * B + step)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(parts_s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=m.shard_2d(0, 1)), tt_x
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        _check(m, y_tr, ref_s, tt_x, f"B={B} gather trace replay {step}", max_ulps=1.0)
    ttnn.release_trace(mesh_device, tid)


def _timed(mesh_device, fn, label) -> float:
    out = fn()  # eager warm-up (compile)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(N_REP):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / N_REP * 1e6
    logger.info(f"CCL {label}: {us:.1f} us per call (traced replay)")
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(out)
    return us


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=50_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_all_reduce_cols_modes_perf(mesh_device, monkeypatch):
    """Traced per-call time of both modes at the decode shapes (after tests/test_ccl_perf.py): a table for the
    ledger, no assertion on the ratio (the block's 4 columns are the 8x4's: the column collective is the same)."""
    m = _mesh(mesh_device)
    table = []
    for B in (4, 32):
        parts, _ = _partials(B, seed=B)
        tt_x = _upload(m, parts, ttnn.DRAM_MEMORY_CONFIG)
        for mode in ("ccl", "gather"):
            monkeypatch.setenv("GLM53_TP_ALLREDUCE", mode)
            us = _timed(mesh_device, lambda: m.all_reduce_cols(tt_x), f"B={B} all_reduce_cols {mode} [1,1,{B},{H}]")
            table.append((mode, B, us))
    logger.info("| all_reduce_cols mode | users/row | us per call (traced replay) |")
    logger.info("|---|---|---|")
    for mode, B, us in table:
        logger.info(f"| {mode} | {B} | {us:.1f} |")
