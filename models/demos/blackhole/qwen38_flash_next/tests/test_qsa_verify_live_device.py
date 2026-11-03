# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_VERIFY_LIVE=1 on the 1x8 mesh: the verify tile score_merge and selection_row work on the live
rows only.  For live rows 1, 3, 5, 8 (and 32 for the no-op case) each program runs with the flag 0 and 1 on the same inputs;
the LIVE rows (and the KV cache) must be bit-identical; the dead rows only have to be finite / the documented constants.
Traced per-call time (slope of two trace lengths) is printed as QSA_LIVE_TIMING lines.
Run (group 2): MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=16,17,18,19,23,22,21,20 QWEN38_FUSED_DEVICE_TEST=1 flock
/home/ttuser/atupe/device.lock.g2 python_env/bin/pytest <this file> -s -p no:cacheprovider -o addopts=
"""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_qsa_mt_row_device import (
    DEVICE_PARAMS,
    MESH_SHAPE,
    bits,
    host,
    traced_us,
    up,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block, qsa_rows

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the held mesh")
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 20_000_000}], indirect=True)
LIVE = pytest.mark.parametrize("live", [5, 1, 3, 8])
WIDTH = 8192
CHUNKS = WIDTH // 1024
MASK = -3.3895313892515355e38


def flag(monkeypatch, on):
    monkeypatch.setenv(qsa_block.QSA_VERIFY_LIVE_ENV, "1" if on else "0")


def report(name, live, us0, us1):
    print("QSA_LIVE_TIMING " + json.dumps({"op": name, "live": live, "flag0_us": us0, "flag1_us": us1}), flush=True)


@MESH
@PARAMS
@LIVE
def test_score_merge_live(mesh_device, monkeypatch, live):
    mesh, g = mesh_device, torch.Generator().manual_seed(100 + live)
    pages = (torch.randn(8 * live * CHUNKS, 1024, generator=g).abs() * 4).to(torch.bfloat16).reshape(1, 1, -1, 1024)
    gathered = up(mesh, pages, ttnn.bfloat16)
    mask_t = (
        torch.where(torch.rand(32, WIDTH, generator=g) < 0.3, MASK, 0.0).to(torch.bfloat16).reshape(1, 1, 32, WIDTH)
    )
    mask = up(mesh, mask_t, ttnn.bfloat16)
    outs, times = {}, {}
    for on in (0, 1):
        flag(monkeypatch, on)
        outs[on] = host(qsa_block.score_merge(gathered, mask, out_rows=32))
        times[on] = traced_us(mesh, lambda: qsa_block.score_merge(gathered, mask, out_rows=32))[0]
    equal = torch.equal(bits(outs[0][..., :live, :]), bits(outs[1][..., :live, :]))
    finite = bool(torch.isfinite(outs[1][..., live:, :].float()).all())
    print(f"QSA_LIVE_SCORE_MERGE live={live} live_rows_equal={equal} dead_finite={finite}", flush=True)
    report("score_merge", live, times[0], times[1])
    assert equal and finite


@MESH
@PARAMS
@LIVE
def test_selection_live(mesh_device, monkeypatch, live):
    mesh, g = mesh_device, torch.Generator().manual_seed(200 + live)
    ids = torch.randint(0, 2**15, (32, 512), generator=g)
    offsets = torch.randint(0, 1000, (32, 2048), generator=g)
    keep = torch.where(torch.rand(32, 2080, generator=g) < 0.8, 0xFFFFFFFF, 0)
    fill = torch.where(keep == 0, torch.randint(0, 2**15, (32, 2080), generator=g), torch.zeros_like(keep))
    sentinel = torch.randint(0, 2**32, (1, 32), generator=g)
    args = (
        up(mesh, ids.reshape(1, 1, 32, 512), ttnn.uint32),
        up(mesh, sentinel.reshape(1, 1, 1, 32), ttnn.uint32),
        up(mesh, offsets.reshape(1, 1, 32, 2048), ttnn.uint32),
        up(mesh, keep.reshape(1, 1, 32, 2080), ttnn.uint32),
        up(mesh, fill.reshape(1, 1, 32, 2080), ttnn.uint32),
    )
    outs, times = {}, {}
    for on in (0, 1):
        flag(monkeypatch, on)
        outs[on] = host(qsa_rows.selection_rows(*args, live_rows=live)).to(torch.int64) & 0xFFFFFFFF
        times[on] = traced_us(mesh, lambda: qsa_rows.selection_rows(*args, live_rows=live))[0]
    outs[0], outs[1] = outs[0].reshape(32, 2080), outs[1].reshape(32, 2080)
    equal = torch.equal(outs[0][:live], outs[1][:live])
    dead = outs[1][live:]
    dead_ok = bool((dead[:, 1:] == 0xFFFFFFFF).all() and (dead[:, 0] == 0).all())
    print(f"QSA_LIVE_SELECTION live={live} live_rows_equal={equal} dead_one_token_row={dead_ok}", flush=True)
    report("selection_rows", live, times[0], times[1])
    assert equal and dead_ok
