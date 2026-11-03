# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""main_tail_rows on the 1x8 mesh: QWEN38_QSA_ROWS_KV_FAST 0 vs 1 (group 2 only, see test_qsa_verify_live_device.py).
test_main_tail_kv_fast: query output and the whole KV cache must be bit-equal for the pages a later read can see
(rows <= P + R - 1 and everything outside the two touched blocks; the zero tail past P + R - 1 may differ).
test_main_tail_zone_run: a few untraced calls for the device profiler (QWEN38_FUSED_ZONES=1)."""

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
CASES = pytest.mark.parametrize(
    "position,live", [(100, 5), (100, 1), (100, 3), (100, 8), (126, 5), (127, 1), (124, 8), (128, 5), (31, 32), (95, 3)]
)
ENV = "QWEN38_QSA_ROWS_KV_FAST"


def inputs(mesh, position, live, seed):
    g = torch.Generator().manual_seed(seed)
    heads, hd = qsa_block.LOCAL_HEADS, qsa_block.HEAD_DIM
    bf = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16)
    tile = lambda t: up(mesh, t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    common = dict(
        qg_ws=tile(bf(1, 1, 32, 2 * heads * hd)),
        k_ws=tile(bf(1, 1, 32, hd)),
        v_ws=tile(bf(1, 1, 32, hd)),
        position=up(mesh, torch.tensor([position]).reshape(1, 1, 1, 1), ttnn.uint32),
        q_norm=tile(bf(1, 1, 1, hd).abs() + 0.5),
        k_norm=tile(bf(1, 1, 1, hd).abs() + 0.5),
        cos=tile(torch.cos(bf(1, 1, 32, 64).float()).to(torch.bfloat16)),
        sin=tile(torch.sin(bf(1, 1, 32, 64).float()).to(torch.bfloat16)),
    )
    rows_u32 = up(mesh, torch.tensor([live]).reshape(1, 1, 1, 1), ttnn.uint32)
    return common, rows_u32, bf(1, 1, 512, 512)


@MESH
@PARAMS
@CASES
def test_main_tail_kv_fast(mesh_device, monkeypatch, position, live):
    mesh = mesh_device
    common, rows_u32, cache_h = inputs(mesh, position, live, 300 + live + position)
    # three configs: 0 = flag off; 1 = KV fast only (no live_rows given); 2 = KV fast + live lanes
    caches = {on: up(mesh, cache_h, ttnn.bfloat16) for on in (0, 1, 2)}
    outs, times = {}, {}
    for on in (0, 1, 2):
        monkeypatch.setenv(ENV, "0" if on == 0 else "1")
        kw = {"live_rows": live} if on == 2 else {}
        call = lambda: qsa_rows.main_tail_rows(**common, kv_cache=caches[on], rows=rows_u32, single_row=False, **kw)
        outs[on] = host(call())
        times[on] = traced_us(mesh, call)[0]
    equal = torch.equal(bits(outs[0]), bits(outs[1]))
    q0, q2 = outs[0].reshape(1, 32, 32, 512), outs[2].reshape(1, 32, 32, 512)
    live_equal = torch.equal(bits(q0[:, :, :live]), bits(q2[:, :, :live]))
    dead_zero = bool((q2[:, :, live:].float() == 0).all())
    c0, c1, c2 = (bits(host(caches[k])).reshape(-1, 512) for k in (0, 1, 2))
    # rows a later read can name: below P + R (written this pass or older); all device copies are equal copies
    seen = position + live
    far = (position // 32 + 2) * 32
    seen_equal = all(torch.equal(c0[:seen], c[:seen]) and torch.equal(c0[far:], c[far:]) for c in (c1, c2))
    full_equal = torch.equal(c0, c1) and torch.equal(c0, c2)
    print(
        f"QSA_KV_FAST P={position} live={live} query_equal={equal} live_lanes_equal={live_equal} dead_lanes_zero={dead_zero} "
        f"visible_cache_equal={seen_equal} full_cache_equal={full_equal}",
        flush=True,
    )
    print(
        "QSA_KV_TIMING "
        + json.dumps(
            {"P": position, "live": live, "flag0_us": times[0], "flag1_us": times[1], "flag1_live_us": times[2]}
        ),
        flush=True,
    )
    assert equal and seen_equal and live_equal and (dead_zero or live == 32)


@MESH
@PARAMS
def test_main_tail_zone_run(mesh_device):
    mesh = mesh_device
    position, live = int(os.environ.get("ZR_POS", "100")), int(os.environ.get("ZR_LIVE", "5"))
    common, rows_u32, cache_h = inputs(mesh, position, live, 7)
    cache = up(mesh, cache_h, ttnn.bfloat16)
    for _ in range(int(os.environ.get("ZR_CALLS", "4"))):
        out = qsa_rows.main_tail_rows(**common, kv_cache=cache, rows=rows_u32, single_row=False)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(out)
