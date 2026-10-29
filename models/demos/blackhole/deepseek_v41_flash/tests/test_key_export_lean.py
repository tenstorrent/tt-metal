# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41_KEY_EXPORT_LEAN: the key hand-off (``Model._export_user_keys``) writes each piece in place at its own offset (the slab viewed as [UD * NP, 1, P, 128], ``fill_cache``)
instead of building the concat of the slab prefix with the piece. (a) CPU: the piece / offset logic. (b) device (skipped without one): old export vs lean export on small slabs, the
resulting slab contents (every row / user / entry) must be BIT-IDENTICAL, and the slab buffer must not move.

    pytest models/demos/blackhole/deepseek_v41_flash/tests/test_key_export_lean.py -s
"""

import os
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import dsv41_model as dm


def _env(monkeypatch, lean, piece=None):
    monkeypatch.delenv("DSV41_KEY_EXPORT_LEAN", raising=False)
    monkeypatch.delenv("DSV41_KEY_EXPORT_PIECE", raising=False)
    if lean:
        monkeypatch.setenv("DSV41_KEY_EXPORT_LEAN", "1")
    if piece:
        monkeypatch.setenv("DSV41_KEY_EXPORT_PIECE", str(piece))


def test_piece_default_off(monkeypatch):
    _env(monkeypatch, False)
    assert dm.key_export_piece(1048640) == 65536
    assert dm.key_export_piece(8192) == 8192
    assert not dm.key_export_lean(1048640)


def test_piece_lean_divides(monkeypatch):
    _env(monkeypatch, True)
    for NA in (1048640, 1048576, 262176, 524320, 65536, 8192, 131104):
        P = dm.key_export_piece(NA)
        assert P % 32 == 0 and P <= 65536
        if dm.key_export_lean(NA):
            assert NA % P == 0
            # pieces tile the slab exactly once: no overlapped / shifted last piece
            js = [min(pj * P, NA - P) for pj in range(-(-NA // P))]
            assert js == list(range(0, NA, P))
    assert dm.key_export_piece(1048640) == 36160 and dm.key_export_lean(1048640)
    assert dm.key_export_piece(1048576) == 65536
    assert not dm.key_export_lean(262176)  # 2^5 * 3 * 2731: no divisor in [4096, 65536]: the old export stays


def test_view_blocks_cover_slab():
    NA, P, UD = 1048640, 36160, 2
    NP = NA // P
    seen = set()
    for u in range(UD):
        for j0 in range(0, NA, P):
            b = u * NP + j0 // P
            assert 0 <= b < UD * NP
            seen.add(b)
            assert (b * P) == (u * NA + j0)  # the view block starts at the user's row j0 of the flat slab
    assert len(seen) == UD * NP


def _stub(md, rows):
    m = types.SimpleNamespace(md=md, rows=rows, log=print)
    m._mp = lambda: ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, md.shape[1]))
    m.log_dram = lambda tag: None
    m._export_state = types.MethodType(dm.Model._export_state, m)
    m.export = types.MethodType(dm.Model._export_user_keys, m)
    return m


def _slab(md, k):
    ttnn.synchronize_device(md)
    return torch.stack([ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(k)])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 5_000_000}, id="plain")],
    indirect=True,
)
@pytest.mark.parametrize(
    "NA,piece,UD",
    [(8192, 2048, 2), (16384, 8192, 2), (32768, 8192, 1), (12288, 8192, 2), (6144, 4096, 2)],
    ids=["4pieces_U2", "2pieces_U2", "4pieces_U1", "div_shift_U2", "no_divisor_fallback_U2"],
)
@torch.no_grad()
def test_lean_equals_old(mesh_device, monkeypatch, NA, piece, UD):
    md = mesh_device
    rows = md.shape[0]
    IDIM, Up = 128, 2
    KL = NA + 96
    g = torch.Generator().manual_seed(11)
    kc0 = torch.randn(rows * UD, 1, NA, IDIM, generator=g).to(torch.bfloat16)
    fifo = torch.randn(rows * Up, 1, KL, IDIM, generator=g).to(torch.bfloat16)
    mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, md.shape[1]))

    def mk(t, dt):
        return ttnn.from_torch(
            t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mp
        )

    ix = types.SimpleNamespace(sh=None, keys=mk(fifo, ttnn.bfloat16), KL=KL, KL_full=KL)
    slabs, addrs = {}, {}
    # (off, n, user, i_src, mesh rows): partial, full-length, other user / slot, no rows (warm)
    calls = [
        (32, min(NA, 3000), UD - 1, 1, [1, 3]),
        (0, NA, 0, 0, [0, 1, 2, 3]),
        (64, NA // 2 + 32, 0, 1, [2]),
        (0, 0, 0, 0, []),
    ]
    for lean in (False, True):
        _env(monkeypatch, lean, piece)
        k = mk(kc0, ttnn.bfloat8_b)
        addrs[lean] = k.buffer_address()
        m = _stub(md, rows)
        for off, n, u, i_src, sel in calls:
            m.export(ix, k, i_src, u, sel, off, n)
        assert k.buffer_address() == addrs[lean]
        slabs[lean] = _slab(md, k)
        ttnn.deallocate(k)
    assert torch.equal(slabs[False], slabs[True]), "lean export differs from the concat export"
    # and the export did something (the slab is not the initial one)
    k0 = mk(kc0, ttnn.bfloat8_b)
    assert not torch.equal(slabs[True], _slab(md, k0))


@pytest.mark.skipif(
    os.environ.get("DSV41_KX_MEASURE") != "1", reason="DSV41_KX_MEASURE=1: transient measurement at a 1M slab"
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 5_000_000}, id="plain")],
    indirect=True,
)
@pytest.mark.parametrize("UD", [1, 2, 4])
@torch.no_grad()
def test_measure_transient(mesh_device, monkeypatch, UD):
    """Allocated DRAM per bank at the MEMLOG point inside the export (temporaries live), per piece, one full-length export of a 1,048,640-row slab (the 1M build), old vs lean."""
    import re

    md = mesh_device
    rows = md.shape[0]
    NA, KL, IDIM, Up = 1048640, 1048640 + 96, 128, 1
    mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, md.shape[1]))

    def mk(t, dt):
        return ttnn.from_torch(
            t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mp
        )

    ix = types.SimpleNamespace(
        sh=None, keys=mk(torch.randn(rows * Up, 1, KL, IDIM).to(torch.bfloat16), ttnn.bfloat16), KL=KL, KL_full=KL
    )
    monkeypatch.setenv("DSV41_MEMLOG", "1")
    res = {}
    for lean in (False, True):
        _env(monkeypatch, lean)
        monkeypatch.setenv("DSV41_MEMLOG", "1")
        k = mk(torch.randn(rows * UD, 1, NA, IDIM).to(torch.bfloat16), ttnn.bfloat8_b)
        m = _stub(md, rows)
        lines = []
        m.log = lines.append
        m.log_dram = types.MethodType(dm.Model.log_dram, m)
        m.export(ix, k, 0, UD - 1, [1, 3], 0, NA)  # (compiles the programs and allocates the persistent state)
        lines.clear()
        m.export(ix, k, 0, UD - 1, [1, 3], 0, NA)
        base = [float(re.search(r"allocated\s+([\d.]+)", ln).group(1)) for ln in lines]
        res[lean] = base
        mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
        res[(lean, "after")] = mv.total_bytes_allocated_per_bank / 2**20
        print(
            f"KXMEASURE UD={UD} lean={int(lean)} pieces={len(base)} allocated MiB/bank per piece: {base} after-export {res[(lean, 'after')]:.1f}"
        )
        ttnn.deallocate(k)
    print(
        f"KXMEASURE UD={UD} peak old {max(res[False]):.1f} lean {max(res[True]):.1f}; growth old {res[False][-1] - res[False][0]:.1f} lean {res[True][-1] - res[True][0]:.1f}"
    )
