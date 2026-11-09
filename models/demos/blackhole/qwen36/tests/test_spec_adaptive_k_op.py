# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_SPEC_ADAPTIVE_K at the gdn_spec_step op level (one random GDN layer, TP mesh, no checkpoint beyond config.json).

The adaptive engine allocates the ring for T_alloc = 8 (8 * M * Nv blocks, M = 8) and the conv-window pair as [8, 3 + 8, C],
and alternates T = 8 verifies (4 users) and T = 4 verifies (8 users) on them. A T = 4 verify that follows a T = 8 one reads
its carry rows at mi up to 7, which gdn_spec_step(mi_max = 7) allows (the default clamps mi to T - 1).

  a. T = 4 with mi_max = 7 on the T_alloc buffers, mi in 0..7 per user == the default T = 4 op on a [B, 7, C] window
     re-aligned on the host (carry rows mi+1 .. mi+3 moved to rows 1..3) and a 4-slot ring whose slot-0 block u is the
     T_alloc ring's block (mi*8 + u): output rows, the ring slots it writes and the new window rows, bit-identical.
  b. mi < 4 everywhere: the T_alloc-buffer call (mi_max = 7) == the default op on the 7-row window / 4-slot ring
     (what a base-family step computes with the flag off), bit-identical.
  c. the 4-user T = 8 call (max_users 8) on the T_alloc buffers writes only users 0..3 (ring and window).

  QWEN36_FABRIC_RING=1 MESH_DEVICE=P150x8 HF_MODEL=... pytest models/demos/blackhole/qwen36/tests/test_spec_adaptive_k_op.py
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tests.test_spec_verify_specstep import _build, _cat0, _cmp
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_page

M, TA = 8, 8


def _rep(mesh, t, dt, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dt,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _call(mesh, g, x, win_a, win_b, ring, mi, T, B, mi_max):
    """One gdn_spec_step on host-built inputs; returns (out [nd, B*T, ...], ring, win_b) host copies."""
    nd = mesh.get_num_devices()
    xa = _rep(mesh, x, ttnn.bfloat16)
    wa, wb = _rep(mesh, win_a, ttnn.bfloat16), _rep(mesh, win_b, ttnn.bfloat16)
    rg = _rep(mesh, ring, ttnn.float32)
    ctrl = _rep(mesh, spec_ctrl_page(list(mi) + [0] * (M - len(mi)), M, g.Nv, parity=0), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    out = g._gdn_spec_step(xa, wa, wb, rg, ctrl, T, B, max_users=M, mi_max=mi_max)
    o = _cat0(mesh, out)
    r = _cat0(mesh, rg).reshape(nd, -1, M, g.Nv, g.Dk, g.Dv)
    w = _cat0(mesh, wb).reshape(nd, M, int(win_b.shape[1]), int(win_b.shape[2]))
    return o, r, w


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("seed", [0, 1])
def test_mi_max_short_after_long(mesh_device, seed, reset_seeds, ensure_gc, monkeypatch):
    mesh = mesh_device
    _, g, _, _ = _build(mesh, M, monkeypatch)
    Nv, Dk, Dv, C, K = g.Nv, g.Dk, g.Dv, g.qkv_dim_tp, g.K
    T = 4
    W = (g.qkvz_dim_tp + 2 * Nv + 31) // 32 * 32
    torch.manual_seed(77 + seed)
    x = torch.randn(1, M * T, W).to(torch.bfloat16)
    ring_a = 0.1 * torch.randn(TA * M * Nv, Dk, Dv)  # the T_alloc ring
    La = K - 1 + TA
    win_a = torch.randn(M, La, C).to(torch.bfloat16)  # E_prev (the half the op reads at parity 0)
    win_b = torch.zeros(M, La, C).to(torch.bfloat16)
    for mi in ([7, 0, 3, 6, 5, 1, 4, 2], [3, 0, 2, 1, 0, 3, 1, 2]):
        # a. / b. the T_alloc call
        o1, r1, w1 = _call(mesh, g, x, win_a, win_b, ring_a, mi, T, M, mi_max=TA - 1)
        # reference: a host-realigned 7-row window and a 4-slot ring holding each user's accepted block at slot 0
        L = K - 1 + T
        win_ref = torch.zeros(M, L, C).to(torch.bfloat16)
        ring_ref = 0.1 * torch.randn(T * M * Nv, Dk, Dv)
        ra = ring_a.reshape(TA, M, Nv, Dk, Dv)
        rr = ring_ref.reshape(T, M, Nv, Dk, Dv)
        for u in range(M):
            win_ref[u, 1:K] = win_a[u, mi[u] + 1 : mi[u] + K]
            rr[0, u] = ra[mi[u], u]
        o2, r2, w2 = _call(mesh, g, x, win_ref, torch.zeros(M, L, C).to(torch.bfloat16), ring_ref, [0] * M, T, M, 0)
        tag = f"mi={mi}"
        _cmp(f"{tag}: output rows", o1[:, : M * T], o2[:, : M * T], True)
        _cmp(f"{tag}: ring slots 0..3", r1[:, :T], r2[:, :T], True)
        _cmp(f"{tag}: new window rows 0..{L - 1}", w1[:, :, :L], w2[:, :, :L], True)
        if max(mi) < T:  # b. the same as the flag-off op on the flag-off buffers at the SAME mi (no realignment)
            win_off = win_a[:, :L].contiguous()
            ring_off = ring_a.reshape(TA, M, Nv, Dk, Dv)[:T].reshape(T * M * Nv, Dk, Dv).contiguous()
            o3, r3, w3 = _call(mesh, g, x, win_off, torch.zeros(M, L, C).to(torch.bfloat16), ring_off, mi, T, M, 0)
            _cmp(f"{tag}: flag-off op output", o1[:, : M * T], o3[:, : M * T], True)
            _cmp(f"{tag}: flag-off op ring", r1[:, :T], r3[:, :T], True)
            _cmp(f"{tag}: flag-off op window", w1[:, :, :L], w3[:, :, :L], True)


@torch.no_grad()
@parametrize_mesh_tp()
def test_wide_t8_writes_only_its_users(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    mesh = mesh_device
    _, g, _, _ = _build(mesh, M, monkeypatch)
    Nv, Dk, Dv, C, K = g.Nv, g.Dk, g.Dv, g.qkv_dim_tp, g.K
    b, T = 4, TA
    W = (g.qkvz_dim_tp + 2 * Nv + 31) // 32 * 32
    torch.manual_seed(5)
    x = torch.randn(1, b * T, W).to(torch.bfloat16)
    ring0 = 0.1 * torch.randn(TA * M * Nv, Dk, Dv)
    La = K - 1 + TA
    win_a = torch.randn(M, La, C).to(torch.bfloat16)
    win_b = torch.randn(M, La, C).to(torch.bfloat16)
    o, r, w = _call(mesh, g, x, win_a, win_b, ring0, [2, 0, 1, 3], T, b, 0)
    nd = mesh.get_num_devices()
    r0 = ring0.reshape(1, TA, M, Nv, Dk, Dv).expand(nd, -1, -1, -1, -1, -1)
    w0 = win_b.float().reshape(1, M, La, C).expand(nd, -1, -1, -1)
    assert not torch.equal(r[:, :, :b], r0[:, :, :b]), "the wide call did not write its users' ring blocks"
    _cmp("wide T=8: ring users >= 4 untouched", r[:, :, b:], r0[:, :, b:], True)
    _cmp("wide T=8: window users >= 4 untouched", w[:, b:], w0[:, b:], True)
    assert torch.isfinite(o[:, : b * T]).all()
