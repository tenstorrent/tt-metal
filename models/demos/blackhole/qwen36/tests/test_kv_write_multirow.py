# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check for QWEN36_KV_WRITE_MULTIROW at the model's call site (TPAttention._write_kv_fused_groups).

The traced verify writes each layer's K/V for n users x T candidates. Flag off, that is T grouped single-row
paged_fused_update_cache calls; flag on, ONE multi-row call (rows_per_user=T). This test runs BOTH on the TP mesh with
the inputs built the way the model builds them, and requires byte-identical caches (and both equal to torch):

  * caches of the 32K B=8 demo: [8 * 544 blocks, 1 head, 64, 256] bf16, so cache page ids reach ~70k tiles;
  * page tables [n, 544] (the demo's per-user table width, i.e. the stick the kernels read), contiguous
    (user u owns blocks [u*544, (u+1)*544), as text_demo does) or a random permutation of the blocks;
  * positions: the per-row kvpos the verify trace derives from its packed aux buffer
    (slice row 0 of a uint32 [3, W] -> typecast int32 -> reshape [n*T]), or a plain int32 [n*T] buffer;
  * candidates p..p+T-1 per user at mixed starts: 32-row tile straddles, 64-row page-block straddles, aligned
    starts and the last slot that fits the table;
  * k / v as [1, n*T, 1, 256] bf16 TILE L1-interleaved (the RoPE output), consumed by the write;
  * every verify row bucket n in (8, 4, 2, 1), and the write captured in a trace and replayed at new positions.

Run (inside the tt-metal container, P150x8 TP mesh: source /tmp/penv.sh):
    pytest models/demos/blackhole/qwen36/tests/test_kv_write_multirow.py -v -s
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.attention import tp as tp_mod
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention

TILE, BLOCK, HD, T = 32, 64, 256, 4
BPU = 544  # blocks per user in the 32K B=8 spec demo (text_demo._run_tp_spec_generation_batched)
NUSERS_CACHE = 8  # the cache always holds 8 users' blocks, as in the demo


def _starts(n):
    """First verify slot per user: tile straddle (29..32), block straddles (62..65, 63..66), aligned (0..3), the last
    slot that fits a 544-block table, and plain mid-tile starts. All candidates of a user are in that user's blocks."""
    s = [32766, 32768, 32797, 32895, 33000, 34812, 32830, 32769]
    return s[:n]


def _rows(starts):
    return [p + j for p in starts for j in range(T)]


def _page_table(n, kind, g):
    if kind == "contiguous":
        return torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(n)])
    perm = torch.randperm(NUSERS_CACHE * BPU, generator=g).to(torch.int32)
    return perm[: n * BPU].reshape(n, BPU).contiguous()


def _attn(mesh, n_max, k_cache, v_cache):
    """The attributes _write_kv_fused_groups / _write_kv_fused_multirow and their shard configs read."""
    a = TPAttention.__new__(TPAttention)
    a.mesh = mesh
    a.HD = HD
    a.B = n_max
    a.paged_k, a.paged_v = k_cache, v_cache
    a._kv_shard_cfg_cache, a._kv_key_cfg_cache, a._kv_rows_cfg_cache = {}, {}, {}
    cols = next(c for c in range(min(8, n_max), 0, -1) if n_max % c == 0)
    # model_config.kv_update_shard_cfg (what _kv_shard_cfg returns at the full decode width)
    a.args = types.SimpleNamespace(
        kv_update_shard_cfg=ttnn.create_sharded_memory_config(
            shape=(TILE, HD),
            core_grid=ttnn.CoreGrid(x=cols, y=n_max // cols),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    )
    return a


def _rep(mesh, t, dtype, layout, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=mem
    )


def _aux_host(rows_pos, width):
    """The verify's packed aux layout (uint32 [3, W]): row 0 = per-row KV positions; rows 1-2 = other words."""
    aux = torch.zeros(3, width, dtype=torch.int32)
    aux[0, : len(rows_pos)] = torch.tensor(rows_pos, dtype=torch.int32)
    aux[1, : len(rows_pos)] = 7
    aux[2] = 12345
    return aux


def _kvpos_from_aux(aux_tt, rows):
    """Qwen36Model._vfy_inputs: kvpos = reshape(typecast(slice(aux row 0)), (rows,))."""
    pos_u = ttnn.slice(aux_tt, [0, 0], [1, rows])
    kv_i = ttnn.typecast(pos_u, ttnn.int32)
    ttnn.deallocate(pos_u)
    return ttnn.reshape(kv_i, (rows,))


def _expected(cache0, x, rows_pos, pt):
    exp = cache0.clone()
    n = pt.shape[0]
    for u in range(n):
        for j in range(T):
            p = rows_pos[u * T + j]
            if p < 0:
                continue
            exp[int(pt[u, p // BLOCK]), 0, p % BLOCK] = x[u * T + j]
    return exp


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _write(a, k_src, v_src, kvpos, n, pt_tt, multirow, monkeypatch):
    """One verify-layer KV write through the model's code, flag on or off. k / v are L1 copies (consumed)."""
    monkeypatch.setenv("QWEN36_KV_WRITE_MULTIROW", "1" if multirow else "0")
    k = ttnn.to_memory_config(k_src, ttnn.L1_MEMORY_CONFIG)
    v = ttnn.to_memory_config(v_src, ttnn.L1_MEMORY_CONFIG)
    a._write_kv_fused_groups(k, v, kvpos, n, pt_tt, None, n * T)


@pytest.fixture(scope="module")
def caches0():
    g = torch.Generator().manual_seed(1234)
    shape = [NUSERS_CACHE * BPU, 1, BLOCK, HD]
    return torch.randn(shape, generator=g).bfloat16(), torch.randn(shape, generator=g).bfloat16()


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize(
    "n, pt_kind, pos_src",
    [
        (8, "contiguous", "aux"),
        (8, "permuted", "aux"),
        (8, "contiguous", "plain"),
        (4, "permuted", "aux"),
        (2, "contiguous", "aux"),
        (1, "permuted", "aux"),
    ],
)
def test_multirow_matches_grouped_at_call_site(mesh_device, n, pt_kind, pos_src, caches0, monkeypatch):
    if not tp_mod.KV_MULTIROW_WRITE_SUPPORTED:
        pytest.skip("this ttnn build has no paged_fused_update_cache(rows_per_user=...)")
    g = torch.Generator().manual_seed(n * 10 + (pt_kind == "permuted"))
    k0, v0 = caches0
    R = n * T
    xk = torch.randn(R, HD, generator=g).bfloat16()
    xv = torch.randn(R, HD, generator=g).bfloat16()
    rows_pos = _rows(_starts(n))
    pt = _page_table(n, pt_kind, g)

    pt_tt = _rep(mesh_device, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    if pos_src == "aux":
        aux_tt = _rep(mesh_device, _aux_host(rows_pos, 128), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        kvpos = _kvpos_from_aux(aux_tt, R)
    else:
        kvpos = _rep(mesh_device, torch.tensor(rows_pos, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    k_src = _rep(mesh_device, xk.reshape(1, R, 1, HD), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    v_src = _rep(mesh_device, xv.reshape(1, R, 1, HD), ttnn.bfloat16, ttnn.TILE_LAYOUT)

    out = {}
    for multirow in (False, True):
        kc = _rep(mesh_device, k0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        vc = _rep(mesh_device, v0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        a = _attn(mesh_device, NUSERS_CACHE, kc, vc)
        _write(a, k_src, v_src, kvpos, n, pt_tt, multirow, monkeypatch)
        ttnn.synchronize_device(mesh_device)
        out[multirow] = (_dev0(kc), _dev0(vc))
        ttnn.deallocate(kc)
        ttnn.deallocate(vc)

    for i, (name, c0, x) in enumerate((("K", k0, xk), ("V", v0, xv))):
        ref, new = out[False][i], out[True][i]
        exp = _expected(c0, x, rows_pos, pt)
        assert torch.equal(ref, exp), f"{name}: grouped write differs from torch ({(ref != exp).sum()} values)"
        bad = (new != exp).reshape(new.shape[0], -1).any(-1).nonzero().flatten().tolist()
        assert not bad, (
            f"{name}: multi-row write differs from the grouped write / torch in {len(bad)} blocks "
            f"(first {bad[:8]}; {(new != exp).sum()} values)"
        )


@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=20 * 1024 * 1024)
@pytest.mark.parametrize("n", [8, 4])
def test_multirow_matches_grouped_in_trace(mesh_device, n, caches0, monkeypatch):
    """The verify's form: persistent aux + page table, the write captured in a trace, replayed after the host restages
    new positions. Two traces (flag off / on) over two cache copies; each replay must match torch exactly."""
    if not tp_mod.KV_MULTIROW_WRITE_SUPPORTED:
        pytest.skip("this ttnn build has no paged_fused_update_cache(rows_per_user=...)")
    g = torch.Generator().manual_seed(99 + n)
    k0, v0 = caches0
    R = n * T
    pt = _page_table(n, "permuted", g)
    pt_tt = _rep(mesh_device, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    first = _rows(_starts(n))
    aux_tt = _rep(mesh_device, _aux_host(first, 128), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    k_in = _rep(mesh_device, torch.zeros(1, R, 1, HD), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    v_in = _rep(mesh_device, torch.zeros(1, R, 1, HD), ttnn.bfloat16, ttnn.TILE_LAYOUT)

    state = {}
    for multirow in (False, True):
        kc = _rep(mesh_device, k0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        vc = _rep(mesh_device, v0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        a = _attn(mesh_device, NUSERS_CACHE, kc, vc)

        def body():
            kvpos = _kvpos_from_aux(aux_tt, R)
            _write(a, k_in, v_in, kvpos, n, pt_tt, multirow, monkeypatch)
            ttnn.deallocate(kvpos)

        body()  # compile outside the trace, as prepare_verify_trace does
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        body()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        state[multirow] = (a, kc, vc, tid)

    # the eager compile pass above wrote the zero k_in / v_in rows at the first positions in both cache copies
    zeros = torch.zeros(R, HD, dtype=torch.bfloat16)
    exp_k, exp_v = _expected(k0, zeros, first, pt), _expected(v0, zeros, first, pt)
    # successive verify steps: each user advances by 1..T committed tokens, the verify window p..p+T-1 moves with it
    starts = _starts(n)
    for step in range(4):
        if step:
            starts = [min(p + 1 + (u + step) % T, BPU * BLOCK - T) for u, p in enumerate(starts)]
        rows_pos = _rows(starts)
        xk = torch.randn(R, HD, generator=g).bfloat16()
        xv = torch.randn(R, HD, generator=g).bfloat16()
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        for host, dev in (
            (_aux_host(rows_pos, 128), aux_tt),
            (xk.reshape(1, R, 1, HD), k_in),
            (xv.reshape(1, R, 1, HD), v_in),
        ):
            dt, lay = (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if dev is aux_tt else (ttnn.bfloat16, ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dt, layout=lay, mesh_mapper=rep), dev)
        for multirow in (False, True):
            ttnn.execute_trace(mesh_device, state[multirow][3], cq_id=0, blocking=True)
        exp_k = _expected(exp_k, xk, rows_pos, pt)
        exp_v = _expected(exp_v, xv, rows_pos, pt)
        for multirow in (False, True):
            _, kc, vc, _ = state[multirow]
            for name, c, exp in (("K", kc, exp_k), ("V", vc, exp_v)):
                got = _dev0(c)
                assert torch.equal(got, exp), (
                    f"step {step} {'multi-row' if multirow else 'grouped'} {name}: {(got != exp).sum()} values "
                    f"differ from torch"
                )
    for multirow in (False, True):
        _, kc, vc, tid = state[multirow]
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(kc)
        ttnn.deallocate(vc)
