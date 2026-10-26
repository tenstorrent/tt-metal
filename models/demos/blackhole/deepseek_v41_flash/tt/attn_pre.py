# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One program replacing concat + nlp_create_qkv_heads_decode + 2x to_memory_config + partial RoPE (3 ops) + paged_update_cache x2.
Env DSV41_ATTN_FUSED_PRE=1.  See tt/attn_kernels/pre_*.cpp."""

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attn_fused import _acc, _cb, _hash, _kernel


def attn_pre(kvn, q, lat, cache, pos, comp_idx, ch, sh, users):
    """kvn [1,1,T,512], q [1,1,T,4096] bf16 TILE (q_b output), lat [1,1,T,512] (already RoPE'd, or None), cache [T,1,L,512] bf16 TILE
    (updated in place: kv rope row -> slot pos[t], lat -> slot comp_idx[t]), ch/sh [1,T,1,512] bf16 TILE tables.
    Returns q_rot [1,T,32,512] bf16 TILE DRAM: row 0 = RoPE'd kv, rows 1..8 the RoPE'd q heads (rest zero)."""
    T = users
    assert (
        T <= 16
        and (cache is None or cache.dtype == ttnn.bfloat16)
        and q.dtype == ttnn.bfloat16
        and kvn.dtype == ttnn.bfloat16
    )
    mesh = kvn.device()
    has_lat = lat is not None
    lat_t = lat if has_lat else kvn
    comp_t = comp_idx if has_lat else pos
    if (
        cache is None
    ):  # no cache write (paged / spec callers): CACHE_PT = 0 compiles the writes out; dummy tensors fill the accessor slots
        assert not has_lat
        cache, comp_t, pos, cache_pt = kvn, kvn, kvn, 0
    else:
        cache_pt = (int(cache.shape[2]) // 32) * 16
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, T, 9, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    nb = 2 * T
    na = min(14 * T, len(allc) - nb)
    cb_, ca_ = allc[:nb], allc[nb : nb + na]
    cset = lambda cs: ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cs]
    )
    setb, seta = cset(cb_), cset(ca_)
    rtb, rta = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for w, (cx, cy) in enumerate(cb_):
        rtb[cx][cy] = [w, 0]
    items, base = 14 * T, 0
    for k, (cx, cy) in enumerate(ca_):
        cnt = items // na + (1 if k < items % na else 0)
        rta[cx][cy] = [base, cnt]
        base += cnt
    bf = ttnn.bfloat16
    tens = [kvn, q, lat_t, cache, out, pos, comp_t, ch, sh]
    common = [t.buffer_address() for t in tens]
    accs = sum([_acc(t) for t in tens[:7]], []) + _acc(ch) + _acc(sh)
    XA, SA = 0, 1
    XB, RB, SB, OB = 2, 3, 4, 5
    cbs = [
        _cb(seta, XA, 1, 2048, bf),
        _cb(seta, SA, 1, 4096, bf),
        _cb(setb, XB, 1, 2048, bf),
        _cb(setb, RB, 1, 2048, bf),
        _cb(setb, SB, 1, 4096, bf),
        _cb(setb, OB, 1, 2048, bf),
    ]
    rdA = _kernel(
        "pre_reader.cpp",
        seta,
        [0, int(has_lat), cache_pt, XA, 0, SA] + accs,
        rta,
        ttnn.ReaderConfigDescriptor(),
        common,
    )
    rdB = _kernel(
        "pre_reader.cpp",
        setb,
        [1, int(has_lat), cache_pt, XB, RB, SB] + accs,
        rtb,
        ttnn.ReaderConfigDescriptor(),
        common,
    )
    wrB = _kernel(
        "pre_writer.cpp",
        setb,
        [int(has_lat), cache_pt, OB, SB] + sum([_acc(t) for t in tens[:7]], []),
        rtb,
        ttnn.WriterConfigDescriptor(),
        common,
    )
    cpB = _kernel(
        "rope_compute.cpp",
        setb,
        [XB, RB, OB],
        ttnn.RuntimeArgs(),
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        [],
    )
    prog = ttnn.ProgramDescriptor(kernels=[rdA, rdB, wrB, cpB], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(0xA71, T, int(has_lat), cache_pt, tuple(accs))
    io = []
    for t in [kvn, q, lat_t, cache, pos, comp_t, ch, sh, out]:
        if not any(t is u for u in io):
            io.append(t)
    ttnn.generic_op(io, prog)
    return out


def qk_heads(kvn, q, ch, sh, users):
    """attn_pre without cache write, restructured for DRAM friendliness (tt/attn_kernels/qkh_*.cpp): one core per (tile column n, UPC users) reads the 9
    whole source tiles of its column.  kvn [1,1,T,512], q [1,1,T,4096] bf16 TILE, ch/sh [1,T,1,512] -> q_rot [1,T,9,512] bf16 TILE DRAM
    (row 0 = RoPE'd kv, rows 1..8 = RoPE'd q heads, rest zero)."""
    import os

    T = users
    assert T <= 32 and q.dtype == ttnn.bfloat16 and kvn.dtype == ttnn.bfloat16
    upc = max(1, min(T, int(os.environ.get("DSV41_QKH_UPC", "8"))))
    upr = max(1, min(T, int(os.environ.get("DSV41_QKH_UPR", "4"))))  # users per RoPE core
    assert upc <= 8 and upr <= 8
    mesh = kvn.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, T, 9, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    jobs_a = [(n, t0, min(upc, T - t0)) for n in range(14) for t0 in range(0, T, upc)]
    jobs_b = [(n, t0, min(upr, T - t0)) for n in (14, 15) for t0 in range(0, T, upr)]
    assert len(jobs_a) + len(jobs_b) <= len(allc)
    ca_, cb_ = allc[: len(jobs_a)], allc[len(jobs_a) : len(jobs_a) + len(jobs_b)]
    cset = lambda cs: ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cs]
    )
    seta, setb = cset(ca_), cset(cb_)
    rta, rtb = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for (cx, cy), j in zip(ca_, jobs_a):
        rta[cx][cy] = list(j)
    for (cx, cy), j in zip(cb_, jobs_b):
        rtb[cx][cy] = list(j)
    bf = ttnn.bfloat16
    SRC, X, R, S, O = range(5)
    cbs_a = [_cb(seta, SRC, 9, 2048, bf), _cb(seta, X, upc, 2048, bf)]
    cbs_b = [
        _cb(setb, SRC, 9, 2048, bf),
        _cb(setb, X, upr, 2048, bf),
        _cb(setb, R, upr, 2048, bf),
        _cb(setb, S, 1, 2048, bf),
        _cb(setb, O, upr, 2048, bf),
    ]
    tens = [kvn, q, out, ch, sh]
    common = [t.buffer_address() for t in tens]
    accs = sum([_acc(t) for t in tens], [])
    rdA = _kernel("qkh_reader.cpp", seta, [0, SRC, X, R, S] + accs, rta, ttnn.ReaderConfigDescriptor(), common)
    rdB = _kernel("qkh_reader.cpp", setb, [1, SRC, X, R, S] + accs, rtb, ttnn.ReaderConfigDescriptor(), common)
    wrB = _kernel("qkh_writer.cpp", setb, [O] + _acc(out), rtb, ttnn.WriterConfigDescriptor(), common)
    cpB = _kernel(
        "qkh_compute.cpp",
        setb,
        [X, R, O],
        rtb,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        [],
    )
    prog = ttnn.ProgramDescriptor(kernels=[rdA, rdB, wrB, cpB], semaphores=[], cbs=cbs_a + cbs_b)
    prog.custom_program_hash = _hash(0xA73, T, upc, upr, tuple(accs))
    ttnn.generic_op([kvn, q, ch, sh, out], prog)
    return out
