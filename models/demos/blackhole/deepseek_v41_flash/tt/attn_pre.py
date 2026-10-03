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
    assert T <= 16 and cache.dtype == ttnn.bfloat16 and q.dtype == ttnn.bfloat16 and kvn.dtype == ttnn.bfloat16
    mesh = kvn.device()
    has_lat = lat is not None
    lat_t = lat if has_lat else kvn
    comp_t = comp_idx if has_lat else pos
    L = int(cache.shape[2])
    cache_pt = (L // 32) * 16
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
    ttnn.generic_op([kvn, q, lat_t, cache, pos, comp_t, ch, sh, out], prog)
    return out
