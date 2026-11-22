# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.7B: the new Python-built programs of PACKED_GLUED_ONECALL=on (tt/packed_glued.py) on the model's mesh, mesh only
(no model), smallest first, bit-exact against the host:
  - ChipRows: per-chip row pieces of a source into a destination (rolls, keeps), ROW_MAJOR rows with blocks (the sdpa
    output's heads), each row's first columns only (the scores' phase roll), and TILE tile rows (bf16, bfloat4_b: the
    indexer's query), several tables of one chunk.
  - GluedSend: the glued window region (own rows, every chip's last 128 rows to the next chip, each active segment's
    carry from its tail tensor at a first page and rotation into its block, every segment's ring pieces into its
    ring), twice (the prepared program's second run, a shared tail tensor).
  - GluedGeo.region_lists: every row of a glued chunk reads exactly its own segment's window positions (host)."""

import types

import pytest
import torch

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"


def _per_chip(mesh, t, dtype, layout=None):
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.attention import _upload

    return _upload(mesh, t, dtype, layout or ttnn.ROW_MAJOR_LAYOUT, (0, 1))


def _host(t):
    import ttnn

    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _segrun(lane, s):
    return types.SimpleNamespace(lane=lane, seg=s, start=s.start, run_end=s.end // 64 * 64, state=None, fields={})


def region_lists_ok(sp: int, tp: int) -> int:
    """Host check (no device): rows of a glued chunk whose window list is not their segment's positions."""
    from models.demos.deepseek_v41_flash_d_p.tt.attention import SENTINEL
    from models.demos.deepseek_v41_flash_d_p.tt.packed import GluedChunk, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.packed_glued import WINDOW, GluedGeo

    mesh = types.SimpleNamespace(shape=(sp, tp))
    chunk = 8192
    segs = [Segment(2, 2048, 3648), Segment(3, 4096, 4873), Segment(6, 9216, 10716), Segment(1, 0, 2048)]
    segs += [Segment(7, 96, 160), Segment(8, 6144, 6304)]
    desc = GluedChunk(chunk, tuple(segs))
    g = GluedGeo(mesh, desc, [_segrun(k, s) for k, s in enumerate(segs)])
    classes = tuple(min(s.start, WINDOW) for s in segs)
    n, R = g.n, g.R
    gl = g.region_lists(classes, list(range(len(segs)))).view(n, R, WINDOW).long()
    pos = torch.full((n, WINDOW + R + 6 * WINDOW), -1, dtype=torch.long)
    rowpos = torch.full((chunk,), -(10**9), dtype=torch.long)
    rowseg = torch.full((chunk,), -1)
    for j, (o, k) in enumerate(zip(g.off, g.rows)):
        rowpos[o : o + k] = segs[j].start + torch.arange(k)
        rowseg[o : o + k] = j
    for q in range(n):
        pos[q, WINDOW : WINDOW + R] = rowpos[q * R : (q + 1) * R]
        if q:
            pos[q, :WINDOW] = rowpos[q * R - WINDOW : q * R]
    for j, s in enumerate(segs):
        pos[:, WINDOW + R + WINDOW * j : WINDOW + R + WINDOW * (j + 1)] = s.start - WINDOW + torch.arange(WINDOW)
    bad = 0
    for m in range(g.used):
        q, i = divmod(m, R)
        j, p = int(rowseg[m]), int(rowpos[m])
        want = list(range(max(p - WINDOW + 1, segs[j].start - classes[j]), p + 1))
        got = [int(pos[q, int(x)]) for x in gl[q, i] if int(x) != SENTINEL]
        bad += int(got != want)
    return bad


def test_glued_region_lists_cpu():
    for sp, tp in ((8, 4), (4, 4), (2, 4), (2, 2)):
        bad = region_lists_ok(sp, tp)
        print(f"PB.7B region lists {sp}x{tp}: {bad} rows wrong")
        assert bad == 0


@mesh_parametrize(timeout_s=900)
def test_glued_ops(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.packed_glued import GLUED_SEGS, ChipTables, chip_rows, glued_send

    sp, tp = tuple(mesh_device.shape)
    n = sp * tp
    R = 8192 // n
    g = torch.Generator().manual_seed(11)
    tables = ChipTables(mesh_device, n)

    # ChipRows, ROW_MAJOR uint32 [1, 4, R, 96] (4 blocks): per chip a roll by a chip-dependent row count
    src = torch.randint(0, 1 << 30, (sp, tp * 4, R, 96), dtype=torch.int32, generator=g)
    s_t = _per_chip(mesh_device, src.view(sp, tp, 4, R, 96).reshape(sp, tp * 4, R, 96), ttnn.uint32)
    # (per chip [1, 4, R, 96]: shard dims (0, 1) of [sp, tp * 4, ...] gives each chip 4 blocks)
    dst0 = torch.randint(0, 1 << 30, (sp, tp * 4, R, 96), dtype=torch.int32, generator=g)
    d_t = _per_chip(mesh_device, dst0, ttnn.uint32)
    rolls = [(32 * (q % 5)) for q in range(n)]
    pieces = [([(a, 0, R - a), (0, R - a, a)] if a else [(0, 0, R)]) for a in rolls]
    chip_rows(s_t, d_t, tables, "roll", pieces)
    keep = [([] if q % 3 == 0 else [(0, 32 * (q % 4), 64)]) for q in range(n)]
    d2 = _per_chip(mesh_device, dst0, ttnn.uint32)
    chip_rows(s_t, d2, tables, "keep", keep)
    got, got2 = _host(d_t), _host(d2)
    bad = 0
    for q in range(n):
        r, c = divmod(q, tp)
        s = src[r, 4 * c : 4 * c + 4]
        want = torch.roll(s, -rolls[q], dims=1)
        bad += int(not torch.equal(got[q].reshape(4, R, 96).to(torch.int32), want))
        w2 = dst0[r, 4 * c : 4 * c + 4].clone()
        for s0, d0, m in keep[q]:
            w2[:, d0 : d0 + m] = s[:, s0 : s0 + m]
        bad += int(not torch.equal(got2[q].reshape(4, R, 96).to(torch.int32), w2))
    print(f"PB.7B ChipRows rows: {bad} chips differ", flush=True)
    assert bad == 0

    # ChipRows, a logits-like ROW_MAJOR bf16 [1, 1, R, 20000]: a roll of each row's first 9024 columns only
    lg = torch.randn(sp, tp, R, 20000, generator=g).bfloat16()
    l_t = _per_chip(mesh_device, lg, ttnn.bfloat16)
    o0 = torch.randn(sp, tp, R, 20000, generator=g).bfloat16()
    o_t = _per_chip(mesh_device, o0, ttnn.bfloat16)
    chip_rows(l_t, o_t, tables, "roll", pieces, cbytes=9024 * 2)
    got = _host(o_t)
    bad = 0
    for q in range(n):
        r, c = divmod(q, tp)
        want = o0[r, c].clone()
        want[:, :9024] = torch.roll(lg[r, c], -rolls[q], dims=0)[:, :9024]
        bad += int(not torch.equal(got[q].reshape(R, 20000), want))
    print(f"PB.7B ChipRows row prefixes: {bad} chips differ", flush=True)
    assert bad == 0
    for t in (l_t, o_t):
        ttnn.deallocate(t)

    # ChipRows, TILE [1, 2, R, 128] bf16 and bfloat4_b: rolls of tile rows
    for dt in (ttnn.bfloat16, ttnn.bfloat4_b):
        x = torch.randn(sp, tp * 2, R, 128, generator=g)
        x_t = _per_chip(mesh_device, x, dt, ttnn.TILE_LAYOUT)
        y_t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 2, R, 128]), dt, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        chip_rows(x_t, y_t, tables, "roll", pieces)
        xs, ys = _host(x_t), _host(y_t)
        bad = 0
        for q in range(n):
            want = torch.roll(xs[q].reshape(2, R, 128).float(), -rolls[q], dims=1)
            bad += int(not torch.equal(ys[q].reshape(2, R, 128).float(), want))
        print(f"PB.7B ChipRows tiles {dt}: {bad} chips differ", flush=True)
        assert bad == 0
        for t in (x_t, y_t):
            ttnn.deallocate(t)

    # GluedSend: region [128 + R + 128 S, 512] bf16, own rows [R, 512], tails [.., 512], rings [192, 512]
    S = GLUED_SEGS
    snd = glued_send(mesh_device, R)
    rep = lambda t: _per_chip(mesh_device, t.expand(sp, tp, *t.shape).contiguous(), ttnn.bfloat16)  # noqa: E731
    for it in range(2):
        own = torch.randn(sp, tp, R, 512, generator=g).bfloat16()
        reg0 = torch.randn(sp, tp, 128 + R + 128 * S, 512, generator=g).bfloat16()
        tails = [torch.randn(128, 512, generator=g).bfloat16() for _ in range(S)]
        rings0 = [torch.randn(192, 512, generator=g).bfloat16() for _ in range(S)]
        region = _per_chip(mesh_device, reg0, ttnn.bfloat16)
        rows_t = _per_chip(mesh_device, own, ttnn.bfloat16)
        ring_t = [rep(t) for t in rings0]
        pieces = [[(32 * it, 64, 32)], [(R - 64, 160, 32), (0, 0, 32)], [], [(96, 128 - 32 * it, 64)], [], []]
        use_ring = [ring_t[0], ring_t[1], None, ring_t[3], ring_t[4], None]
        if it == 0:  # (each active segment its own carry tensor, page 0, no rotation; segments 2 and 5 none)
            tail_t = [rep(t) for t in tails]
            use_tail = [tail_t[0], tail_t[1], None, tail_t[3], tail_t[4], None]
            carries = [(0, 0), (0, 0), None, (0, 0), (0, 0), None]
            carry = list(tails)
        else:  # (segments 0, 1 and 4 share one tensor of three 128-row slabs: bases / rotations, as the snapshot)
            shared = torch.cat([tails[0], tails[1], tails[4]], dim=0)
            sh_t = rep(shared)
            tail_t = [sh_t, rep(tails[3])]
            use_tail = [sh_t, sh_t, None, tail_t[1], sh_t, None]
            carries = [(0, 64), (128, 32), None, (0, 0), (256, 96), None]
            carry = [torch.roll(tails[0], -64, 0), torch.roll(tails[1], -32, 0), None, tails[3]]
            carry += [torch.roll(tails[4], -96, 0), None]
        snd(region, rows_t, use_tail, use_ring, pieces, carries)
        greg = _host(region)
        grings = [_host(t) for t in ring_t]
        bad = 0
        for q in range(n):
            r, c = divmod(q, tp)
            want = reg0[r, c].clone()
            want[128 : 128 + R] = own[r, c]
            if q > 0:
                pr, pc = divmod(q - 1, tp)
                want[:128] = own[pr, pc, R - 128 :]
            for s in range(S):
                if carries[s] is not None:
                    want[128 + R + 128 * s : 128 + R + 128 * (s + 1)] = carry[s]
            ok = torch.equal(greg[q].reshape(-1, 512), want)
            for s in (0, 1, 3, 4):
                wr = rings0[s].clone()
                for s0, d0, m in pieces[s]:
                    wr[d0 : d0 + m] = own[r, c, s0 : s0 + m]
                ok = ok and torch.equal(grings[s][q].reshape(192, 512), wr)
            bad += int(not ok)
        print(f"PB.7B GluedSend run {it}: {bad} chips differ", flush=True)
        assert bad == 0
        for t in [region, rows_t] + tail_t + ring_t:
            ttnn.deallocate(t)
    for t in (s_t, d_t, d2):
        ttnn.deallocate(t)
    tables.free()
