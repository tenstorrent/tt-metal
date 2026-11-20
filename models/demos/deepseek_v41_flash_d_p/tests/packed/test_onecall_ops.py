# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.3A: the new Python-built programs of PACKED_ATTN_ONECALL=on (tt/packed_onecall.py) on the model's mesh, mesh
only (no model), smallest first, bit-exact against the host:
  - LaneCopy: blocks of pages of a source -> a destination on one lane's chips only (the other chips untouched),
    for the history records (320 B pages, owner blocks) and a top-k (2048 B pages, whole tensor).
  - LaneSend: the per-lane window region write: each chip's own rows into its region rows [128, 128 + q_rows), its
    last 128 rows to the next chip of its lane, a lane's first chip its own copy of the lane's tail, the ring pieces
    of each lane into its lane's ring on its lane's chips (twice: the prepared program's second run)."""

import pytest
import torch

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"


def _per_chip(mesh, t, dtype):
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.attention import _upload

    return _upload(mesh, t, dtype, ttnn.ROW_MAJOR_LAYOUT, (0, 1))


def _host(t):
    import ttnn

    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@mesh_parametrize(timeout_s=900)
def test_onecall_ops(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.packed_onecall import Lanes, lane_copy, lane_send

    sp, tp = tuple(mesh_device.shape)
    n = sp * tp
    chunk = 8192
    desc = ChunkDescriptor.packed(chunk, [Segment(1, 0, 2048), Segment(2, 0, 2048), None, Segment(3, 0, 64)])
    lanes = Lanes(mesh_device, desc)
    g = torch.Generator().manual_seed(7)

    # LaneCopy, history records: 32 owner blocks of R rows, n rows each, 320 B
    owners, R, rows = 4, 96, 40
    src = torch.randint(0, 256, (sp, tp, owners * R, 320), dtype=torch.uint8, generator=g)
    dst0 = torch.randint(0, 256, (sp, tp, owners * R, 320), dtype=torch.uint8, generator=g)
    s_t, d_t = _per_chip(mesh_device, src, ttnn.uint8), _per_chip(mesh_device, dst0, ttnn.uint8)
    lane = 1
    lane_copy(mesh_device, lanes.chips[lane], 320, owners, R)(s_t, d_t, rows)
    got = _host(d_t)
    bad = 0
    for q in range(n):
        r, c = divmod(q, tp)
        want = dst0[r, c].clone()
        if lanes.lane_of[q] == lane:
            for b in range(owners):
                want[b * R : b * R + rows] = src[r, c, b * R : b * R + rows]
        bad += int(not torch.equal(got[q].reshape(owners * R, 320), want))
    print(f"PB.3A LaneCopy records: {bad} chips differ", flush=True)
    assert bad == 0

    # LaneCopy, a top-k: [1, 1, Sq, 512] uint32, whole tensor, each lane from its own source
    sq = chunk // n
    tops = [torch.randint(0, 1 << 20, (sp, tp, sq, 512), dtype=torch.int32, generator=g) for _ in range(4)]
    tt = [_per_chip(mesh_device, t, ttnn.uint32) for t in tops]
    comb = _per_chip(mesh_device, torch.zeros(sp, tp, sq, 512, dtype=torch.int32), ttnn.uint32)
    for k in (0, 1, 3):
        lane_copy(mesh_device, lanes.chips[k], 2048)(tt[k], comb, sq)
    got = _host(comb)
    bad = 0
    for q in range(n):
        r, c = divmod(q, tp)
        k = lanes.lane_of[q]
        want = tops[k][r, c] if k in (0, 1, 3) else torch.zeros(sq, 512, dtype=torch.int32)
        bad += int(not torch.equal(got[q].reshape(sq, 512).to(torch.int32), want))
    print(f"PB.3A LaneCopy top-k: {bad} chips differ", flush=True)
    assert bad == 0

    # LaneSend: region [128 + q, 512] bf16, own rows [q, 512], tails per lane [128, 512], rings [192, 512]
    q_rows = lanes.q_rows
    snd = lane_send(mesh_device, lanes)
    for it in range(2):
        own = torch.randn(sp, tp, q_rows, 512, generator=g).bfloat16()
        reg0 = torch.randn(sp, tp, 128 + q_rows, 512, generator=g).bfloat16()
        tails = [torch.randn(128, 512, generator=g).bfloat16() for _ in range(4)]
        rings0 = [torch.randn(192, 512, generator=g).bfloat16() for _ in range(4)]
        region = _per_chip(mesh_device, reg0, ttnn.bfloat16)
        rows_t = _per_chip(mesh_device, own, ttnn.bfloat16)
        rep = lambda t: _per_chip(mesh_device, t.expand(sp, tp, *t.shape).contiguous(), ttnn.bfloat16)  # noqa: E731
        tail_t = [rep(t) for t in tails]
        ring_t = [rep(t) for t in rings0]
        pieces = [[(32 * it, 64, 32)], [(q_rows - 64, 160, 32), (0, 0, 32)], [], [(96, 128 - 32 * it, 64)]]
        use_ring = [ring_t[0], ring_t[1], None, ring_t[3]]
        if it == 0:  # (each lane its own carry tensor, page 0, no rotation)
            use_tail, bases, rots = [tail_t[0], tail_t[1], None, tail_t[3]], None, None
            carry = list(tails)
        else:  # (lanes 0 and 1 share one tensor of two 128-row slabs, as the engine's snapshot tensor: base / rotation)
            shared = torch.cat([tails[0], tails[1]], dim=0)
            sh_t = rep(shared)
            tail_t.append(sh_t)
            use_tail, bases, rots = [sh_t, sh_t, None, tail_t[3]], [0, 128, 0, 0], [64, 32, 0, 0]
            carry = [torch.roll(tails[0], -64, 0), torch.roll(tails[1], -32, 0), None, tails[3]]
        snd(region, rows_t, use_tail, use_ring, pieces, bases, rots)
        greg = _host(region)
        grings = [_host(t) for t in ring_t]
        bad = 0
        for q in range(n):
            r, c = divmod(q, tp)
            k = lanes.lane_of[q]
            j = lanes.rel(q)
            want = reg0[r, c].clone()
            want[128:] = own[r, c]
            if k == 2:
                want[:128] = greg[q].reshape(128 + q_rows, 512)[:128]  # (a lane without a segment: scratch tail)
            elif j == 0:
                want[:128] = carry[k]
            else:
                pr, pc = divmod(q - 1, tp)
                want[:128] = own[pr, pc, q_rows - 128 :]
            ok = torch.equal(greg[q].reshape(128 + q_rows, 512), want)
            if k != 2:
                wr = rings0[k].clone()
                for s0, d0, m in pieces[k]:
                    wr[d0 : d0 + m] = own[r, c, s0 : s0 + m]
                ok = ok and torch.equal(grings[k][q].reshape(192, 512), wr)
            for kk in (0, 1, 3):  # (another lane's ring untouched on this chip)
                if kk != k:
                    ok = ok and torch.equal(grings[kk][q].reshape(192, 512), rings0[kk])
            bad += int(not ok)
        print(f"PB.3A LaneSend run {it}: {bad} chips differ", flush=True)
        assert bad == 0
        for t in [region, rows_t] + tail_t + ring_t:
            ttnn.deallocate(t)
