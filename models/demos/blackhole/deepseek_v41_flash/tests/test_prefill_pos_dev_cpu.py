# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41_PF_POS_DEV (tt/prefill_dyn.py): the compact uploads + the device-side expansion (``derive``) must equal the host-built full-width position tables BIT FOR BIT.

CPU only: ``DynCtx._values`` of the default path (full [C,512] RoPE tables, band mask) is compared with the compact values after the same expansion the trace performs ([ones | cos], [zeros | sin],
[-0.0 | -sin]; select(first, M0, M1)), for many chunk starts (0, 512, ..., 1M - 512, ratios 1 and 2, idle-slot style repeated starts), after the bf16 rounding ``from_torch`` applies.
"""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import prefill_dyn as PD

HEAD_DIM, ROPE_DIM = PD.HEAD_DIM, PD.ROPE_DIM


class FakeAttn:
    """the table interface of tt/attention.py (cos_tab / sin_tab [S, 64] fp32, ``_rope_inputs``)"""

    def __init__(self, S, seed):
        g = torch.Generator().manual_seed(seed)
        ang = torch.rand(S, 32, generator=g) * 50.0
        self.cos_tab = torch.cos(ang).repeat_interleave(2, dim=-1)
        self.sin_tab = torch.sin(ang).repeat_interleave(2, dim=-1)

    def _rope_inputs(self, positions):
        B = positions.shape[0]
        cos, sin = self.cos_tab[positions], self.sin_tab[positions]
        c = torch.ones(B, HEAD_DIM)
        s = torch.zeros(B, HEAD_DIM)
        c[:, HEAD_DIM - ROPE_DIM :], s[:, HEAD_DIM - ROPE_DIM :] = cos, sin
        return c, s


def make_ctx(C, ratios, S, pos_dev):
    ctx = PD.DynCtx.__new__(PD.DynCtx)
    ctx.C, ctx.S_pad = C, S
    ctx.rope_src = {False: FakeAttn(S, 1), True: FakeAttn(S, 2)}
    ctx.ratios = sorted(r for r in ratios if r)
    ctx.L = {r: S // r for r in ctx.ratios}
    ctx.lat_tabs = {r: None for r in ctx.ratios if r > 1}
    ctx.lpk = dict(ctx.lat_tabs)
    ctx.pos_dev = pos_dev
    return ctx


def bf(t):
    return t.to(torch.bfloat16).view(torch.int16)  # bit patterns


def expand(pk):
    n = pk.shape[2]
    c, s, ns = (pk[..., i * ROPE_DIM : (i + 1) * ROPE_DIM] for i in range(3))
    z = torch.zeros(1, 1, n, HEAD_DIM - ROPE_DIM)
    return torch.cat([z + 1, c], -1), torch.cat([z, s], -1), torch.cat([-z, ns], -1)


def check(C, ratios, S, starts):
    old, new = make_ctx(C, ratios, S, False), make_ctx(C, ratios, S, True)
    m0, m1 = bf(new._win_values(0)), bf(new._win_values(C))
    for s0 in starts:
        vo, vn = old._values(s0), new._values(s0)
        for k in old.rope_src:
            for a, b in zip(vo[("tab", k)], expand(vn[("tab", k)][0])):
                assert torch.equal(bf(a), bf(b)), (k, s0)
        for r in old.lat_tabs:
            for a, b in zip(vo[("lat", r)], expand(vn[("lat", r)][0])):
                assert torch.equal(bf(a), bf(b)), (r, s0)
        first = vn["first"][0].item()
        assert torch.equal(bf(vo["win"][0]), m0 if first else m1), s0
        assert first == (1.0 if s0 == 0 else 0.0)
        for r in old.ratios:  # lim / off are unchanged
            for key in (("lim", r), ("off", r)):
                assert torch.equal(vo[key][0], vn[key][0])


def test_pos_dev_equals_host_tables():
    C = 512
    S = 1048576 + 512
    starts = [0, 512, 1024, 1536, 4096, 65536, 262144, 524288, 1048064, 1048576 - 512, 331776]
    check(C, {0, 1, 2}, S, starts)
    check(C, {2}, S, starts)
    check(256, {0, 2}, 65536, [0, 256, 512, 65536 - 256])


def test_pos_dev_select_masks_two_valued():
    C = 512
    ctx = make_ctx(C, {2}, 4096, True)
    ref = {s0: bf(ctx._win_values(s0)) for s0 in range(0, 8192, 128)}
    assert torch.equal(ref[128], ref[4096]) and torch.equal(ref[128], ref[C]), "mask must be constant for s0 >= 128"
    assert not torch.equal(ref[0], ref[128])


def test_pos_dev_rejects_unaligned_start():
    ctx = make_ctx(512, {2}, 4096, True)
    try:
        ctx._values(100)
    except AssertionError:
        return
    raise AssertionError("an unaligned chunk start must be refused")
