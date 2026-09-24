# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.gdn_spec_step (fused GDN spec-verify step) vs an fp32 torch reference, on one device.

  pytest tests/ttnn/unit_tests/operations/experimental/kda/test_gdn_spec_step.py -s

Every test runs at two per-device head geometries: (Nv, Nk) = (12, 4) (TP=4 of the 27B: the a|b gate pair sits in ONE
tile) and (24, 8) (TP=2: 2*Nv = 48 gate columns span TWO tiles, so a[h] and b[h] of a head can come from different tiles).
Cases whose B*Nv exceeds the compute grid (one core per (user, head)) are skipped at (24, 8).
  GDN_GOLDEN_DIR=/path GDN_GOLDEN_MODE=dump   pytest ... -k nv12    # save the device results of the (12, 4) cases
  GDN_GOLDEN_DIR=/path GDN_GOLDEN_MODE=check  pytest ... -k nv12    # assert torch.equal against the saved ones
(the dump/check pair proves a kernel change keeps the one-tile geometry bit-identical).

Covers: (B,T) buckets vs torch (out PCC > 0.999, state PCC > 0.9999; qkvzab carries round_up(B*T, 32) logical rows so
the 'rows outside the users' rows exactly 0' check is not vacuous), the zero-filled output padding rows (bf16/fp32,
even and odd first padding row, R > round_up(B*T, 32) rejected), HOLD (held users' ring blocks and window rows
torch.equal before/after; window rows of non-held users bit-exact), window parity ping-pong, T=8 == 8 chained T=1 runs
(torch.equal; B in {1,2,4,8}, every mi at B=1, HOLD subsets), the seed-shaped [B, K, C] window pair, determinism,
hnew_depth 4 == 2, bf16 odd-row output writes at T=1, the Tier-1 comparison against ttnn.experimental.kda.gdn_decode_step
(fused conv) at T=1 (B=1: the plain op's ungrouped path; B=2: its two-users-per-core grouped path, reported).
"""

import os
import re

import pytest
import torch

import ttnn

Dk, Dv = 128, 128
K = 4
SCALE = Dk**-0.5
L2_EPS = NORM_EPS = 1e-6
HOLD = -1  # int32 bit pattern of the uint32 sentinel 0xFFFFFFFF


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if torch.equal(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def round_up32(n):
    return (n + 31) // 32 * 32


class Geo:
    """Per-device head geometry: [q(Nk*Dk) | k(Nk*Dk) | v(Nv*Dv) | z(Nv*Dv) | a(Nv) | b(Nv)] rows, a|b padded to tiles."""

    def __init__(self, Nv, Nk):
        self.Nv, self.Nk = Nv, Nk
        self.KD, self.VD = Nk * Dk, Nv * Dv
        self.C = 2 * self.KD + self.VD  # 2560 at (12, 4); 5120 at (24, 8)
        self.QKVZ = self.C + self.VD  # 4096; 8192
        self.W = self.QKVZ + round_up32(2 * Nv)  # 4128 (a|b in tile column 128); 8256 (a|b in tile columns 256, 257)
        self.id = f"nv{Nv}"


GEOS = [Geo(12, 4), Geo(24, 8)]


@pytest.fixture(params=GEOS, ids=lambda g: g.id)
def geo(request):
    return request.param


def num_cores(device):
    g = device.compute_with_storage_grid_size()
    return g.x * g.y


def skip_if_over_grid(device, g, B):
    n = num_cores(device)
    if B * g.Nv > n:
        pytest.skip(f"B*Nv = {B}*{g.Nv} = {B * g.Nv} (user, head) items exceed the {n}-core grid")


def _golden(request, **tensors):
    """GDN_GOLDEN_MODE=dump: save the device results; =check: assert torch.equal against the saved ones. One file per
    test node; a test that calls this several times (chained steps, both parities, every mi) merges its keys into that
    file, so every key must be unique within the test (the dump of a re-run overwrites key by key)."""
    d, mode = os.environ.get("GDN_GOLDEN_DIR"), os.environ.get("GDN_GOLDEN_MODE")
    if not d or not mode:
        return
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    p = os.path.join(d, name + ".pt")
    if mode == "dump":
        os.makedirs(d, exist_ok=True)
        saved = torch.load(p) if os.path.exists(p) and getattr(request.node, "_golden_open", False) else {}
        request.node._golden_open = True  # first call of this test truncates a stale file, later calls merge
        saved.update({k: v.clone().cpu() for k, v in tensors.items()})
        torch.save(saved, p)
        print(f"golden dumped: {p} ({len(saved)} tensors)")
    elif mode == "check":
        ref = torch.load(p)
        for k, v in tensors.items():
            assert k in ref, f"{name}: golden file has no {k!r} (keys {sorted(ref)}): re-dump it"
            same = torch.equal(ref[k], v.cpu())
            print(f"golden {k}: bit-identical={same}")
            assert same, f"{name}: {k} differs from the golden dump (max|d|={(ref[k].float() - v.float()).abs().max()})"


def ctrl_page(g, B, T, mi, par, hold=()):
    """[1, Npad] int32: word 0 = parity, words 1..B = mi[u], then the initial ring block per (u,h) (HOLD sentinel for
    ALL Nv heads of a held user)."""
    Nv = g.Nv
    n = 1 + B + B * Nv
    npad = (n + 15) // 16 * 16  # N*4 % 64 == 0
    words = torch.zeros(npad, dtype=torch.int32)
    words[0] = par
    for u in range(B):
        words[1 + u] = int(mi[u])
        for h in range(Nv):
            words[1 + B + u * Nv + h] = HOLD if u in hold else (int(mi[u]) * B + u) * Nv + h
    return words.reshape(1, npad)


def make_rows(g, R):
    """[R, W] fp32 rows of [q|k|v|z|a|b] (bf16-representable); rows u*T + t are the users' rows, the rest is filler."""
    Nv, C, QKVZ, VD, W = g.Nv, g.C, g.QKVZ, g.VD, g.W
    rows = torch.zeros(R, W)
    rows[:, :C] = 0.5 * torch.randn(R, C)
    rows[:, C:QKVZ] = 0.5 * torch.randn(R, VD)
    rows[:, QKVZ : QKVZ + Nv] = 0.3 * torch.randn(R, Nv)
    rows[:, QKVZ + Nv : QKVZ + 2 * Nv] = torch.randn(R, Nv)
    return rows.bfloat16().float()


def reference(g, rows, E_prev, mi, state0, taps, dtb, nea, w, B, T, hold=()):
    """fp32 torch. rows [>= B*T, W]; E_prev [B, >=Lw, C]; mi [B]; state0 [B, Nv, Dk, Dv]; taps [K, C]; dtb/nea [Nv]; w [Dv].
    Returns out [B*T, VD], states [T, B, Nv, Dk, Dv], W_new [B, Lw, C] (held users: zeros / untouched)."""
    Nv, Nk, KD, VD, C, QKVZ = g.Nv, g.Nk, g.KD, g.VD, g.C, g.QKVZ
    Lw = K - 1 + T
    rf = Nv // Nk
    out = torch.zeros(B * T, VD)
    states = torch.zeros(T, B, Nv, Dk, Dv)
    Wn_all = torch.zeros(B, Lw, C)
    for u in range(B):
        if u in hold:
            continue
        m = int(mi[u])
        Wn = torch.cat([E_prev[u, m + 1 : m + K], rows[u * T : (u + 1) * T, :C]], 0)  # [Lw, C]
        Wn_all[u] = Wn
        h = state0[u].clone()
        for t in range(T):
            x = sum(taps[j] * Wn[t + j] for j in range(K))
            conv = torch.nn.functional.silu(x)
            q = conv[:KD].reshape(Nk, Dk).repeat_interleave(rf, 0)
            k = conv[KD : 2 * KD].reshape(Nk, Dk).repeat_interleave(rf, 0)
            v = conv[2 * KD : C].reshape(Nv, Dv)
            r = rows[u * T + t]
            z = r[C:QKVZ].reshape(Nv, Dv)
            a = r[QKVZ : QKVZ + Nv]
            b = r[QKVZ + Nv : QKVZ + 2 * Nv]
            beta = torch.sigmoid(b)
            g = nea * torch.nn.functional.softplus(a + dtb, beta=1.0, threshold=20.0)
            qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_EPS) * SCALE
            kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_EPS)
            h = h * torch.exp(g)[:, None, None]
            vread = torch.einsum("hk,hkv->hv", kn, h)
            delta = beta[:, None] * (v - vread)
            h = h + torch.einsum("hk,hv->hkv", kn, delta)
            o = torch.einsum("hk,hkv->hv", qn, h)
            on = o / torch.sqrt((o * o).mean(-1, keepdim=True) + NORM_EPS) * w[None, :]
            out[u * T + t] = (on * torch.nn.functional.silu(z)).reshape(-1)
            states[t, u] = h
    return out, states, Wn_all


class Case:
    """Device tensors for one (B, T) verify step from random data. Ring block (mi[u]*B + u)*Nv + h holds state0[u,h];
    every other ring block is random (must stay untouched for held users). win[par] rows 0..Lw-1 = E_prev, the rest of
    the pair is random (rows >= even(Lw) of win[1-par] must stay untouched). qkvzab has R logical rows (default B*T;
    rows >= B*T are random filler whose output rows must come back as exact zeros). win_rows = L of the [B, L, C] pair.
    """

    def __init__(
        self, device, g, B, T, seed, mi=None, hold=(), par=0, hnew_depth=2, out_dtype=ttnn.float32, R=None, win_rows=32
    ):
        skip_if_over_grid(device, g, B)
        torch.manual_seed(seed)
        Nv, C, VD, W = g.Nv, g.C, g.VD, g.W
        self.g = g
        self.device, self.B, self.T, self.par, self.hold = device, B, T, par, set(hold)
        self.hnew_depth, self.out_dtype = hnew_depth, out_dtype
        self.Lw = K - 1 + T
        assert win_rows >= self.Lw
        self.L = win_rows
        self.BH = B * Nv
        self.R = B * T if R is None else R
        self.rows = make_rows(g, self.R)
        self.E_prev = (0.5 * torch.randn(B, self.L, C)).bfloat16().float()  # rows 0..Lw-1 are the window
        self.mi = [int(x) for x in torch.randint(0, T, (B,))] if mi is None else list(mi)
        self.state0 = 0.05 * torch.randn(B, Nv, Dk, Dv)
        self.taps = (0.3 * torch.randn(K, C)).bfloat16().float()
        self.dtb = (0.1 * torch.randn(Nv)).bfloat16().float()
        self.nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16().float()
        self.w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16().float()
        ring = 0.05 * torch.randn(T * self.BH, Dk, Dv)
        for u in range(B):
            for h in range(Nv):
                ring[(self.mi[u] * B + u) * Nv + h] = self.state0[u, h]
        self.ring0 = ring
        self.win_other0 = (0.5 * torch.randn(B, self.L, C)).bfloat16().float()
        dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=device)
        self.qkv = dev(self.rows.reshape(1, self.R, W), ttnn.bfloat16)
        wins = [None, None]
        wins[par] = dev(self.E_prev, ttnn.bfloat16)
        wins[1 - par] = dev(self.win_other0, ttnn.bfloat16)
        self.win_a, self.win_b = wins
        self.ring = dev(ring, ttnn.float32)
        self.ctrl = dev(ctrl_page(g, B, T, self.mi, par, self.hold), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.taps_d = dev(self.taps.reshape(1, K, C), ttnn.bfloat16)
        self.dtb_d = dev(self.dtb.reshape(1, 1, Nv), ttnn.float32)
        self.nea_d = dev(self.nea.reshape(1, 1, Nv), ttnn.float32)
        self.w_d = dev(self.w.reshape(1, 1, Dv), ttnn.bfloat16)

    def reference(self):
        return reference(
            self.g,
            self.rows,
            self.E_prev,
            self.mi,
            self.state0,
            self.taps,
            self.dtb,
            self.nea,
            self.w,
            self.B,
            self.T,
            self.hold,
        )

    def run(self, **kw):
        args = dict(
            conv_kernel=K,
            scale=SCALE,
            hnew_depth=self.hnew_depth,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=self.out_dtype,
        )
        args.update(kw)
        return ttnn.experimental.kda.gdn_spec_step(
            self.qkv,
            self.win_a,
            self.win_b,
            self.ring,
            self.ctrl,
            self.taps_d,
            self.dtb_d,
            self.nea_d,
            self.w_d,
            self.g.Nv,
            self.g.Nk,
            Dk,
            Dv,
            self.T,
            self.B,
            self.g.QKVZ,
            **args,
        )

    def snapshot(self, o):
        """out [R, VD] fp32, ring [T*BH, Dk, Dv] fp32, win_prev [B,L,C], win_next [B,L,C] (host copies)."""
        out = ttnn.to_torch(o).reshape(-1, self.g.VD).float()
        ring = ttnn.to_torch(self.ring).reshape(-1, Dk, Dv).float()
        wins = [ttnn.to_torch(self.win_a).float(), ttnn.to_torch(self.win_b).float()]
        return out, ring, wins[self.par], wins[1 - self.par]

    def reseed(self):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self.ring0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self.ring
        )
        wins = [self.win_a, self.win_b]
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self.win_other0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), wins[1 - self.par]
        )


def check_case(c, out, ring, win_prev, win_next, out_ref, st_ref, Wn_ref, tag="", require_pad_rows=False):
    B, T, Lw, BH = c.B, c.T, c.Lw, c.BH
    Nv, C = c.g.Nv, c.g.C
    Lw_w = (Lw + 1) // 2 * 2
    live = [u for u in range(B) if u not in c.hold]
    # outputs: PCC per live user; every row outside the users' rows (the tile padding rows [B*T, R)) exactly 0
    rows_live = sum(([u * T + t for t in range(T)] for u in live), [])
    p_out = pcc(out[rows_live], out_ref[rows_live])
    d_out = (out[rows_live] - out_ref[rows_live]).abs().max().item()
    per_user = [round(pcc(out[u * T : (u + 1) * T], out_ref[u * T : (u + 1) * T]), 6) for u in live]
    assert out.shape[0] == c.R, (out.shape, c.R)
    pad_rows = list(range(B * T, out.shape[0]))
    if require_pad_rows:
        assert pad_rows, "padding-row check would be vacuous"
    assert not pad_rows or torch.equal(
        out[pad_rows], torch.zeros_like(out[pad_rows])
    ), f"rows outside B*T not zero: {[(r, out[r].abs().max().item()) for r in pad_rows if out[r].abs().max() > 0]}"
    assert torch.isfinite(out).all(), "non-finite output"
    # states: PCC over the live users' blocks, held users' blocks untouched
    st_dev = ring.reshape(T, B, Nv, Dk, Dv)
    p_st = pcc(st_dev[:, live], st_ref[:, live])
    d_st = (st_dev[:, live] - st_ref[:, live]).abs().max().item()
    for u in c.hold:
        blocks = [(t * B + u) * Nv + h for t in range(T) for h in range(Nv)]
        assert torch.equal(ring[blocks], c.ring0[blocks]), f"held user {u}: ring blocks changed"
    # windows: win[par] untouched; win[1-par] rows 0..Lw-1 == W_new (bit-exact), rows Lw..Lw_w-1 zero, rest untouched;
    # held users: rows 0..Lw-1 copied through
    assert torch.equal(win_prev, c.E_prev), "win[par] was modified"
    for u in live:
        assert torch.equal(win_next[u, :Lw], Wn_ref[u]), f"user {u}: window rows differ from [E_prev[mi+1:mi+K]; new]"
        assert torch.equal(win_next[u, Lw:Lw_w], torch.zeros(Lw_w - Lw, C)), f"user {u}: pad row not zero"
        assert torch.equal(win_next[u, Lw_w:], c.win_other0[u, Lw_w:]), f"user {u}: rows >= {Lw_w} touched"
    for u in c.hold:
        assert torch.equal(win_next[u, :Lw], c.E_prev[u, :Lw]), f"held user {u}: window rows not copied through"
    print(
        f"{tag}(B={B},T={T},R={c.R},L={c.L},mi={c.mi},hold={sorted(c.hold)}) out pcc={p_out:.7f} max|d|={d_out:.3e} "
        f"per-user={per_user} state pcc={p_st:.7f} max|d|={d_st:.3e} (max|state|={st_ref[:, live].abs().max().item():.3f}); "
        f"{len(pad_rows)} padding rows exactly 0"
    )
    assert p_out > 0.999 and p_st > 0.9999, (p_out, p_st)
    return p_out, p_st


@pytest.mark.parametrize(
    "B,T,hold",
    [
        (1, 8, ()),
        (2, 8, ()),
        (4, 8, ()),
        (8, 4, ()),
        (8, 8, ()),
        (1, 1, ()),
        (8, 1, ()),
        (4, 8, (1, 2)),
        (8, 4, (0, 7)),
    ],
)
def test_vs_reference(device, geo, request, B, T, hold):
    # R = round_up(B*T, 32): the tile padding rows carry random filler and must come back as exact zeros
    c = Case(device, geo, B, T, seed=10 * B + T, hold=hold, par=(B + T) & 1, R=round_up32(B * T))
    out_ref, st_ref, Wn_ref = c.reference()
    o = c.run()
    out, ring, win_prev, win_next = c.snapshot(o)
    check_case(
        c, out, ring, win_prev, win_next, out_ref, st_ref, Wn_ref, tag="vs-torch ", require_pad_rows=B * T % 32 != 0
    )
    _golden(request, out=out, ring=ring, win_next=win_next)
    # determinism: same inputs -> bit-identical outputs, states and windows
    c.reseed()
    o2 = c.run()
    out2, ring2, _, win_next2 = c.snapshot(o2)
    assert torch.equal(out2, out) and torch.equal(ring2, ring) and torch.equal(win_next2, win_next), "non-deterministic"


@pytest.mark.parametrize(
    "B,T,out_dtype",
    [
        (1, 8, ttnn.bfloat16),
        (1, 1, ttnn.bfloat16),
        (1, 1, ttnn.float32),
        (2, 8, ttnn.float32),
        (8, 1, ttnn.bfloat16),
        (3, 1, ttnn.bfloat16),
    ],
)
def test_padding_rows_zero(device, geo, request, B, T, out_dtype):
    """qkvzab with 32 logical rows (random filler beyond B*T): output rows [B*T, 32) are exact zeros for bf16 and fp32
    outputs, for an even ((1,8): rows 8..31) and an odd ((1,1): 1..31; (3,1): 3..31) first padding row (the odd bf16
    case exercises 32 B-aligned zero spans, as the odd-row output writes do)."""
    c = Case(device, geo, B, T, seed=100 + 10 * B + T, out_dtype=out_dtype, R=32)
    out_ref, st_ref, Wn_ref = c.reference()
    o = c.run()
    out, ring, win_prev, win_next = c.snapshot(o)
    check_case(c, out, ring, win_prev, win_next, out_ref, st_ref, Wn_ref, tag="pad-rows ", require_pad_rows=True)
    assert out.shape[0] == 32 and torch.equal(out[B * T :], torch.zeros(32 - B * T, geo.VD))
    print(f"padding rows [{B*T}, 32) exactly 0 ({out_dtype})")
    _golden(request, out=out, ring=ring, win_next=win_next)


def test_rows_beyond_tile_row_rejected(device, geo, expect_error):
    """R > round_up(B*T, 32) would leave whole output tile rows unwritten: the op must refuse it."""
    c = Case(device, geo, 1, 8, seed=3, R=64)
    with expect_error(RuntimeError, "round_up"):
        c.run()


def test_over_grid_rejected(device, geo, expect_error):
    """B*Nv (user, head) items beyond the compute grid: refused by the host validate, not a device hang."""
    n = num_cores(device)
    B = n // geo.Nv + 1
    with expect_error(RuntimeError, "exceed"):
        ctrl = ttnn.from_torch(
            ctrl_page(geo, B, 1, [0] * B, 0), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn.experimental.kda.gdn_spec_step(
            dev(torch.zeros(1, B, geo.W), ttnn.bfloat16),
            dev(torch.zeros(B, K, geo.C), ttnn.bfloat16),
            dev(torch.zeros(B, K, geo.C), ttnn.bfloat16),
            dev(torch.zeros(B * geo.Nv, Dk, Dv), ttnn.float32),
            ctrl,
            dev(torch.zeros(1, K, geo.C), ttnn.bfloat16),
            dev(torch.zeros(1, 1, geo.Nv), ttnn.float32),
            dev(torch.zeros(1, 1, geo.Nv), ttnn.float32),
            dev(torch.ones(1, 1, Dv), ttnn.bfloat16),
            geo.Nv,
            geo.Nk,
            Dk,
            Dv,
            1,
            B,
            geo.QKVZ,
            conv_kernel=K,
            scale=SCALE,
            hnew_depth=2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=ttnn.float32,
        )


def _chained_t1(device, g, request, B, mi, hold=(), seed=77):
    """T=8 verify == 8 chained T=1 steps of the same op (window shift register as materialize_spec_state builds it):
    ring block (t*BH + bh) after the T=8 run torch.equal the chained state after token t; outputs and windows
    torch.equal. Held users: T=8 leaves their ring blocks / windows untouched (check_case), and every chained T=1 step
    leaves their rec_state blocks and window rows untouched too."""
    T = 8
    Nv, Nk, C, VD, W, QKVZ = g.Nv, g.Nk, g.C, g.VD, g.W, g.QKVZ
    c8 = Case(device, g, B, T, seed=seed, mi=mi, hold=hold, par=0)
    out_ref, st_ref, Wn_ref = c8.reference()
    o8 = c8.run()
    out8, ring8, win_prev8, win_next8 = c8.snapshot(o8)
    check_case(c8, out8, ring8, win_prev8, win_next8, out_ref, st_ref, Wn_ref, tag="T8 ")
    _golden(request, **{f"out8_{seed}": out8, f"ring8_{seed}": ring8, f"win_next8_{seed}": win_next8})
    live = [u for u in range(B) if u not in c8.hold]
    # chained T=1: ring1 = B*Nv blocks (rec_state view), window pair [B, 32, C] with rows 0..3 = W[j : j+4]
    dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=device)
    ring1 = dev(c8.state0.reshape(B * Nv, Dk, Dv), ttnn.float32)
    win0 = torch.zeros(B, 32, C)
    for u in range(B):
        win0[u, :K] = c8.E_prev[u, c8.mi[u] : c8.mi[u] + K]  # rows 1..3 = the K-1 carry rows, row 0 = don't care
    wins = [dev(win0, ttnn.bfloat16), dev(torch.zeros(B, 32, C), ttnn.bfloat16)]
    ctrls = [dev(ctrl_page(g, B, 1, [0] * B, par, hold), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) for par in (0, 1)]
    for j in range(T):
        rows_j = torch.stack([c8.rows[u * T + j] for u in range(B)]).reshape(1, B, W)
        qkv1 = dev(rows_j, ttnn.bfloat16)
        par = j & 1
        win_before = ttnn.to_torch(wins[par]).float()
        o1 = ttnn.experimental.kda.gdn_spec_step(
            qkv1,
            wins[0],
            wins[1],
            ring1,
            ctrls[par],
            c8.taps_d,
            c8.dtb_d,
            c8.nea_d,
            c8.w_d,
            Nv,
            Nk,
            Dk,
            Dv,
            1,
            B,
            QKVZ,
            conv_kernel=K,
            scale=SCALE,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=ttnn.float32,
        )
        out1 = ttnn.to_torch(o1).reshape(-1, VD).float()
        st1 = ttnn.to_torch(ring1).reshape(B, Nv, Dk, Dv).float()
        win_new = ttnn.to_torch(wins[1 - par]).float()
        _golden(request, **{f"out1_{seed}_{j}": out1, f"st1_{seed}_{j}": st1, f"win_new_{seed}_{j}": win_new})
        for u in live:
            assert torch.equal(
                st1[u], ring8[(j * B + u) * Nv : (j * B + u + 1) * Nv].reshape(Nv, Dk, Dv)
            ), f"state t={j} u={u}"
            assert torch.equal(
                out1[u], out8[u * T + j]
            ), f"out t={j} u={u}: max|d|={(out1[u] - out8[u * T + j]).abs().max()}"
            assert torch.equal(win_new[u, :K], Wn_ref[u, j : j + K]), f"window t={j} u={u}"
        for u in c8.hold:
            assert torch.equal(st1[u], c8.state0[u]), f"held user {u}: rec_state changed at chained step {j}"
            assert torch.equal(win_new[u], win_before[u]), f"held user {u}: window not copied through at step {j}"
        ttnn.deallocate(qkv1)
        ttnn.deallocate(o1)
    print(f"T8-vs-8xT1 (B={B}, mi={c8.mi}, hold={sorted(c8.hold)}): all {T} states, outputs and windows torch.equal")


@pytest.mark.parametrize("B,hold", [(1, ()), (2, ()), (4, (1,)), (8, (0, 5))])
def test_t8_vs_chained_t1(device, geo, request, B, hold):
    _chained_t1(device, geo, request, B, mi=None, hold=hold, seed=77 + B)


def test_t8_vs_chained_t1_every_mi(device, geo, request):
    """B=1: the carry rows E_prev[mi+1 : mi+K] for every mi in [0, T)."""
    for mi in range(8):
        _chained_t1(device, geo, request, 1, mi=[mi], seed=200 + mi)


def test_seed_window_shape_l4(device, geo, request):
    """Seed-shaped call: T=1 with the window pair as [B, K, C] (L = 4 = Lw), exactly what the T=1 seed passes
    (_conv_win_buf plus a [B, K, C] scratch); both parities."""
    for par in (0, 1):
        c = Case(device, geo, 2, 1, seed=300 + par, par=par, win_rows=K)
        out_ref, st_ref, Wn_ref = c.reference()
        o = c.run()
        out, ring, win_prev, win_next = c.snapshot(o)
        assert win_next.shape == (2, K, geo.C)
        check_case(c, out, ring, win_prev, win_next, out_ref, st_ref, Wn_ref, tag=f"seed-L4 par={par} ")
        _golden(request, **{f"out_{par}": out, f"ring_{par}": ring, f"win_next_{par}": win_next})


def _pack_rows(g, rows, parity=0, both=False):
    """4 torch [C] rows -> [Nv, 4, 32, 32] packed head tiles (chunk c in row 2c + parity), as gdn_decode_step's conv mode
    (user b of a batched call uses parity b & 1)."""
    Nv, Nk, KD = g.Nv, g.Nk, g.KD
    rf = Nv // Nk
    out = torch.zeros(Nv, 4, 32, 32, dtype=torch.bfloat16)
    for h in range(Nv):
        hk = h // rf
        for j, r in enumerate(rows):
            r = r.reshape(-1).to(torch.bfloat16)
            chunks = torch.cat(
                [
                    r[hk * Dk : (hk + 1) * Dk],
                    r[KD + hk * Dk : KD + (hk + 1) * Dk],
                    r[2 * KD + h * Dv : 2 * KD + (h + 1) * Dv],
                ]
            ).reshape(-1, 32)
            n = chunks.shape[0]
            for par in (0, 1) if both else (parity,):
                out[h, j, par : 2 * n + par : 2, :] = chunks
    return out


def _run_plain(device, c, users):
    """ttnn.experimental.kda.gdn_decode_step (fused conv) on the given users of Case c (T = 1, mi = 0 for all): qkv rows
    = the users' rows, packed history of user b at parity b & 1 (as the plain reader packs the new row). Returns
    out [len(users), VD], state [len(users), Nv, Dk, Dv], hist after the op [len(users), Nv, 4, 32, 32]."""
    dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    g = c.g
    Nv, Nk, VD, W, QKVZ = g.Nv, g.Nk, g.VD, g.W, g.QKVZ
    n = len(users)
    qkv = dev(torch.stack([c.rows[u] for u in users]).reshape(1, n, W), ttnn.bfloat16)
    state = dev(torch.stack([c.state0[u] for u in users]).reshape(n, Nv, Dk, Dv), ttnn.float32)
    hist = dev(
        torch.stack(
            [_pack_rows(g, [c.E_prev[u, j].bfloat16() for j in range(4)], parity=b & 1) for b, u in enumerate(users)]
        ),
        ttnn.bfloat16,
    )  # slots 0..3; the plain op reads slots 1..3 as history
    taps = dev(_pack_rows(g, [c.taps[j].bfloat16() for j in range(K)], both=True), ttnn.bfloat16)
    o = ttnn.experimental.kda.gdn_decode_step(
        qkv,
        c.dtb_d,
        c.nea_d,
        state,
        c.w_d,
        Nv,
        Nk,
        Dk,
        Dv,
        scale=SCALE,
        output_dtype=ttnn.float32,
        conv_hist=hist,
        conv_taps=taps,
        qkvz_dim=QKVZ,
    )
    out_p = ttnn.to_torch(o).reshape(-1, VD).float()[:n]
    st_p = ttnn.to_torch(state).reshape(n, Nv, Dk, Dv).float()
    hist_t = ttnn.to_torch(hist).reshape(n, Nv, 4, 32, 32)
    for t in (qkv, state, hist, taps, o):
        ttnn.deallocate(t)
    return out_p, st_p, hist_t


def _cmp(tag, a, b):
    eq = torch.equal(a, b)
    d = (a - b).abs().max().item()
    n = (a != b).sum().item()
    print(f"  {tag}: equal={eq} max|d|={d:.3e} ({n}/{a.numel()} differ)")
    return eq, d, n


@pytest.mark.parametrize("seed", [5, 6, 8, 9, 11])
def test_tier1_vs_gdn_decode_step_conv(device, geo, request, seed):
    """Plain-family identity (ungrouped path): gdn_spec_step(T=1, B=1) vs ttnn.experimental.kda.gdn_decode_step(conv_hist,
    conv_taps) at B=1 on identical inputs (history = E_prev rows 1..3, i.e. mi = 0): out and state must be bit-identical
    (torch.equal)."""
    B, T = 1, 1
    Nv = geo.Nv
    c = Case(device, geo, B, T, seed=seed, mi=[0], par=0)
    out_ref, st_ref, Wn_ref = c.reference()
    o = c.run()
    out, ring, _, win_next = c.snapshot(o)
    check_case(c, out, ring, c.E_prev, win_next, out_ref, st_ref, Wn_ref, tag="tier1 spec ")
    _golden(request, out=out, ring=ring, win_next=win_next)
    out_p, st_p, hist_t = _run_plain(device, c, [0])
    st_s = ring[:Nv].reshape(Nv, Dk, Dv)
    print(
        f"TIER1 seed={seed} gdn_spec_step(T=1,B=1) vs gdn_decode_step(conv, B=1); plain vs torch out pcc={pcc(out_p[0], out_ref[0]):.7f}"
    )
    eq_out, d_out, n_out = _cmp("out", out[0], out_p[0])
    eq_st, d_st, n_st = _cmp("state", st_s, st_p[0])
    # the shifted history the plain op stored == our new window rows 1..3 + the new row
    assert torch.equal(
        hist_t[0], _pack_rows(geo, [win_next[0, j].bfloat16() for j in range(4)])
    ), "window vs shifted history"
    assert eq_out and eq_st, (n_out, d_out, n_st, d_st)


@pytest.mark.parametrize("seed", [21, 22])
def test_tier1_b2_vs_grouped_gdn_decode_step(device, geo, request, seed):
    """Grouped plain path: at B >= 2 gdn_decode_step's factory puts two users per core (gs = 2) and merges their output
    rows with an FPU add (add_tiles_n(out_acc, tmp)) before the copies to `out`. gdn_spec_step(T=1, B=2) is compared
    three ways: vs the grouped plain op at B=2, vs the ungrouped plain op run once per user (B=1), and the plain op
    against itself (B=2 grouped vs B=1). The B=1 form is the reference family (Tier-1 identity); this test REPORTS the
    grouped-path result and asserts only the identity with the ungrouped per-user runs plus closeness (PCC > 0.9999)
    for the grouped path."""
    B, T = 2, 1
    Nv, VD = geo.Nv, geo.VD
    c = Case(device, geo, B, T, seed=seed, mi=[0, 0], par=0)
    out_ref, st_ref, Wn_ref = c.reference()
    o = c.run()
    out, ring, _, win_next = c.snapshot(o)
    check_case(c, out, ring, c.E_prev, win_next, out_ref, st_ref, Wn_ref, tag="tier1-b2 spec ")
    _golden(request, out=out, ring=ring, win_next=win_next)
    st_s = ring[: B * Nv].reshape(B, Nv, Dk, Dv)
    out_g, st_g, hist_g = _run_plain(device, c, [0, 1])  # grouped: 2 users per core
    out_1 = torch.stack([_run_plain(device, c, [u])[0][0] for u in range(B)])  # ungrouped, per user
    st_1 = torch.stack([_run_plain(device, c, [u])[1][0] for u in range(B)])
    print(
        f"TIER1-B2 seed={seed}: gdn_spec_step(T=1,B=2) [spec], gdn_decode_step B=2 grouped [plain2], per-user B=1 [plain1]"
    )
    r_sp2 = [_cmp(f"spec vs plain2 out u={u}", out[u], out_g[u]) for u in range(B)]
    r_sp2s = [_cmp(f"spec vs plain2 state u={u}", st_s[u], st_g[u]) for u in range(B)]
    r_sp1 = [_cmp(f"spec vs plain1 out u={u}", out[u], out_1[u]) for u in range(B)]
    r_sp1s = [_cmp(f"spec vs plain1 state u={u}", st_s[u], st_1[u]) for u in range(B)]
    r_p21 = [_cmp(f"plain2 vs plain1 out u={u}", out_g[u], out_1[u]) for u in range(B)]
    r_p21s = [_cmp(f"plain2 vs plain1 state u={u}", st_g[u], st_1[u]) for u in range(B)]
    for u in range(B):
        assert torch.equal(
            hist_g[u], _pack_rows(geo, [win_next[u, j].bfloat16() for j in range(4)], parity=u & 1)
        ), f"hist u={u}"
    # summary line for the results file
    print(
        f"TIER1-B2 SUMMARY seed={seed}: spec==plain1 out {all(r[0] for r in r_sp1)} state {all(r[0] for r in r_sp1s)}; "
        f"spec==plain2(grouped) out {all(r[0] for r in r_sp2)} (max|d| {max(r[1] for r in r_sp2):.3e}, "
        f"{sum(r[2] for r in r_sp2)}/{B*VD} differ) state {all(r[0] for r in r_sp2s)} (max|d| {max(r[1] for r in r_sp2s):.3e}); "
        f"plain2==plain1 out {all(r[0] for r in r_p21)} state {all(r[0] for r in r_p21s)}"
    )
    assert all(r[0] for r in r_sp1) and all(
        r[0] for r in r_sp1s
    ), "spec(B=2) must equal the ungrouped plain op per user"
    assert pcc(out, out_g) > 0.9999 and pcc(st_s, st_g) > 0.9999


def test_bf16_odd_row_output(device, geo, request):
    """T = 1 with 8 users (4 where 8*Nv exceeds the grid): rows u (odd for odd users) are written as 32 B single
    face-rows in bf16."""
    B = 8 if 8 * geo.Nv <= num_cores(device) else 4
    c = Case(device, geo, B, 1, seed=31, out_dtype=ttnn.bfloat16, par=1)
    out_ref, st_ref, Wn_ref = c.reference()
    o = c.run()
    out, ring, win_prev, win_next = c.snapshot(o)
    p_out, _ = check_case(c, out, ring, win_prev, win_next, out_ref, st_ref, Wn_ref, tag="bf16-odd-rows ")
    per_user = [pcc(out[u], out_ref[u]) for u in range(B)]
    print(f"bf16 ({B},1) per-user out pcc: {[round(p, 6) for p in per_user]}")
    assert min(per_user) > 0.999
    _golden(request, out=out, ring=ring, win_next=win_next)


def test_hnew_depth4(device, geo, request):
    c2 = Case(device, geo, 4, 8, seed=41, hnew_depth=2)
    o2 = c2.run()
    out2, ring2, _, win2 = c2.snapshot(o2)
    c4 = Case(device, geo, 4, 8, seed=41, hnew_depth=4)
    o4 = c4.run()
    out4, ring4, _, win4 = c4.snapshot(o4)
    assert torch.equal(out2, out4) and torch.equal(ring2, ring4) and torch.equal(win2, win4)
    print("hnew_depth 4 == 2: torch.equal (out, ring, window)")
    _golden(request, out=out2, ring=ring2, win=win2)
