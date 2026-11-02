# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.gdn_spec_step (fused GDN spec-verify step) vs an fp32 torch reference, on one device.

  pytest tests/ttnn/unit_tests/operations/experimental/kda/test_gdn_spec_step.py -s

Every test runs at three per-device head geometries: (Nv, Nk) = (6, 2) (TP=8 of the 27B on a Galaxy column: rf = 3,
the a|b gate pair in ONE tile), (12, 4) (TP=4: one tile) and (24, 8) (TP=2: 2*Nv = 48 gate columns span TWO tiles, so
a[h] and b[h] of a head can come from different tiles). Cases whose B*Nv exceeds the compute grid (one core per
(user, head)) are skipped at (24, 8).
Every test also runs in the op's three arithmetic modes: ``plain`` (the original kernels: SFPU gates, SrcA-copied
state), ``fast`` (fast_mode=True: gdn_decode_step's reader-side fp32 gates and dual-packed state add) and
``fast_fp32`` (fast_mode=True, fp32_state=True: plus the exact-fp32 SFPU state decay / add). The Tier-1 tests compare
each mode against gdn_decode_step in the SAME mode (torch.equal); ``fast_fp32`` is what the qwen36 plain decode and
QWEN36_VERIFY_SPEC_STEP=1 verify run. They skip at (24, 8): this tree's gdn_decode_step reads a|b from one tile
(2*Nv <= 32), so the two-tile geometry has no Tier-1 reference.
  GDN_GOLDEN_DIR=/path GDN_GOLDEN_MODE=dump   pytest ... -k nv12    # save the device results of the (12, 4) cases
  GDN_GOLDEN_DIR=/path GDN_GOLDEN_MODE=check  pytest ... -k nv12    # assert torch.equal against the saved ones
(the dump/check pair proves a kernel change keeps the one-tile geometry bit-identical).

Covers: (B,T) buckets vs torch (out PCC > 0.999, state PCC > 0.9999; qkvzab carries round_up(B*T, 32) logical rows so
the 'rows outside the users' rows exactly 0' check is not vacuous), the zero-filled output padding rows (bf16/fp32,
even and odd first padding row, R > round_up(B*T, 32) rejected), HOLD (held users' ring blocks and window rows
torch.equal before/after; window rows of non-held users bit-exact), window parity ping-pong, T=8 == 8 chained T=1 runs
(torch.equal; B in {1,2,4,8}, every mi at B=1, HOLD subsets), the seed-shaped [B, K, C] window pair, determinism,
hnew_depth 4 == 2, bf16 odd-row output writes at T=1, the Tier-1 comparison against ttnn.experimental.kda.gdn_decode_step
(fused conv) at T=1 (B=1: the plain op's ungrouped path; B=2: its two-users-per-core grouped path, reported),
value_blocks 2 / 4 == 1 (torch.equal: out, ring, windows) and the value_blocks timing table (report only).
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


GEOS = [Geo(6, 2), Geo(12, 4), Geo(24, 8)]


@pytest.fixture(params=GEOS, ids=lambda g: g.id)
def geo(request):
    return request.param


class Mode:
    """The op's arithmetic mode: (fast_mode, fp32_state) kwargs of gdn_spec_step AND of the gdn_decode_step it must
    match bit for bit."""

    def __init__(self, id, fast_mode, fp32_state):
        self.id, self.fast_mode, self.fp32_state = id, fast_mode, fp32_state

    @property
    def kw(self):
        return dict(fast_mode=self.fast_mode, fp32_state=self.fp32_state)


MODES = [Mode("plain", False, False), Mode("fast", True, False), Mode("fast_fp32", True, True)]


@pytest.fixture(params=MODES, ids=lambda m: m.id)
def mode(request):
    return request.param


def num_cores(device):
    g = device.compute_with_storage_grid_size()
    return g.x * g.y


def skip_if_over_grid(device, g, B):
    n = num_cores(device)
    if B * g.Nv > n:
        pytest.skip(f"B*Nv = {B}*{g.Nv} = {B * g.Nv} (user, head) items exceed the {n}-core grid")


def skip_if_no_decode_step(g):
    """This tree's gdn_decode_step (the Tier-1 reference) reads the a|b gate pair from ONE tile: its validate rejects
    2*Nv > TILE_WIDTH (gdn_decode_step_device_operation.cpp, "2*Nv <= 32 (a|b in one tile)"). gdn_spec_step itself
    runs the two-tile (24, 8) geometry; only the identity against the decode op has no reference there."""
    if 2 * g.Nv > 32:
        pytest.skip(f"gdn_decode_step needs 2*Nv <= 32 (a|b in one tile); 2*{g.Nv} = {2 * g.Nv} -> no Tier-1 reference")


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
        self,
        device,
        g,
        B,
        T,
        seed,
        mi=None,
        hold=(),
        par=0,
        hnew_depth=2,
        out_dtype=ttnn.float32,
        R=None,
        win_rows=32,
        mode=MODES[0],
    ):
        skip_if_over_grid(device, g, B)
        torch.manual_seed(seed)
        Nv, C, VD, W = g.Nv, g.C, g.VD, g.W
        self.g = g
        self.mode = mode
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
            **self.mode.kw,
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
        (4, 4, ()),  # a T = 4 case that fits (24, 8) too: the two-tile a|b path at a second T
        (8, 8, ()),
        (1, 1, ()),
        (8, 1, ()),
        (4, 8, (1, 2)),
        (8, 4, (0, 7)),
        # the qwen36 engine's buckets (B*(K+1) <= 32): K=3/7/11 -> T=4/8/12
        (1, 4, ()),
        (1, 12, ()),
        (2, 12, ()),
        (3, 8, ()),
        (5, 4, ()),
    ],
)
def test_vs_reference(device, geo, mode, request, B, T, hold):
    # R = round_up(B*T, 32): the tile padding rows carry random filler and must come back as exact zeros
    c = Case(device, geo, B, T, seed=10 * B + T, hold=hold, par=(B + T) & 1, R=round_up32(B * T), mode=mode)
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
def test_padding_rows_zero(device, geo, mode, request, B, T, out_dtype):
    """qkvzab with 32 logical rows (random filler beyond B*T): output rows [B*T, 32) are exact zeros for bf16 and fp32
    outputs, for an even ((1,8): rows 8..31) and an odd ((1,1): 1..31; (3,1): 3..31) first padding row (the odd bf16
    case exercises 32 B-aligned zero spans, as the odd-row output writes do)."""
    c = Case(device, geo, B, T, seed=100 + 10 * B + T, out_dtype=out_dtype, R=32, mode=mode)
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


def test_fp32_state_needs_fast_mode(device, geo, expect_error):
    """fp32_state=True without fast_mode is refused by the host (as gdn_decode_step does)."""
    c = Case(device, geo, 1, 1, seed=4)
    with expect_error(RuntimeError, "fp32_state needs fast_mode"):
        c.run(fast_mode=False, fp32_state=True)


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


def _chained_t1(device, g, request, B, mi, hold=(), seed=77, T=8, mode=MODES[0], value_blocks=None, kernel_opts=None):
    """T-token verify == T chained T=1 steps of the same op (window shift register as the seed path builds it):
    ring block (t*BH + bh) after the T-token run torch.equal the chained state after token t; outputs and windows
    torch.equal. Held users: the T-token run leaves their ring blocks / windows untouched (check_case), and every
    chained T=1 step leaves their rec_state blocks and window rows untouched too."""
    Nv, Nk, C, VD, W, QKVZ = g.Nv, g.Nk, g.C, g.VD, g.W, g.QKVZ
    c8 = Case(device, g, B, T, seed=seed, mi=mi, hold=hold, par=0, mode=mode)
    out_ref, st_ref, Wn_ref = c8.reference()
    vb_kw = {} if value_blocks is None else dict(value_blocks=value_blocks)
    if kernel_opts is not None:
        vb_kw["kernel_opts"] = kernel_opts
    o8 = c8.run(**vb_kw)
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
            **mode.kw,
            **vb_kw,
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
    print(
        f"T{T}-vs-{T}xT1 [{mode.id}] (B={B}, mi={c8.mi}, hold={sorted(c8.hold)}): all {T} states, outputs and windows "
        "torch.equal"
    )


@pytest.mark.parametrize("B,hold", [(1, ()), (2, ()), (4, (1,)), (8, (0, 5))])
def test_t8_vs_chained_t1(device, geo, mode, request, B, hold):
    _chained_t1(device, geo, request, B, mi=None, hold=hold, seed=77 + B, mode=mode)


@pytest.mark.parametrize("B,T", [(8, 4), (1, 12), (2, 12), (1, 4)])
def test_tn_vs_chained_t1_engine_shapes(device, geo, mode, request, B, T):
    """The qwen36 engine's (B, T) buckets (K = 3 / 11): a T-token verify == T chained T=1 steps, in every mode."""
    _chained_t1(device, geo, request, B, mi=None, seed=500 + 10 * B + T, T=T, mode=mode)


def test_t8_vs_chained_t1_every_mi(device, geo, mode, request):
    """B=1: the carry rows E_prev[mi+1 : mi+K] for every mi in [0, T)."""
    for mi in range(8):
        _chained_t1(device, geo, request, 1, mi=[mi], seed=200 + mi, mode=mode)


def test_seed_window_shape_l4(device, geo, mode, request):
    """Seed-shaped call: T=1 with the window pair as [B, K, C] (L = 4 = Lw), exactly what the T=1 seed passes
    (_conv_win_buf plus a [B, K, C] scratch); both parities."""
    for par in (0, 1):
        c = Case(device, geo, 2, 1, seed=300 + par, par=par, win_rows=K, mode=mode)
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
    """ttnn.experimental.kda.gdn_decode_step (fused conv) on the given users of Case c (T = 1, mi = 0 for all), in the
    Case's arithmetic mode (fast_mode / fp32_state): qkv rows = the users' rows, packed history of user b at parity
    b & 1 (as the plain reader packs the new row). Returns out [len(users), VD], state [len(users), Nv, Dk, Dv], hist
    after the op [len(users), Nv, 4, 32, 32]."""
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
        **c.mode.kw,
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
def test_tier1_vs_gdn_decode_step_conv(device, geo, mode, request, seed):
    """Plain-family identity (ungrouped path): gdn_spec_step(T=1, B=1) vs ttnn.experimental.kda.gdn_decode_step(conv_hist,
    conv_taps) at B=1 on identical inputs (history = E_prev rows 1..3, i.e. mi = 0), both in the same arithmetic mode:
    out and state must be bit-identical (torch.equal). ``fast_fp32`` is the qwen36 plain decode's mode."""
    skip_if_no_decode_step(geo)
    B, T = 1, 1
    Nv = geo.Nv
    c = Case(device, geo, B, T, seed=seed, mi=[0], par=0, mode=mode)
    out_ref, st_ref, Wn_ref = c.reference()
    o = c.run()
    out, ring, _, win_next = c.snapshot(o)
    check_case(c, out, ring, c.E_prev, win_next, out_ref, st_ref, Wn_ref, tag="tier1 spec ")
    _golden(request, out=out, ring=ring, win_next=win_next)
    out_p, st_p, hist_t = _run_plain(device, c, [0])
    st_s = ring[:Nv].reshape(Nv, Dk, Dv)
    print(
        f"TIER1 [{mode.id}] seed={seed} gdn_spec_step(T=1,B=1) vs gdn_decode_step(conv, B=1); plain vs torch out "
        f"pcc={pcc(out_p[0], out_ref[0]):.7f}"
    )
    eq_out, d_out, n_out = _cmp("out", out[0], out_p[0])
    eq_st, d_st, n_st = _cmp("state", st_s, st_p[0])
    # the shifted history the plain op stored == our new window rows 1..3 + the new row
    assert torch.equal(
        hist_t[0], _pack_rows(geo, [win_next[0, j].bfloat16() for j in range(4)])
    ), "window vs shifted history"
    assert eq_out and eq_st, (n_out, d_out, n_st, d_st)


@pytest.mark.parametrize("seed", [21, 22])
def test_tier1_b2_vs_grouped_gdn_decode_step(device, geo, mode, request, seed):
    """Grouped plain path: at B >= 2 gdn_decode_step's factory puts two users per core (gs = 2) and merges their output
    rows with an FPU add (add_tiles_n(out_acc, tmp)) before the copies to `out`. gdn_spec_step(T=1, B=2) is compared
    three ways: vs the grouped plain op at B=2, vs the ungrouped plain op run once per user (B=1), and the plain op
    against itself (B=2 grouped vs B=1). The B=1 form is the reference family (Tier-1 identity); this test REPORTS the
    grouped-path result and asserts only the identity with the ungrouped per-user runs plus closeness (PCC > 0.9999)
    for the grouped path."""
    skip_if_no_decode_step(geo)
    B, T = 2, 1
    Nv, VD = geo.Nv, geo.VD
    c = Case(device, geo, B, T, seed=seed, mi=[0, 0], par=0, mode=mode)
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
        f"TIER1-B2 [{mode.id}] seed={seed}: gdn_spec_step(T=1,B=2) [spec], gdn_decode_step B=2 grouped [plain2], "
        "per-user B=1 [plain1]"
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


def test_bf16_odd_row_output(device, geo, mode, request):
    """T = 1 with 8 users (4 where 8*Nv exceeds the grid): rows u (odd for odd users) are written as 32 B single
    face-rows in bf16."""
    B = 8 if 8 * geo.Nv <= num_cores(device) else 4
    c = Case(device, geo, B, 1, seed=31, out_dtype=ttnn.bfloat16, par=1, mode=mode)
    out_ref, st_ref, Wn_ref = c.reference()
    o = c.run()
    out, ring, win_prev, win_next = c.snapshot(o)
    p_out, _ = check_case(c, out, ring, win_prev, win_next, out_ref, st_ref, Wn_ref, tag="bf16-odd-rows ")
    per_user = [pcc(out[u], out_ref[u]) for u in range(B)]
    print(f"bf16 ({B},1) per-user out pcc: {[round(p, 6) for p in per_user]}")
    assert min(per_user) > 0.999
    _golden(request, out=out, ring=ring, win_next=win_next)


def test_hnew_depth4(device, geo, mode, request):
    """hnew_depth 4 == 2 bit for bit (the depth only buffers states for the writer); at fast_fp32 the depth-4 program
    is also the largest L1 footprint the qwen36 wiring uses (B >= 4: hd32 + hn32 + 4 x KV hnew tiles)."""
    c2 = Case(device, geo, 4, 8, seed=41, hnew_depth=2, mode=mode)
    o2 = c2.run()
    out2, ring2, _, win2 = c2.snapshot(o2)
    c4 = Case(device, geo, 4, 8, seed=41, hnew_depth=4, mode=mode)
    o4 = c4.run()
    out4, ring4, _, win4 = c4.snapshot(o4)
    assert torch.equal(out2, out4) and torch.equal(ring2, ring4) and torch.equal(win2, win4)
    print("hnew_depth 4 == 2: torch.equal (out, ring, window)")
    _golden(request, out=out2, ring=ring2, win=win2)


# --------------------------------------------------------------------------- #
# value_blocks: the column split of each (user, head)'s state over up to 4 cores (QWEN36_SPEC_STEP_VALUE_BLOCKS)
# --------------------------------------------------------------------------- #
def effective_value_blocks(device, g, B, requested):
    """The factory's choice: the largest divisor of Dv/32 <= requested (<= 4) whose B*Nv*blocks cores fit the grid."""
    n, vt = num_cores(device), Dv // 32
    for vb in range(min(requested, vt, 4), 1, -1):
        if vt % vb == 0 and B * g.Nv * vb <= n:
            return vb
    return 1


@pytest.mark.parametrize("vb", [2, 4])
@pytest.mark.parametrize(
    "B,T,hold,out_dtype",
    [
        (8, 4, (), ttnn.bfloat16),  # the qwen36 TP=8 B=8 verify bucket (2 blocks on a 13x10 grid)
        (4, 4, (1,), ttnn.bfloat16),
        (2, 4, (), ttnn.float32),
        (1, 4, (), ttnn.bfloat16),
        (8, 1, (0, 5), ttnn.bfloat16),  # the seed shape, odd single-row output writes
        (1, 1, (), ttnn.float32),
        (2, 12, (), ttnn.bfloat16),
    ],
)
def test_value_blocks_bit_identical(device, geo, mode, request, vb, B, T, hold, out_dtype):
    """value_blocks = vb vs 1: out, every ring block (incl. held users' untouched blocks) and both window halves
    torch.equal, at the parity 1 half of the pair too."""
    skip_if_over_grid(device, geo, B)
    eff = effective_value_blocks(device, geo, B, vb)
    if geo.Nv == 6 and B == 8:
        # The qwen36 TP=8 B=8 verify bucket must actually split (2 blocks: 8 * 6 * 4 cores does not fit the grid).
        assert eff == 2, f"TP=8 B=8 value_blocks={vb} resolved to {eff}, expected 2"
    res = []
    for blocks in (1, vb):
        c = Case(device, geo, B, T, seed=91 + B * 10 + T, hold=hold, par=1, out_dtype=out_dtype, mode=mode)
        o = c.run(value_blocks=blocks)
        res.append(c.snapshot(o))
    (out1, ring1, wp1, wn1), (outv, ringv, wpv, wnv) = res
    for name, a, b in (("out", out1, outv), ("ring", ring1, ringv), ("win_prev", wp1, wpv), ("win_next", wn1, wnv)):
        assert torch.equal(a, b), f"value_blocks={vb} (effective {eff}) {name} differs from value_blocks=1"
    print(f"value_blocks {vb} (effective {eff}) == 1 at (B={B}, T={T}, hold={hold}): torch.equal (out, ring, windows)")
    # and still right against the fp32 reference
    c = Case(device, geo, B, T, seed=91 + B * 10 + T, hold=hold, par=1, out_dtype=out_dtype, mode=mode)
    out_ref, st_ref, Wn_ref = c.reference()
    check_case(c, outv, ringv, wpv, wnv, out_ref, st_ref, Wn_ref, tag=f"vb{vb} ")


def test_value_blocks_t8_vs_chained_t1(device, geo, request):
    """A T = 8 step at value_blocks = 4 equals 8 chained T = 1 steps at value_blocks = 4 (torch.equal), in the mode the
    qwen36 wiring runs (fast_fp32)."""
    _chained_t1(device, geo, request, 2, mi=[1, 3], hold=(1,), mode=MODES[2], value_blocks=4)


@pytest.mark.parametrize("B,T", [(8, 4), (4, 4), (2, 4), (1, 4), (8, 1)])
def test_value_blocks_timing(device, request, B, T):
    """Wall time per call (synchronized, 50 calls after 5 warm-ups) at value_blocks 1, 2, 4 in the qwen36 mode
    (fast_fp32, bf16 output, hnew_depth 2) and TP=8 geometry. Report only: the A/B for QWEN36_SPEC_STEP_VALUE_BLOCKS.
    Run with -s to see the table; a Tracy device profile is the authoritative per-op number."""
    import time

    g = GEOS[0]
    skip_if_over_grid(device, g, B)
    times = {}
    for vb in (1, 2, 4):
        c = Case(device, g, B, T, seed=7, out_dtype=ttnn.bfloat16, mode=MODES[2])
        for _ in range(5):
            ttnn.deallocate(c.run(value_blocks=vb))
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(50):
            ttnn.deallocate(c.run(value_blocks=vb))
        ttnn.synchronize_device(device)
        times[vb] = (time.perf_counter() - t0) / 50 * 1e6
    effs = {vb: effective_value_blocks(device, g, B, vb) for vb in times}
    print(
        f"gdn_spec_step (B={B}, T={T}) us/call: "
        + ", ".join(f"vb{vb}(eff {effs[vb]})={t:.1f}" for vb, t in times.items())
    )


# --------------------------------------------------------------------------- #
# kernel_opts: bit-identical kernel restructurings of the fast_mode + fp32_state program (QWEN36_SPEC_STEP_OPT)
# --------------------------------------------------------------------------- #
# 1 fused next-token decay, 2 gate scalars, 4 early src, 8 hoisted q scale, 16 q/k share (value_blocks > 1),
# 32 half-tile conv SFPU tail, 64 gate split (with 2, value_blocks > 1), 128 own-column tail (value_blocks > 1);
# 250 = the recommended QWEN36_SPEC_STEP_OPT (all but 1 and 4). All eight bits at once (255) does not fit the 70656 B
# kernel config buffer at value_blocks 2 (B = 8) or at T = 12.
KERNEL_OPTS = [1, 2, 4, 8, 16, 32, 66, 128, 250]


@pytest.mark.parametrize("opts", KERNEL_OPTS)
@pytest.mark.parametrize("vb", [1, 2, 4])
@pytest.mark.parametrize(
    "B,T,hold,out_dtype",
    [
        (8, 4, (), ttnn.bfloat16),  # the qwen36 TP=8 B=8 verify bucket
        (4, 8, (1,), ttnn.bfloat16),  # the 4-user bucket (K = 7)
        (8, 1, (0, 5), ttnn.bfloat16),  # the seed shape
        (1, 1, (), ttnn.float32),
        (2, 12, (), ttnn.bfloat16),
        (4, 4, (), ttnn.float32),
    ],
)
def test_kernel_opts_bit_identical(device, geo, request, opts, vb, B, T, hold, out_dtype):
    """kernel_opts = opts (at value_blocks vb) vs the original program (kernel_opts 0, value_blocks 1), fast_fp32 mode:
    out, every ring block and both window halves torch.equal."""
    skip_if_over_grid(device, geo, B)
    if geo.Nv == 24 and T >= 12 and vb > 1 and opts & 16:
        # measured: the (24, 8) two-tile a|b geometry at T = 12 with the q/k share is 48 B over the 70656 B kernel config
        # buffer (TT_FATAL at program finalize); the qwen36 TP = 8 / TP = 4 geometries fit at every (B, T) tested
        pytest.skip("Nv = 24, T = 12, value blocks + q/k share: program exceeds the kernel config buffer")
    res = []
    for kw in (dict(), dict(value_blocks=vb, kernel_opts=opts)):
        c = Case(device, geo, B, T, seed=191 + B * 10 + T, hold=hold, par=1, out_dtype=out_dtype, mode=MODES[2])
        o = c.run(**kw)
        res.append(c.snapshot(o))
    (out0, ring0, wp0, wn0), (out1, ring1, wp1, wn1) = res
    for name, a, b in (("out", out0, out1), ("ring", ring0, ring1), ("win_prev", wp0, wp1), ("win_next", wn0, wn1)):
        d = (a.float() - b.float()).abs().max().item()
        assert torch.equal(a, b), f"kernel_opts={opts} value_blocks={vb}: {name} differs from the original (max|d|={d})"
    eff = effective_value_blocks(device, geo, B, vb)
    print(f"kernel_opts {opts} vb {vb} (eff {eff}) == original at (B={B}, T={T}, hold={hold}): torch.equal")


@pytest.mark.parametrize("opts", [250])
@pytest.mark.parametrize("vb", [1, 2, 4])
def test_kernel_opts_t8_vs_chained_t1(device, geo, request, opts, vb):
    """A T = 8 step with kernel_opts equals 8 chained T = 1 steps with kernel_opts (torch.equal), fast_fp32."""
    _chained_t1(device, geo, request, 2, mi=[1, 3], hold=(1,), mode=MODES[2], value_blocks=vb, kernel_opts=opts)


@pytest.mark.parametrize("B,T", [(8, 4), (4, 8), (8, 1)])
def test_kernel_opts_timing(device, request, B, T):
    """Wall time per call (synchronized, 50 calls after 5 warm-ups) of the qwen36 TP=8 fast_fp32 program: the
    original vs value_blocks 4 (clamped to the grid) with kernel_opts 0 / 250. Report only."""
    import time

    g = GEOS[0]
    skip_if_over_grid(device, g, B)
    times = {}
    for vb, opts in ((1, 0), (1, 250), (4, 0), (4, 250)):
        c = Case(device, g, B, T, seed=7, out_dtype=ttnn.bfloat16, mode=MODES[2])
        kw = dict(value_blocks=vb, kernel_opts=opts)
        for _ in range(5):
            ttnn.deallocate(c.run(**kw))
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(50):
            ttnn.deallocate(c.run(**kw))
        ttnn.synchronize_device(device)
        times[(vb, opts)] = (time.perf_counter() - t0) / 50 * 1e6
    effs = {vb: effective_value_blocks(device, g, B, vb) for vb, _ in times}
    print(
        f"gdn_spec_step (B={B}, T={T}) us/call: "
        + ", ".join(f"vb{vb}(eff {effs[vb]})/opts{o}={t:.1f}" for (vb, o), t in times.items())
    )


# --------------------------------------------------------------------------- #
# K4: kernel_opts bit 8 (QWEN36_SPEC_STEP_LAZY) -- chunked (WY) token phase + lazy rollback. NOT bit-identical, and a
# different ring / ctrl contract (models/demos/blackhole/qwen36/tt/gdn/lazy_ref.py): ring slot 0 block (u, h) = D, the
# state before the previous verify's tokens; slot 1 + par = the previous verify's stash (K^T pages, U pages, decay
# page), written by that verify into 1 + (1 - its parity); ctrl mi >= T = FRESH (no replay, conv window at mi 0).
# The truth for every check is the serial recurrence in fp64 from the true committed state.
# --------------------------------------------------------------------------- #
LAZY_BIT = 256
LAZY_OPTS = LAZY_BIT | 250  # the production restructurings + K4 (the op drops 64 / 8 / 1 under K4 itself)


def reference_lazy(g, rows, E_prev, mi_win, S_start, taps, dtb, nea, w, B, T, hold=()):
    """One verify step in fp64 from the committed states S_start [B, Nv, Dk, Dv]: the serial recurrence, plus what the
    K4 stash must hold. Returns out [B*T, VD], states [T, B, Nv, Dk, Dv], deltas [B, Nv, T, Dv], kn [B, Nv, T, Dk],
    dec [B, Nv, T] (fp32 values of exp(g)), W_new [B, Lw, C]."""
    Nv, Nk, KD, VD, C, QKVZ = g.Nv, g.Nk, g.KD, g.VD, g.C, g.QKVZ
    Lw = K - 1 + T
    rf = Nv // Nk
    f64 = torch.float64
    out = torch.zeros(B * T, VD, dtype=f64)
    states = torch.zeros(T, B, Nv, Dk, Dv, dtype=f64)
    deltas = torch.zeros(B, Nv, T, Dv, dtype=f64)
    kns = torch.zeros(B, Nv, T, Dk, dtype=f64)
    decs = torch.zeros(B, Nv, T, dtype=f64)
    Wn_all = torch.zeros(B, Lw, C)
    for u in range(B):
        if u in hold:
            continue
        m = int(mi_win[u])
        Wn = torch.cat([E_prev[u, m + 1 : m + K], rows[u * T : (u + 1) * T, :C]], 0)
        Wn_all[u] = Wn
        h = S_start[u].to(f64).clone()
        for t in range(T):
            x = sum(taps[j].to(f64) * Wn[t + j].to(f64) for j in range(K))
            conv = torch.nn.functional.silu(x)
            q = conv[:KD].reshape(Nk, Dk).repeat_interleave(rf, 0)
            k = conv[KD : 2 * KD].reshape(Nk, Dk).repeat_interleave(rf, 0)
            v = conv[2 * KD : C].reshape(Nv, Dv)
            r = rows[u * T + t].to(f64)
            z = r[C:QKVZ].reshape(Nv, Dv)
            a = r[QKVZ : QKVZ + Nv]
            b = r[QKVZ + Nv : QKVZ + 2 * Nv]
            beta = torch.sigmoid(b)
            gl = nea.to(f64) * torch.nn.functional.softplus(a + dtb.to(f64), beta=1.0, threshold=20.0)
            qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_EPS) * SCALE
            kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_EPS)
            dec = torch.exp(gl)
            h = h * dec[:, None, None]
            delta = beta[:, None] * (v - torch.einsum("hk,hkv->hv", kn, h))
            h = h + torch.einsum("hk,hv->hkv", kn, delta)
            o = torch.einsum("hk,hkv->hv", qn, h)
            on = o / torch.sqrt((o * o).mean(-1, keepdim=True) + NORM_EPS) * w.to(f64)[None, :]
            out[u * T + t] = (on * torch.nn.functional.silu(z)).reshape(-1)
            states[t, u] = h
            deltas[u, :, t] = delta
            kns[u, :, t] = kn
            decs[u, :, t] = dec
    return out, states, deltas, kns, decs, Wn_all


def _ctrl_lazy(g, B, mi_ctrl, par, hold=()):
    """K4 ctrl page: words 1 + u = mi_ctrl[u] (>= T: FRESH), the block words carry only HOLD (their value is the
    engine's (mi*B + u)*Nv + h, as the chain graph builds it, sentinel mi included)."""
    return ctrl_page(g, B, None, mi_ctrl, par, hold)


class LazyCase:
    """Device state for a K4 verify sequence: ring [T*B*Nv, Dk, Dv] with slot 0 = D, slots 1 / 2 = stash (NaN at the
    start: a FRESH row must never read it), the window pair, the gate / tap constants."""

    def __init__(self, device, g, B, T, seed, par0=0, out_dtype=ttnn.bfloat16):
        skip_if_over_grid(device, g, B)
        torch.manual_seed(seed)
        Nv, C = g.Nv, g.C
        self.device, self.g, self.B, self.T, self.par = device, g, B, T, par0
        self.Lw = K - 1 + T
        self.L = 32
        self.out_dtype = out_dtype
        self.S0 = 0.05 * torch.randn(B, Nv, Dk, Dv)
        ring = torch.full((T * B * Nv, Dk, Dv), float("nan"))
        ring[: B * Nv] = self.S0.reshape(B * Nv, Dk, Dv)
        self.E_prev = (0.5 * torch.randn(B, self.L, C)).bfloat16().float()
        self.taps = (0.3 * torch.randn(K, C)).bfloat16().float()
        self.dtb = (0.1 * torch.randn(Nv)).bfloat16().float()
        self.nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16().float()
        self.w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16().float()
        dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=device)
        self.ring = dev(ring, ttnn.float32)
        wins = [None, None]
        wins[par0] = dev(self.E_prev, ttnn.bfloat16)
        wins[1 - par0] = dev((0.5 * torch.randn(B, self.L, C)).bfloat16().float(), ttnn.bfloat16)
        self.win_a, self.win_b = wins
        self.taps_d = dev(self.taps.reshape(1, K, C), ttnn.bfloat16)
        self.dtb_d = dev(self.dtb.reshape(1, 1, Nv), ttnn.float32)
        self.nea_d = dev(self.nea.reshape(1, 1, Nv), ttnn.float32)
        self.w_d = dev(self.w.reshape(1, 1, Dv), ttnn.bfloat16)

    def run(self, rows, mi_ctrl, hold=(), kernel_opts=LAZY_OPTS, value_blocks=4):
        g, B, T = self.g, self.B, self.T
        dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=self.device)
        qkv = dev(rows.reshape(1, rows.shape[0], g.W), ttnn.bfloat16)
        ctrl = dev(_ctrl_lazy(g, B, mi_ctrl, self.par, hold), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.experimental.kda.gdn_spec_step(
            qkv,
            self.win_a,
            self.win_b,
            self.ring,
            ctrl,
            self.taps_d,
            self.dtb_d,
            self.nea_d,
            self.w_d,
            g.Nv,
            g.Nk,
            Dk,
            Dv,
            T,
            B,
            g.QKVZ,
            conv_kernel=K,
            scale=SCALE,
            hnew_depth=2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=self.out_dtype,
            fast_mode=True,
            fp32_state=True,
            value_blocks=value_blocks,
            kernel_opts=kernel_opts,
        )

    def ring_host(self):
        return ttnn.to_torch(self.ring).reshape(self.T, self.B, self.g.Nv, Dk, Dv).float()

    def win_host(self, par):
        return ttnn.to_torch([self.win_a, self.win_b][par]).float()


def _stash_host(ring_slot, B, Nv, T, Kt=Dk // 32, Vt=Dv // 32):
    """ring_slot [B, Nv, Dk, Dv] -> (kt [B, Nv, T, Dk] (token t = column r0 + t of the K^T pages), u [B, Nv, T, Dv],
    dec [B, Nv, T]) per the lazy_ref page layout."""
    kt_all = torch.zeros(B, Nv, T, Dk)
    u_all = torch.zeros(B, Nv, T, Dv)
    dec_all = torch.zeros(B, Nv, T)
    tile = lambda blk, p: blk[(p // Vt) * 32 : (p // Vt + 1) * 32, (p % Vt) * 32 : (p % Vt + 1) * 32]
    for u in range(B):
        r0 = (u * T) % 32
        for h in range(Nv):
            blk = ring_slot[u, h]
            kt = torch.cat([tile(blk, j) for j in range(Kt)], 0)  # [Dk, 32]
            uu = torch.cat([tile(blk, Kt + v) for v in range(Vt)], 1)  # [32, Dv]
            kt_all[u, h] = kt[:, r0 : r0 + T].T
            u_all[u, h] = uu[r0 : r0 + T]
            dec_all[u, h] = tile(blk, Kt + Vt)[r0 : r0 + T, 0]
    return kt_all, u_all, dec_all


@pytest.mark.parametrize("vb", [1, 4])
@pytest.mark.parametrize("B,T", [(8, 4), (4, 8), (2, 4), (1, 3)])
def test_lazy_sequence(device, geo, request, B, T, vb):
    """K4: a 4-step verify sequence (every row FRESH at step 0 over a NaN stash, then random accepted prefixes, window
    parity flipping) vs the fp64 serial recurrence from the true committed state: outputs (PCC > 0.999), ring slot 0 =
    the replayed committed state (step 0: torch.equal D, the FRESH copy; later PCC > 0.9999), the stash written into
    slot 1 + (1 - par) (U, K^T, decay vs the serial deltas / keys / decays), the rebuilt window (torch.equal).
    GDN_LAZY_SEQ_STEPS (default 4) runs a longer sequence: the device state is never re-synced to the fp64 truth, so
    the slot-0 error of the last step is the accumulated drift."""
    g = geo
    if g.Nv == 24 and T >= 8:
        pytest.skip("Nv = 24 at T = 8: not a qwen36 geometry")
    c = LazyCase(device, g, B, T, seed=901 + 10 * B + T)
    S_true = c.S0.double()
    E_prev = c.E_prev
    mi_ctrl = [T] * B  # FRESH
    gen = torch.Generator().manual_seed(77 + B + T)
    Nv, C, Lw = g.Nv, g.C, c.Lw
    Lw_w = (Lw + 1) // 2 * 2
    for step in range(int(os.environ.get("GDN_LAZY_SEQ_STEPS", "4"))):
        rows = make_rows(g, round_up32(B * T) if B * T % 32 else B * T)
        mi_win = [0 if m >= T else m for m in mi_ctrl]
        out_ref, states, deltas, kns, decs, Wn_ref = reference_lazy(
            g, rows, E_prev, mi_win, S_true, c.taps, c.dtb, c.nea, c.w, B, T
        )
        o = c.run(rows, mi_ctrl, value_blocks=vb)
        out = ttnn.to_torch(o).reshape(-1, g.VD).float()[: B * T]
        ttnn.deallocate(o)
        ring = c.ring_host()
        win_next = c.win_host(1 - c.par)
        p_out = pcc(out, out_ref)
        d_out = (out.double() - out_ref).abs().max().item()
        assert torch.isfinite(out).all(), f"step {step}: non-finite output"
        # ring slot 0: the committed state the step replayed to (S_true), exact for a FRESH step
        if step == 0:
            assert torch.equal(ring[0], c.S0), "FRESH step: slot 0 must be D copied bit for bit"
        p_s0 = pcc(ring[0], S_true)
        d_s0 = (ring[0].double() - S_true).abs().max().item()
        # the stash this step wrote (slot 1 + (1 - par))
        kt_d, u_d, dec_d = _stash_host(ring[1 + (1 - c.par)], B, Nv, T)
        p_u = pcc(u_d, deltas)
        p_k = pcc(kt_d, kns)
        d_dec = (dec_d.double() - decs).abs().max().item()
        # the rebuilt conv window (bit-exact, as the serial kernel's)
        for u in range(B):
            assert torch.equal(win_next[u, :Lw], Wn_ref[u]), f"step {step} user {u}: window rows differ"
            assert torch.equal(win_next[u, Lw:Lw_w], torch.zeros(Lw_w - Lw, C)), f"step {step} user {u}: pad row"
        print(
            f"lazy step {step} (B={B},T={T},vb={vb},mi_ctrl={mi_ctrl}): out pcc={p_out:.7f} max|d|={d_out:.3e} | slot0 "
            f"pcc={p_s0:.8f} max|d|={d_s0:.3e} | stash U pcc={p_u:.8f} K pcc={p_k:.8f} dec max|d|={d_dec:.2e}"
        )
        _golden(request, **{f"out_{step}": out, f"ring_{step}": ring})
        assert p_out > 0.999 and p_s0 > 0.9999 and p_u > 0.9999 and p_k > 0.9999 and d_dec < 1e-5
        # commit a random accepted prefix; the next step reads the window this one wrote, at the flipped parity
        mi_next = [int(x) for x in torch.randint(0, T, (B,), generator=gen)]
        S_true = torch.stack([states[mi_next[u], u] for u in range(B)])
        E_prev = win_next
        mi_ctrl = mi_next
        c.par = 1 - c.par


@pytest.mark.parametrize("B,T", [(8, 4), (4, 8)])
def test_lazy_hold_untouched(device, geo, request, B, T):
    """K4 HOLD: a held user's ring blocks (slot 0 and both stash slots) are untouched and its window rows are copied
    through; the other users run normally (outputs vs fp64 serial)."""
    g = geo
    skip_if_over_grid(device, g, B)
    if g.Nv == 24 and T >= 8:
        pytest.skip("Nv = 24 at T = 8: not a qwen36 geometry")
    c = LazyCase(device, g, B, T, seed=933 + B)
    hold = (0, B - 1) if B > 1 else ()
    rows = make_rows(g, round_up32(B * T) if B * T % 32 else B * T)
    before = c.ring_host()
    o = c.run(rows, [T] * B, hold=hold)
    out = ttnn.to_torch(o).reshape(-1, g.VD).float()[: B * T]
    ring = c.ring_host()
    win_next = c.win_host(1 - c.par)
    out_ref, *_ = reference_lazy(g, rows, c.E_prev, [0] * B, c.S0.double(), c.taps, c.dtb, c.nea, c.w, B, T, hold)
    live = [u for u in range(B) if u not in hold]
    rl = sum(([u * T + t for t in range(T)] for u in live), [])
    for u in hold:
        assert torch.equal(ring[:, u].nan_to_num(7.0), before[:, u].nan_to_num(7.0)), f"held user {u}: ring changed"
        assert torch.equal(win_next[u, : c.Lw], c.E_prev[u, : c.Lw]), f"held user {u}: window not copied through"
    p = pcc(out[rl], out_ref[rl])
    print(f"lazy hold {sorted(hold)} (B={B},T={T}): live out pcc={p:.7f}; held ring blocks + windows untouched")
    assert p > 0.999


def test_lazy_vs_serial_kernel(device, request):
    """K4 vs today's serial program (kernel_opts 250) on the same inputs, qwen36 TP=8 (B=8, T=4): a FRESH step (both
    from the same D) and a second step (serial from its own ring slot mi, K4 from its replay). Report: output PCC /
    max|d| between the two and vs fp64, and the share of bf16 output elements that differ. Not bit-identical by
    design."""
    g, B, T = GEOS[0], 8, 4
    c = LazyCase(device, g, B, T, seed=4242)
    # the serial program's own ring: slot 0 = D as well (mi = 0 reads it), the rest garbage
    dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=device)
    ring_s = torch.randn(T * B * g.Nv, Dk, Dv)
    ring_s[: B * g.Nv] = c.S0.reshape(B * g.Nv, Dk, Dv)
    ring_s = dev(ring_s, ttnn.float32)
    wins_s = [dev(c.E_prev, ttnn.bfloat16), dev(torch.zeros(B, c.L, g.C), ttnn.bfloat16)]
    S_true = c.S0.double()
    E_prev = c.E_prev
    mi_l = [T] * B
    mi_s = [0] * B
    gen = torch.Generator().manual_seed(5)
    for step in range(2):
        rows = make_rows(g, B * T)
        par = c.par
        out_ref, states, *_ = reference_lazy(
            g, rows, E_prev, [0 if m >= T else m for m in mi_l], S_true, c.taps, c.dtb, c.nea, c.w, B, T
        )
        ol = c.run(rows, mi_l)
        qkv = dev(rows.reshape(1, B * T, g.W), ttnn.bfloat16)
        ctrl_s = dev(ctrl_page(g, B, T, mi_s, par), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        os_ = ttnn.experimental.kda.gdn_spec_step(
            qkv, wins_s[0], wins_s[1], ring_s, ctrl_s, c.taps_d, c.dtb_d, c.nea_d, c.w_d,
            g.Nv, g.Nk, Dk, Dv, T, B, g.QKVZ, conv_kernel=K, scale=SCALE, hnew_depth=2,
            memory_config=ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16, fast_mode=True, fp32_state=True,
            value_blocks=4, kernel_opts=250,
        )  # fmt: skip
        out_l = ttnn.to_torch(ol).reshape(-1, g.VD).float()[: B * T]
        out_s = ttnn.to_torch(os_).reshape(-1, g.VD).float()[: B * T]
        diff_frac = (out_l != out_s).float().mean().item()
        print(
            f"lazy-vs-serial step {step}: pcc(K4, serial)={pcc(out_l, out_s):.7f} "
            f"max|d|={(out_l - out_s).abs().max():.3e} "
            f"bf16 elements differing {100 * diff_frac:.1f}% | vs fp64: K4 pcc={pcc(out_l, out_ref):.7f} "
            f"max|d|={(out_l.double() - out_ref).abs().max():.3e}, serial pcc={pcc(out_s, out_ref):.7f} "
            f"max|d|={(out_s.double() - out_ref).abs().max():.3e}"
        )
        assert pcc(out_l, out_s) > 0.999
        mi_next = [int(x) for x in torch.randint(0, T, (B,), generator=gen)]
        S_true = torch.stack([states[mi_next[u], u] for u in range(B)])
        E_prev = c.win_host(1 - par)
        mi_l, mi_s = mi_next, mi_next
        c.par = 1 - par  # both programs read win[par] of their own pair and write win[1 - par]


def test_lazy_rejects(device, expect_error):
    """K4 refuses T < 3 / T > 8 and a mode without fast_mode + fp32_state (it must never be dropped silently)."""
    g = GEOS[0]
    c = LazyCase(device, g, 1, 4, seed=3)
    rows = make_rows(g, 4)
    with expect_error(RuntimeError, "3 <= T <= 8"):
        c2 = LazyCase(device, g, 1, 2, seed=3)
        c2.run(make_rows(g, 2), [2])
    with expect_error(RuntimeError, "needs fast_mode and fp32_state"):
        dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=device)
        ttnn.experimental.kda.gdn_spec_step(
            dev(rows.reshape(1, 4, g.W), ttnn.bfloat16), c.win_a, c.win_b, c.ring,
            dev(ctrl_page(g, 1, 4, [4], 0), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            c.taps_d, c.dtb_d, c.nea_d, c.w_d, g.Nv, g.Nk, Dk, Dv, 4, 1, g.QKVZ, conv_kernel=K, scale=SCALE,
            memory_config=ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16, fast_mode=True, fp32_state=False,
            kernel_opts=LAZY_BIT,
        )  # fmt: skip


# --------------------------------------------------------------------------- #
# K4 SFPU gates (kernel_opts bit 9, + bit 11 face) with REALISTIC dt_bias. softplus_tile selects its algorithm on
# INP_FLOAT32; before gdn_step_helpers.hpp defined it, the GDN compute kernels got the BF16 branch, which clamps the
# residual to 0 for |a + dt_bias| > 5: softplus = 0 and decay = 1 (no forgetting) wherever a + dt_bias < -5. The other
# tests draw dt_bias = 0.1 randn and never reach that region. Qwen3.8: 24% of the 2304 GDN heads have dt_bias < -5,
# min about -9. Truth: the fp64 serial recurrence; the K4 program with the writer's exact software gates (no bit 9) is
# the reference arm.
# --------------------------------------------------------------------------- #
SFPU_GATE_BIT = 512
NARROW_GR_BIT = 1024
SFPU_FACE_BIT = 2048
# one TP = 8 slice (Nv = 6) spanning the checkpoint's range (Qwen3.8, 2304 heads: dt_bias min -8.94, 10% -5.93,
# median -3.23, 24.2% < -5; exp(A_log) on the dt_bias < -5 heads: median 0.37, 90% 4.9, max 23.5)
DTB_REALISTIC = [-9.0, -7.5, -6.0, -5.0, -3.0, 0.5]
NEA_REALISTIC = [-23.5, -4.9, -0.37, -0.125, -1.0, -0.33]
SFPU_VARIANTS = {
    "k4": LAZY_OPTS | NARROW_GR_BIT,
    "sfpu": LAZY_OPTS | NARROW_GR_BIT | SFPU_GATE_BIT,
    "sfpu_face": LAZY_OPTS | NARROW_GR_BIT | SFPU_GATE_BIT | SFPU_FACE_BIT,
}


@pytest.mark.parametrize("a_scale", [0.3, 2.0])
@pytest.mark.parametrize("variant", list(SFPU_VARIANTS))
@pytest.mark.parametrize("B,T", [(8, 4), (1, 4)])
def test_lazy_sfpu_gates_realistic_dtbias(device, request, B, T, variant, a_scale):
    """K4 verify sequence (GDN_SFPU_SEQ_STEPS steps, default 6) with dt_bias down to -9 and `a` at a_scale * randn:
    per step the decay the kernel stashed (exp(g_t), fp32) vs fp64 -- max |d dec| over every (user, head, token) and
    over the a + dt_bias < -5 ones separately, the share of those whose decay came out exactly 1.0 (the BF16 clamp
    signature) -- outputs (PCC) and the replayed state in ring slot 0 (max relative error per head class). The state
    is never re-synced to the truth, so slot 0 carries the accumulated drift. Asserts the FP32-branch accuracy."""
    g = GEOS[0]
    assert g.Nv == len(DTB_REALISTIC)
    c = LazyCase(device, g, B, T, seed=1701 + 10 * B + T)
    c.dtb = torch.tensor(DTB_REALISTIC).bfloat16().float()
    c.nea = torch.tensor(NEA_REALISTIC).bfloat16().float()
    dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=device)
    c.dtb_d = dev(c.dtb.reshape(1, 1, g.Nv), ttnn.float32)
    c.nea_d = dev(c.nea.reshape(1, 1, g.Nv), ttnn.float32)
    opts = SFPU_VARIANTS[variant]
    S_true = c.S0.double()
    E_prev = c.E_prev
    mi_ctrl = [T] * B
    gen = torch.Generator().manual_seed(4049 + B + T)
    Nv, QKVZ = g.Nv, g.QKVZ
    neg_heads = c.dtb < -5
    worst = dict(dec_all=0.0, dec_neg=0.0, ones_neg=0.0, s0_neg=0.0, s0_pos=0.0, p_out=1.0)
    for step in range(int(os.environ.get("GDN_SFPU_SEQ_STEPS", "6"))):
        rows = make_rows(g, round_up32(B * T) if B * T % 32 else B * T)
        rows[:, QKVZ : QKVZ + Nv] = (a_scale * torch.randn(rows.shape[0], Nv)).bfloat16().float()
        mi_win = [0 if m >= T else m for m in mi_ctrl]
        out_ref, states, deltas, kns, decs, Wn_ref = reference_lazy(
            g, rows, E_prev, mi_win, S_true, c.taps, c.dtb, c.nea, c.w, B, T
        )
        o = c.run(rows, mi_ctrl, kernel_opts=opts)
        out = ttnn.to_torch(o).reshape(-1, g.VD).float()[: B * T]
        ttnn.deallocate(o)
        ring = c.ring_host()
        _, _, dec_d = _stash_host(ring[1 + (1 - c.par)], B, Nv, T)
        # a + dt_bias per (user, head, token)
        tt_ = torch.stack(
            [rows[u * T : (u + 1) * T, QKVZ : QKVZ + Nv].T + c.dtb[:, None] for u in range(B)]
        )  # [B, Nv, T]
        neg = tt_ < -5
        d_dec = (dec_d.double() - decs).abs()
        dec_all = (d_dec / decs.clamp_min(1e-30)).max().item()  # relative: decays span 1e-3 .. 1
        dec_neg = d_dec[neg].max().item() if neg.any() else 0.0
        ones_neg = (dec_d[neg] == 1.0).float().mean().item() if neg.any() else 0.0
        rel = (ring[0].double() - S_true).flatten(2).norm(dim=-1) / S_true.flatten(2).norm(dim=-1).clamp_min(1e-30)
        s0_neg = rel[:, neg_heads].max().item()
        s0_pos = rel[:, ~neg_heads].max().item()
        p_out = pcc(out, out_ref)
        print(
            f"sfpu-real {variant} a_scale={a_scale} step {step} (B={B},T={T}): a+dtb<-5 share={neg.float().mean():.3f} "
            f"| dec max rel err all={dec_all:.3e}, max|d| a+dtb<-5={dec_neg:.3e} dec==1 share there={ones_neg:.3f} "
            f"| slot0 rel err dtb<-5 heads={s0_neg:.3e} others={s0_pos:.3e} | out pcc={p_out:.7f}"
        )
        for k, v in (("dec_all", dec_all), ("dec_neg", dec_neg), ("ones_neg", ones_neg), ("s0_neg", s0_neg), ("s0_pos", s0_pos)):
            worst[k] = max(worst[k], v)
        worst["p_out"] = min(worst["p_out"], p_out)
        mi_next = [int(x) for x in torch.randint(0, T, (B,), generator=gen)]
        S_true = torch.stack([states[mi_next[u], u] for u in range(B)])
        E_prev = c.win_host(1 - c.par)
        mi_ctrl = mi_next
        c.par = 1 - c.par
    print(f"sfpu-real SUMMARY {variant} a_scale={a_scale} B={B} T={T}: " + " ".join(f"{k}={v:.3e}" for k, v in worst.items()))
    # the FP32 branch: decay within 1e-3 relative everywhere (softplus <= 4e-4 relative), within a few fp32 ulps of the
    # truth where a + dt_bias < -5 (decay near 1; the BF16 clamp's error there is 1 - exp(nea * softplus) ~ 1e-4..1e-2),
    # and slot-0 drift (per (user, head) relative Frobenius error vs fp64, never re-synced) within the exact-gate K4
    # arm's envelope: device, 6 steps, B = 8: 2e-3..4.4e-3 for every variant with the FP32 branch
    assert worst["dec_all"] < 1e-3, f"{variant}: decay max rel err {worst['dec_all']:.3e}"
    assert worst["dec_neg"] < 2e-6, f"{variant}: decay max|d| where a + dt_bias < -5: {worst['dec_neg']:.3e}"
    assert worst["s0_neg"] < 1e-2 and worst["s0_pos"] < 1e-2, f"{variant}: slot-0 drift {worst}"
    assert worst["p_out"] > 0.999
