# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.gdn_spec_step: a speculative-verify step over T = K + 1 candidate tokens per user writes the
recurrent state after every token into pages the caller names (ring_page_idx), so the caller can keep the state of
whichever continuation it accepts. The convolution history after token t is rows t+1 .. t+3 of the window the op
writes into win[1 - par].

The states live in 1024-token bfloat8_b K / V cache blocks of a TP=8 layer slot through alias_view (KvBlocks of
test_gdn_decode_step_paged.py: heads 0..3 in K, heads 4..5 in V). Every per-position output, state and history is
compared with gdn_decode_step run token by token from the same start state, bit for bit, and with the torch
reference."""

import time

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.gdn_decode_step_test_utils import (
    AZ,
    DK,
    DV,
    KV,
    MODE_IDS,
    MODES,
    NK,
    NV,
    SCALE,
    VD,
    C,
    W,
    bits,
    consts_dev,
    from_pooled,
    inputs,
    pack_rows,
    pcc,
    run,
    step_reference,
    tile_dev,
    tiles,
    to_pooled,
    token_rows,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.test_gdn_decode_step_paged import (
    MODEL_MODE,
    TRACE_PARAMS,
    _kv_blocks_or_skip,
)
from tt_py_test_utils_common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()

CONV_K = 4
HOLD = -1  # int32 bit pattern of the HOLD sentinel 0xFFFFFFFF
WIN_ROWS = 32


def round_up32(n):
    return (n + 31) // 32 * 32


def ring_page(kv, block, h):
    """Page of head h of `block` in the page space of ring = kv.rec_k, ring_pool2 = kv.rec_v."""
    pool, page = kv.state_page(block, h)
    return page + (kv.rec_pages if pool == 1 else 0)


def decode_chain(device, consts, h0, hist0, rows, mode):
    """gdn_decode_step on the default layout, batch rows = the len(h0) users, one call per token. h0 [n, NV, DK, DV],
    hist0 [n, NV, 4, 32, 32] (paged layout), rows [T, n, W]. Returns per token (out [n, VD], h, paged history)."""
    n = h0.shape[0]
    state = tile_dev(device, h0.clone(), ttnn.float32)
    hist = tile_dev(device, torch.stack([from_pooled(hist0[b], b) for b in range(n)]), ttnn.bfloat16)
    res = []
    for t in range(rows.shape[0]):
        x = tile_dev(device, rows[t].reshape(1, n, W), ttnn.bfloat16)
        out = ttnn.to_torch(run(x, consts, state, hist, *mode, output_dtype=ttnn.float32)).reshape(n, VD)
        hist_t = ttnn.to_torch(hist)
        res.append((out, ttnn.to_torch(state).clone(), torch.stack([to_pooled(hist_t[b], b) for b in range(n)])))
    return res


def decode_per_user(device, consts, h0, hist0, rows, mode):
    """decode_chain at B = 1 for every user (one value head per core, the arithmetic gdn_spec_step reproduces).
    Returns per token (out [n, VD], h [n, ...], history [n, ...])."""
    per_user = [
        decode_chain(device, consts, h0[u : u + 1], hist0[u : u + 1], rows[:, u : u + 1], mode)
        for u in range(h0.shape[0])
    ]
    return [
        tuple(torch.cat([per_user[u][t][i] for u in range(len(per_user))]) for i in range(3))
        for t in range(rows.shape[0])
    ]


class Verify:
    """One verify step for B users of T tokens on the KV blocks `kv`. starts[u]: block of user u's start state;
    dests[u][t]: block of the state after token t. mi[u]: rows mi+1 .. mi+3 of win[par][u] are the start history.
    hold: users the ctrl page holds. The qkvzab rows past B*T are random filler (their output rows must be 0)."""

    def __init__(self, device, kv, B, T, gen, starts, dests, mi=None, par=0, hold=(), mode=MODEL_MODE):
        self.device, self.kv, self.B, self.T, self.par, self.hold, self.mode = device, kv, B, T, par, set(hold), mode
        self.starts, self.dests = list(starts), [list(d) for d in dests]
        self.mi = [int(x) for x in torch.randint(0, T, (B,), generator=gen)] if mi is None else list(mi)
        self.R = round_up32(B * T)
        self.Lw = CONV_K - 1 + T
        self.rows = torch.zeros(self.R, W, dtype=torch.bfloat16)
        self.rows[: B * T] = token_rows(B * T, gen)
        self.rows[B * T :] = token_rows(self.R - B * T, gen)
        self.win = [(0.5 * torch.randn(B, WIN_ROWS, C, generator=gen)).bfloat16() for _ in range(2)]
        self.qkv = tile_dev(device, self.rows.reshape(1, self.R, W), ttnn.bfloat16)
        self.ctrl = ttnn.from_torch(self.ctrl_host(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self.idx = ttnn.from_torch(self.idx_host(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self.win_dev = [tile_dev(device, w, ttnn.bfloat16) for w in self.win]

    def set_history(self, hist_rows):
        """hist_rows[u]: the 4 history rows [C] of user u, slot 0 (oldest, never read) .. slot 3. They go to rows
        mi .. mi+3 of win[par][u]."""
        for u, r in enumerate(hist_rows):
            for j in range(4):
                self.win[self.par][u, self.mi[u] + j] = r[j]
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self.win[self.par], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.win_dev[self.par]
        )

    def ctrl_host(self):
        words = (1 + self.B + self.B * NV + 15) // 16 * 16
        page = torch.zeros(1, words, dtype=torch.int32)
        page[0, 0] = self.par
        for u in range(self.B):
            page[0, 1 + u] = self.mi[u]
            for h in range(NV):
                page[0, 1 + self.B + u * NV + h] = HOLD if u in self.hold else ring_page(self.kv, self.starts[u], h)
        return page

    def idx_host(self):
        return torch.tensor(
            [
                [ring_page(self.kv, self.dests[u][t], h) for t in range(self.T) for h in range(NV)]
                for u in range(self.B)
            ],
            dtype=torch.int32,
        )

    def write_index(self):
        """Rewrite the contents of the ctrl and index tensors (no new allocation)."""
        for t_dev, host in ((self.ctrl, self.ctrl_host()), (self.idx, self.idx_host())):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), t_dev
            )

    def write_rows(self, rows):
        self.rows[: self.B * self.T] = rows.reshape(self.B * self.T, W)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self.rows.reshape(1, self.R, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.qkv
        )

    def run(self, consts, taps, **kw):
        return ttnn.experimental.kda.gdn_spec_step(
            self.qkv,
            self.win_dev[0],
            self.win_dev[1],
            self.kv.rec_k,
            self.ctrl,
            taps,
            consts["dtb"],
            consts["nea"],
            consts["w"],
            NV,
            NK,
            DK,
            DV,
            self.T,
            self.B,
            AZ,
            conv_kernel=CONV_K,
            scale=SCALE,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=ttnn.float32,
            fast_mode=self.mode[0],
            fp32_state=self.mode[1],
            ring_page_idx=self.idx,
            ring_pool2=self.kv.rec_v,
            **kw,
        )

    def token_rows(self):
        """[T, B, W]: token t of every user, as the decode reference consumes them."""
        return self.rows[: self.B * self.T].reshape(self.B, self.T, W).transpose(0, 1)

    def window_rows(self, u):
        """Rows 0..Lw-1 the op must write for user u: [E_prev[mi+1 : mi+4]; the T new [q|k|v] rows]."""
        m = self.mi[u]
        return torch.cat([self.win[self.par][u, m + 1 : m + CONV_K], self.rows[u * self.T : (u + 1) * self.T, :C]])


def spec_taps(device, inp):
    return tile_dev(device, torch.stack(inp["tap_rows"]).reshape(1, CONV_K, C), ttnn.bfloat16)


def check_window(v, win_next, live):
    """win[1-par]: live users' rows 0..Lw-1 rebuilt (an odd Lw gets an exact zero row Lw), the rest untouched; held
    users' rows copied through from win[par]; win[par] untouched."""
    lw, lw_w = v.Lw, (v.Lw + 1) // 2 * 2
    other = v.win[1 - v.par]
    for u in range(v.B):
        if u in live:
            assert torch.equal(win_next[u, :lw], v.window_rows(u)), f"user {u}: window rows"
            assert not win_next[u, lw:lw_w].any(), f"user {u}: window padding row"
            assert torch.equal(win_next[u, lw_w:], other[u, lw_w:]), f"user {u}: window rows past {lw_w} changed"
        elif u in v.hold:
            assert torch.equal(win_next[u, :lw], v.win[v.par][u, :lw]), f"held user {u}: window not copied through"


def expected_pages(kv, writes):
    """Host model of K and V after the op: kv's pages with writes = [(block, state [NV, DK, DV])] applied in order."""
    k, v = kv.k_pages.clone(), kv.v_pages.clone()
    for block, h_state in writes:
        for h in range(NV):
            pool, page = kv.state_page(block, h)
            (k, v)[pool][page : page + KV] = tiles(h_state[h])
    return k, v


def block_state(kv, pages, block):
    """[NV, KV, 32, 32]: the state tiles of `block` in pages = (K pages, V pages)."""
    return torch.stack([pages[kv.state_page(block, h)[0]][kv.state_page(block, h)[1] :][:KV] for h in range(NV)])


def check_pages(kv, want, skip_blocks=()):
    """K and V equal the host model bit for bit, except the state pages of skip_blocks."""
    k, v, _ = kv.read()
    mk = torch.ones(kv.rec_pages, dtype=torch.bool)
    mv = torch.ones(kv.rec_pages, dtype=torch.bool)
    for block in skip_blocks:
        for h in range(NV):
            pool, page = kv.state_page(block, h)
            (mk, mv)[pool][page : page + KV] = False
    assert torch.equal(bits(k[mk]), bits(want[0][mk])), "K pages differ from the expected states / original bytes"
    assert torch.equal(bits(v[mv]), bits(want[1][mv])), "V pages differ from the expected states / original bytes"


def distinct_blocks(B, T, num_blocks, gen):
    """Random distinct blocks >= 1: (starts [B], dests [B][T])."""
    ids = (torch.randperm(num_blocks - 1, generator=gen)[: B * (T + 1)] + 1).tolist()
    return ids[:B], [ids[B + u * T : B + (u + 1) * T] for u in range(B)]


def num_blocks_for(B, T):
    n = B * (T + 1) + 2
    return n + (n % 2)


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("K_draft", [1, 3])
@pytest.mark.parametrize("B", [1, 8, 16])
def test_spec_step_matches_sequential_decode(device, B, K_draft, fast_mode, fp32_state):
    """Distinct start and destination blocks of both parities. Each position's output, state and convolution
    history equal gdn_decode_step run token by token from the same start bit for bit, both at B = 1 per user and at
    batch B, and follow the torch reference; the padding output rows are 0; no other K / V byte changes. B = 16 puts
    users into the second tile row of the projection rows and of the output."""
    T, mode = K_draft + 1, (fast_mode, fp32_state)
    gen = torch.Generator().manual_seed(1000 + 10 * B + T)
    inp = inputs(B, seed=B + T)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()
    v = Verify(device, kv, B, T, gen, starts, dests, par=(B + T) & 1, mode=mode)
    v.set_history(inp["rows"])
    out = ttnn.to_torch(v.run(consts, taps)).reshape(v.R, VD)
    win_next = ttnn.to_torch(v.win_dev[1 - v.par])

    rows = v.token_rows()
    ref = decode_per_user(device, consts, inp["h"], inp["hist2"], rows, mode)
    batched = decode_chain(device, consts, inp["h"], inp["hist2"], rows, mode)
    assert not out[B * T :].any(), "output rows past B*T are not 0"
    assert torch.equal(ttnn.to_torch(v.win_dev[v.par]), v.win[v.par]), "win[par] changed"
    check_window(v, win_next, range(B))
    writes = []
    for t in range(T):
        out_t, h_t, hist_t = ref[t]
        for u in range(B):
            assert torch.equal(bits(out[u * T + t]), bits(out_t[u])), f"position {t} user {u}: output"
            hist_spec = pack_rows([r for r in win_next[u, t : t + 4]], both=True)
            assert torch.equal(hist_spec, hist_t[u]), f"position {t} user {u}: convolution history"
            writes.append((dests[u][t], h_t[u]))
    check_pages(kv, expected_pages(kv, writes))

    # the batched decode (two users per core at B > 1) and the torch reference
    for t in range(T):
        for u in range(B):
            assert torch.equal(bits(out[u * T + t]), bits(batched[t][0][u])), f"position {t} user {u}: batched output"
            assert torch.equal(bits(ref[t][1][u]), bits(batched[t][1][u])), f"position {t} user {u}: batched state"
            assert torch.equal(ref[t][2][u], batched[t][2][u]), f"position {t} user {u}: batched history"
    k, vv, _ = kv.read()
    taps_f = [t.float() for t in inp["tap_rows"]]
    worst_out, worst_state = 1.0, 1.0
    for u in range(B):
        h_ref, hist_ref = inp["h"][u].clone(), [r.float() for r in inp["rows"][u]]
        for t in range(T):
            o_ref, h_ref, hist_ref = step_reference(
                hist_ref[1:],
                rows[t, u].float(),
                taps_f,
                inp["dtb"].float(),
                inp["nea"].float(),
                h_ref,
                inp["w"].float(),
            )
            worst_out = min(worst_out, pcc(out[u * T + t], o_ref))
            h_dev = block_state(kv, (k, vv), dests[u][t])
            worst_state = min(worst_state, pcc(h_dev, torch.stack([tiles(x) for x in h_ref])))
    logger.info(f"B={B} T={T} mode={mode}: worst out pcc {worst_out:.7f}, worst state pcc {worst_state:.7f} vs torch")
    assert worst_out > 0.999 and worst_state > (0.999 if fast_mode else 0.9999), (worst_out, worst_state)


@pytest.mark.parametrize("K_draft", [1, 3])
@pytest.mark.parametrize("alias_at", ["first", "last"])
def test_spec_step_destination_aliases_the_start(device, K_draft, alias_at):
    """The block of one position is the start block itself (the op reads the start state before it writes any
    state). The other positions go to the blocks right after the start block, so they alternate block parity (fp32
    pages at different bank offsets). Bit-identical to the sequential decode."""
    T, B = K_draft + 1, 4
    gen = torch.Generator().manual_seed(77 + T)
    inp = inputs(B, seed=77 + T)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    n = num_blocks_for(B, T)
    starts = [1 + u * (T + 1) for u in range(B)]
    pos = 0 if alias_at == "first" else T - 1
    dests = [[starts[u] if t == pos else starts[u] + 1 + t for t in range(T)] for u in range(B)]
    kv = _kv_blocks_or_skip(device, n, gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()
    v = Verify(device, kv, B, T, gen, starts, dests)
    v.set_history(inp["rows"])
    out = ttnn.to_torch(v.run(consts, taps)).reshape(v.R, VD)
    ref = decode_per_user(device, consts, inp["h"], inp["hist2"], v.token_rows(), MODEL_MODE)
    writes = []
    for t in range(T):
        for u in range(B):
            assert torch.equal(bits(out[u * T + t]), bits(ref[t][0][u])), f"position {t} user {u}: output"
            writes.append((dests[u][t], ref[t][1][u]))
    check_pages(kv, expected_pages(kv, writes))


def test_spec_step_padding_rows(device):
    """Batch rows without a request: rows 1 and 5 are held (HOLD: no state writes, window copied through), rows 3 and
    6 point every start and destination word at the NaN-filled null block 0. Live rows stay bit-identical to the
    sequential decode; no page outside the null block and the live destinations changes."""
    B, T = 8, 4
    held, nulls = (1, 5), (3, 6)
    live = [u for u in range(B) if u not in held + nulls]
    gen = torch.Generator().manual_seed(5)
    inp = inputs(B, seed=5)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    for u in nulls:
        starts[u], dests[u] = 0, [0] * T
    kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in live + list(held)])
    for h in range(NV):
        pool, page = kv.state_page(0, h)
        (kv.k_pages, kv.v_pages)[pool][page : page + KV] = float("nan")
    kv.write()
    v = Verify(device, kv, B, T, gen, starts, dests, hold=held, par=1)
    v.set_history(inp["rows"])
    out = ttnn.to_torch(v.run(consts, taps)).reshape(v.R, VD)
    win_next = ttnn.to_torch(v.win_dev[1 - v.par])
    ref = decode_per_user(device, consts, inp["h"][live], inp["hist2"][live], v.token_rows()[:, live], MODEL_MODE)
    check_window(v, win_next, live)
    writes = []
    for t in range(T):
        for i, u in enumerate(live):
            assert torch.equal(bits(out[u * T + t]), bits(ref[t][0][i])), f"position {t} user {u}: output"
            writes.append((dests[u][t], ref[t][1][i]))
    check_pages(kv, expected_pages(kv, writes), skip_blocks=[0])


@pytest.mark.parametrize("poison", ["nan", "inf"])
def test_spec_step_ignores_destination_contents(device, poison):
    """Destination state pages filled with NaN or inf before the call: the op overwrites them without reading them,
    so outputs and states equal a run on clean destinations bit for bit."""
    B, T = 8, 4
    gen = torch.Generator().manual_seed(31)
    inp = inputs(B, seed=31)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    results = []
    for fill in (None, float(poison)):
        kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), torch.Generator().manual_seed(3))
        kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
        if fill is not None:
            for u in range(B):
                for t in range(T):
                    for h in range(NV):
                        pool, page = kv.state_page(dests[u][t], h)
                        (kv.k_pages, kv.v_pages)[pool][page : page + KV] = fill
        kv.write()
        v = Verify(device, kv, B, T, torch.Generator().manual_seed(4), starts, dests)
        v.set_history(inp["rows"])
        out = ttnn.to_torch(v.run(consts, taps))
        k, vv, _ = kv.read()
        states = torch.stack([block_state(kv, (k, vv), dests[u][t]) for u in range(B) for t in range(T)])
        results.append((out, states, ttnn.to_torch(v.win_dev[1 - v.par])))
    (o0, s0, w0), (o1, s1, w1) = results
    assert torch.equal(bits(o1), bits(o0)), "output depends on the destination contents"
    assert torch.equal(bits(s1), bits(s0)), "states depend on the destination contents"
    assert torch.equal(w1, w0), "window depends on the destination contents"
    assert torch.isfinite(s1).all(), "a destination page kept a poisoned value"


def run_chain(device, kv, B, T, gen, inp, consts, taps, steps, mode, starts, dest_sets, trace=False):
    """`steps` verify steps; after each one every user accepts a random count a (1..T) and the next step starts from
    the state after token a-1 (its destination block) and the window the last step wrote (mi = a-1, parity flipped).
    Step s writes into dest_sets[s]. Each step is compared with the per-user sequential decode from the state the
    previous step accepted, bit for bit. With trace=True one captured trace runs every step: between replays only
    the contents of the token rows, the ctrl page and the index change."""
    v = Verify(device, kv, B, T, gen, starts, dest_sets[0], par=0, mode=mode)
    v.set_history(inp["rows"])
    h0, hist0 = inp["h"].clone(), inp["hist2"].clone()
    tid = out_dev = None
    if trace:
        snapshot = (kv.k_pages.clone(), kv.v_pages.clone())
        ttnn.deallocate(v.run(consts, taps))  # compile outside the trace, then put the blocks back
        kv.write()
        assert torch.equal(bits(kv.read()[0]), bits(snapshot[0]))
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out_dev = v.run(consts, taps)
        ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        for s in range(steps):
            if s > 0:
                v.write_rows(token_rows(B * T, gen))
                v.write_index()
            if trace:
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                out = ttnn.to_torch(out_dev).reshape(v.R, VD)
            else:
                out = ttnn.to_torch(v.run(consts, taps)).reshape(v.R, VD)
            win_next = ttnn.to_torch(v.win_dev[1 - v.par])
            check_window(v, win_next, range(B))
            ref = decode_per_user(device, consts, h0, hist0, v.token_rows(), mode)
            writes = []
            for t in range(T):
                for u in range(B):
                    assert torch.equal(bits(out[u * T + t]), bits(ref[t][0][u])), f"step {s} pos {t} user {u}: output"
                    hist_spec = pack_rows([r for r in win_next[u, t : t + 4]], both=True)
                    assert torch.equal(hist_spec, ref[t][2][u]), f"step {s} pos {t} user {u}: history"
                    writes.append((v.dests[u][t], ref[t][1][u]))
            want = expected_pages(kv, writes)
            check_pages(kv, want)
            kv.k_pages, kv.v_pages = want
            # accept: the next step starts from position a-1 of this step
            acc = [int(x) for x in torch.randint(1, T + 1, (B,), generator=gen)]
            h0 = torch.stack([ref[acc[u] - 1][1][u] for u in range(B)])
            hist0 = torch.stack([ref[acc[u] - 1][2][u] for u in range(B)])
            v.win[1 - v.par] = win_next
            v.par = 1 - v.par
            v.mi = [a - 1 for a in acc]
            v.starts = [v.dests[u][acc[u] - 1] for u in range(B)]
            if s + 1 < steps:
                v.dests = [list(d) for d in dest_sets[s + 1]]
            logger.info(f"step {s}: bit-identical to the sequential decode; accepted {acc}")
    finally:
        if trace:
            ttnn.release_trace(device, tid)


def chain_blocks(B, T, steps, gen):
    """Distinct blocks: one start per user, then per step one block per (user, position)."""
    n = num_blocks_for(B, T * steps)
    ids = (torch.randperm(n - 1, generator=gen) + 1).tolist()
    starts = ids[:B]
    sets = []
    for s in range(steps):
        base = B + s * B * T
        sets.append([ids[base + u * T : base + (u + 1) * T] for u in range(B)])
    return n, starts, sets


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("K_draft", [1, 3])
def test_spec_step_chained_verifies(device, K_draft, fast_mode, fp32_state):
    """Three verify steps with random acceptance: the convolution history of each start comes from the window the
    previous step wrote (rows mi+1 .. mi+3), the recurrent state from the accepted position's block."""
    B, T, steps = 4, K_draft + 1, 3
    gen = torch.Generator().manual_seed(400 + T)
    inp = inputs(B, seed=400 + T)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    n, starts, sets = chain_blocks(B, T, steps, gen)
    kv = _kv_blocks_or_skip(device, n, gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()
    run_chain(device, kv, B, T, gen, inp, consts, taps, steps, (fast_mode, fp32_state), starts, sets)


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("K_draft", [1, 3])
def test_spec_step_trace_follows_index_contents(device, K_draft):
    """One captured trace replayed for three chained verify steps; between replays only the contents of the token
    rows, the ctrl page (start pages, mi, parity) and the index change. Every replay is bit-identical to the
    sequential decode."""
    B, T, steps = 8, K_draft + 1, 3
    gen = torch.Generator().manual_seed(500 + T)
    inp = inputs(B, seed=500 + T)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    n, starts, sets = chain_blocks(B, T, steps, gen)
    kv = _kv_blocks_or_skip(device, n, gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()
    run_chain(device, kv, B, T, gen, inp, consts, taps, steps, MODEL_MODE, starts, sets, trace=True)


def test_spec_step_program_cache_follows_new_buffers(device):
    """A second KV cache, window pair, ctrl page and index (new addresses, same shapes) reuse the cached program; the
    first allocation keeps what the first call wrote."""
    B, T = 8, 4
    gen = torch.Generator().manual_seed(61)
    inp = inputs(B, seed=61)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    first, entries = None, None
    for allocation in range(2):
        kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), torch.Generator().manual_seed(8))
        kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
        kv.write()
        v = Verify(device, kv, B, T, torch.Generator().manual_seed(9), starts, dests)
        v.set_history(inp["rows"])
        out = ttnn.to_torch(v.run(consts, taps)).reshape(v.R, VD)
        ref = decode_per_user(device, consts, inp["h"], inp["hist2"], v.token_rows(), MODEL_MODE)
        writes = [(dests[u][t], ref[t][1][u]) for t in range(T) for u in range(B)]
        for t in range(T):
            for u in range(B):
                assert torch.equal(bits(out[u * T + t]), bits(ref[t][0][u])), f"allocation {allocation}: output"
        check_pages(kv, expected_pages(kv, writes))
        if first is None:
            first, first_pages, entries = kv, kv.read(), device.num_program_cache_entries()
        else:
            assert kv.k.buffer_address() != first.k.buffer_address()
            assert device.num_program_cache_entries() == entries, "new buffers compiled a new program"
            now = first.read()
            assert torch.equal(bits(now[0]), bits(first_pages[0])) and torch.equal(bits(now[1]), bits(first_pages[1]))


def test_spec_step_refuses_removed_options(device, expect_error):
    """The keywords of the variant with speed options stay in the signature; only their defaults are accepted. The
    window pair must be distinct, fp32_state needs fast_mode, and B * Nv cores must fit the grid."""
    B, T = 2, 4
    gen = torch.Generator().manual_seed(0)
    inp = inputs(B, seed=0)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), gen)
    v = Verify(device, kv, B, T, gen, starts, dests)
    for kw, msg in [
        (dict(value_blocks=4), "value_blocks > 1 is not supported"),
        (dict(kernel_opts=250), "kernel_opts are not supported"),
        (dict(hnew_depth=4), "hnew_depth must be 2"),
        (dict(max_users=B + 1), "max_users must be 0 or B"),
    ]:
        with expect_error(RuntimeError, msg):
            v.run(consts, taps, **kw)
    ttnn.deallocate(v.run(consts, taps, value_blocks=1, max_users=B))  # the defaults' equivalents are accepted
    v.mode = (False, True)
    with expect_error(RuntimeError, "fp32_state needs fast_mode"):
        v.run(consts, taps)
    grid = device.compute_with_storage_grid_size()
    big = grid.x * grid.y // NV + 1
    kv_big = _kv_blocks_or_skip(device, 2, gen)
    vb = Verify(device, kv_big, big, 1, gen, [0] * big, [[0]] * big)
    with expect_error(RuntimeError, "exceed"):
        vb.run(consts, taps)


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("B", [1, 8, 16])
def test_spec_step_traced_latency(device, B):
    """Traced time per T = 4 verify in the model's mode (states in KV blocks, two pools) against four gdn_decode_step
    calls on one state, and against four (copy the previous state and history, then step them in place) rounds that
    keep every position's state. One trace holds N back-to-back calls; the median of REPS replays is reported. Not a
    pass / fail gate."""
    T, n_calls, reps = 4, 32, 11
    gen = torch.Generator().manual_seed(3)
    inp = inputs(B, seed=B)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()
    v = Verify(device, kv, B, T, gen, starts, dests)
    v.set_history(inp["rows"])
    xs = [tile_dev(device, token_rows(B, gen).reshape(1, B, W), ttnn.bfloat16) for _ in range(T)]
    hist_host = torch.stack([from_pooled(inp["hist2"][b], b) for b in range(B)])
    states = [tile_dev(device, inp["h"].clone(), ttnn.float32) for _ in range(T + 1)]
    hists = [tile_dev(device, hist_host, ttnn.bfloat16) for _ in range(T + 1)]

    def decode_x4():
        outs = [run(xs[t], consts, states[0], hists[0], *MODEL_MODE) for t in range(T)]
        for o in outs[:-1]:
            ttnn.deallocate(o)
        return outs[-1]

    def copy_decode_x4():
        outs = []
        for t in range(T):
            ttnn.copy(states[t], states[t + 1])
            ttnn.copy(hists[t], hists[t + 1])
            outs.append(run(xs[t], consts, states[t + 1], hists[t + 1], *MODEL_MODE))
        for o in outs[:-1]:
            ttnn.deallocate(o)
        return outs[-1]

    def per_call_us(call):
        ttnn.deallocate(call())
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(n_calls):
            ttnn.deallocate(call())
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            times.append((time.perf_counter() - t0) / n_calls * 1e6)
        ttnn.release_trace(device, tid)
        return sorted(times)[reps // 2]

    calls = {
        "gdn_spec_step": lambda: v.run(consts, taps),
        "4x gdn_decode_step": decode_x4,
        "4x (copy + gdn_decode_step)": copy_decode_x4,
    }
    res = {k: [] for k in calls}
    for _ in range(2):  # alternate so that drift hits all of them
        for k, call in calls.items():
            res[k].append(per_call_us(call))
    logger.info(f"LATENCY B={B} T={T}: " + ", ".join(f"{k} {min(x):.2f} us" for k, x in res.items()))
