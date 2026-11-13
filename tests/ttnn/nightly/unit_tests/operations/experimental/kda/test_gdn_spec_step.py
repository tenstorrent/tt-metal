# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.gdn_spec_step: a speculative-verify step over T = K + 1 candidate tokens per user writes the
recurrent state after every token into pages the caller names (ring_page_idx), so the caller can keep the state of
whichever continuation it accepts. The convolution history after token t is rows t+1 .. t+3 of the window the op
writes into win[1 - par], or, with conv_hist, slots 1..3 of gdn_decode_step's packed history pages of the block that
takes the state after token t.

The states live in 1024-token bfloat8_b K / V cache blocks of a TP=8 layer slot through alias_view (KvBlocks of
test_gdn_decode_step_paged.py: heads 0..3 in K, heads 4..5 in V). Every per-position output, state and history is
compared with gdn_decode_step run token by token from the same start state, bit for bit, and with the torch
reference. The value_blocks / kernel_opts programs the qwen36 model runs are compared with the minimal program and with
the sequential decode, bit for bit, in both history modes."""

import time

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.gdn_decode_step_test_utils import (
    AZ,
    DK,
    DV,
    HP,
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
    _bf16_tile_raw,
    _fp32_tile_raw,
    _kv_blocks_or_skip,
    _raw_fp32_tile,
)
from tt_py_test_utils_common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()

CONV_K = 4
HOLD = -1  # int32 bit pattern of the HOLD sentinel 0xFFFFFFFF
WIN_ROWS = 32

# kernel_opts bits (GdnSpecStepKernelOpt): gate scalars 2, hoisted q scale 8, q/k share 16, half-tile conv tail 32, gate
# split 64, own-column tail 128, narrow gate reads 1024. The model passes all of them (1274) with fast_mode and
# fp32_state, and value_blocks=4 on a verify (T > 1). value_blocks is clamped to the grid: at Nv = 6 on 110 cores it is
# 4 at B <= 4, 2 at B = 8 and 1 at B = 16.
ALL_KERNEL_OPTS = 2 | 8 | 16 | 32 | 64 | 128 | 1024
OPTION_SETS = {
    "minimal": {},
    "vb4_opts": dict(value_blocks=4, kernel_opts=ALL_KERNEL_OPTS),  # the model's verify
    "opts": dict(kernel_opts=ALL_KERNEL_OPTS),  # the model's T = 1 seed
    "vb4": dict(value_blocks=4),
}
FAST_OPTIONS = [k for k in OPTION_SETS if k != "minimal"]


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

    def windows(self):
        return self.win_dev[0], self.win_dev[1]

    def run(self, consts, taps, **kw):
        return ttnn.experimental.kda.gdn_spec_step(
            self.qkv,
            *self.windows(),
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


@pytest.mark.parametrize("opts", FAST_OPTIONS)
@pytest.mark.parametrize("K_draft", [0, 1, 3])
@pytest.mark.parametrize("B", [1, 8, 16])
def test_spec_step_options_match_minimal_and_sequential_decode(device, B, K_draft, opts):
    """The model's mode with value_blocks / kernel_opts: the same inputs on two copies of the KV blocks, once through
    the minimal program and once with the options. Outputs (including the padding rows), every K / V byte and both
    windows are bit-identical between the two, and every position's output, state and convolution history equals
    gdn_decode_step run token by token. K_draft = 0 is the T = 1 seed step. value_blocks=4 runs 4 blocks per head at
    B = 1, 2 at B = 8 and 1 at B = 16."""
    check_options_against_minimal_and_decode(device, B, K_draft + 1, opts, 2000 + 10 * B + K_draft)


@pytest.mark.parametrize("opts", FAST_OPTIONS)
@pytest.mark.parametrize("K_draft", [0, 1, 3, 7])
@pytest.mark.parametrize("B", [1, 8, 16])
def test_spec_step_history_pages_options_match_minimal_and_sequential_decode(device, B, K_draft, opts):
    """conv_hist with value_blocks / kernel_opts: as test_spec_step_options_match_minimal_and_sequential_decode, with
    the history in the blocks' packed history pages. Outputs, every K / V byte and every history page are
    bit-identical to the minimal program with conv_hist, and every position's output, state and history pages equal
    gdn_decode_step run token by token. With q/k share the head's blocks write disjoint rows of one history page."""
    check_options_against_minimal_and_decode(device, B, K_draft + 1, opts, 5000 + 10 * B + K_draft, hist=True)


@pytest.mark.parametrize("hist", [False, True], ids=["window", "conv_hist"])
@pytest.mark.parametrize("opts", ["vb4_opts", "opts"])
@pytest.mark.parametrize("K_draft", [1, 3])
@pytest.mark.parametrize("B", [1, 8])
def test_spec_step_options_with_fp32_gate_constants(device, B, K_draft, opts, hist):
    """The model's FLOAT32 dt_bias / -exp(A) (values that use the low mantissa bits): with kernel_opts the writer
    reads them for the gate math. Bit-identical to the minimal program and to the sequential decode."""
    check_options_against_minimal_and_decode(
        device, B, K_draft + 1, opts, 3000 + 10 * B + K_draft, fp32_gates=True, hist=hist
    )


@pytest.mark.parametrize("hist", [False, True], ids=["window", "conv_hist"])
@pytest.mark.parametrize("T, B", [(T, B) for T in (1, 2, 3, 4, 8) for B in (4, 8)] + [(12, 1), (12, 2)])
def test_spec_step_model_options_fit_the_kernel_config_buffer(device, B, T, hist):
    """The option set the model passes at each T (value_blocks=4 with kernel_opts on a verify, kernel_opts alone on
    the T = 1 seed) builds and launches in both history modes: the kernel binaries, runtime arguments and DFB configs
    fit the kernel config buffer. Bit-identical to the minimal program. At T = 12 a user's rows must not cross a tile
    row, so B <= 2 (4 blocks per head)."""
    opts = "opts" if T == 1 else "vb4_opts"
    check_options_against_minimal_and_decode(device, B, T, opts, 4000 + 10 * B + T, decode=False, hist=hist)


def check_options_against_minimal_and_decode(device, B, T, opts, seed, fp32_gates=False, decode=True, hist=False):
    """test_spec_step_options_match_minimal_and_sequential_decode for one (B, T, option set); decode=False skips the
    comparison with the sequential decode. hist=True: conv_hist (HistVerify) instead of the window pair."""
    inp = inputs(B, seed=seed)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    if fp32_gates:
        gen = torch.Generator().manual_seed(seed + 3)
        for key in ("dtb", "nea"):
            consts[key] = tile_dev(device, inp[key].float() * (1 + 1e-3 * torch.randn(NV, generator=gen)), ttnn.float32)
    n = num_blocks_for(B, T)
    starts, dests = distinct_blocks(B, T, n, torch.Generator().manual_seed(seed))
    runs = {}
    for name in ("minimal", opts):
        kv = _kv_blocks_or_skip(device, n, torch.Generator().manual_seed(seed + 1))
        kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
        kv.write()
        gen = torch.Generator().manual_seed(seed + 2)
        if hist:
            v = HistVerify(device, kv, B, T, gen, starts, dests)
        else:
            v = Verify(device, kv, B, T, gen, starts, dests, par=(B + T) & 1)
            v.set_history(inp["rows"])
        out = ttnn.to_torch(v.run(consts, taps, **OPTION_SETS[name])).reshape(v.R, VD)
        wins = (None, None) if hist else tuple(ttnn.to_torch(w) for w in v.win_dev)
        runs[name] = (kv, v, out, wins, kv.read())
    _, _, out0, wins0, pages0 = runs["minimal"]
    kv, v, out, wins, pages = runs[opts]
    assert torch.equal(bits(out), bits(out0)), "output differs from the minimal program"
    if not hist:
        assert all(torch.equal(w, w0) for w, w0 in zip(wins, wins0)), "window pair differs from the minimal program"
    for i, name in enumerate(("K", "V", "history")):
        assert torch.equal(bits(pages[i]), bits(pages0[i])), f"{name} pages differ from the minimal program"

    assert not out[B * T :].any(), "output rows past B*T are not 0"
    if not hist:
        win_next = wins[1 - v.par]
        check_window(v, win_next, range(B))
    if not decode:
        return
    ref = decode_per_user(device, consts, inp["h"], inp["hist2"], v.token_rows(), MODEL_MODE)
    writes = []
    for t in range(T):
        out_t, h_t, hist_t = ref[t]
        for u in range(B):
            assert torch.equal(bits(out[u * T + t]), bits(out_t[u])), f"position {t} user {u}: output"
            if not hist:
                hist_spec = pack_rows([r for r in win_next[u, t : t + 4]], both=True)
                assert torch.equal(hist_spec, hist_t[u]), f"position {t} user {u}: convolution history"
            writes.append((dests[u][t], h_t[u], hist_t[u]))
    if hist:
        check_pages(kv, expected_pages_hist(kv, writes))
    else:
        check_pages(kv, expected_pages(kv, [(b, h) for b, h, _ in writes]))


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


@pytest.mark.parametrize("opts", list(OPTION_SETS))
@pytest.mark.parametrize("B", [4, 8])
def test_spec_step_padding_rows(device, B, opts):
    """Batch rows without a request: some rows are held (HOLD: no state writes, window copied through), others point
    every start and destination word at the NaN-filled null block 0. Live rows stay bit-identical to the sequential
    decode; no page outside the null block and the live destinations changes. With value_blocks, B = 4 runs 4 blocks
    per head and B = 8 runs 2."""
    T = 4
    held, nulls = ((1,), (2,)) if B == 4 else ((1, 5), (3, 6))
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
    out = ttnn.to_torch(v.run(consts, taps, **OPTION_SETS[opts])).reshape(v.R, VD)
    win_next = ttnn.to_torch(v.win_dev[1 - v.par])
    ref = decode_per_user(device, consts, inp["h"][live], inp["hist2"][live], v.token_rows()[:, live], MODEL_MODE)
    check_window(v, win_next, live)
    assert not out[B * T :].any(), "output rows past B*T are not 0"
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


def run_chain(device, kv, B, T, gen, inp, consts, taps, steps, mode, starts, dest_sets, trace=False, opts="minimal"):
    """`steps` verify steps; after each one every user accepts a random count a (1..T) and the next step starts from
    the state after token a-1 (its destination block) and the window the last step wrote (mi = a-1, parity flipped).
    Step s writes into dest_sets[s]. Each step is compared with the per-user sequential decode from the state the
    previous step accepted, bit for bit. With trace=True one captured trace runs every step: between replays only
    the contents of the token rows, the ctrl page and the index change. opts: the OPTION_SETS entry of every call."""
    kw = OPTION_SETS[opts]
    v = Verify(device, kv, B, T, gen, starts, dest_sets[0], par=0, mode=mode)
    v.set_history(inp["rows"])
    h0, hist0 = inp["h"].clone(), inp["hist2"].clone()
    tid = out_dev = None
    if trace:
        snapshot = (kv.k_pages.clone(), kv.v_pages.clone())
        ttnn.deallocate(v.run(consts, taps, **kw))  # compile outside the trace, then put the blocks back
        kv.write()
        assert torch.equal(bits(kv.read()[0]), bits(snapshot[0]))
        # The sequential reference compiles here too: a program compiled while the trace exists gets its kernel
        # binaries in memory that a replay may overwrite.
        decode_per_user(device, consts, h0, hist0, v.token_rows(), mode)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out_dev = v.run(consts, taps, **kw)
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
                out = ttnn.to_torch(v.run(consts, taps, **kw)).reshape(v.R, VD)
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
@pytest.mark.parametrize("opts", ["minimal", "vb4_opts"])
@pytest.mark.parametrize("B", [2, 8])
@pytest.mark.parametrize("K_draft", [1, 3])
def test_spec_step_trace_follows_index_contents(device, K_draft, B, opts):
    """One captured trace replayed for three chained verify steps; between replays only the contents of the token
    rows, the ctrl page (start pages, mi, parity) and the index change. Every replay is bit-identical to the
    sequential decode. With value_blocks the head's blocks synchronize through semaphores that every launch resets
    (4 blocks per head at B = 2, 2 at B = 8)."""
    T, steps = K_draft + 1, 3
    gen = torch.Generator().manual_seed(500 + T)
    inp = inputs(B, seed=500 + T)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    n, starts, sets = chain_blocks(B, T, steps, gen)
    kv = _kv_blocks_or_skip(device, n, gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()
    run_chain(device, kv, B, T, gen, inp, consts, taps, steps, MODEL_MODE, starts, sets, trace=True, opts=opts)


@pytest.mark.parametrize("opts", ["minimal", "vb4_opts"])
def test_spec_step_program_cache_follows_new_buffers(device, opts):
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
        out = ttnn.to_torch(v.run(consts, taps, **OPTION_SETS[opts])).reshape(v.R, VD)
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


def test_spec_step_refuses_unsupported_options(device, expect_error):
    """hnew_depth and max_users accept only their defaults; value_blocks is at most 4; kernel_opts accepts only the
    GdnSpecStepKernelOpt bits, and only with fast_mode and fp32_state. fp32_state needs fast_mode, and B * Nv cores
    must fit the grid."""
    B, T = 2, 4
    gen = torch.Generator().manual_seed(0)
    inp = inputs(B, seed=0)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), gen)
    v = Verify(device, kv, B, T, gen, starts, dests)
    for kw, msg in [
        (dict(value_blocks=5), "value_blocks must be in"),
        (dict(kernel_opts=ALL_KERNEL_OPTS | 1), "unsupported kernel_opts bits 0x1"),
        (dict(kernel_opts=4), "unsupported kernel_opts bits 0x4"),
        (dict(kernel_opts=256 | 512), "unsupported kernel_opts bits 0x300"),
        (dict(hnew_depth=4), "hnew_depth must be 2"),
        (dict(max_users=B + 1), "max_users must be 0 or B"),
    ]:
        with expect_error(RuntimeError, msg):
            v.run(consts, taps, **kw)
    ttnn.deallocate(v.run(consts, taps, value_blocks=1, max_users=B))  # the defaults' equivalents are accepted
    for mode in ((False, False), (True, False)):
        v.mode = mode
        with expect_error(RuntimeError, "kernel_opts need fast_mode and fp32_state"):
            v.run(consts, taps, kernel_opts=2)
        ttnn.deallocate(v.run(consts, taps, value_blocks=4))  # value_blocks works in every mode
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
@pytest.mark.parametrize("K_draft", [1, 3])
@pytest.mark.parametrize("B", [1, 8, 16])
def test_spec_step_traced_latency(device, B, K_draft):
    """Traced time per verify in the model's mode (states in KV blocks, two pools), minimal and with the model's
    options, against T gdn_decode_step calls on one state, and against T (copy the previous state and history, then
    step them in place) rounds that keep every position's state. One trace holds N back-to-back calls; the median of
    REPS replays is reported. Not a pass / fail gate."""
    T, n_calls, reps = K_draft + 1, 32, 11
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

    def decode_t():
        outs = [run(xs[t], consts, states[0], hists[0], *MODEL_MODE) for t in range(T)]
        for o in outs[:-1]:
            ttnn.deallocate(o)
        return outs[-1]

    def copy_decode_t():
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
        "gdn_spec_step vb4_opts": lambda: v.run(consts, taps, **OPTION_SETS["vb4_opts"]),
        f"{T}x gdn_decode_step": decode_t,
        f"{T}x (copy + gdn_decode_step)": copy_decode_t,
    }
    res = {k: [] for k in calls}
    for _ in range(2):  # alternate so that drift hits all of them
        for k, call in calls.items():
            res[k].append(per_call_us(call))
    logger.info(f"LATENCY B={B} T={T}: " + ", ".join(f"{k} {min(x):.2f} us" for k, x in res.items()))


# ---- conv_hist: the convolution history in gdn_decode_step's packed pages of the blocks, no window pair


class HistVerify(Verify):
    """Verify with conv_hist = kv.hist and no window pair: the start history is slots 1..3 of the start block's packed
    history pages, and the history after token t goes to slots 1..3 of dests[u][t]."""

    def ctrl_host(self):
        words = (1 + self.B + 2 * self.B * NV + 15) // 16 * 16
        page = torch.zeros(1, words, dtype=torch.int32)
        for u in range(self.B):
            for h in range(NV):
                held = u in self.hold
                page[0, 1 + self.B + u * NV + h] = HOLD if held else ring_page(self.kv, self.starts[u], h)
                page[0, 1 + self.B + self.B * NV + u * NV + h] = HOLD if held else self.kv.hist_page(self.starts[u], h)
        return page

    def idx_host(self):
        hist = [
            [self.kv.hist_page(self.dests[u][t], h) for t in range(self.T) for h in range(NV)] for u in range(self.B)
        ]
        return torch.cat([super().idx_host(), torch.tensor(hist, dtype=torch.int32)], dim=1)

    def windows(self):
        return None, None

    def run(self, consts, taps, **kw):
        return super().run(consts, taps, conv_hist=self.kv.hist, **kw)


def expected_pages_hist(kv, writes):
    """expected_pages with the history: writes = [(block, state [NV, DK, DV], paged history [NV, 4, 32, 32])]. Slots
    1..3 of the block's history pages take the history; slot 0 keeps its bytes."""
    k, v = expected_pages(kv, [(block, state) for block, state, _ in writes])
    raw16 = _fp32_tile_raw(v).view(torch.int16)
    for block, _, hist in writes:
        for h in range(NV):
            for s in range(1, HP):
                q, off = kv._v_slot(kv.hist_page(block, h) + s)
                raw16[q, off : off + 1024] = _bf16_tile_raw(hist[h, s])[0]
    return k, _raw_fp32_tile(raw16.view(torch.int32))


@pytest.mark.parametrize("K_draft", [1, 3])
@pytest.mark.parametrize("B", [1, 8, 16])
def test_spec_step_history_pages_match_sequential_decode(device, B, K_draft):
    """conv_hist: each position's output, state and history pages equal gdn_decode_step run token by token from the
    start block's state and history, bit for bit; no other K / V byte changes. gdn_decode_step then
    continues from the last position's block, which holds a complete state."""
    T = K_draft + 1
    gen = torch.Generator().manual_seed(2000 + 10 * B + T)
    inp = inputs(B, seed=B + T + 7)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()
    v = HistVerify(device, kv, B, T, gen, starts, dests, par=(B + T) & 1)
    out = ttnn.to_torch(v.run(consts, taps)).reshape(v.R, VD)
    ref = decode_per_user(device, consts, inp["h"], inp["hist2"], v.token_rows(), MODEL_MODE)
    assert not out[B * T :].any(), "output rows past B*T are not 0"
    writes = []
    for t in range(T):
        for u in range(B):
            assert torch.equal(bits(out[u * T + t]), bits(ref[t][0][u])), f"position {t} user {u}: output"
            writes.append((dests[u][t], ref[t][1][u], ref[t][2][u]))
    want = expected_pages_hist(kv, writes)
    check_pages(kv, want)

    rows = token_rows(B, gen)
    x = tile_dev(device, rows.reshape(1, B, W), ttnn.bfloat16)
    idx = ttnn.from_torch(
        kv.index_host([dests[u][T - 1] for u in range(B)]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    nxt = run(
        x, consts, kv.rec_k, kv.hist, *MODEL_MODE, output_dtype=ttnn.float32, state_page_idx=idx, state_pool2=kv.rec_v
    )
    nxt = ttnn.to_torch(nxt).reshape(B, VD)
    ref_next = decode_per_user(device, consts, ref[T - 1][1], ref[T - 1][2], rows.reshape(1, B, W), MODEL_MODE)
    for u in range(B):
        assert torch.equal(bits(nxt[u]), bits(ref_next[0][0][u])), f"user {u}: decode after the verify"


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("opts", ["minimal", "vb4_opts", "vb4"])
@pytest.mark.parametrize("B", [2, 8])
@pytest.mark.parametrize("K_draft", [1, 3])
def test_spec_step_history_pages_chain_through_blocks(device, K_draft, B, opts):
    """conv_hist, one captured trace replayed for three verify steps: each step starts from the block of the position
    the previous step accepted (its history written there by that step), and one position per step writes into the
    step's own start block. Between replays only the token rows, the ctrl page and the index change. Every step is
    bit-identical to the sequential decode. With value_blocks (4 blocks per head at B = 2, 2 at B = 8) every block
    reads the start block's history while a block of the head writes the aliased destination: vb4_opts (q/k share)
    splits the history rows between the blocks, vb4 has block 0 write after the head's other blocks are done."""
    T, steps = K_draft + 1, 3
    kw = OPTION_SETS[opts]
    gen = torch.Generator().manual_seed(700 + T)
    inp = inputs(B, seed=700 + T)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    n, starts, sets = chain_blocks(B, T, steps, gen)
    for s in range(steps):
        alias = s % T  # position whose destination is the step's start block
        sets[s] = [[None if t == alias else d for t, d in enumerate(row)] for row in sets[s]]
    kv = _kv_blocks_or_skip(device, n, gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()

    def with_alias(dest_set, start_blocks):
        return [[start_blocks[u] if d is None else d for d in row] for u, row in enumerate(dest_set)]

    v = HistVerify(device, kv, B, T, gen, starts, with_alias(sets[0], starts))
    h0, hist0 = inp["h"].clone(), inp["hist2"].clone()
    snapshot = (kv.k_pages.clone(), kv.v_pages.clone())
    ttnn.deallocate(v.run(consts, taps, **kw))  # compile outside the trace, then put the blocks back
    kv.write()
    assert torch.equal(bits(kv.read()[0]), bits(snapshot[0]))
    # The sequential reference compiles here too: a program compiled while the trace exists gets its kernel binaries
    # in memory that a replay may overwrite.
    decode_per_user(device, consts, h0, hist0, v.token_rows(), MODEL_MODE)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_dev = v.run(consts, taps, **kw)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        for s in range(steps):
            if s > 0:
                v.write_rows(token_rows(B * T, gen))
                v.write_index()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            out = ttnn.to_torch(out_dev).reshape(v.R, VD)
            ref = decode_per_user(device, consts, h0, hist0, v.token_rows(), MODEL_MODE)
            writes = []
            for t in range(T):
                for u in range(B):
                    assert torch.equal(bits(out[u * T + t]), bits(ref[t][0][u])), f"step {s} pos {t} user {u}: output"
                    writes.append((v.dests[u][t], ref[t][1][u], ref[t][2][u]))
            want = expected_pages_hist(kv, writes)
            check_pages(kv, want)
            kv.k_pages, kv.v_pages = want
            acc = [int(x) for x in torch.randint(1, T + 1, (B,), generator=gen)]
            h0 = torch.stack([ref[acc[u] - 1][1][u] for u in range(B)])
            hist0 = torch.stack([ref[acc[u] - 1][2][u] for u in range(B)])
            v.starts = [v.dests[u][acc[u] - 1] for u in range(B)]
            if s + 1 < steps:
                v.dests = with_alias(sets[s + 1], v.starts)
            logger.info(f"step {s}: bit-identical to the sequential decode; accepted {acc}")
    finally:
        ttnn.release_trace(device, tid)


@pytest.mark.parametrize("opts", list(OPTION_SETS))
@pytest.mark.parametrize("B", [4, 8])
def test_spec_step_history_pages_padding_rows(device, B, opts):
    """conv_hist with held rows (no state or history writes) and rows on the NaN-filled null block 0: live rows stay
    bit-identical to the sequential decode, the output rows past B*T are 0, and no page outside the null block and the
    live destinations changes. With value_blocks, B = 4 runs 4 blocks per head and B = 8 runs 2."""
    T = 4
    held, nulls = ((1,), (2,)) if B == 4 else ((1, 5), (3, 6))
    live = [u for u in range(B) if u not in held + nulls]
    gen = torch.Generator().manual_seed(15)
    inp = inputs(B, seed=15)
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
    v = HistVerify(device, kv, B, T, gen, starts, dests, hold=held)
    out = ttnn.to_torch(v.run(consts, taps, **OPTION_SETS[opts])).reshape(v.R, VD)
    ref = decode_per_user(device, consts, inp["h"][live], inp["hist2"][live], v.token_rows()[:, live], MODEL_MODE)
    assert not out[B * T :].any(), "output rows past B*T are not 0"
    writes = []
    for t in range(T):
        for i, u in enumerate(live):
            assert torch.equal(bits(out[u * T + t]), bits(ref[t][0][i])), f"position {t} user {u}: output"
            writes.append((dests[u][t], ref[t][1][i], ref[t][2][i]))
    want = expected_pages_hist(kv, writes)
    k, vv, _ = kv.read()
    k_null, v_null, _ = kv.owned([0])
    assert torch.equal(bits(k[~k_null]), bits(want[0][~k_null])), "K pages outside the null block differ"
    assert torch.equal(bits(vv[~v_null]), bits(want[1][~v_null])), "V pages outside the null block differ"


def test_spec_step_history_pages_refuse_bad_layouts(device, expect_error):
    """conv_hist replaces the window pair (both together, or neither, are refused) and needs ring_page_idx with
    2*T*Nv words per row."""
    B, T = 2, 4
    gen = torch.Generator().manual_seed(1)
    inp = inputs(B, seed=1)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), gen)
    v = Verify(device, kv, B, T, gen, starts, dests)
    with expect_error(RuntimeError, "pass either win_a and win_b"):
        v.run(consts, taps, conv_hist=kv.hist)
    h = HistVerify(device, kv, B, T, gen, starts, dests)
    with expect_error(RuntimeError, "pass either win_a and win_b"):
        Verify.run(h, consts, taps)  # no windows and no conv_hist
    h.idx = v.idx  # T*Nv words per row: no history pages
    with expect_error(RuntimeError, "ring_page_idx must be"):
        h.run(consts, taps)


@pytest.mark.parametrize("hist", [False, True], ids=["window", "conv_hist"])
@pytest.mark.parametrize("K_draft, B", [(7, 1), (7, 4), (8, 1), (8, 3), (11, 1), (11, 2)])
def test_spec_step_long_verifies(device, K_draft, B, hist):
    """T = 8 and 12 (the model's K = 7 and 11) in both history modes: the program fits the kernel config buffer, and
    outputs and states are bit-identical to the sequential decode. T = 9: with conv_hist the 11 packed history tiles
    leave one tile of the writer's bounce buffer before the 2-tile zero block of the output padding rows; a block that
    crossed the buffer end overwrote the ring_page_idx row allocated after it, and the states went to wrong pages."""
    T = K_draft + 1
    gen = torch.Generator().manual_seed(3000 + 10 * B + T)
    inp = inputs(B, seed=B + T + 3)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()
    v = (HistVerify if hist else Verify)(device, kv, B, T, gen, starts, dests)
    if not hist:
        v.set_history(inp["rows"])
    out = ttnn.to_torch(v.run(consts, taps)).reshape(v.R, VD)
    ref = decode_per_user(device, consts, inp["h"], inp["hist2"], v.token_rows(), MODEL_MODE)
    assert not out[B * T :].any(), "output rows past B*T are not 0"
    writes = []
    for t in range(T):
        for u in range(B):
            assert torch.equal(bits(out[u * T + t]), bits(ref[t][0][u])), f"position {t} user {u}: output"
            writes.append((dests[u][t], ref[t][1][u], ref[t][2][u]))
    if hist:
        check_pages(kv, expected_pages_hist(kv, writes))
    else:
        check_pages(kv, expected_pages(kv, [(b, s) for b, s, _ in writes]))


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("K_draft", [1, 3])
@pytest.mark.parametrize("B", [1, 8, 16, 32])
def test_spec_step_history_pages_traced_latency(device, B, K_draft):
    """Traced time per verify in the model's mode (FLOAT32 dt_bias / -exp(A)) with the history in the window pair and
    in the blocks (conv_hist), each minimal and with the model's verify options (vb4_opts), against T gdn_decode_step
    calls on the blocks. B = 32 is two B = 16 calls (110 cores fit B * Nv <= 110). One trace holds N back-to-back
    calls; the median of REPS replays is reported. Not a pass / fail gate."""
    T, n_calls, reps = K_draft + 1, 32, 11
    part = min(B, 16)
    gen = torch.Generator().manual_seed(33)
    inp = inputs(B, seed=B + 33)
    consts, taps = consts_dev(device, inp), spec_taps(device, inp)
    for key in ("dtb", "nea"):
        consts[key] = tile_dev(device, inp[key].float(), ttnn.float32)
    starts, dests = distinct_blocks(B, T, num_blocks_for(B, T), gen)
    kv = _kv_blocks_or_skip(device, num_blocks_for(B, T), gen)
    kv.place([(starts[u], inp["h"][u], inp["hist2"][u]) for u in range(B)])
    kv.write()
    win, pages = [], []
    for p0 in range(0, B, part):
        users = slice(p0, p0 + part)
        win.append(Verify(device, kv, part, T, gen, starts[users], dests[users]))
        win[-1].set_history(inp["rows"][users])
        pages.append(HistVerify(device, kv, part, T, gen, starts[users], dests[users]))
    xs = [tile_dev(device, token_rows(B, gen).reshape(1, B, W), ttnn.bfloat16) for _ in range(T)]
    idx = ttnn.from_torch(kv.index_host(starts), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def decode_t():
        outs = [kv.run(xs[t], consts, idx) for t in range(T)]
        for o in outs[:-1]:
            ttnn.deallocate(o)
        return outs[-1]

    def spec_calls(verifies, kw):
        def call():
            out = None
            for v in verifies:
                if out is not None:
                    ttnn.deallocate(out)
                out = v.run(consts, taps, **kw)
            return out

        return call

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

    calls = {}
    for name in ("minimal", "vb4_opts"):
        calls[f"window pair {name}"] = spec_calls(win, OPTION_SETS[name])
        calls[f"conv_hist {name}"] = spec_calls(pages, OPTION_SETS[name])
    if B <= 16:
        calls[f"{T}x gdn_decode_step"] = decode_t
    res = {k: [] for k in calls}
    for _ in range(2):  # alternate so that drift hits all of them
        for k, call in calls.items():
            res[k].append(per_call_us(call))
    logger.info(f"LATENCY B={B} T={T}: " + ", ".join(f"{k} {min(x):.2f} us" for k, x in res.items()))
