# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.gdn_decode_step with state_page_idx: the recurrent state and the packed convolution history
live in page pools (for example vLLM-owned cache blocks), and a [2*NV, B] index names, per batch row and value head,
the first state page and the first history page. state_pool2 and conv_hist_pool2 extend the page space, so the heads
of one request can be split over a layer's K and V buffers.

Most tests compare with the op on the default [BMAX, ...] layout bit for bit; test_paged_steps_vs_reference compares
with the torch reference."""

import time

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.gdn_decode_step_test_utils import (
    BMAX,
    DK,
    DV,
    HP,
    KV,
    MODE_IDS,
    MODES,
    NV,
    W,
    bits,
    consts_dev,
    default_step,
    from_pooled,
    inputs,
    pack_rows,
    pcc,
    run,
    step_reference,
    tile_dev,
    tiles,
    token_rows,
)
from tt_py_test_utils_common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()

MODEL_MODE = (True, True)  # (fast_mode, fp32_state) as the model runs it
BLOCK = 64  # fp32 pages of one 1024-token bfloat8_b K (or V) block, the two_pools layout
K_HEADS = 4  # two_pools: heads 0..3 in the K pool, heads 4..5 and the whole history in the V pool
TRACE_PARAMS = [{"trace_region_size": 16_000_000}]
FP32_PAGE, BF16_PAGE, BF8_PAGE = 4096, 2048, 1088


def _random_starts(n, run_pages, pool_pages, gen, first=0):
    """n non-overlapping runs of run_pages pages in [first, pool_pages), random order and gaps (any alignment)."""
    cuts = sorted(torch.randint(first, pool_pages - n * run_pages + 1, (n,), generator=gen).tolist())
    order = torch.randperm(n, generator=gen).tolist()
    starts = [0] * n
    for i, j in enumerate(order):
        starts[j] = cuts[i] + i * run_pages
    return starts


class Layout:
    """Where each request's heads live: st[r][h] / hs[r][h] = (pool, first page) of value head h's state (KV fp32
    pages) and history (HP bf16 pages). Pool 0 is `state` / `conv_hist`, pool 1 `state_pool2` / `conv_hist_pool2`.
      rows_in_order: one pool, request r at the default offsets.
      random: one pool, every (request, head) at its own random page of any alignment.
      two_pools: request r owns one BLOCK-page run in the K pool (heads 0..3), one in the V pool (heads 4..5) and
        NV * HP history pages in the V history pool, each at a random page of any alignment.
    With null=True, pages [0, NV * KV) of state pool 0 and [0, NV * HP) of history pool 0 form a null block that no
    request owns (vLLM's block 0); a batch row given request None points there."""

    def __init__(self, n_req, kind, gen, null=False):
        self.null = null
        first_s, first_h = (NV * KV, NV * HP) if null else (0, 0)
        if kind == "rows_in_order":
            assert not null
            self.state_pages, self.hist_pages = [n_req * NV * KV, 0], [n_req * NV * HP, 0]
            self.st = [[(0, (r * NV + h) * KV) for h in range(NV)] for r in range(n_req)]
            self.hs = [[(0, (r * NV + h) * HP) for h in range(NV)] for r in range(n_req)]
        elif kind == "random":
            self.state_pages = [first_s + (n_req * NV + 3) * KV + 13, 0]
            self.hist_pages = [first_h + (n_req * NV + 3) * HP + 7, 0]
            ss = _random_starts(n_req * NV, KV, self.state_pages[0], gen, first_s)
            hs = _random_starts(n_req * NV, HP, self.hist_pages[0], gen, first_h)
            self.st = [[(0, ss[r * NV + h]) for h in range(NV)] for r in range(n_req)]
            self.hs = [[(0, hs[r * NV + h]) for h in range(NV)] for r in range(n_req)]
        else:
            assert kind == "two_pools" and not null
            self.state_pages = [(n_req + 2) * BLOCK + 21, (n_req + 2) * BLOCK + 9]
            self.hist_pages = [3 * NV * HP + 5, (n_req + 2) * NV * HP + 11]
            k = _random_starts(n_req, BLOCK, self.state_pages[0], gen)
            v = _random_starts(n_req, BLOCK, self.state_pages[1], gen)
            c = _random_starts(n_req, NV * HP, self.hist_pages[1], gen)
            self.st = [
                [(0, k[r] + h * KV) if h < K_HEADS else (1, v[r] + (h - K_HEADS) * KV) for h in range(NV)]
                for r in range(n_req)
            ]
            self.hs = [[(1, c[r] + h * HP) for h in range(NV)] for r in range(n_req)]

    def _pages(self, r):
        if r is None:
            return [(0, h * KV) for h in range(NV)], [(0, h * HP) for h in range(NV)]
        return self.st[r], self.hs[r]

    def index_host(self, perm):
        """[2*NV, B] int32 index for batch row b holding request perm[b] (None: the null block); pool 1 pages are
        numbered after pool 0's."""
        g = lambda pool_page, sizes: pool_page[1] + (sizes[0] if pool_page[0] == 1 else 0)
        rows = [[g(self._pages(r)[0][h], self.state_pages) for r in perm] for h in range(NV)]
        rows += [[g(self._pages(r)[1][h], self.hist_pages) for r in perm] for h in range(NV)]
        return torch.tensor(rows, dtype=torch.int32)

    def index(self, device, perm, dtype=ttnn.uint32):
        return ttnn.from_torch(self.index_host(perm), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def host_pools(self, inp, gen, requests=None):
        """Host pools of random values with the requests' state and history placed: (state pools, history pools)."""
        sp = [torch.randn(1, n, 32, 32, generator=gen).float() if n else None for n in self.state_pages]
        hp = [torch.randn(1, n, 32, 32, generator=gen).bfloat16() if n else None for n in self.hist_pages]
        self.place(sp, hp, inp, range(len(self.st)) if requests is None else requests)
        return sp, hp

    def place(self, sp, hp, inp, requests, src=None):
        """Write request r's state and history (from inp entry src[i], default r) into the host pools."""
        for i, r in enumerate(requests):
            s = r if src is None else src[i]
            for h in range(NV):
                pool, page = self.st[r][h]
                sp[pool][0, page : page + KV] = tiles(inp["h"][s, h])
                pool, page = self.hs[r][h]
                hp[pool][0, page : page + HP] = inp["hist2"][s, h]

    def owned_masks(self, requests, with_null=False):
        """Per pool, the pages of `requests` (and of the null block)."""
        s_mask = [torch.zeros(n, dtype=torch.bool) for n in self.state_pages]
        h_mask = [torch.zeros(n, dtype=torch.bool) for n in self.hist_pages]
        for r in list(requests) + ([None] if with_null else []):
            st, hs = self._pages(r)
            for h in range(NV):
                s_mask[st[h][0]][st[h][1] : st[h][1] + KV] = True
                h_mask[hs[h][0]][hs[h][1] : hs[h][1] + HP] = True
        return s_mask, h_mask

    def check(self, pools0, pools, refs, requests, with_null=False):
        """The requests' pages equal refs[r] = (state [NV, DK, DV], paged history [NV, 4, 32, 32]) bit for bit;
        every other page (except the null block's when with_null) is unchanged."""
        (sp0, hp0), (sp, hp) = pools0, pools
        for r in requests:
            h_ref, hist_ref = refs[r]
            for h in range(NV):
                pool, page = self.st[r][h]
                assert torch.equal(
                    bits(sp[pool][0, page : page + KV]), bits(tiles(h_ref[h]))
                ), f"request {r} head {h}: state"
                pool, page = self.hs[r][h]
                assert torch.equal(
                    bits(hp[pool][0, page : page + HP]), bits(hist_ref[h])
                ), f"request {r} head {h}: history"
        s_mask, h_mask = self.owned_masks(requests, with_null)
        for p in range(2):
            if self.state_pages[p]:
                assert torch.equal(bits(sp[p][0, ~s_mask[p]]), bits(sp0[p][0, ~s_mask[p]])), f"state pool {p} changed"
            if self.hist_pages[p]:
                assert torch.equal(bits(hp[p][0, ~h_mask[p]]), bits(hp0[p][0, ~h_mask[p]])), f"history pool {p} changed"


def _pools_dev(device, pools):
    sp, hp = pools
    to_dev = lambda ts, dtype: [tile_dev(device, t, dtype) if t is not None else None for t in ts]
    return to_dev(sp, ttnn.float32), to_dev(hp, ttnn.bfloat16)


def _pools_host(pools_dev):
    to_host = lambda ts: [ttnn.to_torch(t) if t is not None else None for t in ts]
    return to_host(pools_dev[0]), to_host(pools_dev[1])


def _write_pools(pools_dev, pools):
    for t_dev, t_host in zip(pools_dev[0] + pools_dev[1], pools[0] + pools[1]):
        if t_dev is not None:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t_host, dtype=t_dev.dtype, layout=ttnn.TILE_LAYOUT), t_dev)


def _run_pooled(x, consts, pools_dev, idx, fast_mode, fp32_state):
    (sa, sb), (ha, hb) = pools_dev
    return run(x, consts, sa, ha, fast_mode, fp32_state, state_page_idx=idx, state_pool2=sb, conv_hist_pool2=hb)


def _x(device, b, gen):
    return tile_dev(device, token_rows(b, gen).reshape(1, b, W), ttnn.bfloat16)


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("B", [1, 8, 32])
@pytest.mark.parametrize("kind", ["rows_in_order", "random", "two_pools"])
def test_paged_matches_default(device, B, fast_mode, fp32_state, kind):
    """One step with distinct pages per (row, head), request r on batch row r: output, state and history bit-identical
    to the default layout; every other pool page unchanged."""
    gen = torch.Generator().manual_seed(100 + B)
    inp = inputs(B, seed=B)
    consts = consts_dev(device, inp)
    x = _x(device, B, gen)
    out_ref, h_new, hist_new = default_step(
        device, x, consts, list(inp["h"]), list(inp["hist2"]), fast_mode, fp32_state
    )

    lay = Layout(B, kind, gen)
    pools0 = lay.host_pools(inp, gen)
    pools_dev = _pools_dev(device, pools0)
    idx = lay.index(device, list(range(B)), dtype=ttnn.int32 if B == 8 else ttnn.uint32)
    out = ttnn.to_torch(_run_pooled(x, consts, pools_dev, idx, fast_mode, fp32_state))
    assert torch.equal(out, out_ref), "output differs from the default layout"
    lay.check(pools0, _pools_host(pools_dev), {r: (h_new[r], hist_new[r]) for r in range(B)}, range(B))


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("kind", ["random", "two_pools"])
def test_paged_rows_change_parity(device, fast_mode, fp32_state, kind):
    """Eight requests keep their pages while their batch rows change every step, so each request sits on even and odd
    rows in turn. Every step matches the default layout with the same request on the same row bit for bit."""
    B = 8
    gen = torch.Generator().manual_seed(77)
    inp = inputs(B, seed=31)
    consts = consts_dev(device, inp)
    lay = Layout(B, kind, gen)
    pools0 = lay.host_pools(inp, gen)
    pools_dev = _pools_dev(device, pools0)
    ref = {r: (inp["h"][r].clone(), inp["hist2"][r].clone()) for r in range(B)}
    perms = [
        list(range(B)),
        [1, 0, 3, 2, 5, 4, 7, 6],
        [7, 0, 1, 2, 3, 4, 5, 6],
        torch.randperm(B, generator=gen).tolist(),
    ]
    for step, perm in enumerate(perms):  # batch row b holds request perm[b]
        x = _x(device, B, gen)
        out_ref, h_new, hist_new = default_step(
            device, x, consts, [ref[r][0] for r in perm], [ref[r][1] for r in perm], fast_mode, fp32_state
        )
        for b, r in enumerate(perm):
            ref[r] = (h_new[b], hist_new[b])
        out = ttnn.to_torch(_run_pooled(x, consts, pools_dev, lay.index(device, perm), fast_mode, fp32_state))
        assert torch.equal(out, out_ref), f"step {step}: output differs"
        lay.check(pools0, _pools_host(pools_dev), ref, range(B))


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=MODE_IDS)
def test_paged_steps_vs_reference(device, fast_mode, fp32_state):
    """Eight requests run eight steps from state and history held only in their pages, moving between batch rows of
    both parities. After four steps the history in the pages consists of tokens the op shifted in itself. Outputs and
    states follow the torch reference; the history pages equal the reference history exactly."""
    B, steps = 8, 8
    gen = torch.Generator().manual_seed(5)
    inp = inputs(B, seed=5)
    consts = consts_dev(device, inp)
    lay = Layout(B, "random", gen)
    pools_dev = _pools_dev(device, lay.host_pools(inp, gen))
    taps = [t.float() for t in inp["tap_rows"]]
    h_ref = {r: inp["h"][r].clone() for r in range(B)}
    rows_ref = {r: [t.float() for t in inp["rows"][r]] for r in range(B)}
    idx = lay.index(device, list(range(B)))
    for step in range(steps):
        perm = torch.randperm(B, generator=gen).tolist()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(lay.index_host(perm), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), idx
        )
        rows = token_rows(B, gen)
        out = ttnn.to_torch(
            _run_pooled(
                tile_dev(device, rows.reshape(1, B, W), ttnn.bfloat16), consts, pools_dev, idx, fast_mode, fp32_state
            )
        ).reshape(B, -1)
        sp, hp = _pools_host(pools_dev)
        worst_out, worst_state = 1.0, 1.0
        for b, r in enumerate(perm):
            out_r, h_ref[r], rows_ref[r] = step_reference(
                rows_ref[r][1:],
                rows[b].float(),
                taps,
                inp["dtb"].float(),
                inp["nea"].float(),
                h_ref[r],
                inp["w"].float(),
            )
            worst_out = min(worst_out, pcc(out[b], out_r))
            hist_r = pack_rows([t.bfloat16() for t in rows_ref[r]], both=True)
            for h in range(NV):
                pool, page = lay.st[r][h]
                worst_state = min(worst_state, pcc(sp[pool][0, page : page + KV], tiles(h_ref[r][h])))
                pool, page = lay.hs[r][h]
                assert torch.equal(
                    hp[pool][0, page : page + HP], hist_r[h]
                ), f"step {step} request {r} head {h}: history"
        logger.info(f"step {step}: worst out pcc {worst_out:.6f}, worst state pcc {worst_state:.6f}")
        assert worst_out > 0.999 and worst_state > (0.999 if fast_mode else 0.9999), (step, worst_out, worst_state)


@pytest.mark.parametrize("null_fill", ["zero", "random", "nan"])
def test_paged_padding_rows_use_a_null_block(device, null_fill):
    """Batch rows without a request point at one null block (all of them at the same pages). Over three steps the
    live rows match the default layout bit for bit, also when the null block holds NaN, and no page outside the null
    block and the live requests' pages changes."""
    B, steps = 8, 3
    live = {0: 0, 2: 1, 3: 2, 6: 3}  # batch row -> request; rows 1, 4, 5 and 7 are padding
    gen = torch.Generator().manual_seed(9)
    inp = inputs(len(live), seed=9)
    consts = consts_dev(device, inp)
    lay = Layout(len(live), "random", gen, null=True)
    sp, hp = lay.host_pools(inp, gen)
    if null_fill != "random":
        fill = {"zero": 0.0, "nan": float("nan")}[null_fill]
        sp[0][0, : NV * KV] = fill
        hp[0][0, : NV * HP] = fill
    pools0 = (sp, hp)
    pools_dev = _pools_dev(device, pools0)
    perm = [live.get(b) for b in range(B)]
    idx = lay.index(device, perm)
    ref = {r: (inp["h"][r].clone(), inp["hist2"][r].clone()) for r in live.values()}
    zero_h, zero_hist = torch.zeros(NV, DK, DV), torch.zeros(NV, 4, 32, 32, dtype=torch.bfloat16)
    for step in range(steps):
        x = _x(device, B, gen)
        out_ref, h_new, hist_new = default_step(
            device,
            x,
            consts,
            [ref[r][0] if r is not None else zero_h for r in perm],
            [ref[r][1] if r is not None else zero_hist for r in perm],
            *MODEL_MODE,
        )
        out = ttnn.to_torch(_run_pooled(x, consts, pools_dev, idx, *MODEL_MODE))
        for b, r in enumerate(perm):
            if r is not None:
                ref[r] = (h_new[b], hist_new[b])
                assert torch.equal(out[0, b], out_ref[0, b]), f"step {step}: live row {b} output differs"
        lay.check(pools0, _pools_host(pools_dev), ref, live.values(), with_null=True)


@pytest.mark.parametrize("poison", ["nan", "inf"])
def test_paged_ignores_bytes_it_does_not_read(device, poison):
    """The op reads only a request's state pages and history slots 1..3; it overwrites history slot 0 without reading
    it. With slot 0 and every page outside the requests' pages poisoned, outputs, states and histories equal a run
    on clean pools bit for bit, the poisoned pages outside the requests stay as they were, and slot 0 holds the old
    slot 1."""
    B = 8
    value = float(poison)
    gen = torch.Generator().manual_seed(21)
    inp = inputs(B, seed=21)
    consts = consts_dev(device, inp)
    lay = Layout(B, "random", gen)
    clean = lay.host_pools(inp, gen)
    s_mask, h_mask = lay.owned_masks(range(B))
    sp, hp = clone = ([t.clone() for t in clean[0][:1]] + [None], [t.clone() for t in clean[1][:1]] + [None])
    sp[0][0, ~s_mask[0]] = value
    hp[0][0, ~h_mask[0]] = value
    for r in range(B):
        for h in range(NV):
            hp[0][0, lay.hs[r][h][1]] = value  # slot 0
    x = _x(device, B, gen)
    idx = lay.index(device, list(range(B)))
    results = []
    for pools in (clean, clone):
        pools_dev = _pools_dev(device, pools)
        out = ttnn.to_torch(_run_pooled(x, consts, pools_dev, idx, *MODEL_MODE))
        results.append((out, _pools_host(pools_dev)))
    (out_c, (sp_c, hp_c)), (out_p, (sp_p, hp_p)) = results
    assert torch.equal(out_p, out_c), "output differs with poisoned unread bytes"
    assert torch.equal(bits(sp_p[0][0, s_mask[0]]), bits(sp_c[0][0, s_mask[0]])), "state pages differ"
    assert torch.equal(bits(hp_p[0][0, h_mask[0]]), bits(hp_c[0][0, h_mask[0]])), "history pages differ"
    assert torch.equal(bits(sp_p[0][0, ~s_mask[0]]), bits(sp[0][0, ~s_mask[0]])), "unowned state pages changed"
    assert torch.equal(bits(hp_p[0][0, ~h_mask[0]]), bits(hp[0][0, ~h_mask[0]])), "unowned history pages changed"
    for r in range(B):
        for h in range(NV):
            page = lay.hs[r][h][1]
            assert torch.equal(hp_p[0][0, page], clean[1][0][0, page + 1]), f"request {r} head {h}: slot 0"


def test_paged_ignores_unused_history_rows(device):
    """A packed history tile uses rows 0..23 (12 channel chunks, both parities); NaN in rows 24..31 of history slots
    1..3 changes no output, state or used history row."""
    B = 8
    gen = torch.Generator().manual_seed(23)
    inp = inputs(B, seed=23)
    consts = consts_dev(device, inp)
    lay = Layout(B, "random", gen)
    clean = lay.host_pools(inp, gen)
    poisoned = ([clean[0][0].clone(), None], [clean[1][0].clone(), None])
    for r in range(B):
        for h in range(NV):
            page = lay.hs[r][h][1]
            poisoned[1][0][0, page + 1 : page + HP, 24:, :] = float("nan")
    x = _x(device, B, gen)
    idx = lay.index(device, list(range(B)))
    results = []
    for pools in (clean, poisoned):
        pools_dev = _pools_dev(device, pools)
        out = ttnn.to_torch(_run_pooled(x, consts, pools_dev, idx, *MODEL_MODE))
        results.append((out, _pools_host(pools_dev)))
    (out_c, (sp_c, hp_c)), (out_p, (sp_p, hp_p)) = results
    assert torch.equal(out_p, out_c), "output differs with NaN in unused history rows"
    assert torch.equal(bits(sp_p[0]), bits(sp_c[0])), "state differs with NaN in unused history rows"
    assert torch.equal(hp_p[0][..., :24, :], hp_c[0][..., :24, :]), "used history rows differ"


def test_paged_reuses_the_same_pages(device):
    """Three generations of requests run on the same pages with one persistent index tensor: each generation's
    initial state and history are written into the pages a finished generation left, the rows move between steps, and
    every step matches the default layout bit for bit."""
    B = 4
    gen = torch.Generator().manual_seed(33)
    lay = Layout(B, "two_pools", gen)
    first = inputs(B, seed=40)
    consts = consts_dev(device, first)
    pools0 = lay.host_pools(first, gen)
    pools_dev = _pools_dev(device, pools0)
    idx = lay.index(device, list(range(B)))
    for generation in range(3):
        inp = first if generation == 0 else inputs(B, seed=40 + generation)
        inp = {**inp, "w": first["w"], "tap_rows": first["tap_rows"], "dtb": first["dtb"], "nea": first["nea"]}
        if generation > 0:
            sp, hp = _pools_host(pools_dev)
            lay.place(sp, hp, inp, range(B))
            _write_pools(pools_dev, (sp, hp))
        before = _pools_host(pools_dev)
        ref = {r: (inp["h"][r].clone(), inp["hist2"][r].clone()) for r in range(B)}
        for step, perm in enumerate([list(range(B)), [3, 2, 1, 0]]):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(lay.index_host(perm), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), idx
            )
            x = _x(device, B, gen)
            out_ref, h_new, hist_new = default_step(
                device, x, consts, [ref[r][0] for r in perm], [ref[r][1] for r in perm], *MODEL_MODE
            )
            for b, r in enumerate(perm):
                ref[r] = (h_new[b], hist_new[b])
            out = ttnn.to_torch(_run_pooled(x, consts, pools_dev, idx, *MODEL_MODE))
            assert torch.equal(out, out_ref), f"generation {generation} step {step}: output differs"
            lay.check(before, _pools_host(pools_dev), ref, range(B))


def test_paged_program_cache_follows_new_buffers(device):
    """A second set of pools and index (new allocations, same shapes) reuses the cached program: no new program
    cache entry, the new pools get the step, and the first pools stay as the first call left them."""
    B = 8
    gen = torch.Generator().manual_seed(61)
    inp = inputs(B, seed=61)
    consts = consts_dev(device, inp)
    x = _x(device, B, gen)
    out_ref, h_new, hist_new = default_step(device, x, consts, list(inp["h"]), list(inp["hist2"]), *MODEL_MODE)
    refs = {r: (h_new[r], hist_new[r]) for r in range(B)}
    lay = Layout(B, "two_pools", gen)
    pools0 = lay.host_pools(inp, gen)
    first = _pools_dev(device, pools0)
    idx1 = lay.index(device, list(range(B)))
    _run_pooled(x, consts, first, idx1, *MODEL_MODE)
    after_first = _pools_host(first)
    entries = device.num_program_cache_entries()

    second = _pools_dev(device, pools0)
    idx2 = lay.index(device, list(range(B)))
    assert second[0][0].buffer_address() != first[0][0].buffer_address()
    out = ttnn.to_torch(_run_pooled(x, consts, second, idx2, *MODEL_MODE))
    assert device.num_program_cache_entries() == entries, "new buffers compiled a new program"
    assert torch.equal(out, out_ref)
    lay.check(pools0, _pools_host(second), refs, range(B))
    first_now = _pools_host(first)
    for a, b in zip(first_now[0] + first_now[1], after_first[0] + after_first[1]):
        if a is not None:
            assert torch.equal(bits(a), bits(b)), "the second call wrote the first pools"


def test_paged_rejects_bad_index(device, expect_error):
    """The index must be [2*NV, >= B]: a [2, B] index and a short one are refused on the host."""
    B = 8
    gen = torch.Generator().manual_seed(0)
    inp = inputs(B, seed=0)
    consts = consts_dev(device, inp)
    lay = Layout(B, "random", gen)
    (sa, _), (ha, _) = _pools_dev(device, lay.host_pools(inp, gen))
    x = _x(device, B, gen)
    for shape in [(2, B), (2 * NV, B - 1)]:
        idx = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        with expect_error(RuntimeError, "state_page_idx must be"):
            run(x, consts, sa, ha, *MODEL_MODE, state_page_idx=idx)


@pytest.mark.parametrize("pool", ["state_pool2", "conv_hist_pool2"])
def test_paged_rejects_a_second_pool_of_smaller_tiles(device, expect_error, pool):
    """The kernels read a second pool with the first pool's page size: a second pool of 16x32 tiles, with the right
    dtype and enough pages, is refused on the host."""
    B = 8
    gen = torch.Generator().manual_seed(0)
    inp = inputs(B, seed=0)
    consts = consts_dev(device, inp)
    lay = Layout(B, "two_pools", gen)
    sp, hp = lay.host_pools(inp, gen)
    (sa, sb), (ha, hb) = _pools_dev(device, (sp, hp))
    small = lambda t, dtype: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((16, 32)), device=device
    )
    if pool == "state_pool2":
        sb = small(sp[1], ttnn.float32)
    else:
        hb = small(hp[1], ttnn.bfloat16)
    x = _x(device, B, gen)
    idx = lay.index(device, list(range(B)))
    with expect_error(RuntimeError, f"{pool} must use 32x32 tiles"):
        run(x, consts, sa, ha, *MODEL_MODE, state_page_idx=idx, state_pool2=sb, conv_hist_pool2=hb)


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("B", [8, 32])
def test_paged_trace_follows_index(device, B):
    """two_pools layout with 2B requests; request B + i starts from the same data as request i. Capture one trace with
    requests 0..B-1 on rows 0..B-1 and replay; rewrite only the index's contents to requests B..2B-1 on rows shifted by
    one (every request changes parity) and replay. Each replay updates exactly the indexed pages, bit-identical to the
    default layout from the same starting state."""
    gen = torch.Generator().manual_seed(7)
    inp = inputs(B, seed=50 + B)
    consts = consts_dev(device, inp)
    inp2 = {**inp, "h": torch.cat([inp["h"], inp["h"]]), "hist2": torch.cat([inp["hist2"], inp["hist2"]])}
    lay = Layout(2 * B, "two_pools", gen)
    pools0 = lay.host_pools(inp2, gen)
    pools_dev = _pools_dev(device, pools0)
    x = _x(device, B, gen)
    shift = [(b + 1) % B for b in range(B)]  # replay 1: row b holds a copy of request shift[b]
    perms = [list(range(B)), [B + i for i in shift]]
    refs = [
        default_step(device, x, consts, list(inp["h"]), list(inp["hist2"]), *MODEL_MODE),
        default_step(device, x, consts, [inp["h"][i] for i in shift], [inp["hist2"][i] for i in shift], *MODEL_MODE),
    ]
    idx = lay.index(device, perms[0])

    # compile outside the trace, then put the pools back
    _run_pooled(x, consts, pools_dev, idx, *MODEL_MODE)
    _write_pools(pools_dev, pools0)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_dev = _run_pooled(x, consts, pools_dev, idx, *MODEL_MODE)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        before = _pools_host(pools_dev)
        for k, perm in enumerate(perms):
            if k > 0:
                new_idx = ttnn.from_torch(lay.index_host(perm), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn.copy_host_to_device_tensor(new_idx, idx)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            after = _pools_host(pools_dev)
            out_ref, h_new, hist_new = refs[k]
            assert torch.equal(ttnn.to_torch(out_dev), out_ref), f"replay {k}: output differs"
            lay.check(before, after, {r: (h_new[b], hist_new[b]) for b, r in enumerate(perm)}, perm)
            before = after
    finally:
        ttnn.release_trace(device, tid)


# ---- the layout of the 1024-token cache blocks of a TP=8 Qwen3.8 layer slot


def _fp32_tile_raw(t):
    """[n, 32, 32] fp32 tiles -> [n, 1024] int32 in device byte order (four 16x16 faces)."""
    return t.reshape(-1, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(-1, 1024).view(torch.int32)


def _raw_fp32_tile(raw):
    return raw.view(torch.float32).reshape(-1, 2, 2, 16, 16).permute(0, 1, 3, 2, 4).reshape(-1, 32, 32)


def _bf16_tile_raw(t):
    return t.reshape(-1, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(-1, 1024).view(torch.int16)


class KvBlocks:
    """One layer slot of a TP=8 Qwen3.8 model: bfloat8_b K and V caches [N, 1, 1024, 256], one 1024-token block per
    request. A block holds 34,816 bytes per DRAM bank in each of K and V (8.5 fp32 pages), 64 whole fp32 pages per
    block in K: one request's GDN state (6 heads of 16 fp32 pages, 384 KiB) plus its history (24 bf16 pages, 48 KiB)
    does not fit K alone. Heads 0..3 go to the first whole fp32 page rows of the K block, heads 4..5 to the first ones
    of the V block, and the history to the last three bf16 page rows of the V block.

    Host content is kept as the pages of fp32 views of K and V (whole buffers for even N)."""

    def __init__(self, device, num_blocks, gen):
        assert num_blocks % 2 == 0
        self.device, self.n = device, num_blocks
        self.nb = device.dram_grid_size().x * device.dram_grid_size().y
        tiles_per_block = 1024 * 256 // 1024
        self.s = tiles_per_block // self.nb * BF8_PAGE  # bytes per bank per block
        self.hist_tiles = tiles_per_block * BF8_PAGE // BF16_PAGE
        self.hist_first = self.hist_tiles - -(-NV * HP // self.nb) * self.nb
        self.rec_pages = self.nb * (num_blocks * self.s // FP32_PAGE)
        bf8 = lambda: ttnn.from_torch(
            torch.zeros(num_blocks, 1, 1024, 256), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.k, self.v = bf8(), bf8()
        view_shape = [self.rec_pages // 8, 1, DK // 2, DV]
        check = self.s % FP32_PAGE == 0
        self.rec_k = ttnn.experimental.alias_view(self.k, view_shape, ttnn.float32, check_slices=check)
        self.rec_v = ttnn.experimental.alias_view(self.v, view_shape, ttnn.float32, check_slices=check)
        self.hist = ttnn.experimental.alias_view(self.v, [num_blocks, self.hist_tiles, 32, 32], ttnn.bfloat16)
        # random fp32 pages stand for the attention contents of K and V
        self.k_pages = torch.randn(self.rec_pages, 32, 32, generator=gen)
        self.v_pages = torch.randn(self.rec_pages, 32, 32, generator=gen)

    def first_page(self, block):
        """First whole fp32 page of `block` in either view: its first whole page row times the bank count."""
        return self.nb * -(-block * self.s // FP32_PAGE)

    def state_page(self, block, h):
        """(pool, first fp32 page) of head h of the request in `block`."""
        p = self.first_page(block)
        return (0, p + h * KV) if h < K_HEADS else (1, p + (h - K_HEADS) * KV)

    def hist_page(self, block, h):
        return block * self.hist_tiles + self.hist_first + h * HP

    def index_host(self, blocks):
        rows = [[self.state_page(b, h)[1] + (self.rec_pages if h >= K_HEADS else 0) for b in blocks] for h in range(NV)]
        rows += [[self.hist_page(b, h) for b in blocks] for h in range(NV)]
        return torch.tensor(rows, dtype=torch.int32)

    def _v_slot(self, q2):
        """bf16 page q2 of V -> (fp32 page, int16 offset) holding its bytes."""
        row = q2 // self.nb
        return q2 % self.nb + self.nb * (row // 2), (row % 2) * 1024

    def place(self, entries):
        """Host: for each (block, state [NV, DK, DV], paged history [NV, 4, 32, 32]), the request in the block holds
        that state and history."""
        for block, h_state, _ in entries:
            for h in range(NV):
                pool, page = self.state_page(block, h)
                (self.k_pages, self.v_pages)[pool][page : page + KV] = tiles(h_state[h])
        raw16 = _fp32_tile_raw(self.v_pages).view(torch.int16)
        for block, _, hist2 in entries:
            for h in range(NV):
                for s, t in enumerate(_bf16_tile_raw(hist2[h])):
                    q, off = self._v_slot(self.hist_page(block, h) + s)
                    raw16[q, off : off + 1024] = t
        self.v_pages = _raw_fp32_tile(raw16.view(torch.int32))

    def write(self):
        for view, pages in ((self.rec_k, self.k_pages), (self.rec_v, self.v_pages)):
            host = pages.reshape(self.rec_pages // 8, 2, 4, 32, 32).permute(0, 1, 3, 2, 4).reshape(-1, 1, DK // 2, DV)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), view)

    def read(self):
        """Device -> (K fp32 pages, V fp32 pages, history bf16 pages [N * hist_tiles, 32, 32])."""
        k, v = (tiles(ttnn.to_torch(view)) for view in (self.rec_k, self.rec_v))
        return k, v, ttnn.to_torch(self.hist).reshape(-1, 32, 32)

    def owned(self, blocks):
        """Masks of the fp32 K and V pages and the bf16 history pages the requests in `blocks` own."""
        k_mask = torch.zeros(self.rec_pages, dtype=torch.bool)
        v_mask = torch.zeros(self.rec_pages, dtype=torch.bool)
        h_mask = torch.zeros(self.n * self.hist_tiles, dtype=torch.bool)
        for blk in blocks:
            for h in range(NV):
                pool, page = self.state_page(blk, h)
                (k_mask, v_mask)[pool][page : page + KV] = True
                hp = self.hist_page(blk, h)
                h_mask[hp : hp + HP] = True
                for q2 in range(hp, hp + HP):
                    v_mask[self._v_slot(q2)[0]] = True
        return k_mask, v_mask, h_mask

    def run(self, x, consts, idx):
        return run(x, consts, self.rec_k, self.hist, *MODEL_MODE, state_page_idx=idx, state_pool2=self.rec_v)


def _kv_blocks_or_skip(device, num_blocks, gen):
    banks = device.dram_grid_size().x * device.dram_grid_size().y
    if (1024 * 256 // 1024) % banks:
        pytest.skip(f"a 1024-token block does not spread evenly over {banks} DRAM banks")
    return KvBlocks(device, num_blocks, gen)


def _check_kv(kv, got, blocks, refs):
    """The requests in `blocks` hold refs[i] = (state, paged history); every other K and V byte is unchanged."""
    k, v, hist = got
    for blk, (h_ref, hist_ref) in zip(blocks, refs):
        for h in range(NV):
            pool, page = kv.state_page(blk, h)
            assert torch.equal(bits((k, v)[pool][page : page + KV]), bits(tiles(h_ref[h]))), f"block {blk} head {h}"
            hp = kv.hist_page(blk, h)
            assert torch.equal(hist[hp : hp + HP], hist_ref[h]), f"block {blk} head {h}: history"
    k_mask, v_mask, _ = kv.owned(blocks)
    assert torch.equal(bits(k[~k_mask]), bits(kv.k_pages[~k_mask])), "K bytes outside the requests' pages changed"
    assert torch.equal(bits(v[~v_mask]), bits(kv.v_pages[~v_mask])), "V bytes outside the requests' pages changed"


@pytest.mark.parametrize("B", [1, 8, 32])
def test_paged_in_1024_token_kv_blocks(device, B):
    """B requests in odd and even 1024-token blocks of bfloat8_b K and V caches, through alias_view views: one step
    matches the default layout bit for bit and changes no other K or V byte. A second cache allocation reuses the
    cached program."""
    gen = torch.Generator().manual_seed(70 + B)
    num_blocks = 2 * B + 4
    inp = inputs(B, seed=70 + B)
    consts = consts_dev(device, inp)
    blocks = torch.randperm(num_blocks - 1, generator=gen)[:B].add(1).tolist()  # block 0 stays the null block
    x = _x(device, B, gen)
    out_ref, h_new, hist_new = default_step(device, x, consts, list(inp["h"]), list(inp["hist2"]), *MODEL_MODE)
    entries = None
    for allocation in range(2):
        kv = _kv_blocks_or_skip(device, num_blocks, gen)
        kv.place(list(zip(blocks, inp["h"], inp["hist2"])))
        kv.write()
        _check_kv(kv, kv.read(), blocks, list(zip(inp["h"], inp["hist2"])))  # the views place what the host model says
        idx = ttnn.from_torch(kv.index_host(blocks), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = ttnn.to_torch(kv.run(x, consts, idx))
        assert torch.equal(out, out_ref), f"allocation {allocation}: output differs from the default layout"
        _check_kv(kv, kv.read(), blocks, list(zip(h_new, hist_new)))
        if entries is None:
            entries = device.num_program_cache_entries()
            first = kv
        else:
            assert kv.k.buffer_address() != first.k.buffer_address()
            assert device.num_program_cache_entries() == entries, "a new cache allocation compiled a new program"


def test_kv_block_copy_and_clear_with_existing_ops(device):
    """The block operations the model needs exist for bfloat8_b caches: ttnn.slice plus paged_fill_cache copies a
    whole block byte for byte, so the GDN state in a K / V block reaches a block whose pages start at the same bank
    offset (same block parity at 1024 tokens); paged_fill_cache of a zero block clears it. Copies run in order: A to
    B, then B to C. The copied state then steps exactly like the original."""
    gen = torch.Generator().manual_seed(90)
    inp = inputs(1, seed=90)
    consts = consts_dev(device, inp)
    kv = _kv_blocks_or_skip(device, 10, gen)
    a, b, c, z = 3, 5, 9, 7
    kv.place([(a, inp["h"][0], inp["hist2"][0])])
    kv.write()

    def copy_block(src, dst):
        page_table = ttnn.from_torch(torch.tensor([[dst]], dtype=torch.int32), dtype=ttnn.int32, device=device)
        for t in (kv.k, kv.v):
            blk = ttnn.slice(t, (src, 0, 0, 0), (src + 1, 1, 1024, 256), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.experimental.paged_fill_cache(t, blk, page_table, batch_idx=0)

    copy_block(a, b)
    copy_block(b, c)
    zeros = ttnn.from_torch(torch.zeros(1, 1, 1024, 256), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    for t in (kv.k, kv.v):
        ttnn.experimental.paged_fill_cache(
            t,
            zeros,
            ttnn.from_torch(torch.tensor([[z]], dtype=torch.int32), dtype=ttnn.int32, device=device),
            batch_idx=0,
        )
    k, v, hist = kv.read()
    for blk in (b, c):
        for h in range(NV):
            pool, page = kv.state_page(blk, h)
            assert torch.equal(bits((k, v)[pool][page : page + KV]), bits(tiles(inp["h"][0, h]))), f"block {blk}"
            hp = kv.hist_page(blk, h)
            assert torch.equal(hist[hp : hp + HP], inp["hist2"][0, h]), f"block {blk}: history"
    for pages in (k, v):
        lo, hi = kv.first_page(z), kv.first_page(z + 1)
        assert torch.equal(bits(pages[lo:hi]), torch.zeros_like(bits(pages[lo:hi]))), "cleared block is not zero"

    x = _x(device, 2, gen)
    out = ttnn.to_torch(kv.run(x, consts, ttnn.from_torch(kv.index_host([a, c]), dtype=ttnn.uint32, device=device)))
    out_ref, h_new, hist_new = default_step(device, x, consts, [inp["h"][0]] * 2, [inp["hist2"][0]] * 2, *MODEL_MODE)
    assert torch.equal(out, out_ref)
    got = kv.read()
    for blk, i in ((a, 0), (c, 1)):
        for h in range(NV):
            pool, page = kv.state_page(blk, h)
            assert torch.equal(bits((got[0], got[1])[pool][page : page + KV]), bits(tiles(h_new[i][h])))


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("B", [1, 8, 32])
def test_gdn_decode_step_traced_latency(device, B):
    """Traced time per call in the model's mode: the default layout, page pools (two_pools) and the 1024-token K / V
    blocks. One trace holds N back-to-back calls; the median of REPS replays is reported. Not a pass / fail gate."""
    n_calls, reps = 64, 15
    gen = torch.Generator().manual_seed(3)
    inp = inputs(B, seed=B)
    consts = consts_dev(device, inp)
    x = _x(device, B, gen)
    h = torch.zeros(BMAX, NV, DK, DV)
    hist = torch.zeros(BMAX, NV, 4, 32, 32, dtype=torch.bfloat16)
    for b in range(B):
        h[b], hist[b] = inp["h"][b], from_pooled(inp["hist2"][b], b)
    default = (tile_dev(device, h, ttnn.float32), tile_dev(device, hist, ttnn.bfloat16))
    lay = Layout(B, "two_pools", gen)
    pools_dev = _pools_dev(device, lay.host_pools(inp, gen))
    idx = lay.index(device, list(range(B)))
    kv = _kv_blocks_or_skip(device, 2 * B + 2, gen)
    blocks = list(range(1, B + 1))
    kv.place(list(zip(blocks, inp["h"], inp["hist2"])))
    kv.write()
    kv_idx = ttnn.from_torch(kv.index_host(blocks), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def per_call_us(call):
        call()
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
        "default": lambda: run(x, consts, *default, *MODEL_MODE),
        "paged": lambda: _run_pooled(x, consts, pools_dev, idx, *MODEL_MODE),
        "kv_blocks_1024": lambda: kv.run(x, consts, kv_idx),
    }
    res = {k: [] for k in calls}
    for _ in range(2):  # alternate the layouts so drift hits all of them
        for k, call in calls.items():
            res[k].append(per_call_us(call))
    # DRAM traffic of the state per call: state pages read and written, history slots 1..3 read twice (reader and
    # writer), 4 history pages written
    state_bytes = B * NV * (2 * KV * FP32_PAGE + 10 * BF16_PAGE)
    summary = ", ".join(f"{k} {min(v):.2f} us" for k, v in res.items())
    logger.info(f"LATENCY B={B}: {summary}; state traffic {state_bytes / 2**20:.2f} MiB per call")
