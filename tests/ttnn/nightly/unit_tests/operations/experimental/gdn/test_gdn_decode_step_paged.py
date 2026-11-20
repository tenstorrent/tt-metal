# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.gdn.gdn_decode_step with state_page_idx: the recurrent state and the packed convolution history
live in page pools (for example vLLM-owned cache blocks), and a [2*NV, B] index names, per batch row and value head,
the first state page and the first history page. state_pool2 and conv_hist_pool2 extend the page space, so the heads
of one request can be split over a layer's K and V buffers.

Most tests compare with the op on the default [BMAX, ...] layout bit for bit; test_paged_steps_vs_reference compares
with the torch reference."""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.measure.determinism import (
    device_mismatch_marker_is_set,
    fold_mismatches,
    mismatch_report,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.gdn.gdn_decode_step_test_utils import (
    DETERMINISM_RUNS,
    DEVICE_PARAMS,
    DK,
    DV,
    HP,
    K_HEADS,
    KV,
    MODE_IDS,
    MODEL_MODE,
    MODES,
    NV,
    PERF_MARGIN,
    W,
    assert_close_to_reference,
    bits,
    consts_dev,
    default_step,
    inputs,
    kv_blocks_or_skip,
    pack_rows,
    run,
    step_reference,
    tile_dev,
    tiles,
    token_rows,
)
from tt_py_test_utils_common.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device(DEVICE_PARAMS)]

BLOCK = 64  # fp32 pages of one 1024-token bfloat8_b K (or V) block, the two_pools layout
# Device time of one gdn_decode_step at PERF_B rows in the model's call (1024-token K / V blocks), on a P150, and the
# band around it: a slowdown fails, and a speed-up asks for a new measurement.
# Measured 112.7 to 116.2 us over six runs on a host-IOMMU P150b.
PERF_B = 32
DECODE_STEP_DURATION_NS = 113_500


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
        out_ref = torch.zeros(B, NV * DV)
        state_ref, state = (torch.zeros(B, NV, KV, 32, 32) for _ in range(2))
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
            out_ref[b] = out_r
            hist_r = pack_rows([t.bfloat16() for t in rows_ref[r]], both=True)
            for h in range(NV):
                pool, page = lay.st[r][h]
                state[b, h], state_ref[b, h] = sp[pool][0, page : page + KV], tiles(h_ref[r][h])
                pool, page = lay.hs[r][h]
                assert torch.equal(
                    hp[pool][0, page : page + HP], hist_r[h]
                ), f"step {step} request {r} head {h}: history"
        assert_close_to_reference(f"step {step}", fast_mode, fp32_state, out_ref, out, state_ref, state)


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
        kv = kv_blocks_or_skip(device, num_blocks, gen)
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
    kv = kv_blocks_or_skip(device, 10, gen)
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


def _model_call(device, B, seed):
    """The model's gdn_decode_step call: B requests in blocks 1..B of 1024-token K / V blocks, fast_mode and
    fp32_state. Returns (kv, step), where step() runs the call once and returns its output."""
    gen = torch.Generator().manual_seed(seed)
    inp = inputs(B, seed=seed)
    consts = consts_dev(device, inp)
    x = _x(device, B, gen)
    kv = kv_blocks_or_skip(device, 2 * B + 2, gen)
    blocks = list(range(1, B + 1))
    kv.place(list(zip(blocks, inp["h"], inp["hist2"])))
    kv.write()
    idx = ttnn.from_torch(kv.index_host(blocks), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return kv, lambda: kv.run(x, consts, idx)


@pytest.mark.parametrize("B", [1, 8, 32])
def test_gdn_decode_step_is_device_deterministic(device, B):
    """DETERMINISM_RUNS back-to-back calls of the model's call from the same start state, with no sync between them:
    every run's output and every K / V byte equal the first run's. The op updates the state in place, so each run
    first copies the start state back into the blocks on device. The comparison stays on device; one marker is read
    back at the end."""
    kv, step = _model_call(device, B, seed=90 + B)
    views = [kv.rec_k, kv.rec_v]
    start = [ttnn.clone(view) for view in views]

    def run_from_start():
        for src, dst in zip(start, views):
            ttnn.copy(src, dst)
        return step()

    reference = [run_from_start()] + [ttnn.clone(view) for view in views]
    marker, outputs = None, None
    for _ in range(1, DETERMINISM_RUNS):
        if outputs is not None:
            ttnn.deallocate(outputs[0])
        outputs = [run_from_start()] + views
        marker = fold_mismatches(marker, reference, outputs)
    if device_mismatch_marker_is_set(marker):
        raise AssertionError(
            f"gdn_decode_step B={B} is not deterministic:\n"
            + mismatch_report(["output", "K fp32 view", "V fp32 view"], reference, outputs)
        )
    logger.info(f"gdn_decode_step B={B}: {DETERMINISM_RUNS} runs bit-identical")


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("LLK asserts change kernel timing.")
@skip_with_watcher("The watcher changes kernel timing.")
def test_gdn_decode_step_performance(device):
    """Device time of the model's call at PERF_B rows, inside +/- PERF_MARGIN of DECODE_STEP_DURATION_NS."""
    require_realtime_profiler("gdn_decode_step performance")
    _, step = _model_call(device, PERF_B, seed=3)
    ttnn.deallocate(step())  # compile outside the measured call
    out, record = profile_realtime_program(device, step)
    ttnn.deallocate(out)
    duration_ns = record["duration_ns"]
    logger.info(f"gdn_decode_step B={PERF_B}: {duration_ns:.0f} ns (reference {DECODE_STEP_DURATION_NS} ns)")
    if DECODE_STEP_DURATION_NS is None:
        pytest.fail(f"no reference duration: set DECODE_STEP_DURATION_NS from the measured {duration_ns:.0f} ns")
    lower, upper = DECODE_STEP_DURATION_NS * (1 - PERF_MARGIN), DECODE_STEP_DURATION_NS * (1 + PERF_MARGIN)
    assert lower <= duration_ns <= upper, f"duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns"
