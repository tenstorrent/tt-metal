# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.gdn_decode_step with state_page_idx: the recurrent state and the packed conv history live in
page pools (for example vLLM-owned cache blocks) and a [2*Nv, B] index names, per batch row and value head, the first
state page and the first history page. Optional second pools (state_pool2, conv_hist_pool2) extend the page space, so
the heads of one request can be split over two buffers (the K and V cache of a layer). Single device, TP=8 per-device
shapes (Nv=6, Nk=2, Dk=Dv=128).

  pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_paged.py -s
"""
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_gdn_decode_step import KD, VD, Dk, Dv, Nk, Nv, _pack_rows_user

C = 2 * KD + VD
W = C + VD + 32
AZ = C + VD
KV = (Dk // 32) * (Dv // 32)  # fp32 state pages per value head
HP = 4  # bf16 history pages per value head (one per slot)
BMAX = 32
MODES = [(False, False), (True, False), (True, True)]  # (fast_mode, fp32_state); the model runs (True, True)
MODE_IDS = ["default", "fast", "fast_fp32state"]
BLOCK = 64  # fp32 pages of one cache block's K (or V) buffer in layout option 3 at block 1024
K_HEADS = 4  # two_pools: heads 0..3 in the K pool, heads 4..5 and the whole history in the V pool
TRACE_PARAMS = [{"trace_region_size": 16_000_000}]


def _tiles(x):
    """[..., R, Cc] -> [n, 32, 32]: the 32x32 tiles in device page order."""
    *_, R, Cc = x.shape
    return x.reshape(-1, R // 32, 32, Cc // 32, 32).permute(0, 1, 3, 2, 4).reshape(-1, 32, 32)


def _to_pooled(t, b):
    """Packed history tiles [..., 32, 32] of batch row b (chunk c in row 2c + (b & 1)) -> the pooled layout (chunk c
    in rows 2c and 2c + 1)."""
    p = b % 2
    out = t.clone()
    out[..., 1 - p :: 2, :] = t[..., p::2, :]
    return out


def _from_pooled(t, b):
    """The pooled layout (both parities) -> the default layout of batch row b (row 2c + (b & 1), the other rows 0)."""
    p = b % 2
    out = torch.zeros_like(t)
    out[..., p::2, :] = t[..., p::2, :]
    return out


def _inputs(n_req, seed):
    """Per request: state [Nv, Dk, Dv] and pooled history [Nv, 4, 32, 32]; shared constants."""
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen)
    return {
        "h": (0.05 * rn(n_req, Nv, Dk, Dv)).float(),
        "hist2": torch.stack(
            [_pack_rows_user([(0.5 * rn(C)).bfloat16() for _ in range(4)], 0, both=True) for _ in range(n_req)]
        ),
        "w": (1.0 + 0.1 * rn(Dv)).bfloat16(),
        "taps": _pack_rows_user([(0.3 * rn(C)).bfloat16() for _ in range(4)], 0, both=True),
        "dtb": (0.1 * rn(Nv)).bfloat16(),
        "nea": (-torch.exp(0.2 * rn(Nv))).bfloat16(),
    }


def _token_rows(B, gen):
    rows = (0.5 * torch.randn(B, W, generator=gen)).bfloat16()
    rows[:, AZ + 2 * Nv :] = 0
    return rows


def _random_starts(n, run, pool_pages, gen):
    """n non-overlapping start pages of `run` pages each, in random order, with random gaps (any alignment)."""
    cuts = sorted(torch.randint(0, pool_pages - n * run + 1, (n,), generator=gen).tolist())
    order = torch.randperm(n, generator=gen).tolist()
    starts = [0] * n
    for i, j in enumerate(order):
        starts[j] = cuts[i] + i * run
    return starts


class Layout:
    """Where each request's heads live: st[r][h] / hs[r][h] = (pool, first page) of value head h's state (KV fp32 pages)
    and history (4 bf16 pages). pool 0 = `state` / `conv_hist`, pool 1 = `state_pool2` / `conv_hist_pool2`.
      rows_in_order: one pool, request r at the default offsets (the default layout expressed through the index).
      random: one pool, every (request, head) at its own random page of any alignment.
      two_pools: request r owns one BLOCK-page run in the K pool (heads 0..3) and one in the V pool (heads 4..5), and
        24 history pages in the V history pool; every run starts at a random page of any alignment (layout option 3).
        The K history pool exists but holds no request."""

    def __init__(self, n_req, kind, gen):
        self.kind = kind
        if kind == "rows_in_order":
            self.state_pages, self.hist_pages = [n_req * Nv * KV, 0], [n_req * Nv * HP, 0]
            self.st = [[(0, (r * Nv + h) * KV) for h in range(Nv)] for r in range(n_req)]
            self.hs = [[(0, (r * Nv + h) * HP) for h in range(Nv)] for r in range(n_req)]
        elif kind == "random":
            self.state_pages = [(n_req * Nv + 3) * KV + 13, 0]
            self.hist_pages = [(n_req * Nv + 3) * HP + 7, 0]
            ss = _random_starts(n_req * Nv, KV, self.state_pages[0], gen)
            hs = _random_starts(n_req * Nv, HP, self.hist_pages[0], gen)
            self.st = [[(0, ss[r * Nv + h]) for h in range(Nv)] for r in range(n_req)]
            self.hs = [[(0, hs[r * Nv + h]) for h in range(Nv)] for r in range(n_req)]
        else:
            assert kind == "two_pools"
            self.state_pages = [(n_req + 2) * BLOCK + 21, (n_req + 2) * BLOCK + 9]
            self.hist_pages = [3 * Nv * HP + 5, (n_req + 2) * Nv * HP + 11]
            k = _random_starts(n_req, BLOCK, self.state_pages[0], gen)
            v = _random_starts(n_req, BLOCK, self.state_pages[1], gen)
            c = _random_starts(n_req, Nv * HP, self.hist_pages[1], gen)
            self.st = [
                [(0, k[r] + h * KV) if h < K_HEADS else (1, v[r] + (h - K_HEADS) * KV) for h in range(Nv)]
                for r in range(n_req)
            ]
            self.hs = [[(1, c[r] + h * HP) for h in range(Nv)] for r in range(n_req)]

    def index_host(self, perm):
        """[2*Nv, B] int32 index for batch row b holding request perm[b] (pool-1 pages after pool 0's pages)."""
        g = lambda pool_page, sizes: pool_page[1] + (sizes[0] if pool_page[0] == 1 else 0)
        rows = [[g(self.st[r][h], self.state_pages) for r in perm] for h in range(Nv)]
        rows += [[g(self.hs[r][h], self.hist_pages) for r in perm] for h in range(Nv)]
        return torch.tensor(rows, dtype=torch.int32)

    def index(self, device, perm, dtype=ttnn.uint32):
        return ttnn.from_torch(self.index_host(perm), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def host_pools(self, inp, gen):
        """Host pools filled with garbage and every request's state / history placed: (state pools, hist pools)."""
        sp = [torch.randn(1, n, 32, 32, generator=gen).float() if n else None for n in self.state_pages]
        hp = [torch.randn(1, n, 32, 32, generator=gen).bfloat16() if n else None for n in self.hist_pages]
        for r in range(len(self.st)):
            for h in range(Nv):
                pool, page = self.st[r][h]
                sp[pool][0, page : page + KV] = _tiles(inp["h"][r, h])
                pool, page = self.hs[r][h]
                hp[pool][0, page : page + HP] = inp["hist2"][r, h]
        return sp, hp

    def check(self, pools0, pools, refs, requests):
        """The requests' pages equal refs[r] = (state [Nv, Dk, Dv], pooled history [Nv, 4, 32, 32]) bit for bit;
        every other page of every pool is unchanged."""
        (sp0, hp0), (sp, hp) = pools0, pools
        s_mask = [torch.zeros(n, dtype=torch.bool) for n in self.state_pages]
        h_mask = [torch.zeros(n, dtype=torch.bool) for n in self.hist_pages]
        for r in requests:
            h_ref, hist_ref = refs[r]
            for h in range(Nv):
                pool, page = self.st[r][h]
                assert torch.equal(sp[pool][0, page : page + KV], _tiles(h_ref[h])), f"request {r} head {h}: state"
                s_mask[pool][page : page + KV] = True
                pool, page = self.hs[r][h]
                assert torch.equal(hp[pool][0, page : page + HP], hist_ref[h]), f"request {r} head {h}: history"
                h_mask[pool][page : page + HP] = True
        for p in range(2):
            if self.state_pages[p]:
                assert torch.equal(sp[p][0, ~s_mask[p]], sp0[p][0, ~s_mask[p]]), f"state pool {p}: other pages changed"
            if self.hist_pages[p]:
                assert torch.equal(hp[p][0, ~h_mask[p]], hp0[p][0, ~h_mask[p]]), f"hist pool {p}: other pages changed"


def _tile_dev(device, t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _pools_dev(device, pools):
    sp, hp = pools
    to_dev = lambda ts, dtype: [_tile_dev(device, t, dtype) if t is not None else None for t in ts]
    return to_dev(sp, ttnn.float32), to_dev(hp, ttnn.bfloat16)


def _pools_host(pools_dev):
    to_host = lambda ts: [ttnn.to_torch(t) if t is not None else None for t in ts]
    return to_host(pools_dev[0]), to_host(pools_dev[1])


def _consts_dev(device, inp):
    return {k: _tile_dev(device, inp[k], ttnn.bfloat16) for k in ("dtb", "nea", "w", "taps")}


def _run(x, consts, state, hist, fast_mode, fp32_state, state_page_idx=None, state_pool2=None, conv_hist_pool2=None):
    return ttnn.experimental.kda.gdn_decode_step(
        x,
        consts["dtb"],
        consts["nea"],
        state,
        consts["w"],
        Nv,
        Nk,
        Dk,
        Dv,
        scale=Dk**-0.5,
        output_dtype=ttnn.bfloat16,
        conv_hist=hist,
        conv_taps=consts["taps"],
        qkvz_dim=AZ,
        fast_mode=fast_mode,
        fp32_state=fp32_state,
        state_page_idx=state_page_idx,
        state_pool2=state_pool2,
        conv_hist_pool2=conv_hist_pool2,
    )


def _run_pooled(x, consts, pools_dev, idx, fast_mode, fp32_state):
    (sa, sb), (ha, hb) = pools_dev
    return _run(x, consts, sa, ha, fast_mode, fp32_state, state_page_idx=idx, state_pool2=sb, conv_hist_pool2=hb)


def _default_step(device, x, consts, h_rows, hist2_rows, fast_mode, fp32_state):
    """Reference: the op without the index on [BMAX, ...] state / history, batch row b holding h_rows[b] and the
    pooled history hist2_rows[b] repacked for row b's parity. Returns (out, new states, new pooled histories)."""
    B = len(h_rows)
    h = torch.zeros(BMAX, Nv, Dk, Dv)
    hist = torch.zeros(BMAX, Nv, 4, 32, 32, dtype=torch.bfloat16)
    for b in range(B):
        h[b] = h_rows[b]
        hist[b] = _from_pooled(hist2_rows[b], b)
    state_dev, hist_dev = _tile_dev(device, h, ttnn.float32), _tile_dev(device, hist, ttnn.bfloat16)
    out = ttnn.to_torch(_run(x, consts, state_dev, hist_dev, fast_mode, fp32_state))
    h_t, hist_t = ttnn.to_torch(state_dev), ttnn.to_torch(hist_dev)
    return out, [h_t[b] for b in range(B)], [_to_pooled(hist_t[b], b) for b in range(B)]


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("B", [1, 8, 32])
@pytest.mark.parametrize("kind", ["rows_in_order", "random", "two_pools"])
def test_paged_matches_default(device, B, fast_mode, fp32_state, kind):
    """One step, request r on batch row r: output, state and history bit-identical to the default layout; every other
    pool page unchanged."""
    gen = torch.Generator().manual_seed(100 + B)
    inp = _inputs(B, seed=B)
    consts = _consts_dev(device, inp)
    x = _tile_dev(device, _token_rows(B, gen).reshape(1, B, W), ttnn.bfloat16)
    out_ref, h_new, hist_new = _default_step(
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
    inp = _inputs(B, seed=31)
    consts = _consts_dev(device, inp)
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
        x = _tile_dev(device, _token_rows(B, gen).reshape(1, B, W), ttnn.bfloat16)
        out_ref, h_new, hist_new = _default_step(
            device, x, consts, [ref[r][0] for r in perm], [ref[r][1] for r in perm], fast_mode, fp32_state
        )
        for b, r in enumerate(perm):
            ref[r] = (h_new[b], hist_new[b])
        out = ttnn.to_torch(_run_pooled(x, consts, pools_dev, lay.index(device, perm), fast_mode, fp32_state))
        assert torch.equal(out, out_ref), f"step {step}: output differs"
        lay.check(pools0, _pools_host(pools_dev), ref, range(B))


def test_paged_rejects_bad_index(device, expect_error):
    """The index must be [2*Nv, >= B]: a [2, B] index and a short one are refused on the host."""
    B = 8
    gen = torch.Generator().manual_seed(0)
    inp = _inputs(B, seed=0)
    consts = _consts_dev(device, inp)
    lay = Layout(B, "random", gen)
    (sa, _), (ha, _) = _pools_dev(device, lay.host_pools(inp, gen))
    x = _tile_dev(device, _token_rows(B, gen).reshape(1, B, W), ttnn.bfloat16)
    for shape in [(2, B), (2 * Nv, B - 1)]:
        idx = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        with expect_error(RuntimeError, "state_page_idx must be"):
            _run(x, consts, sa, ha, True, True, state_page_idx=idx)


@pytest.mark.parametrize("pool", ["state_pool2", "conv_hist_pool2"])
def test_paged_rejects_a_second_pool_of_smaller_tiles(device, expect_error, pool):
    """The kernels read a second pool with the first pool's page size (32x32 tiles): a second pool of 16x32 tiles, with
    the right dtype and enough pages, is refused on the host."""
    B = 8
    gen = torch.Generator().manual_seed(0)
    inp = _inputs(B, seed=0)
    consts = _consts_dev(device, inp)
    lay = Layout(B, "two_pools", gen)
    (sp, hp) = lay.host_pools(inp, gen)
    (sa, sb), (ha, hb) = _pools_dev(device, (sp, hp))
    small = lambda t, dtype: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((16, 32)), device=device
    )
    if pool == "state_pool2":
        sb = small(sp[1], ttnn.float32)
    else:
        hb = small(hp[1], ttnn.bfloat16)
    x = _tile_dev(device, _token_rows(B, gen).reshape(1, B, W), ttnn.bfloat16)
    idx = lay.index(device, list(range(B)))
    with expect_error(RuntimeError, f"{pool} must use 32x32 tiles"):
        _run(x, consts, sa, ha, True, True, state_page_idx=idx, state_pool2=sb, conv_hist_pool2=hb)


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("B", [8, 32])
def test_paged_trace_follows_index(device, B):
    """two_pools layout with 2B requests; request B + i starts from the same data as request i. Capture one trace with
    requests 0..B-1 on rows 0..B-1 and replay; rewrite only the index's contents to requests B..2B-1 on rows shifted by
    one (every request changes parity) and replay. Each replay updates exactly the indexed pages, bit-identical to the
    default layout from the same starting state."""
    fast_mode, fp32_state = True, True
    gen = torch.Generator().manual_seed(7)
    inp = _inputs(B, seed=50 + B)
    consts = _consts_dev(device, inp)
    inp2 = {**inp, "h": torch.cat([inp["h"], inp["h"]]), "hist2": torch.cat([inp["hist2"], inp["hist2"]])}
    lay = Layout(2 * B, "two_pools", gen)
    pools0 = lay.host_pools(inp2, gen)
    pools_dev = _pools_dev(device, pools0)
    x = _tile_dev(device, _token_rows(B, gen).reshape(1, B, W), ttnn.bfloat16)
    shift = [(b + 1) % B for b in range(B)]  # replay 1: row b holds a copy of request shift[b]
    perms = [list(range(B)), [B + i for i in shift]]
    refs = [
        _default_step(device, x, consts, list(inp["h"]), list(inp["hist2"]), fast_mode, fp32_state),
        _default_step(
            device,
            x,
            consts,
            [inp["h"][i] for i in shift],
            [inp["hist2"][i] for i in shift],
            fast_mode,
            fp32_state,
        ),
    ]
    idx = lay.index(device, perms[0])

    # compile outside the trace, then put the pools back
    _run_pooled(x, consts, pools_dev, idx, fast_mode, fp32_state)
    for t_dev, t_host in zip(pools_dev[0] + pools_dev[1], pools0[0] + pools0[1]):
        if t_dev is not None:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t_host, dtype=t_dev.dtype, layout=ttnn.TILE_LAYOUT), t_dev)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_dev = _run_pooled(x, consts, pools_dev, idx, fast_mode, fp32_state)
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


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("B", [8, 32])
def test_paged_traced_timing(device, B):
    """Traced time per call, default layout vs state_page_idx (two_pools layout), model mode (fast_mode, fp32_state).
    One trace holds N back-to-back calls; the host times REPS replays and reports the median per call."""
    N, REPS = 64, 15
    fast_mode, fp32_state = True, True
    gen = torch.Generator().manual_seed(3)
    inp = _inputs(B, seed=B)
    consts = _consts_dev(device, inp)
    x = _tile_dev(device, _token_rows(B, gen).reshape(1, B, W), ttnn.bfloat16)
    h = torch.zeros(BMAX, Nv, Dk, Dv)
    hist = torch.zeros(BMAX, Nv, 4, 32, 32, dtype=torch.bfloat16)
    for b in range(B):
        h[b], hist[b] = inp["h"][b], _from_pooled(inp["hist2"][b], b)
    default = (_tile_dev(device, h, ttnn.float32), _tile_dev(device, hist, ttnn.bfloat16))
    lay = Layout(B, "two_pools", gen)
    pools_dev = _pools_dev(device, lay.host_pools(inp, gen))
    idx = lay.index(device, list(range(B)))

    def per_call_us(call):
        call()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(N):
            ttnn.deallocate(call())
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            times.append((time.perf_counter() - t0) / N * 1e6)
        ttnn.release_trace(device, tid)
        return sorted(times)[REPS // 2]

    calls = {
        "default": lambda: _run(x, consts, *default, fast_mode, fp32_state),
        "paged": lambda: _run_pooled(x, consts, pools_dev, idx, fast_mode, fp32_state),
    }
    res = {k: [] for k in calls}
    for _ in range(2):  # alternate so drift hits both
        for k, call in calls.items():
            res[k].append(per_call_us(call))
    d, p = min(res["default"]), min(res["paged"])
    print(f"TRACED B={B}: default {d:.2f} us, state_page_idx two pools {p:.2f} us, ratio {p / d:.4f}")
