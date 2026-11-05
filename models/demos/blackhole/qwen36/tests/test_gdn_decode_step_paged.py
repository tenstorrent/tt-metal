# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.gdn_decode_step with state_page_idx: the recurrent state and the packed conv history live in
page pools (for example vLLM-owned cache blocks) and a [2, B] index names each batch row's first page. Single device,
TP=8 per-device shapes (Nv=6, Nk=2, Dk=Dv=128).

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
KV = (Dk // 32) * (Dv // 32)  # fp32 state tiles per value head
STATE_PAGES = Nv * KV  # fp32 state pages per batch row
HIST_PAGES = Nv * 4  # bf16 conv-history pages per batch row
BMAX = 32
MODES = [(False, False), (True, False), (True, True)]  # (fast_mode, fp32_state); the model runs (True, True)
TRACE_PARAMS = [{"trace_region_size": 16_000_000}]


def _tiles(x):
    """[..., R, Cc] -> [n, 32, 32]: the 32x32 tiles in device page order."""
    *_, R, Cc = x.shape
    return x.reshape(-1, R // 32, 32, Cc // 32, 32).permute(0, 1, 3, 2, 4).reshape(-1, 32, 32)


def _inputs(B, seed):
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen)
    rows = (0.5 * rn(B, W)).bfloat16()
    rows[:, AZ + 2 * Nv :] = 0
    cs = [[(0.5 * rn(C)).bfloat16() for _ in range(4)] for _ in range(B)]  # per row: history rows cs0..cs3
    return {
        "h": (0.05 * rn(B, Nv, Dk, Dv)).float(),
        # default layout: row b's history packed with parity b & 1; pooled layout (state_page_idx): both parities
        "hist": torch.stack([_pack_rows_user(cs[b], b) for b in range(B)]),
        "hist2": torch.stack([_pack_rows_user(cs[b], 0, both=True) for b in range(B)]),
        "rows": rows,
        "w": (1.0 + 0.1 * rn(Dv)).bfloat16(),
        "taps": _pack_rows_user([(0.3 * rn(C)).bfloat16() for _ in range(4)], 0, both=True),
        "dtb": (0.1 * rn(Nv)).bfloat16(),
        "nea": (-torch.exp(0.2 * rn(Nv))).bfloat16(),
    }


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


def _random_starts(n, run, pool_pages, gen):
    """n non-overlapping start pages of `run` pages each, in random order, with random gaps (any alignment)."""
    slack = pool_pages - n * run
    cuts = sorted(torch.randint(0, slack + 1, (n,), generator=gen).tolist())
    order = torch.randperm(n, generator=gen).tolist()
    starts = [0] * n
    for i, b in enumerate(order):
        starts[b] = cuts[i] + i * run
    return starts


def _tile_dev(device, t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _idx_dev(device, state_starts, hist_starts, dtype=ttnn.uint32):
    idx = torch.tensor([state_starts, hist_starts], dtype=torch.int32)
    return ttnn.from_torch(idx, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _pools(B, inp, state_starts, hist_starts, state_pool_pages, hist_pool_pages, gen):
    """Host pools [1, P, 32, 32] filled with garbage, row b's tiles placed at its start page."""
    sp = torch.randn(1, state_pool_pages, 32, 32, generator=gen).float()
    hp = torch.randn(1, hist_pool_pages, 32, 32, generator=gen).bfloat16()
    for b in range(B):
        sp[0, state_starts[b] : state_starts[b] + STATE_PAGES] = _tiles(inp["h"][b])
        hp[0, hist_starts[b] : hist_starts[b] + HIST_PAGES] = _tiles(inp["hist2"][b])
    return sp, hp


def _run(device, x, inp_dev, state, hist, fast_mode, fp32_state, state_page_idx=None):
    return ttnn.experimental.kda.gdn_decode_step(
        x,
        inp_dev["dtb"],
        inp_dev["nea"],
        state,
        inp_dev["w"],
        Nv,
        Nk,
        Dk,
        Dv,
        scale=Dk**-0.5,
        output_dtype=ttnn.bfloat16,
        conv_hist=hist,
        conv_taps=inp_dev["taps"],
        qkvz_dim=AZ,
        fast_mode=fast_mode,
        fp32_state=fp32_state,
        state_page_idx=state_page_idx,
    )


def _consts_dev(device, inp):
    return {
        "dtb": _tile_dev(device, inp["dtb"], ttnn.bfloat16),
        "nea": _tile_dev(device, inp["nea"], ttnn.bfloat16),
        "w": _tile_dev(device, inp["w"], ttnn.bfloat16),
        "taps": _tile_dev(device, inp["taps"], ttnn.bfloat16),
    }


def _contiguous_step(device, B, inp, inp_dev, fast_mode, fp32_state):
    """The op without state_page_idx on [BMAX, ...] state / history (rows >= B zero). Returns host (out, h, hist), the
    history converted to the pooled (both parities) layout."""
    h = torch.zeros(BMAX, Nv, Dk, Dv)
    h[:B] = inp["h"]
    hist = torch.zeros(BMAX, Nv, 4, 32, 32, dtype=torch.bfloat16)
    hist[:B] = inp["hist"]
    state_dev = _tile_dev(device, h, ttnn.float32)
    hist_dev = _tile_dev(device, hist, ttnn.bfloat16)
    x = _tile_dev(device, inp["rows"].reshape(1, B, W), ttnn.bfloat16)
    out = _run(device, x, inp_dev, state_dev, hist_dev, fast_mode, fp32_state)
    hist_t = ttnn.to_torch(hist_dev)
    return ttnn.to_torch(out), ttnn.to_torch(state_dev)[:B], torch.stack([_to_pooled(hist_t[b], b) for b in range(B)])


def _check_pools(B, sp0, hp0, sp, hp, state_starts, hist_starts, h_ref, hist_ref):
    """Row b's pages equal the reference (state, pooled history) bit for bit; every other pool page is unchanged."""
    s_mask = torch.zeros(sp.shape[1], dtype=torch.bool)
    h_mask = torch.zeros(hp.shape[1], dtype=torch.bool)
    for b in range(B):
        s0, h0 = state_starts[b], hist_starts[b]
        assert torch.equal(sp[0, s0 : s0 + STATE_PAGES], _tiles(h_ref[b])), f"row {b}: state differs"
        assert torch.equal(hp[0, h0 : h0 + HIST_PAGES], _tiles(hist_ref[b])), f"row {b}: conv history differs"
        s_mask[s0 : s0 + STATE_PAGES] = True
        h_mask[h0 : h0 + HIST_PAGES] = True
    assert torch.equal(sp[0, ~s_mask], sp0[0, ~s_mask]), "unaddressed state pool pages changed"
    assert torch.equal(hp[0, ~h_mask], hp0[0, ~h_mask]), "unaddressed conv-history pool pages changed"


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=["default", "fast", "fast_fp32state"])
@pytest.mark.parametrize("B", [1, 8, 32])
@pytest.mark.parametrize("contiguous_pages", [True, False], ids=["rows_in_order", "random_pages"])
def test_paged_matches_contiguous(device, B, fast_mode, fp32_state, contiguous_pages):
    """state_page_idx vs the default layout: output, state and conv history bit-identical; other pool pages unchanged.
    rows_in_order: row b at page b * per_row (the default layout, expressed through the index). random_pages: random
    non-overlapping start pages in random order, at no particular alignment."""
    gen = torch.Generator().manual_seed(100 + B)
    inp = _inputs(B, seed=B)
    inp_dev = _consts_dev(device, inp)
    out_ref, h_ref, hist_ref = _contiguous_step(device, B, inp, inp_dev, fast_mode, fp32_state)

    if contiguous_pages:
        state_pool_pages, hist_pool_pages = B * STATE_PAGES, B * HIST_PAGES
        state_starts = [b * STATE_PAGES for b in range(B)]
        hist_starts = [b * HIST_PAGES for b in range(B)]
    else:
        state_pool_pages, hist_pool_pages = (B + 3) * STATE_PAGES + 37, (B + 3) * HIST_PAGES + 11
        state_starts = _random_starts(B, STATE_PAGES, state_pool_pages, gen)
        hist_starts = _random_starts(B, HIST_PAGES, hist_pool_pages, gen)
    sp0, hp0 = _pools(B, inp, state_starts, hist_starts, state_pool_pages, hist_pool_pages, gen)
    sp_dev = _tile_dev(device, sp0, ttnn.float32)
    hp_dev = _tile_dev(device, hp0, ttnn.bfloat16)
    idx = _idx_dev(device, state_starts, hist_starts, dtype=ttnn.int32 if B == 8 else ttnn.uint32)
    x = _tile_dev(device, inp["rows"].reshape(1, B, W), ttnn.bfloat16)
    out = ttnn.to_torch(_run(device, x, inp_dev, sp_dev, hp_dev, fast_mode, fp32_state, state_page_idx=idx))

    assert torch.equal(out, out_ref), "output differs from the contiguous layout"
    _check_pools(B, sp0, hp0, ttnn.to_torch(sp_dev), ttnn.to_torch(hp_dev), state_starts, hist_starts, h_ref, hist_ref)


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=["default", "fast", "fast_fp32state"])
def test_paged_rows_change_parity(device, fast_mode, fp32_state):
    """Eight requests keep their pool pages while their batch rows change every step, so each request sits on even
    and odd rows in turn. Each step matches the default layout with the same request on the same row (its state and
    history repacked for that row's parity) bit for bit: output, state and history."""
    B, steps = 8, 4
    gen = torch.Generator().manual_seed(77)
    inp = _inputs(B, seed=31)
    inp_dev = _consts_dev(device, inp)
    perms = [
        list(range(B)),
        [1, 0, 3, 2, 5, 4, 7, 6],
        [7, 0, 1, 2, 3, 4, 5, 6],
        torch.randperm(B, generator=gen).tolist(),
    ]
    assert len(perms) == steps
    state_pool_pages, hist_pool_pages = (B + 2) * STATE_PAGES + 5, (B + 2) * HIST_PAGES + 3
    ss = _random_starts(B, STATE_PAGES, state_pool_pages, gen)
    hs = _random_starts(B, HIST_PAGES, hist_pool_pages, gen)
    sp0, hp0 = _pools(B, inp, ss, hs, state_pool_pages, hist_pool_pages, gen)  # request r starts in pages ss[r], hs[r]
    sp_dev = _tile_dev(device, sp0, ttnn.float32)
    hp_dev = _tile_dev(device, hp0, ttnn.bfloat16)
    ref_h = [inp["h"][r].clone() for r in range(B)]
    ref_hist2 = [inp["hist2"][r].clone() for r in range(B)]
    for step, perm in enumerate(perms):  # batch row b holds request perm[b]
        rows = (0.5 * torch.randn(B, W, generator=gen)).bfloat16()
        rows[:, AZ + 2 * Nv :] = 0
        x = _tile_dev(device, rows.reshape(1, B, W), ttnn.bfloat16)
        # reference: the default layout with request perm[b] in row b
        h = torch.zeros(BMAX, Nv, Dk, Dv)
        hist = torch.zeros(BMAX, Nv, 4, 32, 32, dtype=torch.bfloat16)
        for b, r in enumerate(perm):
            h[b] = ref_h[r]
            hist[b] = _from_pooled(ref_hist2[r], b)
        state_dev, hist_dev = _tile_dev(device, h, ttnn.float32), _tile_dev(device, hist, ttnn.bfloat16)
        out_ref = ttnn.to_torch(_run(device, x, inp_dev, state_dev, hist_dev, fast_mode, fp32_state))
        h_t, hist_t = ttnn.to_torch(state_dev), ttnn.to_torch(hist_dev)
        for b, r in enumerate(perm):
            ref_h[r] = h_t[b]
            ref_hist2[r] = _to_pooled(hist_t[b], b)
        # pooled: the same requests, addressed through state_page_idx
        idx = _idx_dev(device, [ss[r] for r in perm], [hs[r] for r in perm])
        out = ttnn.to_torch(_run(device, x, inp_dev, sp_dev, hp_dev, fast_mode, fp32_state, state_page_idx=idx))
        assert torch.equal(out, out_ref), f"step {step}: output differs"
        _check_pools(B, sp0, hp0, ttnn.to_torch(sp_dev), ttnn.to_torch(hp_dev), ss, hs, ref_h, ref_hist2)


def test_paged_rejects_short_index(device, expect_error):
    """An index with fewer than B columns is refused on the host."""
    B = 8
    inp = _inputs(B, seed=0)
    inp_dev = _consts_dev(device, inp)
    sp = _tile_dev(device, torch.zeros(1, B * STATE_PAGES, 32, 32), ttnn.float32)
    hp = _tile_dev(device, torch.zeros(1, B * HIST_PAGES, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    x = _tile_dev(device, inp["rows"].reshape(1, B, W), ttnn.bfloat16)
    idx = _idx_dev(device, list(range(B - 1)), list(range(B - 1)))
    with expect_error(RuntimeError, "state_page_idx must be"):
        _run(device, x, inp_dev, sp, hp, True, True, state_page_idx=idx)


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("B", [8, 32])
def test_paged_trace_follows_index(device, B):
    """Capture one trace with state_page_idx = page set A, replay; rewrite only the index tensor's contents to a
    disjoint page set B, replay again. Each replay updates exactly the indexed pages, bit-identical to the
    contiguous op on the same starting state."""
    fast_mode, fp32_state = True, True
    gen = torch.Generator().manual_seed(7)
    inp = _inputs(B, seed=50 + B)
    inp_dev = _consts_dev(device, inp)
    state_pool_pages, hist_pool_pages = 2 * B * STATE_PAGES + 64, 2 * B * HIST_PAGES + 64
    s_all = _random_starts(2 * B, STATE_PAGES, state_pool_pages, gen)
    h_all = _random_starts(2 * B, HIST_PAGES, hist_pool_pages, gen)
    sets = [(s_all[:B], h_all[:B]), (s_all[B:], h_all[B:])]
    # both page sets start from the same per-row state
    sp0, hp0 = _pools(B, inp, *sets[0], state_pool_pages, hist_pool_pages, gen)
    for b in range(B):
        sp0[0, sets[1][0][b] : sets[1][0][b] + STATE_PAGES] = _tiles(inp["h"][b])
        hp0[0, sets[1][1][b] : sets[1][1][b] + HIST_PAGES] = _tiles(inp["hist2"][b])
    sp_dev = _tile_dev(device, sp0, ttnn.float32)
    hp_dev = _tile_dev(device, hp0, ttnn.bfloat16)
    x = _tile_dev(device, inp["rows"].reshape(1, B, W), ttnn.bfloat16)
    idx = _idx_dev(device, *sets[0])

    # compile outside the trace, then put the pools back to their initial contents
    _run(device, x, inp_dev, sp_dev, hp_dev, fast_mode, fp32_state, state_page_idx=idx)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(sp0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), sp_dev)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(hp0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), hp_dev)
    out_ref, h_ref, hist_ref = _contiguous_step(device, B, inp, inp_dev, fast_mode, fp32_state)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_dev = _run(device, x, inp_dev, sp_dev, hp_dev, fast_mode, fp32_state, state_page_idx=idx)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        before = (ttnn.to_torch(sp_dev), ttnn.to_torch(hp_dev))
        for k, (state_starts, hist_starts) in enumerate(sets):
            if k > 0:
                new_idx = torch.tensor([state_starts, hist_starts], dtype=torch.int32)
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(new_idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), idx
                )
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            after = (ttnn.to_torch(sp_dev), ttnn.to_torch(hp_dev))
            assert torch.equal(ttnn.to_torch(out_dev), out_ref), f"replay {k}: output differs"
            _check_pools(B, *before, *after, state_starts, hist_starts, h_ref, hist_ref)
            before = after
    finally:
        ttnn.release_trace(device, tid)


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("B", [8, 32])
def test_paged_traced_timing(device, B):
    """Traced time per call, default layout vs state_page_idx (random pages), model mode (fast_mode, fp32_state).
    One trace holds N back-to-back calls; the host times REPS replays and reports the median per call."""
    N, REPS = 64, 15
    fast_mode, fp32_state = True, True
    gen = torch.Generator().manual_seed(3)
    inp = _inputs(B, seed=B)
    inp_dev = _consts_dev(device, inp)
    x = _tile_dev(device, inp["rows"].reshape(1, B, W), ttnn.bfloat16)

    h = torch.zeros(BMAX, Nv, Dk, Dv)
    h[:B] = inp["h"]
    hist = torch.zeros(BMAX, Nv, 4, 32, 32, dtype=torch.bfloat16)
    hist[:B] = inp["hist"]
    contiguous = (_tile_dev(device, h, ttnn.float32), _tile_dev(device, hist, ttnn.bfloat16), None)
    state_pool_pages, hist_pool_pages = (BMAX + 3) * STATE_PAGES + 37, (BMAX + 3) * HIST_PAGES + 11
    state_starts = _random_starts(B, STATE_PAGES, state_pool_pages, gen)
    hist_starts = _random_starts(B, HIST_PAGES, hist_pool_pages, gen)
    sp0, hp0 = _pools(B, inp, state_starts, hist_starts, state_pool_pages, hist_pool_pages, gen)
    paged = (
        _tile_dev(device, sp0, ttnn.float32),
        _tile_dev(device, hp0, ttnn.bfloat16),
        _idx_dev(device, state_starts, hist_starts),
    )

    def per_call_us(state, hist_t, idx):
        _run(device, x, inp_dev, state, hist_t, fast_mode, fp32_state, state_page_idx=idx)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(N):
            out = _run(device, x, inp_dev, state, hist_t, fast_mode, fp32_state, state_page_idx=idx)
            ttnn.deallocate(out)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            times.append((time.perf_counter() - t0) / N * 1e6)
        ttnn.release_trace(device, tid)
        return sorted(times)[REPS // 2], min(times)

    # alternate the two layouts so drift (clocks, temperature) hits both
    results = {"contiguous": [], "paged": []}
    for _ in range(2):
        for name, args in (("contiguous", contiguous), ("paged", paged)):
            results[name].append(per_call_us(*args))
    med = {k: min(m for m, _ in v) for k, v in results.items()}
    best = {k: min(b for _, b in v) for k, v in results.items()}
    print(
        f"TRACED B={B}: contiguous median {med['contiguous']:.2f} us (min {best['contiguous']:.2f}), "
        f"paged median {med['paged']:.2f} us (min {best['paged']:.2f}), "
        f"paged/contiguous {med['paged'] / med['contiguous']:.4f}"
    )
