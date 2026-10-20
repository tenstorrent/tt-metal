# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tt/spec_fused.py: the fused greedy accept + drafter-input reference equals the existing host path
(plugin greedy walk -> propose_draft_tokens -> accepted_plan / _dr_pack_host / _dr_rs_table), element for element.
No device. Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_fused_host.py"""

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_hostfix_host import _propose_engine
from models.demos.blackhole.qwen36.tests.test_spec_hostset_host import _drafter_engine
from models.demos.blackhole.qwen36.tt.spec_engine import (
    MTPSpecEngine,
    accepted_plan,
    draft_rows_fit,
    sanitize_verify_block,
    table_capacity,
)
from models.demos.blackhole.qwen36.tt.spec_fused import fused_accept_aux_host, fused_fit_mode, fused_greedy_reference

try:
    from vllm_tt_plugin.spec_accept import _commit_walks, _greedy_accept
except ImportError:  # pragma: no cover
    pytest.skip("vllm_tt_plugin not importable", allow_module_level=True)

K, T, BS, SCRATCH = 3, 4, 64, 5000


def _case(g, B, nb, edge):
    R = B * T
    cap = nb * BS
    pad = [bool(torch.rand(1, generator=g) < 0.25) for _ in range(B)]
    skip = [bool(torch.rand(1, generator=g) < 0.2) and not p for p in pad]
    nv = torch.randint(0, K + 1, (B,), generator=g)
    nv[torch.tensor(pad)] = 0
    start0 = torch.randint(1, 60, (B,), generator=g)
    if edge:  # near the capacity edge: fits_all / fits_none / mixed all occur
        start0 = cap - K - 1 - K - 1 + torch.randint(0, 2 * K + 4, (B,), generator=g)
    start_pos = start0[:, None] + torch.arange(T)[None, :]
    start_pos[torch.tensor(pad)] = -1
    ids = torch.randint(1, 900, (B, T), generator=g)
    drafts = torch.randint(1, 900, (B, K), generator=g)
    tokens = torch.cat([torch.randint(1, 900, (B, 1), generator=g), drafts], 1)
    for u in range(B):  # forced correct-draft prefix so m covers 0..K
        mm = int(torch.randint(0, K + 1, (1,), generator=g))
        tokens[u, 1 : 1 + mm] = ids[u, :mm]
        if mm < K:
            tokens[u, 1 + mm] = (ids[u, mm] + 1) % 900 + 1
    pt = torch.randint(1, 4000, (B, nb), generator=g, dtype=torch.int32)
    return pad, skip, nv, start_pos, tokens, ids, pt


def _host_path(e_prop, e_dr, pad, nv, start_pos, tokens, ids, pt, nb):
    """The existing host composition. Returns dict or None when propose does not run its drafter."""
    B = len(pad)
    clean, start0, active = sanitize_verify_block(tokens, start_pos, nv, K)
    # plugin walk (spec_accept.py:643 / :1420 via _commit_walks)
    dtok = tokens[:, 1:].clone()
    dtok[torch.arange(K)[None, :] >= nv[:, None]] = -1
    accepted, cand = _greedy_accept(ids.to(torch.int32), dtok, K)
    res = _commit_walks(torch.ones(B, dtype=torch.bool), (accepted, cand), None, nv.to(torch.int32), B, T)
    committed, counts = res.committed_token_ids, res.accepted_counts
    cpos = start0[:, None] + 1 + torch.arange(T)[None, :]
    cpos[torch.tensor(pad)] = -1  # model_runner._committed_positions: padding rows -1
    e_prop.calls.clear()
    out = e_prop.propose_draft_tokens(K, committed, cpos, counts, page_table=pt)
    if not e_prop.calls:
        return None
    ctok, cp, acc, decline, keep = e_prop.calls[0]
    m, _, pending, p = accepted_plan(ctok, cp, acc, B, T)
    m_kv = [mu + int(d and k) for mu, d, k in zip(m, decline, keep)]
    pending = [0 if d else t for d, t in zip(decline, pending)]
    p = [0 if d else x for d, x in zip(decline, p)]
    r = dict(committed=committed, counts=counts, decline=decline, num_valid=out.num_valid, clean=clean)
    dpt = pt.clone()
    dpt[torch.tensor(decline)] = SCRATCH
    r["mtp_pt"] = dpt
    if not all(decline):
        pack, _, _, real = MTPSpecEngine._dr_pack_host(e_dr, ctok, cp, m, pending, p, m_kv)
        MTPSpecEngine._dr_rs_table(e_dr, real, pt)
        r["pack"], r["rs_pt"] = pack, e_dr._dr_rs_cache[2].clone()
    return r


@pytest.mark.parametrize("B", [1, 2, 4, 8])
@pytest.mark.parametrize("delta,S", [(0, 4096), (5, 150)])
def test_fused_equals_host_path(B, delta, S):
    nb = 3
    g = torch.Generator().manual_seed(B * 7 + delta)
    n_fused = n_ms = 0
    seen_m = set()
    for ci in range(6):  # a few cases share one engine (stateful _dr_rs_cache sequence)
        e_dr = _drafter_engine(B, K, delta, S, nb=nb)
        for i in range(40):
            edge = i % 3 == 2
            pad, skip, nv, start_pos, tokens, ids, pt = _case(g, B, nb, edge)
            e_prop = _propose_engine(B, K, pad)
            e_prop._no_draft = {u for u in range(B) if skip[u]}
            e_prop._last_cap_blocks = nb
            ref = _host_path(e_prop, e_dr, pad, nv, start_pos, tokens, ids, pt, nb)
            if ref is None:  # B == 1 skip_all: propose returns early
                continue
            clean, start0, active = sanitize_verify_block(tokens, start_pos, nv, K)
            cap = table_capacity(nb, BS)
            try:
                aux = fused_accept_aux_host(nv, start0, active, skip, cap, K, SCRATCH, delta, S)
            except ValueError:
                n_ms += 1
                assert "mixed" in fused_fit_mode(start0, cap, active, K) and any(
                    md == "mixed" and not s for md, s in zip(fused_fit_mode(start0, cap, active, K), skip)
                )
                continue
            out = fused_greedy_reference(ids.reshape(-1).to(torch.int32), clean.reshape(-1), aux, pt, B=B, K=K)
            n_fused += 1
            assert torch.equal(out.committed, ref["committed"]) and torch.equal(out.counts, ref["counts"])
            assert out.committed.dtype == torch.int32 and out.counts.dtype == torch.int32
            seen_m.update((out.counts - 1).tolist())
            assert out.declined.tolist() == list(ref["decline"])
            assert torch.equal(out.num_valid_out, ref["num_valid"])
            assert torch.equal(out.mtp_pt, ref["mtp_pt"])
            assert out.drafter_runs == ("pack" in ref)
            if "pack" in ref:
                assert torch.equal(out.pack, ref["pack"]), (out.pack, ref["pack"])
                assert torch.equal(out.rs_pt, ref["rs_pt"])
    assert n_fused > 50 and seen_m == set(range(K + 1)), (n_fused, seen_m)
    assert n_ms > 0 or B == 1, n_ms


@pytest.mark.parametrize("K_", [1, 3, 7])
def test_fit_mode_matches_draft_rows_fit(K_):
    for cap in (192, 256):
        start0 = torch.arange(cap - 3 * K_ - 6, cap + 2)
        for active_all in (True, False):
            active = torch.full((len(start0),), active_all)
            modes = fused_fit_mode(start0, cap, active, K_)
            for s, a, md in zip(start0.tolist(), active.tolist(), modes):
                fits = [draft_rows_fit([s + m], K_, cap)[0] for m in range(K_ + 1)]
                want = "fits_none" if not a else "fits_all" if all(fits) else "fits_none" if not any(fits) else "mixed"
                assert md == want
            if active_all:
                assert {"fits_all", "fits_none", "mixed"} <= set(modes) or K_ == 0


def test_aux_layout_and_padding(expect_error):
    aux = fused_accept_aux_host(
        torch.tensor([3, 0]),
        torch.tensor([10, 0]),
        torch.tensor([True, False]),
        [False, False],
        192,
        3,
        SCRATCH,
        -7,
        4096,
    )
    assert aux.shape == (6, 32) and aux.dtype == torch.int32
    w = aux.to(torch.int64) & 0xFFFFFFFF
    assert w[0, :2].tolist() == [3, 0] and w[1, :2].tolist() == [1, 0] and w[2, :2].tolist() == [0, 1]
    assert w[3, :2].tolist() == [0, 0] and w[4, :2].tolist() == [10, 0]
    assert w[5, :5].tolist() == [SCRATCH, 2**32 - 7, 4096, 2, 3]
    assert int(w[5, 5:].sum()) == 0
    with expect_error(ValueError, "draft_rows_fit depends on m"):  # mixed row, not skip
        fused_accept_aux_host(
            torch.tensor([3]), torch.tensor([192 - 3 - 1 - 1]), torch.tensor([True]), [False], 192, 3, 0, 0, 10
        )
