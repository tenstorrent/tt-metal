# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_ATTN_KV_SPLIT=user (tt/kv_split.py, KV-pair user split of the verify / drafter SDPA).

* the flag (default off, "seq" rejected) and the layout / width gates;
* ownership: interleaved users, both pair members own the same count at every even width, bucket b's own users are the
  first b/2 of the B-row list (so one per-device page table serves every bucket by a static slice);
* the selection / scatter / position constants against direct indexing, and Sel @ x / Scat @ y in fp32;
* the pair weights: prepare_attn_qkvg_pair's own-head columns equal prepare_attn_qkv_deint's (so the own q / gate, k,
  v of the split are the unsplit columns), its partner columns are the partner device's, both pair members are equal;
  prepare_attn_wo_pair = the pair's rows of wo;
* the end-to-end algebra: summing every device's split o-proj partial (own rows, pair heads, zeros elsewhere) equals the
  unsplit sum over devices of (own 3 heads, all rows) -- the all-reduce sees the same total;
* flag-off identity: with QWEN36_ATTN_KV_SPLIT unset nothing new is reachable (no plan, no context, no pair weights).
No device. Run: pytest models/demos/blackhole/qwen36/tests/test_kv_split_host.py -q
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tt import kv_split as ks
from models.demos.blackhole.qwen36.tt import tp_common as tpc


def test_flag(monkeypatch):
    monkeypatch.delenv("QWEN36_ATTN_KV_SPLIT", raising=False)
    assert ks.kv_split_mode() is None
    for v in ("0", "off", ""):
        monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT", v)
        assert ks.kv_split_mode() is None
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT", "user")
    assert ks.kv_split_mode() == "user"
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT", "seq")
    with pytest.raises(ValueError):
        ks.kv_split_mode()
    monkeypatch.delenv("QWEN36_ATTN_KV_SPLIT_MIN_CTX", raising=False)
    monkeypatch.delenv("QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT", raising=False)
    assert ks.KVSplitGate.from_env().tables == ks.GATE_DEFAULT  # the context gate (tests/test_kv_split_gate_host.py)
    monkeypatch.delenv("QWEN36_ATTN_KV_SPLIT_PARTS", raising=False)
    assert ks.kv_split_parts() == {"verify", "draft"}
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_PARTS", "verify")
    assert ks.kv_split_parts() == {"verify"}
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_PARTS", "draft, verify")
    assert ks.kv_split_parts() == {"verify", "draft"}
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_PARTS", "prefill")
    with pytest.raises(ValueError):
        ks.kv_split_parts()


def test_gates():
    assert ks.kv_split_layout_ok(8, 4, 1)
    assert not ks.kv_split_layout_ok(4, 4, 1)  # TP4: whole KV heads, no pairs
    assert not ks.kv_split_layout_ok(8, 8, 1)
    assert [n for n in range(1, 9) if ks.split_n_users_ok(n)] == [2, 4, 6, 8]


@pytest.mark.parametrize("B", [2, 4, 6, 8])
def test_ownership(B):
    for r in (0, 1):
        assert ks.own_users(B, r) == list(range(r, B, 2))
    assert len(ks.own_users(B, 0)) == len(ks.own_users(B, 1)) == B // 2
    assert sorted(ks.own_users(B, 0) + ks.own_users(B, 1)) == list(range(B))
    for b in range(2, B + 1, 2):  # a row bucket's own users are a prefix of the B-row list
        for r in (0, 1):
            assert ks.own_users(b, r) == ks.own_users(B, r)[: b // 2]


@pytest.mark.parametrize("n,T", [(8, 4), (4, 4), (2, 4), (8, 1), (6, 1), (2, 8)])
def test_constants(n, T):
    tp, nh = 8, 6
    R, Ro = n * T, (n // 2) * T
    sel = ks.sel_host(n, T, tp, nh).float()
    sc = ks.scatter_host(n, T, tp).float()
    pi = ks.pos_index_host(n, T, tp)
    assert sel.shape == (tp, nh, Ro, R) and sc.shape == (tp, 1, R, Ro) and pi.shape == (tp, Ro)
    x = torch.randn(nh, R, 16)
    pos = torch.arange(R, dtype=torch.int32) * 3 + 11
    for d in range(tp):
        rows = ks.own_rows(n, T, d % 2)
        assert pi[d].tolist() == rows
        assert torch.equal(sel[d] @ x, x[:, rows, :])  # one-hot: exact in fp32
        y = torch.randn(Ro, 16)
        want = torch.zeros(R, 16)
        want[rows] = y
        assert torch.equal(sc[d, 0] @ y, want)
        assert torch.equal(ks.own_positions_host(pos, n, T, d % 2), pos[rows])
        # every row is owned by exactly one member of the pair
        other = ks.own_rows(n, T, 1 - d % 2)
        assert sorted(rows + other) == list(range(R))


def test_own_page_table():
    B, nb, tp = 8, 5, 8
    pt = torch.arange(B * nb, dtype=torch.int32).reshape(B, nb)
    own = ks.own_page_table_host(pt, tp)
    assert own.shape == (tp * B // 2, nb)
    for d in range(tp):
        blk = own[d * B // 2 : (d + 1) * B // 2]
        assert torch.equal(blk, pt[ks.own_users(B, d % 2)])
        for b in (2, 4, 6):  # bucket b reads the first b/2 rows of the device's block
            assert torch.equal(blk[: b // 2], pt[:b][ks.own_users(b, d % 2)])
    with pytest.raises(AssertionError):
        ks.own_page_table_host(pt[:3], tp)


def _toy_weights(tp=8, nh=3, hd=8, nkv=4, dim=16):
    nq = tp * nh
    q_w = torch.randn(nq * hd * 2, dim)  # interleaved [q, gate] per head
    k_w = torch.randn(nkv * hd, dim)
    v_w = torch.randn(nkv * hd, dim)
    o_w = torch.randn(dim, nq * hd)
    k_rep = tpc.replicate_kv_weight(k_w, nkv, tp, hd)
    v_rep = tpc.replicate_kv_weight(v_w, nkv, tp, hd)
    return q_w, k_rep, v_rep, o_w


def test_pair_weights():
    tp, nh, hd, dim = 8, 3, 8, 16
    q_w, k_rep, v_rep, o_w = _toy_weights(tp, nh, hd, 4, dim)
    kv = hd
    deint = tpc.prepare_attn_qkv_deint(q_w, k_rep, v_rep, nh, hd, kv, tp).reshape(tp, -1, dim)
    pair = tpc.prepare_attn_qkvg_pair(q_w, k_rep, v_rep, nh, hd, kv, tp).reshape(tp, -1, dim)
    W = nh * hd
    for d in range(tp):
        p, r = d // 2, d % 2
        dq, dk, dv, dg = deint[d][:W], deint[d][W : W + kv], deint[d][W + kv : W + 2 * kv], deint[d][W + 2 * kv :]
        pq, pk, pv, pg = (
            pair[d][: 2 * W],
            pair[d][2 * W : 2 * W + kv],
            pair[d][2 * W + kv : 2 * W + 2 * kv],
            pair[d][2 * W + 2 * kv :],
        )
        assert pg.shape[0] == 2 * W
        # the own heads sit at pair-global slot r, the partner's at 1 - r: both equal the deint blocks
        assert torch.equal(pq[r * W : (r + 1) * W], dq) and torch.equal(pg[r * W : (r + 1) * W], dg)
        partner = deint[2 * p + 1 - r]
        assert torch.equal(pq[(1 - r) * W : (2 - r) * W], partner[:W])
        assert torch.equal(pg[(1 - r) * W : (2 - r) * W], partner[W + 2 * kv :])
        assert torch.equal(pk, dk) and torch.equal(pv, dv)
        if r:
            assert torch.equal(pair[d], pair[d - 1])  # both members of a pair hold the same pair weight
    # o-projection: shard_w(dim=0) transposes and gives device d rows [d*2W, (d+1)*2W)
    wo_pair = tpc.prepare_attn_wo_pair(o_w, nh, hd, tp).T.reshape(tp, 2 * W, dim)
    wo = o_w.T.reshape(tp, W, dim)  # today's per-device rows
    for d in range(tp):
        p = d // 2
        assert torch.equal(wo_pair[d], torch.cat([wo[2 * p], wo[2 * p + 1]], dim=0))


@pytest.mark.parametrize("n,T", [(8, 4), (4, 4), (8, 1)])
def test_allreduce_total(n, T):
    """sum_d split partial_d == sum_d unsplit partial_d (the all-reduce sees the same total)."""
    tp, nh, hd, dim = 8, 3, 8, 16
    R = n * T
    W = nh * hd
    attn = torch.randn(R, tp * nh, hd)  # every row's attention output for all 24 heads (whatever the SDPA gives)
    gate = torch.randn(R, tp * nh * hd)
    o_w = torch.randn(dim, tp * W)
    gated = (attn.reshape(R, -1) * torch.sigmoid(gate)).double()
    unsplit = sum(gated[:, d * W : (d + 1) * W] @ o_w.T.double()[d * W : (d + 1) * W] for d in range(tp))
    wo_pair = tpc.prepare_attn_wo_pair(o_w, nh, hd, tp).T.double().reshape(tp, 2 * W, dim)
    split = torch.zeros(R, dim, dtype=torch.float64)
    for d in range(tp):
        p = d // 2
        rows = ks.own_rows(n, T, d % 2)
        mask = torch.zeros(R, 1, dtype=torch.float64)
        mask[rows] = 1.0
        split += (gated[:, 2 * p * W : (2 * p + 2) * W] * mask) @ wo_pair[d]
    assert torch.allclose(split, unsplit, rtol=1e-10, atol=1e-9)


def test_reference_model():
    """split_attention_reference (one device's partial) summed over a pair == the pair's heads over all rows."""
    n, T, NH2, HD, dim, S = 4, 4, 6, 8, 12, 40
    R = n * T
    q = torch.randn(R, NH2, HD)
    kc, vc = torch.randn(n, S, HD), torch.randn(n, S, HD)
    pos = torch.tensor([20 + u + j for u in range(n) for j in range(T)])
    gate = torch.randn(R, NH2 * HD)
    wo = torch.randn(NH2 * HD, dim)
    both = ks.split_attention_reference(q, kc, vc, pos, wo, gate, n, T, 0, HD**-0.5) + ks.split_attention_reference(
        q, kc, vc, pos, wo, gate, n, T, 1, HD**-0.5
    )
    full = torch.zeros(R, NH2 * HD)
    for r in range(R):
        u, p = r // T, int(pos[r])
        s = (q[r] @ kc[u, : p + 1].T) * HD**-0.5
        full[r] = (torch.softmax(s, -1) @ vc[u, : p + 1]).reshape(-1)
    assert torch.allclose(both, (full * torch.sigmoid(gate)) @ wo, atol=1e-5)


def test_flag_off_identity(monkeypatch):
    """Flag off: the loader gate is False, so no pair weight, plan or context exists and every dispatch check that
    guards the new paths (``_kv_split_ctx is not None``) stays False."""
    import types

    monkeypatch.delenv("QWEN36_ATTN_KV_SPLIT", raising=False)
    args = types.SimpleNamespace(num_devices=8, n_kv_heads=4, n_local_kv_heads=1)
    assert not ks.kv_split_supported(args)
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT", "user")
    assert ks.kv_split_supported(args)
    args.num_devices = 4
    assert not ks.kv_split_supported(args)
