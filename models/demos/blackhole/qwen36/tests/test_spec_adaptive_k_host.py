# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of QWEN36_SPEC_ADAPTIVE_K (tt/spec_adaptive.py and the state-width ctrl page it needs). No device.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_adaptive_k_host.py
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt import spec_adaptive as sa
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_page, spec_ctrl_words, spec_state_blk_idx
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_chain import chain_reference, pack_ovr_host, res_c_reference
from models.demos.blackhole.qwen36.tt.spec_engine import parked_rows_host, spec_memory_costs
from models.demos.blackhole.qwen36.tt.spec_fused import FUSED_AUX_ROWS
from models.demos.blackhole.qwen36.tt.tp_common import spec_adaptive_k_wide

NV = 6  # value heads per device at TP = 8


# ---------------------------------------------------------------------------------------------------------- flags
def test_wide_flag_parse(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_ADAPTIVE_K_WIDE", raising=False)
    assert spec_adaptive_k_wide(8) == (4, 5)
    monkeypatch.setenv("QWEN36_SPEC_ADAPTIVE_K_WIDE", "2:11")
    assert spec_adaptive_k_wide(8) == (2, 11)
    for bad in ("3:7", "4:8", "8:3", "x", "4:0"):
        monkeypatch.setenv("QWEN36_SPEC_ADAPTIVE_K_WIDE", bad)
        with pytest.raises(ValueError):
            spec_adaptive_k_wide(8)


# --------------------------------------------------------------------------------------------- block geometry
def test_block_to_k_cuts_columns_and_drafts():
    tok = torch.arange(8 * 8, dtype=torch.int32).reshape(8, 8)
    pos = tok + 100
    nv = torch.tensor([7, 3, 0, 2, 5, 1, 3, 4], dtype=torch.int32)
    t, p, n, cut = sa.ak_block_to_k(tok, pos, nv, 3)
    assert t.shape == (8, 4) and p.shape == (8, 4)
    assert torch.equal(t, tok[:, :4]) and torch.equal(p, pos[:, :4])
    assert n.tolist() == [3, 3, 0, 2, 3, 1, 3, 3] and n.dtype == torch.int32
    assert cut == [0, 4, 7]
    # already narrow: unchanged
    t2, p2, n2, cut2 = sa.ak_block_to_k(tok[:, :4], pos[:, :4], torch.clamp(nv, max=3), 3)
    assert torch.equal(t2, tok[:, :4]) and cut2 == []


def test_cols_rows_pad():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    assert sa.ak_cols(x, 4, -1).tolist() == [[1, 2, -1, -1], [3, 4, -1, -1]]
    assert sa.ak_cols(x, 1, -1).tolist() == [[1], [3]]
    assert sa.ak_rows(x, 3, 7).tolist() == [[1, 2], [3, 4], [7, 7]]
    lg = torch.ones(2, 2, 5)
    assert sa.ak_cols(lg, 3, 0).shape == (2, 3, 5) and float(sa.ak_cols(lg, 3, 0)[:, 2].abs().sum()) == 0.0


def test_uniforms_to_k_keeps_bonus_accepts_and_correction():
    u = torch.arange(2 * 9, dtype=torch.float32).reshape(2, 9) / 100  # K_api = 7: [bonus, a1..a7, corr]
    v = sa.ak_uniforms_to_k(u, 3, 7)
    assert v.shape == (2, 5)
    assert torch.equal(v[:, :4], u[:, :4]) and torch.equal(v[:, 4], u[:, 8])
    assert sa.ak_uniforms_to_k(u, 7, 7) is u


def test_compact_perm():
    assert sa.ak_compact_perm([0, 1, 2], 8) is None
    assert sa.ak_compact_perm([1, 5], 8) == [1, 5, 0, 2, 3, 4, 6, 7]
    assert sorted(sa.ak_compact_perm([7], 8)) == list(range(8))


def test_step_rows_and_phys_after():
    phys = [3, 0, 1, 2, 4, 5, 6, 7]
    pos = torch.full((8, 8), -1, dtype=torch.int64)
    pos[0, 0], pos[2, 0] = 10, 20
    assert sa.ak_step_rows(pos, phys) == [1, 3]
    assert sa.ak_phys_after(phys, None, None) == phys
    assert sa.ak_phys_after(phys, None, [1, 0, 2, 3, 4, 5, 6, 7]) == [0, 3, 1, 2, 4, 5, 6, 7]
    assert sa.ak_wide_fits([0, 3], 4) and not sa.ak_wide_fits([0, 4], 4) and not sa.ak_wide_fits([], 4)


# --------------------------------------------------------------------------------------------- family choice
def _state(ready=True):
    st = sa.AdaptiveKState(sa.KFamily("base", 8, 3), sa.KFamily("wide", 4, 7))
    st.ready = ready
    return st


def test_choose_hold_and_switches(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_ADAPTIVE_K_HOLD", "3")
    st = _state()
    base, wide = st.base, st.wide
    # not captured / too many rows / narrow: base
    st0 = _state(False)
    assert sa.ak_choose(st0, [0, 1], "fused_sample", False, False, 3) == (st0.base, False)
    assert sa.ak_choose(st, [0, 1, 2, 3, 4], "fused_sample", False, False, 3) == (base, False)
    assert sa.ak_choose(st, [0], "logits", True, False, 0) == (base, False)
    # a host-staged step that fits switches at once
    assert sa.ak_choose(st, [0, 2], "fused_sample", False, False, 3) == (wide, False)
    # a chained step from the base family waits HOLD steps
    st.last = base
    assert sa.ak_choose(st, [0, 2], "fused_sample", False, True, 3) == (base, False)
    assert sa.ak_choose(st, [0, 2], "fused_sample", False, True, 3) == (base, False)
    assert sa.ak_choose(st, [0, 2], "fused_sample", False, True, 3) == (wide, False)
    # staying wide: no hold
    st.last = wide
    assert sa.ak_choose(st, [0, 1, 2, 3], "fused_sample", False, True, 7) == (wide, False)
    # a row past the width: compaction (after the hold when chained), or base without compaction
    st.last = base
    st.streak = 0
    assert sa.ak_choose(st, [0, 5], "fused_sample", False, False, 3) == (base, False)  # hold: 1
    assert sa.ak_choose(st, [0, 5], "fused_sample", False, False, 3, compact_ok=False) == (base, False)
    # a non-fused step whose rows carry more drafts than the base K must run wide (compacted if needed)
    assert sa.ak_choose(st, [0, 6], "argmax_ids", False, False, 7) == (wide, True)
    assert sa.ak_choose(st, [1, 2], "logits", False, False, 5) == (wide, False)


# ------------------------------------------------------------------------------- ctrl page at the state width
def _mview(B, T, Bs=None, rope_rows=4096):
    mv = SimpleNamespace(
        _vfy_T=T,
        _vfy_B=B,
        _vfy_Nv=NV,
        _vfy_fused=True,
        _vfy_gdn=None,
        _slot_rope_delta=None,
        rope=SimpleNamespace(rope_delta=0, max_seq_len=rope_rows),
    )
    if Bs is not None:
        mv._vfy_state_B = Bs
    return mv


def test_state_width_ctrl_page_layout():
    """The wide family's page (4 users of an 8-wide state) is the 8-user page with mi 0 for users 4..7: words 1..4 the
    users' mi, words 9.. the ring blocks (mi*8 + u)*Nv + h (gdn_spec_step max_users = 8 reads word 1 + u and 1 + M + u*Nv+h).
    """
    mi = [7, 0, 3, 5]
    mv = _mview(4, 8, Bs=8)
    mis, Bs = Qwen36Model._vfy_state_mi(mv, mi)
    assert Bs == 8 and mis == mi + [0, 0, 0, 0]
    page = spec_ctrl_page(mis, 8, NV, parity=1).reshape(-1)
    assert int(page[0]) == 1 and page[1:5].tolist() == mi and page[5:9].tolist() == [0] * 4
    blk = page[9 : 9 + 4 * NV].reshape(4, NV)
    for u in range(4):
        assert blk[u].tolist() == [(mi[u] * 8 + u) * NV + h for h in range(NV)]
    assert torch.equal(page[9 : 9 + 8 * NV], spec_state_blk_idx(mis, 8, NV))
    # flag off (no _vfy_state_B): the B-user page, unchanged
    mis0, Bs0 = Qwen36Model._vfy_state_mi(_mview(8, 4), [1, 2, 3, 0, 0, 1, 2, 3])
    assert Bs0 == 8 and mis0 == [1, 2, 3, 0, 0, 1, 2, 3]


def test_aux_host_wide_vs_base_words():
    pos = [100, 200, 0, 50]
    mi = [6, 1, 0, 7]
    mv = _mview(4, 8, Bs=8)
    rp = Qwen36Model._vfy_row_positions(mv, pos)
    assert rp.tolist() == [p + j for p in pos for j in range(8)]
    aux = Qwen36Model._vfy_aux_host(mv, rp, mi, parity=0)
    assert aux.shape[0] == 3
    ctrl = aux[2]
    want = spec_ctrl_page(mi + [0] * 4, 8, NV, parity=0).reshape(-1)
    want[9 + 4 * NV : 9 + 8 * NV] = 0  # the users past the verify's: zero words (what the chain device writes)
    assert torch.equal(ctrl[: want.numel()].to(torch.int64), want.to(torch.int64))
    assert aux.shape[1] >= spec_ctrl_words(8, NV)


@pytest.mark.parametrize("seed", range(6))
def test_chain_reference_state_width(seed):
    """chain_reference with state_B lays the next ctrl page out at the state width and equals the host builder of the wide
    family's verify aux (Qwen36Model._vfy_aux_host on a 4-user / 8-wide model view)."""
    g = torch.Generator().manual_seed(seed)
    B, K, T, Bs = 4, 7, 8, 8
    counts = torch.randint(1, T + 1, (B,), generator=g)
    committed = torch.randint(0, 1000, (B, T), generator=g)
    committed[torch.arange(T)[None, :] >= counts[:, None]] = -1
    p0_prev = torch.randint(10, 500, (B,), generator=g)
    drafts = torch.randint(0, 1000, (B, K), generator=g)
    act = torch.ones(B, dtype=torch.long)
    act[int(torch.randint(0, B, (1,), generator=g))] = 0
    nv = torch.where(act.bool(), torch.randint(0, K + 1, (B,), generator=g), torch.zeros(B, dtype=torch.long))
    zeros = torch.zeros(B, dtype=torch.long)
    ovr = pack_ovr_host(
        nv, act, 1 - act, zeros, zeros, zeros, None, zeros, 1, K=K, scratch_block=9999, rope_delta=0, rope_rows=8192
    )
    res = res_c_reference(committed, counts, p0_prev, zeros)
    words = spec_ctrl_words(Bs, NV)
    tok, vaux, nxt = chain_reference(
        res, drafts, None, ovr, B=B, K=K, Nv=NV, rope_rows=8192, ctrl_words=words, state_B=Bs
    )
    m = counts - 1
    p0n = ((p0_prev + m + 1) * act).tolist()
    mi = (m * act).tolist()
    mv = _mview(B, T, Bs=Bs, rope_rows=8192)
    want = Qwen36Model._vfy_aux_host(mv, Qwen36Model._vfy_row_positions(mv, p0n), mi, parity=1)
    got = vaux.to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(got[:, : want.shape[1]], want.to(torch.int64) & 0xFFFFFFFF)
    assert nxt.shape == (FUSED_AUX_ROWS, nxt.shape[1])
    # state_B None is the old B-wide page (flag off)
    _, vaux0, _ = chain_reference(res, drafts, None, ovr, B=B, K=K, Nv=NV, rope_rows=8192, ctrl_words=words)
    page0 = vaux0[2].to(torch.int64)
    assert page0[1:5].tolist() == mi and page0[5 : 5 + B * NV].tolist() == spec_state_blk_idx(mi, B, NV).tolist()


# ---------------------------------------------------------------------------------------- park / memory geometry
def test_parked_rows_host_t_alloc():
    """A parked row saved from a T_alloc ring (every slot) restores the slot of its last commit and shifts its window, for a
    commit index of either family (mi up to T_alloc - 1)."""
    D, Ta, C = 2, 8, 5
    ring = torch.arange(D * Ta * NV * 2 * 2, dtype=torch.float32).reshape(D * Ta, NV, 2, 2)
    e_prev = torch.arange(D * (3 + Ta) * C, dtype=torch.float32).reshape(D, 3 + Ta, C)
    for s in (0, 3, 7):
        slot, e = parked_rows_host(ring, e_prev, s, Ta)
        assert torch.equal(slot, ring.reshape(D, Ta, NV, 2, 2)[:, s])
        assert torch.equal(e[:, : 3 + Ta - s], e_prev[:, s:])
        assert float(e[:, 3 + Ta - s :].abs().sum()) == 0.0


def test_memory_costs_ring_t():
    tc = dict(
        hidden_size=5120,
        vocab_size=248320,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_hidden_layers=64,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    c3 = spec_memory_costs(tc, 8, 8, 3, nb=64, buckets=())
    c8 = spec_memory_costs(tc, 8, 8, 3, nb=64, buckets=(), ring_T=8)
    assert c8["per_seq"]["gdn_ring"] == 2 * c3["per_seq"]["gdn_ring"]
    assert spec_memory_costs(tc, 8, 8, 3, nb=64, buckets=(), ring_T=None) == c3
    # 48 GDN layers x 8 users x 4 slots x 6 heads x 64 KiB = 576 MiB per device at T = 4
    assert c3["per_seq"]["gdn_ring"] == 48 * 8 * 4 * 6 * 128 * 128 * 4


# ------------------------------------------------------------------------- wide-family spec SDPA plan (one group)
def _plan_fake():
    """The attributes TPAttention._spec_sdpa_plan / spec_sdpa_pack_heads read: a BH P150 grid (11 x 10), NH = 3 at TP8."""
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention

    grid = SimpleNamespace(x=11, y=10)
    a = SimpleNamespace(
        NH=3,
        _spec_sdpa_cfg_cache={},
        _SPEC_SDPA_L1_FIT=TPAttention._SPEC_SDPA_L1_FIT,
        mesh=SimpleNamespace(compute_with_storage_grid_size=lambda: grid),
    )
    a.spec_sdpa_pack_heads = lambda Tg: TPAttention.spec_sdpa_pack_heads(a, Tg)
    return lambda T, n=1: TPAttention._spec_sdpa_plan(a, T, n)


@pytest.mark.parametrize(
    "env, T, users, want",
    [
        ({}, 8, 4, (8, 4, 13)),  # flag off: the r5 table (2 groups of Tg = 4 per user, 110 // 8 cores)
        ({"QWEN36_SPEC_ADAPTIVE_K": "1"}, 8, 4, (4, 8, 27)),  # one packed group per user, 110 // 4 cores
        ({"QWEN36_SPEC_ADAPTIVE_K": "1"}, 8, 2, (2, 8, 55)),
        ({"QWEN36_SPEC_ADAPTIVE_K": "1"}, 8, 1, (1, 8, 64)),  # the cap of 64 cores per group
        ({"QWEN36_SPEC_ADAPTIVE_K": "1", "QWEN36_SPEC_ADAPTIVE_K_SDPA": "0"}, 8, 4, (8, 4, 13)),
        ({"QWEN36_SPEC_ADAPTIVE_K": "1", "QWEN36_SPEC_SDPA_PACKED": "0"}, 8, 4, (8, 4, 13)),  # unpacked: no 32-row tile
        ({"QWEN36_SPEC_ADAPTIVE_K": "1"}, 4, 8, (8, 4, 13)),  # the base family's T = 4: unchanged
        ({"QWEN36_SPEC_ADAPTIVE_K": "1"}, 12, 2, (6, 4, 18)),  # T = 12: 12 x 4 rows > 32, the table's split
    ],
)
def test_spec_sdpa_plan_one_group(monkeypatch, env, T, users, want):
    from models.demos.blackhole.qwen36.tt.attention import tp as attn_tp

    if not attn_tp.SPEC_SDPA_PACK_SUPPORTED:
        pytest.skip("this ttnn build has no spec_pack_heads")
    for k in ("QWEN36_SPEC_ADAPTIVE_K", "QWEN36_SPEC_ADAPTIVE_K_SDPA"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("QWEN36_SPEC_SDPA_PACKED", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cfg, groups, tg = _plan_fake()(T, users)
    assert (groups, tg, cfg.max_cores_per_head_batch) == want


# ------------------------------------------------------------------ sampled accept at the wide family's K = 7
def _fsh(monkeypatch, k, n_cases=None):
    """tests/test_spec_fused_sampled_host.py (the device-algorithm mirror vs the plugin's accept, and its chi-square
    losslessness test) at K = k drafts: its module constants K / T are read at call time."""
    pytest.importorskip("vllm_tt_plugin.spec_accept")
    from models.demos.blackhole.qwen36.tests import test_spec_fused_sampled_host as fsh

    monkeypatch.setattr(fsh, "K", int(k))
    monkeypatch.setattr(fsh, "T", int(k) + 1)
    if n_cases:
        monkeypatch.setattr(fsh, "N_CASES", int(n_cases))
    return fsh


def test_sampled_accept_k7_matches_plugin(monkeypatch):
    """fused_sampled_reference (the fused sampled accept device's float32 algorithm) == the plugin's
    accept_sampled_support on random candidate blocks at K = 7 (exact outside the 1e-6 boundary classes)."""
    _fsh(monkeypatch, 7, 3000).test_random_blocks_match_plugin()


@pytest.mark.parametrize(
    "temp,k,p,nv,ranks",
    [
        (1.0, 20, 1.0, 7, [0, 0, 1, 0, 0, 2, 0]),
        (0.7, 40, 0.8, 7, [0, 3, 0, 0, 1, 0, 0]),
        (0.9, 30, 0.95, 5, [0, 0, 0, 1, 0, -1, 0]),
    ],
)
def test_sampled_joint_k7_matches_target(monkeypatch, temp, k, p, nv, ranks):
    """Lossless at K = 7: the joint (count, first committed token) of the fused sampled walk over 30000 draws matches the
    exact target distribution (chi-square at p = 0.999, plus the power check that a 3% mass shift is detected)."""
    _fsh(monkeypatch, 7).test_joint_count_first_token_matches_target(temp, k, p, nv, ranks)


# ------------------------------------------------- runner-geometry results satisfy the plugin's AcceptResult rules
def _api_fake(perm):
    """What AdaptiveKMixin._ak_result_api / _ak_api_rows read: the adaptive state, the state width and K_api, the slot
    permutation helper (MTPSpecEngine._perm_slots)."""
    from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

    f = SimpleNamespace(_ak=_state(), _akv_B_state=8, _akv_K_api=5, B=4, K=5, T=6)
    f._B_state, f.K_api = 8, 5
    f._perm_slots = MTPSpecEngine._perm_slots
    f._ak_api_rows = lambda h, x, fill, col0=None: sa.AdaptiveKMixin._ak_api_rows(f, h, x, fill, col0)
    f._ak_in = lambda fam: __import__("contextlib").nullcontext()
    return f


@pytest.mark.parametrize("perm", [None, ([3, 0, 1, 2, 4, 5, 6, 7], [1, 2, 3, 0, 4, 5, 6, 7])])
def test_wide_results_pass_the_plugin_accept_check(perm):
    """A wide-family step (4 engine rows, K = 5) mapped to the runner's [8, 1 + K_api] block: the rows the family did not
    run (engine rows 4..7) must come back as padding rows (count 1, a real column 0, PLACEHOLDER after), the rows it ran
    unchanged. vLLM's AcceptResult (plugin spec_accept.py) rejects PLACEHOLDER inside a committed prefix: the serving
    replay's first adaptive step died on exactly that before the padded rows got a column 0."""
    acc = pytest.importorskip("vllm_tt_plugin.spec_accept")
    from models.demos.blackhole.qwen36.tt.spec_engine import VerifyResult, slot_map_inverse

    st = _state()
    wide = sa.KFamily("wide", 4, 5)
    st.wide = wide
    f = _api_fake(perm)
    f._ak = st
    counts = torch.tensor([3, 1, 6, 2], dtype=torch.int32)
    committed = torch.full((4, 6), sa.PLACEHOLDER, dtype=torch.int32)
    for u, c in enumerate(counts.tolist()):
        committed[u, :c] = torch.arange(100 * (u + 1), 100 * (u + 1) + c, dtype=torch.int32)
    res = VerifyResult("fused_sample", committed.clone(), accepted_token_ids=committed, accepted_counts=counts)
    out = sa.AdaptiveKMixin._ak_result_api(f, res, wide, perm, slot_map_inverse, VerifyResult)
    assert tuple(out.accepted_token_ids.shape) == (8, 6) and tuple(out.accepted_counts.shape) == (8,)
    acc.AcceptResult(out.accepted_token_ids, out.accepted_counts)  # raises on a count / PLACEHOLDER mismatch
    ids = out.accepted_token_ids if perm is None else out.accepted_token_ids[torch.tensor(perm[1])]
    cnt = out.accepted_counts if perm is None else out.accepted_counts[torch.tensor(perm[1])]
    assert torch.equal(ids[:4], committed) and torch.equal(cnt[:4], counts)
    assert cnt[4:].tolist() == [1] * 4 and ids[4:, 0].tolist() == [0] * 4 and bool((ids[4:, 1:] == sa.PLACEHOLDER).all())
    # the deferred readout (_ak_api_rows on a handle of the family) pads the same way
    h = SimpleNamespace(ak_api=dict(perm=perm, B=8, W=6))
    c2 = sa.AdaptiveKMixin._ak_api_rows(f, h, committed, sa.PLACEHOLDER, col0=0)
    n2 = sa.AdaptiveKMixin._ak_api_rows(f, h, counts, 1)
    acc.AcceptResult(c2, n2)


def test_drafts_from_steps_tolerates_unwritten_rows():
    """A bucketed wide-family step leaves the drafter readback rows past its bucket as the (shared) buffers last held them:
    any 32-bit pattern, e.g. a negative bf16 pair read as uint32. _drafts_from_steps must not raise on those; written rows
    (token ids) pass through unchanged (seen on device: steps[1] = [264, 3672, 3208953432, 1058094935] with rows 2-3 padding)."""
    import types

    from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

    eng = types.SimpleNamespace(B=4, K=5, _ak=object())  # adaptive K on: the readback is always tolerant
    steps = [[728, 279, 0, 0], [264, 3672, 3208953432, 1058094935], [5357, 11, 0, 0], [314, 864, 0, 0], [39026, 513, 0, 0]]
    d = MTPSpecEngine._drafts_from_steps(eng, steps)
    assert d.dtype == torch.int32 and tuple(d.shape) == (4, 5)
    assert d[0].tolist() == [728, 264, 5357, 314, 39026] and d[1].tolist() == [279, 3672, 11, 864, 513]
    assert d[2, 1] == 0 and d[3, 1] == 1058094935
