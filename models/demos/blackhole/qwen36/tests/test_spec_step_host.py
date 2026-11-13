# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the packed per-step inputs of the MTP verify and drafter traces (tt/spec_step.py).

* The drafter pack holds, value for value, what the recorded drafter staged per buffer before (anchor_selector,
  reseed_row_plan / reseed_rows_host, draft_step_plan), and its RoPE table rows name the positions plus deltas the host
  rotated by, for every real row.
* The RoPE table rows hold the bytes _rope_tp_cos_sin_decode_torch gives a step of 1, 4, 8 or 32 rows.
* The verify stages one step buffer per replay, and its page tables, the GDN verify tables and the sampler's request
  parameters only when they change.
"""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_engine import (
    MTPSpecEngine,
    accepted_plan,
    anchor_selector,
    draft_step_plan,
    reseed_row_plan,
    reseed_rows_host,
)
from models.demos.blackhole.qwen36.tt.spec_sample import VerifySampler, sampling_plan
from models.demos.blackhole.qwen36.tt.spec_step import (
    DRAFT_PENDING,
    VERIFY_POSITIONS,
    VERIFY_ROPE,
    VERIFY_TOKENS,
    draft_anchor,
    draft_reseed_positions,
    draft_reseed_rope,
    draft_reseed_tokens,
    draft_step_rope,
    draft_step_row,
    drafter_pack,
    pack_u32_rows,
    reseed_real_rows,
    reseed_table,
    rope_rows,
    rope_table_rows,
)

ROPE_THETA = 10_000_000.0
RD = 64


def _rope_model(max_seq_len=4096, rope_delta=0):
    m = Qwen36Model.__new__(Qwen36Model)
    m.args = SimpleNamespace(rope_head_dim=RD, rope_theta=ROPE_THETA, max_seq_len=max_seq_len)
    m.rope = SimpleNamespace(rope_delta=rope_delta)
    m.device = None
    return m


def _host_tables(model, monkeypatch):
    """rope_rm_tables with the upload stood in: the host (cos, sin) [rows, RD] it would replicate."""
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    model._rope_rm = None
    return model.rope_rm_tables()


def _step_case(rng, B, K, rows=4096):
    """One propose of B users: committed block, accepted drafts, verify column-0 positions, decline, keep, deltas."""
    T = K + 1
    m = [rng.randrange(T) for _ in range(B)]
    deltas = [rng.choice([0, 0, -rng.randrange(1, 40), rng.randrange(1, 9)]) for _ in range(B)]
    # a request's first verify position is at least its prompt length, which exceeds -delta (M-RoPE)
    p0 = [rng.randrange(max(1, 1 - d), rows - 2 * T - 9) for d in deltas]
    ctok = torch.full((B, T), -1, dtype=torch.int64)
    for u in range(B):
        ctok[u, : m[u] + 1] = torch.tensor([rng.randrange(248320) for _ in range(m[u] + 1)])
    cpos = torch.tensor([[p0[u] + 1 + j for j in range(T)] for u in range(B)])
    decline = [rng.random() < 0.3 for _ in range(B)]
    keep = [d and rng.random() < 0.5 for d in decline]
    return ctok, cpos, m, p0, decline, keep, deltas


@pytest.mark.parametrize("B,K", [(1, 3), (2, 3), (4, 3), (8, 3), (1, 7), (4, 7), (2, 11)])
def test_the_drafter_pack_holds_the_staged_drafter_inputs(B, K):
    """For random proposes (declined rows, kept rows, M-RoPE deltas), the pack's rows equal the per-buffer values the
    recorded drafter staged before: the anchor one-hot's column, the reseed tokens, positions and table, the pending
    tokens and the step positions (declined rows at 0); every real row's RoPE table row is its position plus delta."""
    rng = random.Random(B * 100 + K)
    T, R, rows, scratch = K + 1, B * (K + 1), 4096, 999
    for _ in range(50):
        ctok, cpos, m, p0, decline, keep, deltas = _step_case(rng, B, K, rows)
        pt = torch.randint(1, 500, (B, 6), dtype=torch.int32)
        rb = rope_rows(p0, deltas, rows)
        pack = drafter_pack(ctok, m, p0, decline, keep, rb, K, rows)
        # anchor: the one-hot's column
        sel = anchor_selector(m, B, T)[0, 0]
        assert pack[draft_anchor(K), :B].tolist() == sel.argmax(-1).tolist()
        # reseed rows (m_kv = m + 1 on a kept declined row)
        m_kv = [mu + int(d and k) for mu, d, k in zip(m, decline, keep)]
        tok, pos, pt_rows = reseed_rows_host(reseed_row_plan(ctok, cpos, m_kv, B, T), pt, T, 6, scratch)
        assert pack[draft_reseed_tokens(K), :R].tolist() == tok.reshape(-1).tolist()
        assert pack[draft_reseed_positions(K), :R].tolist() == pos.tolist()
        assert torch.equal(reseed_table(pt, reseed_real_rows(m, decline, keep, T), T, scratch), pt_rows)
        # draft steps: pending token and positions, declined rows at token 0, positions 0 .. K-1
        _, _, pending, p = accepted_plan(ctok, cpos, [mu + 1 for mu in m], B, T)
        pending = [0 if d else t for d, t in zip(decline, pending)]
        p = [0 if d else pu for d, pu in zip(decline, p)]
        tok0, steps = draft_step_plan(pending, p, K)
        assert pack[DRAFT_PENDING, :B].tolist() == tok0.reshape(-1).tolist()
        for k in range(K):
            assert pack[draft_step_row(k), :B].tolist() == steps[k].tolist()
        # RoPE table rows of every real row: position + its user's delta
        rr = pack[draft_reseed_rope(K), :R].reshape(B, T)
        for u in range(B):
            for i in range(m_kv[u]):
                assert int(rr[u, i]) == int(pos[u * T + i]) + deltas[u]
            if not decline[u]:
                for k in range(K):
                    assert int(pack[draft_step_rope(K, k), u]) == p[u] + k + deltas[u]
        assert not bool(pack[:, max(R, B) :].any()), "padding words stay zero"


@pytest.mark.parametrize("n", [1, 4, 8, 32])
def test_the_rope_table_rows_hold_the_host_cos_and_sin(n, monkeypatch):
    """A row of the device RoPE tables is _rope_tp_cos_sin_decode_torch of that RoPE position, byte for byte, for a step
    of n rows at the deployment's table size; so a lookup of position + delta gives what the host staged for it."""
    model = _rope_model(max_seq_len=262144)
    cos_t, sin_t = _host_tables(model, monkeypatch)
    rows = rope_table_rows(262144)
    assert tuple(cos_t.shape) == tuple(sin_t.shape) == (rows, RD) and cos_t.dtype == torch.bfloat16
    rng = random.Random(n)
    for _ in range(40):
        pos = torch.tensor([rng.randrange(0, 262144) for _ in range(n)], dtype=torch.int32)
        deltas = [rng.choice([0, -rng.randrange(1, 4000)]) for _ in range(n)]
        pos = torch.maximum(pos, torch.tensor([-d for d in deltas], dtype=torch.int32))
        want_cos, want_sin = model._rope_tp_cos_sin_decode_torch(pos, deltas)
        idx = rope_rows(pos, deltas, rows)
        assert torch.equal(cos_t[idx], want_cos.reshape(n, RD)) and torch.equal(sin_t[idx], want_sin.reshape(n, RD))


def test_rope_rows_clamp_only_outside_the_table(expect_error):
    assert rope_rows([0, 5, 4095], 0, 4096).tolist() == [0, 5, 4095]
    assert rope_rows([10, 10, 10], [-3, 0, 2], 4096).tolist() == [7, 10, 12]
    assert rope_rows([2, 4100], [-5, 0], 4096).tolist() == [0, 4095], "padding and junk rows clamp"
    with expect_error(ValueError, "2 M-RoPE deltas for 3 positions"):
        rope_rows([1, 2, 3], [0, 0], 4096)


def test_pack_u32_rows_refuses_what_uint32_rows_cannot_hold(expect_error):
    assert pack_u32_rows([[1, 2], [3]]).tolist() == [[1, 2] + [0] * 30, [3] + [0] * 31]
    with expect_error(ValueError, "negative"):
        pack_u32_rows([[1, -1]])
    with expect_error(ValueError, "holds 32"):
        pack_u32_rows([list(range(33))], width=32)


def _bare_verify_model(B, T, monkeypatch):
    m = _rope_model()
    m._vfy_trace_id, m._vfy_T, m._vfy_B, m._vfy_kv_grouped = "trace", T, B, False
    m._vfy_pt_host = torch.zeros(B, 4, dtype=torch.int32)
    m.gdn_blocks, m.device, m.vocab_size = object(), None, 8
    for name in (
        "_vfy_step_buf",
        "_vfy_kvpt_buf",
        "_vfy_kvpt_users",
        "_vfy_kvpt1_buf",
        "_vfy_ids_out",
        "_vfy_rows_out",
    ):
        setattr(m, name, MagicMock(name=name))
    m._vfy_spec_groups = B
    copies = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda h, d: copies.append((h, d)))
    monkeypatch.setattr(ttnn, "to_torch", lambda t, **kw: torch.zeros(B * T))
    return m, copies


def test_the_verify_stages_one_step_buffer_and_changed_tables_only(monkeypatch):
    """One verify replay stages one step buffer (tokens, KV positions, RoPE table rows); the three page tables are
    copied only when the table differs from the one the device buffers hold."""
    B, T = 2, 4
    m, copies = _bare_verify_model(B, T, monkeypatch)
    tokens = [[5, 6, 7, 8], [9, 9, 9, 9]]
    pt = torch.tensor([[3, 4, 0, 0], [7, 0, 0, 0]], dtype=torch.int32)
    m.verify_traced(tokens, [10, 40], [0] * B, page_tables=pt, rope_deltas=[0, -3])
    assert [d for _, d in copies] == [m._vfy_kvpt_buf, m._vfy_kvpt_users, m._vfy_kvpt1_buf, m._vfy_step_buf]
    step = copies[-1][0]
    assert step[VERIFY_TOKENS, : B * T].tolist() == [5, 6, 7, 8, 9, 9, 9, 9]
    assert step[VERIFY_POSITIONS, : B * T].tolist() == [10, 11, 12, 13, 40, 41, 42, 43]
    assert step[VERIFY_ROPE, : B * T].tolist() == [10, 11, 12, 13, 37, 38, 39, 40]
    copies.clear()
    m.verify_traced(tokens, [11, 41], [0] * B, page_tables=pt.clone(), rope_deltas=[0, -3])
    assert [d for _, d in copies] == [m._vfy_step_buf], "an unchanged table is not copied again"
    copies.clear()
    pt[1, 1] = 8
    m.verify_traced(tokens, [11, 41], [0] * B, page_tables=pt)
    assert [d for _, d in copies][:3] == [m._vfy_kvpt_buf, m._vfy_kvpt_users, m._vfy_kvpt1_buf]
    assert torch.equal(m._vfy_pt_host, pt)


def test_the_sampler_copies_its_request_parameters_when_they_change(monkeypatch):
    """Top-k, top-p and temperature are copied when they change; the seeds whenever they change (every step of an
    unseeded row)."""
    s = VerifySampler.__new__(VerifySampler)
    s.k, s.p, s.temp, s.seeds, s._staged = "k", "p", "temp", "seeds", {}
    s._host = lambda t, dtype: t
    copies = []
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda h, d: copies.append(d))
    params = SimpleNamespace(temperature=[1.0, 0.0], top_k=[20, 1], top_p=[0.95, 1.0], seed=[None, None])
    seeds = iter(range(1, 100))
    plan = lambda: sampling_plan(params, [10, 20], 4, fresh_seed=lambda: next(seeds))
    s.stage(plan())
    assert copies == ["k", "p", "temp", "seeds"]
    copies.clear()
    s.stage(plan())
    assert copies == ["seeds"], "fresh seeds every step, the parameters once"
    params.top_k = [5, 1]
    copies.clear()
    s.stage(plan())
    assert copies == ["k", "seeds"]


def test_the_drafter_derives_its_inputs_from_the_pack(monkeypatch):
    """The drafter trace slices each input from its pack row (spec_step layout), types the positions int32, looks up
    cos and sin by table row, and builds the anchor one-hot from column + 1 against anchor + 1."""
    B, K = 2, 3
    R = B * (K + 1)
    calls = []

    def op(name):
        def f(*a, **kw):
            calls.append((name, a, kw))
            return f"{name}{len(calls)}"

        return f

    for name in ("typecast", "to_layout", "transpose", "add", "eq", "reshape"):
        monkeypatch.setattr(ttnn, name, op(name))
    monkeypatch.setattr(ttnn, "slice", lambda t, s, e: calls.append(("slice", t, s, e)) or ("seg", s[0], e[1]))
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e._dr_pack, e._dr_iota = B, K, K + 1, "pack", "iota"
    lookups = []
    e.model = SimpleNamespace(
        rope_lookup=lambda rows, n: lookups.append((rows, n)) or (f"cos{rows[1]}", f"sin{rows[1]}")
    )
    x, owned = MTPSpecEngine._drafter_inputs(e)
    slices = [(c[2][0], c[3][1]) for c in calls if c[0] == "slice"]
    assert sorted(slices) == sorted(
        [(draft_anchor(K), B), (draft_reseed_rope(K), R), (draft_reseed_tokens(K), R), (draft_reseed_positions(K), R)]
        + [(draft_step_rope(K, k), B) for k in range(K)]
        + [(DRAFT_PENDING, B)]
        + [(draft_step_row(k), B) for k in range(K)]
    )
    assert x["tok0"] == ("seg", DRAFT_PENDING, B) and x["rs_tok"] == ("seg", draft_reseed_tokens(K), R)
    assert lookups == [(("seg", draft_reseed_rope(K), R), R)] + [
        (("seg", draft_step_rope(K, k), B), B) for k in range(K)
    ]
    assert x["rs_cos"] == f"cos{draft_reseed_rope(K)}" and x["cos"] == [f"cos{draft_step_rope(K, k)}" for k in range(K)]
    eq = next(c for c in calls if c[0] == "eq")
    assert eq[1][0] == "iota" and eq[2] == {"dtype": ttnn.bfloat16}
    add = next(c for c in calls if c[0] == "add")
    assert add[1][1] == 1.0 and eq[1][1] == f"add{calls.index(add) + 1}"
    assert x["anchor_sel"] in owned and all(p in owned for p in x["pos"])


def test_the_rope_tables_reach_every_paged_position(monkeypatch):
    """The paged tables can reach past max_seq_len (whole blocks of max_seq_len + K + 1 positions): the tables are
    built for that reach, and rope_table_rows reports the rows they hold."""
    model = _rope_model(max_seq_len=4096)
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    assert model.rope_table_rows() == rope_table_rows(4096)
    cos_t, _ = model.rope_rm_tables(reach=5 * 1024)
    assert model.rope_table_rows() == cos_t.shape[0] == rope_table_rows(5 * 1024)
    want, _ = model._rope_tp_cos_sin_decode_torch(torch.tensor([5100], dtype=torch.int32), [0])
    assert torch.equal(cos_t[5100], want.reshape(RD))
    assert model.rope_rm_tables(reach=10**6)[0] is cos_t, "built once"
