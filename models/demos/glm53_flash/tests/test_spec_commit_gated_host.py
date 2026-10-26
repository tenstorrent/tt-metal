# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests (no device, torch only) of the gated KDA commit of ``tt/spec_decode.py`` (SP1,
``GLM53_SPEC_COMMIT=gated``): the knob, the per-launch write gates vs the blend masks, the torch golden of one gated
``kda_decode`` launch, the commit semantics of both modes on a torch model of ``KdaSpecState`` (V on a scratch + gated
C == V on the live state + replay on the backup + blends == the plain sequential steps per user, bit for bit), and the
``GreedySpecLoop`` driving that model through the fake-target harness of ``test_spec_decode_host.py``.

    pytest --noconftest models/demos/glm53_flash/tests/test_spec_commit_gated_host.py -q
"""

from __future__ import annotations

import pytest
import torch

ttnn = pytest.importorskip("ttnn")  # op.py imports ttnn (no device is opened)
from models.demos.glm53_flash.custom_ops.kda_decode.op import GATE_WORDS, KdaDecode  # noqa: E402
from models.demos.glm53_flash.tests.test_spec_decode_host import (  # noqa: E402
    VOCAB,
    FakeTarget,
    fake_next,
    plain_greedy,
)
from models.demos.glm53_flash.tt.spec_decode import (  # noqa: E402
    SPEC_COMMIT_MODES,
    GreedySpecLoop,
    commit_gate_rows,
    kda_commit_gates,
    kda_commit_masks,
    spec_commit_mode,
)

K_DIM = V_DIM = 128
COMMON = dict(num_heads=2, gate_lower_bound=-5.0, l2_eps=1e-6, norm_eps=1e-5)


# ------------------------------------------------------------------------------------------------ knob + gates
def test_commit_mode_knob():
    assert spec_commit_mode({}) == "blend" and tuple(SPEC_COMMIT_MODES) == ("blend", "gated")
    assert spec_commit_mode({"GLM53_SPEC_COMMIT": "gated"}) == "gated"
    assert spec_commit_mode({"GLM53_SPEC_COMMIT": " GATED "}) == "gated"
    assert spec_commit_mode({"GLM53_SPEC_COMMIT": "blend"}) == "blend"
    raised = False
    try:
        spec_commit_mode({"GLM53_SPEC_COMMIT": "inplace"})
    except ValueError as e:
        raised = "GLM53_SPEC_COMMIT" in str(e)
    assert raised


def test_commit_gates_vs_masks():
    K = 3
    m = torch.tensor([0, 3, 1, 2, 3, 0])
    g = kda_commit_gates(m, K)
    assert g.dtype == torch.int32 and tuple(g.shape) == (K + 1, 6)
    assert g[0].tolist() == [1] * 6, "launch 0 (the anchor) writes every user"
    for j in range(K + 1):
        assert g[j].tolist() == [int(int(mu) >= j) for mu in m.tolist()]
    assert torch.all(g[:-1] >= g[1:]), "monotone: a user rejected at launch j is rejected after it"
    take = kda_commit_masks(m, K)
    g_next = torch.cat([g[1:], torch.zeros(1, 6, dtype=torch.int32)])
    assert torch.equal((g - g_next).float(), take), "take_j = gate_j - gate_{j+1}"
    assert g.sum(0).tolist() == (m + 1).tolist(), "user u takes m_u + 1 launches"
    # the gate pages: row r = mesh row r's B users in the first B words, padding 0
    rows = commit_gate_rows(g[1], users_per_row=3, rows=2)
    assert tuple(rows.shape) == (2, GATE_WORDS) and rows.dtype == torch.int32
    assert rows[0, :3].tolist() == [0, 1, 1] and rows[1, :3].tolist() == [1, 1, 0]
    assert rows[:, 3:].sum() == 0
    for m0 in (torch.zeros(4, dtype=torch.long), torch.full((4,), K)):
        gg = kda_commit_gates(m0, K)
        assert gg[1:].sum() == (0 if int(m0[0]) == 0 else 3 * 4)


# ------------------------------------------------------------------------------------------------ torch model
def _inputs(B, H, seed):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    W = 3 * H * K_DIM
    Hb = -(-H // 32) * 32
    conv = (r(B, 3, W) * 0.8).to(torch.bfloat16).float()
    S = 0.05 * r(B, H, K_DIM, V_DIM)
    taps = (r(4, W) * 0.5).to(torch.bfloat16).float()
    dt_bias = r(H * K_DIM) * 0.5
    a_exp = torch.rand(H, generator=gen).repeat_interleave(K_DIM) * 4.0 + 0.5
    onw = torch.rand(V_DIM, generator=gen) * 0.5 + 0.75
    return dict(conv=conv, S=S, taps=taps, dt_bias=dt_bias, a_exp=a_exp, onw=onw, gen=gen)


def _slab(gen, B, H):
    """One candidate slab: the raw qkv rows (bf16-valued) and the gate pre-activations of B users."""
    W = 3 * H * K_DIM
    Hb = -(-H // 32) * 32
    qkv = torch.randn(B, W, generator=gen).to(torch.bfloat16).float()
    gates = torch.zeros(B, H * K_DIM + H * V_DIM + Hb)
    gates[:, : H * K_DIM] = torch.randn(B, H * K_DIM, generator=gen) * 2.0
    gates[:, H * K_DIM : H * K_DIM + H * V_DIM] = torch.randn(B, H * V_DIM, generator=gen) * 2.0
    gates[:, H * K_DIM + H * V_DIM : H * K_DIM + H * V_DIM + H] = torch.randn(B, H, generator=gen) * 2.0
    return qkv, gates


def _step(x, qkv, gates, S, conv, gate=None):
    """One ``kda_decode`` launch (the torch golden) on (S, conv): plain, or gated per user."""
    args = (qkv, gates, conv, S, x["taps"], x["dt_bias"], x["a_exp"], x["onw"])
    if gate is None:
        return KdaDecode.golden(*args, **COMMON)
    return KdaDecode.golden_gated(*args, gate, **COMMON)


class TorchKdaSpec:
    """``KdaSpecState`` on torch tensors: ``verify(slabs)`` = the K + 1 launches (blend: backup + launches on the live
    state; gated: copy to the scratch + launches there), ``commit(m)`` = the mode's commit. ``live`` is the state the
    model decodes from."""

    def __init__(self, x, K, mode):
        self.x, self.K, self.mode = x, K, mode
        self.live = (x["S"].clone(), x["conv"].clone())
        self.bk = None
        self.slabs = None

    def verify(self, slabs):
        self.slabs = slabs
        self.bk = (self.live[0].clone(), self.live[1].clone())
        S, conv = self.bk if self.mode == "gated" else self.live
        outs = []
        for qkv, g in slabs:
            o, S, conv = _step(self.x, qkv, g, S, conv)
            outs.append(o)
        if self.mode == "gated":
            self.bk = (S, conv)
        else:
            self.live = (S, conv)
        return outs

    def commit(self, m):
        K = self.K
        if self.mode == "gated":
            gates = kda_commit_gates(m, K)
            S, conv = self.live
            for j, (qkv, g) in enumerate(self.slabs):
                _, S, conv = _step(self.x, qkv, g, S, conv, gate=gates[j])
            self.live = (S, conv)
            return
        take = kda_commit_masks(m, K)
        S_bk, conv_bk = self.bk
        S, conv = self.live
        for j, (qkv, g) in enumerate(self.slabs):
            _, S_bk, conv_bk = _step(self.x, qkv, g, S_bk, conv_bk)
            t_rec = take[j].reshape(-1, 1, 1, 1)
            t_conv = take[j].reshape(-1, 1, 1)
            S = S * (1.0 - t_rec) + S_bk * t_rec
            conv = conv * (1.0 - t_conv) + conv_bk * t_conv
        self.live = (S, conv)
        self.bk = None


def _plain_states(x, slabs, B):
    """Per user u: the plain sequential state after launches 0 .. j for every j (the reference of commit(m))."""
    S, conv = x["S"].clone(), x["conv"].clone()
    out = []
    for qkv, g in slabs:
        _, S, conv = _step(x, qkv, g, S, conv)
        out.append((S.clone(), conv.clone()))
    return out


def test_gated_launch_golden_per_user():
    B, H = 4, 2
    x = _inputs(B, H, seed=1)
    qkv, g = _slab(x["gen"], B, H)
    o_ref, S_ref, conv_ref = _step(x, qkv, g, x["S"], x["conv"])
    gate = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    o, S, conv = _step(x, qkv, g, x["S"], x["conv"], gate=gate)
    assert torch.equal(o, o_ref), "the output rows are the plain launch's (the commit discards them)"
    for u in range(B):
        if int(gate[u]):
            assert torch.equal(S[u], S_ref[u]) and torch.equal(conv[u], conv_ref[u]), f"gated user {u}: plain update"
        else:
            assert torch.equal(S[u], x["S"][u]) and torch.equal(conv[u], x["conv"][u]), f"user {u}: bytes kept"
    _, S1, conv1 = _step(x, qkv, g, x["S"], x["conv"], gate=torch.ones(B, dtype=torch.int32))
    assert torch.equal(S1, S_ref) and torch.equal(conv1, conv_ref)
    _, S0, conv0 = _step(x, qkv, g, x["S"], x["conv"], gate=torch.zeros(B, dtype=torch.int32))
    assert torch.equal(S0, x["S"]) and torch.equal(conv0, x["conv"])
    assert not torch.equal(S_ref, x["S"]), "the launch changes the state (the test is not vacuous)"


@pytest.mark.parametrize("K", [1, 2, 3])
def test_gated_commit_equals_blend_commit_and_plain(K):
    """Both modes end with every user at the plain state after m_u + 1 launches, bit for bit, and agree with each
    other; the gated verify leaves the live state untouched; the verify outputs are identical in both modes."""
    B, H = 4, 2
    x = _inputs(B, H, seed=10 + K)
    slabs = [_slab(x["gen"], B, H) for _ in range(K + 1)]
    plain = _plain_states(x, slabs, B)
    m = torch.tensor([u % (K + 1) for u in range(B)])
    blend, gated = TorchKdaSpec(x, K, "blend"), TorchKdaSpec(x, K, "gated")
    o_b, o_g = blend.verify(slabs), gated.verify(slabs)
    assert all(torch.equal(a, b) for a, b in zip(o_b, o_g)), "the verify's launches see the same state in both modes"
    assert torch.equal(gated.live[0], x["S"]) and torch.equal(gated.live[1], x["conv"]), "gated V: live untouched"
    assert torch.equal(gated.bk[0], plain[K][0]) and torch.equal(gated.bk[1], plain[K][1]), "gated V: scratch = S_{K+1}"
    assert torch.equal(blend.live[0], plain[K][0]), "blend V: live = S_{K+1}"
    blend.commit(m)
    gated.commit(m)
    for u in range(B):
        S_ref, conv_ref = plain[int(m[u])]
        assert torch.equal(gated.live[0][u], S_ref[u]) and torch.equal(gated.live[1][u], conv_ref[u]), f"gated u{u}"
        assert torch.equal(blend.live[0][u], S_ref[u]) and torch.equal(blend.live[1][u], conv_ref[u]), f"blend u{u}"
    assert torch.equal(blend.live[0], gated.live[0]) and torch.equal(blend.live[1], gated.live[1])
    # a second iteration from the committed state: the gated path needs no backup to be "spent"
    slabs2 = [_slab(x["gen"], B, H) for _ in range(K + 1)]
    x2 = dict(x, S=gated.live[0].clone(), conv=gated.live[1].clone())
    plain2 = _plain_states(x2, slabs2, B)
    m2 = torch.tensor([(u + 1) % (K + 1) for u in range(B)])
    gated.verify(slabs2)
    gated.commit(m2)
    blend.verify(slabs2)
    blend.commit(m2)
    for u in range(B):
        assert torch.equal(gated.live[0][u], plain2[int(m2[u])][0][u]) and torch.equal(
            gated.live[1][u], plain2[int(m2[u])][1][u]
        )
    assert torch.equal(blend.live[0], gated.live[0]) and torch.equal(blend.live[1], gated.live[1])


# ------------------------------------------------------------------------------------------------ the loop
def _row_inputs(token: int, position: int, B_slot: int, H: int):
    """The KDA block inputs of one decode row: a deterministic function of (token, position) so a reference state
    can be rebuilt from the committed context alone."""
    gen = torch.Generator().manual_seed(1_000_003 * int(token) + 7919 * int(position) + 13 * B_slot)
    qkv, gates = _slab(gen, 1, H)
    return qkv[0], gates[0]


class KdaTrackingTarget(FakeTarget):
    """The fake target whose verify / commit also drive a ``TorchKdaSpec`` (the gated mode) with per-row inputs
    derived from (token, position), and a plain reference state advanced by the committed tokens in order."""

    def __init__(self, prompts, K, acc, x, H, mode="gated", seed=0):
        super().__init__(prompts, K, acc, seed)
        self.spec = TorchKdaSpec(x, K, mode)
        self.x, self.H = x, H
        self.ref = (x["S"].clone(), x["conv"].clone())
        self.ref_steps = self.committed_rows = 0

    def verify(self, tokens, positions):
        ids, d1 = super().verify(tokens, positions)
        N, P = tokens.shape
        slabs = []
        for j in range(P):
            rows = [_row_inputs(int(tokens[u, j]), positions[u] + j, u, self.H) for u in range(N)]
            slabs.append((torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows])))
        self.spec.verify(slabs)
        return ids, d1

    def commit(self, m, positions):
        pending = self.pending.clone()
        super().commit(m, positions)
        self.spec.commit(m)
        # the reference: every user applies its committed rows (anchor + accepted drafts) in order, plain steps
        N = len(m)
        S, conv = self.ref
        for u in range(N):
            for j in range(int(m[u]) + 1):
                qkv, g = _row_inputs(int(pending[u, j]), positions[u] + j, u, self.H)
                _, S_u, conv_u = _step(self.x, qkv[None], g[None], S[u : u + 1], conv[u : u + 1])
                S = torch.cat([S[:u], S_u, S[u + 1 :]])
                conv = torch.cat([conv[:u], conv_u, conv[u + 1 :]])
                self.ref_steps += 1
            self.committed_rows += int(m[u]) + 1
        self.ref = (S, conv)
        assert torch.equal(self.spec.live[0], S) and torch.equal(self.spec.live[1], conv), "commit != plain"


@pytest.mark.parametrize("mode", ["gated", "blend"])
@pytest.mark.parametrize("K, acc", [(1, 0.6), (3, 0.6), (3, 1.0), (2, 0.0)])
def test_loop_commit_semantics_with_kda_state(K, acc, mode):
    """The lossless loop over the KDA-tracking target: the tokens equal plain greedy, and after EVERY commit the
    gated (or blend) live state equals the plain state advanced by the committed tokens (checked inside commit)."""
    prompts = [[1, 2, 3, 4, 5], [9, 8, 7], [4, 4, 4, 4, 4, 4, 4], [2, 7, 1, 8]]
    H = 2
    x = _inputs(len(prompts), H, seed=K * 10 + int(acc * 10))
    target = KdaTrackingTarget(prompts, K, acc, x, H, mode=mode, seed=K)
    anchors = [fake_next(p) for p in prompts]
    loop = GreedySpecLoop(
        len(prompts), K, verify=target.verify, commit=target.commit, draft_chain=target.draft_chain if K > 1 else None
    )
    n = 14
    outs, stats = loop.run(anchors, [len(p) for p in prompts], n)
    for u, p in enumerate(prompts):
        assert outs[u] == plain_greedy(p, n + 1)[1 : n + 1]
    assert target.commits == target.verifies == stats.iterations >= 1
    if acc == 1.0:
        assert stats.mean_accepted > 0.5
    # the per-commit assertion inside KdaTrackingTarget.commit is the check (a loop that skipped a commit or committed
    # the wrong m fails there); the reference took one plain step per committed row, at least one per output token
    assert target.ref_steps == target.committed_rows >= sum(len(o) for o in outs)
