# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests (no device, torch only) of the fused KDA commit of ``tt/spec_decode.py`` (SP2,
``GLM53_SPEC_COMMIT=fused``): the knob, the step page vs the gates / masks, a torch model of ``KdaSpecState`` in the
fused mode (V = one dry fused launch leaving the live state untouched with the sequential launches' outputs, C = one
fused commit launch) against the gated / blend models and the plain sequential steps per user (bits, two
iterations), and the ``GreedySpecLoop`` driving that model through the fake-target harness of
``test_spec_decode_host.py`` with the per-commit plain-reference check of ``test_spec_commit_gated_host.py``.

    pytest --noconftest models/demos/glm53_flash/tests/test_spec_commit_fused_host.py -q
"""

from __future__ import annotations

import pytest
import torch

ttnn = pytest.importorskip("ttnn")  # op.py imports ttnn (no device is opened)
from models.demos.glm53_flash.custom_ops.kda_decode.op import GATE_WORDS, KdaDecode  # noqa: E402
from models.demos.glm53_flash.tests.test_spec_commit_gated_host import (  # noqa: E402
    COMMON,
    KdaTrackingTarget,
    TorchKdaSpec,
    _inputs,
    _plain_states,
    _slab,
)
from models.demos.glm53_flash.tests.test_spec_decode_host import fake_next, plain_greedy  # noqa: E402
from models.demos.glm53_flash.tt.spec_decode import (  # noqa: E402
    SPEC_COMMIT_MODES,
    GreedySpecLoop,
    commit_gate_rows,
    commit_steps_page,
    kda_commit_gates,
    kda_commit_masks,
    kda_commit_steps,
    spec_commit_mode,
)


# ------------------------------------------------------------------------------------------------ knob + page
def test_commit_mode_knob_fused():
    assert "fused" in SPEC_COMMIT_MODES and tuple(SPEC_COMMIT_MODES) == ("blend", "gated", "fused")
    assert spec_commit_mode({}) == "blend", "the default stays today's path"
    assert spec_commit_mode({"GLM53_SPEC_COMMIT": "fused"}) == "fused"
    assert spec_commit_mode({"GLM53_SPEC_COMMIT": " Fused "}) == "fused"
    assert spec_commit_mode({"GLM53_SPEC_COMMIT": "gated"}) == "gated"
    raised = False
    try:
        spec_commit_mode({"GLM53_SPEC_COMMIT": "fuse"})
    except ValueError as e:
        raised = "GLM53_SPEC_COMMIT" in str(e)
    assert raised


def test_commit_steps_vs_gates_and_masks():
    K = 3
    m = torch.tensor([0, 3, 1, 2, 3, 0])
    n = kda_commit_steps(m)
    assert n.dtype == torch.int32 and n.tolist() == [1, 4, 2, 3, 4, 1]
    g = kda_commit_gates(m, K)
    assert torch.equal(g.sum(0).to(torch.int32), n), "n_u = the number of gated launches user u takes"
    take = kda_commit_masks(m, K)
    assert [int(take[:, u].argmax()) + 1 for u in range(6)] == n.tolist(), "n_u = the blend's take launch + 1"
    assert int(n.min()) >= 1, "the anchor row is always committed: the fused commit never leaves a user untouched"
    page = commit_steps_page(m, users_per_row=3, rows=2)
    assert tuple(page.shape) == (2, GATE_WORDS) and page.dtype == torch.int32
    assert page[0, :3].tolist() == [1, 4, 2] and page[1, :3].tolist() == [3, 4, 1] and page[:, 3:].sum() == 0
    assert torch.equal(page, commit_gate_rows(n, 3, 2))
    assert commit_steps_page(torch.zeros(4, dtype=torch.long), 4, 1)[0, :4].tolist() == [1, 1, 1, 1]  # the warm-up


# ------------------------------------------------------------------------------------------------ torch model
class TorchKdaSpecFused(TorchKdaSpec):
    """``KdaSpecState`` in the fused mode on torch tensors: ``verify`` = ONE dry fused launch on the candidate-major
    slab (the live state read only), ``commit`` = ONE fused commit launch with ``n_u = m_u + 1``."""

    def __init__(self, x, K):
        super().__init__(x, K, "fused")

    def _launch(self, slabs, steps, dry):
        qkv = torch.cat([q for q, _ in slabs], dim=0)  # [R, W] candidate-major
        gates = torch.cat([g for _, g in slabs], dim=0)
        S, conv = self.live
        return KdaDecode.golden_fused(
            qkv, gates, conv, S, self.x["taps"], self.x["dt_bias"], self.x["a_exp"], self.x["onw"],
            steps=steps, dry=dry, steps_per_user=self.K + 1, **COMMON,
        )  # fmt: skip

    def verify(self, slabs):
        self.slabs = slabs
        self.bk = None  # no backup / scratch in the fused mode
        before = (self.live[0].clone(), self.live[1].clone())
        o, S, conv = self._launch(slabs, None, dry=True)
        assert torch.equal(S, before[0]) and torch.equal(conv, before[1]), "the dry launch returns the input state"
        self.live = (S, conv)
        B = before[0].shape[0]
        return [o[j * B : (j + 1) * B] for j in range(self.K + 1)]

    def commit(self, m):
        _, S, conv = self._launch(self.slabs, kda_commit_steps(m), dry=False)
        self.live = (S, conv)


@pytest.mark.parametrize("K", [1, 2, 3])
def test_fused_commit_equals_gated_blend_and_plain(K):
    """The fused mode ends every user at the plain state after m_u + 1 launches, bit for bit, agrees with the gated
    and blend models, leaves the live state untouched after V, and V's outputs equal the sequential launches'."""
    B, H = 4, 2
    x = _inputs(B, H, seed=40 + K)
    slabs = [_slab(x["gen"], B, H) for _ in range(K + 1)]
    plain = _plain_states(x, slabs, B)
    m = torch.tensor([u % (K + 1) for u in range(B)])
    fused, gated, blend = TorchKdaSpecFused(x, K), TorchKdaSpec(x, K, "gated"), TorchKdaSpec(x, K, "blend")
    o_f, o_g, o_b = fused.verify(slabs), gated.verify(slabs), blend.verify(slabs)
    assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(o_f, o_g, o_b)), "V outputs: all modes"
    assert torch.equal(fused.live[0], x["S"]) and torch.equal(fused.live[1], x["conv"]), "fused V: live untouched"
    assert fused.bk is None
    fused.commit(m)
    gated.commit(m)
    blend.commit(m)
    for u in range(B):
        S_ref, conv_ref = plain[int(m[u])]
        assert torch.equal(fused.live[0][u], S_ref[u]) and torch.equal(fused.live[1][u], conv_ref[u]), f"fused u{u}"
    assert torch.equal(fused.live[0], gated.live[0]) and torch.equal(fused.live[1], gated.live[1])
    assert torch.equal(fused.live[0], blend.live[0]) and torch.equal(fused.live[1], blend.live[1])
    # a second iteration from the committed state (nothing to spend, nothing to restore)
    slabs2 = [_slab(x["gen"], B, H) for _ in range(K + 1)]
    x2 = dict(x, S=fused.live[0].clone(), conv=fused.live[1].clone())
    plain2 = _plain_states(x2, slabs2, B)
    m2 = torch.tensor([(u + 2) % (K + 1) for u in range(B)])
    for spec in (fused, gated, blend):
        spec.verify(slabs2)
        spec.commit(m2)
    for u in range(B):
        assert torch.equal(fused.live[0][u], plain2[int(m2[u])][0][u]) and torch.equal(
            fused.live[1][u], plain2[int(m2[u])][1][u]
        )
    assert torch.equal(fused.live[0], gated.live[0]) and torch.equal(fused.live[1], blend.live[1])


def test_fused_launch_is_one_op_per_graph():
    """The fused mode's op accounting on the torch model: V and C are one launch each (the model counts them)."""
    B, H, K = 4, 2, 3
    x = _inputs(B, H, seed=77)
    spec = TorchKdaSpecFused(x, K)
    calls = []
    orig = spec._launch

    def counted(slabs, steps, dry):
        calls.append(dry)
        return orig(slabs, steps, dry)

    spec._launch = counted
    slabs = [_slab(x["gen"], B, H) for _ in range(K + 1)]
    spec.verify(slabs)
    spec.commit(torch.tensor([0, 1, 2, 3]))
    assert calls == [True, False], "one dry launch in V, one commit launch in C"


# ------------------------------------------------------------------------------------------------ the loop
class KdaTrackingTargetFused(KdaTrackingTarget):
    """The KDA-state-tracking fake target with the fused torch model."""

    def __init__(self, prompts, K, acc, x, H, seed=0):
        super().__init__(prompts, K, acc, x, H, mode="gated", seed=seed)
        self.spec = TorchKdaSpecFused(x, K)


@pytest.mark.parametrize("K, acc", [(1, 0.6), (3, 0.6), (3, 1.0), (2, 0.0), (3, 0.0)])
def test_loop_commit_semantics_with_fused_kda_state(K, acc):
    """The lossless loop over the fused KDA-tracking target: the tokens equal plain greedy, and after EVERY commit the
    fused live state equals the plain state advanced by the committed tokens (checked inside commit, bits)."""
    prompts = [[1, 2, 3, 4, 5], [9, 8, 7], [4, 4, 4, 4, 4, 4, 4], [2, 7, 1, 8]]
    H = 2
    x = _inputs(len(prompts), H, seed=K * 100 + int(acc * 10))
    target = KdaTrackingTargetFused(prompts, K, acc, x, H, seed=K)
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
    assert target.ref_steps == target.committed_rows >= sum(len(o) for o in outs)
    assert target.spec.bk is None
