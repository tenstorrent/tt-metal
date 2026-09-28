# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of optimizer_lazy_weights: a replayed recipe equals eager torch exactly, a recipe key
changes with anything that changes the bytes, and whatever deferral cannot reproduce taints the build."""

import pytest
import torch
import torch.nn.functional as F

from models.tt_transformers.tests import optimizer_lazy_weights as lazy


@pytest.fixture(autouse=True)
def _fresh_state():
    lazy.reset()
    yield
    lazy.reset()


def _sources(**tensors):
    return lazy.wrap_state_dict(dict(tensors))


def _merged_gate_up(w1, w3, devices=4, pad_quant=32 * 8 * 3):
    """The shape of the model's merged, zero-padded gate/up weight (mlp.py), at a small size."""
    a, b = torch.transpose(w1, -2, -1), torch.transpose(w3, -2, -1)
    k = a.shape[0]
    packed = torch.cat([a.reshape(k, devices, -1), b.reshape(k, devices, -1)], dim=-1)
    n = packed.shape[-1]
    padded = ((n + pad_quant - 1) // pad_quant) * pad_quant
    return F.pad(packed, (0, padded - n)).reshape(k, -1).unsqueeze(0).unsqueeze(0)


def _patterns(t):
    return [
        t.to(torch.bfloat16),
        torch.chunk(t, 4, dim=0)[2],
        t[3:9, ::2],
        t * 0.5 + 1,
        t.view(8, 8, 128).transpose(1, 2).reshape(64, 128).contiguous(),
        t.float().T,
        torch.stack([t, t]),
        t.reshape(2, 32, 128).permute(1, 0, 2),
        torch.split(t, [16, 48])[1],
        t.unsqueeze(0).expand(3, 64, 128),
        torch.cat([t, torch.zeros(64, 32)], dim=-1),
    ]


def test_replay_equals_eager_for_the_ops_models_use():
    x = torch.randn(64, 128)
    s = _sources(x=x)["x"]
    for eager, deferred in zip(_patterns(x), _patterns(s)):
        assert lazy.is_lazy(deferred)
        replayed = lazy.materialize(deferred)
        assert torch.equal(replayed, eager)
        assert replayed.stride() == eager.stride() == deferred.stride()
    assert not lazy.STATE.eager and lazy.STATE.taint is None


def test_merged_padded_weight_is_recorded_not_computed():
    w1, w3 = torch.randn(512, 256, dtype=torch.bfloat16), torch.randn(512, 256, dtype=torch.bfloat16)
    s = _sources(w1=w1, w3=w3)
    deferred = _merged_gate_up(s["w1"], s["w3"])
    assert lazy.is_lazy(deferred) and not lazy.STATE.eager
    assert torch.equal(lazy.materialize(deferred), _merged_gate_up(w1, w3))


def test_recipe_digest_is_stable_and_tracks_every_input():
    w1, w3 = torch.randn(512, 256), torch.randn(512, 256)
    first = _merged_gate_up(*_sources(w1=w1, w3=w3).values()).recipe_digest()
    assert _merged_gate_up(*_sources(w1=w1.clone(), w3=w3.clone()).values()).recipe_digest() == first
    assert _merged_gate_up(*_sources(w1=w1, w3=w3).values(), pad_quant=32 * 8 * 2).recipe_digest() != first
    assert _merged_gate_up(*_sources(w1=w1, w3=w3).values(), devices=2).recipe_digest() != first
    changed = w1.clone()
    changed[7, 3] += 1
    assert _merged_gate_up(*_sources(w1=changed, w3=w3).values()).recipe_digest() != first
    s = _sources(x=torch.randn(8, 8))["x"]
    assert (s * 0.5).recipe_digest() != (s * 0.25).recipe_digest()
    assert s.to(torch.bfloat16).recipe_digest() != s.to(torch.float16).recipe_digest()


def test_an_ordinary_argument_is_copied_into_the_recipe():
    x, pad = torch.randn(4, 4), torch.zeros(4, 2)
    deferred = torch.cat([_sources(x=x)["x"], pad], dim=-1)
    pad.fill_(5.0)  # a later write to the ordinary tensor must not reach the recipe
    assert torch.equal(lazy.materialize(deferred), torch.cat([x, torch.zeros(4, 2)], dim=-1))


def test_reads_and_copies_out_run_eagerly_and_stay_exact():
    x = torch.randn(64, 128)
    s = _sources(x=x)["x"]
    assert s.sum().item() == x.sum().item()
    assert s[0, :3].tolist() == x[0, :3].tolist()
    assert torch.equal(torch.empty(64, 128).copy_(s), x)
    emb = torch.nn.Embedding(64, 128)
    emb.load_state_dict({"weight": s})
    assert torch.equal(emb.weight.data, x)
    assert torch.equal(lazy.materialize(torch.rand_like(s).mul(0) + s), x)  # random op: eager
    assert lazy.STATE.taint is None


@pytest.mark.parametrize(
    "escape",
    [
        lambda t: t.clone().mul_(2),
        lambda t: t.clone().__setitem__(0, 0.0),
        lambda t: t.numpy(),
        lambda t: t.data_ptr(),
    ],
    ids=["inplace", "setitem", "numpy", "data_ptr"],
)
def test_writes_and_memory_escapes_taint_the_build(escape):
    s = _sources(x=torch.randn(8, 8))["x"]
    with pytest.raises(lazy.LazyUnsupported):
        escape(s)
    assert lazy.STATE.taint


def test_an_op_run_eagerly_that_returns_checkpoint_memory_taints():
    s = _sources(x=torch.randn(8, 8))["x"]
    with pytest.raises(lazy.LazyUnsupported):  # as if aten.alias had no meta kernel
        lazy._eager(torch.ops.aten.alias.default, (s,), {}, "test")
    assert lazy.STATE.taint
    lazy.reset()
    s = _sources(x=torch.randn(8, 8))["x"]
    assert torch.equal(lazy._eager(torch.ops.aten.clone.default, (s,), {}, "test"), lazy.materialize(s))
    assert lazy.STATE.taint is None


def test_the_memory_of_a_derived_view_is_never_exposed():
    s = _sources(x=torch.randn(8, 8))["x"]
    # the alias is recorded like any view; asking for its storage would hand out checkpoint memory
    with pytest.raises(lazy.LazyUnsupported):
        torch.ops.aten.alias.default(s).untyped_storage()
    assert lazy.STATE.taint
