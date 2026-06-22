# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the self-conditioning gated MLP reference (#47461/#47463).

Reconciled against transformers ``DiffusionGemmaSelfConditioning``: the module is
``post_norm(inputs_embeds + down(act(gate(pre_norm(signal))) * up(pre_norm(signal))))``
with a scaleless ``post_norm``, and the soft-embedding feeds it.
"""

import pytest
import torch

from models.experimental.diffusion_gemma.reference.self_conditioning import (
    DiffusionGemmaRMSNorm,
    SelfConditioning,
)


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _embedding(vocab, hidden, seed=0):
    return torch.randn(vocab, hidden, generator=_gen(seed))


def test_module_has_expected_params_and_scaleless_post_norm():
    mod = SelfConditioning(16, intermediate_size=40)
    names = dict(mod.named_parameters())
    # the 4 checkpoint weights: pre_norm + gate/up/down; post_norm is scaleless (no weight)
    assert "pre_norm.weight" in names
    assert {"gate_proj.weight", "up_proj.weight", "down_proj.weight"} <= set(names)
    assert not mod.post_norm.with_scale and not hasattr(mod.post_norm, "weight")
    assert names["gate_proj.weight"].shape == (40, 16)
    assert names["down_proj.weight"].shape == (16, 40)


def test_forward_shape_and_is_post_norm_of_sum():
    batch, length, hidden, inter = 2, 8, 16, 40
    mod = SelfConditioning(hidden, intermediate_size=inter)
    emb = torch.randn(batch, length, hidden, generator=_gen(1))
    signal = torch.randn(batch, length, hidden, generator=_gen(2))
    out = mod(emb, signal)
    assert out.shape == (batch, length, hidden)
    # reproduce the exact composition
    normed = mod.pre_norm(signal)
    sc = mod.down_proj(mod._act(mod.gate_proj(normed)) * mod.up_proj(normed))
    assert torch.allclose(out, mod.post_norm(emb + sc), atol=1e-6)


def test_zero_signal_is_post_norm_of_embeds_not_identity():
    # First denoise step / disabled: zero signal -> post_norm(inputs_embeds), NOT
    # inputs_embeds unchanged (the decoder always post-normalizes its embeddings).
    batch, length, vocab, hidden = 2, 8, 32, 16
    mod = SelfConditioning(hidden, intermediate_size=24)
    emb = torch.randn(batch, length, hidden, generator=_gen(3))
    out_disabled = mod.condition(emb, None, _embedding(vocab, hidden), enabled=False)
    assert torch.allclose(out_disabled, mod.post_norm(emb), atol=1e-6)
    assert torch.allclose(out_disabled, mod(emb, torch.zeros_like(emb)), atol=1e-6)
    # and it is NOT just the input embeddings (post_norm rescales)
    assert not torch.allclose(out_disabled, emb, atol=1e-3)


def test_conditioning_signal_changes_output():
    batch, length, vocab, hidden = 1, 4, 20, 12
    torch.manual_seed(0)
    mod = SelfConditioning(hidden, intermediate_size=18)
    emb = torch.randn(batch, length, hidden, generator=_gen(4))
    logits = torch.randn(batch, length, vocab, generator=_gen(5))
    embed_w = _embedding(vocab, hidden, seed=6)
    conditioned = mod.condition(emb, logits, embed_w, enabled=True)
    disabled = mod.condition(emb, logits, embed_w, enabled=False)
    assert not torch.allclose(conditioned, disabled)


def test_soft_embedding_onehot_recovers_token_row():
    vocab, hidden = 20, 12
    emb = _embedding(vocab, hidden, seed=3)
    logits = torch.full((1, 1, vocab), -1e4)
    logits[..., 7] = 1e4
    soft = SelfConditioning.soft_embedding(logits, emb)
    assert torch.allclose(soft[0, 0], emb[7], atol=1e-4)


def test_soft_embedding_is_convex_combination():
    vocab, hidden = 6, 4
    emb = _embedding(vocab, hidden, seed=4)
    soft = SelfConditioning.soft_embedding(torch.randn(1, 5, vocab, generator=_gen(5)), emb)
    assert torch.all(soft <= emb.max(dim=0).values + 1e-5)
    assert torch.all(soft >= emb.min(dim=0).values - 1e-5)


def test_soft_embedding_per_example_mask_zeroes_signal():
    vocab, hidden = 10, 8
    emb = _embedding(vocab, hidden, seed=7)
    logits = torch.randn(3, 5, vocab, generator=_gen(8))
    mask = torch.tensor([True, False, True])
    soft = SelfConditioning.soft_embedding(logits, emb, mask=mask)
    assert torch.all(soft[1] == 0)  # masked example -> zero signal
    assert torch.any(soft[0] != 0) and torch.any(soft[2] != 0)


def test_rmsnorm_matches_reference_formula():
    x = torch.randn(2, 3, 16, generator=_gen(9))
    n = DiffusionGemmaRMSNorm(16, with_scale=False)
    expected = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
    assert torch.allclose(n(x), expected.type_as(x), atol=1e-6)


def test_activation_silu_supported_and_unknown_rejected():
    SelfConditioning(8, activation="silu")  # ok
    with pytest.raises(ValueError):
        SelfConditioning(8, activation="relu6")._act(torch.zeros(1))


def test_deterministic_under_fixed_weights():
    batch, length, hidden = 1, 4, 8
    torch.manual_seed(0)
    mod = SelfConditioning(hidden, intermediate_size=10)
    emb = torch.randn(batch, length, hidden, generator=_gen(6))
    signal = torch.randn(batch, length, hidden, generator=_gen(7))
    a = mod(emb, signal)
    b = mod(emb.clone(), signal.clone())
    assert torch.equal(a, b)
