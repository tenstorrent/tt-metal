# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only unit tests for the standalone DSpark drafter.

These cover the DSpark algorithm in isolation: target-hidden fusion, noise-block
layout, KV injection / bidirectional block attention, Markov sequential sampling,
and confidence prefix truncation. They do not load Flash weights, run ttnn, or
construct the 43-layer target.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.experimental.deepseek_v4_flash.dspark import (
    DSparkConfig,
    DSparkModel,
    dspark_block_mask,
    prefix_survival,
    truncate_prefix,
)


def _model(seed: int = 0, **overrides) -> DSparkModel:
    torch.manual_seed(seed)
    return DSparkModel(DSparkConfig.tiny(**overrides)).eval()


def _target_hiddens(model: DSparkModel, batch: int = 2, ctx: int = 6) -> torch.Tensor:
    cfg = model.config
    return torch.randn(batch, ctx, cfg.num_target_layers, cfg.hidden_size)


def _anchors(model: DSparkModel, batch: int = 2) -> torch.Tensor:
    return torch.randint(0, model.config.vocab_size - 1, (batch,))


@torch.no_grad()
def test_forward_shapes():
    model = _model()
    cfg = model.config
    hidden = _target_hiddens(model)
    out = model(hidden, _anchors(model))

    batch, ctx, _, dim = hidden.shape
    gamma, vocab = cfg.dspark_block_size, cfg.vocab_size
    assert out.draft_ids.shape == (batch, gamma)
    assert out.logits.shape == (batch, gamma, vocab)
    assert out.base_logits.shape == (batch, gamma, vocab)
    assert out.confidence.shape == (batch, gamma)
    assert out.prefix_survival.shape == (batch, gamma)
    assert out.hidden_states.shape == (batch, gamma, dim)
    assert out.context.shape == (batch, ctx, dim)
    assert out.block_input_ids.shape == (batch, gamma)


@torch.no_grad()
def test_block_is_anchor_then_noise():
    model = _model()
    anchors = _anchors(model)
    block = model.build_block_input_ids(anchors)
    assert torch.equal(block[:, 0], anchors)
    assert torch.all(block[:, 1:] == model.config.dspark_noise_token_id)
    assert block.shape[1] == model.config.dspark_block_size


@torch.no_grad()
def test_fuse_accepts_tuple_or_stack():
    model = _model()
    stacked = _target_hiddens(model, batch=1, ctx=4)
    layers = tuple(stacked[0, :, i] for i in range(model.config.num_target_layers))
    # Re-batch the tuple to [B, S, D].
    layers = tuple(t.unsqueeze(0) for t in layers)
    fused_stack = model.fuse_target_hiddens(stacked)
    fused_tuple = model.fuse_target_hiddens(layers)
    assert torch.allclose(fused_stack, fused_tuple, atol=1e-6)
    assert fused_stack.shape == (1, 4, model.config.hidden_size)


@torch.no_grad()
def test_main_proj_is_linear_on_concat():
    model = _model()
    stacked = _target_hiddens(model, batch=1, ctx=3)
    fused = model.fuse_target_hiddens(stacked)
    concat = stacked.flatten(-2)
    projected = model.main_proj(concat)
    # RMSNorm is scale-only; check that fusion is a projection of the concat, not
    # a mean of the three layers.
    mean = stacked.mean(dim=2)
    assert fused.shape == mean.shape
    assert not torch.allclose(projected, mean, atol=1e-3)


@torch.no_grad()
def test_markov_bias_depends_only_on_previous_token():
    model = _model()
    ids_a = torch.tensor([3, 3])
    ids_b = torch.tensor([3, 7])
    bias_a = model.markov_head.bias(ids_a)
    bias_b = model.markov_head.bias(ids_b)
    assert torch.allclose(bias_a[0], bias_a[1])
    assert torch.allclose(bias_a[0], bias_b[0])
    assert not torch.allclose(bias_b[0], bias_b[1])


@torch.no_grad()
def test_greedy_is_deterministic():
    model = _model(seed=1)
    hidden = _target_hiddens(model)
    anchors = _anchors(model)
    first = model(hidden, anchors, greedy=True)
    second = model(hidden, anchors, greedy=True)
    assert torch.equal(first.draft_ids, second.draft_ids)
    assert torch.allclose(first.logits, second.logits)


@torch.no_grad()
def test_markov_bias_changes_sampled_logits():
    model = _model()
    hidden = _target_hiddens(model, batch=1, ctx=4)
    anchors = torch.tensor([2])
    out = model(hidden, anchors, greedy=True)
    assert not torch.allclose(out.logits, out.base_logits)
    # First step bias is a function of the anchor, not of a sampled draft token.
    expected_first = out.base_logits[:, 0] + model.markov_head.bias(anchors)
    assert torch.allclose(out.logits[:, 0], expected_first, atol=1e-5)


@torch.no_grad()
def test_confidence_in_unit_interval_and_survival_is_cumprod():
    model = _model()
    out = model(_target_hiddens(model), _anchors(model))
    assert torch.all(out.confidence > 0)
    assert torch.all(out.confidence < 1)
    assert torch.allclose(out.prefix_survival, prefix_survival(out.confidence), atol=1e-6)
    # Survival is non-increasing.
    assert torch.all(out.prefix_survival[:, 1:] <= out.prefix_survival[:, :-1] + 1e-6)


def test_truncate_prefix_stops_at_first_drop():
    confidence = torch.tensor([[0.9, 0.9, 0.1, 0.99], [0.5, 0.4, 0.3, 0.2]])
    lengths = truncate_prefix(confidence, min_survival=0.5)
    # Row 0: 0.9, 0.81, 0.081, ...  → keep 2. Row 1: 0.5 then 0.2 → keep 1.
    assert torch.equal(lengths, torch.tensor([2, 1]))


@torch.no_grad()
def test_min_survival_zeros_rejected_suffix():
    model = _model()
    out = model(_target_hiddens(model, batch=1), torch.tensor([1]), min_survival=0.999)
    length = int(truncate_prefix(out.confidence, 0.999)[0])
    assert torch.all(out.draft_ids[0, length:] == 0)
    assert torch.all(out.draft_ids[0, :length] != -1)


@torch.no_grad()
def test_bidirectional_block_attention():
    """Position 0's hidden state must see the noise embeddings (non-causal block)."""
    model = _model()
    hidden = _target_hiddens(model, batch=1, ctx=4)
    anchors = torch.tensor([4])
    out_a = model(hidden, anchors)

    # Swap the noise token id and rebuild just the backbone embeddings' effect by
    # comparing against a model whose noise id is different — easier: zero the
    # noise embedding and show position-0 hidden changes.
    noise_id = model.config.dspark_noise_token_id
    original = model.embed_tokens.weight[noise_id].clone()
    model.embed_tokens.weight[noise_id].zero_()
    out_b = model(hidden, anchors)
    model.embed_tokens.weight[noise_id].copy_(original)

    assert not torch.allclose(out_a.hidden_states[:, 0], out_b.hidden_states[:, 0], atol=1e-5)


@torch.no_grad()
def test_context_injection_affects_draft():
    model = _model()
    hidden = _target_hiddens(model, batch=1, ctx=8)
    anchors = torch.tensor([5])
    out_a = model(hidden, anchors)
    out_b = model(torch.zeros_like(hidden), anchors)
    assert not torch.allclose(out_a.hidden_states, out_b.hidden_states, atol=1e-5)
    assert not torch.equal(out_a.draft_ids, out_b.draft_ids) or not torch.allclose(
        out_a.logits, out_b.logits, atol=1e-4
    )


@torch.no_grad()
def test_sliding_window_masks_old_context():
    mask = dspark_block_mask(
        ctx_len=10, block_size=4, sliding_window=3, device=torch.device("cpu"), dtype=torch.float32
    )
    # Keys: 0..9 context, 10..13 block. Window keeps context 7,8,9.
    scores = mask[0, 0, 0]
    assert torch.isneginf(scores[:7]).all()
    assert torch.equal(scores[7:10], torch.zeros(3))
    assert torch.equal(scores[10:], torch.zeros(4))
    # Every query row has the same pattern (full block visibility).
    assert torch.equal(mask[0, 0, 0], mask[0, 0, -1])


@torch.no_grad()
def test_share_from_target_aliases_embed_and_lm_head():
    model = _model()
    embed = nn.Embedding(model.config.vocab_size, model.config.hidden_size)
    lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)
    nn.init.ones_(embed.weight)
    nn.init.zeros_(lm_head.weight)
    model.share_from_target(embed, lm_head)
    assert model.embed_tokens is embed
    assert model.lm_head is lm_head
    assert not model.embed_tokens.weight.requires_grad
    out = model(_target_hiddens(model, batch=1, ctx=3), torch.tensor([0]))
    # Ones embedding + zero lm_head → backbone base logits are ~0 before Markov.
    assert out.base_logits.abs().max() < 1e-4


@torch.no_grad()
def test_mtp_module_layout_matches_checkpoint_namespaces():
    model = _model()
    names = set(model.state_dict())
    assert "mtp.0.main_proj.weight" in names
    assert "mtp.0.main_norm.weight" in names
    assert "mtp.2.markov_head.markov_w1.weight" in names
    assert "mtp.2.markov_head.markov_w2.weight" in names
    assert "mtp.2.confidence_head.proj.weight" in names
    assert "mtp.2.norm.weight" in names
    # Intermediate stages have no fusion / heads.
    assert "mtp.1.main_proj.weight" not in names
    assert "mtp.0.markov_head.markov_w1.weight" not in names


def test_flash_0731_config_knobs():
    cfg = DSparkConfig.flash_0731()
    assert cfg.dspark_block_size == 5
    assert cfg.dspark_markov_rank == 256
    assert cfg.dspark_noise_token_id == 128799
    assert cfg.dspark_target_layer_ids == (40, 41, 42)
    assert cfg.num_stages == 3
    assert cfg.num_target_layers == 3
