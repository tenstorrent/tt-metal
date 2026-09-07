# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the vendored reference. No network, no weights, no device.

Runs at the model's real dimensions on synthetic weights; the model is never shrunk. Each test
pins a property the TTNN port must reproduce independently.

Several tests also implement the plausible wrong choice and assert it differs. For this
architecture that matters: the wrong expert view typechecks, the wrong rotary widening is
finite, and the wrong bias placement scores PCC 0.9999998. Checking only the correct path
would not show whether the wrong one also passes.
"""

import pytest
import torch
import torch.nn.functional as F

from models.common.metrics import compute_max_abs_error, compute_pcc
from models.experimental.nomic_embed_text_v2_moe.common import (
    build_synthetic_model,
    random_input_ids,
    synthetic_state_dict,
)
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import load_vendored_config
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import (
    NomicBertAttention,
    NomicBertMLP,
    NomicBertModel,
    NomicMoELayer,
    apply_rotary_emb,
    build_extended_attention_mask,
    rotate_half,
)

SEQLEN = 16


@pytest.fixture(scope="module")
def cfg():
    return load_vendored_config()


@pytest.fixture(scope="module")
def synthetic_model(cfg):
    return build_synthetic_model(cfg, seed=0)


@pytest.fixture(scope="module")
def moe_layer(cfg, synthetic_model):
    return synthetic_model.encoder.layers[cfg.moe_layers[0]].mlp


def cos_sin(seqlen: int, rotary_dim: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), inv_freq)
    return freqs.cos(), freqs.sin()


def route(moe, x):
    """Router outputs plus the dense routing tensor, which most MoE tests need."""
    _weights, top_weights, top_experts = moe.router(x)
    return top_weights, top_experts, moe.router.dense_weights(top_weights, top_experts)


def test_rotate_half_is_neox_not_interleaved():
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]]))

    x1, x2 = x[..., ::2], x[..., 1::2]
    interleaved = torch.stack((-x2, x1), dim=-1).flatten(-2)
    assert not torch.allclose(rotate_half(x), interleaved)


def test_rotary_position_zero_is_identity(cfg):
    cos, sin = cos_sin(SEQLEN, cfg.rotary_dim, cfg.rotary_emb_base)
    x = torch.randn(2, SEQLEN, cfg.num_attention_heads, cfg.head_dim)
    torch.testing.assert_close(apply_rotary_emb(x, cos, sin)[:, 0], x[:, 0])


def test_rotary_preserves_per_plane_norm(cfg):
    """A rotation preserves the norm of each (i, i + head_dim/2) plane."""
    cos, sin = cos_sin(SEQLEN, cfg.rotary_dim, cfg.rotary_emb_base)
    x = torch.randn(2, SEQLEN, cfg.num_attention_heads, cfg.head_dim)
    out = apply_rotary_emb(x, cos, sin)

    half = cfg.head_dim // 2
    before = x[..., :half] ** 2 + x[..., half:] ** 2
    after = out[..., :half] ** 2 + out[..., half:] ** 2
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)


def test_rotary_encodes_relative_position(cfg):
    """With identical content at every position, the score depends only on the offset."""
    seqlen = 32
    cos, sin = cos_sin(seqlen, cfg.rotary_dim, cfg.rotary_emb_base)
    shape = (1, seqlen, 1, cfg.head_dim)

    q = torch.randn(1, 1, 1, cfg.head_dim).expand(shape).contiguous()
    k = torch.randn(1, 1, 1, cfg.head_dim).expand(shape).contiguous()
    scores = torch.einsum("bshd,bthd->bst", apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin))[0]

    for offset in range(1, 8):
        diag = torch.diagonal(scores, offset=offset)
        torch.testing.assert_close(diag, diag[0].expand_as(diag), rtol=1e-4, atol=1e-4)


def test_rotary_cos_sin_is_concat_duplicated_not_repeat_interleaved(cfg):
    """repeat_interleave gives the GPT-J lane pairing, which with NeoX rotate_half is not a
    rotation. It raises nothing and stays finite."""
    cos, sin = cos_sin(SEQLEN, cfg.rotary_dim, cfg.rotary_emb_base)
    x = torch.randn(2, SEQLEN, cfg.num_attention_heads, cfg.head_dim)

    good = apply_rotary_emb(x, cos, sin)
    bad = x * cos.repeat_interleave(2, dim=-1).unsqueeze(-2) + rotate_half(x) * sin.repeat_interleave(
        2, dim=-1
    ).unsqueeze(-2)

    assert compute_pcc(good, bad) < 0.95

    half = cfg.head_dim // 2
    bad_norm = bad[..., :half] ** 2 + bad[..., half:] ** 2
    good_norm = x[..., :half] ** 2 + x[..., half:] ** 2
    assert compute_max_abs_error(bad_norm, good_norm) > 1e-3


def test_wqkv_is_three_major(cfg):
    """Probed with zero weights and a per-block constant bias, so the split is read off
    directly rather than inferred from a correlation."""
    attn = NomicBertAttention(cfg)
    hidden = cfg.hidden_size
    q_val, k_val, v_val = 1.0, 2.0, 3.0

    with torch.no_grad():
        attn.Wqkv.weight.zero_()
        attn.Wqkv.bias[0:hidden] = q_val
        attn.Wqkv.bias[hidden : 2 * hidden] = k_val
        attn.Wqkv.bias[2 * hidden : 3 * hidden] = v_val

    probe = attn.Wqkv(torch.zeros(1, 4, hidden))
    qkv = probe.view(1, 4, 3, cfg.num_attention_heads, cfg.head_dim)
    assert qkv[:, :, 0].unique().tolist() == [q_val]
    assert qkv[:, :, 1].unique().tolist() == [k_val]
    assert qkv[:, :, 2].unique().tolist() == [v_val]

    # Under a head-major reading the "q" slice would be [..., 0, :]. Heads stride 3 * head_dim
    # across blocks that are hidden_size wide, so it straddles all three constants.
    head_major = probe.view(1, 4, cfg.num_attention_heads, 3, cfg.head_dim)
    assert sorted(head_major[..., 0, :].unique().tolist()) == [q_val, k_val, v_val]


def test_attention_scale_is_inverse_sqrt_head_dim(cfg):
    """Upstream takes SDPA's default scale; norm_factor is only used on the fallback path."""
    assert cfg.attention_scale == pytest.approx(1.0 / cfg.head_dim**0.5)


def test_additive_mask_suppresses_padded_positions(cfg):
    keep = torch.tensor([[1, 1, 0, 0]])
    mask = build_extended_attention_mask(keep, torch.float32)

    assert mask.shape == (1, 1, 1, keep.shape[-1])
    torch.testing.assert_close(mask[0, 0, 0, :2], torch.zeros(2))

    probs = torch.softmax(torch.zeros(1, 1, 4, 4) + mask, dim=-1)
    assert probs[..., 2:].abs().max() == 0.0


def test_block_is_post_norm(cfg, synthetic_model):
    """With both branch outputs zeroed, a post-norm block collapses to norm2(norm1(x)); a
    pre-norm block would return x unchanged."""
    import copy

    block = copy.deepcopy(synthetic_model.encoder.layers[cfg.dense_layers[0]])
    x = torch.randn(1, 8, cfg.hidden_size)

    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.attn.out_proj.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()

    out = block(x)
    torch.testing.assert_close(out, block.norm2(block.norm1(x)), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(out, x, atol=1e-3)


def test_moe_placement_predicate(cfg):
    """`i % every_n == 1`: layer 0 dense, layer 1 MoE. The `== 0` variant is the opposite."""
    assert cfg.moe_layers == (1, 3, 5, 7, 9, 11)
    assert cfg.dense_layers == (0, 2, 4, 6, 8, 10)
    assert not cfg.is_moe_layer(0)


def test_encoder_alternates_moe_and_dense(cfg, synthetic_model):
    for idx, layer in enumerate(synthetic_model.encoder.layers):
        expected = NomicMoELayer if cfg.is_moe_layer(idx) else NomicBertMLP
        assert isinstance(layer.mlp, expected), f"layer {idx} has the wrong MLP kind"


def test_router_topk_weights_are_not_renormalized(cfg, moe_layer):
    """Softmax runs over all experts, then top-k is used as is. Most MoE implementations
    divide by the top-k sum, which is why this needs an explicit test."""
    x = torch.randn(1, 64, cfg.hidden_size)
    weights, top_weights, top_experts = moe_layer.router(x)

    torch.testing.assert_close(weights.sum(-1), torch.ones(weights.shape[0]), rtol=1e-5, atol=1e-5)
    assert (top_weights.sum(-1) < 1.0).all()
    assert top_experts.shape[-1] == cfg.moe_top_k
    assert int(top_experts.max()) < cfg.num_experts

    renormalized = top_weights / top_weights.sum(-1, keepdim=True)
    assert compute_max_abs_error(renormalized, top_weights) > 1e-2


def test_softmax_over_topk_is_not_equivalent_to_topk_of_softmax(cfg, moe_layer):
    """Softmaxing the top-k logits implicitly renormalizes."""
    x = torch.randn(1, 32, cfg.hidden_size)
    _weights, top_weights, _top_experts = moe_layer.router(x)

    logits = moe_layer.router.layer(x.view(-1, cfg.hidden_size))
    wrong = torch.topk(logits, cfg.moe_top_k, dim=-1).values.softmax(dim=-1)

    torch.testing.assert_close(wrong.sum(-1), torch.ones(wrong.shape[0]), rtol=1e-5, atol=1e-5)
    assert compute_max_abs_error(wrong, top_weights) > 1e-2


def test_expert_shared_bias_is_added_once_after_the_sum(cfg, moe_layer):
    """Gating this on max-abs rather than PCC is deliberate: the bug is a near-constant offset
    of (sum(w) - 1) * bias, and PCC mean-centres it away."""
    experts = moe_layer.experts
    x = torch.randn(1, 48, cfg.hidden_size)
    top_weights, top_experts, _dense = route(moe_layer, x)

    correct = experts(x, top_weights, top_experts)

    flat = x.view(-1, cfg.hidden_size)
    buggy = torch.zeros_like(flat)
    expert_mask = F.one_hot(top_experts, num_classes=cfg.num_experts).permute(2, 1, 0)
    for expert_idx in range(cfg.num_experts):
        topk_idx, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.shape[0] == 0:
            continue
        out = (experts.mlp(flat[token_idx], expert_idx) + experts.bias) * top_weights[token_idx, topk_idx, None]
        buggy.index_add_(0, token_idx, out)
    buggy = buggy.reshape(x.shape)

    assert compute_pcc(correct, buggy) > 0.99, "if PCC separated these, this test would be unnecessary"
    assert compute_max_abs_error(correct, buggy) > 1e-3

    predicted_offset = (top_weights.sum(-1) - 1.0).reshape(x.shape[0], x.shape[1], 1) * experts.bias
    torch.testing.assert_close(buggy - correct, predicted_offset, rtol=1e-4, atol=1e-4)


def test_expert_loop_equals_dense_all_experts(cfg, moe_layer):
    """dense_forward is the shape the TTNN port uses, so it must match the upstream loop."""
    x = torch.randn(1, 64, cfg.hidden_size)
    top_weights, top_experts, dense_weights = route(moe_layer, x)

    loop_out = moe_layer.experts(x, top_weights, top_experts)
    dense_out = moe_layer.experts.dense_forward(x, dense_weights)

    assert compute_pcc(loop_out, dense_out) > 0.9999999
    assert compute_max_abs_error(loop_out, dense_out) < 1e-4


def test_w2_transposed_view_typechecks_but_is_garbage(cfg, moe_layer):
    """The element count is symmetric in ffn_hidden and hidden, so the wrong view succeeds and
    every downstream matmul typechecks."""
    experts = moe_layer.experts
    x = torch.randn(1, 32, cfg.hidden_size)
    _top_weights, _top_experts, dense_weights = route(moe_layer, x)

    correct = experts.dense_forward(x, dense_weights)

    w1 = experts.mlp.w1.view(*experts.mlp.expert_shape).transpose(1, 2)
    w2_wrong = experts.mlp.w2.view(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size)
    act = experts.mlp.activation_fn(torch.matmul(x.reshape(1, -1, cfg.hidden_size), w1))
    per_expert = torch.matmul(act, w2_wrong.transpose(1, 2))
    wrong = (per_expert * dense_weights.t().unsqueeze(-1)).sum(0).reshape(x.shape) + experts.bias

    assert abs(compute_pcc(correct, wrong)) < 0.2


def test_expert_axis_is_outer(cfg, moe_layer):
    mlp = moe_layer.experts.mlp
    ffn = cfg.intermediate_size
    for expert_idx in (0, cfg.num_experts // 2, cfg.num_experts - 1):
        w1, w2 = mlp.expert_weights(expert_idx)
        torch.testing.assert_close(w1, mlp.w1[expert_idx * ffn : (expert_idx + 1) * ffn])
        torch.testing.assert_close(w2, mlp.w2[expert_idx * ffn : (expert_idx + 1) * ffn])
        assert w1.shape == (ffn, cfg.hidden_size)


def test_gelu_is_exact_erf_not_tanh():
    """The tanh approximation differs by ~5e-4, which would later read as a device precision
    problem. 1e-5 here is fp32 slack between two spellings of the same function."""
    x = torch.randn(8192) * 3.0
    exact = x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

    torch.testing.assert_close(F.gelu(x, approximate="none"), exact, rtol=1e-5, atol=1e-5)
    assert compute_max_abs_error(F.gelu(x, approximate="tanh"), exact) > 1e-4


def test_token_type_embedding_is_a_single_row_constant(cfg, synthetic_model):
    """type_vocab_size is 1, so the TTNN port can fold this into the word-embedding table at
    load time instead of running a second lookup per forward."""
    assert cfg.type_vocab_size == 1
    emb = synthetic_model.embeddings
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

    torch.testing.assert_close(emb(input_ids), emb.word_embeddings(input_ids) + emb.token_type_embeddings.weight[0])


def test_no_learned_position_embeddings(synthetic_model):
    assert not hasattr(synthetic_model.embeddings, "position_embeddings")


@pytest.mark.parametrize("batch,seqlen", [(1, 1), (1, 4), (2, 8), (3, 17), (1, 128)])
def test_forward_runs_on_small_inputs(cfg, synthetic_model, batch, seqlen):
    input_ids, attention_mask = random_input_ids(batch, seqlen, cfg, seed=batch * 100 + seqlen)
    with torch.no_grad():
        out = synthetic_model(input_ids, attention_mask=attention_mask)

    assert out.shape == (batch, seqlen, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_attention_mask_defaults_to_all_ones(cfg, synthetic_model):
    input_ids, attention_mask = random_input_ids(2, 8, cfg, seed=1)
    with torch.no_grad():
        torch.testing.assert_close(
            synthetic_model(input_ids),
            synthetic_model(input_ids, attention_mask=attention_mask),
        )


def test_padding_does_not_leak_into_kept_positions(cfg, synthetic_model):
    keep = 10
    input_ids, attention_mask = random_input_ids(1, SEQLEN, cfg, seed=7)

    padded_ids = input_ids.clone()
    padded_ids[:, keep:] = cfg.pad_token_id
    padded_mask = attention_mask.clone()
    padded_mask[:, keep:] = 0

    with torch.no_grad():
        full = synthetic_model(input_ids[:, :keep], attention_mask=attention_mask[:, :keep])
        padded = synthetic_model(padded_ids, attention_mask=padded_mask)

    assert compute_pcc(full, padded[:, :keep]) > 0.9999999
    assert compute_max_abs_error(full, padded[:, :keep]) < 1e-4


def test_strict_load_of_synthetic_state_dict_is_clean(cfg):
    """The generated contract is exactly the module tree's parameter set."""
    result = NomicBertModel(cfg).load_state_dict(synthetic_state_dict(cfg), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
