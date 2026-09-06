# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the vendored reference. No network, no weights, no device.

This is the backbone of the suite: it runs anywhere, in seconds, at the model's real
dimensions (never shrunk -- only the weights are synthetic). Every test here pins a
property that the TTNN port must independently reproduce, and most are paired with a
NEGATIVE CONTROL: an implementation of the plausible wrong choice, asserted to differ.

A test that only checks the right answer cannot tell you whether the wrong answer would
also have passed. For this architecture that distinction is load-bearing, because several
of the plausible wrong choices raise no error and score high PCC.
"""

import pytest
import torch
import torch.nn.functional as F

from models.experimental.nomic_embed_text_v2_moe.common import (
    build_synthetic_model,
    max_abs_diff,
    pcc,
    random_input_ids,
    synthetic_state_dict,
)
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import load_vendored_config
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import (
    NomicBertModel,
    apply_rotary_emb,
    rotate_half,
)


@pytest.fixture(scope="module")
def cfg():
    return load_vendored_config()


@pytest.fixture(scope="module")
def synthetic_model(cfg):
    return build_synthetic_model(cfg, seed=0)


def _cos_sin(seqlen, rotary_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), inv_freq)
    return freqs.cos(), freqs.sin()


# =======================================================================================
# Rotary embeddings
# =======================================================================================


def test_rotate_half_is_neox_not_interleaved():
    """`(x1, x2) -> (-x2, x1)` on halves, NOT on even/odd lanes."""
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]]))

    # Negative control: the GPT-J / interleaved map is a genuinely different permutation.
    x1, x2 = x[..., ::2], x[..., 1::2]
    interleaved = torch.stack((-x2, x1), dim=-1).flatten(-2)
    assert not torch.allclose(rotate_half(x), interleaved)


def test_rotary_position_zero_is_identity(cfg):
    cos, sin = _cos_sin(16, cfg.rotary_dim)
    x = torch.randn(2, 16, cfg.num_attention_heads, cfg.head_dim)
    out = apply_rotary_emb(x, cos, sin)
    torch.testing.assert_close(out[:, 0], x[:, 0])


def test_rotary_preserves_per_plane_norm(cfg):
    """A rotation preserves the norm of each (i, i + d/2) plane. This is the closed form."""
    cos, sin = _cos_sin(16, cfg.rotary_dim)
    x = torch.randn(2, 16, cfg.num_attention_heads, cfg.head_dim)
    out = apply_rotary_emb(x, cos, sin)

    half = cfg.head_dim // 2
    before = x[..., :half] ** 2 + x[..., half:] ** 2
    after = out[..., :half] ** 2 + out[..., half:] ** 2
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)


def test_rotary_encodes_relative_position(cfg):
    """`<R(m) q, R(n) k>` depends only on `m - n`; that is the whole point of RoPE."""
    cos, sin = _cos_sin(32, cfg.rotary_dim)
    q = torch.randn(1, 32, 1, cfg.head_dim)
    k = torch.randn(1, 32, 1, cfg.head_dim)
    qr = apply_rotary_emb(q.expand(1, 32, 1, cfg.head_dim).clone(), cos, sin)
    kr = apply_rotary_emb(k.expand(1, 32, 1, cfg.head_dim).clone(), cos, sin)

    # Same content at every position, so the score must depend only on the offset.
    q_const = q[:, :1].expand(1, 32, 1, cfg.head_dim).contiguous()
    k_const = k[:, :1].expand(1, 32, 1, cfg.head_dim).contiguous()
    qr_c = apply_rotary_emb(q_const, cos, sin)
    kr_c = apply_rotary_emb(k_const, cos, sin)

    scores = torch.einsum("bshd,bthd->bst", qr_c, kr_c)[0]
    for offset in range(1, 8):
        diag = torch.diagonal(scores, offset=offset)
        torch.testing.assert_close(diag, diag[0].expand_as(diag), rtol=1e-4, atol=1e-4)
    assert torch.isfinite(qr).all() and torch.isfinite(kr).all()


def test_rotary_cos_sin_is_concat_duplicated_not_repeat_interleaved(cfg):
    """NEGATIVE CONTROL for the single most likely rotary bug.

    Upstream caches cos/sin at HALF width and widens with `repeat(c, "... d -> ... 1 (2 d)")`
    = `concat([c, c])`. Using `repeat_interleave` instead gives the GPT-J lane pairing,
    which composed with NeoX `rotate_half` is not a rotation -- but it is finite, plausible,
    and raises nothing.
    """
    cos, sin = _cos_sin(16, cfg.rotary_dim)
    x = torch.randn(2, 16, cfg.num_attention_heads, cfg.head_dim)

    good = apply_rotary_emb(x, cos, sin)
    cos_i = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
    sin_i = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
    bad = x * cos_i + rotate_half(x) * sin_i

    assert pcc(good, bad) < 0.95, "the interleaved oracle must NOT match the NeoX one"
    # And the wrong one is not norm-preserving, i.e. not a rotation at all.
    half = cfg.head_dim // 2
    bad_norm = bad[..., :half] ** 2 + bad[..., half:] ** 2
    good_norm = x[..., :half] ** 2 + x[..., half:] ** 2
    assert max_abs_diff(bad_norm, good_norm) > 1e-3


# =======================================================================================
# Attention: the fused QKV layout
# =======================================================================================


def test_wqkv_is_three_major(cfg):
    """`Wqkv` emits `[q(768) | k(768) | v(768)]`, heads contiguous inside each block.

    Probed with an integer bias and zero weights, so each of q/k/v is a known constant and
    the split is read off directly rather than inferred from a correlation.
    """
    from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicBertAttention

    attn = NomicBertAttention(cfg)
    H = cfg.hidden_size
    with torch.no_grad():
        attn.Wqkv.weight.zero_()
        attn.Wqkv.bias[0:H] = 1.0  # q
        attn.Wqkv.bias[H : 2 * H] = 2.0  # k
        attn.Wqkv.bias[2 * H : 3 * H] = 3.0  # v

    qkv = attn.Wqkv(torch.zeros(1, 4, H)).view(1, 4, 3, cfg.num_attention_heads, cfg.head_dim)
    assert qkv[:, :, 0].unique().tolist() == [1.0]
    assert qkv[:, :, 1].unique().tolist() == [2.0]
    assert qkv[:, :, 2].unique().tolist() == [3.0]

    # NEGATIVE CONTROL: under a head-major reading, `[..., 0, :]` on the three-axis would be
    # "q". It is not -- it straddles the q/k/v block boundaries and picks up all three
    # constants, because heads stride 192 across a layout whose blocks are 768 wide.
    head_major = attn.Wqkv(torch.zeros(1, 4, H)).view(1, 4, cfg.num_attention_heads, 3, cfg.head_dim)
    assert sorted(head_major[..., 0, :].unique().tolist()) == [1.0, 2.0, 3.0]


def test_attention_scale_is_inverse_sqrt_head_dim(cfg):
    """Upstream takes SDPA's default scale; `norm_factor` is only used on the fallback path."""
    assert cfg.head_dim == 64
    assert abs(cfg.head_dim**-0.5 - 0.125) < 1e-12


def test_additive_mask_suppresses_padded_positions(cfg):
    """The extended mask is additive 0 / dtype-min, and it must actually zero attention."""
    from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import build_extended_attention_mask

    keep = torch.tensor([[1, 1, 0, 0]])
    mask = build_extended_attention_mask(keep, torch.float32)
    assert mask.shape == (1, 1, 1, 4)
    torch.testing.assert_close(mask[0, 0, 0, :2], torch.zeros(2))
    assert (mask[0, 0, 0, 2:] < -1e30).all()

    scores = torch.zeros(1, 1, 4, 4)
    probs = torch.softmax(scores + mask, dim=-1)
    assert probs[..., 2:].abs().max() == 0.0


# =======================================================================================
# Post-norm block structure
# =======================================================================================


def test_block_is_post_norm(cfg, synthetic_model):
    """`norm2(mlp(h) + h)` where `h = norm1(attn(x) + x)` -- residual BEFORE the norm.

    Proven by zeroing each sub-block's output projection in turn: with the branch dead the
    block must collapse to the norm of the residual, which a pre-norm block would not do.
    """
    import copy

    block = copy.deepcopy(synthetic_model.encoder.layers[0])
    x = torch.randn(1, 8, cfg.hidden_size)

    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.attn.out_proj.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()

    out = block(x)
    # Both branches dead => norm2(norm1(x)).
    torch.testing.assert_close(out, block.norm2(block.norm1(x)), rtol=1e-5, atol=1e-5)

    # A pre-norm block would instead have returned x unchanged.
    assert not torch.allclose(out, x, atol=1e-3)


def test_moe_placement_predicate(cfg):
    """`i % 2 == 1`: layer 0 dense, layer 1 MoE. The `== 0` off-by-one must not hold."""
    assert cfg.moe_layers == (1, 3, 5, 7, 9, 11)
    assert cfg.dense_layers == (0, 2, 4, 6, 8, 10)
    assert not cfg.is_moe_layer(0)
    assert cfg.is_moe_layer(1)


def test_encoder_alternates_moe_and_dense(cfg, synthetic_model):
    from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicBertMLP, NomicMoELayer

    for i, layer in enumerate(synthetic_model.encoder.layers):
        expected = NomicMoELayer if cfg.is_moe_layer(i) else NomicBertMLP
        assert isinstance(layer.mlp, expected), f"layer {i} has the wrong MLP kind"


# =======================================================================================
# MoE: router
# =======================================================================================


def test_router_softmax_is_over_all_experts_and_topk_is_not_renormalized(cfg, synthetic_model):
    """The routed weights must sum to LESS than 1. This is the architecture's oddity.

    Softmax runs over all 8 experts, then top-2 is taken and used AS IS. Almost every other
    MoE implementation divides by the top-k sum; doing that here attenuates nothing and
    changes every MoE layer's contribution.
    """
    router = synthetic_model.encoder.layers[1].mlp.router
    x = torch.randn(1, 64, cfg.hidden_size)
    weights, top_weights, top_experts = router(x)

    torch.testing.assert_close(weights.sum(-1), torch.ones(weights.shape[0]), rtol=1e-5, atol=1e-5)
    rowsum = top_weights.sum(-1)
    assert (rowsum < 1.0).all(), "top-k weights appear renormalized"
    assert top_experts.shape[-1] == cfg.moe_top_k
    assert int(top_experts.max()) < cfg.num_experts

    # NEGATIVE CONTROL: what the renormalized variant would have produced.
    renormalized = top_weights / top_weights.sum(-1, keepdim=True)
    assert max_abs_diff(renormalized, top_weights) > 1e-2


def test_softmax_over_topk_is_not_equivalent_to_topk_of_softmax(cfg, synthetic_model):
    """NEGATIVE CONTROL: softmaxing the top-k logits implicitly renormalizes."""
    router = synthetic_model.encoder.layers[1].mlp.router
    x = torch.randn(1, 32, cfg.hidden_size)
    _weights, top_weights, _top_experts = router(x)

    logits = router.layer(x.view(-1, cfg.hidden_size))
    top_logits, _ = torch.topk(logits, cfg.moe_top_k, dim=-1)
    wrong = top_logits.softmax(dim=-1)

    torch.testing.assert_close(wrong.sum(-1), torch.ones(wrong.shape[0]), rtol=1e-5, atol=1e-5)
    assert max_abs_diff(wrong, top_weights) > 1e-2


# =======================================================================================
# MoE: experts
# =======================================================================================


def test_expert_shared_bias_is_added_once_after_the_sum(cfg, synthetic_model):
    """THE test that PCC cannot do.

    Adding the shared bias inside the per-expert loop scales it by the routed-weight sum,
    producing an almost-constant offset of `(sum(w) - 1) * bias`. PCC mean-centres before
    correlating, so that offset is very nearly invisible to it -- measured at PCC 0.99999
    on synthetic weights and 0.9999998 on real ones, above ANY usable threshold. Only an
    absolute metric separates them, so this test gates on max-abs.
    """
    experts = synthetic_model.encoder.layers[1].mlp.experts
    router = synthetic_model.encoder.layers[1].mlp.router
    x = torch.randn(1, 48, cfg.hidden_size)
    _w, top_weights, top_experts = router(x)

    correct = experts(x, top_weights, top_experts)

    # The bug: bias folded into each expert's output before weighting.
    flat = x.view(-1, cfg.hidden_size)
    buggy = torch.zeros_like(flat)
    expert_mask = F.one_hot(top_experts, num_classes=cfg.num_experts).permute(2, 1, 0)
    for e in range(cfg.num_experts):
        topk_idx, token_idx = torch.where(expert_mask[e])
        if token_idx.shape[0] == 0:
            continue
        out = (experts.mlp(flat[token_idx], e) + experts.bias) * top_weights[token_idx, topk_idx, None]
        buggy.index_add_(0, token_idx, out)
    buggy = buggy.reshape(x.shape)

    assert pcc(correct, buggy) > 0.99, "if PCC separated these, the point of this test would be moot"
    assert max_abs_diff(correct, buggy) > 1e-3, "max-abs must separate what PCC cannot"

    # And the offset really is the predicted (sum(w) - 1) * bias.
    predicted = (top_weights.sum(-1) - 1.0).reshape(x.shape[0], x.shape[1], 1) * experts.bias
    torch.testing.assert_close(buggy - correct, predicted, rtol=1e-4, atol=1e-4)


def test_expert_loop_equals_dense_all_experts(cfg, synthetic_model):
    """The batched all-experts formulation the TTNN port uses is arithmetically identical.

    This is the bridge between the upstream ragged gather/scatter and the two
    broadcast-batch matmuls that run on device.
    """
    moe = synthetic_model.encoder.layers[1].mlp
    x = torch.randn(1, 64, cfg.hidden_size)
    _w, top_weights, top_experts = moe.router(x)

    dense_weights = torch.zeros(x.shape[0] * x.shape[1], cfg.num_experts).scatter_(1, top_experts, top_weights)

    loop_out = moe.experts(x, top_weights, top_experts)
    dense_out = moe.experts.dense_forward(x, dense_weights)

    assert pcc(loop_out, dense_out) > 0.9999999
    assert max_abs_diff(loop_out, dense_out) < 1e-4


def test_w2_transposed_view_typechecks_but_is_garbage(cfg, synthetic_model):
    """NEGATIVE CONTROL for the silent expert-orientation failure.

    `w2` is `[E * F, H]`. Viewing it `(E, H, F)` instead of `(E, F, H)` succeeds -- the
    element count is symmetric -- and every downstream matmul typechecks. Nothing raises.
    The output is uncorrelated noise.
    """
    moe = synthetic_model.encoder.layers[1].mlp
    E, Fh, H = cfg.num_experts, cfg.intermediate_size, cfg.hidden_size
    x = torch.randn(1, 32, H)
    _w, top_weights, top_experts = moe.router(x)
    dense_weights = torch.zeros(x.shape[0] * x.shape[1], E).scatter_(1, top_experts, top_weights)

    correct = moe.experts.dense_forward(x, dense_weights)

    flat = x.reshape(1, -1, H)
    w1 = moe.experts.mlp.w1.view(E, Fh, H).transpose(1, 2)
    w2_wrong = moe.experts.mlp.w2.view(E, H, Fh)  # the wrong view -- no error
    act = moe.experts.mlp.activation_fn(torch.matmul(flat, w1))
    per_expert = torch.matmul(act, w2_wrong.transpose(1, 2))
    wrong = (per_expert * dense_weights.t().unsqueeze(-1)).sum(0).reshape(x.shape) + moe.experts.bias

    assert abs(pcc(correct, wrong)) < 0.2, "the wrong expert view must be obviously wrong numerically"


def test_expert_axis_is_outer(cfg, synthetic_model):
    """Expert `e` owns rows `e * F : (e+1) * F` -- the expert axis is outer, not inner."""
    mlp = synthetic_model.encoder.layers[1].mlp.experts.mlp
    Fh, H = cfg.intermediate_size, cfg.hidden_size
    for e in (0, 3, 7):
        w1, w2 = mlp.expert_weights(e)
        torch.testing.assert_close(w1, mlp.w1[e * Fh : (e + 1) * Fh])
        torch.testing.assert_close(w2, mlp.w2[e * Fh : (e + 1) * Fh])
        assert w1.shape == (Fh, H)


# =======================================================================================
# Activation
# =======================================================================================


def test_gelu_is_exact_erf_not_tanh():
    """`activation_function == "gelu"` maps to `approximate="none"`.

    The tanh approximation differs by ~5e-4, which is large enough to be mistaken later for
    a device precision problem and small enough to pass a loose PCC gate.
    """
    x = torch.randn(8192) * 3.0
    exact = x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

    # 1e-5 is fp32 rounding slack between two spellings of the same function; the tanh gap
    # below is ~4.7e-4, so the two are still separated by a factor of ~50.
    torch.testing.assert_close(F.gelu(x, approximate="none"), exact, rtol=1e-5, atol=1e-5)
    assert max_abs_diff(F.gelu(x, approximate="tanh"), exact) > 1e-4


# =======================================================================================
# Embeddings and end-to-end shape behaviour
# =======================================================================================


def test_token_type_embedding_is_a_single_row_constant(cfg, synthetic_model):
    """`type_vocab_size == 1`, so the token-type term is one constant vector.

    That is what lets the TTNN port fold it into the word-embedding table at load time
    instead of running a second embedding lookup per forward.
    """
    assert cfg.type_vocab_size == 1
    emb = synthetic_model.embeddings
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

    folded = emb.word_embeddings(input_ids) + emb.token_type_embeddings.weight[0]
    torch.testing.assert_close(emb(input_ids), folded)


def test_no_learned_position_embeddings(synthetic_model):
    """Position is rotary-only; a learned table would be a second, conflicting signal."""
    assert not hasattr(synthetic_model.embeddings, "position_embeddings")


@pytest.mark.parametrize("batch,seqlen", [(1, 1), (1, 4), (2, 8), (3, 17), (1, 128)])
def test_forward_runs_on_small_inputs(cfg, synthetic_model, batch, seqlen):
    """Issue #54917's acceptance criterion: the model runs on small inputs."""
    input_ids, attention_mask = random_input_ids(batch, seqlen, cfg, seed=batch * 100 + seqlen)
    with torch.no_grad():
        out = synthetic_model(input_ids, attention_mask=attention_mask)
    assert out.shape == (batch, seqlen, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_attention_mask_defaults_to_all_ones(cfg, synthetic_model):
    """Upstream raises AttributeError without a mask; the vendored reference must not."""
    input_ids, attention_mask = random_input_ids(2, 8, cfg, seed=1)
    with torch.no_grad():
        implicit = synthetic_model(input_ids)
        explicit = synthetic_model(input_ids, attention_mask=attention_mask)
    torch.testing.assert_close(implicit, explicit)


def test_padding_does_not_leak_into_kept_positions(cfg, synthetic_model):
    """Right-padding a sequence must not change the outputs at the kept positions."""
    seqlen = 16
    input_ids, attention_mask = random_input_ids(1, seqlen, cfg, seed=7)
    keep = 10

    padded_ids = input_ids.clone()
    padded_ids[:, keep:] = cfg.pad_token_id
    padded_mask = attention_mask.clone()
    padded_mask[:, keep:] = 0

    with torch.no_grad():
        full = synthetic_model(input_ids[:, :keep], attention_mask=attention_mask[:, :keep])
        padded = synthetic_model(padded_ids, attention_mask=padded_mask)

    assert pcc(full, padded[:, :keep]) > 0.9999999
    assert max_abs_diff(full, padded[:, :keep]) < 1e-4


def test_strict_load_of_synthetic_state_dict_is_clean(cfg):
    """The generated contract really is exactly the module tree's parameter set."""
    model = NomicBertModel(cfg)
    result = model.load_state_dict(synthetic_state_dict(cfg), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
