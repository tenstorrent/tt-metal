# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1: the vendored torch reference against the upstream HuggingFace math it was trimmed from.

Same weights on both sides (copied out of the HF model, so ``_init_weights``' ``A_log`` draw and
ones ``dt_bias`` are what both see), same inputs, reduced config. These are two implementations of
one formula, so the bar is near-identity (0.9999+), not the spec's model-level PCC: any real gap
here is a transcription bug, and a transcription bug in the Gated DeltaNet is exactly the kind that
hides behind a plausible number later.

Host only — no ttnn, no checkpoint.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.qwen_3_8_27b_d_p.reference.config import FULL_ATTENTION, LINEAR_ATTENTION
from models.demos.qwen_3_8_27b_d_p.reference.modeling import (
    REF_DTYPE,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
    Qwen35RotaryEmbedding,
    Qwen35TextModel,
    fp16_accumulation,
    l2norm,
    torch_chunk_gated_delta_rule,
)
from models.demos.qwen_3_8_27b_d_p.tests.host.hf_bridge import build_hf_model, reduced_config

# Two torch implementations of the same formula: anything below this is a transcription bug.
IDENTITY_PCC = 0.9999
SEQ = 192  # 3 delta-rule chunks of 64, and not a multiple of the mrope period


@pytest.fixture(autouse=True)
def literal_fp16_accumulation():
    """Every comparison in this file runs the reference with LITERAL fp16 accumulation.

    The reference's default is fp16 storage with fp32 accumulation, because torch has no
    vectorised fp16 GEMM on x86 and the scalar fallback is ~3500x slower — unusable for a
    full-depth golden trace. Here the point is op-for-op parity with upstream, and the reduced
    config is small enough that the slow path costs seconds. ``test_accumulation_modes_agree``
    measures what the two modes differ by.
    """
    with fp16_accumulation():
        yield


@pytest.fixture(scope="module")
def cfg():
    return reduced_config()


@pytest.fixture(scope="module")
def hf_model(cfg):
    return build_hf_model(cfg)


@pytest.fixture(scope="module")
def ref_model(cfg, hf_model):
    """This package's reference, loaded from the HF model's own state dict.

    The key names match one-for-one by construction (the reference was trimmed, not rewritten), so
    ``strict=True`` is itself part of the test: a renamed or dropped parameter fails here rather
    than silently running on a random init.
    """
    model = Qwen35TextModel(cfg).to(REF_DTYPE).eval()
    missing, unexpected = model.load_state_dict(hf_model.state_dict(), strict=False)
    assert not unexpected, f"HF has parameters the reference does not: {unexpected}"
    assert set(missing) == {"lm_head.weight"}, f"reference has unmatched parameters: {missing}"
    torch.manual_seed(1234)
    with torch.no_grad():
        model.lm_head.weight.normal_(0.0, 0.02)
    return model


@pytest.fixture(scope="module")
def hidden(cfg):
    torch.manual_seed(7)
    return torch.randn(1, SEQ, cfg.hidden_size, dtype=REF_DTYPE)


def _check(name: str, expected: torch.Tensor, actual: torch.Tensor, pcc: float = IDENTITY_PCC) -> None:
    assert expected.shape == actual.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    passing, value = comp_pcc(expected.float(), actual.float(), pcc)
    logger.info(f"{name}: {value}")
    assert passing, f"{name} PCC fail: {value}"


# ======================================================================================
# Primitives
# ======================================================================================


def test_rms_norm_vs_hf(cfg, hf_model, hidden):
    ref = Qwen35RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(REF_DTYPE)
    ref.load_state_dict(hf_model.layers[0].input_layernorm.state_dict())
    with torch.no_grad():
        _check("rms_norm", hf_model.layers[0].input_layernorm(hidden), ref(hidden))


def test_gated_rms_norm_vs_hf(cfg, hf_model):
    """The silu-gated GDN output norm. A sigmoid gate instead of silu still produces a smooth,
    plausible tensor — hence an explicit test rather than trusting the composition."""
    hf_norm = hf_model.layers[0].linear_attn.norm
    ref = Qwen35RMSNormGated(cfg.linear_value_head_dim, cfg.rms_norm_eps).to(REF_DTYPE)
    ref.load_state_dict(hf_norm.state_dict())
    torch.manual_seed(3)
    x = torch.randn(SEQ * cfg.linear_num_value_heads, cfg.linear_value_head_dim, dtype=REF_DTYPE)
    z = torch.randn_like(x)
    with torch.no_grad():
        _check("gated_rms_norm", hf_norm(x, z), ref(x, z))


def test_gated_rms_norm_is_not_sigmoid_gated(cfg, hf_model):
    """Guard-rail: pin that the gate is silu, so a later swap to a sigmoid-gated kernel fails loudly."""
    ref = Qwen35RMSNormGated(cfg.linear_value_head_dim, cfg.rms_norm_eps).to(REF_DTYPE)
    ref.load_state_dict(hf_model.layers[0].linear_attn.norm.state_dict())
    torch.manual_seed(3)
    x = torch.randn(64, cfg.linear_value_head_dim, dtype=REF_DTYPE)
    z = torch.randn_like(x)
    with torch.no_grad():
        silu_gated = ref(x, z)
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
        sigmoid_gated = (ref.weight * h.to(REF_DTYPE)) * torch.sigmoid(z.float())
    _, pcc = comp_pcc(silu_gated.float(), sigmoid_gated.float(), 0.0)
    assert float(pcc) < 0.99, f"silu and sigmoid gating are indistinguishable at PCC {pcc} — bad test"


def test_rope_vs_hf(cfg, hf_model):
    ref = Qwen35RotaryEmbedding(cfg)
    position_ids = torch.arange(SEQ)[None, :]
    with torch.no_grad():
        hf_cos, hf_sin = hf_model.rotary_emb(torch.zeros(1, SEQ, cfg.hidden_size, dtype=REF_DTYPE), position_ids)
        cos, sin = ref(position_ids)
    _check("rope_cos", hf_cos, cos)
    _check("rope_sin", hf_sin, sin)
    assert cos.shape[-1] == cfg.rotary_dim == 64


def test_text_only_mrope_reduces_to_plain_rope(cfg):
    """For a text prompt the T/H/W position rows are identical, so interleaved mrope is a no-op and
    cos/sin equal a plain partial RoPE table. The TT side builds the plain table; this is the
    equivalence that licenses it, checked rather than assumed."""
    ref = Qwen35RotaryEmbedding(cfg)
    position_ids = torch.arange(SEQ)[None, :]
    cos, sin = ref(position_ids)
    inv_freq = ref.inv_freq
    freqs = torch.outer(torch.arange(SEQ).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)[None]
    _check("plain_rope_cos", emb.cos().to(REF_DTYPE), cos, pcc=0.99999)
    _check("plain_rope_sin", emb.sin().to(REF_DTYPE), sin, pcc=0.99999)


def test_mlp_vs_hf(cfg, hf_model, ref_model, hidden):
    with torch.no_grad():
        _check("mlp", hf_model.layers[0].mlp(hidden), ref_model.layers[0].mlp(hidden))


def test_l2norm_matches_fla_form():
    """FLA puts eps INSIDE the rsqrt (added to the sum of squares). The x/(||x||+eps) spelling is a
    different function and drifts most on the small-norm rows the delta rule is most sensitive to."""
    torch.manual_seed(5)
    # Scaled so the row's sum of squares is the same order as eps — the regime where the two
    # spellings diverge most, and the one small-norm conv outputs actually land in.
    x = torch.randn(32, 128, dtype=torch.float32) * 1e-4
    ours = l2norm(x, dim=-1, eps=1e-6)
    inside = x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    outside = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    assert torch.equal(ours, inside)
    rel = ((ours - outside).norm() / ours.norm()).item()
    assert rel > 0.01, f"the two l2norm spellings differ by only {rel:.2%} here — bad test regime"


# ======================================================================================
# Gated DeltaNet
# ======================================================================================


def test_chunk_gated_delta_rule_vs_hf(cfg):
    """The scan kernel itself, against upstream's ``torch_chunk_gated_delta_rule``, on the same
    random q/k/v/g/beta. Decoupled from the projections so a failure localises."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as hf_chunk_rule

    torch.manual_seed(11)
    hv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    q = torch.randn(1, SEQ, hv, dk, dtype=REF_DTYPE)
    k = torch.randn(1, SEQ, hv, dk, dtype=REF_DTYPE)
    v = torch.randn(1, SEQ, hv, dv, dtype=REF_DTYPE)
    beta = torch.rand(1, SEQ, hv, dtype=REF_DTYPE)
    g = -torch.rand(1, SEQ, hv, dtype=torch.float32) * 0.5

    with torch.no_grad():
        hf_o, hf_s = hf_chunk_rule(
            q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        o, s = torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
    _check("gdn_scan_out", hf_o, o)
    _check("gdn_scan_state", hf_s, s)


def test_chunk_gated_delta_rule_carries_initial_state(cfg):
    """Splitting the sequence and threading the state must reproduce the one-shot scan exactly.

    This is the property chunked prefill rests on for the 48 GDN layers — the analogue of "chunk N
    attending the prefix left in the KV cache" for the 16 attention ones.
    """
    torch.manual_seed(13)
    hv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    q = torch.randn(1, SEQ, hv, dk, dtype=REF_DTYPE)
    k = torch.randn(1, SEQ, hv, dk, dtype=REF_DTYPE)
    v = torch.randn(1, SEQ, hv, dv, dtype=REF_DTYPE)
    beta = torch.rand(1, SEQ, hv, dtype=REF_DTYPE)
    g = -torch.rand(1, SEQ, hv, dtype=torch.float32) * 0.5

    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    with torch.no_grad():
        full, full_state = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, **kw)
        half = SEQ // 2
        o0, s0 = torch_chunk_gated_delta_rule(
            q[:, :half], k[:, :half], v[:, :half], g=g[:, :half], beta=beta[:, :half], initial_state=None, **kw
        )
        o1, s1 = torch_chunk_gated_delta_rule(
            q[:, half:], k[:, half:], v[:, half:], g=g[:, half:], beta=beta[:, half:], initial_state=s0, **kw
        )
    _check("gdn_scan_chunked_out", full, torch.cat([o0, o1], dim=1), pcc=0.99999)
    _check("gdn_scan_chunked_state", full_state, s1, pcc=0.99999)


def test_gated_deltanet_layer_vs_hf(cfg, hf_model, ref_model, hidden):
    """The whole GDN token mixer: projections -> causal conv -> gates -> scan -> gated norm -> out_proj."""
    idx = cfg.linear_attention_layers[0]
    with torch.no_grad():
        expected = hf_model.layers[idx].linear_attn(hidden_states=hidden, cache_params=None, attention_mask=None)
        actual, state = ref_model.layers[idx].linear_attn(hidden)
    _check("gated_deltanet", expected, actual)
    assert state.conv_state.shape == (1, cfg.gdn_conv_dim, cfg.linear_conv_kernel_dim - 1)
    assert state.recurrent_state.shape == (
        1,
        cfg.linear_num_value_heads,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
    )


def test_gated_deltanet_causal_conv_vs_hf(cfg, hf_model, ref_model, hidden):
    """The 4-tap causal depthwise conv + SiLU alone, in isolation from the scan."""
    idx = cfg.linear_attention_layers[0]
    hf_gdn, ref_gdn = hf_model.layers[idx].linear_attn, ref_model.layers[idx].linear_attn
    with torch.no_grad():
        mixed = ref_gdn.in_proj_qkv(hidden).transpose(1, 2)
        expected = torch.nn.functional.silu(hf_gdn.conv1d(mixed)[:, :, : mixed.shape[-1]])
        actual = ref_gdn.causal_conv(mixed, None)
    _check("gdn_causal_conv", expected, actual)


def test_gated_deltanet_conv_is_causal(cfg, ref_model):
    """Perturbing token t must not change conv outputs before t. Catches an off-by-one in the
    padding trim, which otherwise leaks one token of future into every GDN layer."""
    idx = cfg.linear_attention_layers[0]
    gdn = ref_model.layers[idx].linear_attn
    torch.manual_seed(17)
    x = torch.randn(1, gdn.conv_dim, 64, dtype=REF_DTYPE)
    t = 40
    y = x.clone()
    y[:, :, t] += 5.0
    with torch.no_grad():
        a, b = gdn.causal_conv(x, None), gdn.causal_conv(y, None)
    assert torch.equal(a[:, :, :t], b[:, :, :t]), "conv output before the perturbed token changed"
    assert not torch.equal(a[:, :, t:], b[:, :, t:]), "conv output at/after the perturbed token did not change"


def test_gated_deltanet_gates_vs_hf(cfg, hf_model, ref_model, hidden):
    """beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias), the latter in fp32."""
    idx = cfg.linear_attention_layers[0]
    hf_gdn, ref_gdn = hf_model.layers[idx].linear_attn, ref_model.layers[idx].linear_attn
    with torch.no_grad():
        expected_beta = hf_gdn.in_proj_b(hidden).sigmoid()
        a = hf_gdn.in_proj_a(hidden)
        expected_g = -hf_gdn.A_log.float().exp() * torch.nn.functional.softplus(a.float() + hf_gdn.dt_bias)
        beta, g = ref_gdn.gates(hidden)
    _check("gdn_beta", expected_beta, beta)
    _check("gdn_g", expected_g, g)
    assert g.dtype == torch.float32, "the log-decay must stay fp32 (fp16 exp(A_log) can reach -inf)"
    assert (g <= 0).all(), "the log-decay must be non-positive"


def test_gated_deltanet_head_expansion(cfg, ref_model):
    """q/k are repeat_interleaved, not tiled: value head hv reads key head hv // 3. Getting this
    backwards (``repeat``) is a permutation of the heads that still produces a full-rank result."""
    idx = cfg.linear_attention_layers[0]
    gdn = ref_model.layers[idx].linear_attn
    torch.manual_seed(19)
    conv_out = torch.randn(1, gdn.conv_dim, 64, dtype=REF_DTYPE)
    q, k, v = gdn.split_heads(conv_out)
    assert q.shape == (1, 64, cfg.linear_num_value_heads, cfg.linear_key_head_dim)
    assert v.shape == (1, 64, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
    g = cfg.gdn_num_value_groups
    for hv in range(cfg.linear_num_value_heads):
        assert torch.equal(q[:, :, hv], q[:, :, (hv // g) * g]), f"value head {hv} is not grouped with hv//{g}"


# ======================================================================================
# Full attention
# ======================================================================================


def test_attention_vs_hf(cfg, hf_model, ref_model, hidden):
    idx = cfg.full_attention_layers[0]
    position_ids = torch.arange(SEQ)[None, :]
    with torch.no_grad():
        cos, sin = ref_model.rotary_emb(position_ids)
        mask = torch.zeros(1, 1, SEQ, SEQ, dtype=REF_DTYPE)
        mask.masked_fill_(~torch.ones(SEQ, SEQ, dtype=torch.bool).tril(), torch.finfo(REF_DTYPE).min)
        expected, _ = hf_model.layers[idx].self_attn(
            hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=mask, past_key_values=None
        )
        actual, capture = ref_model.layers[idx].self_attn(hidden, (cos, sin))
    _check("attention", expected, actual)
    assert capture.key.shape == (1, cfg.num_key_value_heads, SEQ, cfg.head_dim)
    assert capture.value.shape == capture.key.shape


def test_attention_output_gate_is_live(cfg, ref_model, hidden):
    """q_proj is 2x wide and the second half gates the attention output. Dropping the gate leaves a
    perfectly well-formed attention block, so pin that it actually changes the answer."""
    idx = cfg.full_attention_layers[0]
    attn = ref_model.layers[idx].self_attn
    assert attn.q_proj.weight.shape[0] == cfg.num_attention_heads * cfg.head_dim * 2
    position_ids = torch.arange(SEQ)[None, :]
    with torch.no_grad():
        cos, sin = ref_model.rotary_emb(position_ids)
        gated, _ = attn(hidden, (cos, sin))
        q, k, v, gate = attn.project(hidden)
        assert gate.shape == (1, SEQ, cfg.num_attention_heads * cfg.head_dim)
        attn.cfg = cfg  # unchanged; the ungated variant is computed by hand below
        ungated = attn.o_proj(_ungated_attention(attn, q, k, v, cos, sin))
    _, pcc = comp_pcc(gated.float(), ungated.float(), 0.0)
    assert float(pcc) < 0.999, f"the output gate barely changes the result (PCC {pcc}) — bad test"


def _ungated_attention(attn, q, k, v, cos, sin) -> torch.Tensor:
    from models.demos.qwen_3_8_27b_d_p.reference.modeling import apply_rotary_pos_emb, repeat_kv

    b, _, s, _ = q.shape
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    ks = repeat_kv(k, attn.num_key_value_groups)
    vs = repeat_kv(v, attn.num_key_value_groups)
    scores = torch.matmul(q.float(), ks.float().transpose(2, 3)) * attn.scaling
    scores = scores.masked_fill(~torch.ones(s, s, dtype=torch.bool).tril(), float("-inf"))
    scores = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(scores, vs).transpose(1, 2).reshape(b, s, -1)


def test_partial_rope_leaves_the_tail_untouched(cfg, ref_model, hidden):
    """Only the first 64 of each 256-wide head is rotated; a full rotation is a silent PCC loss."""
    from models.demos.qwen_3_8_27b_d_p.reference.modeling import apply_rotary_pos_emb

    idx = cfg.full_attention_layers[0]
    attn = ref_model.layers[idx].self_attn
    position_ids = torch.arange(SEQ)[None, :]
    with torch.no_grad():
        cos, sin = ref_model.rotary_emb(position_ids)
        q, k, _, _ = attn.project(hidden)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.equal(q_rot[..., cfg.rotary_dim :], q[..., cfg.rotary_dim :])
    assert torch.equal(k_rot[..., cfg.rotary_dim :], k[..., cfg.rotary_dim :])
    assert not torch.equal(q_rot[..., : cfg.rotary_dim], q[..., : cfg.rotary_dim])


# ======================================================================================
# Layer and whole model
# ======================================================================================


@pytest.mark.parametrize("layer_type", [LINEAR_ATTENTION, FULL_ATTENTION])
def test_decoder_layer_vs_hf(cfg, hf_model, ref_model, hidden, layer_type):
    idx = (cfg.linear_attention_layers if layer_type == LINEAR_ATTENTION else cfg.full_attention_layers)[0]
    position_ids = torch.arange(SEQ)[None, :]
    with torch.no_grad():
        cos, sin = ref_model.rotary_emb(position_ids)
        mask = torch.zeros(1, 1, SEQ, SEQ, dtype=REF_DTYPE)
        mask.masked_fill_(~torch.ones(SEQ, SEQ, dtype=torch.bool).tril(), torch.finfo(REF_DTYPE).min)
        expected = hf_model.layers[idx](
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=None if layer_type == LINEAR_ATTENTION else mask,
            position_ids=position_ids,
            past_key_values=None,
        )
        actual, _ = ref_model.layers[idx](hidden, (cos, sin))
    _check(f"decoder_layer[{layer_type}]", expected, actual)


def test_whole_model_vs_hf(cfg, hf_model, ref_model):
    """All 8 hybrid layers end to end, from token ids. The composition test: catches a layer-type
    dispatch error or a residual-order slip that the per-block tests cannot see."""
    torch.manual_seed(23)
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ))
    with torch.no_grad():
        expected = hf_model(input_ids=input_ids, use_cache=False).last_hidden_state
        actual, states = ref_model(input_ids=input_ids, skip_lm_head=True)
    _check("whole_model", expected, actual)
    assert len(states) == cfg.num_hidden_layers


def test_whole_model_chunked_equals_one_shot(cfg, ref_model):
    """Two chunks threaded through the carried state reproduce the one-shot forward.

    Both mechanisms have to hold simultaneously for this to pass: the GQA layers' K/V prefix AND
    the GDN layers' (conv_state, recurrent_state). It is the host-side statement of P2's goal, so
    a failure here is a reference bug rather than a device one.
    """
    torch.manual_seed(29)
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ))
    half = SEQ // 2
    with torch.no_grad():
        one_shot, _ = ref_model(input_ids=input_ids, skip_lm_head=True)
        h0, states = ref_model(input_ids=input_ids[:, :half], skip_lm_head=True)
        h1, _ = ref_model(input_ids=input_ids[:, half:], start_pos=half, states=states, skip_lm_head=True)
    _check("chunked_model", one_shot, torch.cat([h0, h1], dim=1), pcc=0.999)


def test_accumulation_modes_agree(cfg, ref_model, literal_fp16_accumulation):
    """fp32-accumulated vs literal-fp16-accumulated reference, on the whole reduced model.

    The golden trace is generated with fp32 accumulation; the parity tests above run with fp16.
    This is the bridge between them — it puts a number on the gap rather than leaving it implied.
    Both are legitimate fp16 references; fp32 accumulation is the one a real fp16 GEMM (GPU tensor
    cores, and the Tenstorrent device under HiFi4 + fp32_dest_acc_en) actually performs.
    """
    from models.demos.qwen_3_8_27b_d_p.reference.modeling import _FP32_ACCUM  # noqa: F401

    torch.manual_seed(37)
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ))
    with torch.no_grad():
        fp16_acc, _ = ref_model(input_ids=input_ids, skip_lm_head=True)
    import models.demos.qwen_3_8_27b_d_p.reference.modeling as modeling

    saved = modeling._FP32_ACCUM
    modeling._FP32_ACCUM = True
    try:
        with torch.no_grad():
            fp32_acc, _ = ref_model(input_ids=input_ids, skip_lm_head=True)
    finally:
        modeling._FP32_ACCUM = saved
    _check("accumulation_modes", fp16_acc, fp32_acc, pcc=0.99)


def test_gdn_scan_chunk_size_is_exact(cfg):
    """chunk 32 (what the device op runs) against chunk 64 (upstream's default).

    The chunked gated delta rule is an exact factorisation of the recurrence, not an
    approximation, so the tiling width must not change the answer. The device is forced to 32 by
    the flat q/k/v path's in-kernel L2 norm; this is what licenses comparing its output against a
    reference that runs at 64.
    """
    torch.manual_seed(41)
    hv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    q = torch.randn(1, SEQ, hv, dk, dtype=REF_DTYPE)
    k = torch.randn(1, SEQ, hv, dk, dtype=REF_DTYPE)
    v = torch.randn(1, SEQ, hv, dv, dtype=REF_DTYPE)
    beta = torch.rand(1, SEQ, hv, dtype=REF_DTYPE)
    g = -torch.rand(1, SEQ, hv, dtype=torch.float32) * 0.5

    kw = dict(initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    with torch.no_grad():
        o32, s32 = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=32, **kw)
        o64, s64 = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=64, **kw)
    _check("gdn_scan_chunk32_vs_64_out", o64, o32, pcc=0.99999)
    _check("gdn_scan_chunk32_vs_64_state", s64, s32, pcc=0.99999)
