# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only unit tests for the algebraic restructurings the TTNN functional decoder relies on.

These run without a device and without HF weights. They pin down the three non-obvious
rewrites documented in ``doc/functional_decoder/work_log.md`` §3:

1. the UT transform (``(I - A)^-1``) computed by repeated squaring instead of HF's 63-step
   serial forward-substitution loop,
2. the chunked gated-delta-rule restructuring used by ``prefill_forward`` (including
   carrying the recurrent state across independently-issued prefill chunks),
3. the depthwise causal conv1d expressed as shifted multiply-accumulates, plus the proof
   that HF's 4-column conv state has a dead oldest column,
4. the text-only collapse of interleaved mRoPE onto standard 1-D RoPE,
5. l2-norm expressed as an RMS norm with a rescaled epsilon/weight.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextRotaryEmbedding,
    l2norm,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref
from models.autoports.qwen_qwen3_6_35b_a3b.tt.reference import load_hf_text_config

CHUNK = 64


# ---------------------------------------------------------------------------------------
# 3.1 UT transform by repeated squaring
# ---------------------------------------------------------------------------------------
def _ut_transform_serial(a: torch.Tensor) -> torch.Tensor:
    """HF's forward-substitution loop, lifted out of torch_chunk_gated_delta_rule."""
    attn = a.clone()
    chunk_size = a.shape[-1]
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(chunk_size, dtype=a.dtype, device=a.device)


def ut_transform_squaring(a: torch.Tensor) -> torch.Tensor:
    """(I - A)^-1 for strictly-lower-triangular A, via prod_j (I + A^(2^j))."""
    chunk_size = a.shape[-1]
    assert chunk_size & (chunk_size - 1) == 0, "chunk size must be a power of two"
    eye = torch.eye(chunk_size, dtype=a.dtype, device=a.device)
    acc = eye + a
    power = a
    n = chunk_size.bit_length() - 1  # log2(chunk_size)
    for _ in range(1, n):
        power = power @ power
        acc = acc @ (eye + power)
    return acc


@pytest.mark.parametrize("chunk_size", [32, 64, 128])
@pytest.mark.parametrize("scale", [1.0, 0.1])
def test_ut_transform_squaring_matches_serial(chunk_size, scale):
    """prod_j (I + A^(2^j)) reproduces HF's serial forward substitution bit-for-bit-ish.

    ``scale=0.1`` is the realistic regime: in the model A = -(k_beta k^T) * decay with
    L2-normalised k and beta in (0,1), so |A_ij| <~ 0.1 and (I-A) is well conditioned.
    ``scale=1.0`` is a deliberately ill-conditioned stress case — the two formulations must
    still agree even where the inverse itself has entries ~1e7.
    """
    torch.manual_seed(0)
    a = (scale * torch.randn(2, 3, chunk_size, chunk_size, dtype=torch.float64)).tril(-1)
    serial = _ut_transform_serial(a)
    squared = ut_transform_squaring(a)
    rel = ((serial - squared).abs().max() / serial.abs().max()).item()
    assert rel < 1e-12, f"relative mismatch {rel:g}"


@pytest.mark.parametrize("chunk_size", [32, 64, 128])
def test_ut_transform_squaring_is_the_inverse(chunk_size):
    torch.manual_seed(0)
    a = (0.1 * torch.randn(2, 3, chunk_size, chunk_size, dtype=torch.float64)).tril(-1)
    eye = torch.eye(chunk_size, dtype=torch.float64)
    residual = (ut_transform_squaring(a) @ (eye - a) - eye).abs().max().item()
    assert residual < 1e-12, f"residual {residual:g}"


# ---------------------------------------------------------------------------------------
# 3.2 chunked gated delta rule restructuring
# ---------------------------------------------------------------------------------------
def chunked_gated_delta_rule_restructured(query, key, value, g, beta, initial_state=None, chunk_size=CHUNK):
    """The exact op sequence the TTNN prefill path implements.

    Inputs are HF-layout ``[b, seq, heads, dim]`` (g/beta ``[b, seq, heads]``); returns
    ``(core_attn_out [b, seq, heads, dim], final_state [b, heads, dk, dv])``.
    Differences from HF: L2 norm/scale hoisted out, UT transform by squaring, and the
    per-chunk decay mask built additively (``exp`` after masking) so the upper triangle can
    never overflow.
    """
    query = l2norm(query.float(), dim=-1, eps=1e-6)
    key = l2norm(key.float(), dim=-1, eps=1e-6)
    query, key, value, beta, g = (x.transpose(1, 2).float() for x in (query, key, value, beta, g))
    b, h, seq, dk = key.shape
    dv = value.shape[-1]
    assert seq % chunk_size == 0
    nc = seq // chunk_size

    query = query * (dk**-0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    def to_chunks(x):
        return x.reshape(b, h, nc, chunk_size, x.shape[-1])

    query, key, value, k_beta, v_beta = (to_chunks(x) for x in (query, key, value, k_beta, v_beta))
    g = g.reshape(b, h, nc, chunk_size).cumsum(dim=-1)

    tri_incl = torch.ones(chunk_size, chunk_size, dtype=torch.bool).tril()
    neg_inf = torch.where(tri_incl, 0.0, float("-inf"))
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2) + neg_inf).exp()  # [b,h,nc,C,C], tril incl. diag
    decay_strict = decay * torch.ones(chunk_size, chunk_size).tril(-1)

    a_mat = -(k_beta @ key.transpose(-1, -2)) * decay_strict
    t_mat = ut_transform_squaring(a_mat)
    v_tilde = t_mat @ v_beta
    k_cumdecay = t_mat @ (k_beta * g.exp().unsqueeze(-1))

    state = torch.zeros(b, h, dk, dv, dtype=torch.float32) if initial_state is None else initial_state.float().clone()
    out = torch.empty(b, h, nc, chunk_size, dv, dtype=torch.float32)
    for i in range(nc):
        q_i, k_i, g_i = query[:, :, i], key[:, :, i], g[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)) * decay[:, :, i]
        v_new = v_tilde[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = (q_i * g_i.exp().unsqueeze(-1)) @ state + attn @ v_new
        g_last = g_i[..., -1:]
        state = (
            state * g_last.exp().unsqueeze(-1) + (k_i * (g_last - g_i).exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
        )
    out = out.reshape(b, h, seq, dv).transpose(1, 2)
    return out, state


def _random_delta_inputs(b, seq, seed=0, hk=16, hv=32, dk=128, dv=128):
    torch.manual_seed(seed)
    q = torch.randn(b, seq, hk, dk).repeat_interleave(hv // hk, dim=2)
    k = torch.randn(b, seq, hk, dk).repeat_interleave(hv // hk, dim=2)
    v = torch.randn(b, seq, hv, dv)
    beta = torch.rand(b, seq, hv)
    a_log = torch.empty(hv).uniform_(0, 16).log()
    dt_bias = torch.ones(hv)
    a = torch.randn(b, seq, hv)
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)
    return q, k, v, g, beta


@pytest.mark.parametrize("seq", [64, 128, 256])
@pytest.mark.parametrize("batch", [1, 3])
def test_restructured_chunk_rule_matches_hf(seq, batch):
    q, k, v, g, beta = _random_delta_inputs(batch, seq)
    ref_out, ref_state = torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=CHUNK, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    out, state = chunked_gated_delta_rule_restructured(q, k, v, g, beta)
    assert torch.allclose(ref_out, out, atol=2e-5), (ref_out - out).abs().max()
    assert torch.allclose(ref_state, state, atol=2e-5), (ref_state - state).abs().max()


def test_restructured_chunk_rule_is_chunk_splittable():
    """Prefill issued as several chunks with a carried state == one big prefill."""
    seq = 256
    q, k, v, g, beta = _random_delta_inputs(2, seq, seed=1)
    full_out, full_state = chunked_gated_delta_rule_restructured(q, k, v, g, beta)

    state = None
    outs = []
    for lo in range(0, seq, CHUNK):
        hi = lo + CHUNK
        o, state = chunked_gated_delta_rule_restructured(
            q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi], initial_state=state
        )
        outs.append(o)
    split_out = torch.cat(outs, dim=1)
    assert torch.allclose(full_out, split_out, atol=2e-5), (full_out - split_out).abs().max()
    assert torch.allclose(full_state, state, atol=2e-5)


def test_recurrent_step_matches_hf_single_token():
    """The decode recurrence written as matmuls == HF's elementwise reference."""
    b, hv, dk, dv = 3, 32, 128, 128
    q, k, v, g, beta = _random_delta_inputs(b, 1, seed=2)
    init = torch.randn(b, hv, dk, dv)
    ref_out, ref_state = torch_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=True
    )

    # matmul form (what TTNN runs): state[b,h,dk,dv], k/q as [b,h,1,dk]
    qn = (l2norm(q.float(), dim=-1, eps=1e-6) * (dk**-0.5)).transpose(1, 2)
    kn = l2norm(k.float(), dim=-1, eps=1e-6).transpose(1, 2)
    vv = v.float().transpose(1, 2)
    gg = g.float().transpose(1, 2).unsqueeze(-1)  # [b,h,1,1]
    bb = beta.float().transpose(1, 2).unsqueeze(-1)
    state = init.float() * gg.exp().unsqueeze(-1).squeeze(-1)
    kv_mem = kn @ state  # [b,h,1,dv]
    delta = (vv - kv_mem) * bb
    state = state + kn.transpose(-1, -2) @ delta
    out = (qn @ state).transpose(1, 2)

    assert torch.allclose(ref_out, out, atol=2e-5), (ref_out - out).abs().max()
    assert torch.allclose(ref_state, state, atol=2e-5), (ref_state - state).abs().max()


# ---------------------------------------------------------------------------------------
# 3.3 depthwise causal conv1d as shifted MACs
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("seq", [1, 3, 4, 17, 64])
def test_conv1d_shifted_mac_matches_hf_prefill(seq):
    torch.manual_seed(3)
    channels, kernel = 96, 4
    x = torch.randn(1, channels, seq)
    w = torch.randn(channels, 1, kernel)
    ref = F.silu(F.conv1d(x, w, padding=kernel - 1, groups=channels)[:, :, :seq])

    xp = torch.cat([torch.zeros(1, channels, kernel - 1), x], dim=-1)
    acc = torch.zeros(1, channels, seq)
    for j in range(kernel):
        acc = acc + w[:, 0, j].reshape(1, channels, 1) * xp[:, :, j : j + seq]
    assert torch.allclose(ref, F.silu(acc), atol=1e-5)


def test_conv1d_shifted_mac_matches_hf_with_left_context():
    torch.manual_seed(4)
    channels, kernel, seq = 96, 4, 12
    x = torch.randn(1, channels, seq)
    w = torch.randn(channels, 1, kernel)
    # one-shot conv over the whole 2*seq sequence
    x_all = torch.cat([torch.randn(1, channels, seq), x], dim=-1)
    ref_all = F.silu(F.conv1d(x_all, w, padding=kernel - 1, groups=channels)[:, :, : 2 * seq])
    ref_tail = ref_all[:, :, seq:]

    ctx = x_all[:, :, seq - (kernel - 1) : seq]  # last 3 pre-conv inputs
    xp = torch.cat([ctx, x], dim=-1)
    acc = torch.zeros(1, channels, seq)
    for j in range(kernel):
        acc = acc + w[:, 0, j].reshape(1, channels, 1) * xp[:, :, j : j + seq]
    assert torch.allclose(ref_tail, F.silu(acc), atol=1e-5)


def test_hf_conv_state_oldest_column_is_dead():
    """HF stores kernel_size=4 columns; only the newest 3 can affect any later output."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_causal_conv1d_update

    torch.manual_seed(5)
    channels, kernel = 96, 4
    w = torch.randn(channels, kernel)
    x = torch.randn(1, channels, 1)
    base = torch.randn(1, channels, kernel)

    out_a = torch_causal_conv1d_update(x, base.clone(), w, None, "silu")
    perturbed = base.clone()
    perturbed[:, :, 0] += 100.0  # only the oldest column changes
    out_b = torch_causal_conv1d_update(x, perturbed, w, None, "silu")
    assert torch.allclose(out_a, out_b, atol=1e-6)


# ---------------------------------------------------------------------------------------
# 3.4 text-only mRoPE == standard RoPE
# ---------------------------------------------------------------------------------------
def test_mrope_text_only_is_standard_rope():
    cfg = load_hf_text_config()
    rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
    pos = torch.arange(37).view(1, -1)
    x = torch.zeros(1, 37, cfg.hidden_size)
    cos, sin = rope(x, pos.expand(3, 1, -1).clone())

    rotary_dim = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
    assert cos.shape[-1] == rotary_dim
    inv_freq = 1.0 / (cfg.rope_parameters["rope_theta"] ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = pos.float().T * inv_freq  # [seq, rotary_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    assert torch.allclose(cos[0], emb.cos(), atol=1e-6)
    assert torch.allclose(sin[0], emb.sin(), atol=1e-6)


# ---------------------------------------------------------------------------------------
# 3.5 l2norm as rms_norm
# ---------------------------------------------------------------------------------------
def test_l2norm_as_rms_norm():
    torch.manual_seed(6)
    d, eps = 128, 1e-6
    x = torch.randn(4, 7, d)
    ref = l2norm(x, dim=-1, eps=eps)
    rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps / d) * (d**-0.5)
    assert torch.allclose(ref, rms, atol=1e-6), (ref - rms).abs().max()


# ---------------------------------------------------------------------------------------
# 3.6 the long-context tail references are exact
# ---------------------------------------------------------------------------------------
# The entire advertised-context claim (README section 4, `context_contract.json`,
# `long_context.jsonl`) is measured against `hf_prefill_tail` / `hf_linear_prefill_tail` rather
# than a full `hf_prefill`, because a 262143-token HF forward is not affordable. Those helpers
# split the sequence: fill the cache / advance the state over the head, then run the layer over
# the tail. If that split were not exact, every advertised-context PCC in the stage would be
# measured against the wrong golden and still look fine. These two tests close that gap at a
# length where the full reference *is* affordable, so the comparison is direct.
#
# The parametrisation matters as much as the test. `delta_chunk_size` is 64, so a case whose head
# and chunk lengths are all multiples of 64 exercises none of the interesting arithmetic: the
# advertised-context run is `seq=262143, tail=128, chunk=2048`, i.e. head 262015, whose final
# chunked piece is 1919 tokens (63 mod 64, so HF's internal `pad_size` branch in
# `torch_chunk_gated_delta_rule` fires) and whose tail starts 63 off the global 64-chunk grid.
# The `ragged` cases below reproduce both conditions; the aligned case is kept because it is the
# one that isolates a plain reassociation error from a boundary error.
@pytest.mark.parametrize(
    "kind,layer_idx,seq,tail,chunk,label",
    [
        ("linear", 0, 512, 128, 128, "aligned"),
        ("full", 3, 512, 128, 128, "aligned"),
        # head 447 = 6*64 + 63: the last chunked piece is short *and* not a multiple of 64, and the
        # tail starts at 447 = 63 (mod 64).
        ("linear", 0, 575, 128, 150, "ragged-head"),
        ("full", 3, 575, 128, 150, "ragged-head"),
        # tail itself not a multiple of 64, on top of a ragged head
        ("linear", 0, 575, 129, 150, "ragged-head-and-tail"),
        ("full", 3, 575, 129, 150, "ragged-head-and-tail"),
    ],
)
def test_tail_reference_matches_full_prefill(kind, layer_idx, seq, tail, chunk, label):
    cfg = load_hf_text_config()
    layer = ref.build_hf_layer(cfg, layer_idx, ref.synthetic_layer_state_dict(layer_idx))
    x = ref.synthetic_hidden_states(cfg, 1, seq, seed=91)

    full = ref.hf_prefill(layer, cfg, x, start_pos=0, cache=ref.make_cache(cfg)).output[:, -tail:]
    if kind == "full":
        got = ref.hf_prefill_tail(layer, cfg, x, tail=tail)
    else:
        # chunk smaller than the head so the chunked state advance is actually exercised
        got, _ = ref.hf_linear_prefill_tail(layer, cfg, x, tail=tail, chunk=chunk)

    assert got.shape == full.shape, (got.shape, full.shape)
    # Same math, same dtype (fp32), so this is a tolerance on reassociation only.
    assert torch.allclose(got, full, atol=2e-4), (label, float((got - full).abs().max()))
