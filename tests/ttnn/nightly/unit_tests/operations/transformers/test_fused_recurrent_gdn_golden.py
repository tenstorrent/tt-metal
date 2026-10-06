# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The registered golden of ttnn.transformer.fused_recurrent_gated_delta_rule, checked on the host (no device)."""

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.transformers.gdn_decode_test_utils import (
    SHAPES,
    chained_decode,
    fla_naive_recurrent_gated_delta_rule,
    make_inputs,
)
from ttnn.operations.transformer_golden import l2_norm, recurrent_gated_delta_rule

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
K = V = 128


def _golden():
    return ttnn.get_golden_function(OP)


def test_fused_recurrent_gdn_has_registered_golden_function():
    assert callable(_golden())


@pytest.mark.parametrize("shape", ["qwen27b_tp4", "qwen9b_tp1"])
@pytest.mark.parametrize("B, T", [(1, 1), (2, 4)])
@pytest.mark.parametrize("with_state", [True, False])
def test_golden_matches_fla_naive(shape, B, T, with_state):
    """The op's argument order (q, k, v, g, beta) over FLA naive's (q, k, v, beta, g); q, k pre-normalised on the host
    as FLA naive expects; GQA expanded on the host for FLA. Bit-identical against the vendored copy (same statements in
    the same order), close against an installed FLA."""
    sh = SHAPES[shape]
    x = make_inputs(B, T, sh.num_key_heads, sh.num_value_heads, K, V, seed=1, with_state=with_state)
    o, state = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    fla, source = fla_naive_recurrent_gated_delta_rule()
    groups = sh.num_value_heads // sh.num_key_heads
    o_ref, state_ref = fla(
        x["q"].repeat_interleave(groups, dim=2),
        x["k"].repeat_interleave(groups, dim=2),
        x["v"],
        x["beta"],
        x["g"],
        initial_state=x["initial_state"],
        output_final_state=True,
    )
    assert o.shape == (B, T, sh.num_value_heads, V) and state.shape == (B, sh.num_value_heads, K, V)
    if source == "vendored":
        assert torch.equal(o, o_ref) and torch.equal(state, state_ref)
    else:
        torch.testing.assert_close(o, o_ref)
        torch.testing.assert_close(state, state_ref)


def test_golden_in_reference_l2norm_equals_host_l2norm():
    """use_qk_l2norm=True normalises q and k inside the golden with the module's l2_norm; the same values normalised
    on the host give the same result bit for bit."""
    x = make_inputs(2, 3, 4, 12, K, V, seed=2, normalize_qk=False)
    o_in, s_in = _golden()(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=x["initial_state"],
        output_final_state=True,
        use_qk_l2norm=True,
    )
    o_host, s_host = _golden()(
        l2_norm(x["q"]),
        l2_norm(x["k"]),
        x["v"],
        x["g"],
        x["beta"],
        initial_state=x["initial_state"],
        output_final_state=True,
    )
    assert torch.equal(o_in, o_host) and torch.equal(s_in, s_host)


def test_golden_gqa_is_repeat_interleave():
    x = make_inputs(2, 3, 4, 12, K, V, seed=3)
    got, got_state = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    want, want_state = _golden()(
        x["q"].repeat_interleave(3, dim=2),
        x["k"].repeat_interleave(3, dim=2),
        x["v"],
        x["g"],
        x["beta"],
        initial_state=x["initial_state"],
        output_final_state=True,
    )
    assert torch.equal(got, want) and torch.equal(got_state, want_state)


@pytest.mark.parametrize("K_draft", [3, 7])
def test_golden_multi_token_equals_chained_single_token(K_draft):
    """One call with T = K + 1 and per-token states equals K + 1 chained T = 1 calls; the last per-token state is
    the final state."""
    T = K_draft + 1
    x = make_inputs(2, T, 4, 12, K, V, seed=4)
    o, states = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_per_token_state=True
    )
    _, final = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    o_chain, states_chain = chained_decode(_golden(), x["q"], x["k"], x["v"], x["g"], x["beta"], x["initial_state"])
    assert states.shape == (2, T, 12, K, V)
    assert torch.equal(o, o_chain) and torch.equal(states, states_chain)
    assert torch.equal(states[:, -1], final)


def test_golden_state_output_selection():
    x = make_inputs(1, 2, 4, 12, K, V, seed=5)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
    assert _golden()(*args)[1] is None
    assert _golden()(*args, output_final_state=True)[1].shape == (1, 12, K, V)
    assert _golden()(*args, output_per_token_state=True)[1].shape == (1, 2, 12, K, V)
    # per-token wins over final, as in the op
    assert _golden()(*args, output_final_state=True, output_per_token_state=True)[1].shape == (1, 2, 12, K, V)


def test_golden_per_key_decay_reduces_to_scalar_decay():
    """A rank-4 g whose rows are constant over K is the scalar-decay recurrence, bit for bit."""
    x = make_inputs(2, 3, 12, 12, K, V, seed=6)
    g4 = x["g"].unsqueeze(-1).expand(-1, -1, -1, K).contiguous()
    o3, s3 = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    o4, s4 = _golden()(x["q"], x["k"], x["v"], g4, x["beta"], initial_state=x["initial_state"], output_final_state=True)
    assert torch.equal(o3, o4) and torch.equal(s3, s4)
    # and a genuinely per-key decay scales the state rows: row k of S decays by exp(g[k])
    g_rows = -torch.rand(2, 1, 12, K)
    _, s_rows = recurrent_gated_delta_rule(
        x["q"][:, :1],
        x["k"][:, :1],
        x["v"][:, :1],
        x["beta"][:, :1],
        g_rows,
        initial_state=x["initial_state"],
        output_final_state=True,
    )
    decayed = x["initial_state"] * g_rows[:, 0].exp()[..., :, None]
    k_hat = x["k"][:, 0]  # [B, H, K]
    v_read = (decayed * k_hat[..., None]).sum(-2)
    u = (x["v"][:, 0] - v_read) * x["beta"][:, 0][..., None]
    expected = decayed + k_hat.unsqueeze(-1) * u.unsqueeze(-2)
    torch.testing.assert_close(s_rows, expected)


def test_recurrence_fp64_agrees_with_fp32_to_fp32_precision():
    x = make_inputs(1, 8, 4, 12, K, V, seed=7)
    o32, s32 = recurrent_gated_delta_rule(
        x["q"], x["k"], x["v"], x["beta"], x["g"], initial_state=x["initial_state"], output_final_state=True
    )
    o64, s64 = recurrent_gated_delta_rule(
        x["q"],
        x["k"],
        x["v"],
        x["beta"],
        x["g"],
        initial_state=x["initial_state"],
        output_final_state=True,
        dtype=torch.float64,
    )
    assert o64.dtype == torch.float64 and s64.dtype == torch.float64
    torch.testing.assert_close(o32.double(), o64, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s32.double(), s64, rtol=1e-4, atol=1e-5)
