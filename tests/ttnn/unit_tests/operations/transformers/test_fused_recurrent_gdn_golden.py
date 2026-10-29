# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The registered golden of ttnn.transformer.fused_recurrent_gated_delta_rule, checked on the host (no device)."""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import (
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
    as FLA naive expects; GQA expanded on the host for FLA. Bit-identical: the same statements in the same order."""
    sh = SHAPES[shape]
    x = make_inputs(B, T, sh.num_key_heads, sh.num_value_heads, K, V, seed=1, with_state=with_state)
    o, state = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    groups = sh.num_value_heads // sh.num_key_heads
    o_ref, state_ref = fla_naive_recurrent_gated_delta_rule(
        x["q"].repeat_interleave(groups, dim=2),
        x["k"].repeat_interleave(groups, dim=2),
        x["v"],
        x["beta"],
        x["g"],
        initial_state=x["initial_state"],
        output_final_state=True,
    )
    assert o.shape == (B, T, sh.num_value_heads, V) and state.shape == (B, sh.num_value_heads, K, V)
    assert torch.equal(o, o_ref) and torch.equal(state, state_ref)


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


# ---- the v2 contract in the golden -----------------------------------------------------------------------------------
KDA_OP = ttnn.transformer.fused_recurrent_kda


def _kda_golden():
    return ttnn.get_golden_function(KDA_OP)


def test_golden_norm_keyword_is_the_fla_name():
    """use_qk_l2norm_in_kernel is the op's (FLA's) name of the norm option; the former use_qk_l2norm still works."""
    x = make_inputs(1, 2, 4, 12, K, V, seed=20, normalize_qk=False)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
    o_new, s_new = _golden()(
        *args, initial_state=x["initial_state"], output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    o_old, s_old = _golden()(*args, initial_state=x["initial_state"], output_final_state=True, use_qk_l2norm=True)
    assert torch.equal(o_new, o_old) and torch.equal(s_new, s_old)
    o_host, _ = _golden()(l2_norm(x["q"]), l2_norm(x["k"]), *args[2:], initial_state=x["initial_state"])
    assert torch.equal(o_new, o_host)


def test_golden_gate_in_kernel_gdn_softplus_and_beta_sigmoid():
    """With use_gate_in_kernel the golden computes g = -exp(A_log) * softplus(g_raw + dt_bias) (x > 20 -> x) and with
    use_beta_sigmoid_in_kernel beta = sigmoid(beta_raw), times 2 under allow_neg_eigval; the result equals the plain
    golden fed the gates computed on the host."""
    gen = torch.Generator().manual_seed(21)
    x = make_inputs(2, 3, 4, 12, K, V, seed=21)
    g_raw = torch.randn(2, 3, 12, generator=gen) * 3
    b_raw = torch.randn(2, 3, 12, generator=gen)
    A_log = torch.randn(12, generator=gen) * 0.5
    dt_bias = torch.randn(12, generator=gen)
    g_host = -torch.exp(A_log) * torch.nn.functional.softplus(g_raw + dt_bias, threshold=20)
    for neg in (False, True):
        beta_host = torch.sigmoid(b_raw) * (2.0 if neg else 1.0)
        o, s = _golden()(
            x["q"],
            x["k"],
            x["v"],
            g_raw,
            b_raw,
            initial_state=x["initial_state"],
            output_final_state=True,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=neg,
        )
        o_ref, s_ref = _golden()(
            x["q"], x["k"], x["v"], g_host, beta_host, initial_state=x["initial_state"], output_final_state=True
        )
        assert torch.equal(o, o_ref) and torch.equal(s, s_ref)


def test_golden_gate_in_kernel_kda_bounded_sigmoid():
    """KDA's bounded gate g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias)) with dt_bias [HV*K] viewed as
    [HV, K], through the KDA alias's golden; the softplus form reduces to the GDN formula per key."""
    gen = torch.Generator().manual_seed(22)
    x = make_inputs(1, 2, 12, 12, K, V, seed=22)
    g_raw = torch.randn(1, 2, 12, K, generator=gen)
    A_log = torch.randn(12, generator=gen) * 0.5
    dt_bias = torch.randn(12 * K, generator=gen)
    common = dict(
        initial_state=x["initial_state"], output_final_state=True, use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias
    )
    o, s = _kda_golden()(x["q"], x["k"], x["v"], g_raw, x["beta"], lower_bound=-5.0, **common)
    g_host = -5.0 * torch.sigmoid(torch.exp(A_log)[:, None] * (g_raw + dt_bias.reshape(12, K)))
    o_ref, s_ref = _kda_golden()(
        x["q"], x["k"], x["v"], g_host, x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    assert torch.equal(o, o_ref) and torch.equal(s, s_ref)
    o2, s2 = _kda_golden()(x["q"], x["k"], x["v"], g_raw, x["beta"], **common)
    g_sp = -torch.exp(A_log)[:, None] * torch.nn.functional.softplus(g_raw + dt_bias.reshape(12, K), threshold=20)
    o2_ref, s2_ref = _kda_golden()(
        x["q"], x["k"], x["v"], g_sp, x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    assert torch.equal(o2, o2_ref) and torch.equal(s2, s2_ref)


def test_golden_pool_with_1d_indices_updates_only_the_addressed_slots():
    """state_indices [B] over a pool [slots, HV, K, V]: row b starts from pool[idx[b]]; a negative index skips the
    row (o zero, slot untouched); inplace_final_state returns the pool with exactly the addressed slots rewritten."""
    gen = torch.Generator().manual_seed(23)
    x = make_inputs(3, 2, 4, 12, K, V, seed=23, with_state=False)
    pool = 0.05 * torch.randn(6, 12, K, V, generator=gen)
    idx = torch.tensor([4, -1, 0], dtype=torch.int32)
    o, new_pool = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=pool, state_indices=idx, inplace_final_state=True
    )
    assert new_pool.shape == pool.shape
    for b, slot in enumerate(idx.tolist()):
        if slot < 0:
            assert torch.equal(o[b], torch.zeros_like(o[b]))
            continue
        o_b, s_b = _golden()(
            x["q"][b : b + 1],
            x["k"][b : b + 1],
            x["v"][b : b + 1],
            x["g"][b : b + 1],
            x["beta"][b : b + 1],
            initial_state=pool[slot : slot + 1],
            output_final_state=True,
        )
        assert torch.equal(o[b], o_b[0]) and torch.equal(new_pool[slot], s_b[0])
    untouched = [s for s in range(6) if s not in (4, 0)]
    assert torch.equal(new_pool[untouched], pool[untouched])
    # without inplace_final_state the pool is read only and the final state comes back per row
    _, finals = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=pool, state_indices=idx, output_final_state=True
    )
    assert finals.shape == (3, 12, K, V) and torch.equal(finals[0], new_pool[4]) and torch.equal(finals[2], new_pool[0])


def test_golden_pool_with_2d_indices_and_accepted_tokens():
    """state_indices [B, T] with num_accepted_tokens [B]: row b starts from slot idx[b, n_b - 1] and the state after
    token t goes to slot idx[b, t]; the result equals the chained single-token recurrence from that slot."""
    gen = torch.Generator().manual_seed(24)
    T = 4
    x = make_inputs(2, T, 4, 12, K, V, seed=24, with_state=False)
    pool = 0.05 * torch.randn(9, 12, K, V, generator=gen)
    idx = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    n_acc = torch.tensor([2, 4], dtype=torch.int32)
    o, new_pool = _golden()(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=pool,
        state_indices=idx,
        num_accepted_tokens=n_acc,
        inplace_final_state=True,
    )
    _, per_token = _golden()(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=pool,
        state_indices=idx,
        num_accepted_tokens=n_acc,
        output_per_token_state=True,
    )
    for b in range(2):
        start = int(idx[b, int(n_acc[b]) - 1])
        o_chain, states_chain = chained_decode(
            _golden(),
            x["q"][b : b + 1],
            x["k"][b : b + 1],
            x["v"][b : b + 1],
            x["g"][b : b + 1],
            x["beta"][b : b + 1],
            pool[start : start + 1],
        )
        assert torch.equal(o[b], o_chain[0]) and torch.equal(per_token[b], states_chain[0])
        for t in range(T):
            assert torch.equal(new_pool[int(idx[b, t])], states_chain[0, t])
    assert torch.equal(new_pool[0], pool[0])  # the one slot no index names


def test_golden_state_v_first_transposes_at_the_boundary():
    x = make_inputs(2, 3, 4, 12, K, V, seed=25)
    o_ref, s_ref = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    o, s_vf = _golden()(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=x["initial_state"].transpose(-1, -2).contiguous(),
        output_final_state=True,
        state_v_first=True,
    )
    assert torch.equal(o, o_ref) and torch.equal(s_vf, s_ref.transpose(-1, -2))


def test_golden_cu_seqlens_runs_each_sequence_from_its_own_state():
    """B == 1, cu_seqlens [0, 2, 5]: tokens [0, 2) from initial_state[0], tokens [2, 5) from initial_state[1]."""
    x = make_inputs(1, 5, 4, 12, K, V, seed=26, with_state=False)
    s0 = 0.05 * torch.randn(2, 12, K, V, generator=torch.Generator().manual_seed(26))
    o, finals = _golden()(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=s0,
        cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32),
        output_final_state=True,
    )
    assert finals.shape == (2, 12, K, V)
    for n, (a, b) in enumerate([(0, 2), (2, 5)]):
        o_n, s_n = _golden()(
            x["q"][:, a:b],
            x["k"][:, a:b],
            x["v"][:, a:b],
            x["g"][:, a:b],
            x["beta"][:, a:b],
            initial_state=s0[n : n + 1],
            output_final_state=True,
        )
        assert torch.equal(o[:, a:b], o_n) and torch.equal(finals[n : n + 1], s_n)


def test_kda_golden_is_the_gdn_golden_with_rank4_g(expect_error):
    assert callable(_kda_golden())
    x = make_inputs(1, 2, 12, 12, K, V, seed=27)
    g4 = x["g"].unsqueeze(-1).expand(-1, -1, -1, K).contiguous()
    o_kda, s_kda = _kda_golden()(
        x["q"], x["k"], x["v"], g4, x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    o_gdn, s_gdn = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    assert torch.equal(o_kda, o_gdn) and torch.equal(s_kda, s_gdn)
    with expect_error(AssertionError, "rank 4"):  # a host-side golden assertion, not a device error
        _kda_golden()(x["q"], x["k"], x["v"], x["g"], x["beta"])
