# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP v2 host tests, steps 2 and 4: accept logic, GDN chunk, rows, and the row-serial fakes shared by the later steps."""

from __future__ import annotations

import ast
import inspect
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.reference import (
    _l2_norm,
    build_ngram_hash_spec,
    causal_depthwise_conv1d,
    gated_delta_recurrent,
    mtp_input_fusion,
    ngram_token_ids,
)
from models.demos.blackhole.qwen38_flash_next.tools.mtp_v2_verify_reference import (
    accept_select,
    chain_alphas,
    expected_tokens_per_pass,
    row_serial_torch,
    tf32,
    ulp_distance,
)
from models.demos.blackhole.qwen38_flash_next.tt.gdn import (
    Qwen38GDN,
    Qwen38GDNDimensions,
    Qwen38GDNState,
    Qwen38GDNWeights,
)
from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidual, Qwen38GatedResidualWeights
from models.demos.blackhole.qwen38_flash_next.tt.model import Qwen38FinalMixer, Qwen38FinalMixerWeights
from models.demos.blackhole.qwen38_flash_next.tt.moe import (
    Qwen38ExpertWeights,
    Qwen38MoE,
    Qwen38SharedExpertShard,
    weighted_routed_reduce,
)
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38PLE, Qwen38PLEState, Qwen38PLEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import GREEDY_TIE_BREAK_EPS, TP_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_wrap
from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)

# ============================================================================
# Step 2 accept: device accept logic, tie rule, tok/pass arithmetic (was test_mtp_v2_step2_accept_no_device.py)
# MTP v2 step 2 (c, d): device accept logic on integer ids, the argmax tie rule, and the tok/pass arithmetic.
#
# No device.  The accept model is the design's section 2.6 op sequence in fp32 with the FPU reduce stages
# read through TF32 (the greedy-resolve precision model, silicon-confirmed: fp32 integers survive an FPU
# reduce only below 2048).
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


# Ids seen on the pinned prefixes and at the vocabulary edge: below and above the 2048 TF32 limit.
SMALL_IDS = (17, 15, 16, 21, 12, 20, 11, 0, 2047)
LARGE_IDS = (2048, 95859, 62086, 248044, 248319)
R6_HISTOGRAM = [74, 75, 55, 38, 92]  # accepted drafts 0..4 over 334 rounds (r6 arm, 2026-09-02)


def _s2a_rows(pattern: tuple[int, ...], ids: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify-row argmaxes, drafts and alignment argmaxes realizing one match/mismatch pattern."""

    k = len(pattern)
    cycle = itertools.cycle(ids)
    targets = torch.tensor([next(cycle) for _ in range(k + 1)], dtype=torch.float32)
    drafts = torch.empty(k, dtype=torch.float32)
    for j, match in enumerate(pattern):
        other = next(value for value in ids if value != int(targets[j]))
        drafts[j] = targets[j] if match else other
    alignment = torch.tensor([next(cycle) for _ in range(k + 1)], dtype=torch.float32)
    return targets, drafts, alignment


def _leading_matches(pattern: tuple[int, ...]) -> int:
    return next((j for j, match in enumerate(pattern) if not match), len(pattern))


@pytest.mark.parametrize("k", (3, 4, 5))
def test_accept_count_and_gather_select_are_exact_for_every_pattern(k: int) -> None:
    for pattern in itertools.product((0, 1), repeat=k):
        for ids in (SMALL_IDS, LARGE_IDS, SMALL_IDS + LARGE_IDS):
            targets, drafts, alignment = _s2a_rows(pattern, ids)
            result = accept_select(targets, drafts, alignment)
            accepted = _leading_matches(pattern)
            assert result["accepted"] == accepted, (pattern, ids, result)
            assert result["next_token_gather"] == int(targets[accepted]), (pattern, ids, result)
            assert result["first_draft_gather"] == int(alignment[accepted]), (pattern, ids, result)


@pytest.mark.parametrize("k", (3, 4, 5))
def test_sum_select_is_exact_only_below_the_tf32_integer_limit(k: int) -> None:
    """Design 2.6 writes t' = sum(one_hot * argmax rows); through an FPU reduce that is wrong from id 2048."""

    wrong_large = 0
    for pattern in itertools.product((0, 1), repeat=k):
        small = accept_select(*_s2a_rows(pattern, SMALL_IDS))
        assert small["next_token_sum"] == small["next_token_gather"], pattern
        assert small["first_draft_sum"] == small["first_draft_gather"], pattern
        large = accept_select(*_s2a_rows(pattern, LARGE_IDS))
        wrong_large += large["next_token_sum"] != large["next_token_gather"]
        wrong_large += large["first_draft_sum"] != large["first_draft_gather"]
        # Same ops with an exact reduce (fp32 accumulate of exact inputs): the arithmetic itself is fine.
        exact = accept_select(*_s2a_rows(pattern, LARGE_IDS), fpu_rounding=lambda x: x)
        assert exact["next_token_sum"] == exact["next_token_gather"], pattern
    assert wrong_large > 0, "the TF32 model must reject the sum select on large ids"
    assert tf32(torch.tensor([95859.0, 248044.0])).tolist() == [95872.0, 248064.0]


def test_accept_count_survives_the_fpu_reduce_and_position_update_is_exact() -> None:
    flags = torch.ones(5, dtype=torch.float32)
    assert tf32(flags).sum().item() == 5.0
    for position in (0, 4, 31, 2047, 2048, 32767, 2**16 - 1):
        for accepted in range(6):
            # P <- P + a + 1 as an SFPU add on fp32 (exact below 2**24) or a uint32 add.
            assert float(position) + float(accepted) + 1.0 == float(position + accepted + 1)


def _ranked_owner(values: torch.Tensor) -> torch.Tensor:
    # resolve_greedy_on_device rule, rows batched: bf16 maxima -> fp32 -> minus owner * eps -> argmax.
    assert values.dtype == torch.bfloat16
    ranked = values.to(torch.float32) - torch.arange(TP_SIZE, dtype=torch.float32) * GREEDY_TIE_BREAK_EPS
    return torch.argmax(ranked, dim=-1)


def test_row_batched_owner_tie_break_picks_the_lowest_owner_like_resolve_greedy() -> None:
    torch.manual_seed(7)
    for _ in range(2000):
        rows = torch.randint(1, 6, ()).item()
        values = (torch.randn(rows, TP_SIZE) * 8).to(torch.bfloat16)
        values[0] = values[0, 0]  # all four owners tie on row 0
        if rows > 1:
            values[1, 2] = values[1, 1]  # a two-way tie on row 1
        expected = torch.argmax(values.to(torch.float32), dim=-1)  # first owner on ties (resolve_greedy)
        assert torch.equal(_ranked_owner(values), expected)
        ranked = values.to(torch.float32) - torch.arange(TP_SIZE, dtype=torch.float32) * GREEDY_TIE_BREAK_EPS
        # No row keeps a tie after the shift, so the device argmax kernel's tie rule cannot matter.
        top = ranked.max(dim=-1, keepdim=True).values
        assert torch.all((ranked == top).sum(dim=-1) == 1)


def test_tie_break_shift_keeps_distinct_bf16_logits_ordered() -> None:
    # Adjacent bf16 values from 2**-6 up: the larger one minus 3 * eps stays above the smaller one.
    exponents = torch.arange(-6, 8, dtype=torch.float32)
    for exponent in exponents.tolist():
        base = torch.tensor(2.0**exponent, dtype=torch.bfloat16)
        step = torch.nextafter(base.to(torch.float32), torch.tensor(float("inf")))
        larger = step.to(torch.bfloat16)
        while larger == base:
            step = torch.nextafter(step, torch.tensor(float("inf")))
            larger = step.to(torch.bfloat16)
        assert larger.to(torch.float32) - 3 * GREEDY_TIE_BREAK_EPS > base.to(torch.float32), exponent


def test_r6_histogram_reproduces_the_pooled_alphas_and_tok_per_pass() -> None:
    alphas = chain_alphas(R6_HISTOGRAM)
    assert sum(R6_HISTOGRAM) == 334
    assert [round(alpha, 3) for alpha in alphas] == [0.778, 0.712, 0.703, 0.708]
    mean_accepted = sum(count * drafts for drafts, count in enumerate(R6_HISTOGRAM)) / sum(R6_HISTOGRAM)
    assert round(mean_accepted, 2) == 2.00
    assert round(expected_tokens_per_pass(alphas, 3), 2) == 2.72
    assert round(expected_tokens_per_pass(alphas, 4), 2) == 3.00
    # With the full chain, E[tok/pass] at k=4 is exactly 1 + the mean accepted count.
    assert expected_tokens_per_pass(alphas, 4) == pytest.approx(1.0 + mean_accepted, abs=1e-12)
    # The rounded alphas of the results note give the same two decimals.
    assert round(expected_tokens_per_pass([0.778, 0.712, 0.703], 3), 2) == 2.72
    assert round(expected_tokens_per_pass([0.778, 0.712, 0.703, 0.708], 4), 2) == 3.00


def test_design_section_three_chains_reproduce_the_cost_model_rows(expect_error) -> None:
    measured = [0.791, 0.768, 0.767]  # CPU free-run chain; alpha_4|3 = alpha_5|4 = 0.767 extrapolated
    assert [round(expected_tokens_per_pass(measured, k), 3) for k in (1, 2, 3, 4, 5)] == [
        1.791,
        2.398,
        2.864,
        3.222,
        3.496,
    ]
    geometric = [0.791]
    assert [round(expected_tokens_per_pass(geometric, k), 3) for k in (1, 2, 3, 4, 5)] == [
        1.791,
        2.417,
        2.912,
        3.303,
        3.613,
    ]
    with expect_error(ValueError):  # allow-pytest.raises: pure arithmetic contract
        expected_tokens_per_pass([], 3)


# ============================================================================
# Step 2 GDN chunk (was test_mtp_v2_step2_gdn_chunk_no_device.py)
# MTP v2 step 2 (b): the GDN chunk path against the fp32 step recurrence, with the masked catch-up.
#
# No device.  The chunk model is the torch reference of the chunk kernel (``torch_functional.delta_rule_ops``:
# fp32 WY form, chunk 32 or 64); the step model is the CPU oracle's ``gated_delta_recurrent``.  Per-device
# geometry: 12 value heads, K = V = 128.  q and k are L2-normalized once, up front, in bf16 (the oracle's
# normalize-in-input-dtype rule): torch's bf16 reduction blocks differently for 1 and 5 rows, and a 1-ULP flip
# in a normalized k moves the state by 4e-3, which would hide the algorithmic gap under test.  ULP gaps are
# printed with ``-s``; the gate is fp32 closeness (rtol 1e-4) of the chunk form to the step form.
# The diagonal-regularized WY form of ``ttnn_delta_rule_seq`` (QWEN_GDN_DIAG_ALPHA, default 0.25) is modelled
# as well: only alpha = 0 is the step recurrence.  The conv window rule for the GDN FIR and the dilated PLE conv
# (new history = window rows [committed, committed + state_len)) is checked against the reference convolution.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


S2G_HEADS = 12
S2G_HEAD_DIM = 128
S2G_ROWS = 5
RTOL = 1e-4  # measured: output rel <= 2.1e-6, state rel <= 1.2e-5 against the step recurrence
ATOL = 5e-6


def _s2g_rows(rows: int, seed: int = 1) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "q": _l2_norm(torch.randn(1, rows, S2G_HEADS, S2G_HEAD_DIM).to(torch.bfloat16)),
        "k": _l2_norm(torch.randn(1, rows, S2G_HEADS, S2G_HEAD_DIM).to(torch.bfloat16)),
        "v": torch.randn(1, rows, S2G_HEADS, S2G_HEAD_DIM).to(torch.bfloat16),
        "beta": torch.sigmoid(torch.randn(1, rows, S2G_HEADS)).to(torch.bfloat16),
        "g": -torch.rand(1, rows, S2G_HEADS) * 0.5,  # fp32 log decay, as the oracle's -exp(A_log) * softplus(.)
        "state": torch.randn(1, S2G_HEADS, S2G_HEAD_DIM, S2G_HEAD_DIM) * 0.3,
    }


def _step(x: dict[str, torch.Tensor], rows: slice = slice(None)) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle step recurrence (bf16 output, fp32 state) and the same recurrence's fp32 output."""

    q, k, v, beta, g = (x[name][:, rows] for name in ("q", "k", "v", "beta", "g"))
    output, state = gated_delta_recurrent(q, k, v, g, beta, x["state"], l2_normalize_qk=False)
    output_fp32, state_fp32 = recurrent_gated_delta_rule(
        q, k, v, beta, g, initial_state=x["state"], output_final_state=True
    )
    assert torch.equal(state_fp32, state)  # the two step forms are the same fp32 recurrence
    return output, output_fp32, state


def _chunk(x: dict[str, torch.Tensor], chunk_size: int, rows: slice = slice(None), **overrides):
    values = {**x, **overrides}
    q, k, v, beta, g = (values[name][:, rows] for name in ("q", "k", "v", "beta", "g"))
    return chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=chunk_size, initial_state=values["state"], output_final_state=True
    )


def _report(label: str, output, output_ref_fp32, output_ref_bf16, state, state_ref) -> None:
    out_bf16 = int(ulp_distance(output, output_ref_bf16, torch.bfloat16).max())
    out_fp32 = int(ulp_distance(output, output_ref_fp32, torch.float32).max())
    state_fp32 = int(ulp_distance(state, state_ref, torch.float32).max())
    out_rel = float(((output - output_ref_fp32).abs() / output_ref_fp32.abs().clamp_min(1e-2)).max())
    state_rel = float(((state - state_ref).abs() / state_ref.abs().clamp_min(1e-2)).max())
    print(
        f"{label}: output max ULP bf16 {out_bf16} (vs the oracle's bf16 output) fp32 {out_fp32} rel {out_rel:.2e}; "
        f"state max ULP fp32 {state_fp32} rel {state_rel:.2e}"
    )
    torch.testing.assert_close(output, output_ref_fp32, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(state, state_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("chunk_size", (32, 64))
@pytest.mark.parametrize("rows", (1, 2, 3, 4, 5))
def test_chunk_matches_the_step_recurrence_with_an_initial_state(rows: int, chunk_size: int) -> None:
    x = _s2g_rows(rows)
    step_out, step_out_fp32, step_state = _step(x)
    chunk_out, chunk_state = _chunk(x, chunk_size)
    assert step_out.dtype == torch.bfloat16 and chunk_out.dtype == torch.float32
    assert chunk_out.shape == (1, rows, S2G_HEADS, S2G_HEAD_DIM) and chunk_state.shape == x["state"].shape
    _report(f"T={rows} chunk={chunk_size} chunk vs step", chunk_out, step_out_fp32, step_out, chunk_state, step_state)


@pytest.mark.parametrize("committed", range(S2G_ROWS + 1))
def test_masked_catch_up_equals_the_committed_prefix(committed: int) -> None:
    """Design 2.4 step 1: beta and g of rows at or past the committed count are zeroed; k and v may stay."""

    x = _s2g_rows(S2G_ROWS, seed=2)
    mask = (torch.arange(S2G_ROWS) < committed).to(torch.float32).reshape(1, S2G_ROWS, 1)
    masked = {"beta": (x["beta"].float() * mask).to(torch.bfloat16), "g": x["g"] * mask}
    masked_out, state_masked = _chunk(x, 32, **masked)
    keep = mask.unsqueeze(-1) > 0
    zeros = torch.zeros_like(x["q"])
    _, state_zeroed = _chunk(x, 32, k=torch.where(keep, x["k"], zeros), v=torch.where(keep, x["v"], zeros), **masked)
    # Zeroing k and v as well (the design's padding rule) changes nothing: beta = 0 already makes the write vanish.
    assert torch.equal(state_masked, state_zeroed)
    if committed == 0:
        assert torch.equal(state_masked, x["state"])  # identity update, bitwise in the torch model
        return
    _, state_prefix = _chunk(x, 32, rows=slice(0, committed))
    assert torch.equal(state_masked, state_prefix)  # the masked rows contribute exact zeros
    step_out, step_out_fp32, step_state = _step(x, slice(0, committed))
    _report(
        f"catch-up committed={committed} vs step",
        masked_out[:, :committed],
        step_out_fp32,
        step_out,
        state_masked,
        step_state,
    )
    # Rows before the committed count are unaffected by the mask (the verify outputs of the committed rows).
    full_out, _ = _chunk(x, 32)
    assert torch.equal(masked_out[:, :committed], full_out[:, :committed])


def test_catch_up_mask_counts_committed_rows_not_accepted_drafts() -> None:
    """Row 0 is the base token and is always committed: a accepted drafts commit a + 1 rows.

    The design writes the mask as ``arange(k+1) < a_prev``; with a_prev the accepted-draft count that is one
    row short.  ``a_prev`` there has to be the committed row count (accepted drafts + 1).
    """

    x = _s2g_rows(S2G_ROWS, seed=3)
    for accepted in range(S2G_ROWS):
        short = (torch.arange(S2G_ROWS) < accepted).to(torch.float32).reshape(1, S2G_ROWS, 1)
        _, state_short = _chunk(x, 32, beta=(x["beta"].float() * short).to(torch.bfloat16), g=x["g"] * short)
        _, _, state_committed = _step(x, slice(0, accepted + 1))
        assert not torch.allclose(state_short, state_committed, rtol=1e-3, atol=1e-3), accepted


def _chunk_with_diagonal_regularization(x: dict[str, torch.Tensor], alpha: float, chunk_size: int = 32):
    """The ``ttnn_delta_rule_seq`` WY form: L = I + strict_lower(kk * L_mask) + alpha * diag(kk * L_mask), solved."""

    q, k, v, beta, g = [x[name].transpose(1, 2).contiguous().to(torch.float32) for name in ("q", "k", "v", "beta", "g")]
    T = q.shape[-2]
    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    beta, g = (F.pad(t, (0, pad)) for t in (beta, g))
    B, H, L, K = q.shape
    q = q * K**-0.5
    chunks = lambda t: t.reshape(B, H, -1, chunk_size, t.shape[-1])
    q_c, k_c = chunks(q), chunks(k)
    v_beta_c, k_beta_c = chunks(v * beta[..., None]), chunks(k * beta[..., None])
    decay = g.reshape(B, H, -1, chunk_size).cumsum(-1)
    decay_exp = decay.exp()[..., None]
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()
    kk = (k_beta_c @ k_c.transpose(-1, -2)) * L_mask
    eye = torch.eye(chunk_size)
    L_mat = eye + kk.tril(-1) + alpha * torch.diag_embed(kk.diagonal(dim1=-2, dim2=-1))
    attn = torch.linalg.solve_triangular(L_mat, eye.expand_as(L_mat), upper=False)
    v_corrected = attn @ v_beta_c
    k_cumdecay = attn @ (k_beta_c * decay_exp)
    S = x["state"].to(torch.float32)
    causal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    o = torch.zeros_like(v_corrected)
    for i in range(L // chunk_size):
        q_i, k_i = q_c[:, :, i], k_c[:, :, i]
        intra = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(causal, 0)
        v_new = v_corrected[:, :, i] - k_cumdecay[:, :, i] @ S
        o[:, :, i] = (q_i * decay[:, :, i, :, None].exp()) @ S + intra @ v_new
        S = (
            S * decay[:, :, i, -1, None, None].exp()
            + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    return o.reshape(B, H, -1, o.shape[-1])[:, :, :T].transpose(1, 2).contiguous(), S


def test_seq_reference_diagonal_regularization_is_the_step_recurrence_only_at_alpha_zero() -> None:
    x = _s2g_rows(S2G_ROWS, seed=4)
    step_out, step_out_fp32, step_state = _step(x)
    exact_out, exact_state = _chunk_with_diagonal_regularization(x, alpha=0.0)
    _report("seq WY form alpha=0 vs step", exact_out, step_out_fp32, step_out, exact_state, step_state)
    fla_out, fla_state = _chunk(x, 32)
    _report("seq WY form alpha=0 vs FLA chunk", exact_out, fla_out, fla_out.to(torch.bfloat16), exact_state, fla_state)
    damped_out, damped_state = _chunk_with_diagonal_regularization(x, alpha=0.25)  # the module's default
    out_ulp = int(ulp_distance(damped_out, step_out, torch.bfloat16).max())
    state_rel = float(((damped_state - step_state).abs() / step_state.abs().clamp_min(1e-2)).max())
    print(f"seq WY form alpha=0.25 vs step: output max ULP bf16 {out_ulp}; state max rel {state_rel:.2e}")
    assert out_ulp > 4 and state_rel > 1e-2


def _gdn_weights() -> Qwen38GDNWeights:
    torch.manual_seed(5)
    dims = Qwen38GDNDimensions(
        hidden_size=2560, q_heads=16, value_heads=48, key_head_dim=128, value_head_dim=128, conv_kernel=4
    )
    bf16 = lambda *shape, scale=1.0: (torch.randn(*shape) * scale).to(torch.bfloat16)
    return Qwen38GDNWeights(
        layer_idx=0,
        dimensions=dims,
        rms_norm_eps=1e-6,
        output_gate="sigmoid",
        qkv=bf16(dims.qkv_width, 2560, scale=2560**-0.5),
        z=bf16(dims.value_width, 2560, scale=2560**-0.5),
        a=bf16(48, 2560, scale=2560**-0.5),
        b=bf16(48, 2560, scale=2560**-0.5),
        out=bf16(2560, dims.value_width, scale=dims.value_width**-0.5),
        conv=bf16(dims.qkv_width, 1, 4, scale=0.5),
        dt_bias=bf16(48, scale=0.1),
        A_log=bf16(48, scale=0.1),
        norm=bf16(128, scale=0.1),
    )


def test_gdn_oracle_five_rows_equals_five_sequential_steps() -> None:
    gdn = Qwen38GDN(_gdn_weights())
    hidden = torch.randn(1, S2G_ROWS, 2560).to(torch.bfloat16)
    state = Qwen38GDNState(
        conv=torch.randn(1, gdn.weights.dimensions.qkv_width, 3).to(torch.bfloat16),
        recurrent=torch.randn(1, 48, 128, 128) * 0.3,
    )
    with row_serial_torch():
        output, next_state = gdn.forward(hidden, state)
        rows = []
        threaded = state
        for r in range(S2G_ROWS):
            row, threaded = gdn.forward(hidden[:, r : r + 1], threaded)
            rows.append(row)
    assert torch.equal(output, torch.cat(rows, dim=1))
    assert torch.equal(next_state.conv, threaded.conv)
    assert torch.equal(next_state.recurrent, threaded.recurrent)


@pytest.mark.parametrize(("label", "dilation", "channels"), (("GDN FIR", 1, 2560), ("PLE conv", 3, 640)))
def test_conv_window_rows_give_the_outputs_and_the_next_history(label: str, dilation: int, channels: int) -> None:
    torch.manual_seed(6)
    taps = torch.randn(channels, 4).to(torch.bfloat16)
    state_len = 3 * dilation
    history = torch.randn(1, channels, state_len).to(torch.bfloat16)
    new_rows = torch.randn(1, channels, S2G_ROWS).to(torch.bfloat16)
    window = torch.cat([history, new_rows], dim=-1)  # [history | k+1 new rows], the design's W

    batched, _ = causal_depthwise_conv1d(new_rows, taps, history, dilation=dilation, activation="silu")
    for j in range(S2G_ROWS):
        # Output row j reads window rows j, j + d, j + 2d, j + 3d: the same op on the window slice starting at j.
        single, _ = causal_depthwise_conv1d(
            new_rows[..., j : j + 1], taps, window[..., j : j + state_len], dilation=dilation, activation="silu"
        )
        assert torch.equal(batched[..., j], single[..., 0]), (label, j)
    for committed in range(S2G_ROWS + 1):
        threaded = history
        for j in range(committed):
            _, threaded = causal_depthwise_conv1d(new_rows[..., j : j + 1], taps, threaded, dilation=dilation)
        # New history after committing c rows: window rows [c, c + state_len) (the design's W[a+1 : a+5] with c = a+1).
        assert torch.equal(threaded, window[..., committed : committed + state_len]), (label, committed)


# ============================================================================
# Step 2 rows (was test_mtp_v2_step2_rows_no_device.py)
# MTP v2 step 2 (a): the CPU oracle at R = 5 rows equals its per-row results, stage by stage.
#
# No device, synthetic weights at the model's widths.  GR read/write (full and TP4 forms), final mixer (full
# and TP4), MTP input mixer, PLE (post-lookup equations with the real n-gram hash and a fake table), LM head
# (sharded logits and the owner argmax), and MoE (router/top-k/shared/expert combine, full and EP4 forms).
# Row independence is the property the (k+1)-row verify body relies on; each stage is compared bitwise under
# ``row_serial_torch`` (torch's own bf16 GEMM/reduction blocking differs between M = 1 and M = 5 by up to
# 1 bf16 ULP, which is a torch artifact, not a coupling between rows).
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


S2R_ROWS = 5
HIDDEN = 2560
BRANCHES = 4
RANK = 320
EPS = 1e-6
S2R_TP = 4
VOCAB_SHARD = 2048  # synthetic LM head: 4 x 2048 rows
S2R_EOS = 248044


def _ranges(total: int) -> tuple[tuple[int, int], ...]:
    width = total // S2R_TP
    return tuple((d * width, (d + 1) * width) for d in range(S2R_TP))


def _placement(hidden: int, experts: int = 64) -> SimpleNamespace:
    config = SimpleNamespace(
        hidden_size=hidden,
        residual_branches=BRANCHES,
        residual_rank=RANK,
        residual_width=BRANCHES * hidden,
        rms_norm_eps=EPS,
        num_experts=experts,
        top_k=10,
        norm_topk_prob=True,
    )
    return SimpleNamespace(config=config, hidden_ranges=_ranges(hidden), expert_ranges=_ranges(experts))


def _s2r_bf16(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


def _assert_rows_equal(batched: torch.Tensor, per_row: list[torch.Tensor], label: str) -> None:
    stacked = torch.cat(per_row, dim=1)
    assert stacked.shape == batched.shape, (label, stacked.shape, batched.shape)
    assert stacked.dtype == batched.dtype, (label, stacked.dtype, batched.dtype)
    if not torch.equal(stacked, batched):
        gap = (stacked.float() - batched.float()).abs()
        raise AssertionError(f"{label}: rows differ, max abs {gap.max().item():.3e} at {gap.argmax().item()}")


@pytest.fixture(scope="module")
def residual() -> torch.Tensor:
    torch.manual_seed(1)
    return _s2r_bf16(1, S2R_ROWS, BRANCHES * HIDDEN)


@pytest.fixture(autouse=True)
def _row_serial():
    with row_serial_torch():
        yield


def test_gr_read_and_write_are_row_independent(residual) -> None:
    torch.manual_seed(2)
    weights = Qwen38GatedResidualWeights(
        placement=_placement(HIDDEN),
        layer_index=0,
        block="attn",
        norm=_s2r_bf16(BRANCHES * HIDDEN, scale=0.1),
        down=_s2r_bf16(RANK, BRANCHES * HIDDEN, scale=BRANCHES * HIDDEN**-0.5),
        up=_s2r_bf16(BRANCHES * HIDDEN, RANK, scale=RANK**-0.5),
        inject=_s2r_bf16(BRANCHES, BRANCHES * HIDDEN, scale=(BRANCHES * HIDDEN) ** -0.5),
    )
    gr = Qwen38GatedResidual(weights)
    block_output = _s2r_bf16(1, S2R_ROWS, HIDDEN)

    block_input, state = gr.read(residual)
    written = gr.write(block_output, state)
    rows = [gr.read(residual[:, r : r + 1]) for r in range(S2R_ROWS)]
    _assert_rows_equal(block_input, [row[0] for row in rows], "GR read block input")
    _assert_rows_equal(state.injection, [row[1].injection for row in rows], "GR read injection")
    _assert_rows_equal(written, [gr.write(block_output[:, r : r + 1], rows[r][1]) for r in range(S2R_ROWS)], "GR write")

    shards = gr.shard_residual(residual)
    block_shards, tp_state = gr.read_tp4(shards)
    output_shards = gr.shard_hidden(block_output)
    written_shards = gr.write_tp4(output_shards, tp_state)
    for d in range(S2R_TP):
        per_row = []
        per_row_written = []
        for r in range(S2R_ROWS):
            row_shards = tuple(shard[:, r : r + 1] for shard in shards)
            row_blocks, row_state = gr.read_tp4(row_shards)
            per_row.append(row_blocks[d])
            per_row_written.append(gr.write_tp4(tuple(s[:, r : r + 1] for s in output_shards), row_state)[d])
        _assert_rows_equal(block_shards[d], per_row, f"GR TP4 read shard {d}")
        _assert_rows_equal(written_shards[d], per_row_written, f"GR TP4 write shard {d}")
    _assert_rows_equal(
        tp_state.injection,
        [gr.read_tp4(tuple(s[:, r : r + 1] for s in shards))[1].injection for r in range(S2R_ROWS)],
        "GR TP4 injection",
    )
    assert torch.equal(gr.combine_residual_shards(written_shards), written)


def test_final_mixer_is_row_independent(residual) -> None:
    torch.manual_seed(3)
    weights = Qwen38FinalMixerWeights(
        placement=_placement(HIDDEN),
        norm=_s2r_bf16(BRANCHES * HIDDEN, scale=0.1),
        down=_s2r_bf16(RANK, BRANCHES * HIDDEN, scale=(BRANCHES * HIDDEN) ** -0.5),
        up=_s2r_bf16(BRANCHES * HIDDEN, RANK, scale=RANK**-0.5),
    )
    mixer = Qwen38FinalMixer(weights)
    _assert_rows_equal(mixer(residual), [mixer(residual[:, r : r + 1]) for r in range(S2R_ROWS)], "final mixer")
    shards = mixer.shard_residual(residual)
    outputs = mixer.forward_tp4(shards)
    for d in range(S2R_TP):
        per_row = [mixer.forward_tp4(tuple(s[:, r : r + 1] for s in shards))[d] for r in range(S2R_ROWS)]
        _assert_rows_equal(outputs[d], per_row, f"final mixer TP4 shard {d}")


def test_mtp_input_mixer_is_row_independent(residual) -> None:
    torch.manual_seed(4)
    embedding = _s2r_bf16(1, S2R_ROWS, HIDDEN)
    weights = dict(
        embedding_norm_weight=_s2r_bf16(HIDDEN, scale=0.1),
        hidden_norm_weight=_s2r_bf16(BRANCHES * HIDDEN, scale=0.1),
        fc_embedding_weight=_s2r_bf16(HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        fc_hidden_weight=_s2r_bf16(HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        hc_count=BRANCHES,
        hidden_size=HIDDEN,
        eps=EPS,
    )
    mixed = mtp_input_fusion(embedding, residual, **weights)
    per_row = [mtp_input_fusion(embedding[:, r : r + 1], residual[:, r : r + 1], **weights) for r in range(S2R_ROWS)]
    _assert_rows_equal(mixed, per_row, "MTP input mixer")


class _FakeHostPLEEmbedding:
    """The real n-gram hash and context threading over a small fake table (the checkpoint table is 360 GB away)."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(ngram_size=3, eos_token_id=S2R_EOS, ple_checkpoint_layer=1)
        self.spec = build_ngram_hash_spec(
            unigram_vocab_size=248320,
            ngram_size=3,
            heads_per_ngram=8,
            ngram_vocab_size_base=20_000_000,
            ple_layer_index=0,
            seed=1234,
            divisible_by=128,
        )
        self.embedding_head_dim = HIDDEN // 16
        self.table = _s2r_bf16(4096, self.embedding_head_dim)

    def lookup(self, input_ids: torch.Tensor, previous_context: torch.Tensor | None):
        ids, next_context = ngram_token_ids(input_ids, previous_context, S2R_EOS, self.spec)
        rows = self.table[ids.reshape(-1) % self.table.shape[0]]
        return rows.reshape(*ids.shape, self.embedding_head_dim).flatten(-2), next_context


def test_ple_rows_equal_sequential_steps_with_threaded_state(residual) -> None:
    torch.manual_seed(5)
    weights = Qwen38PLEWeights(
        layer_idx=1,
        hidden_size=HIDDEN,
        residual_branches=BRANCHES,
        embedding_width=HIDDEN,
        conv_kernel=4,
        conv_dilation=3,
        rms_norm_eps=EPS,
        key=_s2r_bf16(BRANCHES * HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        value=_s2r_bf16(HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        norm_key=_s2r_bf16(BRANCHES * HIDDEN, scale=0.1),
        norm_query=_s2r_bf16(BRANCHES * HIDDEN, scale=0.1),
        norm_conv=_s2r_bf16(BRANCHES * HIDDEN, scale=0.1),
        conv=_s2r_bf16(BRANCHES * HIDDEN, 1, 4, scale=0.5),
    )
    ple = Qwen38PLE(_FakeHostPLEEmbedding(), weights)
    input_ids = torch.tensor([[17, 15, 16, 95859, 20]], dtype=torch.long)
    state = Qwen38PLEState(
        token_context=torch.tensor([[S2R_EOS, 12]], dtype=torch.long),
        conv=_s2r_bf16(1, BRANCHES * HIDDEN, 9),
    )

    output, next_state = ple.forward(residual, input_ids, state)
    per_row = []
    threaded = state
    for r in range(S2R_ROWS):
        row_output, threaded = ple.forward(residual[:, r : r + 1], input_ids[:, r : r + 1], threaded)
        per_row.append(row_output)
    _assert_rows_equal(output, per_row, "PLE output")
    assert torch.equal(next_state.token_context, threaded.token_context)
    assert torch.equal(next_state.conv, threaded.conv)
    # The conv state after R rows is the last 9 rows of [history | normalized new rows], as the design's window select.
    assert next_state.conv.shape == (1, BRANCHES * HIDDEN, 9)


def _sharded_greedy(hidden: torch.Tensor, head_shards: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """LM head per vocabulary shard, local argmax, owner = first shard holding the maximum (resolve_greedy)."""

    logits = torch.cat([F.linear(hidden, shard) for shard in head_shards], dim=-1)
    values, indices = zip(*(F.linear(hidden, shard).max(dim=-1) for shard in head_shards))
    values = torch.stack(values, dim=-1).float()
    indices = torch.stack(indices, dim=-1)
    owner = values.argmax(dim=-1, keepdim=True)
    starts = torch.arange(S2R_TP) * VOCAB_SHARD
    return logits, (indices.gather(-1, owner) + starts[owner]).squeeze(-1)


def test_lm_head_logits_and_owner_argmax_are_row_independent() -> None:
    torch.manual_seed(6)
    hidden = _s2r_bf16(1, S2R_ROWS, HIDDEN)
    head_shards = [_s2r_bf16(VOCAB_SHARD, HIDDEN, scale=HIDDEN**-0.5) for _ in range(S2R_TP)]
    # Row 3: copy the winning head row into the next shard, so two owners tie at the maximum.
    logits, _ = _sharded_greedy(hidden, head_shards)
    winner = int(logits[0, 3].argmax())
    head_shards[(winner // VOCAB_SHARD + 1) % S2R_TP][winner % VOCAB_SHARD] = head_shards[winner // VOCAB_SHARD][
        winner % VOCAB_SHARD
    ]
    logits, tokens = _sharded_greedy(hidden, head_shards)
    per_row = [_sharded_greedy(hidden[:, r : r + 1], head_shards) for r in range(S2R_ROWS)]
    _assert_rows_equal(logits, [row[0] for row in per_row], "LM head logits")
    assert tokens.shape == (1, S2R_ROWS)
    assert tokens[0].tolist() == [int(row[1]) for row in per_row]
    # The tie on row 3 resolves to the lowest owner in both forms.
    row3 = logits[0, 3]
    tied = torch.nonzero(row3 == row3.max()).reshape(-1)
    assert tied.numel() >= 2 and int(tokens[0, 3]) == int(tied.min())


class _FakeMoEWeights:
    """Router, shared expert and per-expert weights at a small width; the oracle math is width-generic."""

    def __init__(self, hidden: int, intermediate: int, experts: int) -> None:
        self.placement = _placement(hidden, experts)
        self.config = self.placement.config
        self.expert_ranges = self.placement.expert_ranges
        self.router_weight = _s2r_bf16(experts, hidden, scale=hidden**-0.5)
        self.shared_gate_proj = _s2r_bf16(intermediate, hidden, scale=hidden**-0.5)
        self.shared_up_proj = _s2r_bf16(intermediate, hidden, scale=hidden**-0.5)
        self.shared_down_proj = _s2r_bf16(hidden, intermediate, scale=intermediate**-0.5)
        self.shared_scalar_gate = _s2r_bf16(1, hidden, scale=hidden**-0.5)
        self.experts = [
            Qwen38ExpertWeights(
                _s2r_bf16(2 * intermediate, hidden, scale=hidden**-0.5),
                _s2r_bf16(hidden, intermediate, scale=intermediate**-0.5),
                intermediate,
            )
            for _ in range(experts)
        ]

    def owner(self, expert_index: int) -> int:
        return next(d for d, (start, end) in enumerate(self.expert_ranges) if start <= expert_index < end)

    def expert(self, expert_index: int) -> Qwen38ExpertWeights:
        return self.experts[expert_index]

    def shared_shard(self, device_index: int) -> Qwen38SharedExpertShard:
        width = self.shared_gate_proj.shape[0] // S2R_TP
        start, end = device_index * width, (device_index + 1) * width
        hidden_start, hidden_end = self.placement.hidden_ranges[device_index]
        return Qwen38SharedExpertShard(
            gate_proj=self.shared_gate_proj[start:end],
            up_proj=self.shared_up_proj[start:end],
            down_proj=self.shared_down_proj[:, start:end],
            scalar_gate=self.shared_scalar_gate[:, hidden_start:hidden_end],
        )


def test_moe_router_topk_shared_and_expert_combine_are_row_independent() -> None:
    torch.manual_seed(8)
    moe = Qwen38MoE(_FakeMoEWeights(hidden=512, intermediate=128, experts=64))
    hidden = _s2r_bf16(1, S2R_ROWS, 512)

    output, routing = moe(hidden)
    per_row = [moe(hidden[:, r : r + 1]) for r in range(S2R_ROWS)]
    _assert_rows_equal(output, [row[0] for row in per_row], "MoE output")
    assert torch.equal(routing.indices, torch.cat([row[1].indices for row in per_row], dim=0))
    assert torch.equal(routing.scores, torch.cat([row[1].scores for row in per_row], dim=0))
    assert torch.equal(routing.logits, torch.cat([row[1].logits for row in per_row], dim=0))
    # The five rows touch more distinct experts than any single row: the R1 cost driver, not a numerics change.
    assert len(set(routing.indices.reshape(-1).tolist())) > 10

    ep4 = moe.expert_parallel_forward(hidden)
    per_row_ep4 = [moe.expert_parallel_forward(hidden[:, r : r + 1]) for r in range(S2R_ROWS)]
    for d in range(S2R_TP):
        _assert_rows_equal(ep4.hidden_shards[d], [row.hidden_shards[d] for row in per_row_ep4], f"MoE EP4 shard {d}")
        # Local selection over five rows is the union of the per-row selections, in row order.
        assert ep4.local_selected_experts[d] == tuple(
            expert for row in per_row_ep4 for expert in row.local_selected_experts[d]
        )
    # The score-weighted combine (the device fast-reduce input form [TOP_K, R, H]) is per row.
    expert_outputs = _s2r_bf16(S2R_ROWS, 10, 512)
    combined = weighted_routed_reduce(expert_outputs, routing.scores)
    _assert_rows_equal(
        combined.unsqueeze(0),
        [
            weighted_routed_reduce(expert_outputs[r : r + 1], routing.scores[r : r + 1]).unsqueeze(0)
            for r in range(S2R_ROWS)
        ],
        "weighted routed reduce",
    )


# ============================================================================
# Step 4 rows (was test_mtp_v2_step4_rows_no_device.py)
# MTP v2 step 4: the GDN and PLE multi-row device paths against their 1-row paths, on a torch-backed ttnn.
#
# No device.  ``ttnn`` is replaced in ``ttnn/gdn.py``, ``ttnn/ple.py`` and ``ttnn/layer.py`` by a mesh-of-four
# torch model of the op surface the modules use (elementwise ops in the operand dtype, row-serial matmuls,
# norms and transcendental ops so a row's result does not depend on how many rows share its tensor, collectives
# over the four locals).  The chunk kernel is pluggable: the step form (the 1-row path's fp32 arithmetic) shows
# that everything around the recurrence is bitwise the 1-row path at R = 1; the torch chunk reference (fp32 WY
# form, bf16 q scaling as the composite does it) shows R = 5 rows against five sequential steps and every
# commit(a) against the a + 1 committed steps.  The source pins hold the trace contract of the new methods
# (one full chunk, device-resident constants, no host upload, no per-pass host ints, the 1-row bodies untouched).
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


ROOT = Path(__file__).resolve().parents[1]
GDN_SOURCE = ROOT / "ttnn" / "gdn.py"
PLE_SOURCE = ROOT / "ttnn" / "ple.py"
LAYER_SOURCE = ROOT / "ttnn" / "layer.py"
TP = 4
ROWS = 5
HEADS = 12
HEAD_DIM = 128


# --------------------------------------------------------------------------- torch-backed ttnn


class _DType:
    def __init__(self, name: str, torch_dtype: torch.dtype) -> None:
        self.name, self.torch = name, torch_dtype

    def __repr__(self) -> str:
        return self.name


BF16 = _DType("bfloat16", torch.bfloat16)
FP32 = _DType("float32", torch.float32)
TILE, ROW_MAJOR = "TILE_LAYOUT", "ROW_MAJOR_LAYOUT"
_ids = itertools.count(1)


class PlacementReplicate:
    pass


class PlacementShard:
    def __init__(self, dim: int) -> None:
        self.dim = dim


class TensorTopology:
    def __init__(self, distribution_shape, placements, mesh_coords) -> None:
        self._shape, self._placements, self._coords = tuple(distribution_shape), list(placements), list(mesh_coords)

    def distribution_shape(self):
        return self._shape

    def placements(self):
        return self._placements

    def mesh_coords(self):
        return self._coords


def _topology(shard_dim: int | None) -> TensorTopology:
    second = PlacementReplicate() if shard_dim is None else PlacementShard(shard_dim)
    return TensorTopology((1, TP), [PlacementReplicate(), second], [(0, column) for column in range(TP)])


class FakeTensor:
    """One tensor per mesh device; ``locals`` are the four torch shards."""

    def __init__(self, locals_, dtype: _DType, layout: str = TILE, shard_dim: int | None = None) -> None:
        self.locals = list(locals_)
        assert len(self.locals) == TP and all(t.dtype == dtype.torch for t in self.locals), dtype
        self.dtype, self.layout = dtype, layout
        self.topology = _topology(shard_dim)
        self.tensor_id = next(_ids)
        self.alive = True

    @property
    def shape(self):
        return tuple(self.locals[0].shape)

    def device(self):
        return "mesh"

    def memory_config(self):
        return "DRAM_MEMORY_CONFIG"

    def tensor_topology(self):
        return self.topology

    def update_tensor_topology(self, topology) -> None:
        self.topology = topology

    def torch_shards(self) -> list[torch.Tensor]:
        assert self.alive, "read of a deallocated tensor"
        return self.locals


def _rowwise(fn, x: torch.Tensor) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    return torch.stack([fn(row) for row in rows]).reshape(x.shape)


def _unary(fn, dtype_keep=True):
    def op(tensor, *args, memory_config=None, output_tensor=None, **kwargs):
        results = [_rowwise(fn, t) for t in tensor.torch_shards()]
        if output_tensor is not None:
            assert output_tensor.dtype is tensor.dtype and output_tensor.shape == tuple(results[0].shape)
            for target, result in zip(output_tensor.locals, results):
                target.copy_(result)
            return output_tensor
        return FakeTensor(results, tensor.dtype, tensor.layout)

    return op


def _softplus(tensor, *, beta, threshold, memory_config=None):
    """``ttnn.softplus`` as fp32 torch softplus; the device op's SFPU error (4.7e-4 relative, never a flush to 0 on
    4x p150) is below the 1e-3 envelope the gate probe asserts, so the fake stands in for it honestly."""

    assert tensor.dtype is FP32 and (beta, threshold) == (1.0, 20.0), (tensor.dtype, beta, threshold)
    return FakeTensor(
        [_rowwise(lambda row: F.softplus(row, beta, threshold), t) for t in tensor.torch_shards()], FP32, tensor.layout
    )


def _binary(torch_op):
    def op(
        a,
        b,
        *,
        memory_config=None,
        output_tensor=None,
        dtype=None,
        input_tensor_b_activations=None,
        activations=None,
        fast_and_approximate_mode=None,
    ):
        # Mixed float dtypes compute in fp32; the output keeps input_a's dtype unless ``dtype`` says otherwise.
        out_dtype = dtype or a.dtype
        compute = (
            FP32 if dtype is FP32 or a.dtype is FP32 or (isinstance(b, FakeTensor) and b.dtype is FP32) else a.dtype
        )
        results = []
        for index, x in enumerate(a.torch_shards()):
            y = b.torch_shards()[index] if isinstance(b, FakeTensor) else b
            if isinstance(y, torch.Tensor):
                if input_tensor_b_activations:
                    assert input_tensor_b_activations == ["EXP"]
                    y = torch.exp(y)
                x, y = x.to(compute.torch), y.to(compute.torch)
            result = torch_op(x, y).to(out_dtype.torch)
            # No fused output activation is modeled: the device's fused SOFTPLUS flushed to 0 below about -5.
            assert not activations, activations
            results.append(result)
        if output_tensor is not None:
            assert output_tensor.dtype is out_dtype and output_tensor.shape == tuple(results[0].shape), (
                output_tensor.shape,
                tuple(results[0].shape),
            )
            for target, result in zip(output_tensor.locals, results):
                target.copy_(result)
            return output_tensor
        return FakeTensor(results, out_dtype, a.layout)

    return op


def _compare(torch_op):
    def op(a, b, *, memory_config=None, dtype=None):
        out_dtype = dtype or a.dtype
        results = []
        for index, x in enumerate(a.torch_shards()):
            y = b.torch_shards()[index] if isinstance(b, FakeTensor) else b
            results.append(torch_op(x, y).to(out_dtype.torch))
        return FakeTensor(results, out_dtype, a.layout)

    return op


def _mapped(fn):
    def op(tensor, *args, memory_config=None, **kwargs):
        return FakeTensor([fn(t, *args, **kwargs) for t in tensor.torch_shards()], tensor.dtype, tensor.layout)

    return op


def _slice(tensor, start, end, *, memory_config=None, output_tensor=None):
    index = tuple(slice(s, e) for s, e in zip(start, end))
    pieces = [t[index].clone() for t in tensor.torch_shards()]
    if output_tensor is not None:
        for target, piece in zip(output_tensor.locals, pieces):
            assert target.shape == piece.shape, (target.shape, piece.shape)
            target.copy_(piece)
        return output_tensor
    return FakeTensor(pieces, tensor.dtype, tensor.layout)


def _pad(tensor, padding, value=0.0, *, memory_config=None):
    flat = [amount for pair in reversed(padding) for amount in pair]
    return FakeTensor([F.pad(t, flat, value=value) for t in tensor.torch_shards()], tensor.dtype, tensor.layout)


def _concat(tensors, dim, *, memory_config=None):
    return FakeTensor(
        [torch.cat([t.torch_shards()[index] for t in tensors], dim=dim) for index in range(TP)],
        tensors[0].dtype,
        tensors[0].layout,
    )


def _linear_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    out = torch.stack([(row[None].float() @ w.float())[0] for row in rows])
    return out.reshape(*x.shape[:-1], w.shape[-1]).to(x.dtype)


def _linear(a, w, *, memory_config=None, program_config=None, compute_kernel_config=None):
    return FakeTensor(
        [_linear_rows(x, wt.reshape(wt.shape[-2], wt.shape[-1])) for x, wt in zip(a.torch_shards(), w.torch_shards())],
        a.dtype,
        a.layout,
    )


def _matmul(
    a,
    b,
    *,
    memory_config=None,
    dtype=None,
    program_config=None,
    compute_kernel_config=None,
    optional_output_tensor=None,
):
    """fp32 accumulation over the products, one rounding into the output dtype (the fp32_dest_acc_en form)."""

    out_dtype = dtype or a.dtype
    results = [
        torch.matmul(x.float(), y.float()).to(out_dtype.torch) for x, y in zip(a.torch_shards(), b.torch_shards())
    ]
    if optional_output_tensor is not None:
        assert optional_output_tensor.dtype is out_dtype and optional_output_tensor.shape == tuple(results[0].shape)
        for target, result in zip(optional_output_tensor.locals, results):
            target.copy_(result)
        return optional_output_tensor
    return FakeTensor(results, out_dtype, a.layout)


def _tile_pages(x: torch.Tensor) -> torch.Tensor:
    """The TILE-layout page order of a logical tensor: batch dims, then the 32x32 tiles row by row (zero padded)."""

    *batch, rows, cols = x.shape
    padded_rows, padded_cols = -(-rows // 32) * 32, -(-cols // 32) * 32
    padded = F.pad(x.reshape(-1, rows, cols), (0, padded_cols - cols, 0, padded_rows - rows))
    tiles = padded.reshape(-1, padded_rows // 32, 32, padded_cols // 32, 32).permute(0, 1, 3, 2, 4)
    return tiles.reshape(-1, 32, 32)


def _from_tile_pages(pages: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    *batch, rows, cols = shape
    padded_rows, padded_cols = -(-rows // 32) * 32, -(-cols // 32) * 32
    count = 1
    for size in batch:
        count *= size
    assert pages.shape[0] == count * (padded_rows // 32) * (padded_cols // 32), (pages.shape, shape)
    tiles = pages.reshape(count, padded_rows // 32, padded_cols // 32, 32, 32).permute(0, 1, 3, 2, 4)
    return tiles.reshape(count, padded_rows, padded_cols)[:, :rows, :cols].reshape(shape).clone()


def _view(tensor, shape, *, memory_config=None):
    """``ttnn.experimental.view``: the same device pages read back under a new shape (a metadata change), i.e.
    the tile pages of the input reinterpreted in the output's tile grid.  Modelled faithfully so a view whose
    logical (row-major) meaning differs from a torch reshape is caught."""

    return FakeTensor(
        [_from_tile_pages(_tile_pages(x), tuple(shape)) for x in tensor.torch_shards()], tensor.dtype, tensor.layout
    )


def _rms_norm(tensor, *, epsilon, weight=None, memory_config=None):
    def per_device(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        def norm_row(row: torch.Tensor) -> torch.Tensor:
            row32 = row.float()
            y = row32 * torch.rsqrt(row32.square().mean() + epsilon)
            return y if w is None else y * w.reshape(-1).float()

        return _rowwise(norm_row, x).to(x.dtype)

    weights = weight.torch_shards() if weight is not None else [None] * TP
    return FakeTensor([per_device(x, w) for x, w in zip(tensor.torch_shards(), weights)], tensor.dtype, tensor.layout)


def _all_gather(tensor, *, dim, cluster_axis, memory_config=None):
    full = torch.cat(tensor.torch_shards(), dim=dim)
    return FakeTensor([full.clone() for _ in range(TP)], tensor.dtype, tensor.layout)


def _reduce_scatter(tensor, *, dim, cluster_axis, memory_config=None, topology=None):
    total = sum(t.float() for t in tensor.torch_shards()).to(tensor.dtype.torch)
    return FakeTensor([piece.clone() for piece in torch.chunk(total, TP, dim=dim)], tensor.dtype, tensor.layout, dim)


def _pre_all_gather(tensor, *, dtype, memory_config=None, compute_kernel_config=None):
    def stats(x: torch.Tensor) -> torch.Tensor:
        sums = x.float().square().sum(dim=-1, keepdim=True)  # one stat per (token, branch) row
        return F.pad(sums, (0, 31))

    return FakeTensor([stats(x) for x in tensor.torch_shards()], FP32, tensor.layout)


def _post_all_gather(tensor, gathered, *, epsilon, weight, memory_config=None, compute_kernel_config=None, dtype):
    outputs = []
    for x, stats, w in zip(tensor.torch_shards(), gathered.torch_shards(), weight.torch_shards()):
        total = stats[..., ::32].sum(dim=-1, keepdim=True)  # the four devices' column-0 stats
        mean = total / (TP * x.shape[-1])
        outputs.append((x.float() * torch.rsqrt(mean + epsilon) * w.float()).to(dtype.torch))
    return FakeTensor(outputs, dtype, tensor.layout)


def _sum_last(tensor, dim, keepdim, *, memory_config=None, compute_kernel_config=None):
    assert dim in (3, -1) and keepdim
    return FakeTensor(
        [
            torch.stack([row.sum() for row in x.reshape(-1, x.shape[-1])]).reshape(*x.shape[:-1], 1)
            for x in tensor.torch_shards()
        ],
        tensor.dtype,
        tensor.layout,
    )


def _from_torch(host, *, dtype, layout, device, memory_config, mesh_mapper):
    host = host.to(dtype.torch)
    if mesh_mapper == "replicate":
        return FakeTensor([host.clone() for _ in range(TP)], dtype, layout)
    kind, dims = mesh_mapper
    assert kind == "shard"
    return FakeTensor([piece.clone() for piece in torch.chunk(host, TP, dim=dims[1])], dtype, layout, dims[1])


def _moreh_full(shape, value, device, *, dtype, layout, memory_config):
    return FakeTensor([torch.full(tuple(shape), value, dtype=dtype.torch) for _ in range(TP)], dtype, layout)


def _get_device_tensors(tensor):
    return [FakeTensor([t] * TP, tensor.dtype, tensor.layout) for t in tensor.torch_shards()]


def _copy(source, target):
    for s, t in zip(source.torch_shards(), target.locals):
        t.copy_(s)
    return target


def _deallocate(tensor):
    assert tensor.alive, "double deallocation"
    tensor.alive = False


class FakeChunk:
    """``ttnn.transformer.chunk_gated_delta_rule`` with a pluggable per-device kernel model."""

    def __init__(self) -> None:
        self.impl = torch_chunk_kernel
        self.calls: list[dict] = []

    def chunk_gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
        eye,
        tril,
        ones,
        masks,
        output_head_major=False,
    ):
        # v may arrive rank-3 token-major flat [B, T, HV * V] (the composite's flat-v form: the prep reader
        # addresses head h's tiles at columns 128h..; no head split), or head-split [B, T, HV, V].
        flat_v = len(v.shape) == 3
        heads_v = HEADS if flat_v else v.shape[2]
        self.calls.append(
            {
                "chunk_size": chunk_size,
                "rows": q.shape[1],
                "heads": (q.shape[2], k.shape[2], heads_v),
                "flat_v": flat_v,
                "output_head_major": output_head_major,
                "constants": (eye, tril, ones, masks),
            }
        )
        # T is one full chunk (32) or the long prefill chunk's four chunks (128): pad == 0 either way.
        rows = q.shape[1]
        assert output_final_state and chunk_size == 32 and rows in (32, 128), (chunk_size, q.shape)
        assert q.dtype is BF16 and k.dtype is BF16 and v.dtype is BF16 and g.dtype is FP32 and beta.dtype is FP32
        assert initial_state.dtype is FP32
        if flat_v:
            assert v.shape == (1, rows, HEADS * HEAD_DIM), v.shape
        outputs, states = [], []
        for index in range(TP):
            o, s = self.impl(
                q.torch_shards()[index],
                k.torch_shards()[index],
                v.torch_shards()[index].reshape(1, rows, HEADS, HEAD_DIM) if flat_v else v.torch_shards()[index],
                g.torch_shards()[index],
                beta.torch_shards()[index],
                scale,
                initial_state.torch_shards()[index],
            )
            outputs.append(o)
            states.append(s)
        if output_head_major:
            # [B*HV, T, V] TILE: the kernel's own head-major layout, fp32 on the pinned runtime.
            heads = [o.permute(0, 2, 1, 3).reshape(HEADS, rows, HEAD_DIM).contiguous() for o in outputs]
            return FakeTensor(heads, FP32, TILE), FakeTensor(states, FP32, TILE)
        return FakeTensor(outputs, FP32, ROW_MAJOR), FakeTensor(states, FP32, TILE)  # fp32 token-major, as on device


def torch_chunk_kernel(q, k, v, g, beta, scale, state):
    """The kernel's algorithm in torch: bf16 q scaled in bf16 (the composite's ``multiply(q, scale)``), GQA
    expand of the 4 key heads to the 12 value heads, fp32 WY chunk of 32 rows from ``state``; fp32 output
    (the kernel's output dtype on the pinned runtime, measured on 4x p150)."""

    q_scaled = (q.float() * scale).to(torch.bfloat16).float().repeat_interleave(HEADS // q.shape[2], dim=2)
    k_full = k.float().repeat_interleave(HEADS // k.shape[2], dim=2)
    output, final = chunk_gated_delta_rule(
        q_scaled,
        k_full,
        v.float(),
        g.float(),
        beta.float(),
        chunk_size=32,
        scale=1.0,
        initial_state=state,
        output_final_state=True,
    )
    return output.float(), final


def step_chunk_kernel(q, k, v, g, beta, scale, state):
    """Row 0 through the 1-row path's fp32 step arithmetic (the same torch calls the fake step ops make); rows
    past 0 are zero.  Used to show that everything around the recurrence is bitwise the 1-row path."""

    q_row = q[:, 0].repeat_interleave(HEADS // q.shape[2], dim=1).reshape(1, HEADS, 1, HEAD_DIM).float() * scale
    k_row = k[:, 0].repeat_interleave(HEADS // k.shape[2], dim=1).reshape(1, HEADS, 1, HEAD_DIM).float()
    v_row = v[:, 0].reshape(1, HEADS, 1, HEAD_DIM)
    beta_row = beta[:, 0].reshape(1, HEADS, 1, 1)
    g_row = g[:, 0].reshape(1, HEADS, 1, 1)
    decayed = state * torch.exp(g_row)
    v_read = torch.matmul(k_row, decayed)
    delta = v_row.to(torch.float32) - v_read
    outer = torch.matmul(k_row.transpose(2, 3), delta)
    new_state = decayed + outer * beta_row
    o_row = torch.matmul(
        q_row, new_state
    )  # [1, HEADS, 1, HEAD_DIM] fp32; the rows path casts to bf16 as the 1-row path does
    output = torch.zeros(1, q.shape[1], HEADS, HEAD_DIM, dtype=torch.float32)
    output[:, 0] = o_row[:, :, 0]
    return output, new_state


def make_fake_ttnn(chunk: FakeChunk) -> SimpleNamespace:
    return SimpleNamespace(
        bfloat16=BF16,
        float32=FP32,
        TILE_LAYOUT=TILE,
        ROW_MAJOR_LAYOUT=ROW_MAJOR,
        TILE_SIZE=32,
        DRAM_MEMORY_CONFIG="DRAM_MEMORY_CONFIG",
        L1_MEMORY_CONFIG="L1",
        L1_WIDTH_SHARDED_MEMORY_CONFIG="L1WS",
        Topology=SimpleNamespace(Linear="linear"),
        UnaryOpType=SimpleNamespace(EXP="EXP"),
        UnaryWithParam=lambda op, a, b: (op, a, b),
        MathFidelity=SimpleNamespace(HiFi4="HiFi4", HiFi2="HiFi2"),
        WormholeComputeKernelConfig=lambda **fields: ("compute_kernel_config", tuple(sorted(fields.items()))),
        experimental=SimpleNamespace(view=_view),
        TensorTopology=TensorTopology,
        PlacementReplicate=PlacementReplicate,
        PlacementShard=PlacementShard,
        ShardTensor2dMesh=lambda device, mesh_shape, dims: ("shard", dims),
        from_torch=_from_torch,
        moreh_full=_moreh_full,
        get_device_tensors=_get_device_tensors,
        deallocate=_deallocate,
        copy=_copy,
        all_gather=_all_gather,
        reduce_scatter=_reduce_scatter,
        linear=_linear,
        matmul=_matmul,
        to_memory_config=lambda t, cfg: FakeTensor([x.clone() for x in t.torch_shards()], t.dtype, t.layout),
        to_layout=lambda t, layout, memory_config=None: FakeTensor(
            [x.clone() for x in t.torch_shards()], t.dtype, layout
        ),
        slice=_slice,
        pad=_pad,
        concat=_concat,
        reshape=lambda t, shape, pad_value=None, memory_config=None: FakeTensor(
            [x.reshape(tuple(shape)).clone() for x in t.torch_shards()], t.dtype, t.layout
        ),
        repeat_interleave=_mapped(lambda x, n, dim: torch.repeat_interleave(x, n, dim=dim)),
        repeat=_mapped(lambda x, multipliers: x.repeat(*multipliers)),
        transpose=_mapped(lambda x, a, b: x.transpose(a, b).contiguous()),
        permute=_mapped(lambda x, dims: x.permute(*dims).contiguous()),
        typecast=lambda t, dtype, memory_config=None: FakeTensor(
            [x.to(dtype.torch) for x in t.torch_shards()], dtype, t.layout
        ),
        multiply=_binary(torch.mul),
        add=_binary(torch.add),
        subtract=_binary(torch.sub),
        mac=lambda a, b, c: _binary(torch.add)(_binary(torch.mul)(a, b), c),
        le=_compare(torch.le),
        eq=_compare(torch.eq),
        sigmoid=_unary(torch.sigmoid),
        softplus=_softplus,
        silu=_unary(F.silu),
        sqrt=_unary(torch.sqrt),
        abs=_unary(torch.abs),
        sign=_unary(torch.sign),
        clamp=lambda t, min=None, max=None, memory_config=None: FakeTensor(
            [x.clamp(min=min, max=max) for x in t.torch_shards()], t.dtype, t.layout
        ),
        sum=_sum_last,
        rms_norm=_rms_norm,
        rms_norm_pre_all_gather=_pre_all_gather,
        rms_norm_post_all_gather=_post_all_gather,
        transformer=chunk,
    )


def pin_stock_decode_paths(monkeypatch, fake_ttnn):
    """Route the default-on fused decode sites' stock fallbacks to the fake ttnn.  The row_gather site
    (``fused.resolve_admitted("row_gather")``) and ``edge_trim.hidden_gather`` fall back to the stock all_gather through
    their own ``import ttnn``, i.e. the real op on fake tensors.  Fake tensors fail ``row_gather.admits`` (the call
    is wrapped in try/except), so the stock all_gather of the fake serves, as the inline call did before they existed.
    """

    monkeypatch.setenv(
        "QWEN38_MM_STREAM2", "0"
    )  # default on since 996ae0c00e2; its stock fallback calls the real ttnn.linear
    monkeypatch.setenv("QWEN38_EDGE_TRIM", "0")  # the merged PLE key|value matmul / embed reduce are device-only
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import edge_trim, mm_stream, row_gather, row_reduce

    monkeypatch.setattr(row_gather, "ttnn", fake_ttnn)
    monkeypatch.setattr(row_reduce, "ttnn", fake_ttnn)  # row_reduce_or_reduce_scatter's stock reduce_scatter
    monkeypatch.setattr(mm_stream, "ttnn", fake_ttnn)  # mm_stream.linear: not admitted for fakes -> the stock linear
    monkeypatch.setattr(
        edge_trim,
        "hidden_gather",
        lambda x, *, memory_config: fake_ttnn.all_gather(x, dim=3, cluster_axis=1, memory_config=memory_config),
    )


def install_source_chunk_fake(monkeypatch, chunk):
    """Composition tests replace only the numerical recurrence boundary.

    The real model-owned descriptors have independent device/source-word tests.
    This retains the host oracle for projection, history, masks and ownership.
    """

    def source_chunk(q, k, v, g, beta, initial_state, constants, *, rows_total, scale):
        assert q.shape[1] == rows_total
        eye, tril, ones, masks = constants
        return chunk.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=32,
            output_head_major=True,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )

    monkeypatch.setattr(gdn_module.fused.gdn_source_chunk, "chunk_token_major", source_chunk)
    pin_stock_decode_paths(monkeypatch, make_fake_ttnn(chunk))


class FakeContract:
    """Shape/topology validation is the modules' own ``_require_shape``; placement checks are no-ops here."""

    mesh_shape = (1, TP)
    physical_ids = (0, 1, 2, 3)

    def validate_mesh(self, mesh_device) -> None:
        pass

    def validate_tensor(self, tensor, *, placement, shard_dim=None, require_device=True) -> None:
        assert tensor.alive

    def mark_local_partial(self, tensor, *, replicated_reference, expected_shape) -> None:
        assert tensor.shape == tuple(expected_shape), (tensor.shape, expected_shape)

    def mark_collective_shard(self, tensor, *, replicated_reference, shard_dim, expected_local_shape) -> None:
        assert tensor.shape == tuple(expected_local_shape), (tensor.shape, expected_local_shape)


# --------------------------------------------------------------------------- GDN under the fake


def _bf16(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


def _gdn_oracle_weights() -> Qwen38GDNWeights:
    torch.manual_seed(41)
    dims = Qwen38GDNDimensions(
        hidden_size=2560, q_heads=16, value_heads=48, key_head_dim=128, value_head_dim=128, conv_kernel=4
    )
    return Qwen38GDNWeights(
        layer_idx=0,
        dimensions=dims,
        rms_norm_eps=1e-6,
        output_gate="sigmoid",
        qkv=_bf16(dims.qkv_width, 2560, scale=2560**-0.5),
        z=_bf16(dims.value_width, 2560, scale=2560**-0.5),
        a=_bf16(48, 2560, scale=2560**-0.5),
        b=_bf16(48, 2560, scale=2560**-0.5),
        out=_bf16(2560, dims.value_width, scale=dims.value_width**-0.5),
        conv=_bf16(dims.qkv_width, 1, 4, scale=0.5),
        dt_bias=_bf16(48, scale=0.1),
        A_log=_bf16(48, scale=0.1),
        norm=_bf16(128, scale=0.1),
    )


def _device_gdn_weights(oracle: Qwen38GDNWeights) -> gdn_module.Qwen38TTNNGDNWeights:
    """The host packing of ``Qwen38TTNNGDNWeights.from_checkpoint`` on the fake mesh."""

    shards = [oracle.device_shard(index) for index in range(TP)]
    qkvzab = torch.chunk(gdn_module.pack_projection_columns(shards), TP, dim=3)
    out = [shard.out.transpose(0, 1).contiguous().reshape(1, 1, 1536, 2560) for shard in shards]
    taps = tuple(
        FakeTensor([shard.conv[:, 0, tap].to(torch.bfloat16).reshape(1, 1, 1, 2560) for shard in shards], BF16, TILE, 3)
        for tap in range(4)
    )
    return gdn_module.Qwen38TTNNGDNWeights(
        layer_index=0,
        qkvzab=FakeTensor([piece.clone() for piece in qkvzab], BF16, TILE, 3),
        out=FakeTensor(out, BF16, TILE, 2),
        conv_taps=taps,
        dt_bias=FakeTensor([shard.dt_bias.float().reshape(1, 1, 1, 12) for shard in shards], FP32, TILE, 3),
        neg_exp_A=FakeTensor(
            [(-torch.exp(shard.A_log.float())).reshape(1, 1, 1, 12) for shard in shards], FP32, TILE, 3
        ),
        norm=FakeTensor([oracle.norm.reshape(1, 1, 1, 128).clone() for _ in range(TP)], BF16, TILE),
        projection_dtype=BF16,
    )


def _gdn_module(weights) -> gdn_module.Qwen38TTNNGDN:
    module = object.__new__(gdn_module.Qwen38TTNNGDN)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.weights = weights
    module.collective_topology = "linear"
    for name in (
        "compute_config",
        "projection_compute_config",
        "in_proj_act_memory_config",
        "in_proj_program_config",
        "out_proj_act_memory_config",
        "out_proj_program_config",
        "recurrent_matmul_program_config",
        "recurrent_read_compute_config",
        "recurrent_write_compute_config",
    ):
        setattr(module, name, name)
    return module


@pytest.fixture
def fake(monkeypatch):
    chunk = FakeChunk()
    fake_ttnn = make_fake_ttnn(chunk)
    install_source_chunk_fake(monkeypatch, chunk)
    for module in (gdn_module, ple_module, layer_module):
        monkeypatch.setattr(module, "ttnn", fake_ttnn)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    return SimpleNamespace(ttnn=fake_ttnn, chunk=chunk)


def test_softplus_gate_never_flushes_to_zero_and_tracks_torch_softplus(fake) -> None:
    """The decay gate over a + dt_bias in [-8, 0] (the served range; layer-0 heads reach -5.5): no exact 0,
    within 1e-3 of torch softplus, FP32 out.  The fused SOFTPLUS activation it replaced returned 0 below -5.02."""

    xs = torch.linspace(-8.0, 0.0, 4 * 32, dtype=torch.float32).reshape(1, 1, 4, 32)
    a_fp32 = FakeTensor([xs.clone() for _ in range(TP)], FP32)
    dt_bias = FakeTensor([torch.zeros(1, 1, 1, 32) for _ in range(TP)], FP32)
    gate = gdn_module.softplus_gate(a_fp32, dt_bias, memory_config="L1")
    assert gate.dtype is FP32 and gate.shape == (1, 1, 4, 32)
    for local in gate.torch_shards():
        assert bool((local > 0.0).all()), float(local.min())
        assert float((local - F.softplus(xs)).abs().max()) <= 1e-3
    assert a_fp32.alive and dt_bias.alive  # the helper frees only its own intermediate (the add's output)


@pytest.fixture(scope="module")
def gdn_weights():
    oracle = _gdn_oracle_weights()
    return oracle, _device_gdn_weights(oracle)


def _hidden_sharded(hidden: torch.Tensor) -> FakeTensor:
    """``[1, T, 2560]`` bf16 -> hidden-sharded ``[1, 1, T, 640]``."""

    return FakeTensor(
        [piece.clone() for piece in torch.chunk(hidden.reshape(1, 1, -1, 2560), TP, dim=3)], BF16, TILE, 3
    )


def _cat(tensor: FakeTensor, dim: int) -> torch.Tensor:
    """The four device locals back to one host tensor along their shard dim (a copy)."""

    return torch.cat(tensor.torch_shards(), dim=dim)


def _scalar(tensor: FakeTensor) -> float:
    return float(tensor.torch_shards()[0].reshape(-1)[0])


def _seed_state(module: gdn_module.Qwen38TTNNGDN, seed: int):
    """A non-zero recurrent state and non-zero ring history so the paths exercise the real transition."""

    state = module.allocate_state()
    torch.manual_seed(seed)
    for local in state.recurrent.locals:
        local.copy_(torch.randn(1, 12, 128, 128) * 0.3)
    for slot in state.conv:
        for local in slot.locals:
            local.copy_(torch.randn(1, 1, 1, 2560).to(torch.bfloat16))
    return state


def _clone_state(module, state):
    clone = module.allocate_state()
    for source, target in ((state.recurrent, clone.recurrent), *zip(state.conv, clone.conv)):
        for s, t in zip(source.locals, target.locals):
            t.copy_(s)
    clone.conv_phase = state.conv_phase
    return clone


def _sequential(module, state, hidden: torch.Tensor) -> list[torch.Tensor]:
    outputs = []
    for row in range(hidden.shape[1]):
        result = module.forward_decode(_hidden_sharded(hidden[:, row : row + 1]), state)
        outputs.append(_cat(result.hidden_sharded, 3))
    return outputs


def _rows_run(module, state, hidden: torch.Tensor, constants):
    rows_state = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state, rows_state)
    result = module.forward_rows(_hidden_sharded(hidden), state, rows_state)
    return result, rows_state


def test_forward_rows_full_tile_is_the_row_slice_with_exact_zero_padding_rows(fake, gdn_weights) -> None:
    """``full_tile=True`` returns the whole 32-row output (a fresh buffer): rows below R are the persistent slice's,
    rows past R are exact zeros (zero q rows -> zero recurrent output -> zero norm, gate and projection)."""

    fake.chunk.impl = step_chunk_kernel
    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(17)
    rows = 5
    hidden = torch.randn(1, rows, 2560).to(torch.bfloat16)
    state_slice = _seed_state(module, 13)
    state_tile = _clone_state(module, state_slice)
    constants = module.allocate_rows_constants(rows)
    sliced, _ = _rows_run(module, state_slice, hidden, constants)
    rows_state = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state_tile, rows_state)
    tile = module.forward_rows(_hidden_sharded(hidden), state_tile, rows_state, full_tile=True)
    output = _cat(tile.hidden_rows, 3)
    assert output.shape == (1, 1, CHUNK_ROWS, 2560) and tile.hidden_rows is not rows_state.output
    assert torch.equal(output[:, :, :rows], _cat(sliced.hidden_rows, 3))
    assert torch.count_nonzero(output[:, :, rows:]) == 0
    assert torch.equal(_cat(tile.final_state, 1), _cat(sliced.final_state, 1))
    # The verify layer body takes the tile (no slice landing, no pad) and owns it.
    source = inspect.getsource(mtp_v2._forward_layer_verify)
    assert "full_tile=True" in source and 'label="GDN verify output rows"' not in source
    assert "attention_owner = attention_hidden" in source


def test_forward_rows_at_one_row_is_bitwise_the_one_row_path_around_the_recurrence(fake, gdn_weights) -> None:
    fake.chunk.impl = step_chunk_kernel
    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(7)
    hidden = torch.randn(1, 1, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 11)
    state_rows = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(1)

    initial = _cat(state_step.recurrent, 1)
    step_output = _sequential(module, state_step, hidden)[0]
    result, rows_state = _rows_run(module, state_rows, hidden, constants)

    assert result.hidden_rows.shape == (1, 1, 1, 640)
    assert torch.equal(_cat(result.hidden_rows, 3), step_output)
    assert torch.equal(_cat(result.final_state, 1), _cat(state_step.recurrent, 1))
    # The committed state was not written by the pass; the ring and its phase were not touched.
    assert torch.equal(_cat(state_rows.recurrent, 1), initial)
    assert (
        state_rows.conv_phase == 0 and fake.chunk.calls[-1]["chunk_size"] == 32 and fake.chunk.calls[-1]["rows"] == 32
    )
    # The kernel saw the GQA-expanded q/k (H = HV = 12, no repeat inside the composite), v token-major flat
    # (no head split) and was asked for its head-major output (no untilize / row-major permute round trip).
    assert fake.chunk.calls[-1]["heads"] == (HEADS, HEADS, HEADS) and fake.chunk.calls[-1]["output_head_major"]
    assert fake.chunk.calls[-1]["flat_v"]
    # Rows past the real one reached the kernel as zeros: k = v = 0, beta = 0, g = 0.
    for name, rows_axis, shard_dim in (("q", 1, 2), ("k", 1, 2), ("v", 2, 3), ("beta", 2, 3), ("g", 2, 3)):
        tensor = _cat(getattr(rows_state, name), shard_dim)
        assert torch.count_nonzero(tensor.narrow(rows_axis, 1, 31)) == 0, name


def test_forward_rows_takes_a_zero_padded_input_tile_without_the_pad(fake, gdn_weights) -> None:
    """A caller that hands the full 32-row tile (rows past R already zero) gets the same rows, bitwise, and
    the path skips ``ttnn.pad`` (the persistent zero-padded input tile of the verify body)."""

    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(17)
    hidden = torch.randn(1, ROWS, 2560).to(torch.bfloat16)
    state_a = _seed_state(module, 18)
    state_b = _clone_state(module, state_a)
    constants = module.allocate_rows_constants(ROWS)
    rows_a, rows_b = module.allocate_rows_state(constants), module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state_a, rows_a)
    module.sync_rows_history_from_state(state_b, rows_b)  # the eager sync pads the history tile; not counted
    pads = []
    real_pad = fake.ttnn.pad
    fake.ttnn.pad = lambda *args, **kwargs: pads.append(args[1]) or real_pad(*args, **kwargs)

    result_a = module.forward_rows(_hidden_sharded(hidden), state_a, rows_a)
    assert pads == [[(0, 0), (0, 0), (0, 32 - ROWS), (0, 0)]]
    tile = _hidden_sharded(F.pad(hidden, (0, 0, 0, 32 - ROWS)))
    assert tile.shape == (1, 1, 32, 640)
    result_b = module.forward_rows(tile, state_b, rows_b)
    assert pads == [[(0, 0), (0, 0), (0, 32 - ROWS), (0, 0)]]  # no second pad
    assert torch.equal(_cat(result_b.hidden_rows, 3), _cat(result_a.hidden_rows, 3))
    assert torch.equal(_cat(result_b.final_state, 1), _cat(result_a.final_state, 1))
    for name, shard_dim in (("q", 2), ("k", 2), ("v", 3), ("beta", 3), ("g", 3), ("qkv", 3)):
        assert torch.equal(_cat(getattr(rows_b, name), shard_dim), _cat(getattr(rows_a, name), shard_dim)), name


def _oracle_rows(oracle: Qwen38GDNWeights, hidden: torch.Tensor, state: gdn_module.Qwen38TTNNGDNState):
    # Device d holds q/k heads 4d..4d+3 and v heads 12d..12d+11 as its local [q|k|v] columns, in checkpoint order.
    stacked = torch.stack([_cat(slot, 3).reshape(TP, 2560) for slot in state.conv_window()[:3]], dim=-1)  # [4, 2560, 3]
    q, k, v = (
        stacked[:, :512].reshape(2048, 3),
        stacked[:, 512:1024].reshape(2048, 3),
        stacked[:, 1024:].reshape(6144, 3),
    )
    oracle_state = Qwen38GDNState(
        conv=torch.cat([q, k, v], dim=0).reshape(1, 10240, 3), recurrent=_cat(state.recurrent, 1)
    )
    with row_serial_torch():
        return Qwen38GDN(oracle).forward(hidden, oracle_state)


def test_forward_rows_matches_five_sequential_steps_and_the_cpu_oracle(fake, gdn_weights) -> None:
    oracle, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(8)
    hidden = torch.randn(1, ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 12)
    state_rows = _clone_state(module, state_step)
    oracle_output, oracle_state = _oracle_rows(oracle, hidden, state_rows)
    constants = module.allocate_rows_constants(ROWS)

    step_outputs = torch.cat(_sequential(module, state_step, hidden), dim=2)  # [1, 1, 5, 2560]
    result, _ = _rows_run(module, state_rows, hidden, constants)
    rows_output = _cat(result.hidden_rows, 3)
    final_state = _cat(result.final_state, 1)
    step_state = _cat(state_step.recurrent, 1)

    out_err = (rows_output.float() - step_outputs.float()).abs().max().item()
    state_err = (final_state - step_state).abs().max().item()
    print(
        f"rows vs 5 steps: output max abs {out_err:.2e} (|out| max {step_outputs.float().abs().max().item():.2f}); "
        f"state max abs {state_err:.2e} (|state| max {step_state.abs().max().item():.2f})"
    )
    assert rows_output.shape == (1, 1, ROWS, 2560)
    # The kernel scales q in bf16 (a 2**-9 relative rounding the fp32 step does not have) and the layer output
    # is bf16: allow four bf16 ulps of the output scale on top of the design's 2e-4 recurrence budget.
    assert out_err <= 2e-4 + 4 * 2.0**-8 * step_outputs.float().abs().max().item()
    torch.testing.assert_close(final_state, step_state, rtol=1e-4, atol=1e-5)  # step-2 model: rel <= 1.2e-5
    # The CPU oracle normalizes q/k with its own bf16 rounding (a 1-ulp flip in k moves the state by 4e-3), so
    # this is a wiring check (head order, conv mapping, gate/output path), not a precision gate.
    oracle_err = (
        (rows_output.float().reshape(ROWS, 2560) - oracle_output.float().reshape(ROWS, 2560)).abs().max().item()
    )
    oracle_state_err = (final_state - oracle_state.recurrent).abs().max().item()
    print(f"rows vs CPU oracle: output max abs {oracle_err:.2e}; state max abs {oracle_state_err:.2e}")
    assert oracle_err <= 0.1 * oracle_output.float().abs().max().item()
    assert oracle_state_err <= 0.1 * oracle_state.recurrent.abs().max().item()


@pytest.mark.parametrize("form", ("chunk", "step_on_full_rejection", "step_committed_rows"))
@pytest.mark.parametrize("accepted", range(ROWS))
def test_commit_rows_lands_the_committed_prefix_state_and_history(fake, gdn_weights, accepted: int, form: str) -> None:
    step_form = form == "step_on_full_rejection"
    anchor_form = form == "step_committed_rows"
    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(9)
    hidden = torch.randn(1, ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 13)
    state_rows = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(ROWS)

    _sequential(module, state_step, hidden[:, : accepted + 1])  # the committed prefix, one step at a time
    result, rows_state = _rows_run(module, state_rows, hidden, constants)
    accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
    selectors = gdn_module.build_rows_selectors(accepted_tensor, constants)
    assert [_scalar(t) for t in selectors.onehot_bf16] == [float(j == accepted) for j in range(ROWS)]
    assert selectors.committed_mask.torch_shards()[0].reshape(-1).tolist() == [float(j <= accepted) for j in range(32)]
    # The per-pass history select is the stack row of this accept count: rows 0..2 pick logical window rows
    # accepted + 1 .. accepted + 3 (history rows 0..2 sit at buffer rows 0..2, new row r at 32 + r).
    select = selectors.history_select.torch_shards()[0].float().reshape(32, 64)
    assert selectors.history_select.dtype is BF16 and torch.count_nonzero(select[3:]) == 0
    for index in range(3):
        logical = accepted + 1 + index
        assert select[index].nonzero().reshape(-1).tolist() == [logical if logical < 3 else 32 + logical - 3]
    chunk_calls = len(fake.chunk.calls)
    module.commit_rows(
        state_rows, rows_state, selectors, step_on_full_rejection=step_form, step_committed_rows=anchor_form
    )
    # The re-anchor does not rerun the chunk kernel; the other two forms rerun it once (the masked catch-up).
    assert len(fake.chunk.calls) - chunk_calls == (0 if anchor_form else 1)

    committed = _cat(state_rows.recurrent, 1)
    expected = _cat(state_step.recurrent, 1)  # accepted + 1 committed rows: row 0 (the base token) always is
    err = (committed - expected).abs().max().item()
    print(f"commit a={accepted} form={form}: state vs {accepted + 1} steps max abs {err:.2e}")
    torch.testing.assert_close(committed, expected, rtol=1e-4, atol=1e-5)
    if (accepted == 0 and step_form) or anchor_form:
        # The 1-row step arithmetic over the committed rows is bitwise the 1-row path's state (every a with the
        # re-anchor; a = 0 with the full-rejection switch).
        assert torch.equal(committed, expected)
    if accepted == ROWS - 1 and not anchor_form:
        assert torch.equal(committed, _cat(result.final_state, 1))
    # The history tile's rows 0..2 are the last three q|k|v rows of the committed stream: the 1-row ring's three
    # history slots; the selection matmul leaves exact zeros in rows 3..31.
    expected_history = torch.cat([_cat(slot, 3) for slot in state_step.conv_window()[:3]], dim=2)
    history = _cat(rows_state.history, 3)  # the four device tiles side by side
    assert history.shape == (1, 1, 32, 4 * 2560)
    assert torch.equal(history[:, :, :3], expected_history) and torch.count_nonzero(history[:, :, 3:]) == 0
    # Round trip back to the ring form for the 1-row path.
    module.sync_state_from_rows_history(rows_state, state_rows)
    for slot, expected_slot in zip(state_rows.conv_window()[:3], state_step.conv_window()[:3]):
        assert torch.equal(_cat(slot, 3), _cat(expected_slot, 3))


def test_step_anchor_keeps_the_committed_state_on_the_one_row_path_over_many_passes(fake, gdn_weights) -> None:
    """Twenty-four R = 5 passes with random accept counts (about 70 committed rows; 60 passes ran past the 300 s
    per-test timeout on a loaded host): with ``step_committed_rows`` the committed state stays bitwise the 1-row
    path's after every pass, and the FIR history the ring's; the chunk form's state is reported beside it (the fake
    kernel is the fp32 torch WY form, so its drift here is the algorithm's, not the device's TF32 drift that motivated
    the re-anchor)."""

    _, weights = gdn_weights
    module = _gdn_module(weights)
    generator = torch.Generator().manual_seed(180)
    state_step = _seed_state(module, 21)
    state_anchor = _clone_state(module, state_step)
    state_chunk = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(ROWS)
    rows_anchor = module.allocate_rows_state(constants)
    rows_chunk = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state_anchor, rows_anchor)
    module.sync_rows_history_from_state(state_chunk, rows_chunk)
    committed_rows = 0
    chunk_drift = []
    for index in range(24):
        hidden = torch.randn(1, ROWS, 2560, generator=generator).to(torch.bfloat16)
        accepted = int(torch.randint(0, ROWS, (1,), generator=generator))
        accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
        selectors = gdn_module.build_rows_selectors(accepted_tensor, constants)
        _sequential(module, state_step, hidden[:, : accepted + 1])
        for state, rows_state, anchor in ((state_anchor, rows_anchor, True), (state_chunk, rows_chunk, False)):
            module.forward_rows(_hidden_sharded(hidden), state, rows_state)
            module.commit_rows(state, rows_state, selectors, step_committed_rows=anchor)
        committed_rows += accepted + 1
        expected = _cat(state_step.recurrent, 1)
        anchored = _cat(state_anchor.recurrent, 1)
        chunk_drift.append((_cat(state_chunk.recurrent, 1) - expected).abs().max().item())
        assert torch.equal(anchored, expected), (index, accepted, (anchored - expected).abs().max().item())
        expected_history = torch.cat([_cat(slot, 3) for slot in state_step.conv_window()[:3]], dim=2)
        assert torch.equal(_cat(rows_anchor.history, 3)[:, :, :3], expected_history)
    assert committed_rows >= 60, committed_rows
    scale = _cat(state_step.recurrent, 1).abs().max().item()
    print(
        f"{committed_rows} committed rows over 24 passes: anchored state bitwise the 1-row path's at every pass; "
        f"chunk-form drift max abs {max(chunk_drift):.2e} (state scale {scale:.2f})"
    )


def test_rows_paths_never_upload_or_take_per_pass_host_ints() -> None:
    """Trace contract of the new GDN methods: one full chunk with device-resident constants, no host tensor
    creation, the row count only from the constants, and the 1-row state's ring untouched."""

    tree = ast.parse(GDN_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGDN")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    rows_methods = [
        "forward_rows",
        "commit_rows",
        "_all_gather_rows",
        "_project_rows",
        "_conv_window_rows",
        "_select_rows",
        "_causal_conv_rows",
        "_make_chunk_inputs",
        "_chunk_rows",
        "_chunk_rows_composite",
        "_project_rows_linear",
        "_land_rows_qkv",
        "_gate_and_project_rows",
        "_rows_output_tile",
        "_advance_history_rows",
    ]
    calls = {
        name: [ast.unparse(node.func) for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        for name in rows_methods
    }
    for name, names in calls.items():
        for forbidden in (
            "ttnn.zeros",
            "ttnn.from_torch",
            "ttnn.as_tensor",
            "torch.",
            "ttnn.copy_host_to_device_tensor",
        ):
            assert not any(called.startswith(forbidden) for called in names), (name, forbidden)
        assert "state.advance_conv_window" not in names and "state.conv_window" not in names, name
    for name in ("forward_rows", "commit_rows"):  # the row count comes from the constants, never per call
        for argument in methods[name].args.args:
            annotation = ast.unparse(argument.annotation) if argument.annotation is not None else ""
            assert annotation != "int", f"{name} takes a per-call host int {argument.arg}"
    # The owned rows path hands its views to the public composite with the original scale.
    dispatch = [
        node
        for node in ast.walk(methods["_chunk_rows"])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self._chunk_rows_composite"
    ]
    assert len(dispatch) == 1 and [ast.unparse(argument) for argument in dispatch[0].args] == [
        "q_rows",
        "k_rows",
        "v_rows",
        "g_rows",
        "beta_rows",
        "initial_state",
        "constants",
    ]
    assert calls["_chunk_rows"].count("ttnn.transformer.chunk_gated_delta_rule") == 0
    chunk_call = next(
        node
        for node in ast.walk(methods["_chunk_rows_composite"])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "ttnn.transformer.chunk_gated_delta_rule"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in chunk_call.keywords}
    assert keywords["chunk_size"] == "CHUNK_SIZE" and keywords["output_final_state"] == "True"
    assert keywords["output_head_major"] == "True"  # the kernel's TILE layout; no untilize / RM permute
    assert keywords["initial_state"] == "initial_state" and keywords["scale"] == "HEAD_DIM ** (-0.5)"
    assert all(keywords[name] == f"constants.{name}" for name in ("eye", "tril", "ones", "masks"))
    # q_rows / k_rows are rows_state.q / rows_state.k, or their rank-3 token-major flat views under the slab's
    # gdn_qk_flat glue form (test_ttnn_prefill_glue_no_device pins the conditional); v_rows is the flat view of v.
    assert [ast.unparse(argument) for argument in chunk_call.args] == [
        "q_rows",
        "k_rows",
        "v_rows",
        "g_rows",
        "beta_rows",
    ]
    assert calls["_chunk_rows_composite"].count("ttnn.transformer.chunk_gated_delta_rule") == 1
    assert calls["_chunk_rows_composite"].count("fused.gdn_source_chunk.chunk_token_major") == 1
    source_call = next(
        node
        for node in ast.walk(methods["_chunk_rows_composite"])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "fused.gdn_source_chunk.chunk_token_major"
    )
    source_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in source_call.keywords}
    assert source_keywords == {"rows_total": "constants.tile_rows", "scale": "HEAD_DIM ** (-0.5)"}
    # The forward pass reads the committed state and writes only rows buffers; the commit is the only writer.
    assert "output_tensor=state.recurrent" not in ast.get_source_segment(
        GDN_SOURCE.read_text(), methods["forward_rows"]
    )
    assert "_copy_inplace" in calls["commit_rows"] and "output_tensor=state.recurrent" in ast.get_source_segment(
        GDN_SOURCE.read_text(), methods["commit_rows"]
    )
    # Exact selects: every row read from the [history | qkv] window is a 0/1 selection matmul (HiFi4, fp32
    # accumulation), never a row-unaligned slice (a ROW_MAJOR round trip on this runtime) and never a reduce.
    source = GDN_SOURCE.read_text()
    select = ast.get_source_segment(source, methods["_select_rows"])
    assert "ttnn.matmul(" in select and "compute_kernel_config=self.compute_config" in select
    for name in ("_advance_history_rows", "_causal_conv_rows"):
        body = ast.get_source_segment(source, methods[name])
        assert "self._select_rows(" in body and "ttnn.slice" not in body and "ttnn.sum" not in body, name
    assert "ttnn.slice" not in ast.get_source_segment(source, methods["_conv_window_rows"])
    assert calls["_causal_conv_rows"].count("self._select_rows") == 1  # one per tap in the comprehension
    assert (
        calls["_make_chunk_inputs"].count("ttnn.matmul") == 1
        and "ttnn.repeat_interleave" not in calls["_make_chunk_inputs"]
    )
    gate = ast.get_source_segment(source, methods["_gate_and_project_rows"])
    assert "ttnn.experimental.view(" in gate and "ttnn.to_layout" not in gate and "ttnn.permute" not in gate
    assert "ttnn.pad" not in ast.get_source_segment(source, methods["_causal_conv_rows"])
    # The history select lands through the matmul's own output tensor (no copy); the state copy remains.
    assert "output_tensor=rows_state.history" in ast.get_source_segment(source, methods["_advance_history_rows"])
    assert "_copy_inplace" not in calls["_advance_history_rows"]
    assert gdn_module.CHUNK_SIZE == 32 and gdn_module.CONV_HISTORY_ROWS == 3 and gdn_module.CONV_WINDOW_ROWS == 35
    assert gdn_module.CONV_WINDOW_TILE_ROWS == 64
    tiles = gdn_module.chunk_constant_tiles()
    assert tiles["masks"].shape == (1, 1, 32, 96) and tiles["masks"].sum() == 3 * 16 * 16
    assert torch.equal(tiles["eye"][0, 0], torch.eye(32)) and tiles["tril"].sum() == 32 * 33 / 2


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and torch.equal(left.view(torch.int16), right.view(torch.int16))


def test_rows_selection_matmuls_reproduce_torch_indexing_bitwise() -> None:
    """The 0/1 selection matrices pick exactly the rows torch indexing picks, under the device's arithmetic
    model (HiFi4: exact bf16 x bf16 products; fp32 accumulation; one rounding to bf16).  Each output element
    has a single nonzero term (x * 1.0, the other products exact zeros), so the fp32 sum is x and the bf16
    result is the selected bf16 value, whatever the accumulation order."""

    tiles = gdn_module.rows_window_select_tiles(ROWS)
    taps, stack, expand = tiles["conv_taps"], tiles["history_select_stack"], tiles["qk_expand"]
    for matrix in (*taps, stack, expand):
        assert set(matrix.unique().tolist()) <= {0.0, 1.0}
        assert torch.equal(matrix.to(torch.bfloat16).float(), matrix)  # exactly representable constants
    assert (taps != 0).sum(dim=-1).eq(1).all()  # every tap row selects exactly one window row
    assert (expand != 0).sum(dim=0).eq(1).all()  # every expanded column comes from exactly one q/k column
    for accepted in range(ROWS):
        select = stack[accepted].reshape(32, 64)
        assert (select[:3] != 0).sum(dim=-1).eq(1).all() and torch.count_nonzero(select[3:]) == 0
    assert torch.count_nonzero(stack[ROWS:]) == 0

    torch.manual_seed(29)
    history_tile = _bf16(1, 1, 32, 2560)  # rows 3..31 are never read: random here on purpose
    qkv = _bf16(1, 1, 32, 2560)
    window = torch.cat([history_tile, qkv], dim=2)  # the device window: two whole tiles
    logical = torch.cat([history_tile[:, :, :3], qkv], dim=2)  # [3 history | 32 new] = 35 rows

    def select_rows(select: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        products = select.float().unsqueeze(-1) * rows.float().unsqueeze(-3)  # every product, exact in fp32
        return products.sum(dim=-2).to(torch.bfloat16)  # fp32 accumulation, one rounding

    for tap in range(3):
        assert _bits_equal(select_rows(taps[tap], window[0, 0]), logical[0, 0, tap : tap + 32]), tap
    for accepted in range(ROWS):
        onehot = torch.zeros(1, 32)
        onehot[0, accepted] = 1.0
        select = select_rows(onehot.to(torch.bfloat16), stack.to(torch.bfloat16)).float().reshape(32, 64)
        assert torch.equal(select, stack[accepted].reshape(32, 64))
        selected = select_rows(select, window[0, 0])
        assert _bits_equal(selected[:3], logical[0, 0, accepted + 1 : accepted + 4]), accepted
        assert torch.count_nonzero(selected[3:].float()) == 0
    q = _bf16(1, 1, 32, 512)
    expanded = (q[0, 0].float() @ expand.float()).to(torch.bfloat16)  # one nonzero term per output element
    assert _bits_equal(
        expanded.reshape(1, 32, HEADS, HEAD_DIM), q.reshape(1, 32, 4, HEAD_DIM).repeat_interleave(3, dim=2)
    )


def test_fake_view_is_the_device_tile_order_and_folds_head_major_rows_for_free() -> None:
    """``ttnn.experimental.view`` reinterprets the tile pages: the fake models that, and on it the head-major
    ``[1, 12, 32, 128]`` -> token-major ``[1, 1, 32, 1536]`` fold the gate path uses is a permute + reshape in
    logical terms, at zero data movement on device (tile (h, c) sits at page 4h + c in both shapes)."""

    torch.manual_seed(30)
    head_major = torch.randn(1, HEADS, 32, HEAD_DIM).to(torch.bfloat16)
    viewed = _view(FakeTensor([head_major.clone() for _ in range(TP)], BF16), (1, 1, 32, HEADS * HEAD_DIM))
    expected = head_major.permute(0, 2, 1, 3).reshape(1, 1, 32, HEADS * HEAD_DIM)
    assert _bits_equal(viewed.torch_shards()[0], expected)
    assert not torch.equal(viewed.torch_shards()[0].float(), head_major.reshape(1, 1, 32, -1).float())
    # The GR shapes the model already runs through the view on device, and a plain row-count view.
    branch_major = torch.randn(1, 4, 1, 640).to(torch.bfloat16)
    assert torch.equal(
        _view(FakeTensor([branch_major] * TP, BF16), (1, 1, 1, 2560)).torch_shards()[0],
        branch_major.reshape(1, 1, 1, 2560),
    )
    rows = torch.randn(12, 32, 128).to(torch.bfloat16)
    assert torch.equal(
        _view(FakeTensor([rows] * TP, BF16), (1, 12, 32, 128)).torch_shards()[0], rows.reshape(1, 12, 32, 128)
    )


def test_rows_bodies_are_the_pinned_walk() -> None:
    """The rows-path call order (the trace body's op order) and the stage boundaries the micro-test times."""

    tree = ast.parse(GDN_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGDN")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}

    def walk(name: str) -> list[str]:
        return [
            ast.unparse(node.func).removeprefix("self.")
            for node in sorted(
                (n for n in ast.walk(methods[name]) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
            )
            if ast.unparse(node.func).startswith("self._")
        ]

    # forward_rows gathers and hands the body to the resolved rows form; the chain's own order moved verbatim into
    # the registry's composed body (ttnn/fused/gdn_rows_wrap.rows_body_composed), which is walked right below.
    assert walk("forward_rows") == [
        "_validate_state",
        "_validate_rows_state",
        "_all_gather_rows",
        # the slab body, resolved once: the fused gdn_prefill_rows pair, or the chain's four calls below (which are
        # what the composed callable runs); the 32-row and 128-row bodies take those four directly
        "_gdn_prefill_rows",
        "_rows_body",
    ]
    composed = ast.parse(inspect.getsource(gdn_rows_wrap.rows_body_composed))
    assert [
        ast.unparse(node.func).removeprefix("gdn.")
        for node in sorted(
            (n for n in ast.walk(composed) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
        )
        if ast.unparse(node.func).startswith("gdn._")
    ] == [
        "_project_rows",
        "_causal_conv_rows",
        "_make_chunk_inputs",
        "_chunk_rows",
        "_gate_and_project_rows",
    ]
    assert walk("_project_rows") == ["_project_rows_linear", "_land_rows_qkv"]
    assert walk("_gate_and_project_rows")[-1] == "_rows_output_tile"
    assert walk("commit_rows") == [
        "_validate_state",
        "_validate_rows_state",
        "_advance_history_rows",  # the verify-rows fold's form: the prefix-state pick (a fused call), then the history
        "_step_committed_rows_state",  # the state re-anchor form: no chunk rerun
        "_advance_history_rows",
        "_chunk_rows",
        "_step_row_state",  # the step-on-full-rejection form's row 0
        "_advance_history_rows",
    ]
    assert walk("_step_committed_rows_state") == ["_step_row_state"]
    # the recurrence: the wrap's prim-layout form when it owns the rows state, else the resolved composite / prims
    # direct call; both return through the one validation
    assert walk("_chunk_rows") == ["_validate_chunk_rows", "_chunk_rows_composite", "_validate_chunk_rows"]
    assert walk("_validate_chunk_rows") == []
    assert walk("_chunk_rows_composite") == []
    assert walk("_causal_conv_rows") == ["_shifted_rows_slab", "_conv_window_rows", "_select_rows"]
    assert walk("_advance_history_rows") == ["_conv_window_rows", "_select_rows"]


def test_one_row_gdn_bodies_are_the_pinned_walk() -> None:
    """The 1-row production path is untouched: its method set and call order are the op-diet pins' walk."""

    tree = ast.parse(GDN_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGDN")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward_decode")
    called = [
        ast.unparse(node.func).removeprefix("self.")
        for node in sorted(
            (n for n in ast.walk(forward) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
        )
        if ast.unparse(node.func).startswith("self._")
    ]
    # the conv, recurrent inputs, recurrent step and gate run inside the resolved gdn_step chain (ttnn/fused/gdn_step)
    assert called == ["_validate_state", "_all_gather_hidden", "_project", "_gdn_step", "_out_project"]
    assert "rows" not in ast.get_source_segment(GDN_SOURCE.read_text(), forward)


# --------------------------------------------------------------------------- PLE under the fake

EOS = 2
PLE_TABLE_ROWS = 512


class _FakeResidentLookup:
    """A deterministic stand-in for the n-gram table: the row of (token, context) is a hash of the three ids."""

    def __init__(self) -> None:
        torch.manual_seed(21)
        self.table = _bf16(PLE_TABLE_ROWS, 2560, scale=0.5)

    def _row(self, token: int, context) -> torch.Tensor:
        c0, c1 = (EOS, EOS) if context is None else context
        return self.table[(token * 7919 + c1 * 104729 + c0 * 1299709) % PLE_TABLE_ROWS]

    def lookup_token(self, token: int, context):
        c0, c1 = (EOS, EOS) if context is None else context
        row = self._row(token, context).contiguous()
        return bytearray(row.view(torch.int16).numpy().tobytes()), (c1, token)

    def lookup_tokens(self, tokens, context):
        payload = bytearray()
        contexts = [context]
        for token in tokens:
            row, context = self.lookup_token(int(token), context)
            payload += row
            contexts.append(context)
        return payload, tuple(contexts)

    def lookup(self, input_ids: torch.Tensor, previous_context):
        context = None if previous_context is None else tuple(int(v) for v in previous_context[0])
        token = int(input_ids.reshape(-1)[0])
        row = self._row(token, context)
        _, next_context = self.lookup_token(token, context)
        return row.reshape(1, 1, 2560), torch.tensor([list(next_context)], dtype=torch.long)


def _ple_weights():
    torch.manual_seed(22)
    source = Qwen38PLEWeights(
        layer_idx=1,
        hidden_size=2560,
        residual_branches=4,
        embedding_width=2560,
        conv_kernel=4,
        conv_dilation=3,
        rms_norm_eps=1e-6,
        key=_bf16(4 * 2560, 2560, scale=2560**-0.5),
        value=_bf16(2560, 2560, scale=2560**-0.5),
        norm_key=_bf16(4 * 2560, scale=0.1),
        norm_query=_bf16(4 * 2560, scale=0.1),
        norm_conv=_bf16(4 * 2560, scale=0.1),
        conv=_bf16(4 * 2560, 1, 4, scale=0.5),
    )

    def shard(host: torch.Tensor, dtype: _DType) -> FakeTensor:
        return FakeTensor([piece.clone() for piece in torch.chunk(host.to(dtype.torch), TP, dim=3)], dtype, TILE, 3)

    conv = source.conv.reshape(4, 2560, 4)
    return ple_module.Qwen38TTNNPLEWeights(
        key=shard(ple_module._prepare_key_weight(source.key), BF16),
        value=shard(source.value.transpose(0, 1).reshape(1, 1, 2560, 2560), BF16),
        norm_key=shard((1.0 + source.norm_key.float()).reshape(1, 1, 4, 2560), FP32),
        norm_query=shard((1.0 + source.norm_query.float()).reshape(1, 1, 4, 2560), FP32),
        norm_conv=shard((1.0 + source.norm_conv.float()).reshape(1, 1, 4, 2560), FP32),
        conv_taps=tuple(shard(conv[:, :, tap].reshape(1, 1, 4, 2560), BF16) for tap in range(4)),
        replicated_anchor=FakeTensor([torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE),
        layer_index=1,
    )


def _ple_module():
    module = object.__new__(ple_module.Qwen38TTNNPLE)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.weights = _ple_weights()
    module.collective_topology = "linear"
    module._poisoned_prepared_inputs = []
    module._resident_lookup = _FakeResidentLookup()
    module.host_embedding = module._resident_lookup
    module.compute_config = "compute_config"
    return module


def _residual_rows(residual: torch.Tensor) -> FakeTensor:
    """``[R, 4, 2560]`` bf16 -> token-major hidden-sharded ``[1, R, 4, 640]``."""

    return FakeTensor(
        [piece.clone() for piece in torch.chunk(residual.reshape(1, *residual.shape), TP, dim=3)], BF16, TILE, 3
    )


def _seed_ple_state(module):
    state = module.allocate_state()
    torch.manual_seed(23)
    for slot in state.conv:
        for local in slot.locals:
            local.copy_(torch.randn(1, 1, 4, 640).to(torch.bfloat16))
    return state


def test_ple_rows_equal_sequential_prepared_steps_and_commit_selects_the_history(fake) -> None:
    module = _ple_module()
    torch.manual_seed(24)
    residual = torch.randn(ROWS, 4, 2560).to(torch.bfloat16)
    tokens = (17, 15, 16, 95859, 20)
    state = _seed_ple_state(module)
    rows_state = module.allocate_rows_state(ROWS)
    rows_state.load_from_state(state)
    assert rows_state.token_context is None

    # 1-row path, threaded through the nine slots and the host context.
    deltas, histories, contexts = [], [], [None]
    for row, token in enumerate(tokens):
        prepared = module.prepare_decode_input(torch.tensor([[token]], dtype=torch.long), state)
        result = module.forward_prepared(_residual_rows(residual[row : row + 1]), prepared, state)
        deltas.append(_cat(result.residual_delta, 3))
        histories.append(torch.cat([_cat(slot, 3) for slot in state.conv], dim=1))
        contexts.append(tuple(int(v) for v in state.token_context[0]))

    prepared_rows = module.prepare_rows_input(tokens, rows_state)
    assert prepared_rows.contexts == tuple(contexts) and prepared_rows.embedding_rows.layout == ROW_MAJOR
    result = module.forward_prepared_rows(_residual_rows(residual), prepared_rows, rows_state)
    delta_rows = _cat(result.residual_delta, 3)
    assert delta_rows.shape == (1, ROWS, 4, 2560)
    for row in range(ROWS):
        assert torch.equal(delta_rows[:, row : row + 1], deltas[row]), row

    for accepted in range(ROWS):
        fresh = module.allocate_rows_state(ROWS)
        fresh.load_from_state(_seed_ple_state(module))
        prepared_fresh = module.prepare_rows_input(tokens, fresh)
        module.forward_prepared_rows(_residual_rows(residual), prepared_fresh, fresh)
        constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", module.mesh_contract, rows=ROWS)
        accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
        module.commit_rows(fresh, gdn_module.build_rows_selectors(accepted_tensor, constants))
        module.commit_rows_host(fresh, prepared_fresh, accepted)
        assert torch.equal(_cat(fresh.history, 3), histories[accepted]), accepted
        assert fresh.token_context == contexts[accepted + 1], accepted
        # Round trip to the nine-slot form.
        back = module.allocate_state()
        fresh.store_to_state(back)
        assert torch.equal(torch.cat([_cat(slot, 3) for slot in back.conv], dim=1), histories[accepted])
        assert back.token_context.tolist() == [list(contexts[accepted + 1])]


def test_ple_rows_source_pins() -> None:
    tree = ast.parse(PLE_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNPLE")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    source = PLE_SOURCE.read_text(encoding="utf-8")
    for name in (
        "forward_prepared_rows",
        "_project_rows",
        "_gate_rows",
        "_convolve_rows",
        "_conv_window_rows",
        "commit_rows",
    ):
        calls = [ast.unparse(node.func) for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        assert not any(
            called.startswith(("ttnn.from_torch", "ttnn.zeros", "torch.", "self.resident_lookup")) for called in calls
        ), name
    inject = [ast.unparse(node.func) for node in ast.walk(methods["inject_rows"]) if isinstance(node, ast.Call)]
    assert inject.count("ttnn.permute") == 2 and "self.forward_prepared_rows" in inject and "ttnn.add" in inject
    convolve = ast.get_source_segment(source, methods["_convolve_rows"])
    assert "tap * CONV_DILATION" in convolve and convolve.count("ttnn.slice") == 1
    commit = ast.get_source_segment(source, methods["commit_rows"])
    assert "committed + CONV_STATE_LENGTH" in commit and "fast_and_approximate_mode=False" in commit
    # The 1-row body is untouched: it still shifts the nine slots and consumes the prepared host context.
    convolve_one = ast.get_source_segment(source, methods["_convolve"])
    assert "_copy_inplace(state.conv[index + 1], state.conv[index]" in convolve_one
    forward_one = ast.get_source_segment(source, methods["forward_prepared"])
    assert "state.token_context = prepared.next_token_context" in forward_one


def test_layer_rows_entry_points_are_thin_and_typed() -> None:
    tree = ast.parse(LAYER_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNDecoderLayer")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    for name in ("forward_gdn_rows", "commit_gdn_rows", "forward_ple_rows", "commit_ple_rows"):
        assert name in methods, name
    calls = [ast.unparse(node.func) for node in ast.walk(methods["forward_ple_rows"]) if isinstance(node, ast.Call)]
    assert "ple.inject_rows" in calls and "ttnn.permute" not in calls  # the layer permutes only in _apply_ple
    gdn_calls = [ast.unparse(node.func) for node in ast.walk(methods["forward_gdn_rows"]) if isinstance(node, ast.Call)]
    assert "gdn.forward_rows" in gdn_calls
    # The 1-row layer bodies do not call into the rows path.
    for name in ("forward_decode", "forward_decode_generic", "_apply_ple"):
        called = [ast.unparse(node.func) for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        assert not any(called_name.endswith("_rows") for called_name in called), name


# --------------------------------------------------------------------------- prefill chunk: R = 32
# The chunk step of the C = 32 prefill is forward_rows at the full tile plus commit_rows with every row
# accepted (a = 31; a = r - 1 for the padded last chunk).  These pins hold the same properties as the
# R = 5 tests above at the chunk's row count, under the tolerance model (output <= 2e-4 + 4 bf16
# ulp of the output scale, state <= 3e-3 + 2^-7 of the state scale).

CHUNK_ROWS = 32
CHUNK_COMMITS = (31, 0, 1, 2, 8)


def _output_tolerance(reference: torch.Tensor) -> float:
    return 2e-4 + 4 * 2.0**-8 * reference.float().abs().max().item()


def _state_tolerance(reference: torch.Tensor) -> float:
    return 3e-3 + 2.0**-7 * reference.abs().max().item()


def test_forward_rows_at_the_chunk_size_matches_32_sequential_steps_and_pads_nothing(fake, gdn_weights) -> None:
    oracle, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(31)
    hidden = torch.randn(1, CHUNK_ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 32)
    state_rows = _clone_state(module, state_step)
    oracle_output, oracle_state = _oracle_rows(oracle, hidden, state_rows)
    constants = module.allocate_rows_constants(CHUNK_ROWS)
    assert torch.count_nonzero(constants.row_mask_fp32.torch_shards()[0]) == CHUNK_ROWS  # no padding rows
    pads = []
    real_pad = fake.ttnn.pad
    fake.ttnn.pad = lambda *args, **kwargs: pads.append(args[1]) or real_pad(*args, **kwargs)

    step_outputs = torch.cat(_sequential(module, state_step, hidden), dim=2)
    result, rows_state = _rows_run(module, state_rows, hidden, constants)
    rows_output = _cat(result.hidden_rows, 3)
    final_state = _cat(result.final_state, 1)
    step_state = _cat(state_step.recurrent, 1)

    # The history sync pads its 3-row tile; the forward pass of a full tile pads nothing.
    assert pads == [[(0, 0), (0, 0), (0, 32 - 3), (0, 0)]]
    assert rows_output.shape == (1, 1, CHUNK_ROWS, 2560) and result.hidden_rows is rows_state.output
    out_err = (rows_output.float() - step_outputs.float()).abs().max().item()
    state_err = (final_state - step_state).abs().max().item()
    print(
        f"chunk rows vs 32 steps: output max abs {out_err:.2e} (tol {_output_tolerance(step_outputs):.2e}); state max abs {state_err:.2e} (tol {_state_tolerance(step_state):.2e})"
    )
    assert out_err <= _output_tolerance(step_outputs)
    assert state_err <= _state_tolerance(step_state)
    oracle_err = (
        (rows_output.float().reshape(CHUNK_ROWS, 2560) - oracle_output.float().reshape(CHUNK_ROWS, 2560))
        .abs()
        .max()
        .item()
    )
    assert oracle_err <= 0.1 * oracle_output.float().abs().max().item()
    assert (final_state - oracle_state.recurrent).abs().max().item() <= 0.1 * oracle_state.recurrent.abs().max().item()
    # The committed state and the ring are untouched by the forward pass.
    assert torch.equal(_cat(state_rows.recurrent, 1), _cat(_clone_state(module, state_rows).recurrent, 1))
    assert fake.chunk.calls[-1]["rows"] == 32 and fake.chunk.calls[-1]["chunk_size"] == 32


@pytest.mark.parametrize("accepted", CHUNK_COMMITS)
def test_commit_rows_at_the_chunk_size_lands_the_prefix_state_and_the_history(fake, gdn_weights, accepted: int) -> None:
    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(33)
    hidden = torch.randn(1, CHUNK_ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 34)
    state_rows = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(CHUNK_ROWS)

    _sequential(module, state_step, hidden[:, : accepted + 1])
    result, rows_state = _rows_run(module, state_rows, hidden, constants)
    accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
    selectors = gdn_module.build_rows_selectors(accepted_tensor, constants)
    assert len(selectors.onehot_bf16) == CHUNK_ROWS
    assert selectors.committed_mask.torch_shards()[0].reshape(-1).tolist() == [float(j <= accepted) for j in range(32)]
    module.commit_rows(state_rows, rows_state, selectors)

    committed = _cat(state_rows.recurrent, 1)
    expected = _cat(state_step.recurrent, 1)
    err = (committed - expected).abs().max().item()
    print(
        f"chunk commit a={accepted}: state vs {accepted + 1} steps max abs {err:.2e} (tol {_state_tolerance(expected):.2e})"
    )
    assert err <= _state_tolerance(expected)
    if accepted == CHUNK_ROWS - 1:
        # A full chunk commits the forward pass's own final state: no second numerics.
        assert torch.equal(committed, _cat(result.final_state, 1))
    expected_history = torch.cat([_cat(slot, 3) for slot in state_step.conv_window()[:3]], dim=2)
    history = _cat(rows_state.history, 3)
    assert torch.equal(history[:, :, :3], expected_history) and torch.count_nonzero(history[:, :, 3:]) == 0
    state_rows.conv_phase = (accepted + 1) % 4  # the hand-off: the host sets the phase, the sync fills the ring
    module.sync_state_from_rows_history(rows_state, state_rows)
    for slot, expected_slot in zip(state_rows.conv_window()[:3], state_step.conv_window()[:3]):
        assert torch.equal(_cat(slot, 3), _cat(expected_slot, 3))
    # The next 1-row step agrees on both sides (the ring, phase and state are the decode's).
    torch.manual_seed(35)
    next_hidden = torch.randn(1, 1, 2560).to(torch.bfloat16)
    step_next = _sequential(module, state_step, next_hidden)[0]
    rows_next = _sequential(module, state_rows, next_hidden)[0]
    assert (rows_next.float() - step_next.float()).abs().max().item() <= _output_tolerance(step_next)


@pytest.mark.parametrize("accepted", (CHUNK_ROWS - 1, 4))
def test_chunk_path_commit_with_the_step_anchor_is_bitwise_the_one_row_path(fake, gdn_weights, accepted: int) -> None:
    """The prefill chunk's GDN commit (rows = 32 = CHUNK_ROWS: the accept scalar 31 for a full chunk, r - 1 for the
    padded tail of r real rows) with ``step_committed_rows`` (the layer's ``gdn_step_anchor``): the committed state
    is bitwise the 1-row path's after the committed rows and the FIR history is the ring's; the chunk-kernel form is
    reported beside it (the fake kernel is the fp32 torch WY form, so its drift is the algorithm's, not the device's
    TF32 drift that grows with the prompt length and motivated the re-anchor)."""

    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(32)
    hidden = torch.randn(1, CHUNK_ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 33)
    state_anchor = _clone_state(module, state_step)
    state_chunk = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(CHUNK_ROWS)
    accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)

    _sequential(module, state_step, hidden[:, : accepted + 1])
    committed = {}
    for state, anchor in ((state_anchor, True), (state_chunk, False)):
        result, rows_state = _rows_run(module, state, hidden, constants)
        selectors = gdn_module.build_rows_selectors(accepted_tensor, constants)
        assert selectors.committed_mask.torch_shards()[0].reshape(-1).tolist() == [
            float(j <= accepted) for j in range(CHUNK_ROWS)
        ]
        chunk_calls = len(fake.chunk.calls)
        module.commit_rows(state, rows_state, selectors, step_committed_rows=anchor)
        assert len(fake.chunk.calls) - chunk_calls == (0 if anchor else 1)
        _deallocate(result.final_state)
        committed[anchor] = _cat(state.recurrent, 1)
        if anchor:
            expected_history = torch.cat([_cat(slot, 3) for slot in state_step.conv_window()[:3]], dim=2)
            history = _cat(rows_state.history, 3)
            assert torch.equal(history[:, :, :3], expected_history) and torch.count_nonzero(history[:, :, 3:]) == 0
    expected = _cat(state_step.recurrent, 1)
    chunk_err = (committed[False] - expected).abs().max().item()
    print(f"chunk commit a={accepted}: anchored state bitwise the 1-row path's; chunk-form max abs {chunk_err:.2e}")
    assert torch.equal(committed[True], expected)
    torch.testing.assert_close(committed[False], expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("accepted", (31, 0, 1, 8))
def test_ple_rows_at_the_chunk_size_equal_32_prepared_steps_and_commit_selects_the_history(fake, accepted: int) -> None:
    module = _ple_module()
    torch.manual_seed(36)
    residual = torch.randn(CHUNK_ROWS, 4, 2560).to(torch.bfloat16)
    tokens = tuple(int(v) for v in torch.randint(0, 200_000, (CHUNK_ROWS,)))
    state = _seed_ple_state(module)
    rows_state = module.allocate_rows_state(CHUNK_ROWS)
    rows_state.load_from_state(state)

    deltas, histories, contexts = [], [], [None]
    for row, token in enumerate(tokens):
        prepared = module.prepare_decode_input(torch.tensor([[token]], dtype=torch.long), state)
        result = module.forward_prepared(_residual_rows(residual[row : row + 1]), prepared, state)
        deltas.append(_cat(result.residual_delta, 3))
        histories.append(torch.cat([_cat(slot, 3) for slot in state.conv], dim=1))
        contexts.append(tuple(int(v) for v in state.token_context[0]))

    prepared_rows = module.prepare_rows_input(tokens, rows_state)
    assert prepared_rows.contexts == tuple(contexts) and prepared_rows.embedding_rows.shape == (1, 1, CHUNK_ROWS, 640)
    result = module.forward_prepared_rows(_residual_rows(residual), prepared_rows, rows_state)
    delta_rows = _cat(result.residual_delta, 3)
    assert delta_rows.shape == (1, CHUNK_ROWS, 4, 2560)
    for row in range(CHUNK_ROWS):
        assert torch.equal(delta_rows[:, row : row + 1], deltas[row]), row

    constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", module.mesh_contract, rows=CHUNK_ROWS)
    accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
    module.commit_rows(rows_state, gdn_module.build_rows_selectors(accepted_tensor, constants))
    module.commit_rows_host(rows_state, prepared_rows, accepted)
    assert torch.equal(_cat(rows_state.history, 3), histories[accepted])
    assert rows_state.token_context == contexts[accepted + 1]
    back = module.allocate_state()
    rows_state.store_to_state(back)
    assert torch.equal(torch.cat([_cat(slot, 3) for slot in back.conv], dim=1), histories[accepted])
    assert back.token_context.tolist() == [list(contexts[accepted + 1])]
