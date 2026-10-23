# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP v2 step 2 (b): the GDN chunk path against the fp32 step recurrence, with the masked catch-up.

No device.  The chunk model is the torch reference of the chunk kernel (``torch_functional.delta_rule_ops``:
fp32 WY form, chunk 32 or 64); the step model is the CPU oracle's ``gated_delta_recurrent``.  Per-device
geometry: 12 value heads, K = V = 128.  q and k are L2-normalized once, up front, in bf16 (the oracle's
normalize-in-input-dtype rule): torch's bf16 reduction blocks differently for 1 and 5 rows, and a 1-ULP flip
in a normalized k moves the state by 4e-3, which would hide the algorithmic gap under test.  ULP gaps are
printed with ``-s``; the gate is fp32 closeness (rtol 1e-4) of the chunk form to the step form.
The diagonal-regularized WY form of ``ttnn_delta_rule_seq`` (QWEN_GDN_DIAG_ALPHA, default 0.25) is modelled
as well: only alpha = 0 is the step recurrence.  The conv window rule for the GDN FIR and the dilated PLE conv
(new history = window rows [committed, committed + state_len)) is checked against the reference convolution.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.reference import _l2_norm, causal_depthwise_conv1d, gated_delta_recurrent
from models.demos.blackhole.qwen38_flash_next.tools.mtp_v2_verify_reference import row_serial_torch, ulp_distance
from models.demos.blackhole.qwen38_flash_next.tt.gdn import (
    Qwen38GDN,
    Qwen38GDNDimensions,
    Qwen38GDNState,
    Qwen38GDNWeights,
)
from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)

HEADS = 12
HEAD_DIM = 128
ROWS = 5
RTOL = 1e-4  # measured: output rel <= 2.1e-6, state rel <= 1.2e-5 against the step recurrence
ATOL = 5e-6


def _rows(rows: int, seed: int = 1) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "q": _l2_norm(torch.randn(1, rows, HEADS, HEAD_DIM).to(torch.bfloat16)),
        "k": _l2_norm(torch.randn(1, rows, HEADS, HEAD_DIM).to(torch.bfloat16)),
        "v": torch.randn(1, rows, HEADS, HEAD_DIM).to(torch.bfloat16),
        "beta": torch.sigmoid(torch.randn(1, rows, HEADS)).to(torch.bfloat16),
        "g": -torch.rand(1, rows, HEADS) * 0.5,  # fp32 log decay, as the oracle's -exp(A_log) * softplus(.)
        "state": torch.randn(1, HEADS, HEAD_DIM, HEAD_DIM) * 0.3,
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
    x = _rows(rows)
    step_out, step_out_fp32, step_state = _step(x)
    chunk_out, chunk_state = _chunk(x, chunk_size)
    assert step_out.dtype == torch.bfloat16 and chunk_out.dtype == torch.float32
    assert chunk_out.shape == (1, rows, HEADS, HEAD_DIM) and chunk_state.shape == x["state"].shape
    _report(f"T={rows} chunk={chunk_size} chunk vs step", chunk_out, step_out_fp32, step_out, chunk_state, step_state)


@pytest.mark.parametrize("committed", range(ROWS + 1))
def test_masked_catch_up_equals_the_committed_prefix(committed: int) -> None:
    """Design 2.4 step 1: beta and g of rows at or past the committed count are zeroed; k and v may stay."""

    x = _rows(ROWS, seed=2)
    mask = (torch.arange(ROWS) < committed).to(torch.float32).reshape(1, ROWS, 1)
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

    x = _rows(ROWS, seed=3)
    for accepted in range(ROWS):
        short = (torch.arange(ROWS) < accepted).to(torch.float32).reshape(1, ROWS, 1)
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
    x = _rows(ROWS, seed=4)
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
    hidden = torch.randn(1, ROWS, 2560).to(torch.bfloat16)
    state = Qwen38GDNState(
        conv=torch.randn(1, gdn.weights.dimensions.qkv_width, 3).to(torch.bfloat16),
        recurrent=torch.randn(1, 48, 128, 128) * 0.3,
    )
    with row_serial_torch():
        output, next_state = gdn.forward(hidden, state)
        rows = []
        threaded = state
        for r in range(ROWS):
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
    new_rows = torch.randn(1, channels, ROWS).to(torch.bfloat16)
    window = torch.cat([history, new_rows], dim=-1)  # [history | k+1 new rows], the design's W

    batched, _ = causal_depthwise_conv1d(new_rows, taps, history, dilation=dilation, activation="silu")
    for j in range(ROWS):
        # Output row j reads window rows j, j + d, j + 2d, j + 3d: the same op on the window slice starting at j.
        single, _ = causal_depthwise_conv1d(
            new_rows[..., j : j + 1], taps, window[..., j : j + state_len], dilation=dilation, activation="silu"
        )
        assert torch.equal(batched[..., j], single[..., 0]), (label, j)
    for committed in range(ROWS + 1):
        threaded = history
        for j in range(committed):
            _, threaded = causal_depthwise_conv1d(new_rows[..., j : j + 1], taps, threaded, dilation=dilation)
        # New history after committing c rows: window rows [c, c + state_len) (the design's W[a+1 : a+5] with c = a+1).
        assert torch.equal(threaded, window[..., committed : committed + state_len]), (label, committed)
