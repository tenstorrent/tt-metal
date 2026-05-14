# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
GDN recurrence correctness test.

Reference: ``torch_recurrent_gated_delta_rule`` from Hugging Face ``transformers``
``modular_qwen3_next.py`` (Qwen3-Next linear attention recurrence).

https://github.com/huggingface/transformers/blob/ddb841f48888e3fcf50c3f2a570ac9774aa7373c/src/transformers/models/qwen3_next/modular_qwen3_next.py#L294

Compares a naive ttnn implementation (``gated_delta.gdn_recurrence_fused_inplace``)
over sequence length ``SEQ_LEN`` (64) against that reference. No conv1d or projections.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.unit_tests.gated_delta import (
    chunked_gdn_recurrence_experimental_op,
    chunked_gdn_recurrence_fused_inplace,
    gdn_recurrence_fused_inplace,
)
from models.common.utility_functions import comp_pcc

import tracy

# Sequence length for this test (prefill-style recurrence along time).

# Qwen3-0.6B linear-attention head layout (HF ``Qwen3NextGatedDeltaNet``):
# Q/K projections use ``num_k_heads``; V, g, and beta use ``num_v_heads``.
# Before GDN, Q and K are repeated along the head dim to match V (see modular_qwen3_next.py).
NUM_QK_HEADS = 16
NUM_V_HEADS = 48
V_HEADS_PER_QK_HEAD = NUM_V_HEADS // NUM_QK_HEADS


def expand_qk_heads_for_gdn(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    num_qk_heads: int = NUM_QK_HEADS,
    num_v_heads: int = NUM_V_HEADS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat Q/K head dim to ``num_v_heads`` (HF ``repeat_interleave`` on head axis)."""
    factor = num_v_heads // num_qk_heads
    assert num_v_heads % num_qk_heads == 0
    if factor > 1:
        query = query.repeat_interleave(factor, dim=1)
        key = key.repeat_interleave(factor, dim=1)
    return query, key


# --- Reference (aligned with HF modular_qwen3_next.py, linked above) -----------------


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """Match FLA / Qwen3-Next ``l2norm``."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    """Port of HF ``torch_recurrent_gated_delta_rule`` (same tensor semantics)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    last_recurrent_states: list[torch.Tensor] = []
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        last_recurrent_states.append(last_recurrent_state.clone())
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    last_recurrent_states = torch.stack(last_recurrent_states, dim=2)
    core_attn_out = core_attn_out.contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state, last_recurrent_states


def preprocess_like_hf(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same preprocessing as inside ``torch_recurrent_gated_delta_rule`` before the time loop."""
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    return query, key, value, beta, g


def pearson_correlation_coefficient(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between the flattened elements of ``a`` and ``b``."""
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_gdn_kernel_correctness(mesh_device, reset_seeds):
    """Match HF recurrent GDN over ``SEQ_LEN`` (64) tokens vs naive ttnn stepping (Qwen3-0.6B heads)."""

    SEQ_LEN = 64
    device = mesh_device
    batch_size = 2
    Dk, Dv = 128, 128
    num_cores = min(10, batch_size * NUM_V_HEADS)

    logger.info(
        f"Testing GDN: batch={batch_size}, qk_heads={NUM_QK_HEADS}, v_heads={NUM_V_HEADS}, "
        f"seq_len={SEQ_LEN}, Dk={Dk}, Dv={Dv}"
    )

    torch.manual_seed(432)
    query = torch.randn(batch_size, NUM_QK_HEADS, SEQ_LEN, Dk, dtype=torch.float32)
    key = torch.randn(batch_size, NUM_QK_HEADS, SEQ_LEN, Dk, dtype=torch.float32)
    value = torch.randn(batch_size, NUM_V_HEADS, SEQ_LEN, Dv, dtype=torch.float32)
    g = torch.randn(batch_size, NUM_V_HEADS, SEQ_LEN, dtype=torch.float32) * 0.5 - 1.0
    beta = torch.randn(batch_size, NUM_V_HEADS, SEQ_LEN, dtype=torch.float32)

    query, key = expand_qk_heads_for_gdn(query, key)

    out_ref, state_ref = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # out_ref: [B, H, S, Dv] (same layout as ``core_attn_out`` in the reference loop)
    logger.info(f"Reference output: shape={out_ref.shape}, range=[{out_ref.min():.4f}, {out_ref.max():.4f}]")

    q_f, k_f, v_f, beta_f, g_f = preprocess_like_hf(query, key, value, g, beta, use_qk_l2norm_in_kernel=True)

    def to_tt(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    state_tt = ttnn.from_torch(
        torch.zeros(batch_size, NUM_V_HEADS, Dk, Dv, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    q_tt = to_tt(q_f)
    k_tt = to_tt(k_f)
    v_tt = to_tt(v_f)
    g_tt = to_tt(g_f)
    beta_tt = to_tt(beta_f)

    step_outputs: list[torch.Tensor] = []
    for t in range(SEQ_LEN):
        q_t = q_tt[:, :, t : t + 1]
        k_t = k_tt[:, :, t : t + 1]
        v_t = v_tt[:, :, t : t + 1]
        g_t = g_tt[:, :, t : t + 1]
        beta_t = beta_tt[:, :, t : t + 1]

        output_tt = gdn_recurrence_fused_inplace(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            state_tt,
            num_cores=num_cores,
            iter=t,
        )
        step_outputs.append(ttnn.to_torch(output_tt).float().squeeze(1).clone())

    out_tt_stacked = torch.stack(step_outputs, dim=2)  # [BH, S, Dv]
    out_tt_cpu = out_tt_stacked.reshape(batch_size, NUM_V_HEADS, SEQ_LEN, Dv).contiguous()
    state_tt_cpu = ttnn.to_torch(state_tt).float()

    logger.info(f"ttnn output: shape={out_tt_cpu.shape}, range=[{out_tt_cpu.min():.4f}, {out_tt_cpu.max():.4f}]")

    out_ref_f = out_ref.float()
    out_diff = (out_ref_f - out_tt_cpu).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()

    pcc = pearson_correlation_coefficient(out_ref_f, out_tt_cpu)

    logger.info("Output comparison:")
    logger.info(f"  Max diff: {out_max_diff:.6f}")
    logger.info(f"  Mean diff: {out_mean_diff:.6f}")
    logger.info(f"  PCC: {pcc:.6f}")

    state_diff = (state_ref - state_tt_cpu).abs()
    state_max_diff = state_diff.max().item()
    state_pcc = pearson_correlation_coefficient(state_ref, state_tt_cpu)

    logger.info("State comparison:")
    logger.info(f"  Max diff: {state_max_diff:.6f}")
    logger.info(f"  PCC: {state_pcc:.6f}")

    assert pcc > 0.999, f"Output PCC too low: {pcc:.6f}"
    assert state_pcc > 0.999, f"State PCC too low: {state_pcc:.6f}"

    logger.info(f"PASSED: ttnn matches HF reference (output PCC={pcc:.4f}, state PCC={state_pcc:.4f})")


@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("SEQ_LEN", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("num_heads", [NUM_V_HEADS])
def test_chunked_gdn_kernel_correctness(device, reset_seeds, num_heads, SEQ_LEN):
    """Match HF recurrent GDN over ``SEQ_LEN`` (64) tokens vs naive ttnn stepping."""
    batch_size = 4
    Dk, Dv = 128, 256

    logger.info(
        f"Testing GDN: batch={batch_size}, qk_heads={NUM_QK_HEADS}, v_heads={NUM_V_HEADS}, "
        f"seq_len={SEQ_LEN}, Dk={Dk}, Dv={Dv}"
    )

    torch.manual_seed(432)
    query = torch.randn(batch_size, num_heads, SEQ_LEN, Dk, dtype=torch.float32)
    key = torch.randn(batch_size, num_heads, SEQ_LEN, Dk, dtype=torch.float32)
    value = torch.randn(batch_size, num_heads, SEQ_LEN, Dv, dtype=torch.float32)
    g = torch.randn(batch_size, num_heads, SEQ_LEN, dtype=torch.float32) * 0.5 - 1.0
    beta = torch.randn(batch_size, num_heads, SEQ_LEN, dtype=torch.float32)
    state = torch.randn(batch_size, NUM_V_HEADS, Dk, Dv, dtype=torch.bfloat16)
    out_ref, state_ref, _ = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # out_ref: [B, H, S, Dv] (same layout as ``core_attn_out`` in the reference loop)
    logger.info(f"Reference output: shape={out_ref.shape}, range=[{out_ref.min():.4f}, {out_ref.max():.4f}]")

    q_f, k_f, v_f, beta_f, g_f = preprocess_like_hf(query, key, value, g, beta, use_qk_l2norm_in_kernel=True)
    # Shapes [B, H, S, *]
    num_pairs = batch_size * num_heads

    def to_tt(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    state_tt = ttnn.from_torch(
        state,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    q_tt = to_tt(q_f)
    k_tt = to_tt(k_f)
    v_tt = to_tt(v_f)
    g_tt = to_tt(g_f)
    beta_tt = to_tt(beta_f)

    # step_outputs: list[torch.Tensor] = []
    # for t in range(SEQ_LEN):
    #     q_t =       q_tt[:, :, t:t+1]
    #     k_t =       k_tt[:, :, t:t+1]
    #     v_t =       v_tt[:, :, t:t+1]
    #     g_t =       g_tt[:, :, t:t+1]
    #     beta_t = beta_tt[:, :, t:t+1]

    #     output_tt = gdn_recurrence_fused_inplace(
    #         q_t,
    #         k_t,
    #         v_t,
    #         g_t,
    #         beta_t,
    #         state_tt,
    #         num_cores=num_cores,
    #         iter=t,
    #     )
    #     step_outputs.append(ttnn.to_torch(output_tt).float().squeeze(1).clone())
    tracy.signpost(
        "chunked_gdn_op_start",
        f"batch={batch_size}, qk_heads={NUM_QK_HEADS}, v_heads={NUM_V_HEADS}, seq_len={SEQ_LEN}, Dk={Dk}, Dv={Dv}",
    )
    out_tt_stacked = chunked_gdn_recurrence_fused_inplace(q_tt, k_tt, v_tt, g_tt, beta_tt, state_tt)
    tracy.signpost(
        "chunked_gdn_op_stop",
        f"batch={batch_size}, qk_heads={NUM_QK_HEADS}, v_heads={NUM_V_HEADS}, seq_len={SEQ_LEN}, Dk={Dk}, Dv={Dv}",
    )
    out_tt_cpu = out_tt_stacked.reshape(batch_size, NUM_V_HEADS, SEQ_LEN, Dv).contiguous()
    state_tt_cpu = ttnn.to_torch(state_tt).float()

    logger.info(f"ttnn output: shape={out_tt_cpu.shape}, range=[{out_tt_cpu.min():.4f}, {out_tt_cpu.max():.4f}]")

    out_ref_f = out_ref.float()
    out_diff = (out_ref_f - out_tt_cpu).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()

    pcc = pearson_correlation_coefficient(out_ref_f, out_tt_cpu)

    logger.info("Output comparison:")
    logger.info(f"  Max diff: {out_max_diff:.6f}")
    logger.info(f"  Mean diff: {out_mean_diff:.6f}")
    logger.info(f"  PCC: {pcc:.6f}")

    state_diff = (state_ref - state_tt_cpu).abs()
    state_max_diff = state_diff.max().item()
    state_pcc = pearson_correlation_coefficient(state_ref, state_tt_cpu)

    logger.info("State comparison:")
    logger.info(f"  Max diff: {state_max_diff:.6f}")
    logger.info(f"  PCC: {state_pcc:.6f}")

    assert pcc > 0.999, f"Output PCC too low: {pcc:.6f}"
    assert state_pcc > 0.999, f"State PCC too low: {state_pcc:.6f}"

    logger.info(f"PASSED: ttnn matches HF reference (output PCC={pcc:.4f}, state PCC={state_pcc:.4f})")


@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("SEQ_LEN", [64, 128, 256, 512, 1024])
def test_chunked_gdn_experimental_op_correctness(device, reset_seeds, SEQ_LEN):
    """Match HF recurrent GDN vs ``ttnn.experimental.chunked_gated_delta`` (Qwen3-0.6B heads)."""
    batch_size = 1
    Dk, Dv = 128, 256

    logger.info(
        f"Testing chunked GDN op: batch={batch_size}, qk_heads={NUM_QK_HEADS}, v_heads={NUM_V_HEADS}, "
        f"seq_len={SEQ_LEN}, Dk={Dk}, Dv={Dv}"
    )

    torch.manual_seed(432)
    query = torch.randn(batch_size, NUM_QK_HEADS, SEQ_LEN, Dk, dtype=torch.float32)
    key = torch.randn(batch_size, NUM_QK_HEADS, SEQ_LEN, Dk, dtype=torch.float32)
    value = torch.randn(batch_size, NUM_V_HEADS, SEQ_LEN, Dv, dtype=torch.float32)
    g = torch.randn(batch_size, NUM_V_HEADS, SEQ_LEN, dtype=torch.float32) * 0.5 - 1.0
    beta = torch.randn(batch_size, NUM_V_HEADS, SEQ_LEN, dtype=torch.float32)

    query, key = expand_qk_heads_for_gdn(query, key)

    state_host = torch.randn(batch_size, NUM_V_HEADS, Dk, Dv, dtype=torch.float32)

    out_ref, state_ref, all_states_ref = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=state_host,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    logger.info(f"Reference output: shape={out_ref.shape}, range=[{out_ref.min():.4f}, {out_ref.max():.4f}]")

    q_f, k_f, v_f, beta_f, g_f = preprocess_like_hf(query, key, value, g, beta, use_qk_l2norm_in_kernel=True)

    def to_tt(t: torch.Tensor, layout: ttnn.Layout = ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
        )

    state_tt = ttnn.from_torch(
        state_host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    q_tt = to_tt(q_f)
    k_tt = to_tt(k_f, ttnn.ROW_MAJOR_LAYOUT)
    v_tt = to_tt(v_f)
    g_tt = to_tt(g_f)
    beta_tt = to_tt(beta_f)

    tracy.signpost(
        "chunked_gdn_op_start",
        f"batch={batch_size}, qk_heads={NUM_QK_HEADS}, v_heads={NUM_V_HEADS}, seq_len={SEQ_LEN}, Dk={Dk}, Dv={Dv}",
    )
    out_tt_cpu, all_states_cpu = chunked_gdn_recurrence_experimental_op(q_tt, k_tt, v_tt, g_tt, beta_tt, state_tt)
    tracy.signpost(
        "chunked_gdn_op_stop",
        f"batch={batch_size}, qk_heads={NUM_QK_HEADS}, v_heads={NUM_V_HEADS}, seq_len={SEQ_LEN}, Dk={Dk}, Dv={Dv}",
    )

    out_tt_cpu = out_tt_cpu.float().reshape(batch_size, NUM_V_HEADS, SEQ_LEN, Dv)
    all_states_cpu = all_states_cpu.float().reshape(batch_size, NUM_V_HEADS, SEQ_LEN, Dk, Dv)
    state_tt_cpu = all_states_cpu[:, :, -1, :, :]
    # factor = factor.float().reshape(batch_size, NUM_V_HEADS, SEQ_LEN, Dk, Dk)
    # bktv = bktv.float().reshape(batch_size, NUM_V_HEADS, SEQ_LEN, Dk, Dv)
    # g_exp = torch.exp(g).reshape(batch_size, NUM_V_HEADS, SEQ_LEN, 1, 1)
    # exp_state = state_host.reshape(batch_size, NUM_V_HEADS, 1, Dk, Dv).expand(batch_size, NUM_V_HEADS, SEQ_LEN, Dk, Dv)
    # torch_ref = torch.matmul(factor, exp_state * g_exp) + bktv

    # pcc = pearson_correlation_coefficient(torch_out, torch_ref)
    # logger.info(f"PCC: {pcc:.6f}")
    # torch.set_printoptions(sci_mode=False)
    # assert pcc > 0.999, f"PCC too low: {pcc:.6f}"
    # logger.info(f"PASSED: chunked_gated_delta op matches HF reference (PCC={pcc:.4f})")
    # max_diff = diff.max().item()
    # mean_diff = diff.mean().item()
    # min_diff = diff.min().item()
    # logger.info(f"Max diff: {max_diff:.6f}")
    # logger.info(f"Mean diff: {mean_diff:.6f}")
    # assert max_diff < 0.15, f"Max diff too high: {max_diff:.6f}"

    # out_tt_cpu = out_tt_stacked
    # state_tt_cpu = ttnn.to_torch(state_tt).float()

    # logger.info(f"ttnn output: shape={out_tt_cpu.shape}, range=[{out_tt_cpu.min():.4f}, {out_tt_cpu.max():.4f}]")

    out_ref_f = out_ref.float()
    out_diff = (out_ref_f - out_tt_cpu).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()

    pcc = pearson_correlation_coefficient(out_ref_f, out_tt_cpu)

    logger.info("Output comparison:")
    logger.info(f"  Max diff: {out_max_diff:.6f}")
    logger.info(f"  Mean diff: {out_mean_diff:.6f}")
    logger.info(f"  PCC: {pcc:.6f}")

    state_diff = (state_ref - state_tt_cpu).abs()
    state_max_diff = state_diff.max().item()
    state_pcc = pearson_correlation_coefficient(state_ref, state_tt_cpu)

    logger.info("State comparison:")
    logger.info(f"  Max diff: {state_max_diff:.6f}")
    logger.info(f"  PCC: {state_pcc:.6f}")

    assert pcc > 0.999, f"Output PCC too low: {pcc:.6f}"
    assert state_pcc > 0.999, f"State PCC too low: {state_pcc:.6f}"

    logger.info(
        f"PASSED: chunked_gated_delta op matches HF reference " f"(output PCC={pcc:.4f}, state PCC={state_pcc:.4f})"
    )
