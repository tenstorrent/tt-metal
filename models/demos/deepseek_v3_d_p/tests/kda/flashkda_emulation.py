# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU emulation of FlashKDA's precision, for error budgets where no FlashKDA output exists.

FlashKDA (a3e42bbb ``csrc/smxx/fwd_kernel2.cuh``) processes 16-token chunks and stores the recurrent state
in BF16 after every chunk. The 2026-10-07 reference comparison (``tt-work/artifacts/kda-accuracy``, "R4")
reproduced FlashKDA's layer output at PCC >= 0.99994 on Kimi-K3 layers 1-3 with exactly that emulation: an
FP32 chunked recurrence with 16-token chunks whose state is rounded to BF16 after each chunk. Around it, the
layer follows vLLM's BF16 activations: every stage output (projections, convolution, gate, beta, recurrence,
gated norm, output) is rounded to BF16.

This is a model of FlashKDA, not FlashKDA: budgets from it must be validated against recorded FlashKDA
outputs (the R1 trace) before they gate anything.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.ops import (
    causal_depthwise_conv_reference,
    kda_gate_reference,
    l2_norm_reference,
    sigmoid_gated_rms_norm_reference,
)
from models.demos.deepseek_v3_d_p.reference.kda.tests.fla_naive_kda import naive_chunk_kda

FLASHKDA_CHUNK = 16


def flashkda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 output ``[B, T, H, V]`` and final state ``[B, H, K, V]`` of the FlashKDA-precision recurrence.

    Takes raw (un-normalized) q/k like ``kda_recurrent_reference``. A partial last chunk is padded with
    zero-update tokens (q = k = 0, gate = 0, beta = 0), which leave the state unchanged except for the
    BF16 rounding FlashKDA also applies at the end of a chunk.
    """
    batch, sequence, heads, key_dim = q.shape
    pad = -sequence % FLASHKDA_CHUNK

    def padded(tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor.float(), (0, 0) * (tensor.ndim - 2) + (0, pad)) if pad else tensor.float()

    state = (
        torch.zeros(batch, heads, key_dim, v.shape[-1])
        if initial_state is None
        else initial_state.float().bfloat16().float()
    )
    output, final_state = naive_chunk_kda(
        padded(l2_norm_reference(q)),
        padded(l2_norm_reference(k)),
        padded(v),
        padded(gate),
        padded(beta),
        initial_state=state,
        output_final_state=True,
        chunk_size=FLASHKDA_CHUNK,
        compute_dtype=torch.float32,
        state_dtype=torch.bfloat16,
    )
    return output[:, :sequence], final_state


def flashkda_layer(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config: KDAConfig,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 layer output ``[B, T, hidden]`` and FP32 final recurrent state, with FlashKDA/vLLM precision.

    The convolution history starts empty; ``initial_state`` is the recurrent state only.
    """

    def bf16(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.bfloat16().float()

    def project(inputs: torch.Tensor, name: str) -> torch.Tensor:
        return bf16(F.linear(inputs, weights[name].float()))

    hidden = hidden_states.float()
    batch, sequence, _ = hidden.shape
    heads, key_dim, value_dim = config.num_heads, config.head_k_dim, config.head_v_dim

    def convolved(name: str) -> torch.Tensor:
        output, _ = causal_depthwise_conv_reference(
            project(hidden, f"{name}_proj.weight"), weights[f"{name}_conv1d.weight"]
        )
        return bf16(output)

    q = convolved("q").reshape(batch, sequence, heads, key_dim)
    k = convolved("k").reshape(batch, sequence, heads, key_dim)
    v = convolved("v").reshape(batch, sequence, heads, value_dim)
    raw_gate = project(project(hidden, "f_a_proj.weight"), "f_b_proj.weight").reshape(batch, sequence, heads, key_dim)
    gate = bf16(kda_gate_reference(raw_gate, weights["A_log"], weights["dt_bias"], config.gate_lower_bound))
    beta = bf16(torch.sigmoid(project(hidden, "b_proj.weight")))
    recurrence, final_state = flashkda_recurrence(q, k, v, gate, beta, initial_state)
    if config.use_full_rank_gate:
        output_gate = project(hidden, "g_proj.weight")
    else:
        output_gate = project(project(hidden, "g_a_proj.weight"), "g_b_proj.weight")
    normalized = sigmoid_gated_rms_norm_reference(
        bf16(recurrence),
        output_gate.reshape(batch, sequence, heads, value_dim),
        weights["o_norm.weight"],
        config.norm_eps,
    )
    output = F.linear(bf16(normalized.reshape(batch, sequence, -1)), weights["o_proj.weight"].float())
    return output.bfloat16(), final_state
