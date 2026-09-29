# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the independent KDA operation semantics."""

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda.ops import (
    causal_depthwise_conv_reference,
    kda_gate_reference,
    kda_recurrent_reference,
    sigmoid_gated_rms_norm_reference,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_equal


def test_causal_convolution_split_equivalence() -> None:
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(2, 7, 32, generator=generator)
    weight = torch.randn(32, 1, 4, generator=generator)
    full_output, full_state = causal_depthwise_conv_reference(inputs, weight)
    first_output, first_state = causal_depthwise_conv_reference(inputs[:, :5], weight)
    last_output, split_state = causal_depthwise_conv_reference(inputs[:, 5:], weight, first_state)
    assert_accurate(
        full_output,
        torch.cat((first_output, last_output), dim=1),
        name="split convolution output",
        pcc_threshold=0.999999,
    )
    assert_equal(full_state, split_state, name="split convolution state")


def test_causal_convolution_state_has_compact_storage() -> None:
    inputs = torch.randn(2, 64, 32)
    weight = torch.randn(32, 1, 4)
    _, state = causal_depthwise_conv_reference(inputs, weight)
    assert state.untyped_storage().nbytes() == state.numel() * state.element_size()


def test_gate_formulas() -> None:
    raw = torch.tensor([[[[-2.0, 0.5], [1.0, 3.0]]]])
    a_log = torch.log(torch.tensor([[[[2.0], [4.0]]]]))
    bias = torch.tensor([0.1, -0.2, 0.3, -0.4])
    assert_equal(-a_log.exp() * F.softplus(raw + bias.reshape(1, 1, 2, 2)), kda_gate_reference(raw, a_log, bias))
    bounded = kda_gate_reference(raw, a_log, bias, lower_bound=-5.0)
    assert_equal(-5.0 * torch.sigmoid(a_log.exp() * (raw + bias.reshape(1, 1, 2, 2))), bounded)
    assert bool(((-5.0 <= bounded) & (bounded <= 0.0)).all())


def test_vector_decay_reduces_to_scalar_recurrence() -> None:
    generator = torch.Generator().manual_seed(23)
    q = torch.randn(1, 4, 2, 32, generator=generator)
    k = torch.randn(1, 4, 2, 32, generator=generator)
    v = torch.randn(1, 4, 2, 32, generator=generator)
    beta = torch.sigmoid(torch.randn(1, 4, 2, generator=generator))
    scalar_gate = -F.softplus(torch.randn(1, 4, 2, generator=generator))
    initial_state = 0.05 * torch.randn(1, 2, 32, 32, generator=generator)
    vector_output, vector_state = kda_recurrent_reference(
        q, k, v, scalar_gate.unsqueeze(-1).expand_as(q), beta, initial_state
    )

    q_norm = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6) * (32**-0.5)
    k_norm = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    state = initial_state.float().clone()
    scalar_outputs = []
    for token in range(q.shape[1]):
        state *= scalar_gate[:, token].exp().unsqueeze(-1).unsqueeze(-1)
        residual = v[:, token] - torch.einsum("bhk,bhkv->bhv", k_norm[:, token], state)
        state += torch.einsum("bhk,bhv->bhkv", k_norm[:, token], beta[:, token].unsqueeze(-1) * residual)
        scalar_outputs.append(torch.einsum("bhk,bhkv->bhv", q_norm[:, token], state))
    scalar_output = torch.stack(scalar_outputs, dim=1)
    assert_accurate(scalar_output, vector_output, pcc_threshold=0.999999)
    assert_accurate(state, vector_state, pcc_threshold=0.999999)


def test_output_norm_uses_sigmoid_gate() -> None:
    inputs = torch.tensor([[[[1.0, 2.0, -3.0]]]])
    gate = torch.tensor([[[[-2.0, 0.0, 2.0]]]])
    weight = torch.tensor([0.5, 1.0, 1.5])
    expected = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-5)
    expected = expected * weight * torch.sigmoid(gate)
    assert_equal(expected, sigmoid_gated_rms_norm_reference(inputs, gate, weight, eps=1e-5))
