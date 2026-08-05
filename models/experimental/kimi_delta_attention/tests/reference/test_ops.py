# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the independent KDA specification."""

import torch
import torch.nn.functional as F

from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    recurrent_gated_delta_rule,
)
from models.experimental.kimi_delta_attention.reference.ops import (
    causal_depthwise_conv_reference,
    kda_gate_reference,
    kda_recurrent_reference,
    sigmoid_gated_rms_norm_reference,
)


def test_causal_convolution_split_equivalence() -> None:
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(2, 7, 32, generator=generator)
    weight = torch.randn(32, 1, 4, generator=generator)

    full_output, full_state = causal_depthwise_conv_reference(inputs, weight)
    first_output, first_state = causal_depthwise_conv_reference(inputs[:, :5], weight)
    last_output, split_state = causal_depthwise_conv_reference(inputs[:, 5:], weight, first_state)

    assert torch.allclose(full_output, torch.cat((first_output, last_output), dim=1), rtol=1e-6, atol=1e-6)
    assert torch.equal(full_state, split_state)


def test_gate_matches_authoritative_formula() -> None:
    raw = torch.tensor([[[[-2.0, 0.5], [1.0, 3.0]]]])
    a_log = torch.log(torch.tensor([[[[2.0], [4.0]]]]))
    bias = torch.tensor([0.1, -0.2, 0.3, -0.4])

    actual = kda_gate_reference(raw, a_log, bias)
    expected = -a_log.exp() * F.softplus(raw + bias.reshape(1, 1, 2, 2))

    assert torch.equal(actual, expected)
    assert torch.all(actual < 0)


def test_bounded_gate_matches_kimi_k3_formula() -> None:
    raw = torch.tensor([[[[-2.0, 0.5], [1.0, 3.0]]]])
    a_log = torch.log(torch.tensor([[[[2.0], [4.0]]]]))
    bias = torch.tensor([0.1, -0.2, 0.3, -0.4])

    actual = kda_gate_reference(raw, a_log, bias, lower_bound=-5.0)
    expected = -5.0 * torch.sigmoid(a_log.exp() * (raw + bias.reshape(1, 1, 2, 2)))

    assert torch.equal(actual, expected)
    assert torch.all((-5.0 <= actual) & (actual <= 0.0))


def test_vector_decay_reduces_to_trusted_scalar_gdn() -> None:
    generator = torch.Generator().manual_seed(23)
    q = torch.randn(1, 4, 2, 32, generator=generator)
    k = torch.randn(1, 4, 2, 32, generator=generator)
    v = torch.randn(1, 4, 2, 32, generator=generator)
    beta = torch.sigmoid(torch.randn(1, 4, 2, generator=generator))
    scalar_gate = -F.softplus(torch.randn(1, 4, 2, generator=generator))
    vector_gate = scalar_gate.unsqueeze(-1).expand(-1, -1, -1, 32)
    initial_state = 0.05 * torch.randn(1, 2, 32, 32, generator=generator)

    kda_output, kda_state = kda_recurrent_reference(q, k, v, vector_gate, beta, initial_state)
    gdn_output, gdn_state = recurrent_gated_delta_rule(
        q,
        k,
        v,
        beta,
        scalar_gate,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm=True,
    )

    assert gdn_state is not None
    assert torch.allclose(kda_output, gdn_output, rtol=1e-5, atol=1e-6)
    assert torch.allclose(kda_state, gdn_state, rtol=1e-5, atol=1e-6)


def test_output_norm_uses_sigmoid_gate() -> None:
    inputs = torch.tensor([[[[1.0, 2.0, -3.0]]]])
    gate = torch.tensor([[[[-2.0, 0.0, 2.0]]]])
    weight = torch.tensor([0.5, 1.0, 1.5])

    actual = sigmoid_gated_rms_norm_reference(inputs, gate, weight, eps=1e-5)
    normalized = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-5)
    expected = normalized * weight * torch.sigmoid(gate)

    assert torch.equal(actual, expected)
