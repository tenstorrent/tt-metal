# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the independent KDA specification."""

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda.ops import (
    causal_depthwise_conv_reference,
    kda_gate_reference,
    kda_recurrent_reference,
    sigmoid_gated_rms_norm_reference,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import assert_accurate, assert_equal
from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    recurrent_gated_delta_rule,
)


def test_causal_convolution_split_equivalence() -> None:
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(2, 7, 32, generator=generator)
    weight = torch.randn(32, 1, 4, generator=generator)

    full_output, full_state = causal_depthwise_conv_reference(inputs, weight)
    first_output, first_state = causal_depthwise_conv_reference(inputs[:, :5], weight)
    last_output, split_state = causal_depthwise_conv_reference(inputs[:, 5:], weight, first_state)

    split_output = torch.cat((first_output, last_output), dim=1)
    assert_accurate(full_output, split_output, name="split convolution output", pcc_threshold=0.999999)
    assert_equal(full_state, split_state, name="split convolution state")


def test_gate_matches_authoritative_formula() -> None:
    raw = torch.tensor([[[[-2.0, 0.5], [1.0, 3.0]]]])
    a_log = torch.log(torch.tensor([[[[2.0], [4.0]]]]))
    bias = torch.tensor([0.1, -0.2, 0.3, -0.4])

    actual = kda_gate_reference(raw, a_log, bias)
    expected = -a_log.exp() * F.softplus(raw + bias.reshape(1, 1, 2, 2))

    assert_equal(expected, actual, name="gate")
    assert_equal(torch.ones_like(actual, dtype=torch.bool), actual < 0, name="gate is negative")


def test_bounded_gate_matches_kimi_k3_formula() -> None:
    raw = torch.tensor([[[[-2.0, 0.5], [1.0, 3.0]]]])
    a_log = torch.log(torch.tensor([[[[2.0], [4.0]]]]))
    bias = torch.tensor([0.1, -0.2, 0.3, -0.4])

    actual = kda_gate_reference(raw, a_log, bias, lower_bound=-5.0)
    expected = -5.0 * torch.sigmoid(a_log.exp() * (raw + bias.reshape(1, 1, 2, 2)))

    assert_equal(expected, actual, name="bounded gate")
    in_bounds = (-5.0 <= actual) & (actual <= 0.0)
    assert_equal(torch.ones_like(in_bounds), in_bounds, name="bounded gate range")


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
    assert_accurate(gdn_output, kda_output, name="KDA vs GDN output", pcc_threshold=0.99999)
    assert_accurate(gdn_state, kda_state, name="KDA vs GDN state", pcc_threshold=0.99999)


def test_output_norm_uses_sigmoid_gate() -> None:
    inputs = torch.tensor([[[[1.0, 2.0, -3.0]]]])
    gate = torch.tensor([[[[-2.0, 0.0, 2.0]]]])
    weight = torch.tensor([0.5, 1.0, 1.5])

    actual = sigmoid_gated_rms_norm_reference(inputs, gate, weight, eps=1e-5)
    normalized = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-5)
    expected = normalized * weight * torch.sigmoid(gate)

    assert_equal(expected, actual, name="sigmoid gated RMSNorm")
