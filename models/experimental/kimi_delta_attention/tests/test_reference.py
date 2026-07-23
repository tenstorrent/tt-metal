# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the independent KDA specification."""

import pytest
import torch
import torch.nn.functional as F

from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    recurrent_gated_delta_rule,
)
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import (
    causal_depthwise_conv_reference,
    kda_forward_reference,
    kda_gate_reference,
    kda_recurrent_reference,
    sigmoid_gated_rms_norm_reference,
    validate_reference_weights,
)


def _config() -> KDAConfig:
    return KDAConfig(
        hidden_size=64,
        num_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=4,
    )


def _random_weights(config: KDAConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260723)

    def normal(*shape: int, scale: float = 0.05) -> torch.Tensor:
        return scale * torch.randn(*shape, generator=generator)

    hidden = config.hidden_size
    key_rank, value_rank = config.head_k_dim, config.head_v_dim
    weights = {
        "q_proj.weight": normal(config.q_dim, hidden),
        "k_proj.weight": normal(config.k_dim, hidden),
        "v_proj.weight": normal(config.v_dim, hidden),
        "q_conv1d.weight": normal(config.q_dim, 1, config.conv_kernel_size, scale=0.2),
        "k_conv1d.weight": normal(config.k_dim, 1, config.conv_kernel_size, scale=0.2),
        "v_conv1d.weight": normal(config.v_dim, 1, config.conv_kernel_size, scale=0.2),
        "A_log": torch.log(torch.linspace(1.0, 4.0, config.num_heads)).reshape(1, 1, config.num_heads, 1),
        "f_a_proj.weight": normal(key_rank, hidden),
        "f_b_proj.weight": normal(config.num_heads * key_rank, key_rank),
        "dt_bias": normal(config.num_heads * key_rank),
        "b_proj.weight": normal(config.num_heads, hidden),
        "g_a_proj.weight": normal(value_rank, hidden),
        "g_b_proj.weight": normal(config.num_heads * value_rank, value_rank),
        "o_norm.weight": 1.0 + normal(value_rank),
        "o_proj.weight": normal(hidden, config.num_heads * value_rank),
    }
    return weights


def test_target_config_mapping() -> None:
    config = KDAConfig.from_model_config(
        {
            "hidden_size": 2304,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
            },
        }
    )

    assert config.hidden_size == 2304
    assert config.num_heads == 32
    assert config.head_k_dim == config.head_v_dim == 128
    assert config.q_dim == config.k_dim == config.v_dim == 4096
    assert config.conv_kernel_size == 4


@pytest.mark.parametrize("field", ["hidden_size", "num_heads", "head_k_dim", "head_v_dim"])
def test_config_rejects_nonpositive_dimensions(field: str, expect_error) -> None:
    values = {
        "hidden_size": 64,
        "num_heads": 2,
        "head_k_dim": 32,
        "head_v_dim": 32,
        "conv_kernel_size": 4,
        "norm_eps": 1e-5,
    }
    values[field] = 0
    with expect_error(ValueError, field):
        KDAConfig(**values)


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


def test_full_layer_split_equivalence() -> None:
    config = _config()
    weights = _random_weights(config)
    hidden = torch.randn(1, 6, config.hidden_size, generator=torch.Generator().manual_seed(29))

    full_output, full_state = kda_forward_reference(hidden, weights, config)
    first_output, first_state = kda_forward_reference(hidden[:, :4], weights, config)
    last_output, split_state = kda_forward_reference(hidden[:, 4:], weights, config, first_state)
    split_output = torch.cat((first_output, last_output), dim=1)

    assert torch.allclose(full_output, split_output, rtol=1e-5, atol=1e-6)
    assert torch.allclose(full_state.recurrent, split_state.recurrent, rtol=1e-5, atol=1e-6)
    assert torch.equal(full_state.q_convolution, split_state.q_convolution)
    assert torch.equal(full_state.k_convolution, split_state.k_convolution)
    assert torch.equal(full_state.v_convolution, split_state.v_convolution)


def test_weight_validation_reports_exact_name_and_shape(expect_error) -> None:
    config = _config()
    weights = _random_weights(config)
    weights["q_proj.weight"] = torch.empty(config.q_dim, config.hidden_size + 1)

    with expect_error(ValueError, r"q_proj\.weight shape .* !="):
        validate_reference_weights(weights, config)
