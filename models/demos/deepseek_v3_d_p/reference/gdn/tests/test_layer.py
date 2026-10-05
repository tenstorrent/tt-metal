# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the GDN reference layer against transformers (no device fixture).

Run with a mock cluster so the repository conftest's ttnn import touches no hardware:
    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_8xP150.yaml> pytest models/demos/deepseek_v3_d_p/reference/gdn/tests
"""

from dataclasses import replace

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.layer import GDNReferenceState, gdn_forward_reference
from models.demos.deepseek_v3_d_p.reference.gdn.tests.helpers import TINY, random_weights


class _Qwen4ExpRMSNormGated(torch.nn.Module):
    """``Qwen4ExpTextRMSNormGated`` of transformers@56d3afc0 ``models/qwen4_exp/modeling_qwen4_exp.py:176-192``,
    transcribed: the installed transformers has no ``qwen4_exp``. The Flash-Next GDN layer equals the
    ``Qwen3_5GatedDeltaNet`` with this norm (tt-work ``artifacts/scripts/gdn_flash_next_layer_diff.out``)."""

    def __init__(self, hidden_size: int, eps: float, activation: str) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        from transformers.activations import ACT2FN

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def _transformers_layer(config: GDNConfig, weights: dict[str, torch.Tensor]) -> torch.nn.Module:
    """transformers' own Qwen3_5GatedDeltaNet (torch fallback) in FP32; for a sigmoid gate its norm is replaced by
    the qwen4_exp norm with ``activation=output_gate_type``, which is the whole qwen4_exp GDN difference."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    hf_config = Qwen3_5TextConfig(
        hidden_size=config.hidden_size,
        linear_num_key_heads=config.num_key_heads,
        linear_num_value_heads=config.num_value_heads,
        linear_key_head_dim=config.head_k_dim,
        linear_value_head_dim=config.head_v_dim,
        linear_conv_kernel_dim=config.conv_kernel_size,
        rms_norm_eps=config.norm_eps,
        hidden_act="silu",
    )
    layer = hf.Qwen3_5GatedDeltaNet(hf_config, layer_idx=0)
    if config.output_gate_activation != "silu":
        layer.norm = _Qwen4ExpRMSNormGated(config.head_v_dim, config.norm_eps, config.output_gate_activation)
    layer = layer.float()
    layer.load_state_dict({k: v.float() for k, v in weights.items()})
    return layer


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_reference_matches_transformers_layer(activation: str):
    """Independent check: the reference equals transformers' GDN layer in FP32 for both output-gate activations."""
    config = replace(TINY, output_gate_activation=activation)
    weights = random_weights(config)
    x = torch.randn(1, 70, config.hidden_size, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    with torch.no_grad():
        expected = _transformers_layer(config, weights)(x.float())[0]
    actual, _ = gdn_forward_reference(x[0], weights, config)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


def test_output_gate_activation_is_observable():
    """Negative control for the test above: the two activations give clearly different outputs on the same inputs."""
    weights = random_weights(TINY)
    x = torch.randn(70, TINY.hidden_size, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    silu, _ = gdn_forward_reference(x, weights, TINY)
    sigmoid, _ = gdn_forward_reference(x, weights, replace(TINY, output_gate_activation="sigmoid"))
    assert (silu - sigmoid).norm() > 0.1 * silu.norm()


def test_reference_state_matches_transformers_recurrence():
    """Final recurrent state against transformers' torch_recurrent_gated_delta_rule, nonzero initial state."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    from models.demos.deepseek_v3_d_p.reference.gdn.layer import delta_rule_recurrence

    g = torch.Generator().manual_seed(2)
    T, H, K, V = 37, 3, 16, 16
    q, k = torch.randn(T, H, K, generator=g), torch.randn(T, H, K, generator=g)
    v = torch.randn(T, H, V, generator=g)
    gate = -torch.rand(T, H, generator=g) * 3
    beta = torch.rand(T, H, generator=g)
    s0 = torch.randn(H, K, V, generator=g)
    hf_out, hf_state = hf.torch_recurrent_gated_delta_rule(
        q[None], k[None], v[None], gate[None], beta[None], s0[None], True, use_qk_l2norm_in_kernel=True
    )
    qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5
    kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    out, state = delta_rule_recurrence(qn, kn, v, gate, beta, s0)
    torch.testing.assert_close(out, hf_out[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state, hf_state[0], rtol=1e-5, atol=1e-5)


def test_chained_chunks_equal_one_pass():
    weights = random_weights(TINY)
    x = torch.randn(96, TINY.hidden_size, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    whole, whole_state = gdn_forward_reference(x, weights, TINY)
    state, outs = GDNReferenceState.zeros(TINY), []
    for lo, hi in ((0, 32), (32, 61), (61, 96)):
        out, state = gdn_forward_reference(x[lo:hi], weights, TINY, state)
        outs.append(out)
    torch.testing.assert_close(torch.cat(outs), whole, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.recurrent, whole_state.recurrent, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.conv, x[-3:].float() @ weights["in_proj_qkv.weight"].float().T)
