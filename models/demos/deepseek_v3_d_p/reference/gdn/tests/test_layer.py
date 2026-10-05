# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the GDN reference layer against transformers (no device fixture).

Run with a mock cluster so the repository conftest's ttnn import touches no hardware:
    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_8xP150.yaml> pytest models/demos/deepseek_v3_d_p/reference/gdn/tests
"""

import torch

from models.demos.deepseek_v3_d_p.reference.gdn.layer import PREFIX, GdnShape, GdnState, gdn_layer_reference

TINY = GdnShape(hidden=64, num_k_heads=2, num_v_heads=6, head_k_dim=16, head_v_dim=16, conv_kernel=4, eps=1e-6)


def _tiny_weights(shape: GdnShape, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def w(*dims, scale):
        return (torch.randn(*dims, generator=g) * scale).to(torch.bfloat16)

    return {
        PREFIX + "in_proj_qkv.weight": w(shape.conv_dim, shape.hidden, scale=0.2),
        PREFIX + "in_proj_z.weight": w(shape.value_dim, shape.hidden, scale=0.2),
        PREFIX + "in_proj_a.weight": w(shape.num_v_heads, shape.hidden, scale=0.2),
        PREFIX + "in_proj_b.weight": w(shape.num_v_heads, shape.hidden, scale=0.2),
        PREFIX + "out_proj.weight": w(shape.hidden, shape.value_dim, scale=0.2),
        PREFIX + "conv1d.weight": w(shape.conv_dim, 1, shape.conv_kernel, scale=0.5),
        PREFIX + "A_log": w(shape.num_v_heads, scale=1.0),
        PREFIX + "dt_bias": w(shape.num_v_heads, scale=1.0),
        PREFIX + "norm.weight": (1 + w(shape.head_v_dim, scale=0.1).float()).to(torch.bfloat16),
    }


def test_reference_matches_transformers_layer():
    """Independent check: the reference equals transformers' own Qwen3_5GatedDeltaNet (torch fallback) in FP32."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        hidden_size=TINY.hidden,
        linear_num_key_heads=TINY.num_k_heads,
        linear_num_value_heads=TINY.num_v_heads,
        linear_key_head_dim=TINY.head_k_dim,
        linear_value_head_dim=TINY.head_v_dim,
        linear_conv_kernel_dim=TINY.conv_kernel,
        rms_norm_eps=TINY.eps,
        hidden_act="silu",
    )
    layer = hf.Qwen3_5GatedDeltaNet(config, layer_idx=0).float()
    weights = _tiny_weights(TINY)
    layer.load_state_dict({k[len(PREFIX) :]: v.float() for k, v in weights.items()})
    x = torch.randn(1, 70, TINY.hidden, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    with torch.no_grad():
        expected = layer(x.float())[0]
    actual, _ = gdn_layer_reference(weights, TINY, x[0])
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


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
    weights = _tiny_weights(TINY)
    x = torch.randn(96, TINY.hidden, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    whole, whole_state = gdn_layer_reference(weights, TINY, x)
    state, outs = GdnState.zeros(TINY), []
    for lo, hi in ((0, 32), (32, 61), (61, 96)):
        out, state = gdn_layer_reference(weights, TINY, x[lo:hi], state)
        outs.append(out)
    torch.testing.assert_close(torch.cat(outs), whole, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.recurrent, whole_state.recurrent, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.conv, x[-3:].float() @ weights[PREFIX + "in_proj_qkv.weight"].float().T)
