# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for gated_delta_net (immutable spec for the implementer).

Each case is ((1, B*T, C), (B, Hv, K, V)); the two shapes fix every tensor of the call.
"""

import pytest
import torch

import ttnn
from eval.metrics import default_tolerance
from ttnn.operations.gated_delta_net import gated_delta_net

HISTORY = 3
L2_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0


def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def reference(mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight, epsilon=1e-6):
    """M1..M14 in float32, token by token."""
    f32 = torch.float32
    mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight = (
        t.to(f32) for t in (mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight)
    )
    batch, value_heads, key_dim, value_dim = recurrent_state.shape
    rows, channels = mixed_qkv.shape[-2:]
    tokens = rows // batch
    key_heads = (channels - value_heads * value_dim) // (2 * key_dim)
    ratio = value_heads // key_heads

    x = mixed_qkv.reshape(batch, tokens, channels)
    hist = conv_state.reshape(batch, HISTORY, channels)
    window = torch.cat([hist, x], dim=1)
    conv = sum(conv_weight[j] * window[:, j : j + tokens] for j in range(HISTORY + 1))
    y = conv * _sigmoid(conv)
    conv_state_out = window[:, tokens : tokens + HISTORY].reshape(1, HISTORY * batch, channels)

    split = key_heads * key_dim
    q = y[..., :split].reshape(batch, tokens, key_heads, key_dim)
    k = y[..., split : 2 * split].reshape(batch, tokens, key_heads, key_dim)
    v = y[..., 2 * split :].reshape(batch, tokens, value_heads, value_dim)
    beta = _sigmoid(b.reshape(batch, tokens, value_heads))
    u_gate = a.reshape(batch, tokens, value_heads) + dt_bias
    softplus = torch.where(
        u_gate > SOFTPLUS_THRESHOLD, u_gate, torch.log1p(torch.exp(torch.clamp(u_gate, max=SOFTPLUS_THRESHOLD)))
    )
    g = -torch.exp(A_log) * softplus
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_EPS) * key_dim**-0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_EPS)
    q = q.repeat_interleave(ratio, dim=2)
    k = k.repeat_interleave(ratio, dim=2)

    state = recurrent_state.clone()
    outs = []
    for t in range(tokens):
        state = state * torch.exp(g[:, t])[..., None, None]
        read = (state * k[:, t][..., None]).sum(-2)
        delta = beta[:, t][..., None] * (v[:, t] - read)
        state = state + k[:, t][..., None] * delta[..., None, :]
        outs.append((state * q[:, t][..., None]).sum(-2))
    o = torch.stack(outs, dim=1)
    normed = o / torch.sqrt((o * o).mean(-1, keepdim=True) + epsilon) * norm_weight
    gate = z.reshape(batch, tokens, value_heads, value_dim)
    output = (normed * gate * _sigmoid(gate)).reshape(1, rows, value_heads * value_dim)
    return output, conv_state_out, state


def make_inputs(inputs):
    (_, rows, channels), (batch, value_heads, key_dim, value_dim) = inputs
    f32 = torch.float32
    return {
        "mixed_qkv": torch.randn(1, rows, channels, dtype=f32),
        "z": torch.randn(1, rows, value_heads * value_dim, dtype=f32),
        "b": 2.0 * torch.randn(1, rows, value_heads, dtype=f32),
        "a": 2.0 * torch.randn(1, rows, value_heads, dtype=f32),
        "conv_state": torch.randn(1, HISTORY * batch, channels, dtype=f32),
        "recurrent_state": 0.5 * torch.randn(batch, value_heads, key_dim, value_dim, dtype=f32),
        "conv_weight": 0.5 * torch.randn(HISTORY + 1, channels, dtype=f32),
        "A_log": torch.linspace(-5.6, 3.6, value_heads, dtype=f32),
        "dt_bias": torch.linspace(19.3, -9.0, value_heads, dtype=f32),
        "norm_weight": 0.55 + 0.85 * torch.rand(value_dim, dtype=f32),
    }


ORDER = ("mixed_qkv", "z", "b", "a", "conv_state", "recurrent_state", "conv_weight", "A_log", "dt_bias", "norm_weight")


def to_device(tensors, device, *, state_dtype=ttnn.float32, poison=None):
    out = []
    for name in ORDER:
        dtype = state_dtype if name == "recurrent_state" else ttnn.float32
        t = ttnn.from_torch(
            tensors[name], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if poison is not None and name != "recurrent_state":
            t = ttnn.fill_implicit_tile_padding(t, poison)
        out.append(t)
    return out


def compute_config(fp32_dest_acc_en=True):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=False
    )


def assert_close(actual, expected, label):
    pcc_min, rms_max = default_tolerance(ttnn.float32, True)
    actual = actual.to(torch.float32).flatten()
    expected = expected.to(torch.float32).flatten()
    assert torch.isfinite(actual).all(), f"{label}: non-finite values"
    pcc = torch.corrcoef(torch.stack([actual.double(), expected.double()]))[0, 1].item()
    rms = torch.sqrt(((actual - expected).double() ** 2).mean()).item()
    scale = expected.std().item()
    rel_rms = rms / scale if scale > 1e-12 else rms
    assert pcc >= pcc_min, f"{label}: PCC {pcc:.6f} < {pcc_min}"
    assert rel_rms <= rms_max, f"{label}: relative RMS {rel_rms:.6f} > {rms_max}"


def check_call(returned, expected, inputs):
    assert isinstance(returned, (tuple, list)) and len(returned) == 3
    out, cs, rs = returned
    (_, rows, channels), (batch, value_heads, key_dim, value_dim) = inputs
    assert tuple(out.shape) == (1, rows, value_heads * value_dim)
    assert tuple(cs.shape) == (1, HISTORY * batch, channels)
    assert tuple(rs.shape) == (batch, value_heads, key_dim, value_dim)
    assert out.dtype == ttnn.float32 and cs.dtype == ttnn.float32 and rs.dtype == ttnn.float32
    assert out.layout == ttnn.TILE_LAYOUT and cs.layout == ttnn.TILE_LAYOUT and rs.layout == ttnn.TILE_LAYOUT
    assert_close(ttnn.to_torch(out), expected[0], "output")
    assert torch.equal(ttnn.to_torch(cs).to(torch.float32), expected[1]), "conv_state_out not bit-exact"
    rs_torch = ttnn.to_torch(rs).to(torch.float32)
    assert_close(rs_torch, expected[2], "recurrent_state_out")
    for b_idx in range(batch):
        for h_idx in range(value_heads):
            assert_close(rs_torch[b_idx, h_idx], expected[2][b_idx, h_idx], f"state pair ({b_idx}, {h_idx})")
    return out, cs, rs


SHAPES = [
    pytest.param(((1, 1, 256), (1, 3, 32, 64)), id="decode_b1_small"),
    pytest.param(((1, 3, 256), (3, 3, 32, 64)), id="decode_b3_small"),
    pytest.param(((1, 16, 256), (16, 3, 32, 64)), id="decode_b16_small"),
    pytest.param(((1, 40, 256), (40, 3, 32, 64)), id="decode_b40_two_tile_rows"),
    pytest.param(((1, 1, 2560), (1, 12, 128, 128)), id="decode_b1_chip"),
    pytest.param(((1, 8, 2560), (8, 12, 128, 128)), id="decode_b8_chip"),
    pytest.param(((1, 2, 256), (1, 3, 32, 64)), id="prefill_t2"),
    pytest.param(((1, 5, 256), (1, 3, 32, 64)), id="prefill_t5"),
    pytest.param(((1, 32, 256), (1, 3, 32, 64)), id="prefill_t32_aligned"),
    pytest.param(((1, 33, 256), (1, 3, 32, 64)), id="prefill_t33"),
    pytest.param(((1, 100, 256), (1, 3, 32, 64)), id="prefill_t100"),
    pytest.param(((1, 64, 256), (1, 2, 32, 64)), id="prefill_r1"),
    pytest.param(((1, 40, 320), (1, 2, 64, 32)), id="prefill_k_gt_v"),
    pytest.param(((1, 5, 1760), (1, 33, 32, 32)), id="prefill_hv33"),
    pytest.param(((1, 64, 2560), (1, 12, 128, 128)), id="prefill_t64_chip"),
]


@pytest.mark.parametrize("inputs", SHAPES)
def test_gated_delta_net(device, inputs):
    torch.manual_seed(42)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


def test_epsilon_kwarg(device):
    torch.manual_seed(42)
    inputs = ((1, 33, 256), (1, 3, 32, 64))
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER), epsilon=1e-2)
    returned = gated_delta_net(*to_device(tensors, device), epsilon=1e-2)
    check_call(returned, expected, inputs)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 3, 256), (3, 3, 32, 64)), id="decode_b3"),
        pytest.param(((1, 33, 256), (1, 3, 32, 64)), id="prefill_t33"),
    ],
)
@pytest.mark.parametrize("poison", [1e4, float("inf")], ids=["finite", "inf"])
def test_padding_does_not_leak(device, inputs, poison):
    torch.manual_seed(42)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device, poison=poison), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


def test_states_carry_across_calls(device):
    """One prefill call of 33 tokens, then 3 decode calls: both returned states feed the next call."""
    torch.manual_seed(42)
    total = 36
    geometry = (1, 3, 32, 64)
    tensors = make_inputs(((1, total, 256), geometry))
    weights = [tensors[name] for name in ("conv_weight", "A_log", "dt_bias", "norm_weight")]
    torch_cs, torch_rs = tensors["conv_state"], tensors["recurrent_state"]
    dev_cs = dev_rs = None
    position = 0
    for tokens in (33, 1, 1, 1):
        part = {name: tensors[name][:, position : position + tokens] for name in ("mixed_qkv", "z", "b", "a")}
        expected = reference(part["mixed_qkv"], part["z"], part["b"], part["a"], torch_cs, torch_rs, *weights)
        call = {**tensors, **part, "conv_state": torch_cs, "recurrent_state": torch_rs}
        dev = to_device(call, device)
        if dev_cs is not None:
            dev[4], dev[5] = dev_cs, dev_rs
        returned = gated_delta_net(*dev, compute_kernel_config=compute_config())
        _, dev_cs, dev_rs = check_call(returned, expected, ((1, tokens, 256), geometry))
        torch_cs, torch_rs = expected[1], expected[2]
        position += tokens


def test_inputs_not_modified(device):
    torch.manual_seed(42)
    inputs = ((1, 3, 256), (3, 3, 32, 64))
    tensors = make_inputs(inputs)
    dev = to_device(tensors, device)
    gated_delta_net(*dev, compute_kernel_config=compute_config())
    assert torch.equal(ttnn.to_torch(dev[4]).to(torch.float32), tensors["conv_state"])
    assert torch.equal(ttnn.to_torch(dev[5]).to(torch.float32), tensors["recurrent_state"])


def test_refuses_fp32_dest_acc_disabled(device, expect_error):
    torch.manual_seed(42)
    tensors = make_inputs(((1, 1, 256), (1, 3, 32, 64)))
    with expect_error(NotImplementedError, ""):
        gated_delta_net(*to_device(tensors, device), compute_kernel_config=compute_config(False))


def test_rejects_nonpositive_epsilon(device, expect_error):
    torch.manual_seed(42)
    tensors = make_inputs(((1, 1, 256), (1, 3, 32, 64)))
    with expect_error((ValueError, RuntimeError), ""):
        gated_delta_net(*to_device(tensors, device), epsilon=0.0)


def test_rejects_non_float32_state(device, expect_error):
    torch.manual_seed(42)
    tensors = make_inputs(((1, 1, 256), (1, 3, 32, 64)))
    with expect_error((ValueError, RuntimeError), ""):
        gated_delta_net(*to_device(tensors, device, state_dtype=ttnn.bfloat16))


def test_rejects_bad_conv_state_rows(device, expect_error):
    torch.manual_seed(42)
    tensors = make_inputs(((1, 3, 256), (3, 3, 32, 64)))
    tensors["conv_state"] = tensors["conv_state"][:, :6]
    with expect_error((ValueError, RuntimeError), ""):
        gated_delta_net(*to_device(tensors, device))
