# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for `onorm` — the KDA s6 gated-RMSNorm output tail.

IMMUTABLE SPEC. The implementer must not modify this file; it defines what
"correct" means for the op.

    out = flatten_heads( RMSNorm_over_V(o) * weight ) * sigmoid(gate)

`o` is head-major [B, T, HV, V] (heads on the tiled row axis, V the reduction
axis); the output is flat token-major [B, T, HV*V] with feature = head*V + chan.
HV = 32 value-heads, V = 128 head_dim, so the flat width is 4096. T is always a
multiple of 32. Everything is bfloat16 / TILE.
"""

import pytest
import torch

import ttnn
from ttnn.operations.onorm import onorm

from tests.ttnn.utils_for_testing import assert_with_pcc


# Fixed KDA s6 head geometry (TP=1) — see eval/golden_tests/onorm/feature_spec.py.
HV = 32
V = 128
FLAT = HV * V  # 4096

# Same thresholds as the golden suite — keyed by dtype only.
PCC = {
    torch.float32: 0.999,
    torch.bfloat16: 0.995,
}


def torch_onorm(o, gate, weight, epsilon):
    """Reference: RMSNorm over V per head -> * weight -> flatten -> * sigmoid(gate)."""
    B, T, hv, v = o.shape
    o_f32 = o.to(torch.float32)
    ms = o_f32.pow(2).mean(dim=-1, keepdim=True)  # mean over V, per (b, t, h)
    normed = o_f32 * torch.rsqrt(ms + epsilon)
    normed = normed * weight.to(torch.float32).reshape(1, 1, 1, v)
    flat = normed.reshape(B, T, hv * v)  # head-major -> flat (feature = head*V + chan)
    return flat * torch.sigmoid(gate.to(torch.float32))


def _run_onorm(device, batch, tokens, epsilon, pass_epsilon, compute_kernel_config):
    torch.manual_seed(42)

    torch_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    torch_gate = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    torch_weight = torch.randn(1, 1, 1, V, dtype=torch.bfloat16)

    tt_o = ttnn.from_torch(torch_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gate = ttnn.from_torch(torch_gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {}
    if pass_epsilon:
        kwargs["epsilon"] = epsilon
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    tt_out = onorm(tt_o, tt_gate, tt_weight, **kwargs)

    assert list(tt_out.shape) == [batch, tokens, FLAT]
    assert tt_out.dtype == ttnn.bfloat16
    assert tt_out.layout == ttnn.TILE_LAYOUT

    expected = torch_onorm(torch_o, torch_gate, torch_weight, epsilon)
    actual = ttnn.to_torch(tt_out).to(torch.float32)

    assert_with_pcc(expected, actual, PCC[torch.bfloat16])


@pytest.mark.parametrize(
    "batch, tokens",
    [
        (1, 32),  # single token tile-row (smallest real block)
        (1, 64),  # multi tile-row
        (1, 128),  # 4 tile-rows
        (1, 640),  # bringup profiling length, 20 tile-rows (non-square, > grid-ish)
        (2, 64),  # multi-batch
        (3, 96),  # multi-batch x multi tile-row
    ],
)
def test_onorm(device, batch, tokens):
    """Default epsilon, default compute config: onorm(o, gate, weight)."""
    _run_onorm(device, batch, tokens, 1e-5, pass_epsilon=False, compute_kernel_config=None)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6, 1e-2])
def test_onorm_epsilon(device, epsilon):
    """Explicit epsilon override: onorm(o, gate, weight, epsilon=...)."""
    _run_onorm(device, 1, 64, epsilon, pass_epsilon=True, compute_kernel_config=None)


def test_onorm_compute_kernel_config(device):
    """Explicit compute config override: onorm(o, gate, weight, compute_kernel_config=cfg)."""
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    _run_onorm(device, 1, 64, 1e-5, pass_epsilon=False, compute_kernel_config=cfg)


def test_onorm_default_compute_kernel_config_is_exported(device):
    """The None-resolution path must go through a single exported factory."""
    from ttnn.operations.onorm import default_compute_kernel_config

    cfg = default_compute_kernel_config()
    assert cfg is not None
    _run_onorm(device, 1, 32, 1e-5, pass_epsilon=False, compute_kernel_config=cfg)
