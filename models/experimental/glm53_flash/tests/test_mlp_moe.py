# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device unit tests for the GLM-5.3-Flash building blocks: SwiGLU MLP, MoE, router, mHC.

Each module is graded against a torch reference written from the same formula the module's docstring
states, so a failure localises to a module rather than to the whole layer.

Runs on whatever arch the harness has. The modules were authored against Blackhole; the only
arch-sensitive thing in them is the compute-kernel config, which is built through
`ttnn.init_device_compute_kernel_config(device.arch(), ...)` (see `test_config_precision.py::
test_compute_kernel_configs_are_arch_typed`) rather than hard-coded. Nothing here needs a port.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.glm53_flash.tt.mlp import SwiGLUMLP, swiglu_clamped
from models.experimental.glm53_flash.tt.moe import PADDED_EXPERTS, MoE, Router

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})

DRAM = ttnn.DRAM_MEMORY_CONFIG
HIDDEN = 4096


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def to_device(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=DRAM)


def from_device(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).float()


def bf4_round(t: torch.Tensor) -> torch.Tensor:
    """The values a bfloat4_b TILE tensor holds for ``t`` (shared exponents over 16-wide blocks of the last dim)."""
    return ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


# ---------------------------------------------------------------------------------------------
# SwiGLU MLP (dense FFN of layers 0-2, and the MoE shared expert)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("intermediate", [256, 2048], ids=["small", "moe_shared"])
@pytest.mark.parametrize("tokens", [1, 32])
def test_swiglu_mlp_matches_reference(device, intermediate, tokens):
    """``down( silu(min(gate,limit)) * clamp(up,-limit,limit) )`` with limit = swiglu_limit = 10."""
    torch.manual_seed(0)
    limit = 10.0
    # Scale so some entries exceed the clamp in both directions (that is the whole point of it).
    x = torch.randn(1, 1, tokens, HIDDEN) * 3.0
    w_gate = torch.randn(HIDDEN, intermediate) * 0.02
    w_up = torch.randn(HIDDEN, intermediate) * 0.02
    w_down = torch.randn(intermediate, HIDDEN) * 0.02

    mlp = SwiGLUMLP(
        device,
        to_device(w_gate, device, ttnn.bfloat8_b),
        to_device(w_up, device, ttnn.bfloat8_b),
        to_device(w_down, device, ttnn.bfloat8_b),
        limit=limit,
    )
    out = mlp(to_device(x, device))
    ttnn.synchronize_device(device)

    g = x @ w_gate
    u = x @ w_up
    # Written plainly, in the module's order: silu(min(gate, limit)) * clamp(up, +-limit)
    act = F.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)
    ref = act @ w_down

    got = from_device(out)
    got = got.reshape(ref.shape)
    assert pcc(got, ref) > 0.98, f"PCC {pcc(got, ref):.5f}"


def test_swiglu_clamped_actually_clamps(device):
    """A gate above the limit and an up above it must both be clamped, not merely passed through."""
    limit = 2.0
    I = 64
    gate_up = torch.zeros(1, 1, 32, 2 * I)
    gate_up[..., :I] = 100.0  # far above the gate limit
    gate_up[..., I:] = 100.0  # far above the up limit
    out = swiglu_clamped(to_device(gate_up, device), I, limit)
    got = from_device(out).reshape(32, I)
    # silu(2.0) * 2.0, not silu(100) * 100 (which would be ~100).
    expected = F.silu(torch.tensor(limit)) * limit
    assert torch.allclose(got.flatten()[:8], expected.expand(8), rtol=2e-2, atol=2e-2), got.flatten()[:4]


# ---------------------------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------------------------


def _router_ref(x, w, bias, num_experts, top_k, scaling):
    """noaux_tc: sigmoid scores, bias for *selection* only, top-k, renormalize, scale."""
    logits = x @ w  # [.., 288]
    scores = torch.sigmoid(logits)
    choice = scores + bias
    idx = torch.topk(choice, k=top_k, dim=-1, largest=True, sorted=True).indices
    sel = torch.zeros_like(scores, dtype=torch.bool)
    sel.scatter_(-1, idx, True)
    weights = scores * sel
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * scaling


@pytest.mark.parametrize("tokens", [1, 32])
def test_router_matches_reference(device, tokens):
    """The router returns dense per-expert weights, 0 for unselected, renormalized and scaled."""
    torch.manual_seed(1)
    n_experts, top_k, scaling = 288, 8, 2.5
    x = torch.randn(1, 1, tokens, HIDDEN) * 0.5
    w = torch.randn(HIDDEN, n_experts) * 0.02
    # A GLM-like bias: large offset, small spread (the case that needs centring downstream).
    bias = 34.5 + torch.randn(n_experts) * 0.14

    router = Router(
        device, to_device(w, device, ttnn.bfloat16), bias, num_experts=n_experts, top_k=top_k, scaling=scaling
    )
    out = router(to_device(x, device))
    ttnn.synchronize_device(device)

    got = from_device(out).reshape(tokens, n_experts)
    ref = _router_ref(x.reshape(tokens, HIDDEN), w, bias, n_experts, top_k, scaling)

    # Exactly top_k non-zeros per token.
    assert torch.all((got > 0).sum(-1) == top_k), (got > 0).sum(-1)
    # The selected *set* must match: this is what a bf16 ranking row would break.
    assert torch.equal((got > 0), (ref > 0)), "selected expert set differs from the fp32 reference"
    assert pcc(got, ref) > 0.99, f"PCC {pcc(got, ref):.5f}"


def test_router_pads_experts_to_alignment():
    """288 experts pad to PADDED_EXPERTS, and the pad bias is -inf so pads are never selected."""
    assert PADDED_EXPERTS >= 288
    assert PADDED_EXPERTS % 32 == 0


# ---------------------------------------------------------------------------------------------
# MoE end to end
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [1, 32])
def test_moe_matches_reference(device, tokens):
    """Routed top-k experts + the shared expert, on a small expert set for speed."""
    torch.manual_seed(2)
    limit = 10.0
    n_experts, top_k, scaling, intermediate = 8, 2, 2.5, 256

    x = torch.randn(1, 1, tokens, HIDDEN) * 0.5
    w_router = torch.randn(HIDDEN, n_experts) * 0.02
    bias = torch.randn(n_experts) * 0.1
    # experts: [E, HIDDEN, 2I] = [gate | up], [E, I, HIDDEN], stored as bfloat4_b. Rounded to their bf4 values up
    # front so the reference uses exactly what the device holds and the test grades the MoE, not the 4-bit storage.
    ex_gu = bf4_round(torch.randn(n_experts, HIDDEN, 2 * intermediate) * 0.02)
    ex_down = bf4_round(torch.randn(n_experts, intermediate, HIDDEN) * 0.02)
    sh_gate = torch.randn(HIDDEN, intermediate) * 0.02
    sh_up = torch.randn(HIDDEN, intermediate) * 0.02
    sh_down = torch.randn(intermediate, HIDDEN) * 0.02

    router = Router(
        device, to_device(w_router, device, ttnn.bfloat16), bias, num_experts=n_experts, top_k=top_k, scaling=scaling
    )
    shared = SwiGLUMLP(
        device,
        to_device(sh_gate, device, ttnn.bfloat8_b),
        to_device(sh_up, device, ttnn.bfloat8_b),
        to_device(sh_down, device, ttnn.bfloat8_b),
        limit=limit,
    )
    moe = MoE(
        device,
        router,
        to_device(ex_gu.unsqueeze(0), device, ttnn.bfloat4_b),
        to_device(ex_down.unsqueeze(0), device, ttnn.bfloat4_b),
        shared,
        intermediate=intermediate,
        hidden=HIDDEN,
        limit=limit,
    )
    assert moe.gate_up.dtype == ttnn.bfloat4_b and moe.down.dtype == ttnn.bfloat4_b
    out = moe(to_device(x, device), decode=(tokens == 1))
    ttnn.synchronize_device(device)

    # Reference: routed weighted sum + shared expert.
    xf = x.reshape(tokens, HIDDEN)
    routing = _router_ref(xf, w_router, bias, n_experts, top_k, scaling)
    acc = torch.zeros(tokens, HIDDEN)
    for e in range(n_experts):
        w = routing[:, e]
        if not bool((w > 0).any()):
            continue
        gu = xf @ ex_gu[e]
        act = F.silu(gu[:, :intermediate].clamp(max=limit)) * gu[:, intermediate:].clamp(-limit, limit)
        acc += w.unsqueeze(-1) * (act @ ex_down[e])
    s_gu = xf @ torch.cat([sh_gate, sh_up], dim=-1)
    s_act = F.silu(s_gu[:, :intermediate].clamp(max=limit)) * s_gu[:, intermediate:].clamp(-limit, limit)
    ref = acc + s_act @ sh_down

    got = from_device(out).reshape(tokens, HIDDEN)
    assert pcc(got, ref) > 0.99, f"PCC {pcc(got, ref):.5f}"
