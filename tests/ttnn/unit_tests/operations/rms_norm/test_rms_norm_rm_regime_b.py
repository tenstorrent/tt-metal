# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Refinement 4 — ROW_MAJOR routed through the cross-core mcast all-gather (Regime B RM).
#
# Before R4, the ROW_MAJOR path was row-parallel only (_regime_rm_descriptor): a
# wide-W RM row that exceeds one core's L1 budget had no Regime-B fallback and
# under-parallelized (often a single core). After unifying the kernels (one reader/
# compute/writer gated by layout_is_rm x num_partials), the RM legs route through
# the SAME mcast all-gather as TILE Regime B.
#
# These tests (a) PROVE a wide-W ROW_MAJOR row routes through Regime B — by building
# the program descriptor and asserting it carries the two mcast semaphores — and
# (b) pin the numerical correctness of that path against the torch reference.
#
# DO NOT DELETE — documents the Refinement 4 RM-through-mcast contract.

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


# Shapes the host heuristic routes to Regime B RM (few 32-stick blocks, wide W).
# (aligned and W-non-aligned to exercise the shard padding in the gather path).
RM_REGIME_B_SHAPES = [
    (1, 1, 32, 4096),  # K=64, Wt_s=2
    (1, 1, 32, 8192),  # K=64, Wt_s=4
    (1, 1, 64, 8192),  # 2 block-groups x K=32
    (1, 32, 8192),  # 3D, 1 block-group, wide
    (1, 1, 32, 8190),  # W non-aligned (last shard padded)
]


def _torch_rms_norm(x, gamma=None, eps=1e-6):
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    return out * gamma.to(torch.float32).reshape(-1) if gamma is not None else out


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _rel_rms(a, b):
    return (a.float() - b.float()).pow(2).mean().sqrt().item() / (b.float().pow(2).mean().sqrt().item() or 1.0)


_TOL = {ttnn.bfloat16: (0.999, 0.04), ttnn.float32: (0.9999, 0.02)}


def _routes_through_regime_b(device, shape, dtype, gamma):
    """Build the descriptor and confirm it is Regime B RM (carries the mcast semaphores).

    Regime B allocates DATA_READY + CONSUMED (the mcast handshake) and, since
    Refinement 9 (Part A), a third PRODUCED counter for the root-relay transport — so a
    Regime-B descriptor has >= 2 semaphores while Regime A (row-parallel) has none."""
    ti = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_t = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    prog, _ = pd.create_program_descriptor(ti, out_t, gamma, 1e-6, None)
    return len(prog.semaphores) >= 2


@pytest.mark.parametrize("dtype", [pytest.param(ttnn.bfloat16, id="bf16"), pytest.param(ttnn.float32, id="fp32")])
@pytest.mark.parametrize("shape", RM_REGIME_B_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rm_regime_b_routing(device, shape, dtype):
    """A wide-W ROW_MAJOR row must route through the Regime B mcast all-gather."""
    assert _routes_through_regime_b(
        device, shape, dtype, None
    ), f"shape {shape} ({dtype}) did NOT route through Regime B RM (no mcast semaphores)"


@pytest.mark.parametrize("dtype", [pytest.param(ttnn.bfloat16, id="bf16"), pytest.param(ttnn.float32, id="fp32")])
@pytest.mark.parametrize(
    "gamma_mode",
    [
        pytest.param("no_gamma", id="no_gamma"),
        pytest.param("gamma_tile", id="gamma_tile"),
        pytest.param("gamma_rm", id="gamma_rm"),
    ],
)
@pytest.mark.parametrize("shape", RM_REGIME_B_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rm_regime_b_correctness(device, shape, dtype, gamma_mode):
    """Numerical correctness of the RM-through-mcast path vs the torch reference."""
    torch.manual_seed(0)
    tdt = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(shape, dtype=torch.float32).to(tdt)

    gamma = torch_gamma = None
    if gamma_mode != "no_gamma":
        W = shape[-1]
        torch_gamma = torch.randn(W, dtype=torch.float32).to(tdt)
        glayout = ttnn.TILE_LAYOUT if gamma_mode == "gamma_tile" else ttnn.ROW_MAJOR_LAYOUT
        gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=glayout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Confirm this shape actually exercises Regime B (else the test is vacuous).
    assert _routes_through_regime_b(device, shape, dtype, gamma), f"{shape} not Regime B RM"

    ti = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_out = rms_norm(ti, gamma=gamma)
    assert ttnn_out.layout == ttnn.ROW_MAJOR_LAYOUT  # output layout matches input
    out = ttnn.to_torch(ttnn_out)
    assert tuple(out.shape) == tuple(shape)

    expected = _torch_rms_norm(torch_input, gamma=torch_gamma)
    pcc_min, rms_max = _TOL[dtype]
    pcc = _pcc(out, expected)
    relrms = _rel_rms(out, expected)
    assert pcc >= pcc_min, f"PCC {pcc:.6f} < {pcc_min} (shape={shape}, {dtype}, {gamma_mode})"
    assert relrms <= rms_max, f"relRMS {relrms:.5f} > {rms_max} (shape={shape}, {dtype}, {gamma_mode})"
