# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 9 — cross-core all-reduce transport + combine + K re-tune.

Covers, WITHOUT a profiler build (correctness + host-side routing assertions; the
device-time bake-off lives in test_rms_norm_perf.py):

  1. The root-relay gather-then-broadcast transport (transport_mode==1) is the production
     default for Regime B, and a Regime-B descriptor carries the new PRODUCED semaphore.
  2. BOTH transports (baseline mcast all-gather, mode 0; root-relay, mode 1) are
     numerically correct vs torch on wide-W Regime-B shapes — all-ones is exact and random
     PCC is at the bf16 band. This guards the single-reduce combine (Part B) and the new
     transport leg (Part A) together: every core must end with the FULLY-combined global
     Σx² regardless of how the K partials were gathered.
  3. The re-tuned _select_k proxy (Part C) picks the measured-optimal K under root-relay.
"""

import pytest
import torch

import ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def _torch_ref(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


# Wide-W shapes that route through Regime B (the cross-core all-reduce path).
_REGIME_B_SHAPES = [(1, 1, 32, 4096), (1, 1, 32, 8192), (1, 1, 64, 8192), (1, 1, 32, 16384), (1, 1, 64, 12288)]


@pytest.fixture
def _restore_force_hooks():
    """Save/restore the measurement-only module hooks the tests poke."""
    saved = (desc._FORCE_REGIME, desc._FORCE_TRANSPORT, desc._FORCE_K)
    yield
    desc._FORCE_REGIME, desc._FORCE_TRANSPORT, desc._FORCE_K = saved


def test_root_relay_is_production_default(device, _restore_force_hooks):
    """Regime B defaults to the root-relay transport and carries the PRODUCED semaphore."""
    # transport selection is a pure host decision — assert it directly.
    assert desc._select_transport(16) == desc.TRANSPORT_ROOT_RELAY
    assert desc._FORCE_TRANSPORT is None  # not pinned in the committed module

    # A built Regime-B descriptor must allocate DATA_READY + CONSUMED + PRODUCED (3 sems).
    desc._FORCE_REGIME = "B"
    shape = (1, 1, 32, 8192)
    ti = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    prog, _ = desc.create_program_descriptor(ti, out_t, None, 1e-6, None)
    assert len(prog.semaphores) == 3, f"Regime B should carry DATA_READY+CONSUMED+PRODUCED, got {len(prog.semaphores)}"


@pytest.mark.parametrize("mode", [desc.TRANSPORT_MCAST_ALLGATHER, desc.TRANSPORT_ROOT_RELAY],
                         ids=["mcast_allgather", "root_relay"])
@pytest.mark.parametrize("shape", _REGIME_B_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("gamma_on", [False, True], ids=["no_gamma", "gamma"])
def test_transport_correctness(device, _restore_force_hooks, mode, shape, gamma_on):
    """Both transports produce the correct global Σx² (all-ones exact + random PCC)."""
    desc._FORCE_REGIME = "B"
    desc._FORCE_TRANSPORT = mode
    W = shape[-1]
    g = None
    if gamma_on:
        gt = torch.randn(W)
        g = ttnn.from_torch(gt.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # all-ones: RMS(ones)*gamma == gamma (or 1.0 with no gamma) exactly.
    xo = torch.ones(*shape)
    to = ttnn.from_torch(xo.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    oo = ttnn.to_torch(rms_norm(to, gamma=g)).float()
    expect_ones = (gt.float().expand_as(oo[..., :W]) if gamma_on else torch.ones_like(oo))
    assert (oo - expect_ones).abs().max().item() < 0.1, "all-ones not exact"

    # random vs torch
    x = torch.randn(*shape)
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(ti, gamma=g)).float()
    ref = _torch_ref(x)
    if gamma_on:
        ref = ref * gt.float()
    p = _pcc(out, ref)
    assert p >= 0.99, f"mode={mode} shape={shape} gamma={gamma_on} PCC={p:.5f} below band"


def test_k_retune_picks_measured_optima():
    """Part C: the re-tuned proxy picks the K measured fastest under root-relay."""
    class _Grid:
        x = 8
        y = 8

    grid = _Grid()
    total_cores = 64
    # (Wt, num_row_groups, expected_K) — every entry was device-measured optimal (changelog R9).
    cases = [
        (64, 1, 16),    # 2048
        (128, 1, 16),   # 4096
        (256, 1, 32),   # 8192
        (384, 2, 24),   # 12288 — K=24 (3 core-rows/group) beats K=16/K=32
        (512, 1, 32),   # 16384
    ]
    for Wt, rg, expected in cases:
        K = desc._select_k(Wt, rg, grid, total_cores, False, ttnn.bfloat16, True, ttnn.bfloat16)
        assert K == expected, f"_select_k(Wt={Wt}, rg={rg}) = {K}, expected {expected}"
