# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe shapes for mhc_post (run with run_safe_pytest.sh --profile). Correctness is the
acceptance test's job; this only dispatches each shape once for the device profiler."""
import pytest
import torch
import ttnn

from ttnn.operations.mhc_post import mhc_post

N = 4


def _dev(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("T,C", [(160, 5120), (640, 1792), (640, 7168), (1, 7168), (4096, 1792)])
def test_perf_plain_fp32(T, C, device):
    torch.manual_seed(0)
    f = torch.randn(1, 1, T, C)
    x = torch.randn(1, 1, T, N * C)
    post = torch.rand(1, 1, T, N)
    comb = torch.rand(1, 1, T, N * N)
    out = mhc_post(_dev(f, device), _dev(x, device), _dev(post, device), _dev(comb, device))
    ttnn.synchronize_device(device)
    assert list(out.shape) == [1, 1, T, N * C]


@pytest.mark.parametrize("mode", ["plain", "stats"])
@pytest.mark.parametrize("fidelity", [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4], ids=["HiFi2", "HiFi4"])
@pytest.mark.parametrize("T,C", [(160, 5120), (640, 5120)])
def test_perf_bf16_packed(T, C, fidelity, mode, device):
    """DeepSeek-V4.1 call: bf16 streams + sublayer, packed mixes [pre | post | comb]."""
    torch.manual_seed(0)
    bf = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mixes = _dev(torch.rand(1, 1, T, N * (N + 2)), device)
    kw = dict(post_col=N, comb_col=2 * N)
    if mode == "stats":
        kw["stats_weight"] = _dev(torch.randn(1, 1, N * C, 32) * 0.01, device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False)
    out = mhc_post(
        bf(torch.randn(1, 1, T, C)), bf(torch.randn(1, 1, T, N * C)), mixes, mixes, compute_kernel_config=cfg, **kw
    )
    ttnn.synchronize_device(device)
    assert out is not None
