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
