# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profiling driver for g6_zeros_like on the production cases (run with run_safe_pytest.sh --profile).

Correctness lives in test_g6_zeros_like.py; this file only issues the calls the profiler measures.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros_like import g6_zeros_like
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd


@pytest.mark.parametrize(
    "shape,dtype,torch_dtype",
    [
        pytest.param((1, 12, 128, 128), ttnn.float32, torch.float32, id="float32"),
        pytest.param((5, 1, 32, 256), ttnn.bfloat8_b, torch.bfloat16, id="bfloat8_b"),
    ],
)
def test_profile(device, shape, dtype, torch_dtype):
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for _ in range(3):
        if dtype == ttnn.float32:
            out = g6_zeros_like(x)
        else:
            # bfloat8_b is outside Phase 0 SUPPORTED; dispatch the same program directly.
            out = ttnn.allocate_tensor_on_device(x.spec, device)
            ttnn.generic_op([out], pd.get_program_descriptor(out))
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(device)


@pytest.mark.parametrize("row_wise", [False, True], ids=["col", "row"])
@pytest.mark.parametrize("num_cores_cap", [None, 96, 64, 48, 32, 24, 16, 8], ids=lambda c: f"cap{c}")
def test_profile_core_knobs_fp32(device, monkeypatch, num_cores_cap, row_wise):
    """Grid-synchronization and placement lamps: same kernel, host knobs only."""
    monkeypatch.setattr(pd, "NUM_CORES_CAP", num_cores_cap)
    monkeypatch.setattr(pd, "ROW_WISE", row_wise)
    shape = (1, 12, 128, 128)
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for _ in range(3):
        out = g6_zeros_like(x)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(device)


@pytest.mark.parametrize("noc", [ttnn.NOC.NOC_1, ttnn.NOC.NOC_0], ids=["noc1", "noc0"])
def test_profile_writer_noc_fp32(device, monkeypatch, noc):
    """Placement lamp: the same writer on either NoC."""
    monkeypatch.setattr(pd, "WRITER_NOC", noc)
    shape = (1, 12, 128, 128)
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for _ in range(5):
        out = g6_zeros_like(x)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(device)
