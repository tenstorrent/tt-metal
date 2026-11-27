# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profiling driver for g6_zeros_like on the production cases (run with run_safe_pytest.sh --profile).

Correctness lives in test_g6_zeros_like.py; this file only issues the calls the profiler measures.
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros_like import g6_zeros_like
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd

PROFILE_CALLS = 12


@pytest.mark.parametrize("with_optional_tensor", [False, True], ids=["new_output", "into_optional"])
@pytest.mark.parametrize(
    "shape,dtype,torch_dtype",
    [
        pytest.param((1, 12, 128, 128), ttnn.float32, torch.float32, id="float32"),
        pytest.param((5, 1, 32, 256), ttnn.bfloat8_b, torch.bfloat16, id="bfloat8_b"),
    ],
)
def test_profile(device, shape, dtype, torch_dtype, with_optional_tensor):
    """Production calls at the shipped knob values; PROFILE_CALLS samples per case."""
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    target = ttnn.allocate_tensor_on_device(x.spec, device) if with_optional_tensor else None
    for _ in range(PROFILE_CALLS):
        out = g6_zeros_like(x, optional_tensor=target)
        ttnn.synchronize_device(device)
        if target is None:
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


@pytest.mark.parametrize("row_wise", [False, True], ids=["col", "row"])
@pytest.mark.parametrize("num_cores_cap", [None, 20, 8], ids=lambda c: f"cap{c}")
def test_profile_core_knobs_bf8(device, monkeypatch, num_cores_cap, row_wise):
    """Grid-synchronization and placement lamps on the bfloat8_b production case (P = 40)."""
    monkeypatch.setattr(pd, "NUM_CORES_CAP", num_cores_cap)
    monkeypatch.setattr(pd, "ROW_WISE", row_wise)
    shape = (5, 1, 32, 256)
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for _ in range(5):
        out = g6_zeros_like(x)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(device)


SWEEP_CALLS = PROFILE_CALLS
SWEEP_ORDER_FILE = "generated/g6_zeros_like_sweep_order.txt"


def _sweep(device, request, shape, dtype, torch_dtype):
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for _ in range(SWEEP_CALLS):
        out = g6_zeros_like(x)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(device)
    # Profiler rows come out in call order; this file maps each block of SWEEP_CALLS rows to its config.
    os.makedirs(os.path.dirname(SWEEP_ORDER_FILE), exist_ok=True)
    with open(SWEEP_ORDER_FILE, "a") as f:
        f.write(f"{request.node.callspec.id} {SWEEP_CALLS}\n")


@pytest.mark.parametrize("scratch_pages", [1, 4], ids=lambda s: f"scratch{s}")
@pytest.mark.parametrize("row_wise", [False, True], ids=["col", "row"])
@pytest.mark.parametrize("num_cores_cap", [None, 90, 80, 70, 64, 16], ids=lambda c: f"cap{c}")
@pytest.mark.parametrize(
    "regime,assignment",
    [
        ("page_split_local_zero", "bank_major"),
        ("bank_run_coalesced", "bank_major"),
        ("bank_run_coalesced", "stagger"),
        ("bank_run_coalesced", "nearest"),
        ("bank_run_coalesced", "segment_rows"),
    ],
    ids=["page_split", "bank_major", "stagger", "nearest", "segment_rows"],
)
def test_profile_write_stream_fp32(
    device, monkeypatch, request, regime, assignment, num_cores_cap, row_wise, scratch_pages
):
    """Write-stream lamps on the perf-flagged shape: regime, bank assignment, core count, placement, transfer size."""
    monkeypatch.setattr(pd, "REGIME", regime)
    monkeypatch.setattr(pd, "BANK_ASSIGNMENT", assignment)
    monkeypatch.setattr(pd, "NUM_CORES_CAP", num_cores_cap)
    monkeypatch.setattr(pd, "ROW_WISE", row_wise)
    monkeypatch.setattr(pd, "ZERO_SCRATCH_PAGES", scratch_pages)
    _sweep(device, request, (1, 12, 128, 128), ttnn.float32, torch.float32)


@pytest.mark.parametrize("row_wise", [False, True], ids=["col", "row"])
@pytest.mark.parametrize("num_cores_cap", [None, 20, 8], ids=lambda c: f"cap{c}")
@pytest.mark.parametrize(
    "regime,assignment",
    [
        ("page_split_local_zero", "segment_rows"),
        ("bank_run_coalesced", "bank_major"),
        ("bank_run_coalesced", "segment_rows"),
    ],
    ids=["page_split", "bank_major", "segment_rows"],
)
def test_profile_write_stream_bf8(device, monkeypatch, request, regime, assignment, num_cores_cap, row_wise):
    """Guard for the write-stream knobs on the bfloat8_b production case (P = 40 < grid)."""
    monkeypatch.setattr(pd, "REGIME", regime)
    monkeypatch.setattr(pd, "BANK_ASSIGNMENT", assignment)
    monkeypatch.setattr(pd, "NUM_CORES_CAP", num_cores_cap)
    monkeypatch.setattr(pd, "ROW_WISE", row_wise)
    _sweep(device, request, (5, 1, 32, 256), ttnn.bfloat8_b, torch.bfloat16)
