# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-only ablation driver (run with run_safe_pytest.sh --profile): the shipped plan with a writer that
issues no NoC payload, i.e. the launch / setup / barrier floor of the op. Output is wrong by design."""

from pathlib import Path

import torch
import ttnn

from ttnn.operations.g6_zeros_like import g6_zeros_like
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd

ABLATED_WRITER = str(Path(__file__).parent / "probes" / "ablate_no_writes_writer.cpp")


def test_profile_no_payload_fp32(device, monkeypatch):
    monkeypatch.setattr(pd, "WRITER_KERNEL", ABLATED_WRITER)
    monkeypatch.setattr(pd, "_PROGRAM_CACHE", {})
    x = ttnn.from_torch(
        torch.zeros((1, 12, 128, 128), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for _ in range(12):
        out = g6_zeros_like(x)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(device)
