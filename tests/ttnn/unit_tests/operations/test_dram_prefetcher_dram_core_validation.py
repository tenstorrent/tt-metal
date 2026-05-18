# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Validation tests for ttnn.dram_prefetcher's new DRAM-core mode (run_on_dram_cores=True).

These exercise only the host-side argument validation; no kernels run.
"""

import os
import pytest
import torch
import ttnn


def _is_blackhole(device) -> bool:
    arch = getattr(device, "arch", lambda: None)()
    if arch is None:
        return True
    return "BLACKHOLE" in str(arch).upper()


def _dram_programmable_enabled() -> bool:
    return os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") == "1"


def _make_dummy_tensor(device) -> ttnn.Tensor:
    # Minimal DRAM tensor; the op validation only checks structural things, so this is enough.
    t = torch.zeros(32, 32, dtype=torch.bfloat16)
    return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)


def test_run_on_dram_cores_requires_dram_sender_global_cb(device):
    if not _is_blackhole(device):
        pytest.skip("DRAM-core mode requires Blackhole")
    if not _dram_programmable_enabled():
        pytest.skip("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set")

    tensor = _make_dummy_tensor(device)
    addrs = _make_dummy_tensor(device)
    with pytest.raises(RuntimeError, match="run_on_dram_cores=true requires a dram_sender_global_cb"):
        ttnn.dram_prefetcher(
            [tensor, addrs],
            num_layers=1,
            global_cb=None,
            run_on_dram_cores=True,
            dram_sender_global_cb=None,
        )


def test_run_on_dram_cores_rejects_global_cb(device):
    if not _is_blackhole(device):
        pytest.skip("DRAM-core mode requires Blackhole")
    if not _dram_programmable_enabled():
        pytest.skip("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set")

    tensor = _make_dummy_tensor(device)
    addrs = _make_dummy_tensor(device)

    # Build both GCBs.
    sender = ttnn.CoreCoord(0, 0)
    receivers = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1))})
    global_cb = ttnn.create_global_circular_buffer(device, [(sender, receivers)], 1024)

    dram_recv = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    dram_gcb = ttnn.create_dram_sender_global_circular_buffer(device, [(0, dram_recv)], 1024)

    with pytest.raises(RuntimeError, match="incompatible with global_cb"):
        ttnn.dram_prefetcher(
            [tensor, addrs],
            num_layers=1,
            global_cb=global_cb,
            run_on_dram_cores=True,
            dram_sender_global_cb=dram_gcb,
        )
