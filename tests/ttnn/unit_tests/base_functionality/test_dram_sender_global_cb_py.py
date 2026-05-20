# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import ttnn


def _is_blackhole(device) -> bool:
    arch = getattr(device, "arch", lambda: None)()
    if arch is None:
        return True
    return "BLACKHOLE" in str(arch).upper()


def _dram_programmable_enabled() -> bool:
    return os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") == "1"


def _single_recv(x: int, y: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))})


def test_create_dram_sender_global_circular_buffer_single_bank(device):
    if not _is_blackhole(device):
        pytest.skip("DramSenderGCB requires Blackhole")
    if not _dram_programmable_enabled():
        pytest.skip("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set")

    bank_to_receivers = [(0, _single_recv(0, 0))]
    gcb = ttnn.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, 1024)
    assert gcb.size() == 1024
    assert gcb.sender_cores().num_cores() == 1
    assert gcb.receiver_cores().num_cores() == 1


def test_create_dram_sender_global_circular_buffer_multi_bank_disjoint(device):
    if not _is_blackhole(device):
        pytest.skip("DramSenderGCB requires Blackhole")
    if not _dram_programmable_enabled():
        pytest.skip("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set")

    # Each bank gets a disjoint single-core receiver set.
    bank_to_receivers = [(b, _single_recv(b, 0)) for b in range(4)]
    gcb = ttnn.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, 2048)
    assert gcb.size() == 2048
    assert gcb.sender_cores().num_cores() == 4
    assert gcb.receiver_cores().num_cores() == 4
