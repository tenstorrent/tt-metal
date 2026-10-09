# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def test_capture_closes_when_the_body_raises(expect_error):
    with expect_error(ValueError, "boom"):
        with ttnn.jit_telemetry.capture() as cap:
            raise ValueError("boom")
    assert cap.stats == {}


def test_capture_records_a_program_that_fails_the_size_check(expect_error):
    # A ~1 KiB kernel-config buffer: the program fails tt-metal's size check before dispatch.
    device = ttnn.open_device(device_id=0, worker_l1_size=ttnn.get_max_worker_l1_unreserved_size() - 1024)
    try:
        a = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        with ttnn.jit_telemetry.capture() as cap:
            with expect_error(RuntimeError, "too large for kernel config buffer"):
                ttnn.add(a, a)
    finally:
        ttnn.close_device(device)
    assert cap.stats["program_config_size.total.TENSIX"]["max"] > 1024
