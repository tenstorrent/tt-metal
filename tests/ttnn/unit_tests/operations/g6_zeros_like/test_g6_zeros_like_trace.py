# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace capture and replay of g6_zeros_like on the float32 production case (BB-16).

The model replays its calls from captured traces. A replay must rewrite the buffer the captured
call named, every time, and must not allocate or build. Between replays the buffer is refilled
with NaN on the host side of the queue, so a replay that skipped a page leaves NaN behind.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros_like import g6_zeros_like

SHAPE = (1, 12, 128, 128)
TRACE_REGION_SIZE = 200_000


def _to_device(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _host(t):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)


def _assert_positive_zero_bits(out):
    bits = ttnn.to_torch(out).contiguous().view(torch.int32)
    wrong = int((bits != 0).sum().item())
    assert wrong == 0, f"{wrong} of {bits.numel()} elements are not +0.0"


@pytest.mark.use_module_device({"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize("with_optional", [True, False], ids=["into_optional", "new_output"])
def test_trace_replay_fp32(device, with_optional):
    torch.manual_seed(0)
    x = _to_device(
        torch.randint(-(2**31), 2**31, SHAPE, dtype=torch.int64).to(torch.int32).view(torch.float32), device
    )
    nan_host = _host(torch.full(SHAPE, float("nan")))
    o = _to_device(torch.full(SHAPE, float("nan")), device) if with_optional else None

    # Eager call first: compiles the program and fills the program cache, as the model does before capture.
    eager = g6_zeros_like(x, optional_tensor=o)
    _assert_positive_zero_bits(eager)
    if not with_optional:
        ttnn.deallocate(eager)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    out = g6_zeros_like(x, optional_tensor=o)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    if with_optional:
        assert out.buffer_address() == o.buffer_address(), "returned tensor is not optional_tensor"

    try:
        for _ in range(2):
            ttnn.copy_host_to_device_tensor(nan_host, out)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            _assert_positive_zero_bits(out)
    finally:
        ttnn.release_trace(device, trace_id)
