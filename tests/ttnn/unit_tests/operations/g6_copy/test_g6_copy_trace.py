# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""BB-16: the model replays g6_copy from a captured trace. A replay must move the source's current
contents into the destination's buffer, and equal the eager call bit for bit."""

import pytest
import torch
import ttnn

from ttnn.operations.g6_copy import g6_copy

pytestmark = pytest.mark.use_module_device({"trace_region_size": 1 << 20})

SHAPES = [(1, 12, 128, 128), (16, 12, 128, 128)]


def _host(t):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)


def _bits(t):
    return t.contiguous().view(torch.int32)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_g6_copy_trace_replay(device, shape):
    torch.manual_seed(0)
    first, second = torch.randn(shape), torch.randn(shape)
    src = _host(first).to(device, ttnn.DRAM_MEMORY_CONFIG)
    dst = _host(torch.zeros(shape)).to(device, ttnn.DRAM_MEMORY_CONFIG)

    # Eager call first: it builds the descriptor and compiles the kernels outside the capture.
    g6_copy(src, dst)
    eager = ttnn.to_torch(dst)
    assert torch.equal(_bits(eager), _bits(first))

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    out = g6_copy(src, dst)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    assert out is dst

    try:
        ttnn.copy_host_to_device_tensor(_host(torch.zeros(shape)), dst)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(_bits(ttnn.to_torch(dst)), _bits(eager)), "replay differs from the eager call"

        ttnn.copy_host_to_device_tensor(_host(second), src)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(_bits(ttnn.to_torch(dst)), _bits(second)), "replay did not copy the new source"
    finally:
        ttnn.release_trace(device, trace_id)
