# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qualify the cache fill on all four QB2 replicas, including poisoned replay."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tt.cache_initialization import allocate_cache_zero
from models.demos.qwen38_27b_qb2.tt.decoder_tp import validate_qb2_mesh


def assert_replicated_value(tensor, value):
    shards = ttnn.get_device_tensors(tensor)
    assert len(shards) == 4
    for shard in shards:
        actual = ttnn.to_torch(shard)
        assert torch.all(actual == value)
        if value == 0:
            assert not torch.signbit(actual).any()


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1048576}], indirect=True)
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((8192, 1, 32, 256), ttnn.bfloat8_b),
        ((8192, 1, 32, 256), ttnn.bfloat16),
        ((1, 12, 128, 128), ttnn.float32),
        ((2, 12, 128, 128), ttnn.float32),
    ],
)
def test_device_cache_zero(mesh_device, monkeypatch, shape, dtype):
    validate_qb2_mesh(mesh_device)
    owner = SimpleNamespace(device=mesh_device, _cache_initialization_pending=None)
    allocate = ttnn.allocate_tensor_on_device

    def allocate_poisoned(*args, **kwargs):
        tensor = allocate(*args, **kwargs)
        ttnn.full_like(tensor, 7.0, optional_tensor=tensor)
        assert_replicated_value(tensor, 7.0)
        return tensor

    # A fresh allocation must be initialized even when its storage is dirty.
    with monkeypatch.context() as patch:
        patch.setattr(ttnn, "allocate_tensor_on_device", allocate_poisoned)
        output = allocate_cache_zero(owner, shape, dtype=dtype)
    trace = None
    try:
        assert tuple(output.shape) == tuple(output.padded_shape) == shape
        assert output.dtype == dtype
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert_replicated_value(output, 0)
        identity = output.buffer_unique_id()

        # Warm and capture the same in-place fill used during setup. Poison
        # after capture and before every replay so a missing replay must fail.
        result = ttnn.zeros_like(output, optional_tensor=output)
        assert result.buffer_unique_id() == identity
        ttnn.synchronize_device(mesh_device)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            result = ttnn.zeros_like(output, optional_tensor=output)
        finally:
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        assert result.buffer_unique_id() == identity
        for poison in (7.0, -3.0):
            ttnn.full_like(output, poison, optional_tensor=output)
            assert_replicated_value(output, poison)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            assert output.buffer_unique_id() == identity
            assert_replicated_value(output, 0)
    finally:
        ttnn.synchronize_device(mesh_device)
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        ttnn.deallocate(output)
