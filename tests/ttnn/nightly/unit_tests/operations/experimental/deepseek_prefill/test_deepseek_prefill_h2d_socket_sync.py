# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal single-device data-flow test for ttnn.experimental.deepseek_prefill.h2d_socket_sync.

test_h2d_stream_service.py only validates the worker_cores/metadata *binding surface*
(it never runs a successful worker-sync forward). This drives the real device-side
handshake end to end on the smallest possible mesh (1x1): push a chunk through the
service, run the C++ h2d_socket_sync op, and check that

  * the op's output equals the pushed bytes (direct, per-device, bit-exact),
  * the inline 3xuint32 metadata round-trips, and
  * the op is program-cached (built once on the first call, pure cache hits after).

H2DStreamService claims an FD dispatch-column service core and DMA-pins host memory, so it
requires Blackhole (any single card or Galaxy; see service_core_manager.cpp) plus fast
dispatch. This module is skipped on Wormhole and under slow dispatch.
"""

import struct

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_slow_dispatch

pytestmark = [
    skip_for_slow_dispatch(),
    pytest.mark.skipif(
        not is_blackhole(),
        reason="H2DStreamService requires Blackhole (service-core claims); see service_core_manager.cpp",
    ),
]

# int32 source viewed as UINT32; compared as int64 (lossless for 0..INT32_MAX).
_DTYPE_TORCH = torch.int32
_DTYPE_TTNN = ttnn.uint32
_DTYPE_SIZE = 4
_RANDINT_HIGH = 2**31  # negatives would break the int64 readback compare
_METADATA_SIZE_BYTES = 12  # 3 x uint32: [slot_id, actual_start, actual_end]
_ISL = 640  # tokens/device: one socket page, tile-aligned
_NUM_ITERS = 3


@pytest.mark.parametrize("mesh_device", [1], indirect=True)  # 1x1 single-device mesh
def test_h2d_socket_sync_single_device(mesh_device):
    shape_list = [mesh_device.shape[0], 1, _ISL]
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape(shape_list),
        dtype=_DTYPE_TTNN,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    per_row_bytes = _ISL * _DTYPE_SIZE

    # Replicate across the (1x1) mesh; matches the service's default mapper, and lets the
    # readback compare each device's shard against the un-sharded source directly.
    placements = [ttnn.PlacementReplicate() for _ in range(mesh_device.shape.dims())]
    iter_mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(placements=placements))

    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=8 * per_row_bytes,
        scratch_cb_size_bytes=per_row_bytes,
        worker_cores=worker_cores,
        metadata_size_bytes=_METADATA_SIZE_BYTES,
    )

    op_cache_delta = []
    for i in range(_NUM_ITERS):
        # Vary the data every iteration so a stale/clobbered output would fail the compare.
        gen = torch.Generator()
        gen.manual_seed(i)
        src = torch.randint(low=0, high=_RANDINT_HIGH, size=shape_list, dtype=_DTYPE_TORCH, generator=gen)
        expected_host = ttnn.from_torch(src, spec=global_spec, mesh_mapper=iter_mapper)
        meta = struct.pack("<III", i, 0, _ISL)  # slot_id=i, [actual_start, actual_end)

        service.forward_to_tensor_bytes(src.contiguous().numpy(), metadata=meta)

        pre = mesh_device.num_program_cache_entries()
        tokens, tt_meta = ttnn.experimental.deepseek_prefill.h2d_socket_sync(
            service, metadata_size_bytes=_METADATA_SIZE_BYTES
        )
        op_cache_delta.append(mesh_device.num_program_cache_entries() - pre)

        # Direct, bit-exact: the op's output must equal the pushed source on every device.
        actual_subs = ttnn.get_device_tensors(tokens)
        expected_subs = ttnn.get_device_tensors(expected_host)
        assert len(actual_subs) == len(expected_subs), f"iter {i}: shard count mismatch"
        for d, (actual, expected) in enumerate(zip(actual_subs, expected_subs)):
            a_t = ttnn.to_torch(actual).view(-1).to(torch.int64)
            e_t = ttnn.to_torch(expected).view(-1).to(torch.int64)
            assert torch.equal(a_t, e_t), f"iter {i} device {d}: token bytes mismatch"

        # Inline metadata round-trips (replicated across the mesh; read device 0).
        meta_vals = ttnn.to_torch(ttnn.get_device_tensors(tt_meta)[0]).view(-1).to(torch.int64)[:3].tolist()
        assert meta_vals == [i, 0, _ISL], f"iter {i}: metadata mismatch {meta_vals}"

    # Program cache: compiled once on the first call, pure cache hits afterward.
    assert op_cache_delta[0] >= 1, f"expected a program-cache entry on the first call: {op_cache_delta}"
    assert all(d == 0 for d in op_cache_delta[1:]), f"op recompiled instead of cache-hitting: {op_cache_delta}"

    service.barrier()
    del service
