# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal single-device test for ttnn.experimental.deepseek_prefill.inbound_socket_service_sync.

The test runs on mesh 1x1, pushes a chunk through the service, runs the inbound_socket_service_sync op,
and checks that:

  * the op's output equals the pushed bytes (direct, per-device, bit-exact),
  * the inline 3xuint32 metadata round-trips, and
  * the op is program-cached (built once on the first call, cache hits after).
"""

import struct

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_slow_dispatch

pytestmark = [
    pytest.mark.requires_host_iommu,
    skip_for_slow_dispatch(),
    pytest.mark.skipif(
        not is_blackhole(),
        reason="H2DStreamService requires Blackhole (service-core claims); see service_core_manager.cpp",
    ),
]

_DTYPE_TORCH = torch.int32
_DTYPE_TTNN = ttnn.uint32
_DTYPE_SIZE = 4
_METADATA_SIZE_BYTES = 12  # 3 x uint32: [slot_id, actual_start, actual_end]
_ISL = 640
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

    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        max_socket_page_size_bytes=per_row_bytes,
        worker_cores=worker_cores,
        metadata_size_bytes=_METADATA_SIZE_BYTES,
    )
    logger.info(
        f"[inbound_socket_service_sync] service built: mesh={tuple(mesh_device.shape)}, isl={_ISL}, "
        f"per_row_bytes={per_row_bytes}, metadata_size_bytes={_METADATA_SIZE_BYTES}, iters={_NUM_ITERS}"
    )

    op_cache_delta = []
    for i in range(_NUM_ITERS):
        torch.manual_seed(i)
        src = torch.randint(0, 2**31, shape_list, dtype=_DTYPE_TORCH)  # < 2**31 -> lossless int64 compare
        meta = struct.pack("<III", i, 0, _ISL)  # slot_id=i, [actual_start, actual_end)

        service.forward_to_tensor_bytes(src.contiguous().numpy(), metadata=meta)

        pre = mesh_device.num_program_cache_entries()
        tokens, tt_meta = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=_METADATA_SIZE_BYTES
        )
        op_cache_delta.append(mesh_device.num_program_cache_entries() - pre)

        actual = ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).view(-1).to(torch.int64)
        assert torch.equal(actual, src.view(-1).to(torch.int64)), f"iter {i}: token bytes mismatch"

        meta_vals = ttnn.to_torch(ttnn.get_device_tensors(tt_meta)[0]).view(-1).to(torch.int64)[:3].tolist()
        assert meta_vals == [i, 0, _ISL], f"iter {i}: metadata mismatch {meta_vals}"

        logger.info(
            f"[inbound_socket_service_sync] iter {i}: synced {per_row_bytes} B/row, tokens byte-exact, "
            f"metadata={meta_vals}, program_cache_delta={op_cache_delta[i]}"
        )

    assert op_cache_delta[0] >= 1, f"expected a program-cache entry on the first call: {op_cache_delta}"
    assert all(d == 0 for d in op_cache_delta[1:]), f"op recompiled instead of cache-hitting: {op_cache_delta}"

    logger.info(
        f"[inbound_socket_service_sync] PASS: {_NUM_ITERS} iters byte-exact + metadata round-trip, "
        f"program-cached (deltas={op_cache_delta})"
    )

    service.barrier()
    del service
