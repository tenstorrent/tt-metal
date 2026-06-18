# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper around the C++ ``ttnn.experimental.deepseek_prefill.h2d_socket_sync`` op.
"""

import ttnn


def h2d_socket_sync(
    service: ttnn.H2DStreamService,
    worker_cores: ttnn.CoreRange | None = None,
    *,
    metadata_size_bytes: int = 0,
) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
    """Wait for the next H2D transfer to land in ``service.get_backing_tensor()``,
    copy it into a fresh device tensor, and ack the service core.

    Args:
        service: A persistent H2DStreamService constructed with `worker_cores` set
            to the same CoreRange passed here. Provides the data-ready semaphore,
            per-coord consumed counter, and per-coord service-core coordinates.
        worker_cores: Worker CoreRange — must match the `worker_cores` the service
            was constructed with. Each core runs one iteration of the
            wait → copy slice → ack protocol.
        metadata_size_bytes: When > 0, must match the value passed to the service
            constructor. The kernel additionally copies the inline metadata
            multicast by the service core (lives at `service.get_metadata_addr()`
            in worker L1) into a fresh DRAM tensor; this function then returns
            `(tokens_tensor, metadata_tensor)` instead of just the tokens tensor.
            Must be a multiple of 4 bytes (we expose the metadata buffer as a
            uint32 tensor).

    Returns:
        When `metadata_size_bytes == 0`: a single ttnn.Tensor with the same
        per-shard spec as `service.get_backing_tensor()`.
        When `metadata_size_bytes > 0`: a tuple `(tokens_tensor, metadata_tensor)`
        where the metadata tensor has shape `[1, 1, 1, metadata_size_bytes // 4]`
        uint32 ROW_MAJOR DRAM, replicated across the mesh (each coord's worker
        writes the same metadata it received from its service core multicast).
    """
    outputs = ttnn.experimental.deepseek_prefill.h2d_socket_sync(service, metadata_size_bytes=metadata_size_bytes)
    if metadata_size_bytes > 0:
        return outputs[0], outputs[1]
    return outputs[0]
