# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper around the native C++ ``ttnn.experimental.h2d_socket_sync`` op.

This module used to build a ``MeshProgramDescriptor`` and dispatch it via
``ttnn.generic_op`` on every call. Rebuilding + dispatching that program per
iteration dominated prefill runtime (~1.2 s/chunk; see RUNNER_PERF_INVESTIGATION).
The logic now lives in C++ as a program-cached device operation at
``ttnn/cpp/ttnn/operations/experimental/h2d_socket_sync/`` — the program is built
once and later calls only patch the fresh output address via BufferBindings.

This wrapper preserves the original Python call contract so callers
(``prefill_runner.py``) are unchanged.
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
        service: A persistent ``H2DStreamService`` constructed with ``worker_cores``
            set (and ``metadata_size_bytes`` if the metadata path is used).
        worker_cores: Accepted for backwards-compatibility only. The worker grid is
            now read from the service itself (``service.get_worker_cores()``); when
            provided it must match.
        metadata_size_bytes: When > 0, must match the value the service was
            constructed with; the op then also returns the inline metadata tensor.

    Returns:
        The tokens tensor, or ``(tokens, metadata)`` when ``metadata_size_bytes > 0``.
    """
    outputs = ttnn.experimental.h2d_socket_sync(service, metadata_size_bytes=metadata_size_bytes)
    if metadata_size_bytes > 0:
        return outputs[0], outputs[1]
    return outputs[0]
