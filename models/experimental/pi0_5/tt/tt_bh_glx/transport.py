# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-chip transport for the BH-Galaxy pipeline.

v2 (this file) uses fabric **mesh sockets** with PR #45765's
``ttnn.experimental.send_direct_async`` / ``recv_direct_async`` — the
sender writes the tensor directly into the receiver's pre-allocated
output tensor, bypassing the socket FIFO. Cross-chip handoff happens
entirely on-device; no host-bounce, no ``ttnn.synchronize_device`` calls
(trace-compatible).

API:
    transport = SocketTransport()
    out_buf = transport.allocate_recv_buffer(spec, dst_mesh)   # ONCE at __init__
    transport.send(src_tensor, dst_mesh, out_buf)              # per call
    # out_buf now holds the transferred contents

The send_direct_async path requires the receiver tensor to be a real
device tensor on `dst_mesh` with the same shape/dtype/layout as the
source. We cache one (send_socket, recv_socket) pair per (src_mesh,
dst_mesh) since pair construction is non-trivial (involves cross-mesh
fabric routing) and the per-hop wiring is stable across calls.

v1 (host-bounce) ``send_via_host`` is kept as a fallback under
``PI05_GLX_TRANSPORT=host`` for A/B testing without rebuilding.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import ttnn


_HANDSHAKE_PAGE_SIZE = 4096  # bytes; small — direct-mode socket only uses FIFO for the handshake

# send_direct_async runs one worker core per SocketConnection, round-robined onto
# the adjacent chip pair's 2 forwarding fabric links. 2 connections measured faster
# on the full all-socket traced e2e: 67.1 → 63.5 ms (14.9 → 15.8 infer/s), torch
# PCC unchanged (0.9971 → 0.9974), 2026-06-13 same-box A/B. Gain is modest (~5%, not
# 2×) because the pipeline is serialization/dispatch-bound, not socket-bandwidth-
# bound. The C++ "multiple sender cores on a single device" line is a warning only —
# no correctness impact here. Multi-conn support is in the .so (committed 7009cd6a47c);
# this is Python-only. Set PI05_SOCK_CONN=1 to A/B.
_N_SOCK_CONN = int(os.environ.get("PI05_SOCK_CONN", "2"))


def send_via_host(src_tensor, dst_submesh, memory_config=None):
    """LEGACY host-bounce path (v1). Kept for A/B testing via PI05_GLX_TRANSPORT=host."""
    host = ttnn.to_torch(src_tensor)
    return ttnn.from_torch(
        host,
        dtype=src_tensor.dtype,
        layout=src_tensor.layout,
        device=dst_submesh,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )


class SocketTransport:
    """Pre-cached socket pairs + receiver buffers between per-chip submeshes.

    Each (src_mesh, dst_mesh) pair gets ONE (send_socket, recv_socket) tuple
    AND ONE pre-allocated receiver tensor (matching the first sent tensor's
    spec), built lazily on first ``send`` and reused for every subsequent
    transfer between the same pair. Both meshes must be 1x1 submeshes carved
    from the same parent mesh with FABRIC_2D enabled (see
    mesh_setup.open_galaxy_mesh).

    Pipelines should call ``send(src, dst)`` instead of the legacy
    ``send_via_host`` — the transport handles socket creation, receiver
    buffer allocation, and the direct-write transfer with no host bounce
    and no ``ttnn.synchronize_device`` calls.
    """

    def __init__(self):
        # Key: (id(src_mesh), id(dst_mesh)) → (send_sock, recv_sock)
        self._pairs: Dict[Tuple[int, int], Tuple] = {}
        # Key: (id(src_mesh), id(dst_mesh)) → pre-allocated receiver tensor
        self._recv_bufs: Dict[Tuple[int, int], object] = {}

    def _pair(self, src_mesh, dst_mesh):
        key = (id(src_mesh), id(dst_mesh))
        pair = self._pairs.get(key)
        if pair is None:
            # For 1x1 submeshes, the only mesh coordinate is (0,0). One worker
            # core per connection (sender (i,0) → receiver (i,1)); N connections
            # spread send_direct_async across the chip pair's 2 fabric links.
            conns = [
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(i, 0)),
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(i, 1)),
                )
                for i in range(_N_SOCK_CONN)
            ]
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, _HANDSHAKE_PAGE_SIZE * 4)
            cfg = ttnn.SocketConfig(conns, mem)
            pair = ttnn.create_socket_pair(src_mesh, dst_mesh, cfg)
            self._pairs[key] = pair
        return pair

    def allocate_recv_buffer(self, src_template, dst_mesh):
        """Allocate a receiver tensor on dst_mesh that matches src_template's spec.

        Call ONCE per (src, dst) hop at pipeline __init__; reuse the returned
        tensor for every subsequent ``send`` call to that destination.
        """
        return ttnn.allocate_tensor_on_device(src_template.spec, dst_mesh)

    def send(self, src_tensor, dst_mesh, *, out_buf=None, tag=None):
        """Direct-write transfer src_tensor → receiver buffer via fabric socket.

        Buffer cache is keyed by (src_mesh, dst_mesh, tag). When two distinct
        tensors travel between the same chip pair (e.g. K then V in KV
        migration), pass a different ``tag`` to each so they get separate
        receiver buffers. Default tag=None covers stages where each chip
        pair carries exactly one tensor per inference.

        If ``out_buf`` is None, a receiver buffer is lazily allocated on
        ``dst_mesh`` matching ``src_tensor.spec`` and cached for every
        subsequent call with the same (src, dst, tag).

        No ``ttnn.synchronize_device`` calls — TTNN's normal command-queue
        ordering on the sender + receiver meshes preserves the data
        dependency for the op that consumes the receiver buffer next.
        """
        src_mesh = src_tensor.device()
        send_sock, recv_sock = self._pair(src_mesh, dst_mesh)
        if out_buf is None:
            key = (id(src_mesh), id(dst_mesh), tag)
            out_buf = self._recv_bufs.get(key)
            if out_buf is None:
                out_buf = ttnn.allocate_tensor_on_device(src_tensor.spec, dst_mesh)
                self._recv_bufs[key] = out_buf
        ttnn.experimental.send_direct_async(src_tensor, send_sock)
        ttnn.experimental.recv_direct_async(out_buf, recv_sock)
        return out_buf


def make_transport():
    """Factory — returns SocketTransport unless PI05_GLX_TRANSPORT=host."""
    if os.environ.get("PI05_GLX_TRANSPORT", "").lower() == "host":
        return None  # callers should use send_via_host directly in that mode
    return SocketTransport()
