# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SplitSocketTransport -- subclass of the BYTE-UNCHANGED Galaxy ``tt_bh_glx.SocketTransport``.

Framework mandate (plan §5): subclass the Galaxy transport rather than re-vendoring it
wholesale. The Galaxy parent provides only a TAG-LESS ``_pair(src, dst)`` (2-arg),
TAG-LESS ``allocate_recv_buffer(src_template, dst_mesh)``, a ``send(...)`` (exercised by
the Galaxy tests), and ``__init__(self)`` (no kwargs); it LACKS ``prepare`` / ``send_only``
/ ``recv_only`` / ``close`` / ``_check_payload`` / ``_connections``, and its ``_pair``
hard-codes ``_N_SOCK_CONN`` (module-default 2) via a module constant.

This subclass VENDORS the tag-keyed split helpers from the tt_symbiote
``core/d2d_transport.SocketTransport`` PRIVATELY (shadowing the parent's tag-less versions),
keys the pair/recv caches by ``(id(src), id(dst), tag)`` (so the velocity_wrap pair on the
stage[-1]<->stage[0] hop with a distinct tag does not collide with the hop tags), bakes
``num_connections=1`` (the tt_symbiote streamed default; the subclass builds its OWN
SocketConnection list -> the Galaxy module-default of 2 is irrelevant to our instances),
and NEVER calls the parent's tag-less ``send`` (which the Galaxy tests exercise; calling it
would risk SC4). The 3-arg tag-keyed ``_pair`` / ``allocate_recv_buffer`` shadows affect
ONLY our instances -> SC4 safe (tt_bh_glx byte-unchanged).

ZERO tt_symbiote imports. The ONLY external dependency is the byte-unchanged Galaxy
``tt_bh_glx.transport.SocketTransport`` (imported, never edited).
"""
from __future__ import annotations

import ttnn

# Import the BYTE-UNCHANGED Galaxy transport (never edited). Importing it does NOT pull
# tt_symbiote -- tt_bh_glx is self-contained.
from models.experimental.pi0_5.tt.tt_bh_glx.transport import SocketTransport as _GlxSocketTransport

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

# Import-time guard: the fabric direct-write socket ops must exist in the active ttnn build
# (the unchanged tt_bh_glx.transport already uses these against the same build -> present).
assert hasattr(ttnn.experimental, "send_direct_async") and hasattr(ttnn.experimental, "recv_direct_async"), (
    f"d2d socket ops absent from this ttnn ({ttnn.__file__}); a tt-metal build with "
    f"send_direct_async/recv_direct_async is required for SplitSocketTransport."
)

_HANDSHAKE_PAGE_SIZE = 4096  # direct mode uses the FIFO only for the handshake
_L1 = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)


def _as_l1(t):
    if t.memory_config().buffer_type != ttnn.BufferType.L1:
        return ttnn.to_memory_config(t, _L1)
    return t


class SplitSocketTransport(_GlxSocketTransport):
    """Tag-keyed split-API socket transport for the streamed (trace) denoise path."""

    def __init__(self, *, num_connections: int = 1, page_size: int = _HANDSHAKE_PAGE_SIZE):
        super().__init__()  # parent __init__ takes no kwargs; sets its own (unused) caches
        # Our OWN tag-keyed caches: (id(src), id(dst), tag). These shadow the parent's
        # tag-less caches for our instances only.
        self._num_connections = num_connections
        self._page_size = page_size
        self._pairs = {}
        self._recv_bufs = {}

    # ------------------------------------------------------------------ private split API
    def _connections(self, src_mesh):
        # Build the SocketConnection list directly with OUR num_connections (the Galaxy
        # parent hard-codes _N_SOCK_CONN=2 via a module constant; we override here).
        sender = [ttnn.CoreCoord(i, 0) for i in range(self._num_connections)]
        recv = [ttnn.CoreCoord(i, 1) for i in range(self._num_connections)]
        conns = []
        for coord in ttnn.MeshCoordinateRange(src_mesh.shape):
            for s, r in zip(sender, recv):
                conns.append(ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, s), ttnn.MeshCoreCoord(coord, r)))
        return conns

    def _pair(self, src_mesh, dst_mesh, tag):  # SHADOWS parent (3-arg, tag-keyed)
        key = (id(src_mesh), id(dst_mesh), tag)
        pair = self._pairs.get(key)
        if pair is None:
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, self._page_size * 4)
            cfg = ttnn.SocketConfig(self._connections(src_mesh), mem)
            pair = ttnn.create_socket_pair(src_mesh, dst_mesh, cfg)
            self._pairs[key] = pair
        return pair

    def allocate_recv_buffer(self, src_template, dst_mesh, *, tag=None):  # SHADOWS (tag-keyed)
        key = (id(src_template.device()), id(dst_mesh), tag)
        buf = self._recv_bufs.get(key)
        if buf is None:
            buf = ttnn.allocate_tensor_on_device(src_template.spec, dst_mesh)
            self._recv_bufs[key] = buf
        return buf

    def _check_payload(self, src_tensor):
        # Direct-write reads the source from L1 (a DRAM source silently corrupts the transfer).
        assert (
            src_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), f"d2d payload must be L1-resident, got {src_tensor.memory_config().buffer_type}"
        # aligned page must fit the socket FIFO (verbatim from d2d_transport.py:92-95)
        aligned = getattr(src_tensor, "buffer_aligned_page_size", None)
        if callable(aligned):
            ap = src_tensor.buffer_aligned_page_size()
            assert self._page_size * 4 >= ap, f"socket fifo {self._page_size * 4} < aligned_page {ap}"

    def prepare(self, src_tensor, dst_mesh, *, tag=None):
        """Build+cache the socket pair and receiver buffer for this hop without issuing ops."""
        src_tensor = _as_l1(src_tensor)
        self._check_payload(src_tensor)
        send_sock, recv_sock = self._pair(src_tensor.device(), dst_mesh, tag)
        out_buf = self.allocate_recv_buffer(src_tensor, dst_mesh, tag=tag)
        return send_sock, recv_sock, out_buf

    def send_only(self, src_tensor, send_sock):
        ttnn.experimental.send_direct_async(src_tensor, send_sock)

    def recv_only(self, out_buf, recv_sock):
        ttnn.experimental.recv_direct_async(out_buf, recv_sock)

    def send(self, src_tensor, dst_mesh, *, out_buf=None, tag=None):
        """Eager d2d transfer (reference path + eager adapter chain). Uses the tag-keyed split
        API -- does NOT call the parent's tag-less send. No synchronize_device (trace-compatible)."""
        src_tensor = _as_l1(src_tensor)
        send_sock, recv_sock, buf = self.prepare(src_tensor, dst_mesh, tag=tag)
        if out_buf is not None:
            buf = out_buf
        self.send_only(src_tensor, send_sock)
        self.recv_only(buf, recv_sock)
        return buf

    def close(self):
        self._pairs.clear()
        self._recv_bufs.clear()
