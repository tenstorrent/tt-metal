# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TCP protocol between a disaggregated prefill worker (``demo/d_p_prefill_worker.py``, metal_2's whole-mesh prefill on
its own Galaxy) and a decode server that takes its prefills from it (``demo/server.py --prefill-backend d_p``).

A message is an 8-byte big-endian header length, a JSON header, then the raw blobs the header lists in ``blobs``
(byte counts, in order). Requests:

* ``{"op": "info"}`` -> ``{"max_seq", "chunk", "layers", "mesh"}``.
* ``{"op": "prefill", "blobs": [4 * n]}`` + the prompt's ids (int32) -> zero or more ``{"progress": end}`` messages
  (after each chunk is enqueued), then the states: ``{"seq_len", "seconds", "fields": [[layer, name, rows, cols]],
  "blobs": [...]}`` with one bf16 ``[rows, cols]`` blob per field, in the host format of :mod:`.d_p_bridge`.

Any reply may instead be ``{"error": message}``.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Callable, Optional, Sequence

import torch

_LEN = struct.Struct(">Q")
_MAX_HEADER = 1 << 24


class WorkerError(RuntimeError):
    """The worker reported an error for a request."""


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError(f"peer closed the connection after {got} of {n} bytes")
        got += k
    return buf


def send_msg(sock: socket.socket, header: dict, blobs: Sequence = ()) -> None:
    views = [memoryview(b).cast("B") for b in blobs]
    head = json.dumps({**header, "blobs": [v.nbytes for v in views]}).encode()
    sock.sendall(_LEN.pack(len(head)) + head)
    for v in views:
        sock.sendall(v)


def recv_msg(sock: socket.socket) -> tuple[dict, list[bytearray]]:
    (n,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    if n > _MAX_HEADER:
        raise ConnectionError(f"header of {n} bytes: not this protocol")
    header = json.loads(_recv_exact(sock, n))
    return header, [_recv_exact(sock, k) for k in header.get("blobs", [])]


def _bf16_bytes(t: torch.Tensor) -> memoryview:
    return memoryview(t.to(torch.bfloat16).contiguous().view(torch.int16).numpy())


def send_states(sock: socket.socket, host: dict, seconds: float) -> None:
    fields, blobs = [], []
    for i in sorted(host["layers"]):
        for name, t in host["layers"][i].items():
            fields.append([int(i), name, int(t.shape[0]), int(t.shape[1])])
            blobs.append(_bf16_bytes(t))
    send_msg(sock, {"seq_len": int(host["seq_len"]), "seconds": float(seconds), "fields": fields}, blobs)


def _states(header: dict, blobs: list[bytearray]) -> dict:
    layers: dict = {}
    for (i, name, rows, cols), blob in zip(header["fields"], blobs):
        t = torch.frombuffer(blob, dtype=torch.int16).view(torch.bfloat16).reshape(rows, cols)
        layers.setdefault(int(i), {})[name] = t
    return {"seq_len": int(header["seq_len"]), "seconds": float(header["seconds"]), "layers": layers}


class PrefillClient:
    """The decode server's end: one connection to the worker at ``address`` (``host:port``), one request at a time."""

    def __init__(self, address: str, timeout: float = 3600.0):
        host, _, port = address.rpartition(":")
        self.address = address
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info = self._call({"op": "info"})[0]

    def _call(self, header: dict, blobs: Sequence = ()):
        send_msg(self.sock, header, blobs)
        return self._reply()

    def _reply(self):
        header, blobs = recv_msg(self.sock)
        if "error" in header:
            raise WorkerError(f"prefill worker {self.address}: {header['error']}")
        return header, blobs

    def prefill(self, ids: Sequence[int], on_progress: Optional[Callable[[int], None]] = None) -> dict:
        """Host states of ``ids`` prefilled from position 0 (:mod:`.d_p_bridge`'s format, plus ``seconds``: the
        worker's own prefill time)."""
        send_msg(self.sock, {"op": "prefill"}, [torch.tensor(list(ids), dtype=torch.int32).numpy()])
        while True:
            header, blobs = self._reply()
            if "progress" not in header:
                return _states(header, blobs)
            if on_progress is not None:
                on_progress(int(header["progress"]))

    def close(self) -> None:
        self.sock.close()
