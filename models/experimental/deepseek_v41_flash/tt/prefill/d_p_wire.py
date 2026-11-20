# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TCP protocol between the disaggregated prefill server (``demo/prefill_server.py``: metal_2's whole-mesh prefill on its
own Galaxy) and the decode servers that send it prompts and take the KV cache entries back.

A message is an 8-byte big-endian header length, a JSON header, then the raw blobs the header lists in ``blobs``
(byte counts, in order). A connection carries one request at a time; requests:

* ``{"op": "ping"}`` -> ``{"pong": true}``.
* ``{"op": "info"}`` -> ``{"version", "max_seq", "chunk", "align", "layers", "kv_sources", "window", "vocab",
  "mesh"}`` (``kv_sources``: layer -> compression ratio).
* ``{"op": "prefill"}`` + one blob, the prompt's ids (int32, a multiple of ``align`` tokens, at most ``max_seq``)
  -> zero or more ``{"progress": end}`` (each chunk enqueued), then the KV entries: ``{"seq_len", "seconds",
  "queued", "fields": [[layer, name, rows, cols], ...]}`` with one bf16 ``[rows, cols]`` blob per field:

  * ``kv_tail`` of every layer: the window's K=V rows of positions ``[seq_len - window, seq_len)``, in order,
  * ``compressed_kv`` and ``idx_keys`` of each KV source layer: ``seq_len / ratio`` rows.

  ``seconds`` is the prefill's device time, ``queued`` how long the request waited for another one to finish.

Any reply may instead be ``{"error": message}``; the connection stays usable after it.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Callable, Optional, Sequence

import torch

VERSION = 1
_LEN = struct.Struct(">Q")
_MAX_HEADER = 1 << 24


class PrefillServerError(RuntimeError):
    """The prefill server rejected or failed a request."""


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


def send_states(sock: socket.socket, host: dict, seconds: float, queued: float = 0.0) -> None:
    fields, blobs = [], []
    for i in sorted(host["layers"]):
        for name, t in host["layers"][i].items():
            fields.append([int(i), name, int(t.shape[0]), int(t.shape[1])])
            blobs.append(_bf16_bytes(t))
    header = {"seq_len": int(host["seq_len"]), "seconds": float(seconds), "queued": float(queued), "fields": fields}
    send_msg(sock, header, blobs)


def _states(header: dict, blobs: list[bytearray]) -> dict:
    layers: dict = {}
    for (i, name, rows, cols), blob in zip(header["fields"], blobs):
        t = torch.frombuffer(blob, dtype=torch.int16).view(torch.bfloat16).reshape(rows, cols)
        layers.setdefault(int(i), {})[name] = t
    return {
        "seq_len": int(header["seq_len"]),
        "seconds": float(header["seconds"]),
        "queued": float(header.get("queued", 0.0)),
        "bytes": sum(len(b) for b in blobs),
        "layers": layers,
    }


class PrefillClient:
    """A decode server's connection to the prefill server at ``address`` (``host:port``), one request at a time.

    :meth:`prefill` returns the KV entries in :mod:`.d_p_bridge`'s host format (``{"seq_len", "layers": {i: {...}}}``,
    plus ``seconds``, ``queued`` and ``bytes``), ready for :func:`~.d_p_bridge.to_prefill_states`."""

    def __init__(self, address: str, timeout: float = 3600.0):
        host, _, port = address.rpartition(":")
        self.address = address
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info = self._call({"op": "info"})[0]
        self.info.pop("blobs", None)
        if self.info.get("version") != VERSION:
            raise PrefillServerError(f"{address} speaks protocol {self.info.get('version')}, this client {VERSION}")

    def __enter__(self) -> "PrefillClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _call(self, header: dict, blobs: Sequence = ()):
        send_msg(self.sock, header, blobs)
        return self._reply()

    def _reply(self):
        header, blobs = recv_msg(self.sock)
        if "error" in header:
            raise PrefillServerError(f"prefill server {self.address}: {header['error']}")
        return header, blobs

    def ping(self) -> bool:
        return bool(self._call({"op": "ping"})[0].get("pong"))

    def prefill(self, ids: Sequence[int], on_progress: Optional[Callable[[int], None]] = None) -> dict:
        """KV entries of ``ids`` prefilled from position 0. ``on_progress(end)`` as the server enqueues each chunk."""
        send_msg(self.sock, {"op": "prefill"}, [torch.tensor(list(ids), dtype=torch.int32).numpy()])
        while True:
            header, blobs = self._reply()
            if "progress" not in header:
                return _states(header, blobs)
            if on_progress is not None:
                on_progress(int(header["progress"]))

    def close(self) -> None:
        self.sock.close()
