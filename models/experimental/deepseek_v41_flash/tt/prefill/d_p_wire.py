# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TCP protocol between the disaggregated prefill server (``demo/prefill_server.py``: whole-mesh prefill on its
own Galaxy) and the decode servers that send it prompts and take the KV cache entries back.

A message is an 8-byte big-endian header length, a JSON header, then the raw blobs the header lists in ``blobs``
(byte counts, in order). A connection carries one request at a time; requests:

* ``{"op": "ping"}`` -> ``{"pong": true}``.
* ``{"op": "stats"}`` -> ``{"requests", "tokens", "seconds": {stage: seconds}}``: the server's per-stage profile
  summed over every request it served (stages: ``demo/prefill_server.py`` ``PrefillService``).
* ``{"op": "info"}`` -> ``{"version", "max_seq", "chunk", "align", "layers", "kv_sources", "window", "vocab",
  "mesh", "cache_slots"}`` (``kv_sources``: layer -> compression ratio; ``cache_slots``: the server's prefix-cache
  slots).
* ``{"op": "prefill", "have": H, "cache": true}`` + one blob, the prompt's ids (int32, a multiple of ``align``
  tokens, at most ``max_seq``, always the whole prompt from position 0) -> zero or more ``{"progress": end}``
  (each chunk enqueued), then the KV entries: ``{"seq_len", "base", "cached", "seconds", "queued", "fields":
  [[layer, name, rows, cols], ...]}`` with one bf16 ``[rows, cols]`` blob per field:

  * ``kv_tail`` of every layer: the window's K=V rows of positions ``[seq_len - window, seq_len)``, in order,
  * ``compressed_kv`` and ``idx_keys`` of each KV source layer: the entries of positions ``[base, seq_len)``,
    ``(seq_len - base) / ratio`` rows (possibly 0, a 0-byte blob).

  ``have`` (default 0) is how many leading tokens of the prompt the client already holds the entries of: a
  multiple of ``align`` up to the prompt length, and the reply's ``base``. ``cached`` is how many tokens the server
  took from its prefix cache instead of computing (``cached == seq_len``: nothing computed, ``seconds`` is 0); it
  never changes which entries are sent. ``cache: false`` skips the lookup (the result is still kept).
  ``seconds`` is the prefill's device time, ``queued`` how long the request waited for another one to finish.
* ``{"op": "prefill_push", "have": H, "cache": true, "targets": {...}}`` + the ids blob, as ``prefill``, but the
  server writes the state straight into the client's device memory over TT-Fabric (``tt/prefill/d_p_push.py``;
  both servers ranks of one fabric job, ``PREFILL_D_P_FABRIC_PUSH.md``): ``targets`` is what the decode model's
  ``remote_commit_targets`` returned for the session (its ``seq_len`` the prompt's length, its ``start`` ``have``).
  Reply: zero or more ``{"progress": end}``, then ``{"seq_len", "base", "cached", "seconds", "queued", "bytes",
  "pages", "push_seconds"}`` and no blobs, once every write has left the server's mesh (``bytes`` / ``pages``:
  what it wrote, ``push_seconds``: how long that took).
* ``{"op": "prefill_warm"}`` + the ids blob (servers whose ``info`` has ``"warm": true``): prefill the prompt into
  the server's prefix cache only, for a request the client cannot place yet (all its decode slots taken), so that its
  ``prefill`` / ``prefill_push`` later computes nothing. It runs after every other request waiting and stops after a
  chunk when one comes in. Reply ``{"seq_len", "done", "cached", "seconds", "queued"}``: ``done`` how far it got
  (``seq_len``: all of it), the cache keeping ``[0, done)``, so sending it again continues from there.

Images (servers whose ``info`` has ``"image_rows"`` > 0): a ``prefill`` / ``prefill_push`` prompt may hold image-span
tokens as the decode server lays them out (``tt/vision.py``: ``vocab + 0 .. 2`` the span delimiters, ``vocab + 3 + s``
the vision row in the decode server's slot ``s``) when the header names every image whose rows it uses: ``"images":
[{"key", "slots", "rows"}, ...]``, ``key`` the image (the sha1 of its bytes), ``slots`` the decode slots of its rows
in order, ``rows`` true when the request carries them: then one more blob per such image, in order, its bf16
``[len(slots), hidden]`` vision rows. The server keeps an image's rows by key (``demo/prefill_server.py`` ImageBook);
when it holds neither them nor the blob it replies ``{"need": [key, ...]}`` instead, having run nothing, and the
client sends the request again with those rows.

Any reply may instead be ``{"error": message}``; the connection stays usable after it.
"""

from __future__ import annotations

import json
import socket
import struct
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from urllib.parse import urlparse

import torch

VERSION = 3
_LEN = struct.Struct(">Q")
_MAX_HEADER = 1 << 24


class PrefillServerError(RuntimeError):
    """The prefill server rejected or failed a request."""


def parse_prefill_address(address: str) -> tuple[str, int]:
    """Parse a prefill server URL or ``host:port`` string into ``(host, port)``.

    Accepts ``tcp://host:port``, ``http://host:port``, or bare ``host:port`` (including IPv6 ``[::1]:9041``).
    """
    raw = address.strip()
    if not raw:
        raise ValueError("empty prefill server address")
    if "://" in raw:
        parsed = urlparse(raw)
        if parsed.scheme not in ("tcp", "http", "https"):
            raise ValueError(f"unsupported prefill URL scheme {parsed.scheme!r} in {address!r}")
        if not parsed.hostname or parsed.port is None:
            raise ValueError(f"prefill URL must include host and port, got {address!r}")
        return parsed.hostname, int(parsed.port)
    parsed = urlparse(f"tcp://{raw}")
    if not parsed.hostname or parsed.port is None:
        raise ValueError(f"prefill address must be host:port or tcp://host:port, got {address!r}")
    return parsed.hostname, int(parsed.port)


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
    # memoryview.cast refuses a view with a zero in its shape (a 0-row field).
    views = [memoryview(b).cast("B") if memoryview(b).nbytes else memoryview(b"") for b in blobs]
    head = json.dumps({**header, "blobs": [v.nbytes for v in views]}).encode()
    sock.sendall(_LEN.pack(len(head)) + head)
    for v in views:
        sock.sendall(v)


def recv_msg(sock: socket.socket, times: Optional[dict] = None) -> tuple[dict, list[bytearray]]:
    """One message. ``times`` gets ``wait`` (seconds until its first bytes came), ``transfer`` (the rest) and
    ``bytes``."""
    t0 = time.perf_counter()
    (n,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    t1 = time.perf_counter()
    if n > _MAX_HEADER:
        raise ConnectionError(f"header of {n} bytes: not this protocol")
    header = json.loads(_recv_exact(sock, n))
    blobs = [_recv_exact(sock, k) for k in header.get("blobs", [])]
    if times is not None:
        times["wait"] = t1 - t0
        times["transfer"] = time.perf_counter() - t1
        times["bytes"] = _LEN.size + n + sum(len(b) for b in blobs)
    return header, blobs


def _bf16_bytes(t: torch.Tensor) -> memoryview:
    return memoryview(t.to(torch.bfloat16).contiguous().view(torch.int16).numpy())


def send_states(
    sock: socket.socket,
    host: dict,
    seconds: float,
    queued: float = 0.0,
    base: Optional[int] = None,
    cached: int = 0,
) -> None:
    fields, blobs = [], []
    for i in sorted(host["layers"]):
        for name, t in host["layers"][i].items():
            fields.append([int(i), name, int(t.shape[0]), int(t.shape[1])])
            blobs.append(_bf16_bytes(t))
    header = {
        "seq_len": int(host["seq_len"]),
        "base": int(host.get("base", 0) if base is None else base),
        "cached": int(cached),
        "seconds": float(seconds),
        "queued": float(queued),
        "fields": fields,
    }
    send_msg(sock, header, blobs)


def _states(header: dict, blobs: list[bytearray]) -> dict:
    layers: dict = {}
    for (i, name, rows, cols), blob in zip(header["fields"], blobs):
        if rows == 0:  # torch.frombuffer refuses an empty buffer
            t = torch.empty(0, cols, dtype=torch.bfloat16)
        else:
            t = torch.frombuffer(blob, dtype=torch.int16).view(torch.bfloat16).reshape(rows, cols)
        layers.setdefault(int(i), {})[name] = t
    return {
        "seq_len": int(header["seq_len"]),
        "base": int(header.get("base", 0)),
        "cached": int(header.get("cached", 0)),
        "seconds": float(header["seconds"]),
        "queued": float(header.get("queued", 0.0)),
        "bytes": sum(len(b) for b in blobs),
        "layers": layers,
    }


@dataclass
class WireImage:
    """One image of a prompt (module docstring): ``key`` the image, ``slots`` the decode slots of its rows in order,
    ``rows()`` its bf16 ``[len(slots), hidden]`` vision rows (read only when sent), ``send_rows`` whether to send them
    without being asked (the server is not known to hold them)."""

    key: str
    slots: Sequence[int]
    rows: Callable[[], torch.Tensor]
    send_rows: bool = True


class PrefillClient:
    """A decode server's connection to the prefill server at ``address`` (``host:port`` or ``tcp://host:port``),
    one request at a time.

    :meth:`prefill` returns the KV entries in host format (``{"seq_len", "base", "layers": {i: {...}}}``,
    plus ``cached``, ``seconds``, ``queued`` and ``bytes``), ready for
    :meth:`~..model.DeepSeekV41Model.commit_host_prefill_state` (``start=base``).
    """

    def __init__(self, address: str, timeout: float = 3600.0):
        host, port = parse_prefill_address(address)
        self.address = f"{host}:{port}"
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sent_images: set = set()  # (_prompt: the keys whose rows the last prompt sent)
        # The last prompt's client-side seconds: "send" (the request out, image rows included), "first_reply"
        # (from sent to the first reply: the server's queue and first chunk), "replies" (from there to the last
        # reply), "recv" (reading the last reply's bytes: the KV entries without push), "need_rounds".
        self.times: dict = {}
        try:
            self.info = self._call({"op": "info"})[0]
            self.info.pop("blobs", None)
            if self.info.get("version") != VERSION:
                raise PrefillServerError(
                    f"{self.address} speaks protocol {self.info.get('version')}, this client {VERSION}"
                )
        except Exception:
            self.sock.close()
            raise

    def __enter__(self) -> "PrefillClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _call(self, header: dict, blobs: Sequence = ()):
        send_msg(self.sock, header, blobs)
        return self._reply()

    def _reply(self, times: Optional[dict] = None):
        header, blobs = recv_msg(self.sock, times)
        if "error" in header:
            raise PrefillServerError(f"prefill server {self.address}: {header['error']}")
        return header, blobs

    def ping(self) -> bool:
        return bool(self._call({"op": "ping"})[0].get("pong"))

    def stats(self) -> dict:
        """The server's per-stage seconds summed over every request it served (``stats``)."""
        header, _ = self._call({"op": "stats"})
        header.pop("blobs", None)
        return header

    def _send_prompt(self, header: dict, ids: Sequence[int], images: Sequence["WireImage"], rows: set) -> None:
        blobs = [torch.tensor(list(ids), dtype=torch.int32).numpy()]
        if images:
            entries = []
            for image in images:
                send = image.key in rows
                entries.append({"key": image.key, "slots": [int(s) for s in image.slots], "rows": send})
                if send:
                    blobs.append(_bf16_bytes(image.rows()))
            header = {**header, "images": entries}
        send_msg(self.sock, header, blobs)

    def _prompt(self, header: dict, ids, images, on_progress, finish):
        """Send a prompt (``prefill`` / ``prefill_push``) and read the replies to ``finish(header, blobs)``; a
        ``need`` reply sends it again with those images' rows. :attr:`sent_images` then holds the keys whose rows
        went to the server (the caller need not send them again)."""
        images = list(images or ())
        rows = {image.key for image in images if image.send_rows}
        times = self.times = {"send": 0.0, "need_rounds": 0}
        while True:
            t0 = time.perf_counter()
            self._send_prompt(header, ids, images, rows)
            t_sent = time.perf_counter()
            times["send"] += t_sent - t0
            first = None
            while True:
                msg: dict = {}
                reply, blobs = self._reply(msg)
                if first is None:
                    first = time.perf_counter() - msg["transfer"]
                    times["first_reply"] = first - t_sent
                if "need" in reply:
                    missing = set(reply["need"]) - {image.key for image in images}
                    if missing or set(reply["need"]) <= rows:
                        raise PrefillServerError(f"prefill server {self.address} asked for the rows of {reply['need']}")
                    rows |= set(reply["need"])
                    times["need_rounds"] += 1
                    break
                if "progress" in reply:
                    if on_progress is not None:
                        on_progress(int(reply["progress"]))
                    continue
                self.sent_images = rows
                times["replies"] = time.perf_counter() - msg["transfer"] - first
                times["recv"] = msg["transfer"]
                times["recv_bytes"] = msg["bytes"]
                return finish(reply, blobs)

    def prefill(
        self,
        ids: Sequence[int],
        on_progress: Optional[Callable[[int], None]] = None,
        have: int = 0,
        cache: bool = True,
        images: Sequence["WireImage"] = (),
    ) -> dict:
        """KV entries of ``ids`` (the whole prompt from position 0) from position ``have`` on: the client already
        holds those of the first ``have`` tokens. ``on_progress(end)`` as the server enqueues each chunk. ``images``:
        every image whose rows ``ids`` use (module docstring)."""
        header = {"op": "prefill", "have": int(have), "cache": bool(cache)}
        return self._prompt(header, ids, images, on_progress, _states)

    def prefill_push(
        self,
        ids: Sequence[int],
        targets: dict,
        on_progress: Optional[Callable[[int], None]] = None,
        have: int = 0,
        cache: bool = True,
        images: Sequence["WireImage"] = (),
    ) -> dict:
        """As :meth:`prefill`, but the server writes the entries into the device memory ``targets`` describes
        (``remote_commit_targets`` of the session) and replies with what it wrote: ``{"seq_len", "base", "cached",
        "seconds", "queued", "bytes", "pages", "push_seconds"}``."""
        header = {"op": "prefill_push", "have": int(have), "cache": bool(cache), "targets": targets}

        def finish(reply, _blobs):
            reply.pop("blobs", None)
            return reply

        return self._prompt(header, ids, images, on_progress, finish)

    def prefill_warm(self, ids: Sequence[int]) -> dict:
        """Prefill the text prompt ``ids`` into the server's prefix cache only (``prefill_warm``, module docstring):
        ``{"seq_len", "done", "cached", "seconds", "queued"}``."""

        def finish(reply, _blobs):
            reply.pop("blobs", None)
            return reply

        return self._prompt({"op": "prefill_warm"}, ids, (), None, finish)

    def close(self) -> None:
        self.sock.close()
