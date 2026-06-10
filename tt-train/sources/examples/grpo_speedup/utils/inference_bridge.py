# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Two-rank inference RPC over MPI for the GRPO ttml + ttt split.

After ``WeightBridge.transfer_weights`` lands a fresh weight dict on the
tt-transformers rank, the ttt rank typically wants to keep serving
``generate()`` calls from the ttml rank rather than running a single
one-shot completion. This module provides:

- :class:`TttInferenceServer` -- runs on the ttt rank. Calls
  ``serve_forever(generate_fn, on_transfer=...)`` and blocks waiting for
  requests. Each ``OP_GENERATE`` message turns into one
  ``generate_fn(prompts, ...)`` invocation; the resulting token-id lists
  are sent back. Each ``OP_REQUEST_TRANSFER`` message invokes the
  caller-supplied ``on_transfer`` callback (typically a
  ``WeightBridge.transfer_weights`` + ``Transformer.update_weights``
  sequence) with no response. An ``OP_SHUTDOWN`` message returns from
  ``serve_forever``.

- :class:`TttInferenceClient` -- runs on the ttml rank.
  ``remote_generate`` posts an inference request and blocks for the
  response. ``request_transfer`` posts an ``OP_REQUEST_TRANSFER`` and
  returns (no response expected); pair it with the matching
  ``WeightBridge.transfer_weights(hf_dict)`` on the next line so the two
  ranks rendezvous inside the transfer. ``shutdown`` posts an
  ``OP_SHUTDOWN`` and returns.

Both sides exchange small JSON payloads (token-id lists + a few sampling
knobs) over the same blocking ``send_bytes`` / ``recv_bytes`` primitives
that :mod:`utils.weight_bridge` uses, so the inference protocol piggybacks
on the existing distributed context.

Failure semantics
=================

If ``generate_fn`` raises, the exception propagates out of
``serve_forever`` and the server process dies. The client, blocked on
``recv_bytes`` for the response, will then never hear back -- under
``tt-run`` / Open MPI this triggers the whole world to abort. **Both
processes die together on any server-side generate failure**; do not
wrap ``generate_fn`` in a ``try`` / ``except`` inside the loop.

Tag namespace
=============

The inference protocol uses MPI tags 10-13 to stay clear of
:mod:`utils.weight_bridge`'s 1-6 namespace. Both modules can be active on
the same pair of ranks simultaneously without tag collisions.

Wire format
===========

Every request and response starts with a fixed 16-byte header
``(op: u32, body_len: u64, reserved: u32)`` in little-endian order. The
body, when present, is a UTF-8 JSON object whose shape depends on ``op``:

``GENERATE`` request::

    {
        "prompts":        [[int, int, ...], ...],
        "max_new_tokens": int,
        "temperature":    float,
        "top_p":          float,
        "seed":           int | null,
    }

``GENERATE`` response::

    {"completions": [[int, ...], ...]}

``REQUEST_TRANSFER`` request: no body. No response. The server invokes
its ``on_transfer`` callback before returning to the dispatch loop.

``SHUTDOWN`` request: no body. No response.
"""

from __future__ import annotations

import json
import struct
from typing import Callable, List, Optional

import ttnn
from ttnn._ttnn.multi_device import recv_bytes as _mpi_recv_bytes
from ttnn._ttnn.multi_device import send_bytes as _mpi_send_bytes


# ---------------------------------------------------------------------------
# Wire-format constants
# ---------------------------------------------------------------------------

OP_GENERATE: int = 1
OP_SHUTDOWN: int = 2
OP_REQUEST_TRANSFER: int = 3

# 16-byte request/response header: op (u32), body_len (u64), reserved (u32),
# little-endian. The reserved word is currently always zero; kept so the
# header is 8-byte aligned and future versions can grow without changing
# the layout.
_HEADER_FMT: str = "<IQI"
_HEADER_LEN: int = struct.calcsize(_HEADER_FMT)

# Reserve tags >= 10 so the inference protocol does not collide with the
# WeightBridge 1..6 tag namespace.
_INFER_REQ_HDR_TAG: int = 10
_INFER_REQ_BODY_TAG: int = 11
_INFER_RES_HDR_TAG: int = 12
_INFER_RES_BODY_TAG: int = 13


GenerateFn = Callable[..., List[List[int]]]


def _require_distributed_context(owner: str) -> int:
    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError(
            f"{owner}: ttnn distributed context is not initialized. "
            "Call ttnn.init_distributed_context() (ttt side) or "
            "AutoContext.get_instance().initialize_distributed_context(*sys.argv) "
            "(ttml side) before constructing the bridge."
        )
    return int(ttnn.distributed_context_get_rank())


# ---------------------------------------------------------------------------
# Server (ttt rank)
# ---------------------------------------------------------------------------


class TttInferenceServer:
    """Blocking inference RPC server, runs on the tt-transformers rank.

    The constructor is cheap. The loop is entered via
    :meth:`serve_forever`, which blocks until the peer sends
    ``OP_SHUTDOWN``. Each ``OP_GENERATE`` request dispatches a single
    ``generate_fn(prompts, **kwargs)`` call and the resulting token-id
    lists are shipped back. Each ``OP_REQUEST_TRANSFER`` request invokes
    the caller-supplied ``on_transfer`` callback (no response) and then
    returns to the dispatch loop.

    If ``generate_fn`` or ``on_transfer`` raises, the exception is
    allowed to propagate out of :meth:`serve_forever`; the server
    process dies and MPI tears down the world. There is no per-request
    error response.
    """

    def __init__(self, *, peer_rank: int) -> None:
        local_rank = _require_distributed_context("TttInferenceServer")
        if int(peer_rank) == local_rank:
            raise RuntimeError(
                f"TttInferenceServer: peer_rank={peer_rank} matches local rank "
                f"{local_rank}; client and server must be on different ranks."
            )
        self.peer_rank: int = int(peer_rank)

    def serve_forever(
        self,
        generate_fn: GenerateFn,
        on_transfer: Optional[Callable[[], None]] = None,
    ) -> None:
        """Block accepting requests until the peer sends ``OP_SHUTDOWN``.

        ``generate_fn`` is called with positional ``prompts`` (a
        ``List[List[int]]``) and the following keyword arguments
        forwarded from the request payload:

        - ``max_new_tokens`` (int, required)
        - ``temperature`` (float, default 0.0)
        - ``top_p`` (float, default 1.0)
        - ``seed`` (int | None, default None)

        The return value must be a list of ``len(prompts)`` lists of
        ``int``-coercible tokens.

        ``on_transfer`` is invoked (with no arguments) on every
        ``OP_REQUEST_TRANSFER`` message. Typical contents are a
        ``WeightBridge.transfer_weights`` receive followed by
        ``Transformer.update_weights``. If the peer ever sends
        ``OP_REQUEST_TRANSFER`` and ``on_transfer`` is ``None``, the
        server raises (and the process dies, per the failure contract).
        """
        while True:
            hdr = _mpi_recv_bytes(_HEADER_LEN, self.peer_rank, _INFER_REQ_HDR_TAG)
            op, body_len, _reserved = struct.unpack(_HEADER_FMT, hdr)

            if op == OP_SHUTDOWN:
                # SHUTDOWN carries no body and gets no response.
                return

            if op == OP_REQUEST_TRANSFER:
                # REQUEST_TRANSFER carries no body and gets no response.
                # The on_transfer callback is expected to rendezvous with
                # the peer on its own MPI tags (e.g. via WeightBridge).
                if on_transfer is None:
                    raise RuntimeError(
                        "TttInferenceServer: OP_REQUEST_TRANSFER received " "but no on_transfer callback was provided."
                    )
                on_transfer()
                continue

            if op != OP_GENERATE:
                raise RuntimeError(f"TttInferenceServer: unknown op {op}")

            body_bytes = _mpi_recv_bytes(int(body_len), self.peer_rank, _INFER_REQ_BODY_TAG) if body_len else b""
            req = json.loads(body_bytes.decode("utf-8"))

            # No try/except: if generate_fn raises, this whole process dies
            # and the client's blocking response recv hangs -> MPI aborts
            # the world. That is the intended failure mode.
            completions = generate_fn(
                req["prompts"],
                max_new_tokens=int(req["max_new_tokens"]),
                temperature=float(req.get("temperature", 0.0)),
                top_p=float(req.get("top_p", 1.0)),
                seed=(None if req.get("seed") is None else int(req["seed"])),
            )

            response_body = json.dumps({"completions": [[int(t) for t in c] for c in completions]}).encode("utf-8")
            response_hdr = struct.pack(_HEADER_FMT, 0, len(response_body), 0)
            _mpi_send_bytes(response_hdr, self.peer_rank, _INFER_RES_HDR_TAG)
            _mpi_send_bytes(response_body, self.peer_rank, _INFER_RES_BODY_TAG)


# ---------------------------------------------------------------------------
# Client (ttml rank)
# ---------------------------------------------------------------------------


class TttInferenceClient:
    """Blocking inference RPC client, runs on the ttml rank.

    Each :meth:`remote_generate` call posts one request and blocks until
    the response lands. Call :meth:`shutdown` exactly once when the
    client side is done driving the server; the server's
    :meth:`TttInferenceServer.serve_forever` returns immediately after.

    The client has no try/except around the wait: if the server process
    dies mid-request, the client's blocking ``recv_bytes`` will not return
    and MPI will eventually abort the world. That is the intended failure
    mode.
    """

    def __init__(self, *, peer_rank: int) -> None:
        local_rank = _require_distributed_context("TttInferenceClient")
        if int(peer_rank) == local_rank:
            raise RuntimeError(
                f"TttInferenceClient: peer_rank={peer_rank} matches local rank "
                f"{local_rank}; client and server must be on different ranks."
            )
        self.peer_rank: int = int(peer_rank)

    def remote_generate(
        self,
        prompts: List[List[int]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[List[int]]:
        """Run inference on the ttt rank and return token-id lists."""
        req_body = json.dumps(
            {
                "prompts": [[int(t) for t in p] for p in prompts],
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "seed": None if seed is None else int(seed),
            }
        ).encode("utf-8")
        req_hdr = struct.pack(_HEADER_FMT, OP_GENERATE, len(req_body), 0)
        _mpi_send_bytes(req_hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
        _mpi_send_bytes(req_body, self.peer_rank, _INFER_REQ_BODY_TAG)

        res_hdr = _mpi_recv_bytes(_HEADER_LEN, self.peer_rank, _INFER_RES_HDR_TAG)
        _op, body_len, _reserved = struct.unpack(_HEADER_FMT, res_hdr)
        res_body = _mpi_recv_bytes(int(body_len), self.peer_rank, _INFER_RES_BODY_TAG) if body_len else b""
        payload = json.loads(res_body.decode("utf-8")) if res_body else {}
        return [[int(t) for t in c] for c in payload.get("completions", [])]

    def request_transfer(self) -> None:
        """Tell the server to run its ``on_transfer`` callback.

        No response is expected. Pair this call with the matching
        peer-side action (typically ``bridge.transfer_weights(hf_dict)``)
        on the next line so the two ranks rendezvous inside that action;
        otherwise the server will block forever inside ``on_transfer``.
        """
        hdr = struct.pack(_HEADER_FMT, OP_REQUEST_TRANSFER, 0, 0)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)

    def shutdown(self) -> None:
        """Send ``OP_SHUTDOWN`` to the server; no response is expected."""
        hdr = struct.pack(_HEADER_FMT, OP_SHUTDOWN, 0, 0)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
