# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-rank rollout RPC: remote generate + weight push over MPI.

This module hosts the two user-facing RPC endpoints that sit on top of the
lower-level :class:`~utils.weight_bridge.WeightBridge` weight transport:

- :class:`MPIRolloutServer` -- runs on the ttt rank. Owns an internal
  ``WeightBridge`` (constructed with ``role="ttt"``) and the
  ``generate_fn`` + ``on_weights_received`` callbacks supplied to its
  constructor. Calling :meth:`MPIRolloutServer.serve_forever` blocks
  until the peer sends ``OP_SHUTDOWN``. Op dispatch:

    * ``OP_GENERATE`` -> ``generate_fn(prompts, ...)``; result shipped
      back as JSON.
    * ``OP_REQUEST_TRANSFER`` -> internal ``bridge.receive_weights()``
      to receive the HF dict, then ``on_weights_received(hf_dict)``
      (typically ``Transformer.update_weights``), then a post-transfer
      barrier.
    * ``OP_SHUTDOWN`` -> return.

- :class:`MPIRolloutClient` -- runs on the ttml rank. Owns an internal
  ``WeightBridge`` (constructed with ``role="ttml"``). Public methods:

    * :meth:`MPIRolloutClient.remote_generate` -- run inference on the
      ttt rank, blocks for the response.
    * :meth:`MPIRolloutClient.send_weights` -- single-call weight
      update: posts ``OP_REQUEST_TRANSFER``, runs the
      ``bridge.send_weights(hf_dict)`` transport, and waits on the
      post-transfer barrier. The user side only needs this one call.
    * :meth:`MPIRolloutClient.shutdown` -- post ``OP_SHUTDOWN``.

Constructor handshake
=====================

Both ``MPIRolloutServer`` and ``MPIRolloutClient`` call
``bridge.connect()`` in their own constructors. That call is a two-rank
handshake (plus any transport setup the concrete bridge needs) which
blocks until the peer constructs its own object. As a consequence:

- You **cannot** get a usable client/server instance back from
  ``__init__`` until the peer also constructs theirs.
- Any code that runs after construction is past the handshake; the
  classic "client tries to RPC before server is ready" deadlock is
  impossible.

Best practice is still to construct both objects as early as possible
in each rank's flow (immediately after the mesh device is open) so the
handshake completes quickly and the two ranks then proceed in parallel.

Failure semantics
=================

If ``generate_fn`` or ``on_weights_received`` raises, the exception
propagates out of :meth:`MPIRolloutServer.serve_forever`; the server
process dies and MPI tears down the world. The client, blocked on
``recv_bytes`` for the response, will then never hear back. **Both
processes die together on any server-side callback failure**; do not
wrap these callbacks in a ``try`` / ``except`` inside the loop.

Tag namespace
=============

- ``WeightBridge`` (see :mod:`utils.weight_bridge`): MPI tags 1..4.
- Rollout RPC (this module): MPI tags 10..13 (request header/body,
  response header/body).

Disjoint tag namespaces let both protocols share the same MPI
distributed context simultaneously without crosstalk.

Wire format -- rollout RPC
==========================

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

``REQUEST_TRANSFER`` request: no body. No response. The server runs its
weight-receive + ``on_weights_received`` callback + barrier sequence
before returning to the dispatch loop.

``SHUTDOWN`` request: no body. No response.
"""

from __future__ import annotations

import json
import struct
from typing import Callable, List, Optional

from ttnn._ttnn.multi_device import recv_bytes as _mpi_recv_bytes
from ttnn._ttnn.multi_device import send_bytes as _mpi_send_bytes

from .weight_bridge import (
    TTML_RANK,
    TTT_RANK,
    WeightBridge,
    _require_distributed_context,
)

# ---------------------------------------------------------------------------
# Public constants -- rollout RPC op codes
# ---------------------------------------------------------------------------

OP_GENERATE: int = 1
OP_SHUTDOWN: int = 2
OP_REQUEST_TRANSFER: int = 3


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Rollout RPC tag namespace (10..13). Disjoint from WeightBridge (1..6).
_INFER_REQ_HDR_TAG: int = 10
_INFER_REQ_BODY_TAG: int = 11
_INFER_RES_HDR_TAG: int = 12
_INFER_RES_BODY_TAG: int = 13

# 16-byte request/response header: op (u32), body_len (u64), reserved (u32),
# little-endian. The reserved word is currently always zero; kept so the
# header is 8-byte aligned and future versions can grow without changing
# the layout.
_HEADER_FMT: str = "<IQI"
_HEADER_LEN: int = struct.calcsize(_HEADER_FMT)


GenerateFn = Callable[..., List[List[int]]]
OnWeightsReceivedFn = Callable[[List[dict]], None]


# ---------------------------------------------------------------------------
# MPIRolloutServer -- ttt rank
# ---------------------------------------------------------------------------


class MPIRolloutServer:
    """Blocking inference RPC server, runs on the tt-transformers rank.

    Drives an injected :class:`WeightBridge` (built by the caller via
    ``init_receiver``) for the OP_REQUEST_TRANSFER side effect. The
    constructor calls ``bridge.connect()`` -- it blocks until the peer
    constructs its :class:`MPIRolloutClient`, so returning from ``__init__``
    guarantees both ranks are past the handshake.

    :meth:`serve_forever` then blocks reading op headers and
    dispatches:

    - ``OP_GENERATE``: calls ``generate_fn(prompts, **kwargs)`` and
      ships the result back as JSON.
    - ``OP_REQUEST_TRANSFER``: runs ``bridge.receive_weights()`` to
      receive one weight dict per submesh, calls
      ``on_weights_received(list_of_dicts)`` (if supplied), and then runs
      ``bridge.barrier()`` to fence the transfer.
    - ``OP_SHUTDOWN``: returns.

    If ``generate_fn`` or ``on_weights_received`` raises, the exception
    is allowed to propagate out of :meth:`serve_forever`; the server
    process dies and MPI tears down the world. There is no per-request
    error response.
    """

    def __init__(
        self,
        *,
        peer_rank: int,
        bridge: WeightBridge,
        generate_fn: GenerateFn,
        on_weights_received: Optional[OnWeightsReceivedFn] = None,
    ) -> None:
        local_rank = _require_distributed_context("MPIRolloutServer")
        if local_rank != TTT_RANK:
            raise RuntimeError(f"MPIRolloutServer must run on TTT_RANK={TTT_RANK} (got local rank {local_rank}).")
        if int(peer_rank) != TTML_RANK:
            raise RuntimeError(f"MPIRolloutServer: peer_rank must be TTML_RANK={TTML_RANK} (got {peer_rank}).")

        self.peer_rank: int = int(peer_rank)
        self._generate_fn: GenerateFn = generate_fn
        self._on_weights_received: Optional[OnWeightsReceivedFn] = on_weights_received

        # The caller builds the concrete WeightBridge (via init_receiver)
        # and injects it; the server only drives it.
        # connect() performs the bridge handshake -- it blocks until the peer
        # constructs its MPIRolloutClient, so returning from __init__ guarantees
        # both ranks are past the handshake.
        self._bridge: WeightBridge = bridge
        self._bridge.connect()

    def serve_forever(self) -> None:
        """Block accepting requests until the peer sends ``OP_SHUTDOWN``.

        Uses the ``generate_fn`` and ``on_weights_received`` callbacks
        supplied to :meth:`__init__`; no further callbacks are passed
        in here.
        """
        while True:
            hdr = _mpi_recv_bytes(_HEADER_LEN, self.peer_rank, _INFER_REQ_HDR_TAG)
            op, body_len, _reserved = struct.unpack(_HEADER_FMT, hdr)

            if op == OP_SHUTDOWN:
                # SHUTDOWN carries no body and gets no response.
                return

            if op == OP_REQUEST_TRANSFER:
                # REQUEST_TRANSFER carries no body and gets no response.
                # The peer is now inside its matching
                # client.send_weights(hf_dict) call; the bridge
                # exchange below pairs up with it on tags 1..4.
                per_target = self._bridge.receive_weights()  # list[dict], one per receiver submesh
                assert per_target is not None, "WeightBridge.receive_weights must return a list[dict]"
                if self._on_weights_received is not None:
                    self._on_weights_received(per_target)
                self._bridge.barrier()
                continue

            if op != OP_GENERATE:
                raise RuntimeError(f"MPIRolloutServer: unknown op {op}")

            body_bytes = _mpi_recv_bytes(int(body_len), self.peer_rank, _INFER_REQ_BODY_TAG) if body_len else b""
            req = json.loads(body_bytes.decode("utf-8"))

            # No try/except: if generate_fn raises, this whole process dies
            # and the client's blocking response recv hangs -> MPI aborts
            # the world. That is the intended failure mode.
            completions = self._generate_fn(
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
# MPIRolloutClient -- ttml rank
# ---------------------------------------------------------------------------


class MPIRolloutClient:
    """Blocking inference RPC client, runs on the ttml rank.

    Drives an injected :class:`WeightBridge` (built by the caller via
    ``init_sender``) so the user only has to call :meth:`send_weights`
    once to push a fresh weight dict to the peer -- the op-code message
    and the bridge transport are bundled together inside that single method.

    The constructor performs the bridge handshake -- it blocks until
    the peer constructs its :class:`MPIRolloutServer`. Returning from
    ``__init__`` therefore guarantees that both ranks are past the
    handshake and the bridge transport is ready.

    No try/except is used around blocking receives: if the server
    process dies mid-request, the client's blocking ``recv_bytes`` will
    not return and MPI will eventually abort the world. That is the
    intended failure mode.
    """

    def __init__(
        self,
        *,
        peer_rank: int,
        bridge: WeightBridge,
    ) -> None:
        local_rank = _require_distributed_context("MPIRolloutClient")
        if local_rank != TTML_RANK:
            raise RuntimeError(f"MPIRolloutClient must run on TTML_RANK={TTML_RANK} (got local rank {local_rank}).")
        if int(peer_rank) != TTT_RANK:
            raise RuntimeError(f"MPIRolloutClient: peer_rank must be TTT_RANK={TTT_RANK} (got {peer_rank}).")

        self.peer_rank: int = int(peer_rank)

        # The caller builds the concrete WeightBridge (via init_sender) and injects it.
        self._bridge: WeightBridge = bridge
        self._bridge.connect()

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

    def send_weights(self, hf_dict: dict[str, "ttnn.Tensor"]) -> None:
        """Push a fresh HF-keyed weight dict to the ttt rank in one call.

        Internally:

        1. Send the ``OP_REQUEST_TRANSFER`` header (tag 10, no body).
        2. Run ``bridge.send_weights(hf_dict)`` -- ships the manifest
           plus each tensor over the bridge transport (tags 1, 2 +
           per-weight blobs on the host bridge).
        3. Run ``bridge.barrier()`` -- post-transfer fence matching the
           server-side barrier in ``serve_forever``.

        The user never has to interleave a separate request op with the
        bridge call; the whole exchange is bundled here.
        """
        hdr = struct.pack(_HEADER_FMT, OP_REQUEST_TRANSFER, 0, 0)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
        self._bridge.send_weights(hf_dict)
        self._bridge.barrier()

    def shutdown(self) -> None:
        """Send ``OP_SHUTDOWN`` to the server; no response is expected."""
        hdr = struct.pack(_HEADER_FMT, OP_SHUTDOWN, 0, 0)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
