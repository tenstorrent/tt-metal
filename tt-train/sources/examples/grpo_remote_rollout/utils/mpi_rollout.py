# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-rank rollout RPC (``MPIRolloutServer`` / ``MPIRolloutClient``) on top
of :class:`~utils.weight_bridge.WeightBridge`.

Gotchas:
- Both constructors call ``bridge.connect()``, a two-rank handshake that blocks
  until the peer constructs its own object -- so both ranks must construct.
- Server-side callback (``generate_fn`` / ``on_weights_received``) failures are
  not caught: the server dies, the client's blocking recv never returns, MPI
  aborts the world. That is the intended failure mode; do not wrap in try/except.
- Rollout RPC uses MPI tags 10..13, disjoint from WeightBridge (1..6), so both
  protocols share one MPI context without crosstalk.
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

OP_GENERATE: int = 22001
OP_SHUTDOWN: int = 22002
OP_REQUEST_TRANSFER: int = 22003

_INFER_REQ_HDR_TAG: int = 22010
_INFER_REQ_BODY_TAG: int = 22011
_INFER_RES_HDR_TAG: int = 22012
_INFER_RES_BODY_TAG: int = 22013

# 16-byte header: op (u32), body_len (u64), reserved (u32), little-endian.
_HEADER_FMT: str = "<IQI"
_HEADER_LEN: int = struct.calcsize(_HEADER_FMT)


GenerateFn = Callable[..., List[List[int]]]
OnWeightsReceivedFn = Callable[[List[dict]], None]


class MPIRolloutServer:
    """Blocking inference RPC server, runs on the tt-transformers rank."""

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

        # connect() blocks until the peer constructs its MPIRolloutClient.
        self._bridge: WeightBridge = bridge
        self._bridge.connect()

    def serve_forever(self) -> None:
        """Block accepting requests until the peer sends ``OP_SHUTDOWN``."""
        while True:
            hdr = _mpi_recv_bytes(_HEADER_LEN, self.peer_rank, _INFER_REQ_HDR_TAG)
            op, body_len, _reserved = struct.unpack(_HEADER_FMT, hdr)

            if op == OP_SHUTDOWN:
                return

            if op == OP_REQUEST_TRANSFER:
                # Peer is inside its matching client.send_weights(); the bridge
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

            # No try/except: generate_fn raising kills this process and hangs the
            # client's response recv -> MPI aborts the world (intended).
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


class MPIRolloutClient:
    """Blocking inference RPC client, runs on the ttml rank.

    The constructor's ``bridge.connect()`` blocks until the peer constructs its
    :class:`MPIRolloutServer`.
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

        Sends OP_REQUEST_TRANSFER, then bridge.send_weights, then bridge.barrier
        (matching the server-side barrier in serve_forever).
        """
        hdr = struct.pack(_HEADER_FMT, OP_REQUEST_TRANSFER, 0, 0)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
        self._bridge.send_weights(hf_dict)
        self._bridge.barrier()

    def shutdown(self) -> None:
        """Send ``OP_SHUTDOWN`` to the server; no response is expected."""
        hdr = struct.pack(_HEADER_FMT, OP_SHUTDOWN, 0, 0)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
