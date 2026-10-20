# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-rank rollout RPC (``MPIRolloutServer`` / ``MPIRolloutClient``) on top
of :class:`~utils.weight_bridge.WeightBridge`.

Gotchas:
- Both constructors call ``bridge.connect()``, a two-rank handshake that blocks
  until the peer constructs its own object -- so both ranks must construct.
- Server-side sampler failures (``generate`` / ``update_weights``) are sent
  back as an ``OP_ERROR`` response and raised by the client. Any other failure
  inside an exchange leaves the peer blocked in a matching MPI call, so that
  rank prints the traceback and calls ``os._exit(1)``.
- ``MPIRolloutClient.shutdown()`` also runs at interpreter exit, so the server
  stops even if the client's owner never calls it.
- Rollout RPC tags are disjoint from WeightBridge tags (and from the fabric
  MPI tag range), so both protocols share one MPI context without crosstalk.
"""

from __future__ import annotations

import atexit
import json
import logging
import struct
import traceback
from typing import List, Tuple

from ttnn import distributed_context_recv_bytes as _mpi_recv_bytes
from ttnn import distributed_context_send_bytes as _mpi_send_bytes

from ..distributed import exit_on_error
from ..grpo_trainer import RolloutSampler
from .weight_bridge import (
    TTML_RANK,
    TTT_RANK,
    WeightBridge,
    _require_distributed_context,
)

OP_GENERATE: int = 22001
OP_SHUTDOWN: int = 22002
OP_REQUEST_TRANSFER: int = 22003
OP_ERROR: int = 22004

_INFER_REQ_HDR_TAG: int = 22010
_INFER_REQ_BODY_TAG: int = 22011
_INFER_RES_HDR_TAG: int = 22012
_INFER_RES_BODY_TAG: int = 22013

# 16-byte header: op (u32), body_len (u64), reserved (u32), little-endian.
# OP_REQUEST_TRANSFER carries the new weight version in ``reserved``.
# A response echoes the request op, or is OP_ERROR with the traceback as the body.
_HEADER_FMT: str = "<IQI"
_HEADER_LEN: int = struct.calcsize(_HEADER_FMT)


class MPIRolloutServer:
    """Blocking rollout RPC server, runs on the tt-transformers rank and serves ``sampler``."""

    def __init__(
        self,
        *,
        peer_rank: int,
        bridge: WeightBridge,
        sampler: RolloutSampler,
    ) -> None:
        local_rank = _require_distributed_context("MPIRolloutServer")
        if local_rank != TTT_RANK:
            raise RuntimeError(f"MPIRolloutServer must run on TTT_RANK={TTT_RANK} (got local rank {local_rank}).")
        if int(peer_rank) != TTML_RANK:
            raise RuntimeError(f"MPIRolloutServer: peer_rank must be TTML_RANK={TTML_RANK} (got {peer_rank}).")

        self.peer_rank: int = int(peer_rank)
        self._sampler: RolloutSampler = sampler

        # connect() blocks until the peer constructs its MPIRolloutClient.
        self._bridge: WeightBridge = bridge
        self._bridge.connect()

    def serve_forever(self) -> None:
        """Block accepting requests until the peer sends ``OP_SHUTDOWN``, then close the sampler."""
        with exit_on_error():
            self._serve()
        self._sampler.close()

    def _serve(self) -> None:
        while True:
            hdr = _mpi_recv_bytes(_HEADER_LEN, self.peer_rank, _INFER_REQ_HDR_TAG)
            op, body_len, reserved = struct.unpack(_HEADER_FMT, hdr)
            if op == OP_SHUTDOWN:
                logging.info("MPIRolloutServer: rank %d sent OP_SHUTDOWN, exiting", self.peer_rank)
                return
            elif op == OP_GENERATE:
                self._serve_generate(body_len)
            elif op == OP_REQUEST_TRANSFER:
                self._serve_transfer(version=reserved)
            else:
                raise RuntimeError(f"MPIRolloutServer: unknown op {op}")

    def _serve_generate(self, body_len: int) -> None:
        body_bytes = _mpi_recv_bytes(int(body_len), self.peer_rank, _INFER_REQ_BODY_TAG) if body_len else b""
        req = json.loads(body_bytes.decode("utf-8"))
        try:
            batch = self._sampler.generate(req["prompts"])
            body = json.dumps(
                {
                    "completions": [[int(t) for t in c] for c in batch.completions],
                    "logprobs": [batch.logprobs[r, : len(c)].tolist() for r, c in enumerate(batch.completions)],
                    "weight_version": int(batch.weight_version),
                }
            ).encode("utf-8")
            op = OP_GENERATE
        except Exception:
            op, body = OP_ERROR, traceback.format_exc().encode("utf-8")
        self._respond(op, body)

    def _serve_transfer(self, version: int) -> None:
        # Peer is inside its matching client.send_weights(); the bridge
        # exchange below pairs up with it on tags 1..4.
        per_target = self._bridge.receive_weights()  # list[dict], one per receiver submesh
        if per_target is None:
            raise RuntimeError("WeightBridge.receive_weights must return a list[dict]")
        try:
            self._sampler.update_weights(per_target, version=version)
            op, body = OP_REQUEST_TRANSFER, b""
        except Exception:
            op, body = OP_ERROR, traceback.format_exc().encode("utf-8")
        self._bridge.barrier()
        self._respond(op, body)

    def _respond(self, op: int, body: bytes) -> None:
        _mpi_send_bytes(struct.pack(_HEADER_FMT, op, len(body), 0), self.peer_rank, _INFER_RES_HDR_TAG)
        if body:
            _mpi_send_bytes(body, self.peer_rank, _INFER_RES_BODY_TAG)


class MPIRolloutClient:
    """Blocking inference RPC client, runs on the ttml rank.

    The constructor's ``bridge.connect()`` blocks until the peer constructs its
    :class:`MPIRolloutServer`. ``shutdown()`` also runs at interpreter exit.
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
        self._shut_down = False
        atexit.register(self.shutdown)

    def remote_generate(self, prompts: List[List[int]]) -> Tuple[List[List[int]], List[List[float]], int]:
        """Generate on the ttt rank. Returns ``(completions, per-token log-probs, weight_version)``.

        The ttt rank's sampler expands each prompt into its own number of completions, prompt-major.
        """
        req_body = json.dumps({"prompts": [[int(t) for t in p] for p in prompts]}).encode("utf-8")
        req_hdr = struct.pack(_HEADER_FMT, OP_GENERATE, len(req_body), 0)
        with exit_on_error():
            _mpi_send_bytes(req_hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
            _mpi_send_bytes(req_body, self.peer_rank, _INFER_REQ_BODY_TAG)
        payload = json.loads(self._recv_response().decode("utf-8"))
        completions = [[int(t) for t in c] for c in payload["completions"]]
        logprobs = [[float(x) for x in lp] for lp in payload["logprobs"]]
        return completions, logprobs, int(payload["weight_version"])

    def send_weights(self, hf_dict: dict[str, "ttnn.Tensor"], *, version: int) -> None:
        """Push a fresh HF-keyed weight dict to the ttt rank as weight version ``version``.

        Sends OP_REQUEST_TRANSFER, then bridge.send_weights, then bridge.barrier,
        then receives the response (matching _serve_transfer).
        """
        version = int(version)
        if not 0 <= version < 2**32:
            raise ValueError(f"weight version {version} must be in [0, 2**32) to fit the u32 header field")
        hdr = struct.pack(_HEADER_FMT, OP_REQUEST_TRANSFER, 0, version)
        with exit_on_error():
            _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
            self._bridge.send_weights(hf_dict)
            self._bridge.barrier()
        self._recv_response()

    def shutdown(self) -> None:
        """Send ``OP_SHUTDOWN`` to the server once; no response is expected."""
        if self._shut_down:
            return
        self._shut_down = True
        hdr = struct.pack(_HEADER_FMT, OP_SHUTDOWN, 0, 0)
        with exit_on_error():
            _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)

    def _recv_response(self) -> bytes:
        with exit_on_error():
            res_hdr = _mpi_recv_bytes(_HEADER_LEN, self.peer_rank, _INFER_RES_HDR_TAG)
            op, body_len, _reserved = struct.unpack(_HEADER_FMT, res_hdr)
            body = _mpi_recv_bytes(int(body_len), self.peer_rank, _INFER_RES_BODY_TAG) if body_len else b""
        if op == OP_ERROR:
            raise RuntimeError(f"rollout rank {self.peer_rank} failed:\n{body.decode('utf-8')}")
        return body
