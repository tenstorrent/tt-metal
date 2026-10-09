# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-rank rollout RPC (``MPIRolloutServer`` / ``MPIRolloutClient``) on top
of :class:`~utils.weight_bridge.WeightBridge`.

Gotchas:
- Both constructors call ``bridge.connect()``, a two-rank handshake that blocks
  until the peer constructs its own object -- so both ranks must construct.
- Server-side sampler failures (``generate`` / ``update_weights``) are not
  caught: the server dies, the client's blocking recv never returns, MPI
  aborts the world. That is the intended failure mode; do not wrap in try/except.
- Rollout RPC tags are disjoint from WeightBridge tags (and from the fabric
  MPI tag range), so both protocols share one MPI context without crosstalk.
"""

from __future__ import annotations

import json
import struct
from typing import List, Tuple

from ttnn import distributed_context_recv_bytes as _mpi_recv_bytes
from ttnn import distributed_context_send_bytes as _mpi_send_bytes

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

_INFER_REQ_HDR_TAG: int = 22010
_INFER_REQ_BODY_TAG: int = 22011
_INFER_RES_HDR_TAG: int = 22012
_INFER_RES_BODY_TAG: int = 22013

# 16-byte header: op (u32), body_len (u64), reserved (u32), little-endian.
# OP_REQUEST_TRANSFER carries the new weight version in ``reserved``.
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
        try:
            self._serve()
        finally:
            self._sampler.close()

    def _serve(self) -> None:
        while True:
            hdr = _mpi_recv_bytes(_HEADER_LEN, self.peer_rank, _INFER_REQ_HDR_TAG)
            op, body_len, reserved = struct.unpack(_HEADER_FMT, hdr)

            if op == OP_SHUTDOWN:
                return

            if op == OP_REQUEST_TRANSFER:
                # Peer is inside its matching client.send_weights(); the bridge
                # exchange below pairs up with it on tags 1..4.
                per_target = self._bridge.receive_weights()  # list[dict], one per receiver submesh
                if per_target is None:
                    raise RuntimeError("WeightBridge.receive_weights must return a list[dict]")
                self._sampler.update_weights(per_target, version=reserved)
                self._bridge.barrier()
                continue

            if op != OP_GENERATE:
                raise RuntimeError(f"MPIRolloutServer: unknown op {op}")

            body_bytes = _mpi_recv_bytes(int(body_len), self.peer_rank, _INFER_REQ_BODY_TAG) if body_len else b""
            req = json.loads(body_bytes.decode("utf-8"))

            # No try/except: generate raising kills this process and hangs the
            # client's response recv -> MPI aborts the world (intended).
            batch = self._sampler.generate(req["prompts"])

            response_body = json.dumps(
                {
                    "completions": [[int(t) for t in c] for c in batch.completions],
                    "logprobs": [batch.logprobs[r, : len(c)].tolist() for r, c in enumerate(batch.completions)],
                    "weight_version": int(batch.weight_version),
                }
            ).encode("utf-8")
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

    def remote_generate(self, prompts: List[List[int]]) -> Tuple[List[List[int]], List[List[float]], int]:
        """Generate on the ttt rank. Returns ``(completions, per-token log-probs, weight_version)``.

        The ttt rank's sampler expands each prompt into its own number of completions, prompt-major.
        """
        req_body = json.dumps({"prompts": [[int(t) for t in p] for p in prompts]}).encode("utf-8")
        req_hdr = struct.pack(_HEADER_FMT, OP_GENERATE, len(req_body), 0)
        _mpi_send_bytes(req_hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
        _mpi_send_bytes(req_body, self.peer_rank, _INFER_REQ_BODY_TAG)

        res_hdr = _mpi_recv_bytes(_HEADER_LEN, self.peer_rank, _INFER_RES_HDR_TAG)
        _op, body_len, _reserved = struct.unpack(_HEADER_FMT, res_hdr)
        res_body = _mpi_recv_bytes(int(body_len), self.peer_rank, _INFER_RES_BODY_TAG) if body_len else b""
        payload = json.loads(res_body.decode("utf-8"))
        completions = [[int(t) for t in c] for c in payload["completions"]]
        logprobs = [[float(x) for x in lp] for lp in payload["logprobs"]]
        return completions, logprobs, int(payload["weight_version"])

    def send_weights(self, hf_dict: dict[str, "ttnn.Tensor"], *, version: int) -> None:
        """Push a fresh HF-keyed weight dict to the ttt rank as weight version ``version``.

        Sends OP_REQUEST_TRANSFER, then bridge.send_weights, then bridge.barrier
        (matching the server-side barrier in serve_forever).
        """
        version = int(version)
        if not 0 < version < 2**32:
            raise ValueError(f"weight version {version} must be in (0, 2**32) to fit the u32 header field")
        hdr = struct.pack(_HEADER_FMT, OP_REQUEST_TRANSFER, 0, version)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
        self._bridge.send_weights(hf_dict)
        self._bridge.barrier()

    def shutdown(self) -> None:
        """Send ``OP_SHUTDOWN`` to the server; no response is expected."""
        hdr = struct.pack(_HEADER_FMT, OP_SHUTDOWN, 0, 0)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
