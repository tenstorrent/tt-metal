# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-rank weight transport: a small ``WeightBridge`` abstraction.

``WeightBridge`` is a pure interface (ABC) with four methods -- ``connect``,
``send_weights``, ``receive_weights`` and ``barrier``. One concrete transport
implements it:

- :class:`HostWeightBridge` -- moves each weight to host (``torch.save`` over
  MPI) and re-materialises it replicated onto every receiver target (submesh).
  No fabric; the cost is a host/PCIe round-trip of ~model size per target.

Instances are built with the ``init_sender`` / ``init_receiver`` classmethod
factories (there is no public role-switching constructor).

Weight-dict format
==================

``dict[str, ttnn.Tensor]`` -- HuggingFace safetensors-style keys mapping to
weight tensors. Every tensor is ``bfloat16``, ``TILE_LAYOUT``,
``DRAM_MEMORY_CONFIG`` (interleaved), 4D ``(1, 1, rows, cols)`` and *fully
replicated* across its local mesh (every mesh axis ``PlacementReplicate``).
``send_weights`` requires this on the sender's ``[1, N]`` mesh.
``receive_weights`` returns a ``list[dict]`` -- one dict per receiver target
(submesh) -- each tensor satisfying the same contract on that target. Keys
are iterated in sorted order. This matches ``LlamaCompositeKV.export_to_hf_dict``
output and the ``Transformer.update_weights`` input contract.

Tag namespace
=============

- ``WeightBridge`` (this module): MPI tags 1..6 (manifest length/body, two
  handshake directions, weight-blob length/body for the host transport).
- Rollout RPC (:mod:`utils.mpi_rollout`): MPI tags 10..13.

Disjoint tag namespaces let both protocols share the same MPI distributed
context simultaneously without crosstalk.
"""

from __future__ import annotations

import io
import json
import struct
from abc import ABC, abstractmethod
from typing import List, Optional

import ttnn
from ttnn._ttnn.multi_device import recv_bytes as _mpi_recv_bytes
from ttnn._ttnn.multi_device import send_bytes as _mpi_send_bytes


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

SENDER_RANK: int = 0
RECEIVER_RANK: int = 1

# Back-compat aliases (used by utils.mpi_rollout and the tests).
TTML_RANK: int = SENDER_RANK
TTT_RANK: int = RECEIVER_RANK


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_ROLE_SENDER: str = "sender"
_ROLE_RECEIVER: str = "receiver"

# MPI tag namespace (1..6).
_MANIFEST_LEN_TAG: int = 1
_MANIFEST_BODY_TAG: int = 2
# Per-direction handshake tags so a send cannot self-match the same rank's recv.
_HANDSHAKE_TAG_FROM_SENDER: int = 3
_HANDSHAKE_TAG_FROM_RECEIVER: int = 4
# Host transport: per-weight length-prefixed blobs.
_WEIGHT_LEN_TAG: int = 5
_WEIGHT_BLOB_TAG: int = 6

_HANDSHAKE_PAYLOAD: bytes = b"ready"


# ---------------------------------------------------------------------------
# Module-level shared helpers (both impls use these; the ABC stays pure)
# ---------------------------------------------------------------------------


def _require_distributed_context(owner: str) -> int:
    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError(
            f"{owner}: ttnn distributed context is not initialized. "
            "Call ttnn.init_distributed_context() (receiver) or "
            "AutoContext.get_instance().initialize_distributed_context(*sys.argv) "
            "(sender) before constructing the bridge."
        )
    return int(ttnn.distributed_context_get_rank())


def _check_role_rank(owner: str, role: str, local_rank: int) -> None:
    if role not in (_ROLE_SENDER, _ROLE_RECEIVER):
        raise ValueError(f"{owner}: role must be {_ROLE_SENDER!r} or {_ROLE_RECEIVER!r}, got {role!r}")
    expected = SENDER_RANK if role == _ROLE_SENDER else RECEIVER_RANK
    if local_rank != expected:
        raise RuntimeError(f"{owner}: role={role!r} but local MPI rank is {local_rank} (expected {expected}).")


def _handshake(role: str, peer_rank: int) -> None:
    """Two-rank barrier: each side eager-sends, then blocks on recv."""
    if role == _ROLE_SENDER:
        _mpi_send_bytes(_HANDSHAKE_PAYLOAD, peer_rank, _HANDSHAKE_TAG_FROM_SENDER)
        _mpi_recv_bytes(len(_HANDSHAKE_PAYLOAD), peer_rank, _HANDSHAKE_TAG_FROM_RECEIVER)
    else:
        _mpi_send_bytes(_HANDSHAKE_PAYLOAD, peer_rank, _HANDSHAKE_TAG_FROM_RECEIVER)
        _mpi_recv_bytes(len(_HANDSHAKE_PAYLOAD), peer_rank, _HANDSHAKE_TAG_FROM_SENDER)


def _shape_to_list(shape) -> List[int]:
    return [int(d) for d in shape]


def _is_fully_replicated(tensor: "ttnn.Tensor") -> bool:
    placements = tensor.tensor_topology().placements()
    return all(isinstance(p, ttnn.PlacementReplicate) for p in placements)


def _validate_source_tensor(key: str, tensor: "ttnn.Tensor") -> None:
    """Enforce the ``Transformer.update_weights`` input contract before the wire."""
    if tensor.dtype != ttnn.bfloat16:
        raise ValueError(
            f"WeightBridge: tensor {key!r} has dtype={tensor.dtype}, expected ttnn.bfloat16 "
            "(Transformer.update_weights requires bfloat16)."
        )
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"WeightBridge: tensor {key!r} has layout={tensor.layout}, expected ttnn.TILE_LAYOUT.")
    if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError(
            f"WeightBridge: tensor {key!r} is not in DRAM_MEMORY_CONFIG "
            f"(memory_config={tensor.memory_config()}). Move the parameter to DRAM-interleaved before "
            "sending; sharded L1 is not supported."
        )
    if not _is_fully_replicated(tensor):
        placements = tensor.tensor_topology().placements()
        raise ValueError(
            f"WeightBridge: tensor {key!r} is not fully replicated across the mesh (placements={placements}). "
            "The bridge only supports replicated weights; on the sender this typically means TP/CP is enabled -- "
            "use DDP-only (line topology like [1, N])."
        )


def _send_manifest(peer_rank: int, weights: dict, keys: List[str]) -> None:
    """Send a length-prefixed JSON manifest (per-key shape/dtype/layout)."""
    manifest = {
        "version": 1,
        "entries": [
            {
                "key": k,
                "shape": _shape_to_list(weights[k].shape),
                "dtype": weights[k].dtype.name,
                "layout": weights[k].layout.name,
            }
            for k in keys
        ],
    }
    body = json.dumps(manifest).encode("utf-8")
    _mpi_send_bytes(struct.pack("<Q", len(body)), peer_rank, _MANIFEST_LEN_TAG)
    _mpi_send_bytes(body, peer_rank, _MANIFEST_BODY_TAG)


def _recv_manifest(peer_rank: int) -> dict:
    header = _mpi_recv_bytes(8, peer_rank, _MANIFEST_LEN_TAG)
    (n,) = struct.unpack("<Q", header)
    body = _mpi_recv_bytes(int(n), peer_rank, _MANIFEST_BODY_TAG)
    return json.loads(body.decode("utf-8"))


def _device0_to_host(tensor: "ttnn.Tensor"):
    """Bring device 0's (replicated) copy of ``tensor`` to host as torch."""
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _replicate_from_host(host_tensor, device: "ttnn.MeshDevice") -> "ttnn.Tensor":
    """Materialise ``host_tensor`` replicated across ``device`` (bf16/TILE/DRAM)."""
    return ttnn.from_torch(
        host_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
    )


def _torch_save_bytes(t) -> bytes:
    import torch  # noqa: F401 -- torch is only needed on the host transport path

    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _torch_load_bytes(blob: bytes):
    import torch

    return torch.load(io.BytesIO(blob), weights_only=True)


# ---------------------------------------------------------------------------
# WeightBridge -- pure interface
# ---------------------------------------------------------------------------


class WeightBridge(ABC):
    """Cross-rank transport for a replicated HF-keyed weight dict.

    Weight-dict format (see the module docstring): ``dict[str, ttnn.Tensor]``,
    each tensor bf16 / TILE / DRAM-interleaved / 4D ``(1, 1, *, *)`` / fully
    replicated across its mesh. ``send_weights`` requires this on the sender
    mesh; ``receive_weights`` returns a ``list[dict]`` (one per receiver
    submesh), each satisfying the contract on that submesh.

    Build instances with the concrete impls' ``init_sender`` / ``init_receiver``
    classmethods. Lifecycle: construct -> ``connect()`` (both ranks) ->
    ``send_weights`` (sender) / ``receive_weights`` (receiver), paired and the
    same number of times -> ``barrier()``.
    """

    @abstractmethod
    def connect(self) -> None:
        """Handshake with the peer and open any transport. Both ranks call once."""

    @abstractmethod
    def send_weights(self, weights: dict[str, "ttnn.Tensor"]) -> None:
        """Sender only: send the replicated weight dict to the receiver."""

    @abstractmethod
    def receive_weights(self) -> List[dict[str, "ttnn.Tensor"]]:
        """Receiver only: return one weight dict per submesh target."""

    @abstractmethod
    def barrier(self) -> None:
        """Two-rank fence so the sender keeps its source tensors alive until drained."""


# ---------------------------------------------------------------------------
# HostWeightBridge -- MPI byte transfer, no fabric
# ---------------------------------------------------------------------------


class HostWeightBridge(WeightBridge):
    """Move weights via host: ``torch.save`` over MPI, then re-upload to each target.

    Always works (no fabric); the cost is a host/PCIe round-trip of ~model size
    per receiver target. The receiver's ``mesh`` argument is accepted for a
    uniform factory signature but unused -- host uploads straight to each submesh.
    """

    def __init__(
        self,
        *,
        role: str,
        peer_rank: int,
        mesh: Optional["ttnn.MeshDevice"] = None,
        submeshes: Optional[List["ttnn.MeshDevice"]] = None,
    ) -> None:
        local_rank = _require_distributed_context("HostWeightBridge")
        _check_role_rank("HostWeightBridge", role, local_rank)
        self._role = role
        self._peer_rank = int(peer_rank)
        if role == _ROLE_SENDER:
            if mesh is None:
                raise ValueError("HostWeightBridge.init_sender requires a mesh (the [1, N] sender mesh).")
            self._mesh = mesh
            self._targets: Optional[List[ttnn.MeshDevice]] = None
        else:
            if not submeshes:
                raise ValueError("HostWeightBridge.init_receiver requires a non-empty submeshes list.")
            self._mesh = mesh  # accepted for signature parity; unused by the host path
            self._targets = list(submeshes)

    @classmethod
    def init_sender(cls, *, mesh: "ttnn.MeshDevice", peer_rank: int) -> "HostWeightBridge":
        return cls(role=_ROLE_SENDER, peer_rank=peer_rank, mesh=mesh)

    @classmethod
    def init_receiver(
        cls, *, mesh: "ttnn.MeshDevice", peer_rank: int, submeshes: List["ttnn.MeshDevice"]
    ) -> "HostWeightBridge":
        return cls(role=_ROLE_RECEIVER, peer_rank=peer_rank, mesh=mesh, submeshes=submeshes)

    def connect(self) -> None:
        _handshake(self._role, self._peer_rank)

    def send_weights(self, weights: dict[str, "ttnn.Tensor"]) -> None:
        if self._role != _ROLE_SENDER:
            raise RuntimeError("HostWeightBridge.send_weights called on a receiver.")
        keys = sorted(weights.keys())
        for k in keys:
            _validate_source_tensor(k, weights[k])
        _send_manifest(self._peer_rank, weights, keys)
        for k in keys:
            blob = _torch_save_bytes(_device0_to_host(weights[k]))
            _mpi_send_bytes(struct.pack("<Q", len(blob)), self._peer_rank, _WEIGHT_LEN_TAG)
            _mpi_send_bytes(blob, self._peer_rank, _WEIGHT_BLOB_TAG)

    def receive_weights(self) -> List[dict[str, "ttnn.Tensor"]]:
        if self._role != _ROLE_RECEIVER:
            raise RuntimeError("HostWeightBridge.receive_weights called on a sender.")
        manifest = _recv_manifest(self._peer_rank)
        dicts: List[dict[str, ttnn.Tensor]] = [{} for _ in self._targets]
        for entry in manifest["entries"]:
            (blob_len,) = struct.unpack("<Q", _mpi_recv_bytes(8, self._peer_rank, _WEIGHT_LEN_TAG))
            host_tensor = _torch_load_bytes(_mpi_recv_bytes(int(blob_len), self._peer_rank, _WEIGHT_BLOB_TAG))
            for i, target in enumerate(self._targets):
                dicts[i][entry["key"]] = _replicate_from_host(host_tensor, target)
        return dicts

    def barrier(self) -> None:
        _handshake(self._role, self._peer_rank)
