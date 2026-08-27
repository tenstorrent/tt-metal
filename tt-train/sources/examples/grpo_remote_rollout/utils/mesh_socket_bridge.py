# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MeshSocket-based :class:`WeightBridge`: cross-rank fabric transport for a
replicated HF-keyed weight dict.

Both sides must open a ``[1, 1]`` mesh (single device). Transport goes over ONE
cross-rank :class:`ttnn.MeshSocket` with a single ``(0,0) -> (0,0)`` connection.
Tensor order + per-tensor spec is exchanged via a JSON manifest sent over MPI
(reuses :mod:`utils.weight_bridge` helpers); the fabric carries only the tensor
bytes.

Why one socket and not one-per-tensor: tt-metal PR #48757 shows that multiple
MeshSockets in parallel corrupt tensors from the third transfer onward, so we
stream every parameter sequentially through a single socket instead.

Requirements:
- ``ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)`` before either mesh
  opens (fabric is what MeshSocket routes over).
- The MGD used by tt-run must declare fabric ``connections`` between the two
  mesh instances. See ``configurations/1_1/mgd.textproto`` under the weight-
  bridge test dir for a working template.
"""

from __future__ import annotations

from typing import Any, List, Optional

import ttnn

from .weight_bridge import (
    RECEIVER_RANK,
    SENDER_RANK,
    WeightBridge,
    _ROLE_RECEIVER,
    _ROLE_SENDER,
    _check_role_rank,
    _handshake,
    _recv_manifest,
    _require_distributed_context,
    _send_manifest,
    _validate_source_tensor,
)

# Single (0,0)->(0,0) socket connection between the two [1,1] meshes.
_LOGICAL_CORE = (0, 0)

# FIFO capacity between the sender and receiver. MeshSocket streams tensor
# pages through it, so this only needs to hold a few pages; the socket is a
# credit-based ring and the send/recv side advance in lockstep at the tile
# granularity. 64 KiB comfortably fits several bf16 tiles.
_DEFAULT_FIFO_SIZE_BYTES = 64 * 1024


def _require_single_device_mesh(role: str, mesh: Any) -> None:
    """Enforce mesh.shape == (1, 1) so the single (0,0)->(0,0) socket is well-defined."""
    shape = tuple(int(d) for d in mesh.shape)
    if shape != (1, 1):
        raise ValueError(
            f"MeshSocketWeightBridge ({role}): mesh.shape must be (1, 1), got {shape}. "
            "This bridge only supports single-device meshes on each side; open a "
            "[1, 1] mesh with the matching MGD before constructing it."
        )


def _dtype_from_name(name: str) -> "ttnn.DataType":
    """Turn a ttnn.DataType name (e.g. ``\"BFLOAT16\"``) back into the enum member."""
    dt = getattr(ttnn.DataType, name, None)
    if dt is None:
        raise ValueError(f"MeshSocketWeightBridge: unknown ttnn dtype name {name!r}")
    return dt


def _layout_from_name(name: str) -> "ttnn.Layout":
    """Turn a ttnn.Layout name (e.g. ``\"TILE\"``) back into the enum member."""
    layout = getattr(ttnn.Layout, name, None)
    if layout is None:
        raise ValueError(f"MeshSocketWeightBridge: unknown ttnn layout name {name!r}")
    return layout


class MeshSocketWeightBridge(WeightBridge):
    """Ship a replicated HF-keyed weight dict over one cross-rank MeshSocket.

    Lifecycle mirrors the ABC exactly:
    ``__init__`` -> ``connect()`` (both ranks) -> ``send_weights`` (sender)
    / ``receive_weights`` (receiver) -> ``barrier()``. ``send_weights`` and
    ``receive_weights`` may be called any number of times between ``connect``
    and process teardown; the socket is created once by ``connect`` and reused.
    """

    def __init__(
        self,
        *,
        role: str,
        mesh: "ttnn.MeshDevice",
        submeshes: Optional[List["ttnn.MeshDevice"]] = None,
        fifo_size_bytes: int = _DEFAULT_FIFO_SIZE_BYTES,
    ) -> None:
        local_rank = _require_distributed_context("MeshSocketWeightBridge")
        _check_role_rank("MeshSocketWeightBridge", role, local_rank)
        _require_single_device_mesh(role, mesh)

        self._role = role
        self._mesh = mesh
        self._peer_rank = RECEIVER_RANK if role == _ROLE_SENDER else SENDER_RANK
        self._fifo_size_bytes = int(fifo_size_bytes)

        if role == _ROLE_SENDER:
            if submeshes is not None:
                raise ValueError("MeshSocketWeightBridge.init_sender: 'submeshes' must be None on the sender.")
            self._targets: Optional[List[ttnn.MeshDevice]] = None
        else:
            if not submeshes or len(submeshes) != 1:
                raise ValueError(
                    "MeshSocketWeightBridge.init_receiver: 'submeshes' must be a list of length 1 "
                    "(single-device receiver); "
                    f"got len={len(submeshes) if submeshes else 0}."
                )
            _require_single_device_mesh("receiver submesh", submeshes[0])
            self._targets = list(submeshes)

        # Cached MeshSocket, populated by connect().
        self._socket: Any = None

    @classmethod
    def init_sender(
        cls,
        *,
        mesh: "ttnn.MeshDevice",
        peer_rank: int,
        fifo_size_bytes: int = _DEFAULT_FIFO_SIZE_BYTES,
    ) -> "MeshSocketWeightBridge":
        if int(peer_rank) != RECEIVER_RANK:
            raise ValueError(
                f"MeshSocketWeightBridge.init_sender: peer_rank must be RECEIVER_RANK="
                f"{RECEIVER_RANK} (got {peer_rank})."
            )
        return cls(role=_ROLE_SENDER, mesh=mesh, fifo_size_bytes=fifo_size_bytes)

    @classmethod
    def init_receiver(
        cls,
        *,
        mesh: "ttnn.MeshDevice",
        peer_rank: int,
        submeshes: List["ttnn.MeshDevice"],
        fifo_size_bytes: int = _DEFAULT_FIFO_SIZE_BYTES,
    ) -> "MeshSocketWeightBridge":
        if int(peer_rank) != SENDER_RANK:
            raise ValueError(
                f"MeshSocketWeightBridge.init_receiver: peer_rank must be SENDER_RANK="
                f"{SENDER_RANK} (got {peer_rank})."
            )
        return cls(role=_ROLE_RECEIVER, mesh=mesh, submeshes=submeshes, fifo_size_bytes=fifo_size_bytes)

    # ---- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Two-rank handshake, then construct the shared MeshSocket.

        The MPI handshake up front makes the ordering visible in logs (both
        ranks reached ``connect``). The subsequent ``ttnn.MeshSocket(...)``
        constructor is itself a blocking cross-rank handshake, so both sides
        must reach it or one side hangs.
        """
        _handshake(self._role, self._peer_rank)

        connections = [
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_LOGICAL_CORE)),
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_LOGICAL_CORE)),
            )
        ]
        mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, self._fifo_size_bytes)
        socket_config = ttnn.SocketConfig(
            connections,
            mem_config,
            sender_rank=SENDER_RANK,
            receiver_rank=RECEIVER_RANK,
        )
        self._socket = ttnn.MeshSocket(self._mesh, socket_config)

    def send_weights(self, weights: dict[str, "ttnn.Tensor"]) -> None:
        if self._role != _ROLE_SENDER:
            raise RuntimeError("MeshSocketWeightBridge.send_weights called on a receiver.")
        if self._socket is None:
            raise RuntimeError("MeshSocketWeightBridge.send_weights: call connect() first.")

        keys = sorted(weights.keys())
        for k in keys:
            _validate_source_tensor(k, weights[k])

        # Manifest (order + spec) travels over MPI so the receiver knows exactly
        # what to allocate and in which order to post recv_async.
        _send_manifest(self._peer_rank, weights, keys)

        # Stream all tensor bytes over the fabric. No inter-tensor sync: the
        # single MeshSocket queues them in order and the receiver pulls them in
        # order via recv_async posted in the same key order.
        for k in keys:
            ttnn.experimental.send_async(weights[k], self._socket)

        # Drain the fabric so the following barrier is meaningful (barrier is
        # host-only MPI; without this sync, the peer's barrier can complete
        # while the device is still pushing bytes, and the caller may then free
        # or mutate the source tensors underneath the in-flight send).
        ttnn.synchronize_device(self._mesh)

    def receive_weights(self) -> List[dict[str, "ttnn.Tensor"]]:
        if self._role != _ROLE_RECEIVER:
            raise RuntimeError("MeshSocketWeightBridge.receive_weights called on a sender.")
        if self._socket is None:
            raise RuntimeError("MeshSocketWeightBridge.receive_weights: call connect() first.")
        assert self._targets is not None  # enforced in __init__ for receivers.

        manifest = _recv_manifest(self._peer_rank)
        entries = manifest["entries"]
        target = self._targets[0]

        # Allocate first, in manifest order, so we can post recv_async without
        # racing against another allocation on the target device.
        allocated: List[tuple[str, "ttnn.Tensor"]] = []
        for entry in entries:
            spec = ttnn.TensorSpec(
                entry["shape"],
                _dtype_from_name(entry["dtype"]),
                _layout_from_name(entry["layout"]),
            )
            tensor = ttnn.allocate_tensor_on_device(spec, target)
            allocated.append((entry["key"], tensor))

        # Post recv_async in the exact order the sender used for send_async.
        for _key, tensor in allocated:
            ttnn.experimental.recv_async(tensor, self._socket)

        # Drain all recvs so the returned tensors are fully written before we
        # hand them back to the caller (who will do worker.update_weights).
        ttnn.synchronize_device(self._mesh)

        return [{k: t for k, t in allocated}]

    def barrier(self) -> None:
        _handshake(self._role, self._peer_rank)
