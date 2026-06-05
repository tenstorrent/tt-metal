# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-rank weight transfer between a ttml submesh and a tt-transformers
submesh.

Two MPI ranks each open their own ``ttnn.MeshDevice`` (typically half of a
T3K). Rank ``TTML_RANK`` hosts the ttml model; rank ``TTT_RANK`` hosts the
tt-transformers model. Periodically the ttml side calls
``model.export_to_hf_dict()`` and the resulting HF-keyed dict of on-device
``ttnn.Tensor`` handles needs to land, materially equivalent, on the ttt
side as input to ``Transformer.update_weights(hf_dict, hf_rope=False)``.

The bridge uses only ttnn primitives -- it does not depend on
``ttml.autograd.AutoContext`` or ``ttml.core.distributed.SocketManager``,
so the ttt rank can stay ttml-free and just open a mesh device via
``ttnn.open_mesh_device(...)``.

Protocol
========

1. Length-prefixed JSON manifest of ``(key, shape, dtype, layout)`` entries
   is sent first over plain host MPI
   (``ttnn._ttnn.multi_device.send_bytes`` / ``recv_bytes``) so the
   receiver knows what to allocate and in what order.
2. Each tensor is then streamed in manifest key order over a single
   cached fabric socket (``ttnn.MeshSocket`` +
   ``ttnn.experimental.send_async`` / ``recv_async``). One
   ``SocketConnection`` is created per ``(row, col)`` of the local mesh
   so all chips on each side participate in the transfer (matches the
   fan-out logic in ``ttml/core/distributed/socket_manager.cpp``).
3. A single ``ttnn.synchronize_device`` at the end of each direction
   waits for all per-tensor transfers to drain.

Mesh-shape contract
===================

Both submeshes must have the same shape because the per-``(row, col)``
fan-out pairs ``coord -> coord`` on each side. The manifest carries the
sender's mesh shape and the receiver asserts on match. ``[1, 4]`` on
both sides (DDP-only line topology) is the configuration this code is
written for; ``[2, 2]`` on both sides also works.

Replicated-only contract
========================

``Transformer.update_weights`` documents its input as
*replicated, DRAM-interleaved, TILE, bfloat16* (see
``models/tt_transformers/tt/model.py:172-175``), and its leaf
``Attention.update`` raises ``NotImplementedError`` for
``num_devices_per_group > 1`` (see
``models/tt_transformers/tt/attention.py:603-608``). To keep the bridge
honest, ``_send_state`` asserts every source tensor is fully replicated
on every mesh axis (every ``placement`` is ``ttnn.PlacementReplicate``),
DRAM-interleaved, ``TILE_LAYOUT``, and ``bfloat16``. With DDP-only on
the ttml side, ``export_to_hf_dict`` produces exactly that. Enable TP or
mix in any sharded mesh axis on the ttml side and this bridge will
fail-fast on the first sharded weight rather than silently mis-deliver.
"""

from __future__ import annotations

import json
import struct
from typing import Optional

import ttnn

from ttnn._ttnn.multi_device import recv_bytes as _mpi_recv_bytes
from ttnn._ttnn.multi_device import send_bytes as _mpi_send_bytes


TTML_RANK: int = 0
TTT_RANK: int = 1

_MANIFEST_LEN_TAG: int = 1
_MANIFEST_BODY_TAG: int = 2

_ROLE_TTML: str = "ttml"
_ROLE_TTT: str = "ttt"

# Bandwidth-delay product is roughly 10GB/s * 1us = 10MB; use an 8x safety
# margin and allocate 80MB. Mirrors ``_make_socket_mem_config`` in
# ``tt-train/sources/ttml/core/distributed/socket_manager.cpp``.
_SOCKET_FIFO_BYTES: int = 10 * 1024 * 1024 * 2 * 4  # 80 MB


def _dtype_to_name(dtype: "ttnn.DataType") -> str:
    return dtype.name


def _dtype_from_name(name: str) -> "ttnn.DataType":
    return getattr(ttnn.DataType, name)


def _layout_to_name(layout: "ttnn.Layout") -> str:
    return layout.name


def _layout_from_name(name: str) -> "ttnn.Layout":
    return getattr(ttnn.Layout, name)


def _shape_to_list(shape) -> list[int]:
    return [int(d) for d in shape]


def _make_socket_mem_config() -> "ttnn.SocketMemoryConfig":
    """80 MB DRAM fifo, mirroring ttml's ``_make_socket_mem_config``."""
    return ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, _SOCKET_FIFO_BYTES)


def _make_socket_connection_config(mesh_shape: list[int]) -> list["ttnn.SocketConnection"]:
    """One ``coord -> coord`` SocketConnection per mesh chip (CoreCoord(0,0)).

    Mirrors ``_make_socket_connection_config`` in
    ``tt-train/sources/ttml/core/distributed/socket_manager.cpp``: every
    sender chip pairs with the same-coord receiver chip so all devices
    on each side participate in the transfer.
    """
    if len(mesh_shape) != 2:
        raise ValueError(
            f"WeightBridge: only 2D meshes are supported (got mesh_shape={mesh_shape}); "
            f"line topologies should be expressed as [1, N] or [N, 1]."
        )
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
    connections: list[ttnn.SocketConnection] = []
    for row in range(rows):
        for col in range(cols):
            mesh_core_coord = ttnn.MeshCoreCoord(
                ttnn.MeshCoordinate(row, col),
                ttnn.CoreCoord(0, 0),
            )
            connections.append(ttnn.SocketConnection(mesh_core_coord, mesh_core_coord))
    return connections


def _is_fully_replicated(tensor: "ttnn.Tensor") -> bool:
    """True iff every mesh axis has a ``PlacementReplicate`` placement."""
    placements = tensor.tensor_topology().placements()
    return all(isinstance(p, ttnn.PlacementReplicate) for p in placements)


def _alloc_replicated_template(
    shape: list[int],
    dtype: "ttnn.DataType",
    layout: "ttnn.Layout",
    device: "ttnn.MeshDevice",
) -> "ttnn.Tensor":
    """Allocate a fresh replicated tensor on ``device`` matching the spec.

    Contents are uninitialized and overwritten by ``recv_async``. On a
    multi-device mesh, ``allocate_tensor_on_device`` produces a tensor
    whose per-device buffers each have ``shape`` -- the natural
    "replicated" layout.
    """
    spec = ttnn.TensorSpec(shape, dtype, layout)
    return ttnn.allocate_tensor_on_device(spec, device)


class WeightBridge:
    """Stream an HF-keyed weight dict from the ttml rank to the ttt rank.

    Both ranks instantiate one. ``role="ttml"`` on the sending rank and
    ``role="ttt"`` on the receiving rank. ``peer_rank`` is the other
    rank's id. ``device`` is the local mesh -- on the ttml side this is
    typically ``ttml.autograd.AutoContext.get_instance().get_device()``,
    on the ttt side it is whatever ``ttnn.open_mesh_device(...)``
    returned (the bridge does not look at AutoContext).

    Use ``transfer_state`` for a single rank-aware call, or
    ``send_state`` / ``recv_state`` for the explicit per-side flow.
    """

    def __init__(
        self,
        *,
        role: str,
        peer_rank: int,
        device: "ttnn.MeshDevice",
    ) -> None:
        if role not in (_ROLE_TTML, _ROLE_TTT):
            raise ValueError(f"WeightBridge: role must be {_ROLE_TTML!r} or {_ROLE_TTT!r}, got {role!r}")
        if not ttnn.distributed_context_is_initialized():
            raise RuntimeError(
                "WeightBridge: ttnn distributed context is not initialized. "
                "Call ttnn.init_distributed_context() (ttt side) or "
                "AutoContext.get_instance().initialize_distributed_context(*sys.argv) "
                "(ttml side) before constructing the bridge."
            )

        self.role = role
        self.peer_rank = int(peer_rank)
        self.device = device
        self.rank = int(ttnn.distributed_context_get_rank())

        if role == _ROLE_TTML and self.rank != TTML_RANK:
            raise RuntimeError(
                f"WeightBridge: role={role!r} but local MPI rank is {self.rank} " f"(expected TTML_RANK={TTML_RANK})"
            )
        if role == _ROLE_TTT and self.rank != TTT_RANK:
            raise RuntimeError(
                f"WeightBridge: role={role!r} but local MPI rank is {self.rank} " f"(expected TTT_RANK={TTT_RANK})"
            )

        # Lazily constructed on first transfer; caching avoids rebuilding
        # the SocketConfig + MeshSocket per tensor.
        self._socket: Optional[ttnn.MeshSocket] = None

    def _get_socket(self) -> "ttnn.MeshSocket":
        if self._socket is not None:
            return self._socket
        mesh_shape = _shape_to_list(self.device.shape)
        socket_config = ttnn.SocketConfig(
            _make_socket_connection_config(mesh_shape),
            _make_socket_mem_config(),
            sender_rank=TTML_RANK,
            receiver_rank=TTT_RANK,
        )
        self._socket = ttnn.MeshSocket(self.device, socket_config)
        return self._socket

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def transfer_state(
        self,
        source: Optional[dict[str, "ttnn.Tensor"]] = None,
    ) -> Optional[dict[str, "ttnn.Tensor"]]:
        """Single rank-aware entry point.

        On the ttml rank, pass ``source=ttml_model.export_to_hf_dict()``
        and the bridge sends. Returns ``None``.

        On the ttt rank, pass nothing and the bridge receives. Returns
        the materialized dict, ready to feed to
        ``ttt.model.update_weights(received, hf_rope=False)``.
        """
        if self.role == _ROLE_TTML:
            if source is None:
                raise ValueError("WeightBridge.transfer_state on ttml rank requires source=hf_dict")
            self.send_state(source)
            return None
        if source is not None:
            raise ValueError("WeightBridge.transfer_state on ttt rank must not pass source=...")
        return self.recv_state()

    def send_state(self, hf_dict: dict[str, "ttnn.Tensor"]) -> None:
        """Send the full HF-keyed weight dict to the ttt rank.

        Order of the keys is sorted-by-name so the receiver iterates in
        a matching deterministic order. Tied embeddings (the
        ``model.embed_tokens.weight`` / ``lm_head.weight`` pair that
        ``export_to_hf_dict`` returns as the same handle for
        ``weight_tying=Enabled``) are sent twice on the wire -- the
        receiver allocates separate destination buffers under each key,
        which is exactly what ``Transformer.update_weights`` expects.

        Each tensor is asserted to be fully replicated, DRAM-interleaved,
        ``TILE_LAYOUT``, and ``bfloat16`` -- the contract documented by
        ``Transformer.update_weights``. Anything else fails fast here
        rather than silently mis-delivering on the wire.
        """
        if self.role != _ROLE_TTML:
            raise RuntimeError(f"WeightBridge.send_state called with role={self.role!r}; expected {_ROLE_TTML!r}")

        keys = sorted(hf_dict.keys())
        for k in keys:
            self._validate_source_tensor(k, hf_dict[k])

        manifest = {
            "version": 1,
            "sender_mesh_shape": _shape_to_list(self.device.shape),
            "entries": [
                {
                    "key": k,
                    "shape": _shape_to_list(hf_dict[k].shape),
                    "dtype": _dtype_to_name(hf_dict[k].dtype),
                    "layout": _layout_to_name(hf_dict[k].layout),
                }
                for k in keys
            ],
        }
        body = json.dumps(manifest).encode("utf-8")

        # Length-prefixed manifest. recv_bytes requires nbytes upfront,
        # so the sender announces the body size first (8-byte
        # little-endian uint64) and then the body itself.
        _mpi_send_bytes(struct.pack("<Q", len(body)), self.peer_rank, _MANIFEST_LEN_TAG)
        _mpi_send_bytes(body, self.peer_rank, _MANIFEST_BODY_TAG)

        socket = self._get_socket()
        for k in keys:
            ttnn.experimental.send_async(hf_dict[k], socket)
        ttnn.synchronize_device(self.device)

    def recv_state(self) -> dict[str, "ttnn.Tensor"]:
        """Receive a full HF-keyed weight dict from the ttml rank.

        Allocates one fresh template tensor per manifest entry on the
        local mesh (replicated, matching ``(shape, dtype, layout)``)
        then ``recv_async``s into it. Total device-memory footprint is
        the entire model state simultaneously, until the caller passes
        the dict to ``Transformer.update_weights`` and drops the
        reference.
        """
        if self.role != _ROLE_TTT:
            raise RuntimeError(f"WeightBridge.recv_state called with role={self.role!r}; expected {_ROLE_TTT!r}")

        header = _mpi_recv_bytes(8, self.peer_rank, _MANIFEST_LEN_TAG)
        (n,) = struct.unpack("<Q", header)
        body = _mpi_recv_bytes(int(n), self.peer_rank, _MANIFEST_BODY_TAG)
        manifest = json.loads(body.decode("utf-8"))

        sender_shape = manifest["sender_mesh_shape"]
        receiver_shape = _shape_to_list(self.device.shape)
        if sender_shape != receiver_shape:
            raise RuntimeError(
                "WeightBridge: ttml and ttt submeshes must have the same shape "
                "(per-(row, col) socket fan-out pairs coord->coord on each side). "
                f"ttml mesh {sender_shape} != ttt mesh {receiver_shape}."
            )

        socket = self._get_socket()
        hf_dict: dict[str, ttnn.Tensor] = {}
        for entry in manifest["entries"]:
            template = _alloc_replicated_template(
                shape=entry["shape"],
                dtype=_dtype_from_name(entry["dtype"]),
                layout=_layout_from_name(entry["layout"]),
                device=self.device,
            )
            received = ttnn.experimental.recv_async(template, socket)
            # ``recv_async`` returns a list with the populated output tensor
            # (see ``ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async_nanobind.cpp``).
            hf_dict[entry["key"]] = received[0] if isinstance(received, (list, tuple)) and received else template

        ttnn.synchronize_device(self.device)
        return hf_dict

    def barrier(self) -> None:
        """Best-practice fence after a transfer.

        Lets the ttml rank hold its source tensors live until the ttt
        rank finishes copying them into TTT's destination buffers (in
        particular the per-layer K/V slices that ``export_to_hf_dict``
        newly allocates would otherwise race against ``update_weights``
        if the ttml rank freed them too eagerly).
        """
        ttnn.distributed_context_barrier()

    # ------------------------------------------------------------------ #
    # Internal validation                                                #
    # ------------------------------------------------------------------ #

    def _validate_source_tensor(self, key: str, tensor: "ttnn.Tensor") -> None:
        """Enforce the ``Transformer.update_weights`` input contract on
        every source tensor before we put it on the wire."""
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
                f"(memory_config={tensor.memory_config()}). Move the parameter to DRAM-interleaved "
                "before exporting; sharded L1 is not supported by the current bridge protocol."
            )
        if not _is_fully_replicated(tensor):
            placements = tensor.tensor_topology().placements()
            raise ValueError(
                f"WeightBridge: tensor {key!r} is not fully replicated across the mesh "
                f"(placements={placements}). The bridge currently only supports replicated weights; "
                "sharded weights would need a host-gather + re-shard path that is not implemented. "
                "On the ttml side this typically means TP (or CP) is enabled -- use DDP-only "
                "(line topology like [1, N]) for now."
            )
