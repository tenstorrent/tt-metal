# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Inference RPC + weight transport between a ttml rank and a ttt rank.

This module exposes three classes, ordered from the lowest-level transport
up to the user-facing entry points:

- :class:`WeightBridge` -- low-level cross-rank weight transport. Both
  ranks construct one, call :meth:`WeightBridge.connect`, and then any
  number of :meth:`WeightBridge.transfer_weights` calls move
  HuggingFace-keyed weight dicts over a cached ``ttnn.MeshSocket``.
  Tag namespace 1..6 (manifest length, manifest body, two handshake
  directions, two mesh-shape directions). The transport is symmetric in
  the sense that both ranks must call ``connect()`` once and
  ``transfer_weights`` exactly the same number of times.

- :class:`TttInferenceServer` -- runs on the ttt rank. Owns an internal
  ``WeightBridge`` (constructed with ``role="ttt"``) and the
  ``generate_fn`` + ``on_weights_received`` callbacks supplied to its
  constructor. Calling :meth:`serve_forever` blocks until the peer sends
  ``OP_SHUTDOWN``. Op dispatch:

    * ``OP_GENERATE`` -> ``generate_fn(prompts, ...)``; result shipped
      back as JSON.
    * ``OP_REQUEST_TRANSFER`` -> internal ``bridge.transfer_weights()``
      to receive the HF dict, then ``on_weights_received(hf_dict)``
      (typically ``Transformer.update_weights``), then a post-transfer
      barrier.
    * ``OP_SHUTDOWN`` -> return.

- :class:`TttInferenceClient` -- runs on the ttml rank. Owns an internal
  ``WeightBridge`` (constructed with ``role="ttml"``). Public methods:

    * :meth:`TttInferenceClient.remote_generate` -- run inference on the
      ttt rank, blocks for the response.
    * :meth:`TttInferenceClient.transfer_weights` -- single-call weight
      update: posts ``OP_REQUEST_TRANSFER``, runs the
      ``bridge.transfer_weights(hf_dict)`` transport, and waits on the
      post-transfer barrier. The user side only needs this one call.
    * :meth:`TttInferenceClient.shutdown` -- post ``OP_SHUTDOWN``.

Constructor handshake
=====================

Both ``TttInferenceServer`` and ``TttInferenceClient`` call
``bridge.connect()`` in their own constructors. That call is a two-rank
handshake (handshake payload + mesh-shape exchange + ``MeshSocket``
open) which blocks until the peer constructs its own object. As a
consequence:

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
propagates out of :meth:`serve_forever`; the server process dies and
MPI tears down the world. The client, blocked on ``recv_bytes`` for the
response, will then never hear back. **Both processes die together on
any server-side callback failure**; do not wrap these callbacks in a
``try`` / ``except`` inside the loop.

Tag namespace
=============

- ``WeightBridge``: MPI tags 1..6 (manifest length/body, two handshake
  directions, two mesh-shape directions).
- Inference RPC: MPI tags 10..13 (request header/body, response
  header/body).

Disjoint tag namespaces let both protocols share the same MPI
distributed context simultaneously without crosstalk.

Wire format -- inference RPC
============================

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

import ttnn
from ttnn._ttnn.multi_device import recv_bytes as _mpi_recv_bytes
from ttnn._ttnn.multi_device import send_bytes as _mpi_send_bytes


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

TTML_RANK: int = 0
TTT_RANK: int = 1

OP_GENERATE: int = 1
OP_SHUTDOWN: int = 2
OP_REQUEST_TRANSFER: int = 3


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# WeightBridge tag namespace (1..6).
_MANIFEST_LEN_TAG: int = 1
_MANIFEST_BODY_TAG: int = 2
# Per-direction tags so a send cannot self-match the same rank's recv.
_HANDSHAKE_TAG_FROM_TTML: int = 3
_HANDSHAKE_TAG_FROM_TTT: int = 4
_SHAPE_TAG_FROM_TTML: int = 5
_SHAPE_TAG_FROM_TTT: int = 6

_HANDSHAKE_PAYLOAD: bytes = b"ready"

# Fixed-size 2D mesh shape exchange: two little-endian uint32s.
_SHAPE_PAYLOAD_BYTES: int = 8

_ROLE_TTML: str = "ttml"
_ROLE_TTT: str = "ttt"

# Bandwidth-delay product is roughly 10GB/s * 1us = 10MB; use an 8x safety
# margin and allocate 80MB. Mirrors ``_make_socket_mem_config`` in
# ``tt-train/sources/ttml/core/distributed/socket_manager.cpp``.
_SOCKET_FIFO_BYTES: int = 10 * 1024 * 1024 * 2 * 4  # 80 MB

# Inference RPC tag namespace (10..13). Disjoint from WeightBridge above.
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
OnWeightsReceivedFn = Callable[[dict], None]


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


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


def _make_socket_connection_config(
    sender_shape: list[int],
    receiver_shape: list[int],
) -> list["ttnn.SocketConnection"]:
    """Build the ``SocketConnection`` list shared by both ranks.

    Two regimes:

    1. ``sender_shape == receiver_shape``: cartesian per-``(row, col)``
       fan-out -- every chip on each side participates. Mirrors
       ``_make_socket_connection_config`` in
       ``tt-train/sources/ttml/core/distributed/socket_manager.cpp``.
    2. ``sender_shape != receiver_shape``: a single
       ``(0, 0) -> (0, 0)`` connection. Because every source tensor is
       fully replicated (enforced in ``_validate_source_tensor``), the
       chip at ``(0, 0)`` of the sender mesh carries the same payload
       the cartesian fan-out would have. The receiver mesh just needs a
       ``(0, 0)`` device, which every non-empty mesh has. Throughput is
       capped by one fabric link.
    """
    if len(sender_shape) != 2 or len(receiver_shape) != 2:
        raise ValueError(
            f"WeightBridge: only 2D meshes are supported (got sender_shape={sender_shape}, "
            f"receiver_shape={receiver_shape}); line topologies should be expressed as [1, N] or [N, 1]."
        )
    if sender_shape == receiver_shape:
        rows, cols = int(sender_shape[0]), int(sender_shape[1])
        connections: list[ttnn.SocketConnection] = []
        for row in range(rows):
            for col in range(cols):
                mesh_core_coord = ttnn.MeshCoreCoord(
                    ttnn.MeshCoordinate(row, col),
                    ttnn.CoreCoord(0, 0),
                )
                connections.append(ttnn.SocketConnection(mesh_core_coord, mesh_core_coord))
        return connections

    coord_zero = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
    return [ttnn.SocketConnection(coord_zero, coord_zero)]


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
# WeightBridge -- low-level cross-rank weight transport
# ---------------------------------------------------------------------------


class WeightBridge:
    """Stream an HF-keyed weight dict from the ttml rank to the ttt rank.

    Both ranks instantiate one. ``role="ttml"`` on the sending rank and
    ``role="ttt"`` on the receiving rank. ``peer_rank`` is the other
    rank's id. ``device`` is the local mesh -- on the ttml side this is
    typically ``ttml.autograd.AutoContext.get_instance().get_device()``,
    on the ttt side it is whatever ``ttnn.open_mesh_device(...)``
    returned (the bridge does not look at AutoContext).

    Lifecycle
    ---------

    1. ``WeightBridge(...)`` -- cheap, non-blocking; validates args.
    2. ``bridge.connect()`` -- both ranks call once; handshakes and
       opens the ``MeshSocket``. Blocks until the peer also calls it.
    3. ``transfer_weights`` / ``send_state`` / ``recv_state`` -- moves
       the weights; may be called multiple times.
    4. ``bridge.barrier()`` -- optional post-transfer fence.

    Transfer methods raise ``RuntimeError`` if ``connect()`` has not
    been called.

    Mesh-shape contract
    -------------------

    Both ranks exchange their local mesh shapes during ``connect()`` and
    build the same ``SocketConfig`` from that pair. Two regimes are
    supported:

    - Symmetric (``ttml.shape == ttt.shape``): cartesian per-``(row, col)``
      fan-out so every chip on each side participates. ``[1, 4]`` on
      both sides (DDP-only line topology) is the canonical
      configuration; ``[2, 2]`` on both sides also works.
    - Asymmetric (any pair of non-empty 2D meshes, e.g.
      ``[1, 2] -> [1, 1]``): a single ``(0, 0) -> (0, 0)`` connection.
      Every source tensor is required to be fully replicated (enforced
      in ``_validate_source_tensor``), so the ttml chip at ``(0, 0)``
      carries the same payload the cartesian fan-out would have, and
      the other ttml chips sit idle for the duration of the transfer.

    Replicated-only contract
    ------------------------

    ``Transformer.update_weights`` documents its input as *replicated,
    DRAM-interleaved, TILE, bfloat16*, and its leaf ``Attention.update``
    raises ``NotImplementedError`` for ``num_devices_per_group > 1``. To
    keep the bridge honest, ``send_state`` asserts every source tensor
    is fully replicated on every mesh axis (every ``placement`` is
    ``ttnn.PlacementReplicate``), DRAM-interleaved, ``TILE_LAYOUT``,
    and ``bfloat16``. With DDP-only on the ttml side,
    ``export_to_hf_dict`` produces exactly that. Enable TP or mix in any
    sharded mesh axis on the ttml side and this bridge will fail-fast
    on the first sharded weight rather than silently mis-deliver.
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
        local_rank = _require_distributed_context("WeightBridge")

        self.role = role
        self.peer_rank = int(peer_rank)
        self.device = device
        self.rank = local_rank

        if role == _ROLE_TTML and self.rank != TTML_RANK:
            raise RuntimeError(
                f"WeightBridge: role={role!r} but local MPI rank is {self.rank} (expected TTML_RANK={TTML_RANK})"
            )
        if role == _ROLE_TTT and self.rank != TTT_RANK:
            raise RuntimeError(
                f"WeightBridge: role={role!r} but local MPI rank is {self.rank} (expected TTT_RANK={TTT_RANK})"
            )

        # Populated by ``connect()``; ``None`` triggers fail-fast in
        # ``send_state``/``recv_state``.
        self._socket: Optional[ttnn.MeshSocket] = None

    # ------------------------------------------------------------------ #
    # Synchronisation primitives                                         #
    # ------------------------------------------------------------------ #

    def _handshake(self) -> None:
        """Two-rank barrier: each side eager-sends, then blocks on recv."""
        if self.role == _ROLE_TTML:
            _mpi_send_bytes(_HANDSHAKE_PAYLOAD, self.peer_rank, _HANDSHAKE_TAG_FROM_TTML)
            _mpi_recv_bytes(len(_HANDSHAKE_PAYLOAD), self.peer_rank, _HANDSHAKE_TAG_FROM_TTT)
        else:
            _mpi_send_bytes(_HANDSHAKE_PAYLOAD, self.peer_rank, _HANDSHAKE_TAG_FROM_TTT)
            _mpi_recv_bytes(len(_HANDSHAKE_PAYLOAD), self.peer_rank, _HANDSHAKE_TAG_FROM_TTML)

    def _exchange_mesh_shapes(self) -> tuple[list[int], list[int]]:
        """Exchange 2D mesh shapes; return ``(sender_shape, receiver_shape)``.

        Both ranks need the same ``SocketConfig`` (the descriptor-exchange
        in ``MeshSocket::connect_with_peer`` validates the peer's config
        against the local config), so each side must know both shapes
        before constructing the socket. The asymmetric case --
        e.g. ttml on ``[1, 2]`` and ttt on ``[1, 1]`` -- depends on this
        exchange to land on the same single-connection config on both
        sides.
        """
        local_shape = _shape_to_list(self.device.shape)
        if len(local_shape) != 2:
            raise ValueError(
                f"WeightBridge: only 2D meshes are supported (got local mesh_shape={local_shape}); "
                f"line topologies should be expressed as [1, N] or [N, 1]."
            )
        payload = struct.pack("<II", int(local_shape[0]), int(local_shape[1]))
        if self.role == _ROLE_TTML:
            _mpi_send_bytes(payload, self.peer_rank, _SHAPE_TAG_FROM_TTML)
            peer_bytes = _mpi_recv_bytes(_SHAPE_PAYLOAD_BYTES, self.peer_rank, _SHAPE_TAG_FROM_TTT)
        else:
            _mpi_send_bytes(payload, self.peer_rank, _SHAPE_TAG_FROM_TTT)
            peer_bytes = _mpi_recv_bytes(_SHAPE_PAYLOAD_BYTES, self.peer_rank, _SHAPE_TAG_FROM_TTML)
        peer_shape = list(struct.unpack("<II", peer_bytes))
        if self.role == _ROLE_TTML:
            return local_shape, peer_shape
        return peer_shape, local_shape

    def _require_connected(self, op: str) -> "ttnn.MeshSocket":
        if self._socket is None:
            raise RuntimeError(
                f"WeightBridge.{op}() called before WeightBridge.connect(). "
                "Both ranks must call bridge.connect() once after constructing "
                "the bridge and before any transfer."
            )
        return self._socket

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Handshake with the peer and open the ``MeshSocket``. Idempotent.

        Both ranks must call this exactly once before any transfer. The
        handshake pins both ranks to the same point so the
        ``MeshSocket`` constructor's MPI descriptor exchange does not
        trip its 10s timeout when pre-work is asymmetric. A subsequent
        shape exchange lets each side build the same ``SocketConfig``
        when the two submeshes have different shapes.
        """
        if self._socket is not None:
            return
        self._handshake()
        sender_shape, receiver_shape = self._exchange_mesh_shapes()
        socket_config = ttnn.SocketConfig(
            _make_socket_connection_config(sender_shape, receiver_shape),
            _make_socket_mem_config(),
            sender_rank=TTML_RANK,
            receiver_rank=TTT_RANK,
        )
        self._socket = ttnn.MeshSocket(self.device, socket_config)

    def transfer_weights(
        self,
        source: Optional[dict[str, "ttnn.Tensor"]] = None,
    ) -> Optional[dict[str, "ttnn.Tensor"]]:
        """Single rank-aware entry point. Requires a prior ``connect()``.

        On the ttml rank, pass ``source=ttml_model.export_to_hf_dict()``
        and the bridge sends. Returns ``None``.

        On the ttt rank, pass nothing and the bridge receives. Returns
        the materialized dict, ready to feed to
        ``ttt.model.update_weights(received, hf_rope=False)``.
        """
        if self.role == _ROLE_TTML:
            if source is None:
                raise ValueError("WeightBridge.transfer_weights on ttml rank requires source=hf_dict")
            self.send_state(source)
            return None
        if source is not None:
            raise ValueError("WeightBridge.transfer_weights on ttt rank must not pass source=...")
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
        socket = self._require_connected("send_state")

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
        socket = self._require_connected("recv_state")

        header = _mpi_recv_bytes(8, self.peer_rank, _MANIFEST_LEN_TAG)
        (n,) = struct.unpack("<Q", header)
        body = _mpi_recv_bytes(int(n), self.peer_rank, _MANIFEST_BODY_TAG)
        manifest = json.loads(body.decode("utf-8"))

        # Mesh shapes were already exchanged and the SocketConfig
        # negotiated in connect(); the manifest still carries
        # ``sender_mesh_shape`` for forward compatibility / debugging
        # but the receiver does not need to consume it.

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
        """Post-transfer fence so the sender does not free source
        tensors before the receiver drains them."""
        self._handshake()

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


# ---------------------------------------------------------------------------
# TttInferenceServer -- ttt rank
# ---------------------------------------------------------------------------


class TttInferenceServer:
    """Blocking inference RPC server, runs on the tt-transformers rank.

    Composes an internal :class:`WeightBridge` (with ``role="ttt"``) for
    the OP_REQUEST_TRANSFER side effect. The constructor performs the
    bridge handshake -- it blocks until the peer constructs its
    :class:`TttInferenceClient`. Returning from ``__init__`` therefore
    guarantees that both ranks are past the handshake and the
    ``MeshSocket`` is open.

    :meth:`serve_forever` then blocks reading op headers and
    dispatches:

    - ``OP_GENERATE``: calls ``generate_fn(prompts, **kwargs)`` and
      ships the result back as JSON.
    - ``OP_REQUEST_TRANSFER``: runs ``bridge.transfer_weights()`` to
      receive the HF dict, calls ``on_weights_received(hf_dict)`` (if
      supplied), and then runs ``bridge.barrier()`` to fence the
      transfer.
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
        device: "ttnn.MeshDevice",
        generate_fn: GenerateFn,
        on_weights_received: Optional[OnWeightsReceivedFn] = None,
    ) -> None:
        local_rank = _require_distributed_context("TttInferenceServer")
        if local_rank != TTT_RANK:
            raise RuntimeError(f"TttInferenceServer must run on TTT_RANK={TTT_RANK} (got local rank {local_rank}).")
        if int(peer_rank) != TTML_RANK:
            raise RuntimeError(f"TttInferenceServer: peer_rank must be TTML_RANK={TTML_RANK} (got {peer_rank}).")

        self.peer_rank: int = int(peer_rank)
        self._generate_fn: GenerateFn = generate_fn
        self._on_weights_received: Optional[OnWeightsReceivedFn] = on_weights_received

        # The bridge handshake (open MeshSocket) is what makes the
        # constructor blocking. Subsequent serve_forever() calls can
        # rely on the socket being live.
        self._bridge: WeightBridge = WeightBridge(role=_ROLE_TTT, peer_rank=int(peer_rank), device=device)
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
                # client.transfer_weights(hf_dict) call; the bridge
                # exchange below pairs up with it on tags 1..6.
                hf_dict = self._bridge.transfer_weights()
                assert hf_dict is not None, "WeightBridge.transfer_weights on ttt rank must return a dict"
                if self._on_weights_received is not None:
                    self._on_weights_received(hf_dict)
                self._bridge.barrier()
                continue

            if op != OP_GENERATE:
                raise RuntimeError(f"TttInferenceServer: unknown op {op}")

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
# TttInferenceClient -- ttml rank
# ---------------------------------------------------------------------------


class TttInferenceClient:
    """Blocking inference RPC client, runs on the ttml rank.

    Composes an internal :class:`WeightBridge` (with ``role="ttml"``)
    so the user only has to call :meth:`transfer_weights` once to push
    a fresh weight dict to the peer -- the op-code message and the
    bridge transport are bundled together inside that single method.

    The constructor performs the bridge handshake -- it blocks until
    the peer constructs its :class:`TttInferenceServer`. Returning from
    ``__init__`` therefore guarantees that both ranks are past the
    handshake and the ``MeshSocket`` is open.

    No try/except is used around blocking receives: if the server
    process dies mid-request, the client's blocking ``recv_bytes`` will
    not return and MPI will eventually abort the world. That is the
    intended failure mode.
    """

    def __init__(
        self,
        *,
        peer_rank: int,
        device: "ttnn.MeshDevice",
    ) -> None:
        local_rank = _require_distributed_context("TttInferenceClient")
        if local_rank != TTML_RANK:
            raise RuntimeError(f"TttInferenceClient must run on TTML_RANK={TTML_RANK} (got local rank {local_rank}).")
        if int(peer_rank) != TTT_RANK:
            raise RuntimeError(f"TttInferenceClient: peer_rank must be TTT_RANK={TTT_RANK} (got {peer_rank}).")

        self.peer_rank: int = int(peer_rank)

        self._bridge: WeightBridge = WeightBridge(role=_ROLE_TTML, peer_rank=int(peer_rank), device=device)
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

    def transfer_weights(self, hf_dict: dict[str, "ttnn.Tensor"]) -> None:
        """Push a fresh HF-keyed weight dict to the ttt rank in one call.

        Internally:

        1. Send the ``OP_REQUEST_TRANSFER`` header (tag 10, no body).
        2. Run ``bridge.transfer_weights(hf_dict)`` -- ships the manifest
           plus each tensor over the cached ``MeshSocket`` (tags 1, 2 +
           socket).
        3. Run ``bridge.barrier()`` -- post-transfer fence matching the
           server-side barrier in ``serve_forever``.

        The user never has to interleave a separate request op with the
        bridge call; the whole exchange is bundled here.
        """
        hdr = struct.pack(_HEADER_FMT, OP_REQUEST_TRANSFER, 0, 0)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
        self._bridge.transfer_weights(hf_dict)
        self._bridge.barrier()

    def shutdown(self) -> None:
        """Send ``OP_SHUTDOWN`` to the server; no response is expected."""
        hdr = struct.pack(_HEADER_FMT, OP_SHUTDOWN, 0, 0)
        _mpi_send_bytes(hdr, self.peer_rank, _INFER_REQ_HDR_TAG)
