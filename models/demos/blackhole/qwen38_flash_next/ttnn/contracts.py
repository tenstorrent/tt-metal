# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed TTNN mesh and tensor-placement contracts.

Numerically identical tensors with different mesh topology are not
interchangeable in this model.  Every device module calls these guards after
uploads, cache loads, layout changes, and collectives.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import torch

import ttnn

TP_SIZE = 4
MESH_SHAPE = (1, TP_SIZE)
# The fused kernels use all compute cores; fabric must not reserve Tensix cores.
# Serving workers apply these model defaults before opening the mesh.
FABRIC_CONFIG = {
    "config": ttnn.FabricConfig.FABRIC_1D,
    "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
    "fabric_tensix_config": ttnn.FabricTensixConfig.DISABLED,
}
POSITION_SCALAR_SHAPE = (1, 1, 1, 1)
POSITION_INDEX_ROW_SHAPE = (1, 1, 1, ttnn.TILE_SIZE)
# P & ~3 lane mask.  Kept in a tensor: values >= 2**31 must not take the scalar
# operand path of the binary ops.
BLOCK_START_LANE_MASK = 0xFFFFFFFC
UINT32_LIMIT = 2**32
# Rows of one prefill chunk: one 32-row tile, so every decode matmul keeps its program.
CHUNK_ROWS = ttnn.TILE_SIZE
# Rows of one long prefill chunk: four tiles.  The DRAM-sharded decode matmuls run once per tile (the
# pinned runtime admits one row tile); every other op runs natively on the four tiles.
LONG_CHUNK_ROWS = 4 * ttnn.TILE_SIZE
CHUNK_ROW_COUNTS = (CHUNK_ROWS, LONG_CHUNK_ROWS)
# Rows of one prefill slab (the opt-in layer-major form): a multiple of the long chunk, 256 .. 4096.  Every dense
# linear runs as one 2D-multicast matmul on an interleaved copy of its weight; the routed experts run in 128-token
# moe_compute calls; the GDN kernel carries the state through the slab in one call.
SLAB_ROW_STEP = LONG_CHUNK_ROWS
MIN_SLAB_ROWS = 2 * LONG_CHUNK_ROWS
MAX_SLAB_ROWS = 32 * LONG_CHUNK_ROWS
DEFAULT_SLAB_ROWS = 16 * LONG_CHUNK_ROWS


def is_slab_rows(rows) -> bool:
    """True for a prefill slab row count (a multiple of 128 in 256 .. 4096; never 32 or 128)."""

    return (
        not isinstance(rows, bool)
        and type(rows) is int
        and MIN_SLAB_ROWS <= rows <= MAX_SLAB_ROWS
        and rows % SLAB_ROW_STEP == 0
    )


def chunk_row_tiles(rows: int) -> int:
    """The row tiles of a chunk form (1 or 4) or of a slab (8 .. 128); rejects any other row count."""

    if rows not in CHUNK_ROW_COUNTS and not is_slab_rows(rows):
        raise ValueError(
            f"prefill chunk rows must be one of {CHUNK_ROW_COUNTS} or a slab row count "
            f"(a multiple of {SLAB_ROW_STEP} in {MIN_SLAB_ROWS}..{MAX_SLAB_ROWS}), got {rows!r}"
        )
    return rows // ttnn.TILE_SIZE


# Batched decode: lane u of a [1,1,1,32] row (position, token) or row u of a [.., 32, ..] tile is user u; every
# per-lane tensor is one tile tall, so the lane count is 1..32 and the row count of a multi-row path is the same
# contract (1 = decode, 5 = the MTP verifier, 32 = the prefill chunk).
MAX_LANES = ttnn.TILE_SIZE
# The GDN ring slot is one host int per trace, so every resident lane shares ``P mod GDN_RESIDUE_CLASSES``
# (gdn.py CONV_KERNEL_SIZE); a lane is admitted only at a step whose residue is its position's.
GDN_RESIDUE_CLASSES = 4


def require_lane_count(lanes, *, label: str = "lane count", error_type: type[Exception] = ValueError) -> int:
    """``lanes`` as an exact int in [1, MAX_LANES]; the error names the actual value and the admitted range."""

    value = _exact_integer(lanes, label=label, error_type=error_type)
    if not 1 <= value <= MAX_LANES:
        raise error_type(f"{label} must be in [1,{MAX_LANES}], got {value}")
    return value


def admission_wait_steps(resident_residue: int, position: int) -> int:
    """Steps (0..3) a lane at ``position`` waits before it may join lanes whose positions are ``resident_residue``
    mod 4 at the current step: after ``w`` steps the resident residue is ``resident_residue + w``, so the lane joins
    at the first step where that equals ``position mod 4``.  Ragged lengths are otherwise free."""

    residue = _exact_integer(resident_residue, label="resident residue", error_type=ValueError)
    if not 0 <= residue < GDN_RESIDUE_CLASSES:
        raise ValueError(f"resident residue must be in [0,{GDN_RESIDUE_CLASSES}), got {residue}")
    value = _exact_integer(position, label="admission position", error_type=ValueError)
    if value < 0:
        raise ValueError(f"admission position must be non-negative, got {value}")
    return (value - residue) % GDN_RESIDUE_CLASSES


class TensorPlacement(str, Enum):
    HIDDEN_SHARDED = "hidden_sharded"
    EXPERT_SHARDED = "expert_sharded"
    ROUTER_SHARDED = "router_sharded"
    INTERMEDIATE_SHARDED = "intermediate_sharded"
    VOCAB_SHARDED = "vocab_sharded"
    HEAD_SHARDED = "head_sharded"
    KV_PAIR_GROUPED = "kv_pair_grouped"
    REPLICATED = "replicated"
    LOCAL_PARTIAL = "local_partial"


def _exact_integer(value, *, label: str, error_type: type[Exception] = RuntimeError) -> int:
    """Return an integer-protocol value without accepting scalar aliases.

    Nanobind integer results and other legitimate bound integer types implement
    ``__index__``.  Decimal strings and floating-point values do not.  ``bool``
    also implements ``__index__``, so reject it explicitly before conversion.
    """

    if isinstance(value, bool):
        raise error_type(f"{label} must be an exact integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise error_type(f"{label} must be an exact integer, got {value!r}") from error
    if isinstance(result, bool) or not isinstance(result, int):
        raise error_type(f"{label} must be an exact integer, got {value!r}")
    return result


def _shape_tuple(
    value,
    *,
    label: str = "shape",
    error_type: type[Exception] = RuntimeError,
) -> tuple[int, ...]:
    try:
        items = tuple(value)
    except TypeError as error:
        raise error_type(f"{label} must be an iterable of exact integers") from error
    return tuple(
        _exact_integer(item, label=f"{label}[{index}]", error_type=error_type) for index, item in enumerate(items)
    )


def _placement_name(value) -> str:
    return type(value).__name__


def tensor_metadata(tensor) -> str:
    """The four fields every metadata contract checks, for actual-vs-expected error text."""

    return (
        f"shape={list(_shape_tuple(tensor.shape))} padded_shape={list(_shape_tuple(tensor.padded_shape))} "
        f"dtype={tensor.dtype} layout={tensor.layout}"
    )


@dataclass(frozen=True)
class Qwen38TensorBackingIdentity:
    """Exact ownership identity for one distributed device allocation.

    ``tensor_id`` identifies a TTNN graph object and Python object identity
    identifies only one wrapper.  Neither is an allocation identity.  Device
    lifetime ledgers instead bind every local backing buffer to its exact mesh
    coordinate and physical device.  This permits two wrappers over the same
    four buffers to share one release while rejecting ambiguous topology.
    """

    distribution_shape: tuple[int, int]
    members: tuple[tuple[tuple[int, int], int, int], ...]


def qwen38_tensor_backing_identity(tensor: Any) -> Qwen38TensorBackingIdentity:
    """Return the exact ordered 1x4 physical backing identity of ``tensor``.

    The function is deliberately fail closed.  A device object without the
    full tensor-topology and per-device buffer-address APIs cannot be safely
    deduplicated for deallocation and must remain owned until mesh teardown.
    """

    try:
        device = tensor.device()
        topology = tensor.tensor_topology()
        distribution_shape = _shape_tuple(
            topology.distribution_shape(),
            label="backing tensor distribution shape",
        )
        coordinates = tuple(
            _shape_tuple(coordinate, label=f"backing tensor mesh coordinate[{index}]")
            for index, coordinate in enumerate(tuple(topology.mesh_coords()))
        )
        mesh_shape = _shape_tuple(device.shape, label="backing tensor mesh-device shape")
        physical_ids = _shape_tuple(
            device.get_device_ids(),
            label="backing tensor mesh-device physical IDs",
        )
        local_tensors = tuple(ttnn.get_device_tensors(tensor))
    except BaseException as error:
        raise RuntimeError("cannot resolve exact distributed tensor backing identity") from error

    expected_coordinates = tuple((0, column) for column in range(TP_SIZE))
    if (
        distribution_shape != MESH_SHAPE
        or mesh_shape != MESH_SHAPE
        or coordinates != expected_coordinates
        or len(physical_ids) != TP_SIZE
        or len(set(physical_ids)) != TP_SIZE
        or any(physical_id < 0 for physical_id in physical_ids)
        or len(local_tensors) != TP_SIZE
    ):
        raise RuntimeError(
            "distributed tensor backing topology differs from exact 1x4: "
            f"distribution={distribution_shape} device={mesh_shape} coordinates={coordinates} "
            f"physical_ids={physical_ids} local_count={len(local_tensors)}"
        )

    members: list[tuple[tuple[int, int], int, int]] = []
    for index, (coordinate, expected_physical_id, local_tensor) in enumerate(
        zip(coordinates, physical_ids, local_tensors, strict=True)
    ):
        try:
            local_device = local_tensor.device()
            local_coordinates = tuple(
                _shape_tuple(item, label=f"backing tensor local device coordinate[{index}]")
                for item in tuple(local_tensor.device_coords())
            )
            physical_id = _exact_integer(
                device.get_device_id(ttnn.MeshCoordinate(*coordinate)),
                label=f"backing tensor parent-mesh physical ID[{index}]",
            )
            buffer_address = _exact_integer(
                local_tensor.buffer_address(),
                label=f"backing tensor local buffer address[{index}]",
            )
        except BaseException as error:
            raise RuntimeError(f"cannot resolve backing tensor member at mesh coordinate {coordinate}") from error
        if (
            local_device is not device
            or local_coordinates != (coordinate,)
            or physical_id != expected_physical_id
            or buffer_address < 0
        ):
            raise RuntimeError(
                f"backing tensor member {coordinate} resolved parent/coords/physical/address "
                f"{local_device is device}/{local_coordinates}/{physical_id}/{buffer_address}, "
                f"expected parent/coords/physical True/{(coordinate,)}/{expected_physical_id}"
            )
        members.append((coordinate, physical_id, buffer_address))
    return Qwen38TensorBackingIdentity(MESH_SHAPE, tuple(members))


def replicate_tensor_2d_mesh_mapper(mesh_device):
    """Replicate over both axes of the model's explicit ``(1, 4)`` topology."""

    return ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
            ttnn.MeshShape(*MESH_SHAPE),
        ),
    )


@dataclass(frozen=True)
class Qwen38MeshContract:
    """The physical and logical mesh admitted by the four-P150 model.

    ``physical_ids`` are the TTNN IDs returned by ``MeshDevice.get_device_ids``
    in mesh-coordinate order.  The caller must derive and record their node,
    BDF, board-id, and fabric mapping before opening the mesh.
    """

    physical_ids: tuple[int, int, int, int]
    mesh_shape: tuple[int, int] = MESH_SHAPE

    def __post_init__(self) -> None:
        canonical_mesh_shape = _shape_tuple(self.mesh_shape, label="mesh_shape", error_type=ValueError)
        if canonical_mesh_shape != MESH_SHAPE:
            raise ValueError(f"Qwen3.8-Flash-Next requires mesh {MESH_SHAPE}, got {self.mesh_shape}")
        if not isinstance(self.physical_ids, tuple):
            raise ValueError(f"expected four distinct physical IDs, got {self.physical_ids}")
        try:
            canonical_physical_ids = _shape_tuple(
                self.physical_ids,
                label="physical_ids",
                error_type=ValueError,
            )
        except ValueError as error:
            raise ValueError(f"expected four distinct physical IDs, got {self.physical_ids}") from error
        if (
            len(canonical_physical_ids) != TP_SIZE
            or len(set(canonical_physical_ids)) != TP_SIZE
            or any(device_id < 0 for device_id in canonical_physical_ids)
        ):
            raise ValueError(f"expected four distinct physical IDs, got {self.physical_ids}")
        object.__setattr__(self, "mesh_shape", canonical_mesh_shape)
        object.__setattr__(self, "physical_ids", canonical_physical_ids)

    def validate_mesh(self, mesh_device) -> None:
        actual_shape = _shape_tuple(mesh_device.shape, label="opened mesh shape")
        if actual_shape != MESH_SHAPE:
            raise RuntimeError(f"opened mesh shape {actual_shape} does not match leased {self.mesh_shape}")
        actual_count = _exact_integer(mesh_device.get_num_devices(), label="opened mesh device count")
        if actual_count != TP_SIZE:
            raise RuntimeError(f"opened mesh has {actual_count} devices, expected {TP_SIZE}")
        actual_ids = _shape_tuple(mesh_device.get_device_ids(), label="opened mesh physical IDs")
        if actual_ids != self.physical_ids:
            raise RuntimeError(
                f"opened mesh physical order {actual_ids} does not match leased order {self.physical_ids}"
            )

    def validate_tensor(
        self,
        tensor,
        *,
        placement: TensorPlacement,
        shard_dim: int | None = None,
        require_device: bool = True,
    ) -> None:
        """Validate the actual ``tensor_topology()``, not only its values.

        All model tensors use a 2-D distribution shape ``(1, 4)``.  A sharded
        tensor must replicate over the size-one row and shard over the four-way
        column; an intentionally replicated tensor must replicate on both axes.
        Pair-grouped KV is represented by an expanded two-head tensor sharded on
        the head dimension, not by a replicated topology.
        """

        device = tensor.device()
        if require_device:
            if device is None:
                raise RuntimeError(f"{placement.value} tensor is not device resident")
            self.validate_mesh(device)
        topology = tensor.tensor_topology()
        distribution_shape = _shape_tuple(
            topology.distribution_shape(),
            label=f"{placement.value} tensor distribution shape",
        )
        if distribution_shape != self.mesh_shape:
            raise RuntimeError(f"{placement.value} tensor distribution {distribution_shape} must be {self.mesh_shape}")
        raw_coords = tuple(topology.mesh_coords())
        coords = tuple(
            _shape_tuple(coord, label=f"{placement.value} tensor mesh coordinate[{index}]")
            for index, coord in enumerate(raw_coords)
        )
        expected_coords = tuple((0, column) for column in range(TP_SIZE))
        if coords != expected_coords:
            raise RuntimeError(f"{placement.value} tensor mesh-coordinate order {coords} must be {expected_coords}")

        placements = tuple(topology.placements())
        names = tuple(_placement_name(item) for item in placements)
        if len(placements) != 2:
            raise RuntimeError(f"{placement.value} tensor must have two placement axes, got {names}")
        if placement in (TensorPlacement.REPLICATED, TensorPlacement.LOCAL_PARTIAL):
            if names != ("PlacementReplicate", "PlacementReplicate"):
                raise RuntimeError(f"{placement.value} tensor has unexpected placements {names}")
            return

        if shard_dim is None:
            raise ValueError(f"shard_dim is required for {placement.value}")
        expected_dim = _exact_integer(shard_dim, label=f"{placement.value} shard_dim", error_type=ValueError)
        if names != ("PlacementReplicate", "PlacementShard"):
            raise RuntimeError(f"{placement.value} tensor has unexpected placements {names}")
        actual_dim = _exact_integer(
            getattr(placements[1], "dim"),
            label=f"{placement.value} tensor PlacementShard.dim",
        )
        if actual_dim != expected_dim:
            raise RuntimeError(f"{placement.value} tensor shards dim {actual_dim}, expected {expected_dim}")

    @staticmethod
    def require_equal_local_shapes(tensors: Iterable, *, label: str) -> tuple[int, ...]:
        shapes = tuple(_shape_tuple(tensor.shape, label=f"{label} tensor shape") for tensor in tensors)
        if not shapes:
            raise ValueError(f"{label} has no tensors")
        if len(set(shapes)) != 1:
            raise RuntimeError(f"{label} local shapes differ across the mesh: {shapes}")
        return shapes[0]

    def mark_local_partial(self, tensor, *, replicated_reference, expected_shape: tuple[int, ...]) -> None:
        """Label a full-shape per-device additive partial without copying values.

        TTNN has no placement kind for values which have the complete local shape
        but differ by device pending an all-reduce/reduce-scatter.  Such tensors
        must carry Replicate *shape* metadata so the following collective sees a
        full logical operand; ``LOCAL_PARTIAL`` records the stronger semantic
        invariant in this model.  This method is only valid after a row-parallel
        operation whose local output width is complete.
        """

        self.validate_tensor(replicated_reference, placement=TensorPlacement.REPLICATED)
        if _shape_tuple(tensor.shape) != _shape_tuple(expected_shape):
            raise RuntimeError(
                f"local partial shape {_shape_tuple(tensor.shape)} does not match expected {expected_shape}"
            )
        reference_topology = replicated_reference.tensor_topology()
        topology = ttnn.TensorTopology(
            reference_topology.distribution_shape(),
            [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
            reference_topology.mesh_coords(),
        )
        tensor.update_tensor_topology(topology)
        self.validate_tensor(tensor, placement=TensorPlacement.LOCAL_PARTIAL)

    def mark_collective_shard(
        self,
        tensor,
        *,
        replicated_reference,
        shard_dim: int,
        expected_local_shape: tuple[int, ...],
    ) -> None:
        """Repair the known legacy reduce-scatter topology omission.

        Standard ``ttnn.reduce_scatter`` changes the physical/local tensor shape
        but currently has no ``compute_output_topologies`` hook.  Call this only
        immediately after that collective; it records the actual shard placement
        without moving or replicating data.
        """

        canonical_shard_dim = _exact_integer(shard_dim, label="collective shard_dim", error_type=ValueError)
        self.validate_tensor(replicated_reference, placement=TensorPlacement.REPLICATED)
        if _shape_tuple(tensor.shape) != _shape_tuple(expected_local_shape):
            raise RuntimeError(
                f"collective shard shape {_shape_tuple(tensor.shape)} does not match expected "
                f"{expected_local_shape}"
            )
        reference_topology = replicated_reference.tensor_topology()
        topology = ttnn.TensorTopology(
            reference_topology.distribution_shape(),
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(canonical_shard_dim)],
            reference_topology.mesh_coords(),
        )
        tensor.update_tensor_topology(topology)
        self.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=canonical_shard_dim)


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def same_buffer(left, right) -> bool:
    """Whether two device tensors are one allocation: a zero-cost view (``ttnn.reshape`` of a ROW_MAJOR tensor, or of
    a TILE tensor that keeps its last dimension) and its source share a buffer under different tensor ids.  This is
    the test that decides whether a reshape's SOURCE may be freed while the result is still read: freeing the source
    of a view frees the result's rows (the 2026-09-26 slab defect, docs/NUMERICS.md); ``_tensor_key`` compares tensor
    ids, which a view never shares, and must not decide it."""

    left_address, right_address = getattr(left, "buffer_address", None), getattr(right, "buffer_address", None)
    if not callable(left_address) or not callable(right_address):
        # a host stand-in (the no-device tests' fakes) has no device buffer and nothing to alias: the id test decides
        return _tensor_key(left) == _tensor_key(right)
    return int(left_address()) == int(right_address())


def _host_uint32(shape: tuple[int, ...], value, *, label: str) -> torch.Tensor:
    value = _exact_integer(value, label=label, error_type=ValueError)
    if not 0 <= value < UINT32_LIMIT:
        raise ValueError(f"{label} must be in [0,{UINT32_LIMIT}), got {value}")
    return torch.full(shape, value, dtype=torch.int64).to(torch.uint32)


def _require_uint32_row(tensor, shape: tuple[int, ...], mesh_contract: Qwen38MeshContract, *, label: str) -> None:
    if _shape_tuple(tensor.shape) != shape or tensor.dtype != ttnn.uint32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"{label} must be UINT32 ROW_MAJOR {list(shape)}, got {tensor_metadata(tensor)}")
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)


@dataclass(frozen=True)
class Qwen38TTNNDevicePosition:
    """Device-resident absolute position ``P`` of the token a generic decode body decodes.

    ``scalar`` is the only source of the position inside the captured body: it
    is read at the start, every derived quantity comes from exact UINT32 device
    ops, and :meth:`advance` increments it in place as the body's last op so a
    replay alone moves to the next position.  The host writes it only through
    :meth:`reset`.  All three tensors keep their addresses for the life of the
    object.
    """

    scalar: Any
    ones_row: Any
    block_start_mask_row: Any
    mesh_device: Any = field(repr=False, compare=False)
    mesh_contract: Qwen38MeshContract = field(repr=False, compare=False)
    # QWEN38_FUSED=position_advance binds ttnn/fused/position_derive.advance (P += count as one in-place program) in
    # allocate(); None runs the chain's add + copy.
    _fused_advance: Any = field(default=None, repr=False, compare=False)
    # The rotary shift ``S`` (UINT32 ``[1,1,1,1]``): the rotary index of the token at ``P`` is ``P - S`` on every
    # axis (``mrope.Qwen38MRoPEPositions.shift``: 0 for text, the images' accumulated ``max(h, w) // 2 - h * w // 4``
    # otherwise), ``S <= P & ~3``.  Host-written with the position (:meth:`reset`); the cache, the KV row, the block
    # index and the masks stay on ``P``.  ``rope_shift`` is its host mirror.  None: a position built without one (a
    # text-only stand-in): the rotary rows are the plain rows.
    shift: Any = None
    rope_shift: int = 0

    @classmethod
    def allocate(
        cls, mesh_device, mesh_contract: Qwen38MeshContract, *, position: int = 0, rope_shift: int = 0
    ) -> Qwen38TTNNDevicePosition:
        mesh_contract.validate_mesh(mesh_device)
        require_rope_shift(rope_shift, position)
        uploaded: list[Any] = []
        try:
            for shape, value, label in (
                (POSITION_SCALAR_SHAPE, position, "device position"),
                (POSITION_INDEX_ROW_SHAPE, 1, "device position ones row"),
                (POSITION_INDEX_ROW_SHAPE, BLOCK_START_LANE_MASK, "device position block-start mask row"),
                (POSITION_SCALAR_SHAPE, rope_shift, "device rotary shift"),
            ):
                tensor = ttnn.from_torch(
                    _host_uint32(shape, value, label=label),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
                )
                uploaded.append(tensor)
                _require_uint32_row(tensor, shape, mesh_contract, label=label)
        except BaseException:
            for tensor in uploaded:
                ttnn.deallocate(tensor)
            raise
        from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_kernels

        fused_advance = None
        if fused_kernels.enabled("position_advance"):
            fused_advance = fused_kernels.kernel("position_advance").fused
        return cls(
            uploaded[0], uploaded[1], uploaded[2], mesh_device, mesh_contract, fused_advance, uploaded[3], rope_shift
        )

    def reset(self, position: int, rope_shift: int | None = None) -> None:
        """Host write of ``P`` and of the rotary shift ``S`` (outside any trace); the only host path into ``scalar``
        and ``shift``.  ``rope_shift`` None keeps the mirrored shift (a position rewind inside one request); a new
        sequence passes its shift (0 for text)."""

        if rope_shift is None:
            rope_shift = self.rope_shift
        require_rope_shift(rope_shift, position)
        if self.shift is None and rope_shift:
            raise RuntimeError("this device position carries no rotary shift tensor")
        host = ttnn.from_torch(
            _host_uint32(POSITION_SCALAR_SHAPE, position, label="device position"),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self.scalar)
        if rope_shift != self.rope_shift:  # the shift changes only between sequences: one more write then
            shift_host = ttnn.from_torch(
                _host_uint32(POSITION_SCALAR_SHAPE, rope_shift, label="device rotary shift"),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(shift_host, self.shift)
            object.__setattr__(self, "rope_shift", rope_shift)  # the frozen dataclass's mirror (as the fused hook)

    def advance(self) -> None:
        """In-trace ``P += 1``; must be the last op of the model body."""

        if self._fused_advance is not None:
            self._advance_fused(1)
            return
        key = _tensor_key(self.scalar)
        advanced = ttnn.add(self.scalar, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        copied = ttnn.copy(advanced, self.scalar)
        if _tensor_key(self.scalar) != key or (copied is not None and _tensor_key(copied) != key):
            raise RuntimeError("device position advance did not write the resident scalar in place")
        ttnn.deallocate(advanced)

    def _advance_fused(self, count: int) -> None:
        """``P += count`` as one in-place program (ttnn/fused/position_derive.advance), the chain's add + copy."""

        key = _tensor_key(self.scalar)
        written = self._fused_advance(self.scalar, count)
        if _tensor_key(self.scalar) != key or (written is not None and _tensor_key(written) != key):
            raise RuntimeError("device position advance did not write the resident scalar in place")

    def advance_by(self, count: int) -> None:
        """In-trace ``P += count`` (the prefill chunk's ``CHUNK_ROWS``); must be the last op of the chunk body."""

        if isinstance(count, bool) or type(count) is not int or count <= 0:
            raise ValueError(f"device position advance needs a positive int, got {count!r}")
        if self._fused_advance is not None:
            self._advance_fused(count)
            return
        key = _tensor_key(self.scalar)
        advanced = ttnn.add(self.scalar, count, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        copied = ttnn.copy(advanced, self.scalar)
        if _tensor_key(self.scalar) != key or (copied is not None and _tensor_key(copied) != key):
            raise RuntimeError("device position advance did not write the resident scalar in place")
        ttnn.deallocate(advanced)

    def index_row(self):
        """``[1,1,1,32]`` UINT32 row with every lane ``= P`` (an embedding index row after reshape)."""

        row = ttnn.multiply(self.ones_row, self.scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _require_uint32_row(row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device position index row")
        return row

    def block_start_index_row(self, index_row):
        """``[1,1,1,32]`` UINT32 row with every lane ``= P & ~3`` (the current index block's first position)."""

        row = ttnn.bitwise_and(index_row, self.block_start_mask_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _require_uint32_row(row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device position block-start row")
        return row

    def rope_index_row(self, index_row):
        """``[1,1,1,32]`` UINT32 row with every lane ``= P - S``: the RoPE table row of the token at ``P`` (an exact
        UINT32 subtract; ``S <= P`` by construction of the shift)."""

        if self.shift is None:  # no shift tensor: the plain row, as a fresh tensor (the callers release it)
            row = ttnn.add(index_row, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            row = ttnn.subtract(index_row, self.shift, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _require_uint32_row(row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device rotary index row")
        return row

    def rope_block_start_index_row(self, index_row):
        """``[1,1,1,32]`` UINT32 row with every lane ``= (P & ~3) - S``: the RoPE table row of the token at the
        current index block's first position (a text token of this sequence, so its shift is ``S``)."""

        block_start_row = self.block_start_index_row(index_row)
        if self.shift is None:
            return block_start_row
        row = ttnn.subtract(block_start_row, self.shift, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(block_start_row)
        _require_uint32_row(row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device rotary block-start row")
        return row

    def read(self) -> int:
        """Diagnostic readback of ``P`` from coordinate 0 (outside any trace)."""

        return int(ttnn.to_torch(ttnn.get_device_tensors(self.scalar)[0]).reshape(-1)[0].item())

    def read_shift(self) -> int:
        """Diagnostic readback of ``S`` from coordinate 0 (outside any trace)."""

        return int(ttnn.to_torch(ttnn.get_device_tensors(self.shift)[0]).reshape(-1)[0].item())

    def deallocate(self) -> None:
        for tensor in (self.scalar, self.ones_row, self.block_start_mask_row, self.shift):
            if tensor is not None:
                ttnn.deallocate(tensor)


def require_rope_shift(rope_shift, position) -> int:
    """The rotary shift: an exact int in ``[0, P & ~3]``, so neither the token's rotary index ``P - S`` nor its index
    block's first token's ``(P & ~3) - S`` wraps (the block's first token is a text token of this sequence whose
    rotary position is non-negative; ``mrope.Qwen38MRoPEPositions.tail_is_plain`` is the prompt-side form of the
    same rule)."""

    value = _exact_integer(rope_shift, label="rotary shift", error_type=ValueError)
    position = _exact_integer(position, label="device position", error_type=ValueError)
    limit = max(position, 0) & BLOCK_START_LANE_MASK
    if not 0 <= value <= limit:
        raise ValueError(f"rotary shift must be in [0, {limit}] (the position's index block start), got {value}")
    return value


@dataclass
class Qwen38TTNNDevicePositionRow:
    """Device-resident per-lane positions: the ``[1,1,1,32]`` UINT32 row with lane ``u = P_u``.

    The batched form of :class:`Qwen38TTNNDevicePosition`: the row itself is the
    embedding index row of the RoPE lookup (lane ``u`` reads row ``P_u``), the
    per-lane QSA inputs are derived from it in column form (``qsa.derive_qsa_lane_inputs``),
    and :meth:`advance` adds 1 to every lane in place as the body's last op.  All
    resident lanes share ``P_u mod 4`` (``GDN_RESIDUE_CLASSES``): one trace and one
    host ring slot serve the step, so a lane joins only at a step whose residue is
    its position's (:func:`admission_wait_steps`).  Lanes ``lanes..31`` are idle and
    hold the row's residue (a valid position of the class; their regions receive
    harmless writes).  ``positions`` is the host mirror the host writes come from;
    the 1-row class is untouched.
    """

    row: Any
    block_start_mask_row: Any
    lanes: int
    positions: list[int]
    mesh_device: Any = field(repr=False, compare=False)
    mesh_contract: Qwen38MeshContract = field(repr=False, compare=False)
    # Lane u's rotary shift ``S_u`` (UINT32 ``[1,1,1,32]``; idle lanes 0; ``S_u <= P_u & ~3``) and its host mirror:
    # the RoPE table row of lane u's token at ``P_u`` is ``P_u - S_u`` (see ``Qwen38TTNNDevicePosition.shift``).
    # None: a row built without one (a text-only stand-in): the rotary rows are the plain rows.
    shift_row: Any = None
    shifts: list[int] = field(default_factory=lambda: [0] * MAX_LANES)

    @staticmethod
    def _lane_shifts(shifts, positions: list[int], *, lanes: int) -> list[int]:
        """The 32 lane shifts (active lanes as given, 0 when None, idle lanes 0), each in ``[0, P_u]``."""

        values = [0] * lanes if shifts is None else [int(value) for value in shifts]
        if len(values) != lanes:
            raise ValueError(f"shift row needs one shift per active lane: got {len(values)}, expected {lanes}")
        for lane, value in enumerate(values):
            require_rope_shift(value, positions[lane])
        return values + [0] * (MAX_LANES - lanes)

    @staticmethod
    def _lane_positions(positions, *, lanes: int) -> list[int]:
        """The 32 lane values (active lanes as given, idle lanes at the shared residue), residue checked."""

        values = [
            _exact_integer(value, label=f"lane {index} position", error_type=ValueError)
            for index, value in enumerate(positions)
        ]
        if len(values) != lanes:
            raise ValueError(f"position row needs one position per active lane: got {len(values)}, expected {lanes}")
        for lane, value in enumerate(values):
            if not 0 <= value < UINT32_LIMIT:
                raise ValueError(f"lane {lane} position must be in [0,{UINT32_LIMIT}), got {value}")
        residue = values[0] % GDN_RESIDUE_CLASSES
        for lane, value in enumerate(values):
            if value % GDN_RESIDUE_CLASSES != residue:
                raise ValueError(
                    f"lane {lane} position {value} has residue {value % GDN_RESIDUE_CLASSES}, expected the row's "
                    f"residue {residue} (lane 0 at {values[0]}); wait {admission_wait_steps(residue, value)} steps"
                )
        return values + [residue] * (MAX_LANES - lanes)

    @staticmethod
    def _host_row(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64).reshape(POSITION_INDEX_ROW_SHAPE).to(torch.uint32)

    @classmethod
    def allocate(
        cls, mesh_device, mesh_contract: Qwen38MeshContract, positions, *, lanes: int, shifts=None
    ) -> Qwen38TTNNDevicePositionRow:
        mesh_contract.validate_mesh(mesh_device)
        lanes = require_lane_count(lanes, label="position row lanes")
        values = cls._lane_positions(positions, lanes=lanes)
        shift_values = cls._lane_shifts(shifts, values, lanes=lanes)
        uploaded: list[Any] = []
        try:
            for host, label in (
                (cls._host_row(values), "device position row"),
                (
                    _host_uint32(
                        POSITION_INDEX_ROW_SHAPE, BLOCK_START_LANE_MASK, label="device position block-start mask row"
                    ),
                    "device position block-start mask row",
                ),
                (cls._host_row(shift_values), "device rotary shift row"),
            ):
                tensor = ttnn.from_torch(
                    host,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
                )
                uploaded.append(tensor)
                _require_uint32_row(tensor, POSITION_INDEX_ROW_SHAPE, mesh_contract, label=label)
        except BaseException:
            for tensor in uploaded:
                ttnn.deallocate(tensor)
            raise
        return cls(uploaded[0], uploaded[1], lanes, values, mesh_device, mesh_contract, uploaded[2], shift_values)

    @property
    def residue(self) -> int:
        return self.positions[0] % GDN_RESIDUE_CLASSES

    def validate(self) -> None:
        if len(self.positions) != MAX_LANES or len(self.shifts) != MAX_LANES:
            raise RuntimeError(f"position row mirror holds {len(self.positions)} lanes, expected {MAX_LANES}")
        self._lane_positions(self.positions[: self.lanes], lanes=self.lanes)
        self._lane_shifts(self.shifts[: self.lanes], self.positions, lanes=self.lanes)
        _require_uint32_row(self.row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device position row")
        if self.shift_row is not None:
            _require_uint32_row(
                self.shift_row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device rotary shift row"
            )

    def _write_mirror(self) -> None:
        """Host write of the whole position row and shift row from the mirrors (outside any trace); the only host
        path into ``row`` and ``shift_row``."""

        targets = [(self.positions, self.row)]
        if self.shift_row is not None:
            targets.append((self.shifts, self.shift_row))
        elif any(self.shifts):
            raise RuntimeError("this position row carries no rotary shift row")
        for values, tensor in targets:
            host = ttnn.from_torch(
                self._host_row(values),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(host, tensor)

    def reset(self, positions, shifts=None) -> None:
        """Rewrite every active lane (a new batch); the lanes must share one residue.  ``shifts`` are the lanes'
        rotary shifts (None: text, all 0)."""

        self.positions = self._lane_positions(positions, lanes=self.lanes)
        self.shifts = self._lane_shifts(shifts, self.positions, lanes=self.lanes)
        self._write_mirror()

    def admit(self, lane: int, position: int, rope_shift: int = 0) -> None:
        """Rewrite one lane at a step of its residue class: ``position mod 4`` must equal :attr:`residue`.
        ``rope_shift`` is the lane's rotary shift (0 for a text prompt)."""

        lane = _exact_integer(lane, label="admitted lane", error_type=ValueError)
        if not 0 <= lane < self.lanes:
            raise ValueError(f"admitted lane must be in [0,{self.lanes}), got {lane}")
        value = _exact_integer(position, label="admitted position", error_type=ValueError)
        if not 0 <= value < UINT32_LIMIT:
            raise ValueError(f"admitted position must be in [0,{UINT32_LIMIT}), got {value}")
        wait = admission_wait_steps(self.residue, value)
        if wait:
            raise ValueError(
                f"lane {lane} admission at position {value} has residue {value % GDN_RESIDUE_CLASSES}, expected the "
                f"row's residue {self.residue}; admit it {wait} steps later"
            )
        shift = require_rope_shift(rope_shift, value)
        self.positions[lane] = value
        self.shifts[lane] = shift
        self._write_mirror()

    def advance(self) -> None:
        """In-trace ``P_u += 1`` for every lane; must be the last op of the model body."""

        key = _tensor_key(self.row)
        advanced = ttnn.add(self.row, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        copied = ttnn.copy(advanced, self.row)
        if _tensor_key(self.row) != key or (copied is not None and _tensor_key(copied) != key):
            raise RuntimeError("device position row advance did not write the resident row in place")
        ttnn.deallocate(advanced)
        self.positions = [value + 1 for value in self.positions]

    def advance_replayed(self) -> None:
        """The mirror's ``P_u += 1`` after a captured body replayed: the trace advanced the device row without the
        Python body, so the host that schedules admissions against :attr:`residue` calls this once per replay."""

        self.positions = [value + 1 for value in self.positions]

    def index_row(self):
        """A fresh ``[1,1,1,32]`` UINT32 copy of the row (lane ``u = P_u``): the RoPE embedding index row."""

        row = ttnn.add(self.row, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _require_uint32_row(row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device position lane index row")
        return row

    def block_start_index_row(self, index_row):
        """``[1,1,1,32]`` UINT32 row with lane ``u = P_u & ~3`` (lane ``u``'s current index block start)."""

        row = ttnn.bitwise_and(index_row, self.block_start_mask_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _require_uint32_row(
            row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device position lane block-start row"
        )
        return row

    def rope_index_row(self, index_row):
        """``[1,1,1,32]`` UINT32 row with lane ``u = P_u - S_u``: lane u's RoPE table row."""

        if self.shift_row is None:
            row = ttnn.add(index_row, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            row = ttnn.subtract(index_row, self.shift_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _require_uint32_row(row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device rotary lane index row")
        return row

    def rope_block_start_index_row(self, index_row):
        """``[1,1,1,32]`` UINT32 row with lane ``u = (P_u & ~3) - S_u``: the RoPE table row of lane u's block start."""

        block_start_row = self.block_start_index_row(index_row)
        if self.shift_row is None:
            return block_start_row
        row = ttnn.subtract(block_start_row, self.shift_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(block_start_row)
        _require_uint32_row(
            row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device rotary lane block-start row"
        )
        return row

    def read(self) -> list[int]:
        """Diagnostic readback of the 32 lanes from coordinate 0 (outside any trace)."""

        values = ttnn.to_torch(ttnn.get_device_tensors(self.row)[0]).reshape(-1).to(torch.int64) & (UINT32_LIMIT - 1)
        return [int(value) for value in values.tolist()]

    def deallocate(self) -> None:
        for tensor in (self.row, self.block_start_mask_row, self.shift_row):
            if tensor is not None:
                ttnn.deallocate(tensor)
