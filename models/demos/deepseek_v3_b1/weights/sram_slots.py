# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side orchestration for SRAM hot expert weight slots.

This module owns everything that turns a ranked candidate list of routed
experts into a populated :class:`SramCompressedExpertSlots` on device:

  * the four per-projection per-core L1 ``CompressedTensor`` builders
    (``_build_l1_compressed_tensor*``) plus the cache-vs-direct dispatcher
    :func:`_route_l1_compressed_tensor`,
  * the address-based trim machinery (:func:`_compute_sram_trim_budget`,
    :func:`_refresh_lowest_addr_from_alloc`, the per-core L1 footprint /
    lowest-address introspection helpers),
  * the main entry point :func:`prepare_compressed_sram_slots` that walks
    candidates, predicts each expert's per-core cost via
    :func:`_predict_expert_per_core_bytes`, and stops at the first expert
    whose footprint would push any tracked core's lowest L1 address below
    ``boundary_addr``.

Pure host-side preprocessing math (assignments, byte-cost prediction,
routing-frequency ranking) lives in
:mod:`models.demos.deepseek_v3_b1.weights.transforms.sram_experts`.  Disk
caching lives in
:mod:`models.demos.deepseek_v3_b1.weights.cache.sram_compressed_cache`.
This module ties the two together with the on-device allocator.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.weights.cache import CacheConfig, SourceTensorSelection, SramCompressedTensorTarget
from models.demos.deepseek_v3_b1.weights.cache.sram_compressed_cache import (
    assigner_fingerprint,
    get_or_create_sram_compressed_expert,
    to_canonical_mesh_mapper,
)
from models.demos.deepseek_v3_b1.weights.overlap.spec import _core_list
from models.demos.deepseek_v3_b1.weights.specs.overlap_configs import GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
from models.demos.deepseek_v3_b1.weights.transforms.moe import preprocess_gate_up, shared_down_torch_for_cache
from models.demos.deepseek_v3_b1.weights.transforms.sram_experts import (
    _SHARED_EXPERT_DOWN_K_PER_DEV,
    _SHARED_EXPERT_DOWN_N_PER_CORE,
    SramExpertCoreGrids,
    _predict_expert_per_core_bytes,
)


@dataclass
class SramCompressedExpertSlots:
    """SRAM (L1) hot expert slots using per-core CompressedTensor.

    Each slot holds one expert's gate/up/down projection weights as a per-core
    L1 CompressedTensor.  These CTs are directly usable as ``sram_cts`` in
    :class:`~micro_ops.matmul_expert.op.ExpertKernel`.

    Slot management is decoupled from the MoE op itself — populate slots at
    weight preparation time, then pass the CTs through when combining.
    """

    # Already device-resident per-core L1 CompressedTensors; skip the
    # host->device upload walk in `two_phase_upload`.
    __upload_passthrough__ = True

    num_slots: int
    slot_experts: list[int]
    gate_proj: list[CompressedTensor]
    up_proj: list[CompressedTensor]
    down_proj: list[CompressedTensor]

    def is_dram_flags(self, num_total_experts: int) -> list[int]:
        """Build SRAM/DRAM routing flags for ``create_expert_selection_meta``.

        Returns a list of length *num_total_experts* where SRAM-resident
        experts are ``0`` and all others are ``1``.
        """
        flags = [1] * num_total_experts
        for eid in self.slot_experts:
            flags[eid] = 0
        return flags


# Combined per-core cap for persistent attention weights + SRAM-hot experts.
# Applies to every core in the SRAM binding set (gate ∪ up ∪ down grids):
# ``attn_bytes[c] + sram_hot_expert_bytes[c] <= _COMBINED_ATTN_SRAM_CAP_BYTES``.
# Shared expert L1 weights are intentionally *not* counted here -- they are
# staged after the SRAM trim runs and land below the SRAM band in the
# ``worker_l1_size - cap`` reserve along with runtime scratch (CBs, activation
# shards, allocator bookkeeping).  Raise the cap only if scratch + shared
# headroom measurements confirm it's safe.
_COMBINED_ATTN_SRAM_CAP_BYTES = 960 * 1024


# Dataclass field names of the shared expert L1 weights inside
# ``DeepSeekV3MoELayerWeights`` -- skipped by ``eager_upload_l1_lockstep`` so
# they don't contribute to the SRAM trim's per-core ``initial_lowest_addr``.
_SHARED_EXPERT_FIELDS = ("shared_gate_proj", "shared_up_proj", "shared_down_proj")


# ---------------------------------------------------------------------------
# Per-core L1 footprint / lowest-address introspection helpers
# ---------------------------------------------------------------------------


def _tensor_l1_per_core_footprint(tt_tensor: ttnn.Tensor) -> tuple[int, list[tuple[int, int]]]:
    """Return ``(per_core_bytes, cores)`` for an L1-sharded ``ttnn.Tensor``.

    Returns ``(0, [])`` when the tensor is not in L1 or not sharded.  Handles
    both tile-layout tensors (bfp8 / bfp4 / bfloat16) and the row-major uint32
    backing of fused ``OverlappedTensor`` buffers.
    """
    mc = tt_tensor.memory_config()
    if mc.buffer_type != ttnn.BufferType.L1 or mc.shard_spec is None:
        return 0, []
    shape = mc.shard_spec.shape
    cores = _core_list(mc.shard_spec.grid)
    if tt_tensor.layout == ttnn.TILE_LAYOUT:
        tile = tt_tensor.tile
        tile_h, tile_w = tile.tile_shape
        per_core_bytes = (shape[0] // tile_h) * (shape[1] // tile_w) * tile.get_tile_size(tt_tensor.dtype)
    else:
        # Fused OverlappedTensor buffers land here: uint32 backing + row-major
        # WIDTH_SHARDED.  ``shape`` is in uint32 words so 4 bytes per element.
        elem_bytes = 4 if tt_tensor.dtype == ttnn.uint32 else tt_tensor.element_size()
        per_core_bytes = shape[0] * shape[1] * elem_bytes
    return per_core_bytes, cores


def per_core_l1_usage_on_cores(
    tensors: list[Any],
    target_cores: set[tuple[int, int]] | list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Per-core L1 bytes already reserved on each core in ``target_cores``.

    Walks a flat list of ``OverlappedTensor`` / ``ttnn.Tensor`` (``None``
    entries are ignored) and returns a dict mapping ``(x, y) -> bytes`` for
    every core in ``target_cores`` that carries at least one reservation.
    Fused buffers shared between multiple ``OverlappedTensor`` views are
    deduped by underlying ``fused_tensor.tensor_id`` so the lockstep L1
    reservation is counted once.

    Only the *reservation* is counted, not per-subtensor offsets -- this
    matches how the allocator sees lockstep L1 blocks on Tenstorrent devices.
    Cores in ``target_cores`` that hold no reservation are omitted from the
    returned dict; callers should default missing keys to 0.
    """
    target = set(target_cores)
    seen_fused: set[int] = set()
    per_core: dict[tuple[int, int], int] = {}
    for obj in tensors:
        if obj is None:
            continue
        if hasattr(obj, "fused_tensor"):
            fused = obj.fused_tensor
            tid = fused.tensor_id
            if tid in seen_fused:
                continue
            seen_fused.add(tid)
            tt_tensor = fused
        elif isinstance(obj, ttnn.Tensor):
            tt_tensor = obj
        else:
            continue
        per_core_bytes, cores = _tensor_l1_per_core_footprint(tt_tensor)
        if per_core_bytes == 0:
            continue
        for c in cores:
            if c in target:
                per_core[c] = per_core.get(c, 0) + per_core_bytes
    return per_core


def max_per_core_l1_usage_on_cores(
    tensors: list[Any],
    target_cores: set[tuple[int, int]] | list[tuple[int, int]],
) -> int:
    """Max L1 bytes already reserved on any core in ``target_cores``.

    Scalar wrapper around :func:`per_core_l1_usage_on_cores`; returns 0 when
    no reservation touches ``target_cores``.
    """
    per_core = per_core_l1_usage_on_cores(tensors, target_cores)
    return max(per_core.values(), default=0)


def per_core_l1_lowest_addr_on_cores(
    tensors: list[Any],
    target_cores: set[tuple[int, int]] | list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Per-core lowest occupied L1 address across the given tensors.

    Walks the same flat list as :func:`per_core_l1_usage_on_cores`
    (``OverlappedTensor`` / ``ttnn.Tensor``; ``None`` ignored) and queries
    each tensor's on-device buffer address.  Returns ``dict[(x, y) -> int]``
    mapping each core in ``target_cores`` that holds at least one allocation
    to the **lowest** address occupied by any tracked tensor on that core.

    Top-down L1 allocation means a smaller address ⇒ more memory consumed.
    Cores in ``target_cores`` with no tracked allocation are omitted from
    the returned dict; callers should treat them as having unbounded
    headroom.

    Lockstep tensors (the common case for attention weights) report the
    same address on every core in their grid via ``buffer_address()``.
    Per-core-allocated tensors report a distinct address per core via
    ``experimental_per_core_buffer_address(core)``.

    Tensors that are not on device (host-only) are silently skipped --
    address queries are only meaningful post-allocation.
    """
    target = set(target_cores)
    seen_fused: set[int] = set()
    lowest: dict[tuple[int, int], int] = {}
    for obj in tensors:
        if obj is None:
            continue
        if hasattr(obj, "fused_tensor"):
            fused = obj.fused_tensor
            tid = fused.tensor_id
            if tid in seen_fused:
                continue
            seen_fused.add(tid)
            tt_tensor = fused
        elif isinstance(obj, ttnn.Tensor):
            tt_tensor = obj
        else:
            continue
        if not ttnn.is_tensor_storage_on_device(tt_tensor):
            continue
        mc = tt_tensor.memory_config()
        if mc.buffer_type != ttnn.BufferType.L1 or mc.shard_spec is None:
            continue
        cores = _core_list(mc.shard_spec.grid)
        is_per_core = bool(getattr(tt_tensor, "is_per_core_allocated", lambda: False)())
        for c_xy in cores:
            if c_xy not in target:
                continue
            if is_per_core:
                addr = tt_tensor.experimental_per_core_buffer_address(ttnn.CoreCoord(c_xy[0], c_xy[1]))
            else:
                addr = tt_tensor.buffer_address()
            prev = lowest.get(c_xy)
            if prev is None or addr < prev:
                lowest[c_xy] = addr
    return lowest


# ---------------------------------------------------------------------------
# Per-core L1 CompressedTensor builders (cache-aware)
# ---------------------------------------------------------------------------


def _route_l1_compressed_tensor(
    weight: torch.Tensor,
    *,
    memory_config,
    per_core_allocation: bool,
    mesh_mapper_config,
    assigner: CompressedTensorAssigner | None = None,
    assignment: np.ndarray | None = None,
    device,
    tile_hw: int,
    cache_config: CacheConfig | None,
    sram_cache_target_name: str | None,
    sram_cache_source_keys: tuple[str, ...] | None,
) -> CompressedTensor:
    """Build one per-core L1 CompressedTensor, optionally routing through the SRAM disk cache.

    When ``cache_config`` is a :class:`TensorCache`-backed config and a
    cache target name + source keys are provided, the call is dispatched to
    :func:`get_or_create_sram_compressed_expert` — warm runs hydrate the
    CompressedTensor from a content-addressed object on disk and skip the
    BFP pack pipeline entirely.  Otherwise, falls back to the legacy direct
    construction (``CompressedTensor.from_torch`` / ``from_bspm``) so
    behaviour is unchanged when no cache is wired in.
    """
    assert (assigner is None) != (assignment is None), "Provide exactly one of assigner / assignment"
    use_cache = cache_config is not None and sram_cache_target_name is not None and sram_cache_source_keys is not None
    if not use_cache:
        # Direct cold construction — preserves pre-cache behaviour for
        # callers that don't wire in a CacheConfig (unit tests, ephemeral
        # paths).  Mirrors the original ``_build_l1_compressed_tensor*``
        # branches; the BSPM path here historically does **not** request
        # ``per_core_allocation`` / ``mesh_mapper_config`` (TP=1 only),
        # so we keep that quirk here.
        if assigner is not None:
            return CompressedTensor.from_torch(
                weight.float(),
                assigner,
                device=device,
                memory_config=memory_config,
                per_core_allocation=per_core_allocation,
                mesh_mapper_config=mesh_mapper_config,
            )
        return CompressedTensor.from_bspm(
            weight.float(),
            assignment,
            device=device,
            memory_config=memory_config,
        )

    target = SramCompressedTensorTarget(
        name=sram_cache_target_name,
        tensor_shape=tuple(int(d) for d in weight.shape),
        tile_hw=tile_hw,
        memory_config=memory_config,
        per_core_allocation=per_core_allocation,
        mesh_mapper_config=to_canonical_mesh_mapper(mesh_mapper_config),
        assigner_fingerprint=assigner_fingerprint(assigner) if assigner is not None else "",
        assignment_hash=hashlib.sha256(np.ascontiguousarray(assignment).tobytes()).hexdigest()[:16]
        if assignment is not None
        else "",
    )
    fp = cache_config.context.fingerprint(
        source=SourceTensorSelection(names=tuple(sram_cache_source_keys)),
        target=target,
    )
    return get_or_create_sram_compressed_expert(
        cache_config.cache,
        fp,
        device,
        weight_provider=lambda w=weight: w,
        assigner=assigner,
        assignment=assignment,
        memory_config=memory_config,
        per_core_allocation=per_core_allocation,
        mesh_mapper_config=mesh_mapper_config,
        tile_hw=tile_hw,
    )


def _build_l1_compressed_tensor(
    weight: torch.Tensor,
    core_grid: ttnn.CoreRangeSet,
    *,
    assigner: CompressedTensorAssigner | None = None,
    assignment: np.ndarray | None = None,
    device=None,
    mesh_mapper_config=None,
    tile_hw: int = 32,
    cache_config: CacheConfig | None = None,
    sram_cache_target_name: str | None = None,
    sram_cache_source_keys: tuple[str, ...] | None = None,
) -> CompressedTensor:
    """Create a single per-core L1 CompressedTensor for one expert projection.

    Uses WIDTH_SHARDED on *core_grid*: each core holds (K, N_per_core) where
    N_per_core = N / num_cores.  Exactly one of *assigner* or *assignment* must
    be provided.

    Args:
        weight: Float weight tensor.  For single-device: ``(K, N)``.
                For multi-device with ``PlacementReplicate``: ``(K, N)``.
                For multi-device with ``PlacementShard`` (e.g. shared-expert-
                style TP8): ``(mesh_rows, mesh_cols, K_per_device, N_per_device)``
                where K/N have already been split along the mesh dims.
        core_grid: L1 compute cores for the shard.  ``N_per_device`` must be
                divisible by ``num_cores`` and the resulting ``per_core_N``
                must be tile-aligned (``% tile_hw == 0``).
        assigner: ``CompressedTensorAssigner`` (mutually exclusive with *assignment*).
        assignment: Pre-computed BSPM tile-level assignment array (mutually
                    exclusive with *assigner*).
        device: Device/mesh to allocate on (``None`` keeps host-only).
        mesh_mapper_config: ``ttnn.MeshMapperConfig`` for multi-device.
        tile_hw: Tile dimension for alignment check (default 32).
    """
    assert (assigner is None) != (assignment is None), "Provide exactly one of assigner or assignment"

    num_cores = core_grid.num_cores()
    if weight.ndim == 4:
        K, N = weight.shape[2], weight.shape[3]
    else:
        K, N = weight.shape[0], weight.shape[1]
    assert N % num_cores == 0, f"N ({N}) must be divisible by num_cores ({num_cores})"
    per_core_N = N // num_cores
    assert per_core_N % tile_hw == 0, (
        f"per_core_N ({per_core_N}) must be a multiple of tile_hw ({tile_hw}); "
        f"got N={N} on {num_cores} cores. Shrink the per-device core grid or "
        f"pick a mesh mapper whose per-device N stays tile-aligned."
    )
    assert K % tile_hw == 0, f"K ({K}) must be a multiple of tile_hw ({tile_hw})"

    shard_spec = ttnn.ShardSpec(core_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    return _route_l1_compressed_tensor(
        weight,
        memory_config=mem_config,
        # WIDTH_SHARDED on N: per_core_allocation is on for the assigner path
        # (mixed-precision needs per-core sizing) and off for the BSPM path
        # (legacy single-device behaviour).
        per_core_allocation=(assigner is not None),
        mesh_mapper_config=mesh_mapper_config if assigner is not None else None,
        assigner=assigner,
        assignment=assignment,
        device=device,
        tile_hw=tile_hw,
        cache_config=cache_config,
        sram_cache_target_name=sram_cache_target_name,
        sram_cache_source_keys=sram_cache_source_keys,
    )


def _build_l1_compressed_tensor_height_sharded(
    preprocessed_weight: torch.Tensor,
    core_range_set: ttnn.CoreRangeSet,
    shard_shape: tuple[int, int],
    *,
    assigner: CompressedTensorAssigner | None = None,
    assignment: np.ndarray | None = None,
    device=None,
    mesh_mapper_config=None,
    tile_hw: int = 32,
    cache_config: CacheConfig | None = None,
    sram_cache_target_name: str | None = None,
    sram_cache_source_keys: tuple[str, ...] | None = None,
) -> CompressedTensor:
    """Per-core L1 HEIGHT_SHARDED CompressedTensor for gate/up projections.

    Mirrors the shared expert's HEIGHT_SHARDED gate/up layout: each of the
    ``core_range_set.num_cores()`` cores holds a single ``shard_shape =
    (sh, sw)`` shard, so the per-device tensor shape is
    ``(num_cores * sh, sw)``.  That per-device shape is produced by
    :func:`preprocess_gate_up`, which applies the shared expert's
    ``reshuffle_block_to_height_sharded`` tile permutation and optionally
    TP-stacks across ``moe_tp`` slabs into the full multi-device tensor
    ``(mesh_rows * num_cores * sh, mesh_cols * sw)``.

    ``CompressedTensor.from_torch`` with ``mesh_mapper_config=[Shard(0),
    Shard(1)]`` splits that back to ``(num_cores * sh, sw)`` per device,
    where the HEIGHT_SHARDED memory config spreads the height across the
    ``num_cores`` cores one shard at a time (CRS iteration order).

    .. note::
       The ``core_range_set`` passed here **must** match the one used by
       ``preprocess_gate_up`` (i.e. ``GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
       .gate_core_range_set`` / ``.up_core_range_set``) so the block
       permutation inside ``reshuffle_block_to_height_sharded`` and the
       memory config's core ordering line up.

    Args:
        preprocessed_weight: Output of :func:`preprocess_gate_up` for one
            expert / one projection (``shared_gate_proj`` or
            ``shared_up_proj``).  Shape:

            * single-device (``mesh_mapper_config=None``):
              ``(num_cores * sh, sw)``.
            * multi-device with ``[Shard(0), Shard(1)]``:
              ``(mesh_rows * num_cores * sh, mesh_cols * sw)``.
        core_range_set: Shared-expert gate or up core range set (64 cores
            under the default ``k_par=n_par=8`` spec).
        shard_shape: ``(sh, sw)`` per-core shard shape (e.g. ``(896, 32)``).
        assigner: Mixed-precision assigner (mutually exclusive with
            *assignment*).
        assignment: Pre-computed BSPM tile assignment (mutually exclusive
            with *assigner*).
        device: Device / mesh device (``None`` for host-only).
        mesh_mapper_config: ``ttnn.MeshMapperConfig`` for multi-device
            (``PlacementShard(0), PlacementShard(1)`` to match
            :func:`preprocess_gate_up`'s TP8 stacking).
        tile_hw: Tile dimension (default 32).
    """
    assert (assigner is None) != (assignment is None), "Provide exactly one of assigner or assignment"

    sh, sw = shard_shape
    assert sh % tile_hw == 0, f"shard height ({sh}) must be a multiple of tile_hw ({tile_hw})"
    assert sw % tile_hw == 0, f"shard width ({sw}) must be a multiple of tile_hw ({tile_hw})"

    shard_spec = ttnn.ShardSpec(core_range_set, [sh, sw], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    return _route_l1_compressed_tensor(
        preprocessed_weight,
        memory_config=mem_config,
        per_core_allocation=(assigner is not None),
        mesh_mapper_config=mesh_mapper_config if assigner is not None else None,
        assigner=assigner,
        assignment=assignment,
        device=device,
        tile_hw=tile_hw,
        cache_config=cache_config,
        sram_cache_target_name=sram_cache_target_name,
        sram_cache_source_keys=sram_cache_source_keys,
    )


def _build_l1_compressed_tensor_width_sharded_shared_down(
    preprocessed_weight: torch.Tensor,
    core_range_set: ttnn.CoreRangeSet,
    shard_shape: tuple[int, int],
    *,
    assigner: CompressedTensorAssigner | None = None,
    assignment: np.ndarray | None = None,
    device=None,
    mesh_mapper_config=None,
    tile_hw: int = 32,
    cache_config: CacheConfig | None = None,
    sram_cache_target_name: str | None = None,
    sram_cache_source_keys: tuple[str, ...] | None = None,
) -> CompressedTensor:
    """Per-core L1 WIDTH_SHARDED CompressedTensor for the down projection.

    Mirrors the shared expert's down layout: each of the 112 matmul cores
    holds a single ``shard_shape = (K_per_dev, N_per_core) = (256, 64)``
    shard.  The per-device tensor shape is ``(K_per_dev, N_down =
    N_per_core * num_cores = 7168)``, produced by
    :func:`shared_down_torch_for_cache` which reshape-permute-reshapes the
    raw routed-expert ``(K=2048, N=7168)`` down weight into a
    mesh-friendly ``(mesh_rows * K_per_dev, mesh_cols * N_down) =
    (1024, 14336)`` stacked layout.  ``CompressedTensor.from_torch`` with
    ``mesh_mapper_config=[Shard(0), Shard(1)]`` splits it back to
    ``(K_per_dev, N_down)`` per device.

    Unlike :func:`_build_l1_compressed_tensor`, the shard shape is passed
    explicitly: the preprocessed tensor's 2D dims already encode the mesh
    split, so deriving ``per_core_N`` from ``N / num_cores`` (as that helper
    does) would compute the WRONG number after
    :func:`shared_down_torch_for_cache` coalesces the mesh_cols axis into
    the width dimension.

    Args:
        preprocessed_weight: Output of :func:`shared_down_torch_for_cache`
            for one expert's down weight.  Shape:

            * single-device (``moe_tp=1``): ``(K_per_dev, N_down) =
              (256, 7168)``.
            * multi-device TP8: ``(mesh_rows * K_per_dev, mesh_cols *
              N_down) = (1024, 14336)``.
        core_range_set: Shared expert's 112-core down grid
            (``DOWN_PROJ_SINGLE_DEVICE_SPEC.build_matmul_core_grid()``).
        shard_shape: ``(K_per_dev, N_per_core) = (256, 64)`` per-core
            shard.  Must be tile-aligned on both axes.
        assigner / assignment: Exactly one required (see
            :func:`_build_l1_compressed_tensor`).
        device: Device / mesh device.
        mesh_mapper_config: ``[Shard(0), Shard(1)]`` under TP>1 to match
            :func:`shared_down_torch_for_cache`'s stacked layout.
        tile_hw: Tile dimension (default 32).
    """
    assert (assigner is None) != (assignment is None), "Provide exactly one of assigner or assignment"

    sh, sw = shard_shape
    assert sh % tile_hw == 0, f"shard height ({sh}) must be a multiple of tile_hw ({tile_hw})"
    assert sw % tile_hw == 0, f"shard width ({sw}) must be a multiple of tile_hw ({tile_hw})"

    shard_spec = ttnn.ShardSpec(core_range_set, [sh, sw], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    return _route_l1_compressed_tensor(
        preprocessed_weight,
        memory_config=mem_config,
        per_core_allocation=(assigner is not None),
        mesh_mapper_config=mesh_mapper_config if assigner is not None else None,
        assigner=assigner,
        assignment=assignment,
        device=device,
        tile_hw=tile_hw,
        cache_config=cache_config,
        sram_cache_target_name=sram_cache_target_name,
        sram_cache_source_keys=sram_cache_source_keys,
    )


# ---------------------------------------------------------------------------
# Address-based trim helpers
# ---------------------------------------------------------------------------


def _compute_sram_trim_budget(
    device,
    persistent_tensors: list,
    sram_core_grids: SramExpertCoreGrids,
    worker_l1_size: int,
    combined_attn_sram_cap_bytes: int,
) -> tuple[int, dict[tuple[int, int], int], int]:
    """Compute ``(boundary_addr, initial_lowest_addr, l1_top_addr)`` for the
    address-based SRAM expert trim.

    Walks ``persistent_tensors`` (the persistent attention reservations)
    restricted to the SRAM binding cores (gate ∪ up ∪ down), queries the
    allocator for its worker-L1 base, and turns the combined attn+SRAM cap
    into a hard floor address:

        l1_top_addr    = worker_l1_unreserved_base + worker_l1_size
        boundary_addr  = l1_top_addr - combined_attn_sram_cap_bytes

    ``initial_lowest_addr`` is bootstrapped from real allocator addresses
    of the persistent attention L1 tensors -- callers must therefore
    ensure those tensors are on device *before* invoking this helper
    (cache-backed providers go through :func:`eager_upload_l1_lockstep`
    in ``prepare_moe_layer_weights``).  This guarantees the
    "attn-first, SRAM-below" allocation order the per-core invariant
    ``attn[c] + sram[c] <= cap`` relies on.

    Shared expert L1 weights are deliberately *not* in scope here: they
    are uploaded after SRAM and land below ``boundary_addr`` in the
    ``worker_l1_size - cap`` reserve.  Host-staged tensors passed in are
    silently ignored by the underlying address query, so leaving shared
    fields out of ``persistent_tensors`` is just a clarity-of-intent
    decision rather than a correctness one.

    The returned triple is the input contract of
    :func:`prepare_compressed_sram_slots`.
    """
    sram_core_set = (
        set(_core_list(sram_core_grids.gate))
        | set(_core_list(sram_core_grids.up))
        | set(_core_list(sram_core_grids.down))
    )
    initial_lowest_addr = per_core_l1_lowest_addr_on_cores(persistent_tensors, sram_core_set)
    worker_l1_unreserved_base = ttnn.get_allocator_base_address(device, ttnn.BufferType.L1)
    l1_top_addr = worker_l1_unreserved_base + worker_l1_size
    boundary_addr = l1_top_addr - combined_attn_sram_cap_bytes
    return boundary_addr, initial_lowest_addr, l1_top_addr


def _refresh_lowest_addr_from_alloc(
    curr_addr: dict[tuple[int, int], int],
    cts: tuple[tuple[CompressedTensor, ttnn.CoreRangeSet], ...],
) -> None:
    """Refine *curr_addr* (per-core lowest L1 address) from real allocator state.

    For each ``(CompressedTensor, grid)`` pair, walk the grid's cores and
    query the CT's per-core L1 base address; record it as the new lowest
    occupied address whenever it sits below the current entry.  Top-down
    L1 allocation means freshly allocated tensors land *below* anything
    previously allocated, so the new address is the natural new floor.

    Mutates *curr_addr* in place.
    """
    for ct, grid in cts:
        for c_obj in ttnn.corerange_to_cores(grid):
            c_xy = (c_obj.x, c_obj.y)
            new_addr = ct.get_data_l1_address_per_core(c_obj)
            prev = curr_addr.get(c_xy)
            if prev is None or new_addr < prev:
                curr_addr[c_xy] = new_addr


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _key(layer_idx: int, suffix: str) -> str:
    """Build the HF state-dict key for a per-layer tensor."""
    return f"model.layers.{layer_idx}.{suffix}"


def prepare_compressed_sram_slots(
    device,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    initial_expert_indices: list[int],
    core_grids: SramExpertCoreGrids,
    assigner: CompressedTensorAssigner | None = None,
    *,
    boundary_addr: int,
    initial_lowest_addr: dict[tuple[int, int], int],
    l1_top_addr: int,
    assignment_provider: Callable[[int, int], np.ndarray] | None = None,
    move_to_device: bool = True,
    tile_hw: int = 32,
    cache_config: CacheConfig | None = None,
) -> SramCompressedExpertSlots:
    """Allocate SRAM hot expert slots as per-core L1 CompressedTensors.

    Each expert's gate, up, and down projections are independently allocated in
    L1 as WIDTH_SHARDED ``CompressedTensor`` objects with ``per_core_allocation=True``.

    The returned :class:`SramCompressedExpertSlots` holds CTs that can be
    passed directly to ``ExpertKernel.op()`` as ``sram_cts``.

    Address-boundary budgeting (always on): ``initial_expert_indices`` is
    treated as a *ranked candidate list* and allocation stops at the first
    expert whose predicted per-core footprint would push any tracked core's
    lowest occupied L1 address *below* ``boundary_addr``.  Concretely, for
    each candidate the predicted delta ``Δ_c`` is compared against the
    current per-core lowest-allocated address ``A_c`` (bootstrapped from
    ``initial_lowest_addr``, defaulting to ``l1_top_addr`` for cores not
    yet touched); the candidate is accepted iff
    ``A_c - Δ_c >= boundary_addr`` for every core ``c`` in the expert's
    binding grid.  After each accepted expert lands on device, the address
    map ``A_c`` is refreshed from real allocator addresses so prediction
    drift is bounded to a single expert.  To allocate without budgeting,
    pass ``boundary_addr=worker_l1_unreserved_base`` (the allocator's own
    floor); this keeps the check trivially satisfied and is the idiomatic
    "no trim" call.

    Args:
        device: Device or mesh device.
        state_dict: HuggingFace state dict with expert weights.
        layer_idx: Decoder layer index.
        initial_expert_indices: Ranked candidate list of global expert
            indices (first candidate is preferred).
        core_grids: :class:`SramExpertCoreGrids` assigning a distinct
            (tile-aligned, ``N``-divisible) grid to each projection, matching
            the routed-expert pipeline's per-projection layout (e.g.
            64 / 64 / 112 cores on Blackhole).  For symmetric unit-test
            weights use :meth:`SramExpertCoreGrids.uniform`.
        assigner: ``CompressedTensorAssigner`` for on-the-fly mixed-precision
            encoding.  Mutually exclusive with ``assignment_provider``.
        boundary_addr: Lower-bound L1 byte address that allocations must
            stay above.  Typically ``l1_top_addr - cap_bytes`` for a tight
            cap, or ``worker_l1_unreserved_base`` for the effectively-no-trim
            case.
        initial_lowest_addr: Per-core lowest already-occupied L1 address
            (e.g. from :func:`per_core_l1_lowest_addr_on_cores` over the
            attention / shared / routed reservations).  Cores not present
            default to ``l1_top_addr``.  Pass ``{}`` when the allocator is
            fresh.
        l1_top_addr: Absolute top of the worker-L1 allocator region
            (``worker_l1_unreserved_base + worker_l1_size``).  Used to
            bootstrap untouched cores' "current allocation floor" before
            any expert lands.
        assignment_provider: Callable ``(expert_idx, proj_idx) -> np.ndarray``
            returning a pre-computed tile assignment (e.g. from a BSPM file).
            Mutually exclusive with ``assigner``.  Only used on the TP=1
            single-device path; the TP>1 height-sharded path always uses
            ``assigner``.
        move_to_device: Whether to place tensors on device.
        tile_hw: Tile dimension for cost prediction (default 32).
    """
    assert (assigner is None) != (assignment_provider is None), "Provide exactly one of assigner or assignment_provider"
    assert l1_top_addr > boundary_addr, f"l1_top_addr ({l1_top_addr}) must be > boundary_addr ({boundary_addr})"
    # Per-core L1 CompressedTensors cannot be host-staged; they must be
    # allocated directly on device.  Catch this early to surface a clear
    # error instead of a NoneType crash deep inside CompressedTensor.from_torch.
    assert move_to_device, (
        "prepare_compressed_sram_slots requires move_to_device=True; SRAM hot expert slots are "
        "per-core L1 tensors and have no host-stage path. The DRAM weight path's "
        "move_to_device=False does not apply here."
    )
    grids = core_grids
    requested_experts = list(initial_expert_indices)
    logger.info(
        "Preparing up to {} compressed SRAM slots for layer {} (experts: {})",
        len(requested_experts),
        layer_idx,
        requested_experts,
    )
    logger.info(
        "  SRAM core grids: gate={} cores, up={} cores, down={} cores",
        grids.gate.num_cores(),
        grids.up.num_cores(),
        grids.down.num_cores(),
    )
    # Lowest pre-existing per-core address tells us how much room is left
    # above boundary_addr at the start; min over cores is the tightest
    # initial headroom.
    initial_min_addr = min(initial_lowest_addr.values(), default=l1_top_addr)
    logger.info(
        "  SRAM trim boundary_addr={} (l1_top={}, headroom={} bytes), "
        "initial_min_lowest_addr={} (initial headroom above boundary={} bytes)",
        boundary_addr,
        l1_top_addr,
        l1_top_addr - boundary_addr,
        initial_min_addr,
        initial_min_addr - boundary_addr,
    )
    t0 = time.perf_counter()

    device_for_torch = device if move_to_device else None
    gate_cts: list[CompressedTensor] = []
    up_cts: list[CompressedTensor] = []
    down_cts: list[CompressedTensor] = []

    # Match the shared-expert TP8 convention (see `_shared_down_tensor_target`
    # and `preprocess_gate_up`): expert weights are 2D-sharded across the 4x2
    # mesh with dim-0 (K) split along mesh_rows and dim-1 (N) split along
    # mesh_cols.  This reduces per-core L1 footprint by ~2.4x vs full replicate,
    # letting ~2-3x more hot experts fit per core. The consuming kernel must
    # emit a cross-mesh_rows reduce at the down output (shared expert's
    # existing pipeline already does this -- see `MoeOp` wiring TODO).
    mesh_mapper_config = None
    mesh_rows, mesh_cols = 1, 1
    if device_for_torch is not None:
        num_mesh_devices = getattr(device_for_torch, "get_num_devices", lambda: 1)()
        if num_mesh_devices > 1:
            mesh_shape = device_for_torch.shape
            mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
            mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)])

    moe_tp = mesh_rows * mesh_cols
    # Under TP>1 we adopt the shared-expert gate/up layout: HEIGHT_SHARDED on
    # 64 cores with the `reshuffle_block_to_height_sharded` block permutation
    # + TP stacking via `preprocess_gate_up`.  The per-core shard is the
    # natively tile-aligned (sh, sw) = (896, 32) — unlike WIDTH_SHARDED on 64
    # cores which would need per_core_N=16 (not tile-aligned) for gate/up.
    # See sram_experts_plan.md (phase 2) for the full rationale.
    #
    # Single-device tests (e.g. `test_prepare_compressed_sram_slots` with
    # synthetic K=N=256 weights) don't match the shared expert's
    # (7168, 256)-per-device geometry, so we keep the old WIDTH_SHARDED path
    # there.  Real deployment is TP8 so this branch is exercised end-to-end.
    use_shared_expert_gate_up = moe_tp > 1
    gate_up_cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC

    def _shard_for_mesh(w: torch.Tensor) -> torch.Tensor:
        """Reshape ``(K, N)`` -> ``(mesh_rows, mesh_cols, K/rows, N/cols)`` for TP8."""
        if mesh_rows == 1 and mesh_cols == 1:
            return w
        K, N = w.shape
        assert K % mesh_rows == 0, f"K ({K}) must be divisible by mesh_rows ({mesh_rows})"
        assert N % mesh_cols == 0, f"N ({N}) must be divisible by mesh_cols ({mesh_cols})"
        return w.reshape(mesh_rows, K // mesh_rows, mesh_cols, N // mesh_cols).permute(0, 2, 1, 3).contiguous()

    # Address-boundary trim state: only tracks per-core lowest occupied L1
    # address (bootstrapped from real allocator state via the attention prep).
    # Cores not present default to ``l1_top_addr`` (assumed empty).  After
    # each accepted expert is allocated, ``curr_addr`` is refreshed from real
    # allocator addresses so prediction drift stays bounded to a single
    # expert.
    curr_addr: dict[tuple[int, int], int] = dict(initial_lowest_addr)
    mesh_shape_for_predict = (mesh_rows, mesh_cols)
    selected_experts: list[int] = []

    for slot_idx, expert_idx in enumerate(requested_experts):
        gate_key = _key(layer_idx, f"mlp.experts.{expert_idx}.gate_proj.weight")
        up_key = _key(layer_idx, f"mlp.experts.{expert_idx}.up_proj.weight")
        down_key = _key(layer_idx, f"mlp.experts.{expert_idx}.down_proj.weight")

        gate_full = state_dict[gate_key].T.contiguous()
        up_full = state_dict[up_key].T.contiguous()
        # Read down once and reuse for both predict and build.  The underlying
        # tensor is cached by ``LazyStateDict`` but the ``.T.contiguous()``
        # memcpy was previously paid twice (predict arg + build site).
        down_raw = state_dict[down_key].T.contiguous()

        # Predict the next expert's per-core L1 cost (host-side, one-step
        # lookahead) so we can stop *before* an over-budget allocation.
        predicted_delta = _predict_expert_per_core_bytes(
            expert_idx,
            gate_full,
            up_full,
            down_raw,
            grids,
            assigner=assigner,
            assignment_provider=assignment_provider,
            tile_hw=tile_hw,
            mesh_shape=mesh_shape_for_predict,
        )
        # Cores untouched so far are assumed empty: their next allocation
        # would land just below ``l1_top_addr`` (top-down growth).
        overflow_core = next(
            (c for c, add in predicted_delta.items() if curr_addr.get(c, l1_top_addr) - add < boundary_addr),
            None,
        )
        if overflow_core is not None:
            proj_addr = curr_addr.get(overflow_core, l1_top_addr) - predicted_delta[overflow_core]
            logger.info(
                "  Stopping at expert {} (slot {}): core {} projected addr {} "
                "would fall below boundary {} (delta={} bytes); selected {} experts so far",
                expert_idx,
                slot_idx,
                overflow_core,
                proj_addr,
                boundary_addr,
                predicted_delta[overflow_core],
                len(selected_experts),
            )
            break

        # Per-projection SRAM disk-cache identity: each (layer, expert,
        # projection) pair gets its own fingerprint, sourced from the
        # exact HF state-dict tensor name(s) it consumes.  The name
        # doubles as a human-friendly cache label.
        gate_cache_name = f"sram_layer{layer_idx}_expert{expert_idx}_gate_proj"
        up_cache_name = f"sram_layer{layer_idx}_expert{expert_idx}_up_proj"
        down_cache_name = f"sram_layer{layer_idx}_expert{expert_idx}_down_proj"
        if use_shared_expert_gate_up:
            # Shared-expert-style HEIGHT_SHARDED gate/up: reshuffle +
            # TP-stack once per expert, then build CTs for both projections
            # off the same preprocessed dict.
            preprocessed = preprocess_gate_up(gate_full, up_full, moe_tp, mesh_rows, mesh_cols)
            gate_cts.append(
                _build_l1_compressed_tensor_height_sharded(
                    preprocessed["shared_gate_proj"],
                    gate_up_cfg.gate_core_range_set,
                    gate_up_cfg.shard_shape,
                    assigner=assigner,
                    device=device_for_torch,
                    mesh_mapper_config=mesh_mapper_config,
                    cache_config=cache_config,
                    sram_cache_target_name=gate_cache_name,
                    sram_cache_source_keys=(gate_key,),
                )
            )
            up_cts.append(
                _build_l1_compressed_tensor_height_sharded(
                    preprocessed["shared_up_proj"],
                    gate_up_cfg.up_core_range_set,
                    gate_up_cfg.shard_shape,
                    assigner=assigner,
                    device=device_for_torch,
                    mesh_mapper_config=mesh_mapper_config,
                    cache_config=cache_config,
                    sram_cache_target_name=up_cache_name,
                    sram_cache_source_keys=(up_key,),
                )
            )
        else:
            gate_w = _shard_for_mesh(gate_full)
            up_w = _shard_for_mesh(up_full)
            gate_cts.append(
                _build_l1_compressed_tensor(
                    gate_w,
                    grids.gate,
                    assigner=assigner,
                    assignment=assignment_provider(expert_idx, 0) if assignment_provider is not None else None,
                    device=device_for_torch,
                    mesh_mapper_config=mesh_mapper_config,
                    cache_config=cache_config,
                    sram_cache_target_name=gate_cache_name,
                    sram_cache_source_keys=(gate_key,),
                )
            )
            up_cts.append(
                _build_l1_compressed_tensor(
                    up_w,
                    grids.up,
                    assigner=assigner,
                    assignment=assignment_provider(expert_idx, 1) if assignment_provider is not None else None,
                    device=device_for_torch,
                    mesh_mapper_config=mesh_mapper_config,
                    cache_config=cache_config,
                    sram_cache_target_name=up_cache_name,
                    sram_cache_source_keys=(up_key,),
                )
            )

        # Down projection (Phase 3): under TP>1 use the shared expert's
        # K-reshape trick — ``shared_down_torch_for_cache`` stacks the
        # routed-expert ``(K=2048, N=7168)`` down weight into a
        # ``(mesh_rows * K_per_dev, mesh_cols * N_down) = (1024, 14336)``
        # layout that, under ``mesh_mapper_config=[Shard(0), Shard(1)]``,
        # yields per-device ``(K_per_dev=256, N_down=7168)`` — giving 112
        # cores × ``(256, N_per_core=64)`` shards (16 tiles per core).  This
        # matches shared expert's exact down data flow so the kernel can
        # reuse shared expert's data-movement code verbatim.
        #
        # Caveat (activation K mismatch): routed expert activations arrive
        # at ``K_dev=512`` (from the TP8 gate/up outputs), but this down
        # layout expects ``K_dev=256`` inputs.  Bridging that mismatch
        # (extra K-slab reshuffle on activations, or per-core
        # ``sram_k_offsets`` indirection) is a Phase 4 / kernel-side
        # concern.  See sram_experts_plan.md "Risks/caveats" for details.
        #
        # Under TP=1 the synthetic test geometry (K=N=256) doesn't match
        # ``shared_down_torch_for_cache``'s hardcoded ``(2048, 7168)``
        # shape assertion, so we fall back to the legacy WIDTH_SHARDED
        # path for single-device tests.
        # ``down_raw`` was read above at I/O time and is reused here.
        if use_shared_expert_gate_up:
            down_preprocessed = shared_down_torch_for_cache(down_raw, moe_tp, (mesh_rows, mesh_cols))
            down_cts.append(
                _build_l1_compressed_tensor_width_sharded_shared_down(
                    down_preprocessed,
                    grids.down,
                    (_SHARED_EXPERT_DOWN_K_PER_DEV, _SHARED_EXPERT_DOWN_N_PER_CORE),
                    assigner=assigner,
                    device=device_for_torch,
                    mesh_mapper_config=mesh_mapper_config,
                    cache_config=cache_config,
                    sram_cache_target_name=down_cache_name,
                    sram_cache_source_keys=(down_key,),
                )
            )
        else:
            down_w = _shard_for_mesh(down_raw)
            down_cts.append(
                _build_l1_compressed_tensor(
                    down_w,
                    grids.down,
                    assigner=assigner,
                    assignment=assignment_provider(expert_idx, 2) if assignment_provider is not None else None,
                    device=device_for_torch,
                    mesh_mapper_config=mesh_mapper_config,
                    cache_config=cache_config,
                    sram_cache_target_name=down_cache_name,
                    sram_cache_source_keys=(down_key,),
                )
            )
        selected_experts.append(expert_idx)
        logger.debug("  SRAM slot {} ← expert {} prepared", slot_idx, expert_idx)

        # Refresh per-core lowest occupied L1 address from the real
        # allocator so the next iteration's fit check uses ground truth
        # (not prefix-summed host predictions).
        _refresh_lowest_addr_from_alloc(
            curr_addr,
            cts=(
                (gate_cts[-1], grids.gate),
                (up_cts[-1], grids.up),
                (down_cts[-1], grids.down),
            ),
        )

    num_slots = len(selected_experts)
    logger.info(
        "Compressed SRAM slots for layer {} done in {:.3f}s ({} of {} slots accepted)",
        layer_idx,
        time.perf_counter() - t0,
        num_slots,
        len(requested_experts),
    )
    return SramCompressedExpertSlots(
        num_slots=num_slots,
        slot_experts=selected_experts,
        gate_proj=gate_cts,
        up_proj=up_cts,
        down_proj=down_cts,
    )
