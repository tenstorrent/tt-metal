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
from models.demos.deepseek_v3_b1.weights.transforms.moe import preprocess_gate_up, shared_down_torch_for_cache
from models.demos.deepseek_v3_b1.weights.transforms.sram_experts import (
    SramExpertCoreGrids,
    _MemoryConfigSpec,
    _predict_expert_per_core_bytes,
    sram_projection_specs,
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
    spec: _MemoryConfigSpec,
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

    The ``(layout, core_range_set, shard_shape)`` decision lives on
    ``spec``: WIDTH_SHARDED with auto-derived ``(K, N // num_cores)`` for
    the TP=1 fallback (legacy gate/up/down), HEIGHT_SHARDED with explicit
    ``(sh, sw)`` for the shared-expert TP>1 gate/up after
    :func:`preprocess_gate_up`, and WIDTH_SHARDED with explicit
    ``(K_per_dev, N_per_core)`` for the shared-expert TP>1 down after
    :func:`shared_down_torch_for_cache`.  Exactly one of *assigner* or
    *assignment* must be provided.

    Args:
        weight: Float weight tensor.  Shape varies with ``spec``:
                * WIDTH_SHARDED auto-derive (``shard_shape=None``):
                  ``(K, N)`` (2D) or
                  ``(mesh_rows, mesh_cols, K_per_device, N_per_device)``
                  (4D, mesh-pre-sharded).
                * Explicit shard shape: 2D per-device tensor whose dims
                  match the upstream preprocess (e.g. ``(num_cores * sh,
                  sw)`` after :func:`preprocess_gate_up`, or
                  ``(K_per_dev, N_down)`` after
                  :func:`shared_down_torch_for_cache`).
        spec: ``_MemoryConfigSpec`` describing the on-device layout.
        assigner: ``CompressedTensorAssigner`` (mutually exclusive with *assignment*).
        assignment: Pre-computed BSPM tile-level assignment array (mutually
                    exclusive with *assigner*).
        device: Device / mesh to allocate on (``None`` keeps host-only).
        mesh_mapper_config: ``ttnn.MeshMapperConfig`` for multi-device.
                            Only honoured on the assigner path; the BSPM
                            path historically does not request multi-device
                            sharding (TP=1 only).
        tile_hw: Tile dimension for alignment check (default 32).
    """
    assert (assigner is None) != (assignment is None), "Provide exactly one of assigner or assignment"

    mem_config = spec.materialize(weight, tile_hw=tile_hw)

    return _route_l1_compressed_tensor(
        weight,
        memory_config=mem_config,
        # per_core_allocation is on for the assigner path (mixed-precision
        # needs per-core sizing) and off for the BSPM path (legacy
        # single-device behaviour).
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


def _shard_for_mesh(w: torch.Tensor, mesh_rows: int, mesh_cols: int) -> torch.Tensor:
    """Reshape ``(K, N)`` -> ``(mesh_rows, mesh_cols, K/rows, N/cols)`` for TP>1.

    Identity for TP=1; under multi-device meshes splits ``K`` along
    ``mesh_rows`` and ``N`` along ``mesh_cols`` so a subsequent
    ``mesh_mapper_config=[Shard(0), Shard(1)]`` walks the resulting 4D tensor
    into per-device ``(K/rows, N/cols)`` slabs.  Used by the TP=1 fallback
    builders (``_build_l1_compressed_tensor``); the TP>1 shared-expert paths
    do their own preprocessing via ``preprocess_gate_up`` /
    ``shared_down_torch_for_cache``.
    """
    if mesh_rows == 1 and mesh_cols == 1:
        return w
    K, N = w.shape
    assert K % mesh_rows == 0, f"K ({K}) must be divisible by mesh_rows ({mesh_rows})"
    assert N % mesh_cols == 0, f"N ({N}) must be divisible by mesh_cols ({mesh_cols})"
    return w.reshape(mesh_rows, K // mesh_rows, mesh_cols, N // mesh_cols).permute(0, 2, 1, 3).contiguous()


@dataclass(frozen=True)
class _ProjBuildPlan:
    """Per-projection build inputs for one expert in ``prepare_compressed_sram_slots``.

    Bundles the ``(weight, spec, cache identity)`` tuple driving one
    :func:`_build_l1_compressed_tensor` call.  :func:`_expert_build_plans`
    returns three plans per expert (gate, up, down), hiding the TP>1 vs TP=1
    layout split behind a single uniform call shape — the loop body in
    :func:`prepare_compressed_sram_slots` no longer has to fork on layout.

    Fields:
      proj_idx: 0=gate, 1=up, 2=down — also the index ``assignment_provider``
        is keyed on under TP=1.
      weight: Already preprocessed for the chosen layout (``preprocess_gate_up``
        output under TP>1 gate/up; ``shared_down_torch_for_cache`` output under
        TP>1 down; ``_shard_for_mesh`` output under TP=1).
      spec: :class:`_MemoryConfigSpec` describing the on-device layout
        (``(layout, core_range_set, shard_shape)``).
      cache_name / source_key: Per-projection cache identity (target name
        plus the single HF state-dict key it consumes).
      use_assignment_provider: When ``True`` and an ``assignment_provider`` is
        wired into :func:`prepare_compressed_sram_slots`, that provider
        supplies the assignment for this plan; otherwise the call falls back
        to the shared ``assigner``.  Currently ``True`` only on the TP=1
        fallback.
    """

    proj_idx: int
    weight: torch.Tensor
    spec: _MemoryConfigSpec
    cache_name: str
    source_key: str
    use_assignment_provider: bool = False


def _expert_build_plans(
    *,
    layer_idx: int,
    expert_idx: int,
    gate_full: torch.Tensor,
    up_full: torch.Tensor,
    down_raw: torch.Tensor,
    gate_key: str,
    up_key: str,
    down_key: str,
    grids: SramExpertCoreGrids,
    use_shared_expert_layout: bool,
    moe_tp: int,
    mesh_rows: int,
    mesh_cols: int,
) -> tuple[_ProjBuildPlan, _ProjBuildPlan, _ProjBuildPlan]:
    """Build the (gate, up, down) :class:`_ProjBuildPlan` triple for one expert.

    The per-projection layout decision (``_MemoryConfigSpec`` per name) is
    delegated to :func:`sram_projection_specs` -- single source of truth
    shared with the cost predictor in
    :mod:`models.demos.deepseek_v3_b1.weights.transforms.sram_experts`.
    Weight preprocessing branches on TP layout: under TP>1 gate/up ride
    :func:`preprocess_gate_up`'s reshuffle output and down rides
    :func:`shared_down_torch_for_cache`; under TP=1 all three use the
    :func:`_shard_for_mesh` mesh-presharding pre-pass.
    """

    def name(proj: str) -> str:
        return f"sram_layer{layer_idx}_expert{expert_idx}_{proj}_proj"

    specs = sram_projection_specs(grids, (mesh_rows, mesh_cols))
    keys = {"gate": gate_key, "up": up_key, "down": down_key}

    if use_shared_expert_layout:
        pp = preprocess_gate_up(gate_full, up_full, moe_tp, mesh_rows, mesh_cols)
        down_pp = shared_down_torch_for_cache(down_raw, moe_tp, (mesh_rows, mesh_cols))
        weights = {"gate": pp["shared_gate_proj"], "up": pp["shared_up_proj"], "down": down_pp}
        use_provider = False
    else:
        weights = {
            "gate": _shard_for_mesh(gate_full, mesh_rows, mesh_cols),
            "up": _shard_for_mesh(up_full, mesh_rows, mesh_cols),
            "down": _shard_for_mesh(down_raw, mesh_rows, mesh_cols),
        }
        use_provider = True

    return tuple(
        _ProjBuildPlan(
            proj_idx=i,
            weight=weights[proj],
            spec=specs[proj],
            cache_name=name(proj),
            source_key=keys[proj],
            use_assignment_provider=use_provider,
        )
        for i, proj in enumerate(("gate", "up", "down"))
    )


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
    # existing pipeline already does this).
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

        # Per-projection SRAM disk-cache identity, layout decision, and
        # weight preprocessing all live in ``_expert_build_plans``: under
        # TP>1 each projection rides the shared-expert pipeline
        # (HEIGHT_SHARDED gate/up via ``preprocess_gate_up``,
        # WIDTH_SHARDED shared-down via ``shared_down_torch_for_cache``);
        # under TP=1 all three use the WIDTH_SHARDED legacy fallback.
        #
        # Caveat (activation K mismatch under TP>1): routed expert
        # activations arrive at ``K_dev=512`` (from the TP8 gate/up
        # outputs), but the down layout expects ``K_dev=256`` inputs.
        # Bridging that mismatch (extra K-slab reshuffle on activations,
        # or per-core ``sram_k_offsets`` indirection) is a Phase 4 /
        # kernel-side concern.  See sram_experts_plan.md "Risks/caveats"
        # for details.
        plans = _expert_build_plans(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            gate_full=gate_full,
            up_full=up_full,
            down_raw=down_raw,
            gate_key=gate_key,
            up_key=up_key,
            down_key=down_key,
            grids=grids,
            use_shared_expert_layout=use_shared_expert_gate_up,
            moe_tp=moe_tp,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
        )
        out_lists = (gate_cts, up_cts, down_cts)
        for plan, out in zip(plans, out_lists):
            if plan.use_assignment_provider and assignment_provider is not None:
                # TP=1 + BSPM provider: precomputed tile assignment per (expert, projection).
                # The builder's own asserter still enforces "exactly one of assigner/assignment".
                assigner_arg = None
                assignment_arg = assignment_provider(expert_idx, plan.proj_idx)
            else:
                assigner_arg = assigner
                assignment_arg = None
            out.append(
                _build_l1_compressed_tensor(
                    plan.weight,
                    plan.spec,
                    assigner=assigner_arg,
                    assignment=assignment_arg,
                    device=device_for_torch,
                    mesh_mapper_config=mesh_mapper_config,
                    cache_config=cache_config,
                    sram_cache_target_name=plan.cache_name,
                    sram_cache_source_keys=(plan.source_key,),
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
