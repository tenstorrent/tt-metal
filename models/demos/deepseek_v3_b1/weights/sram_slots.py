# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side orchestration for SRAM hot expert weight slots.

This module owns everything that turns a ranked candidate list of routed
experts into a populated :class:`SramCompressedExpertSlots` on device:

  * the per-projection per-core L1 ``CompressedTensor`` builder
    :func:`build_sram_routed_proj_ct` (with batch wrapper
    :func:`build_sram_routed_proj_cts`) — includes the optional disk-cache
    dispatch via :func:`get_or_create_sram_compressed_expert`,
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
from models.demos.deepseek_v3_b1.compressed_tensor import BFP_MANT_BITS, CompressedTensor
from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import bfp_tile_packed_size
from models.demos.deepseek_v3_b1.weights.cache import CacheConfig, SourceTensorSelection, SramCompressedTensorTarget
from models.demos.deepseek_v3_b1.weights.cache.sram_compressed_cache import (
    get_or_create_sram_compressed_expert,
    to_canonical_mesh_mapper,
)
from models.demos.deepseek_v3_b1.weights.overlap.spec import _core_list
from models.demos.deepseek_v3_b1.weights.specs.overlap_configs import GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
from models.demos.deepseek_v3_b1.weights.transforms.sram_experts import (
    SramExpertCoreGrids,
    _predict_expert_per_core_bytes,
    reduce_per_device_max,
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


def build_sram_routed_proj_cts(
    mesh_device,
    sram_expert_ids: list,
    full_torch_weights_per_device: dict,
    core_grid: ttnn.CoreRangeSet,
    num_tiles_k: int,
    n_parallel: int,
    per_core_N: int,
    *,
    tile_w: int = 32,
    assignment_per_expert: dict | None = None,
    cache_config: CacheConfig | None = None,
    cache_target_name_fn: "Callable[[int], str] | None" = None,
    cache_source_keys_fn: "Callable[[int], tuple[str, ...]] | None" = None,
) -> list[CompressedTensor]:
    """Batch wrapper around :func:`build_sram_routed_proj_ct` for a list of experts.

    Each slot's CT is built via :func:`build_sram_routed_proj_ct` with
    ``is_cb_backing_expert=(slot_idx == 0)``.  When ``assignment_per_expert`` is
    ``None``, every (expert, device) gets a uniform-BFP4 constant assignment
    (BFP4 = index 1 in COMPRESSED_FORMATS); otherwise the per-(expert, device)
    arrays come from the caller (e.g. BSPM file).

    Optional per-slot disk cache identity: provide ``cache_config`` plus
    ``cache_target_name_fn(eid) -> str`` and ``cache_source_keys_fn(eid) -> tuple``
    to persist the post-pack byte buffers under a content-addressed key.
    """
    moe_tp = mesh_device.shape[0] * mesh_device.shape[1]
    cts: list[CompressedTensor] = []
    for slot_idx, eid in enumerate(sram_expert_ids):
        weights = full_torch_weights_per_device[eid]
        if assignment_per_expert is None:
            per_dev_asgn = [
                np.full((weights[d].shape[0] // tile_w, weights[d].shape[1] // tile_w), 1, dtype=np.int8)
                for d in range(moe_tp)
            ]
        else:
            per_dev_asgn = assignment_per_expert[eid]
        cts.append(
            build_sram_routed_proj_ct(
                mesh_device=mesh_device,
                full_torch_weight_per_device=weights,
                core_grid=core_grid,
                num_tiles_k=num_tiles_k,
                n_parallel=n_parallel,
                per_core_N=per_core_N,
                is_cb_backing_expert=(slot_idx == 0),
                bspm_assignment_per_device=per_dev_asgn,
                tile_w=tile_w,
                cache_config=cache_config,
                cache_target_name=cache_target_name_fn(eid) if cache_target_name_fn else None,
                cache_source_keys=cache_source_keys_fn(eid) if cache_source_keys_fn else None,
            )
        )
    return cts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_sram_routed_proj_ct(
    *,
    mesh_device,
    full_torch_weight_per_device: list[torch.Tensor],
    core_grid: ttnn.CoreRangeSet,
    num_tiles_k: int,
    n_parallel: int,
    per_core_N: int,
    is_cb_backing_expert: bool,
    bspm_assignment_per_device: list[np.ndarray],
    tile_w: int = 32,
    cache_config: CacheConfig | None = None,
    cache_target_name: str | None = None,
    cache_source_keys: tuple[str, ...] | None = None,
) -> CompressedTensor:
    """Build one per-expert, per-projection SRAM L1 CompressedTensor.

    Assignment-based: caller provides per-(device) tile assignment.  For
    uniform-BFP4 the assignment is a constant array filled with the BFP4 format
    index (= 1 in COMPRESSED_FORMATS); for BSPM the assignment comes from a
    .bspm file.

    Layout decision:
      * ``k_parallel_factor = num_cores // n_parallel == 1`` → WIDTH_SHARDED
        (down projection at TP8).
      * else → HEIGHT_SHARDED with per-core shards stacked along H in row-major
        ``(k_idx, n_idx)`` order (gate/up at TP8).

    Disk cache: when ``cache_config`` is disk-backed AND ``cache_target_name`` +
    ``cache_source_keys`` are provided, the post-pack byte buffers are persisted
    so warm runs hydrate without re-packing.  Ephemeral cache configs and
    missing cache identity → cold pack (no persistence).
    """
    assert bspm_assignment_per_device is not None, "assignment required"
    num_sram_cores = len(ttnn.corerange_to_cores(core_grid))
    assert num_sram_cores % n_parallel == 0
    mesh_shape = mesh_device.shape
    mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
    num_devices = mesh_rows * mesh_cols
    assert len(full_torch_weight_per_device) == num_devices

    k_parallel_factor = num_sram_cores // n_parallel
    is_width_sharded = k_parallel_factor == 1

    if is_width_sharded:
        sram_b_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_grid,
                [num_tiles_k * tile_w, per_core_N * tile_w],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    else:
        sram_b_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_grid,
                [num_tiles_k * tile_w, per_core_N * tile_w],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    per_dev_shards = []
    for dev_idx in range(num_devices):
        b_full = full_torch_weight_per_device[dev_idx]
        if is_width_sharded:
            per_dev_shards.append(b_full)
        else:
            shards = []
            for i in range(num_sram_cores):
                k_idx = i // n_parallel
                n_idx = i % n_parallel
                k_start = k_idx * num_tiles_k * tile_w
                k_end = k_start + num_tiles_k * tile_w
                n_start = n_idx * per_core_N * tile_w
                n_end = n_start + per_core_N * tile_w
                shards.append(b_full[k_start:k_end, n_start:n_end])
            per_dev_shards.append(torch.cat(shards, dim=0))

    if is_width_sharded:
        b_4d = torch.stack(per_dev_shards).reshape(
            mesh_rows, mesh_cols, num_tiles_k * tile_w, num_sram_cores * per_core_N * tile_w
        )
    else:
        b_4d = torch.stack(per_dev_shards).reshape(
            mesh_rows, mesh_cols, num_sram_cores * num_tiles_k * tile_w, per_core_N * tile_w
        )

    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)])
    min_shard_bytes = bfp_tile_packed_size(BFP_MANT_BITS["bfp4"], tile_w) if is_cb_backing_expert else 1

    # Shuffle per-device assignment to match the per-core layout above.
    shuffled = []
    for asg_per_dev in bspm_assignment_per_device:
        if is_width_sharded:
            shuffled.append(asg_per_dev)
        else:
            parts = []
            for i in range(num_sram_cores):
                k_idx = i // n_parallel
                n_idx = i % n_parallel
                k_start = k_idx * num_tiles_k
                k_end = k_start + num_tiles_k
                n_start = n_idx * per_core_N
                n_end = n_start + per_core_N
                parts.append(asg_per_dev[k_start:k_end, n_start:n_end])
            shuffled.append(np.concatenate(parts, axis=0))
    full_assignment = np.concatenate(shuffled, axis=0)

    use_cache = cache_config is not None and cache_target_name is not None and cache_source_keys is not None
    if use_cache:
        target = SramCompressedTensorTarget(
            name=cache_target_name,
            tensor_shape=tuple(int(d) for d in b_4d.shape),
            tile_hw=tile_w,
            memory_config=sram_b_mem,
            per_core_allocation=True,
            mesh_mapper_config=to_canonical_mesh_mapper(mesh_mapper_config),
            assigner_fingerprint="",
            assignment_hash=hashlib.sha256(np.ascontiguousarray(full_assignment).tobytes()).hexdigest()[:16],
        )
        fp = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=tuple(cache_source_keys)),
            target=target,
        )
        return get_or_create_sram_compressed_expert(
            cache_config.cache,
            fp,
            mesh_device,
            weight_provider=lambda: b_4d,
            assignment=full_assignment,
            memory_config=sram_b_mem,
            per_core_allocation=True,
            mesh_mapper_config=mesh_mapper_config,
            tile_hw=tile_w,
            min_shard_bytes=min_shard_bytes,
        )

    return CompressedTensor(
        b_4d,
        full_assignment,
        device=mesh_device,
        memory_config=sram_b_mem,
        per_core_allocation=True,
        mesh_mapper_config=mesh_mapper_config,
        min_shard_bytes=min_shard_bytes,
    )


def _key(layer_idx: int, suffix: str) -> str:
    """Build the HF state-dict key for a per-layer tensor."""
    return f"model.layers.{layer_idx}.{suffix}"


def prepare_compressed_sram_slots(
    device,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    initial_expert_indices: list[int],
    core_grids: SramExpertCoreGrids,
    *,
    assignment_provider: Callable[[int, int], np.ndarray],
    boundary_addr: int,
    initial_lowest_addr: dict[tuple[int, int], int],
    l1_top_addr: int,
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
    assert assignment_provider is not None, "assignment_provider required"
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

    # Mesh layout: expert weights are 2D-sharded across the (mesh_rows,
    # mesh_cols) mesh with dim-0 (K) split along mesh_rows and dim-1 (N)
    # split along mesh_cols.  Reduces per-core L1 footprint by ~2.4x vs
    # full replicate.  The consuming kernel must emit a cross-mesh_rows
    # reduce at the down output.
    mesh_mapper_config = None
    mesh_rows, mesh_cols = 1, 1
    if device_for_torch is not None:
        num_mesh_devices = getattr(device_for_torch, "get_num_devices", lambda: 1)()
        if num_mesh_devices > 1:
            mesh_shape = device_for_torch.shape
            mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
            mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)])

    moe_tp = mesh_rows * mesh_cols
    # Layout: TP>1 gate/up at 64 cores → HEIGHT_SHARDED (k_par=8, n_par=8)
    # with per-core tile-aligned (sh, sw) = (896, 32); TP>1 down at 112
    # cores → WIDTH_SHARDED (k_par=1, n_par=112).  TP=1 degenerates with
    # mesh_rows=mesh_cols=1 through the same build_sram_routed_proj_ct
    # path (only used by host-only unit tests).

    # Per-projection n_parallel decision — loop-invariant; matches the
    # triple passed to :func:`build_sram_routed_proj_ct` below.  Hoisted out
    # of the loop so the predictor and allocator share the exact same value.
    num_gate_cores = len(ttnn.corerange_to_cores(grids.gate))
    num_up_cores = len(ttnn.corerange_to_cores(grids.up))
    num_down_cores = len(ttnn.corerange_to_cores(grids.down))
    # Gate/up follow the shared-expert HEIGHT_SHARDED overlap layout when
    # the grid matches that spec's core count (k_par, n_par from the spec
    # itself — no magic numbers).  Otherwise fall back to WIDTH_SHARDED
    # (k_par=1, n_par=num_cores).  Down is always WIDTH_SHARDED.
    _OVERLAP_GATE_CORES = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC.gate_core_range_set.num_cores()
    _OVERLAP_UP_CORES = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC.up_core_range_set.num_cores()
    _OVERLAP_N_PARALLEL = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC.n_parallel
    gate_n_parallel = _OVERLAP_N_PARALLEL if num_gate_cores == _OVERLAP_GATE_CORES else num_gate_cores
    up_n_parallel = _OVERLAP_N_PARALLEL if num_up_cores == _OVERLAP_UP_CORES else num_up_cores
    down_n_parallel = num_down_cores
    n_parallel_per_proj = (gate_n_parallel, up_n_parallel, down_n_parallel)

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

        # Predict the next expert's per-device per-core L1 cost (host-side,
        # one-step lookahead) so we can stop *before* an over-budget
        # allocation.  Reduce via max-over-devices so the chosen expert
        # count fits the worst-case device under BSPM (where different
        # devices receive different tile-format slabs).  Under uniform BFP4
        # the reduction is a no-op (every device has identical per-core
        # bytes).
        predicted_delta_per_device = _predict_expert_per_core_bytes(
            expert_idx,
            gate_full,
            up_full,
            down_raw,
            grids,
            assigner=None,
            assignment_provider=assignment_provider,
            tile_hw=tile_hw,
            mesh_shape=mesh_shape_for_predict,
            n_parallel_per_proj=n_parallel_per_proj,
        )
        predicted_delta = reduce_per_device_max(predicted_delta_per_device)
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

        # Per-projection layout & per-slot CT construction via
        # :func:`build_sram_routed_proj_ct` (uniform for TP=1 and TP>1).
        _is_cb_backing = len(selected_experts) == 0
        # Per-expert per-device weight slabs.  Hoisted n_parallel_per_proj
        # above drives k_par; per_core_N / num_tiles_k fall out of the
        # (K, N_per_dev) split.
        gate_full_kn = gate_full.reshape(gate_full.shape[0], gate_full.shape[1])
        up_full_kn = up_full.reshape(up_full.shape[0], up_full.shape[1])
        down_full_kn = down_raw.reshape(down_raw.shape[0], down_raw.shape[1])
        K_gate, N_gate_full = gate_full_kn.shape
        K_up, N_up_full = up_full_kn.shape
        K_down_full, N_down = down_full_kn.shape
        N_gate_per_dev = N_gate_full // moe_tp
        N_up_per_dev = N_up_full // moe_tp
        K_down_per_dev = K_down_full // moe_tp
        gate_per_dev = [
            gate_full_kn[:, d * N_gate_per_dev : (d + 1) * N_gate_per_dev].contiguous() for d in range(moe_tp)
        ]
        up_per_dev = [up_full_kn[:, d * N_up_per_dev : (d + 1) * N_up_per_dev].contiguous() for d in range(moe_tp)]
        down_per_dev = [
            down_full_kn[d * K_down_per_dev : (d + 1) * K_down_per_dev, :].contiguous() for d in range(moe_tp)
        ]
        gate_k_parallel = num_gate_cores // gate_n_parallel
        up_k_parallel = num_up_cores // up_n_parallel
        gate_num_tiles_k = K_gate // tile_hw // gate_k_parallel
        up_num_tiles_k = K_up // tile_hw // up_k_parallel
        gate_per_core_N = N_gate_per_dev // tile_hw // gate_n_parallel
        up_per_core_N = N_up_per_dev // tile_hw // up_n_parallel
        down_num_tiles_k = K_down_per_dev // tile_hw
        down_per_core_N = N_down // tile_hw // num_down_cores

        def _bspm_per_dev(proj_idx: int, full_assignment) -> list[np.ndarray]:
            """Slice the full BSPM assignment along the per-TP-device shard axis."""
            if full_assignment is None:
                return None
            if proj_idx in (0, 1):
                # gate/up: column-shard N across devices.
                n_tiles_per_dev = full_assignment.shape[1] // moe_tp
                return [
                    np.ascontiguousarray(full_assignment[:, d * n_tiles_per_dev : (d + 1) * n_tiles_per_dev])
                    for d in range(moe_tp)
                ]
            # down: row-shard K across devices.
            k_tiles_per_dev = full_assignment.shape[0] // moe_tp
            return [
                np.ascontiguousarray(full_assignment[d * k_tiles_per_dev : (d + 1) * k_tiles_per_dev, :])
                for d in range(moe_tp)
            ]

        gate_assignment_full = assignment_provider(expert_idx, 0)
        up_assignment_full = assignment_provider(expert_idx, 1)
        down_assignment_full = assignment_provider(expert_idx, 2)

        def _cache_target_name(_proj: str, _eid: int = expert_idx) -> str:
            return f"sram_layer{layer_idx}_expert{_eid}_{_proj}_proj"

        def _cache_source_keys(_proj: str, _eid: int = expert_idx) -> tuple[str, ...]:
            return (_key(layer_idx, f"mlp.experts.{_eid}.{_proj}_proj.weight"),)

        gate_ct = build_sram_routed_proj_ct(
            mesh_device=device_for_torch,
            full_torch_weight_per_device=gate_per_dev,
            core_grid=grids.gate,
            num_tiles_k=gate_num_tiles_k,
            n_parallel=gate_n_parallel,
            per_core_N=gate_per_core_N,
            is_cb_backing_expert=_is_cb_backing,
            bspm_assignment_per_device=_bspm_per_dev(0, gate_assignment_full),
            cache_config=cache_config,
            cache_target_name=_cache_target_name("gate"),
            cache_source_keys=_cache_source_keys("gate"),
        )
        up_ct = build_sram_routed_proj_ct(
            mesh_device=device_for_torch,
            full_torch_weight_per_device=up_per_dev,
            core_grid=grids.up,
            num_tiles_k=up_num_tiles_k,
            n_parallel=up_n_parallel,
            per_core_N=up_per_core_N,
            is_cb_backing_expert=_is_cb_backing,
            bspm_assignment_per_device=_bspm_per_dev(1, up_assignment_full),
            cache_config=cache_config,
            cache_target_name=_cache_target_name("up"),
            cache_source_keys=_cache_source_keys("up"),
        )
        down_ct = build_sram_routed_proj_ct(
            mesh_device=device_for_torch,
            full_torch_weight_per_device=down_per_dev,
            core_grid=grids.down,
            num_tiles_k=down_num_tiles_k,
            n_parallel=down_n_parallel,
            per_core_N=down_per_core_N,
            is_cb_backing_expert=_is_cb_backing,
            bspm_assignment_per_device=_bspm_per_dev(2, down_assignment_full),
            cache_config=cache_config,
            cache_target_name=_cache_target_name("down"),
            cache_source_keys=_cache_source_keys("down"),
        )
        gate_cts.append(gate_ct)
        up_cts.append(up_ct)
        down_cts.append(down_ct)
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
