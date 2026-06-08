# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side preprocessing for SRAM hot expert weights.

Pure preprocessing layer that turns one expert's float weight tensors into
the host-side artifacts the SRAM hot expert orchestrator
(:mod:`models.demos.deepseek_v3_b1.weights.sram_slots`) needs to do its
device-side work:

  * mixed-precision tile assignments (via :class:`CompressedTensorAssigner`)
  * per-core L1 byte costs (for trim budgeting and one-step lookahead)
  * per-layer ranking by routing frequency

Nothing in this module touches a device or a cache; it's all in-memory math.
That keeps the orchestrator (which *does* touch device/cache) free to focus
on allocation order, cache hit/miss handling, and address-based budgeting
without dragging the assigner machinery along.

Companion module to :mod:`weights.cache.sram_compressed_cache` (which
handles the disk-cache side of SRAM hot experts).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensorAssigner
from models.demos.deepseek_v3_b1.weights.specs.overlap_configs import (
    DOWN_PROJ_SINGLE_DEVICE_SPEC,
    GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
)

#: Per-layer SRAM hot expert configuration.
#:
#: Maps ``layer_idx → list[expert_idx]`` specifying which routed experts to
#: pre-load into L1 SRAM slots for each MoE layer.  Layers absent from the
#: dict get no SRAM slots.
#:
#: Example::
#:
#:     config: SramHotExpertConfig = {
#:         3: [10, 42, 100],   # layer 3: 3 hot experts
#:         7: [5, 20],         # layer 7: 2 hot experts
#:     }
SramHotExpertConfig = dict[int, list[int]]


@dataclass(frozen=True)
class SramExpertCoreGrids:
    """Per-projection core grids for SRAM hot experts.

    The routed-expert pipeline uses three distinct, tile-aligned core grids:
      * ``gate`` — "A" compute grid (e.g. 64 cores, N=2048)
      * ``up``   — "B" compute grid, disjoint from A (e.g. 64 cores, N=2048)
      * ``down`` — "down" grid covering the full matmul core set
                  (e.g. 112 cores, N=7168)

    Matching these grids is required so each projection's ``N`` is divisible
    by the per-projection core count, and so the ExpertKernel can locate each
    shard on the cores it expects.  For symmetric unit-test weights (single
    ``(K, N)`` shared by all three projections) pass the same ``CoreRangeSet``
    for ``gate``/``up``/``down``.
    """

    gate: ttnn.CoreRangeSet
    up: ttnn.CoreRangeSet
    down: ttnn.CoreRangeSet

    @classmethod
    def uniform(cls, core_grid: ttnn.CoreRangeSet) -> "SramExpertCoreGrids":
        """Broadcast a single ``CoreRangeSet`` to all three projections."""
        return cls(gate=core_grid, up=core_grid, down=core_grid)

    @classmethod
    def shared_expert_mirror(cls) -> "SramExpertCoreGrids":
        """Gate/up/down CRS matching the shared-expert overlap specs (single source).

        Uses :data:`GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC` for gate/up and
        :meth:`DOWN_PROJ_SINGLE_DEVICE_SPEC.build_matmul_core_grid` for down —
        the same canonical ``CoreRangeSet`` objects ``prepare_compressed_sram_slots``
        assumes under TP>1 (HEIGHT_SHARDED gate/up after ``preprocess_gate_up``,
        WIDTH_SHARDED down after ``shared_down_torch_for_cache``).  Call sites
        (tests, demos) should prefer this over re-stitching CRS from overlap_configs.
        """
        return cls(
            gate=GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC.gate_core_range_set,
            up=GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC.up_core_range_set,
            down=DOWN_PROJ_SINGLE_DEVICE_SPEC.build_matmul_core_grid(),
        )


# Shared-expert down projection per-device shard dimensions, mirrored from
# :func:`shared_down_torch_for_cache` in ``weights/transforms/moe.py``.  The
# function hardcodes ``K_down_per_device=256`` and ``N_per_core=64`` for the
# shared expert's ``DOWN_PROJ_SINGLE_DEVICE_SPEC`` (112 matmul cores → total
# per-device N = 7168).  SRAM hot experts reuse this layout verbatim under
# TP>1 so the kernel can share shared expert's data-movement code.  Update
# these constants only if the shared-expert down spec changes.
_SHARED_EXPERT_DOWN_K_PER_DEV = 256
_SHARED_EXPERT_DOWN_N_PER_CORE = 64


@dataclass(frozen=True)
class _MemoryConfigSpec:
    """Per-projection L1 memory layout spec for SRAM hot experts.

    Captures the ``(layout, core_range_set, shard_shape)`` triple needed to
    construct a ``ttnn.MemoryConfig``.  Single source of truth shared between
    the cost predictor (:func:`compute_expert_l1_bytes_per_core`) and the
    on-device allocator (``_expert_build_plans`` /
    :func:`_build_l1_compressed_tensor` in
    :mod:`models.demos.deepseek_v3_b1.weights.sram_slots`); per-projection
    instances are produced by :func:`sram_projection_specs`.

    Fields:
      layout: ``WIDTH_SHARDED`` (gate/up/down under TP=1; shared-down under
        TP>1) or ``HEIGHT_SHARDED`` (shared-gate/up under TP>1).
      core_range_set: ``CoreRangeSet`` to bind the shard to.
      shard_shape: Per-core ``(sh, sw)`` shard shape, or ``None`` for the
        WIDTH_SHARDED auto-derive path used by the TP=1 fallback (computes
        ``(K, N // num_cores)`` from the weight's shape).  Auto-derive is
        only valid for ``WIDTH_SHARDED``; explicit shapes are required
        everywhere else (HEIGHT_SHARDED, or the shared-down WIDTH_SHARDED
        whose 2D weight dims already encode the mesh split, see
        :func:`shared_down_torch_for_cache`).
    """

    layout: ttnn.TensorMemoryLayout
    core_range_set: ttnn.CoreRangeSet
    shard_shape: tuple[int, int] | None = None

    def materialize(self, weight: torch.Tensor, *, tile_hw: int) -> ttnn.MemoryConfig:
        """Resolve the spec against *weight* into a concrete ``ttnn.MemoryConfig``.

        For ``shard_shape=None`` (TP=1 WIDTH_SHARDED auto-derive only) the
        per-core shape is ``(K, N // num_cores)`` taken from ``weight.shape``;
        ``weight.ndim==4`` is supported for the mesh-pre-sharded
        ``(mesh_rows, mesh_cols, K_per_device, N_per_device)`` carrier.
        Otherwise the explicit ``shard_shape`` is used verbatim.
        """
        if self.shard_shape is None:
            assert (
                self.layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            ), f"auto-derived shard shape only supported for WIDTH_SHARDED; got {self.layout}"
            num_cores = self.core_range_set.num_cores()
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
            sh, sw = K, per_core_N
        else:
            sh, sw = self.shard_shape
            assert sh % tile_hw == 0, f"shard height ({sh}) must be a multiple of tile_hw ({tile_hw})"
            assert sw % tile_hw == 0, f"shard width ({sw}) must be a multiple of tile_hw ({tile_hw})"
        shard_spec = ttnn.ShardSpec(self.core_range_set, [sh, sw], ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(self.layout, ttnn.BufferType.L1, shard_spec)


def sram_projection_specs(
    core_grids: SramExpertCoreGrids,
    mesh_shape: tuple[int, int],
) -> dict[str, _MemoryConfigSpec]:
    """Return one :class:`_MemoryConfigSpec` per projection (gate / up / down).

    Single source of truth for the SRAM hot expert TP-layout decision shared
    between cost prediction (:func:`compute_expert_l1_bytes_per_core`) and
    on-device allocation (``prepare_compressed_sram_slots``).

    Under TP>1: HEIGHT_SHARDED gate/up via :func:`preprocess_gate_up`'s
    reshuffle output, WIDTH_SHARDED shared-down via
    :func:`shared_down_torch_for_cache`'s K-reshape output.  Under TP=1:
    WIDTH_SHARDED everything with auto-derived ``(K, N // num_cores)`` shards
    from the per-projection ``core_grids``.
    """
    moe_tp = mesh_shape[0] * mesh_shape[1]
    if moe_tp > 1:
        cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
        return {
            "gate": _MemoryConfigSpec(
                layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                core_range_set=cfg.gate_core_range_set,
                shard_shape=cfg.shard_shape,
            ),
            "up": _MemoryConfigSpec(
                layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                core_range_set=cfg.up_core_range_set,
                shard_shape=cfg.shard_shape,
            ),
            "down": _MemoryConfigSpec(
                layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                core_range_set=core_grids.down,
                shard_shape=(_SHARED_EXPERT_DOWN_K_PER_DEV, _SHARED_EXPERT_DOWN_N_PER_CORE),
            ),
        }
    return {
        "gate": _MemoryConfigSpec(layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, core_range_set=core_grids.gate),
        "up": _MemoryConfigSpec(layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, core_range_set=core_grids.up),
        "down": _MemoryConfigSpec(layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, core_range_set=core_grids.down),
    }


def _quantize_dequantize_bfp_fn(x, fmt_str):
    """Dequantizer used by ``CompressedTensorAssigner`` for bfp{0,2,4,8}."""
    from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import quantize_dequantize_bfp

    mant_map = {"bfp8": 7, "bfp4": 3, "bfp2": 1, "bfp0": 0}
    return quantize_dequantize_bfp(x, mant_map[fmt_str])


def _expert_assignments_and_shapes(
    expert_idx: int,
    gate_full: torch.Tensor,
    up_full: torch.Tensor,
    down_full: torch.Tensor,
    *,
    assigner: CompressedTensorAssigner | None,
    assignment_provider: Callable[[int, int], np.ndarray] | None,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Compute (assignments, shapes) for one expert's three projections.

    Drives :func:`compute_expert_l1_bytes_per_core` for budgeting.  Mutually
    exclusive: provide ``assigner`` (CPU-side mixed-precision assignment) or
    ``assignment_provider`` (pre-computed BSPM tile maps), matching the
    contract of ``prepare_compressed_sram_slots``.

    Float32 cast is required: ``CompressedTensorAssigner`` runs through
    ``np.asarray(..., dtype=np.float32)`` and rejects bfloat16.
    """
    assert (assigner is None) != (assignment_provider is None), "Provide exactly one of assigner or assignment_provider"
    weights = [gate_full.float(), up_full.float(), down_full.float()]
    assignments: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    for proj_idx, w in enumerate(weights):
        if assigner is not None:
            result = assigner.assign(w, _quantize_dequantize_bfp_fn)
            assignments.append(result.assignment)
        else:
            assignments.append(assignment_provider(expert_idx, proj_idx))
        shapes.append(tuple(w.shape))
    return assignments, shapes


def _predict_expert_per_core_bytes(
    expert_idx: int,
    gate_full: torch.Tensor,
    up_full: torch.Tensor,
    down_full: torch.Tensor,
    core_grids: SramExpertCoreGrids,
    *,
    assigner: CompressedTensorAssigner | None,
    assignment_provider: Callable[[int, int], np.ndarray] | None,
    tile_hw: int = 32,
    mesh_shape: tuple[int, int] = (1, 1),
) -> dict[tuple[int, int], int]:
    """Predict per-core L1 byte cost of a single expert (host-side lookahead).

    Wraps :func:`_expert_assignments_and_shapes` +
    :func:`compute_expert_l1_bytes_per_core` so the trim loop in
    ``prepare_compressed_sram_slots`` can stop *before* an over-budget
    allocation is attempted.  The same assignments are recomputed inside
    ``CompressedTensor.from_torch`` during the actual allocation -- worth
    the duplicated CPU work since predicting and allocating live in
    different code paths today.
    """
    assignments, shapes = _expert_assignments_and_shapes(
        expert_idx,
        gate_full,
        up_full,
        down_full,
        assigner=assigner,
        assignment_provider=assignment_provider,
    )
    return compute_expert_l1_bytes_per_core(assignments, shapes, core_grids, tile_hw=tile_hw, mesh_shape=mesh_shape)


def compute_expert_l1_bytes_per_core(
    assignments: list[np.ndarray],
    tensor_shapes: list[tuple[int, int]],
    core_grids: SramExpertCoreGrids,
    tile_hw: int = 32,
    mesh_shape: tuple[int, int] = (1, 1),
) -> dict[tuple[int, int], int]:
    """Per-core L1 byte cost for one expert across all projections.

    Each projection's shard-to-core tile mapping is computed via
    ``compute_shard_page_mapping``, then per-tile byte sizes (determined by
    the assignment codes) are summed per ``(core.x, core.y)``.  The per-core
    totals are accumulated across all projections and returned keyed by
    ``(x, y)``.  Cores that hold no shard for a given projection simply
    contribute zero bytes for that projection.

    Args:
        assignments: 3 projection assignment arrays in order
            ``[gate, up, down]``, each ``(tiles_h, tiles_w)`` with tt-metal
            format codes (0=bfp8, 1=bfp4, 2=bfp2, 3=bfp0).
        tensor_shapes: 3 projection ``(K, N)`` **logical** shapes (pre-TP,
            i.e. full tensor) in the same order.  Must match assignment tile
            counts.  TP division is applied via *mesh_shape*.
        core_grids: :class:`SramExpertCoreGrids` giving one grid per
            projection.  Use :meth:`SramExpertCoreGrids.uniform` for symmetric
            unit-test weights.  The grids refer to **per-device** cores.
        tile_hw: Tile dimension (default 32).
        mesh_shape: ``(mesh_rows, mesh_cols)`` for TP-sharded SRAM slots
            mirroring the shared-expert TP8 layout (dim-0 split along
            mesh_rows, dim-1 split along mesh_cols).  Defaults to ``(1, 1)``
            (no TP).

    Returns:
        ``dict[(core.x, core.y) -> bytes]`` summed across all projections.
        Use :func:`compute_expert_l1_bytes` for the scalar max-per-core cost.
    """
    from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import compute_shard_page_mapping
    from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import bfp_tile_packed_size

    _MANT_BITS = {0: 7, 1: 3, 2: 1, 3: 0}
    tile_byte_lut = np.array(
        [bfp_tile_packed_size(_MANT_BITS[c], tile_hw) for c in range(4)],
        dtype=np.int64,
    )

    assert len(assignments) == 3 and len(tensor_shapes) == 3, (
        "compute_expert_l1_bytes expects exactly 3 projections in order [gate, up, down], "
        f"got {len(assignments)} assignments / {len(tensor_shapes)} shapes"
    )
    projection_names = ("gate", "up", "down")
    mesh_rows, mesh_cols = mesh_shape
    moe_tp = mesh_rows * mesh_cols
    # Single-source TP-layout decision: ``sram_projection_specs`` returns the
    # same per-projection ``_MemoryConfigSpec`` triple consumed by the
    # on-device allocator in ``prepare_compressed_sram_slots``.  Branching
    # below is driven entirely by ``spec.layout`` / ``spec.shard_shape`` so
    # the predictor and allocator cannot disagree on layout policy.
    specs = sram_projection_specs(core_grids, mesh_shape)

    per_core_totals: dict[tuple[int, int], int] = {}

    for assignment, (K, N), proj_name in zip(assignments, tensor_shapes, projection_names):
        spec = specs[proj_name]
        if spec.layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            # Shared-expert HEIGHT_SHARDED gate/up (TP>1 only).  Each device
            # holds ONE tp slab of the logical (K, N): full K, per_device_n =
            # N/moe_tp columns.  Within the slab,
            # ``reshuffle_block_to_height_sharded`` splits into k_par * n_par
            # blocks, permutes by ``_crs_shard_permutation`` to match CRS
            # iteration order, and stacks along height.  We approximate
            # max-per-core bytes using tp_idx=0's slab (device (0, 0));
            # different tp slabs can differ slightly in tile-format
            # distribution but the per-core block structure is identical.
            cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
            sh, sw = spec.shard_shape
            k_par, n_par = cfg.k_parallel, cfg.n_parallel
            sh_tiles = sh // tile_hw
            sw_tiles = sw // tile_hw
            core_range_set = spec.core_range_set
            num_cores = core_range_set.num_cores()
            assert (
                num_cores == k_par * n_par
            ), f"shared-expert {proj_name} grid must have {k_par * n_par} cores; got {num_cores}"
            assert (
                K == cfg.gate_proj_shape[0]
            ), f"shared-expert {proj_name} requires K == {cfg.gate_proj_shape[0]}; got {K}"
            per_device_n = cfg.gate_proj_shape[1]
            assert N == per_device_n * moe_tp, (
                f"shared-expert {proj_name} requires N == {per_device_n * moe_tp} "
                f"(per_device_n={per_device_n} * moe_tp={moe_tp}); got {N}"
            )

            # Extract tp_idx=0 slab (first per_device_n cols) and apply the
            # reshuffle permutation to obtain the per-core block order that
            # matches the on-device HEIGHT_SHARDED memory config.
            per_device_n_tiles = n_par * sw_tiles
            assignment_dev = assignment[:, :per_device_n_tiles]
            a = assignment_dev.reshape(k_par, sh_tiles, n_par, sw_tiles).transpose(0, 2, 1, 3).copy()
            block_shards = a.reshape(k_par * n_par, sh_tiles, sw_tiles)
            perm = cfg._crs_shard_permutation(core_range_set)
            reshuffled_flat = block_shards[list(perm)].reshape(-1, sw_tiles).ravel()

            shard_spec = ttnn.ShardSpec(core_range_set, [sh, sw], ttnn.ShardOrientation.ROW_MAJOR)
            mem_config = ttnn.MemoryConfig(spec.layout, ttnn.BufferType.L1, shard_spec)
            stacked_h = num_cores * sh
            shard_mapping = compute_shard_page_mapping([stacked_h, sw], mem_config, tile_hw)

            for core, page_indices in shard_mapping:
                shard_bytes = int(tile_byte_lut[reshuffled_flat[list(page_indices)]].sum())
                key = (core.x, core.y)
                per_core_totals[key] = per_core_totals.get(key, 0) + shard_bytes
            continue

        # WIDTH_SHARDED path.  Covers:
        #   (a) shared-expert down under TP>1 (``spec.shard_shape`` is the
        #       explicit (K_per_dev=256, N_per_core=64) pair): per-device
        #       slab = assignment[0:8, 0:224] (tp_idx=0, device (0, 0) in
        #       logical tile coords -- strided across mesh rows/cols but
        #       the per-core max-bytes is identical on all devices).
        #   (b) single-device (TP=1) gate/up/down (``spec.shard_shape`` is
        #       None): top-left (K, N) slab with per_core_N = N / num_cores.
        num_cores = spec.core_range_set.num_cores()
        if spec.shard_shape is not None:
            K_dev, per_core_N = spec.shard_shape
            N_dev = per_core_N * num_cores
            expected_down_shape = (K_dev * moe_tp, per_core_N * num_cores)
            assert (K, N) == expected_down_shape, (
                f"shared-expert down requires logical (K, N) == {expected_down_shape} "
                f"(K_per_dev * moe_tp, N_per_core * num_cores); got ({K}, {N})"
            )
        else:
            assert K % mesh_rows == 0, f"K ({K}) must be divisible by mesh_rows ({mesh_rows})"
            assert N % mesh_cols == 0, f"N ({N}) must be divisible by mesh_cols ({mesh_cols})"
            K_dev = K // mesh_rows
            N_dev = N // mesh_cols
            per_core_N = N_dev // num_cores

        shard_spec = ttnn.ShardSpec(spec.core_range_set, [K_dev, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(spec.layout, ttnn.BufferType.L1, shard_spec)
        shard_mapping = compute_shard_page_mapping([K_dev, N_dev], mem_config, tile_hw)

        # Per-device assignment slab in logical tile coords.  For device
        # (0, 0) both the legacy and shared-expert-down paths reduce to the
        # top-left ``(K_dev_tiles, N_dev_tiles)`` block (the shared-expert
        # K-reshape trick strides K-rows across devices but device (0, 0)
        # always lands on logical tile rows [0, K_per_dev/tile_hw)).
        K_dev_tiles = K_dev // tile_hw
        N_dev_tiles = N_dev // tile_hw
        assignment_dev = assignment[:K_dev_tiles, :N_dev_tiles]
        assignment_flat = assignment_dev.ravel()

        for core, page_indices in shard_mapping:
            shard_bytes = int(tile_byte_lut[assignment_flat[list(page_indices)]].sum())
            key = (core.x, core.y)
            per_core_totals[key] = per_core_totals.get(key, 0) + shard_bytes

    return per_core_totals


def compute_expert_l1_bytes(
    assignments: list[np.ndarray],
    tensor_shapes: list[tuple[int, int]],
    core_grids: SramExpertCoreGrids,
    tile_hw: int = 32,
    mesh_shape: tuple[int, int] = (1, 1),
) -> int:
    """Max per-core L1 byte cost for one expert across all projections.

    Scalar wrapper around :func:`compute_expert_l1_bytes_per_core`; returns
    the byte total on the most loaded core (or 0 when no projection lands on
    any core).  Retained as a diagnostic/summary helper; the per-core dict
    form is what ``prepare_compressed_sram_slots`` uses for
    attention-plus-SRAM budget accounting against real allocator addresses.
    """
    per_core = compute_expert_l1_bytes_per_core(
        assignments, tensor_shapes, core_grids, tile_hw=tile_hw, mesh_shape=mesh_shape
    )
    return max(per_core.values()) if per_core else 0


def _load_routing_frequencies(path: Path | None = None) -> dict[int, list[int]]:
    """Load per-layer routing frequencies from a JSON file.

    The file maps ``"layer_idx" → list[int]`` where each list has 256 entries
    (one per routed expert) representing activation counts from calibration.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "data" / "routing_frequencies.json"
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_sram_hot_expert_config(
    layer_indices: list[int],
    routing_frequencies: dict[int, list[int]],
) -> SramHotExpertConfig:
    """Rank SRAM hot expert candidates by routing frequency (host-only).

    For each layer in ``layer_indices``, experts are sorted by their
    activation count in ``routing_frequencies`` (descending) with
    zero-frequency experts dropped.  The returned config is a pure
    *candidate ranking* -- no host-side budgeting is performed; the actual
    per-core L1 fit is decided device-side by
    ``prepare_compressed_sram_slots`` against an absolute L1 address
    boundary (see ``prepare_moe_layer_weights``).

    Args:
        layer_indices: MoE layer indices to consider.
        routing_frequencies: ``layer_idx → list[int]`` activation counts.

    Returns:
        ``SramHotExpertConfig`` mapping ``layer_idx → ranked list[expert_idx]``.
    """
    config: SramHotExpertConfig = {}

    for layer_idx in layer_indices:
        freqs = routing_frequencies.get(layer_idx)
        if freqs is None:
            logger.warning("No routing frequencies for layer {}, skipping", layer_idx)
            continue

        ranked_experts = [e for e in sorted(range(len(freqs)), key=lambda e: freqs[e], reverse=True) if freqs[e] > 0]
        if ranked_experts:
            config[layer_idx] = ranked_experts
            logger.info(
                "Layer {}: {} SRAM expert candidates ranked by routing frequency", layer_idx, len(ranked_experts)
            )

    return config
