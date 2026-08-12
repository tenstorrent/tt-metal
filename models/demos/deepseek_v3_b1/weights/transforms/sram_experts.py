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
from models.demos.deepseek_v3_b1.compressed_tensor import COMPRESSED_FORMATS, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_layer
from models.demos.deepseek_v3_b1.model_dimensions import RoutedExpert as _RoutedExpert
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
        passes to :func:`build_sram_routed_proj_ct` under TP>1 (HEIGHT_SHARDED
        gate/up, WIDTH_SHARDED down).  Call sites (tests, demos) should prefer
        this over re-stitching CRS from overlap_configs.
        """
        return cls(
            gate=GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC.gate_core_range_set,
            up=GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC.up_core_range_set,
            down=DOWN_PROJ_SINGLE_DEVICE_SPEC.build_matmul_core_grid(),
        )


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


_BFP4_FMT_IDX = COMPRESSED_FORMATS.index("bfp4")
# (K, N) full-tensor shapes per projection (0=gate, 1=up, 2=down) in routed
# expert compute layout.  Drives the assignment_provider's full-tile-grid sizing.
_PROJ_SHAPES = (
    (_RoutedExpert.K, _RoutedExpert.GATE_PROJ_N),  # gate
    (_RoutedExpert.K, _RoutedExpert.GATE_PROJ_N),  # up
    (_RoutedExpert.GATE_PROJ_N, _RoutedExpert.K),  # down
)


def _proj_tile_shape(proj_idx: int, num_banks: int, tile_w: int) -> tuple[int, int]:
    """``(tiles_h_full, tiles_w_full_padded)`` for one projection.

    Padding rounds the N axis up to a full ``num_banks * tile_w`` row so the
    BSPM tile grid matches the DRAM-side padded layout (see CompressedTensor
    bank-aligned packing).
    """
    K_full, N_full = _PROJ_SHAPES[proj_idx]
    pad_step = num_banks * tile_w
    N_padded_full = ((N_full + pad_step - 1) // pad_step) * pad_step
    return K_full // tile_w, N_padded_full // tile_w


def make_sram_assignment_provider(
    *,
    bspm_path: Path | None,
    num_banks: int,
    layer_idx: int,
    tile_w: int = 32,
) -> Callable[[int, int], np.ndarray]:
    """Build the ``(expert_idx, proj_idx) -> assignment`` closure consumed by
    :func:`prepare_compressed_sram_slots`.

    Two modes, same allocator path:

    * **With BSPM** (``bspm_path`` set): load codes from the .bspm file once,
      then slice per (expert, projection).  Per-tile precision varies per
      expert/projection.
    * **Without BSPM**: return a constant array of BFP4 format indices
      sized to the projection's full padded tile grid.  Every tile is
      uniform BFP4.

    Both modes return a 2D ``(tiles_h_full, tiles_w_full_padded)`` int array
    matching the projection's bank-aligned layout.
    """
    if bspm_path is not None:
        logger.info("Loading BSPM for SRAM auto-fit at layer {}: {}", layer_idx, bspm_path)
        bspm_data = load_bspm_for_layer(str(bspm_path))

        def _provider(expert_idx: int, proj_idx: int) -> np.ndarray:
            tiles_h_full, tiles_w_full_padded = _proj_tile_shape(proj_idx, num_banks, tile_w)
            assert (
                expert_idx < bspm_data["n_experts"]
            ), f"SRAM expert {expert_idx} out of range for BSPM (n_experts={bspm_data['n_experts']})"
            return np.ascontiguousarray(
                bspm_data["codes"][expert_idx, proj_idx].reshape(tiles_w_full_padded, tiles_h_full).T
            )

        return _provider

    def _uniform_bfp4_provider(expert_idx: int, proj_idx: int) -> np.ndarray:
        tiles_h_full, tiles_w_full_padded = _proj_tile_shape(proj_idx, num_banks, tile_w)
        return np.full((tiles_h_full, tiles_w_full_padded), _BFP4_FMT_IDX, dtype=np.int8)

    return _uniform_bfp4_provider


def _predict_expert_per_core_bytes(
    expert_idx: int,
    gate_full: torch.Tensor,
    up_full: torch.Tensor,
    down_full: torch.Tensor,
    core_grids: SramExpertCoreGrids,
    n_parallel_per_proj: tuple[int, int, int],
    *,
    assigner: CompressedTensorAssigner | None,
    assignment_provider: Callable[[int, int], np.ndarray] | None,
    tile_hw: int = 32,
    mesh_shape: tuple[int, int] = (1, 1),
) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    """Predict per-device, per-core L1 byte cost of a single expert.

    Wraps :func:`_expert_assignments_and_shapes` +
    :func:`compute_expert_l1_bytes_per_core` so the trim loop in
    ``prepare_compressed_sram_slots`` can stop *before* an over-budget
    allocation is attempted.  The same assignments are recomputed inside
    ``CompressedTensor.from_torch`` during the actual allocation -- worth
    the duplicated CPU work since predicting and allocating live in
    different code paths today.

    See :func:`compute_expert_l1_bytes_per_core` for ``n_parallel_per_proj``
    semantics — production callers should pass the same triple they pass to
    :func:`build_sram_routed_proj_ct` so the two stay in lock-step.

    Returns ``dict[(mesh_row, mesh_col) -> dict[(x, y) -> bytes]]``; under
    uniform BFP4 all devices match, under BSPM they diverge and callers
    must reduce via max-over-devices for a uniform-across-devices placement.
    """
    assignments, shapes = _expert_assignments_and_shapes(
        expert_idx,
        gate_full,
        up_full,
        down_full,
        assigner=assigner,
        assignment_provider=assignment_provider,
    )
    return compute_expert_l1_bytes_per_core(
        assignments,
        shapes,
        core_grids,
        n_parallel_per_proj,
        tile_hw=tile_hw,
        mesh_shape=mesh_shape,
    )


def reduce_per_device_max(
    per_device_totals: dict[tuple[int, int], dict[tuple[int, int], int]],
) -> dict[tuple[int, int], int]:
    """Reduce per-device per-core bytes via max-over-devices.

    Picks the worst-case (largest) byte total for each core across all
    mesh coordinates -- the right reduction for a uniform-across-devices
    SRAM trim budget under BSPM, where different devices receive different
    tile-format slabs and would otherwise diverge on expert count.
    """
    out: dict[tuple[int, int], int] = {}
    for dev_totals in per_device_totals.values():
        for core_xy, byte_cnt in dev_totals.items():
            prev = out.get(core_xy, 0)
            if byte_cnt > prev:
                out[core_xy] = byte_cnt
    return out


def compute_expert_l1_bytes_per_core(
    assignments: list[np.ndarray],
    tensor_shapes: list[tuple[int, int]],
    core_grids: SramExpertCoreGrids,
    n_parallel_per_proj: tuple[int, int, int],
    tile_hw: int = 32,
    mesh_shape: tuple[int, int] = (1, 1),
) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    """Per-device, per-core L1 byte cost for one expert across all projections.

    Mirrors :func:`build_sram_routed_proj_ct`'s geometry exactly so the trim
    loop can stop *before* an over-budget allocation.  Per-projection layout
    is driven by ``n_parallel_per_proj`` (same triple the caller passes to
    :func:`build_sram_routed_proj_ct`):

      * ``k_par = num_cores // n_par`` ⇒ HEIGHT_SHARDED block stack when
        ``k_par > 1``, WIDTH_SHARDED single-row when ``k_par == 1``.
      * Per-core block is ``(K_per_dev/k_par, N_per_dev/n_par)`` in tiles,
        indexed as ``(k_idx, n_idx) = (i // n_par, i % n_par)`` for the
        i-th core in row-major ``corerange_to_cores`` order — identical to
        the allocator's per-core slab assignment.

    Per-device shard axis:
      * gate / up — column-shard ``N`` across ``moe_tp`` devices.
      * down — row-shard ``K`` across ``moe_tp`` devices.

    Under uniform BFP4 every device's per-core dict is identical.  Under
    BSPM the format-code distribution differs per device, so the trim loop
    must reduce via :func:`reduce_per_device_max` for a uniform-across-
    devices placement.

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
        n_parallel_per_proj: ``(gate, up, down)`` triple of ``n_parallel``
            values — same values the caller passes to
            :func:`build_sram_routed_proj_ct`.
        tile_hw: Tile dimension (default 32).
        mesh_shape: ``(mesh_rows, mesh_cols)`` for TP-sharded SRAM slots
            (dim-0 split along mesh_rows, dim-1 split along mesh_cols).
            Defaults to ``(1, 1)`` (no TP).

    Returns:
        ``dict[(mesh_row, mesh_col) -> dict[(core.x, core.y) -> bytes]]``
        summed across all projections.  Use :func:`compute_expert_l1_bytes`
        for the scalar worst-case cost across all devices and cores.
    """
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
    mesh_rows, mesh_cols = mesh_shape
    moe_tp = mesh_rows * mesh_cols
    per_proj_grids = (core_grids.gate, core_grids.up, core_grids.down)
    per_proj_is_down = (False, False, True)

    per_device_totals: dict[tuple[int, int], dict[tuple[int, int], int]] = {
        (r, c): {} for r in range(mesh_rows) for c in range(mesh_cols)
    }

    for proj_idx, (assignment, (K, N), grid, is_down) in enumerate(
        zip(assignments, tensor_shapes, per_proj_grids, per_proj_is_down)
    ):
        cores = ttnn.corerange_to_cores(grid, row_wise=True)
        num_cores = len(cores)
        n_par = n_parallel_per_proj[proj_idx]
        assert (
            n_par > 0 and num_cores % n_par == 0
        ), f"n_parallel ({n_par}) must divide num_cores ({num_cores}) for projection {proj_idx}"
        k_par = num_cores // n_par

        # Allocator (``prepare_compressed_sram_slots``) slices linearly by
        # ``moe_tp`` (= mesh_rows * mesh_cols), assigning the d-th N-slab
        # (gate/up) or K-slab (down) to linear device d.  Predictor MUST
        # divide by ``moe_tp`` — dividing by ``mesh_cols`` / ``mesh_rows``
        # would under-slice on a 4×2 mesh and overestimate per-core bytes.
        if is_down:
            assert K % moe_tp == 0, f"down K ({K}) must be divisible by moe_tp ({moe_tp})"
            K_per_dev = K // moe_tp
            N_per_dev = N
        else:
            K_per_dev = K
            assert N % moe_tp == 0, f"gate/up N ({N}) must be divisible by moe_tp ({moe_tp})"
            N_per_dev = N // moe_tp

        K_per_dev_tiles = K_per_dev // tile_hw
        N_per_dev_tiles = N_per_dev // tile_hw
        assert K_per_dev_tiles % k_par == 0, f"K_per_dev_tiles ({K_per_dev_tiles}) not divisible by k_par ({k_par})"
        assert N_per_dev_tiles % n_par == 0, f"N_per_dev_tiles ({N_per_dev_tiles}) not divisible by n_par ({n_par})"
        K_block_tiles = K_per_dev_tiles // k_par
        N_block_tiles = N_per_dev_tiles // n_par

        for tp_idx in range(moe_tp):
            tp_coord = (tp_idx // mesh_cols, tp_idx % mesh_cols)
            if is_down:
                # Row-shard K across devices.
                assignment_dev = assignment[tp_idx * K_per_dev_tiles : (tp_idx + 1) * K_per_dev_tiles, :N_per_dev_tiles]
            else:
                # Column-shard N across devices.
                assignment_dev = assignment[:K_per_dev_tiles, tp_idx * N_per_dev_tiles : (tp_idx + 1) * N_per_dev_tiles]

            dev_totals = per_device_totals[tp_coord]
            for i, core in enumerate(cores):
                k_idx = i // n_par
                n_idx = i % n_par
                block = assignment_dev[
                    k_idx * K_block_tiles : (k_idx + 1) * K_block_tiles,
                    n_idx * N_block_tiles : (n_idx + 1) * N_block_tiles,
                ]
                shard_bytes = int(tile_byte_lut[block.ravel()].sum())
                key = (core.x, core.y)
                dev_totals[key] = dev_totals.get(key, 0) + shard_bytes

    return per_device_totals


def compute_expert_l1_bytes(
    assignments: list[np.ndarray],
    tensor_shapes: list[tuple[int, int]],
    core_grids: SramExpertCoreGrids,
    n_parallel_per_proj: tuple[int, int, int],
    tile_hw: int = 32,
    mesh_shape: tuple[int, int] = (1, 1),
) -> int:
    """Max per-core L1 byte cost for one expert across all devices and projections.

    Scalar wrapper around :func:`compute_expert_l1_bytes_per_core`; returns
    the worst-case byte total across every (device, core) (or 0 when no
    projection lands on any core).  Retained as a diagnostic/summary helper;
    the per-device per-core dict form is what ``prepare_compressed_sram_slots``
    uses for attention-plus-SRAM budget accounting against real allocator
    addresses.
    """
    per_device = compute_expert_l1_bytes_per_core(
        assignments,
        tensor_shapes,
        core_grids,
        tile_hw=tile_hw,
        mesh_shape=mesh_shape,
        n_parallel_per_proj=n_parallel_per_proj,
    )
    return max(
        (b for per_core in per_device.values() for b in per_core.values()),
        default=0,
    )


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
