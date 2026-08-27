# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Canonical WH Galaxy `Prefetcher2D` construction policy.

The sender/receiver mapping, dummy-core padding, and global circular-buffer
size are properties of the Wormhole Galaxy decode topology, not of any model.
A model creates exactly one prefetcher for its mesh, registers its prefetched
decode weights in issue order, seals registration, and hands the resolved
contexts to its module configs.
"""

from __future__ import annotations

from typing import Any

import ttnn
from models.common.models.galaxy.recipes import prefetch_sender_cores
from models.common.models.galaxy.resources import GalaxyModePlan, GalaxyResourcesConfig
from models.common.modules.prefetcher import Prefetcher2D, Prefetcher2DConfig, Prefetcher2DModeConfig

GALAXY_GLOBAL_CB_SIZE = 728 * 1088

_RECEIVER_COLUMN_PAIRS = tuple(((1, y), (2, y)) for y in (9, 0, 4, 5)) + tuple(
    ((5, y), (6, y)) for y in (0, 9, 1, 7, 6, 2, 4, 5)
)
_DUMMY_SENDER_COORDS = ((0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (4, 3), (4, 8))
_DUMMY_RECEIVER_RANGES = (
    ((3, 0, 3, 0), (1, 1, 3, 1)),
    ((1, 2, 3, 2),),
    ((1, 3, 3, 3), (3, 4, 3, 4)),
    ((3, 5, 3, 5), (1, 6, 3, 6)),
    ((1, 7, 3, 7),),
    ((1, 8, 3, 8), (3, 9, 3, 9)),
    ((5, 3, 6, 3),),
    ((5, 8, 6, 8),),
)


def _ranges(coordinates: tuple[tuple[int, int, int, int], ...]) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in coordinates]
    )


def galaxy_sender_receiver_mapping() -> tuple[tuple[Any, Any], ...]:
    """Return the canonical `(sender core, receiver core set)` prefetch mapping."""

    senders = prefetch_sender_cores() + tuple(ttnn.CoreCoord(x, y) for x, y in _DUMMY_SENDER_COORDS)
    receivers = tuple(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(*end))})
        for start, end in _RECEIVER_COLUMN_PAIRS
    ) + tuple(_ranges(coordinates) for coordinates in _DUMMY_RECEIVER_RANGES)
    if len(senders) != len(receivers):
        raise ValueError("Galaxy prefetch sender and receiver counts must match")
    return tuple(zip(senders, receivers))


def galaxy_address_memory_config(weight_count: int) -> ttnn.MemoryConfig:
    """Return the packed weight-address placement on the real sender cores."""

    if weight_count <= 0:
        raise ValueError("prefetched weight count must be positive")
    senders = prefetch_sender_cores()
    sender_cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in senders])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_cores, [1, weight_count], ttnn.ShardOrientation.ROW_MAJOR),
    )


def galaxy_dram_prefetch_start(*, tensors_per_layer: int, num_layers: int) -> Any:
    """Return the Galaxy decode prefetch producer.

    The DRAM prefetcher streams one layer's tensor set at a time and walks the
    remaining layers through the packed address table, so it receives the first
    layer's tensors plus the address metadata rather than every weight.
    """

    if tensors_per_layer <= 0 or num_layers <= 0:
        raise ValueError("tensors_per_layer and num_layers must be positive")

    def start(context: Any) -> Any:
        weights = list(context.weights[:tensors_per_layer])
        if len(weights) != tensors_per_layer:
            raise RuntimeError(
                f"decode prefetch requires {tensors_per_layer} registered weights per layer, got {len(weights)}"
            )
        return ttnn.dram_prefetcher(
            weights + [context.weight_address_metadata],
            num_layers=num_layers,
            global_cb=context.global_cb,
        )

    return start


def _mode_config(plan: GalaxyModePlan) -> Prefetcher2DModeConfig:
    return Prefetcher2DModeConfig(
        mode=plan.mode,
        sub_devices=plan.sub_devices,
        worker_sub_device_id=plan.worker_sub_device_id,
        stall_group=plan.stall_group,
        local_l1_size=plan.local_l1_size,
    )


def build_galaxy_prefetcher_config(
    mesh_device: Any,
    resources_config: GalaxyResourcesConfig,
    *,
    expected_weight_count: int,
    global_cb_size: int | None = GALAXY_GLOBAL_CB_SIZE,
    prefetch_num_layers: int = 1,
    defer_global_cb: bool = True,
    release_global_cb_on_prefill: bool = False,
) -> Prefetcher2DConfig:
    """Resolve the prefetcher policy that matches a Galaxy resource config.

    ``defer_global_cb`` defaults to ``True`` here, and only here: on this mesh
    the global CB's ~774 kB of L1 per sender/receiver core makes every prefill
    program that needs static circular buffers on those cores unplaceable, and
    the Galaxy models all run prefill before decode. See the field's own
    docstring in ``Prefetcher2DConfig``; the production Galaxy prefetcher makes
    the same choice for the same reason.
    """

    return Prefetcher2DConfig(
        mesh_device=mesh_device,
        architecture=resources_config.architecture,
        prefill=_mode_config(resources_config.prefill),
        decode=_mode_config(resources_config.decode),
        sender_receiver_mapping=galaxy_sender_receiver_mapping(),
        global_cb_size=global_cb_size,
        expected_weight_count=expected_weight_count,
        address_repeat_count=len(prefetch_sender_cores()),
        address_memory_config=galaxy_address_memory_config(expected_weight_count),
        address_mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        prefetch_num_layers=prefetch_num_layers,
        mesh_shape=resources_config.mesh_shape,
        defer_global_cb=defer_global_cb,
        # Defaults to False: see the field's docstring. Deferring the allocation is
        # safe and qualified; releasing and recreating it per mode is not, and it is
        # opt-in until something re-qualifies the decode path behind it.
        release_global_cb_on_prefill=release_global_cb_on_prefill,
    )


def build_galaxy_prefetcher(
    mesh_device: Any,
    resources_config: GalaxyResourcesConfig,
    *,
    expected_weight_count: int,
    global_cb_size: int | None = GALAXY_GLOBAL_CB_SIZE,
    prefetch_num_layers: int = 1,
    release_global_cb_on_prefill: bool = False,
    **injections: Any,
) -> Prefetcher2D:
    """Create an initialized, unsealed `Prefetcher2D` for one Galaxy mesh."""

    prefetcher = Prefetcher2D(
        build_galaxy_prefetcher_config(
            mesh_device,
            resources_config,
            expected_weight_count=expected_weight_count,
            global_cb_size=global_cb_size,
            prefetch_num_layers=prefetch_num_layers,
            release_global_cb_on_prefill=release_global_cb_on_prefill,
        ),
        **injections,
    )
    try:
        prefetcher.initialize()
    except BaseException:
        prefetcher.cleanup()
        raise
    return prefetcher
