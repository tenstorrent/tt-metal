# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cache-coupled, bijective expert placement on an eight-bank 8x4 Blackhole Galaxy.

Applies to any MoE whose routed experts are spread evenly over the 32 chips in column-major order
(global expert = (col * 8 + row) * experts_per_chip + slot) with one dispatch group per mesh column,
for example GLM-5.3 (256 experts, 8 per chip) and Kimi-K3 (896 experts, 28 per chip). A placement
may move an expert to any chip row of its own column, never across columns, so dispatch groups and
the expert count per chip are unchanged.

Scoring and stable selection stay in logical expert order. Only the final IDs
are translated; the prepared BF4 cache stores the matching original weight bytes.
"""

import json
from pathlib import Path

MESH_SHAPE = (8, 4)
NUM_DEVICES = MESH_SHAPE[0] * MESH_SHAPE[1]
NUM_GROUPS = MESH_SHAPE[1]
PLACEMENT_FILE = "layer_{layer}.expert_placement.json"


def validate_mapping(mapping, num_experts=None):
    if not isinstance(mapping, list) or any(type(x) is not int for x in mapping):
        raise ValueError("Expert placement requires a list of integer expert IDs")
    n = len(mapping)
    if n == 0 or n % NUM_DEVICES or (num_experts is not None and n != num_experts):
        raise ValueError(f"Expert placement requires {num_experts or 'a multiple of 32'} expert IDs, got {n}")
    if sorted(mapping) != list(range(n)):
        raise ValueError("Expert placement must retain every expert exactly once")
    group_size = n // NUM_GROUPS
    if any(logical // group_size != physical // group_size for logical, physical in enumerate(mapping)):
        raise ValueError("Expert placement must preserve the four TP dispatch groups")
    return mapping


def load_placement(
    cache_dir,
    layer,
    mesh_device,
    *,
    num_experts,
    top_k,
    experts_per_chip,
    dispatch_group_size,
    num_dispatch_groups,
    expert_dtype,
    cache_only,
):
    if cache_dir is None:
        return None
    marker = Path(cache_dir) / PLACEMENT_FILE.format(layer=layer)
    if not marker.exists():
        return None
    import ttnn

    data = json.loads(marker.read_text())
    if data.get("version") != 1 or data.get("complete") is not True:
        raise ValueError(f"Incomplete or unsupported placement marker: {marker}")
    if data.get("kind") != "bijective_within_tp_group" or data.get("layer") != layer:
        raise ValueError(f"Placement kind or layer mismatch: {marker}")
    if not cache_only:
        raise ValueError("A placed expert cache requires cache-only loading")
    if (
        mesh_device.arch() != ttnn.Arch.BLACKHOLE
        or tuple(mesh_device.shape) != MESH_SHAPE
        or mesh_device.get_num_devices() != NUM_DEVICES
    ):
        raise ValueError("Expert placement requires an 8x4 Blackhole Galaxy")
    if (
        (dispatch_group_size, num_dispatch_groups) != MESH_SHAPE
        or experts_per_chip * NUM_DEVICES != num_experts
        or not 0 < top_k <= 32
    ):
        raise ValueError("Expert placement requires experts spread evenly over SP=8/TP=4 and top-k <= 32")
    if expert_dtype != ttnn.bfloat4_b or data.get("expert_dtype") != "BFLOAT4_B":
        raise ValueError("Expert placement requires the original BF4 expert cache")
    if data.get("mesh_shape") != list(MESH_SHAPE) or data.get("num_experts") != num_experts:
        raise ValueError("Placement cache topology or expert count mismatch")
    mapping = validate_mapping(data.get("logical_to_physical"), num_experts)
    inverse = validate_mapping(data.get("physical_to_logical"), num_experts)
    if any(inverse[p] != e for e, p in enumerate(mapping)):
        raise ValueError("Placement inverse does not match its expert map")
    return mapping
