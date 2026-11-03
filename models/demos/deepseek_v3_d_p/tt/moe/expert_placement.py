# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cache-coupled, bijective placement for GLM-5.3 on an eight-bank 8x4 Galaxy.

Scoring and stable selection stay in logical expert order. Only the final IDs
are translated; the prepared BF4 cache stores the matching original weight bytes.
"""

import json
from pathlib import Path

NUM_EXPERTS = 256
GROUP_SIZE = 64
PLACEMENT_FILE = "layer_{layer}.expert_placement.json"


def validate_mapping(mapping):
    if not isinstance(mapping, list) or len(mapping) != NUM_EXPERTS or any(type(x) is not int for x in mapping):
        raise ValueError("GLM placement requires 256 integer expert IDs")
    if sorted(mapping) != list(range(NUM_EXPERTS)):
        raise ValueError("GLM placement must retain every expert exactly once")
    if any(logical // GROUP_SIZE != physical // GROUP_SIZE for logical, physical in enumerate(mapping)):
        raise ValueError("GLM placement must preserve the four TP dispatch groups")
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
        or tuple(mesh_device.shape) != (8, 4)
        or mesh_device.get_num_devices() != 32
    ):
        raise ValueError("GLM placement requires an 8x4 Blackhole Galaxy")
    if (num_experts, top_k, experts_per_chip, dispatch_group_size, num_dispatch_groups) != (256, 8, 8, 8, 4):
        raise ValueError("GLM placement requires 256 experts, top-8, and SP=8/TP=4")
    if expert_dtype != ttnn.bfloat4_b or data.get("expert_dtype") != "BFLOAT4_B":
        raise ValueError("GLM placement requires the original BF4 expert cache")
    if data.get("mesh_shape") != [8, 4] or data.get("num_experts") != 256:
        raise ValueError("Placement cache topology or expert count mismatch")
    mapping = validate_mapping(data.get("logical_to_physical"))
    inverse = validate_mapping(data.get("physical_to_logical"))
    if any(inverse[p] != e for e, p in enumerate(mapping)):
        raise ValueError("Placement inverse does not match its expert map")
    return mapping
