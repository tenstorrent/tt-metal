# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the flat weight cache's layout key (flat_expert.FlatRoutedExpert cache_prefix) of GLM-5.3-Flash's plan at
each MIMO_FL_ND (FLAT_KEY_ND, comma list), for a 2 x 4 mesh's gids - which cached layout a plan option reproduces."""
import hashlib
import os

import pytest

import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_cache_key(mesh_device):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import WEIGHT_LAYOUT_KEYS
    from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import flat_layout

    gids = [list(range(d * 36, (d + 1) * 36)) for d in range(8)]
    for nd in os.environ.get("FLAT_KEY_ND", "0").split(","):
        lo = flat_layout(max_down=int(nd)) if nd != "0" else flat_layout()
        lay = dict(
            ttnn.bringup.flat_routed_expert_plan(mesh_device, 4096, 2048, 36, 288, 5120, False, 1, False, False, lo)
        )
        layout = {k: lay[k] for k in WEIGHT_LAYOUT_KEYS if k in lay and (k != "dw_rot" or lay[k])}
        if lay.get("rgs"):
            layout["rgs"] = tuple(lay["rgs"])
        if lay.get("hg"):
            layout["hg"] = True
        if lay.get("nu"):
            layout["nu"] = lay["nu"]
        key = hashlib.sha1(repr((sorted(layout.items()), gids, 8)).encode()).hexdigest()[:12]
        print(f"FLATKEY nd {nd}: down {len(lay['down'])} key lay{key}", flush=True)
