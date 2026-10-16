# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the flat_routed_expert plans at the precision harness's shapes (8 experts, capacity 512), side by side."""
import json
import os

import pytest

import ttnn

SHAPES = {"glm53": (6144, 2048, False), "glm53_hbf16": (6144, 2048, True), "k27": (7168, 2048, False), "k3": (3584, 3072, False)}


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_plan_cmp(mesh_device):
    out = {}
    for name, (H, I, hb) in SHAPES.items():
        p = ttnn.bringup.flat_routed_expert_plan(mesh_device, H, I, 8, 64, 512, False, 1, False, hb)
        out[name] = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(p).items()}
    json.dump(out, open(os.environ["PLAN_OUT"], "w"), default=lambda o: list(o) if not isinstance(o, (int, float, str)) else o)
