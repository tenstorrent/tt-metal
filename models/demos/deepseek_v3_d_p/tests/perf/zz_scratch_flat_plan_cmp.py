# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: flat_routed_expert plans side by side (PLAN_SHAPES: name=H:I:E:NG:M:h_bf16, ...; default the precision
harness's shapes and the production per-chip loads). Reports the reader-tail out region risk: pcd_r > max(pcds)."""
import json
import os

import pytest

import ttnn

DEFAULT = (
    "glm53=6144:2048:8:64:512:0,glm53_hbf16=6144:2048:8:64:512:1,k27=7168:2048:8:64:512:0,k3=3584:3072:8:64:512:0,"
    "glm53_lb=6144:2048:32:256:8192:0,glm53_flash_lb=4096:2048:36:288:8192:0,k27_lb=7168:2048:48:384:8192:0,"
    "k3_4x4=3584:3072:56:896:8192:0,k3_lb=3584:3072:56:448:8192:0"
)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_plan_cmp(mesh_device):
    out = {}
    for item in os.environ.get("PLAN_SHAPES", DEFAULT).split(","):
        name, spec = item.split("=")
        H, I, E, NG, M, hb = (int(v) for v in spec.split(":"))
        p = ttnn.bringup.flat_routed_expert_plan(mesh_device, H, I, E, NG, M, False, 1, False, bool(hb))
        d = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(p).items()}
        out[name] = d
        print(f"PLANCMP {name:16s} H {H} I {I} E {E}: mt {d['mt']} rdown {d['rdown']} pcd max {max(d['pcds'])} "
              f"pcd_r {d['pcd_r']} arena_tiles {d['arena_tiles']}"
              + ("  <- tails wider than the down cores (out region was undersized before the fix)" if d["rdown"] and d["pcd_r"] > max(d["pcds"]) else ""))
    if os.environ.get("PLAN_OUT"):
        json.dump(out, open(os.environ["PLAN_OUT"], "w"), default=lambda o: o if isinstance(o, (int, float, str)) else list(o))
