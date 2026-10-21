# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: flat_routed_expert's plan at each model's per-chip expert shape (32 chips) with pure DRISC (no Tensix weight
readers), on a real Blackhole (DRISC needs the DRAM cores; bh-lb-09 has the Galaxy-like 12 x 10 grid). Prints the role
cores, the per-role arena layout and the cells the flat plan leaves to the combine."""

import json

import pytest
from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import DRISC_ONLY, GLM_FLASH, flat_layout, k3_gu64

import ttnn

SHAPES = {
    "Kimi K3": (3584, 3072, 28, 896, {**k3_gu64(True), "drisc": True, "drisc_pure": 1}),
    "Kimi K2.7": (7168, 2048, 12, 384, {**DRISC_ONLY, "drisc": True}),
    "GLM-5.3": (6144, 2048, 8, 256, {**DRISC_ONLY, "drisc": True}),
    "MiniMax M3": (6144, 3072, 4, 128, {**DRISC_ONLY, "drisc": True}),
    "GLM-5.3-Flash": (4096, 2048, 9, 288, {**GLM_FLASH, "drisc": True, "drisc_pure": 1}),
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_plan_galaxy(device):
    g = device.compute_with_storage_grid_size()
    out = {"grid": [g.x, g.y]}
    for name, (H, I, epc, E, lay) in SHAPES.items():
        try:
            p = ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
                device,
                H,
                I,
                epc,
                E,
                4096,
                weights_bf8=False,
                pin=1,
                x_bf16=False,
                h_bf16=False,
                layout=flat_layout(lay),
            )
        except RuntimeError as e:
            out[name] = {"error": str(e).splitlines()[0][:200]}
            continue
        d = {k: p[k] for k in ("np", "nk_gu", "nd", "x_slots", "hbuf", "arena_tiles", "rdown", "drisc")}
        keys = ("gu", "down", "relays", "readers", "gu_idle", "arena_cores") + (
            ("tensix_readers",) if "tensix_readers" in p else ()
        )
        d["roles"] = {k: len(p[k]) for k in keys}
        rd_key = "tensix_readers" if "tensix_readers" in p else "readers"
        taken = {tuple(c) for k in ("arena_cores", "gu", "down", rd_key, "relays", "coords") for c in p[k]}
        d["free_cells"] = sorted([(x, y) for y in range(g.y) for x in range(g.x) if (x, y) not in taken])
        d["off"] = dict(p["role_offsets"])
        out[name] = d
    print("PLANS " + json.dumps(out), flush=True)
