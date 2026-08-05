# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin: remap GLM-5's column-12 core assignments onto column 11.

Why: on a 1x-harvested BH Galaxy the compute grid is 12x10, so column 12 does not exist.
GLM-5's BlazeConfig pins `moe_router_gate_mm_cores` to x=12 (a 13x10 assumption), and both
the router and routed-expert tests then fail with

    Circular buffer core range [12-0 - 12-0] ... exceeds device compute grid (12x10)

Unlike the shared-expert grid (which is hardcoded to a 64/64 gate/up split that only
balances at 130 cores) this one is a plain config value, so it is remappable. Column 11
rows 0-7 are free: GridConfig puts the sender at (11,9) and the idle phantom at (11,8).

`pytest_configure` runs before test-module import, so the `from ... import GLM5_BLAZE_CONFIG`
name binding in the tests picks up the patched object.

DIAGNOSTIC. This says whether column 12 is the ONLY 13-column assumption on the
routed-expert path -- it is not a proposed fix.
"""

import dataclasses


def pytest_configure(config):
    import os

    from blaze.models.glm_5_1 import glm_5_1_blaze_config as mod

    cfg = mod.GLM5_BLAZE_CONFIG
    remapped = {}
    for field in ("moe_router_gate_mm_cores", "moe_shared_expert_up_coords", "moe_shared_expert_gate_coords"):
        coords = getattr(cfg, field, ())
        if not coords:
            continue
        if any(x >= 12 for x, _ in coords):
            # Clamp any column >= 12 down to 11, dropping duplicates that result.
            seen, out = set(), []
            for x, y in coords:
                t = (min(x, 11), y)
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            remapped[field] = tuple(out)

    # sender_core is a single (col, row), not a list. GLM-5 pins it to (12, 9); on a 12-wide
    # grid GridConfig.from_device would compute (11, 9) itself, so this just agrees with it.
    sc = getattr(cfg, "sender_core", None)
    if sc and sc[0] >= 12:
        remapped["sender_core"] = (11, sc[1])
        print(f"[remap_col12] sender_core: {sc} -> {remapped['sender_core']}")

    # Gate-MM core count must equal num_experts // 32 (one core per 32-expert tile).
    # GLM-5's config carries 8 (256 experts). Driving the op at GLM-4.7-Flash's 64 experts
    # while leaving 8 gate cores wired in HANGS the device -- observed 20+ min with no
    # output where the GLM-5 shape completes in ~190 s -- because the handshake is sized
    # for one count while a different set of cores participates. Truncate to match.
    n_exp = os.environ.get("BLAZE_AB_NUM_EXPERTS", "").strip()
    if n_exp:
        want = int(n_exp) // 32
        gm = remapped.get("moe_router_gate_mm_cores", cfg.moe_router_gate_mm_cores)
        if len(gm) != want:
            remapped["moe_router_gate_mm_cores"] = tuple(gm[:want])
            print(f"[remap_col12] gate_mm_cores truncated {len(gm)} -> {want} for {n_exp} experts")

    if remapped:
        mod.GLM5_BLAZE_CONFIG = dataclasses.replace(cfg, **remapped)
        for k, v in remapped.items():
            if v and isinstance(v[0], tuple):
                print(f"[remap_col12] {k}: -> {len(v)} cores, max x = {max(x for x, _ in v)}")
    else:
        print("[remap_col12] nothing to remap")
