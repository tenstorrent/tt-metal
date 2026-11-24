# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Stage 2: GLM gate/up cores on x chains, today (mode 0) vs K split in pairs (mode 1); see ks_chain.py."""

import pytest
from loguru import logger

from models.demos.glm_ks_expert.ks_chain import ChainCfg, plan_geometry, run_chain

CASES = {
    "today_2x2": ChainCfg(mode=0, W=2, H=2, relay_y=0, nsb=8),
    "ks_2x2": ChainCfg(mode=1, W=2, H=2, relay_y=0, nsb=8),
    "today_1core": ChainCfg(mode=0, W=1, H=1, relay_y=0),
    "today_1core_w": ChainCfg(mode=0, W=1, H=1, relay_y=0, w_stream=True),
    "today_west": ChainCfg(mode=0),
    "ks_west": ChainCfg(mode=1),
    "ks_west_xs12": ChainCfg(mode=1, xs=12),
    "ks_west_xs16": ChainCfg(mode=1, xs=16),
    "today_west_w": ChainCfg(mode=0, w_stream=True),
    "ks_west_w": ChainCfg(mode=1, w_stream=True),
    "ks_west_xs16_w": ChainCfg(mode=1, xs=16, w_stream=True),
    # the chip: the GLM plan's 64 gate/up cores (both rectangles), relays, banks
    "today_chip": ChainCfg(mode=0, w_stream=True, geometry=plan_geometry()),
    "ks_chip_snake": ChainCfg(mode=1, w_stream=True, geometry=plan_geometry("snake")),
    "ks_chip_snake_xs12": ChainCfg(mode=1, w_stream=True, xs=12, geometry=plan_geometry("snake")),
    "ks_chip_snake_xs16": ChainCfg(mode=1, w_stream=True, xs=16, geometry=plan_geometry("snake")),
    "today_chip_down": ChainCfg(mode=0, w_stream=True, geometry=plan_geometry(), down_load=True),
    "ks_chip_snake_down": ChainCfg(mode=1, w_stream=True, geometry=plan_geometry("snake"), down_load=True),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.timeout(600)
def test_kc_chain(device, name):
    cfg = CASES[name]
    r = run_chain(device, cfg)
    logger.info(
        f"KCHAIN {name}: {r['cores']} cores, {r['nb']} x blocks a sub-block; {cfg.nsb} sub-blocks {r['total_us']:.1f} us, "
        f"first sub-block {r['first_sb_us']:.1f} us, steady {r['steady_sb_us']:.2f} us a sub-block; h max err "
        f"{r['max_err']:.2e} min pcc {r['min_pcc']:.6f}" + (f"; weights {r['dram_gbs']:.0f} GB/s" if "dram_gbs" in r else "") + (f"; y min pcc {r['y_min_pcc']:.6f}" if "y_min_pcc" in r else "")
    )
    assert r["min_pcc"] > 0.999
