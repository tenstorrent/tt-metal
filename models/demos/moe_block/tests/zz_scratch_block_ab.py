# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: one stage at a time ops vs fused (head / router placement / tail) for a model at a row count on the 2x4 -
one eager call each, PCC against the host model on the device's routing (localizes a block-level PCC loss)."""

import pytest
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

from models.demos.moe_block.moe_block import KIMI_K3, MoEBlock, MoEOptions, WeightCache
from models.demos.moe_block.tests.test_moe_block import TRACE_REGION, C, R, _fab
from models.demos.moe_block.tests.test_moe_block_matrix import mesh_model, run_checked, scenario_weights

ROWS = 768
VARIANTS = {
    "auto": MoEOptions(),
    "tail_ops": MoEOptions(tail="ops"),
    "head_ops": MoEOptions(head="ops"),
    "router_engine": MoEOptions(router="engine"),
    "both_ops": MoEOptions(head="ops", tail="ops"),
}


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (R, C),
            _fab(
                fabric_payload_size=fabric_payload(KIMI_K3.routed_width),
                l1_small_size=0,
                trace_region_size=TRACE_REGION,
            ),
            id="k3",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_block_ab(mesh_device, device_params, variant):
    m = mesh_model("k3", R * C)
    w = scenario_weights(m, "uniform", R * C)
    try:
        block = MoEBlock(mesh_device, m, ROWS * R, w, WeightCache(path=None, layer=0), options=VARIANTS[variant])
    except (AssertionError, ValueError, RuntimeError) as e:
        pytest.skip(f"{variant}: does not build ({str(e)[:200]})")
    try:
        stats = run_checked(mesh_device, block, m, w, f"ab {variant}", eager=2, replays=0, nx=2)
        logger.info(f"AB {variant}: PCC {min(stats['pcc']):.5f}; {block.summary}")
    except AssertionError as e:
        logger.info(f"AB {variant}: FAIL {str(e)[:120]}; {block.summary}")
        raise
    finally:
        block.close()
