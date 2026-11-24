# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A bare fabric all-to-all on the LoudBox 2 x 4 (a2a.py): the GLM-5.3 combine's traffic (16 experts a chip, M 384:
768 rows of 12320 B from every chip to every other chip) and variants, timed as device time (slowest chip, after a
mesh barrier)."""

import statistics
import sys
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.glm_ks_expert.a2a import A2ACfg, build, router_cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests/ttnn/unit_tests/operations/experimental/moe_block"))
from op_checks import MeshBarrier, op_device_times  # noqa: E402

OPS = {"a2a": "glm_ks_expert/kernels/a2a"}


def dev_params(payload):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": router_cfg(payload), "l1_small_size": 8192}


CASES = {
    "glm_rows_pay12320": (12320, A2ACfg(PB=12320)),
    "glm_rows_pay15232": (15232, A2ACfg(PB=12320)),
    "glm_bfp8_rows_pay6560": (6560, A2ACfg(PB=6560)),
    "glm_rows_1link": (12320, A2ACfg(PB=12320, links=1)),
    "glm_rows_ws": (12320, A2ACfg(PB=12320, placement="ws")),
    "glm_bfp8_rows_ws": (6560, A2ACfg(PB=6560, placement="ws")),
}


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params, name", [(dev_params(p), n) for n, (p, _) in CASES.items()], ids=list(CASES), indirect=["device_params"]
)
def test_a2a(mesh_device, name):
    cfg = CASES[name][1]
    mpd, io, bytes_out, ns = build(mesh_device, cfg)
    barrier = MeshBarrier(mesh_device)
    try:
        ttnn.generic_op(io, mpd)  # (warm-up: program build)
        ttnn.synchronize_device(mesh_device)

        def run():
            for _ in range(5):
                barrier()
                ttnn.generic_op(io, mpd)

        t = op_device_times(mesh_device, run, OPS)["a2a"]
    finally:
        barrier.close()
    us = statistics.median(t) / 1e3
    mb = bytes_out[(0, 0)] / 1e6
    logger.info(
        f"A2A {name}: {us:.1f} us (runs {[round(x / 1e3) for x in t]}), {mb:.1f} MB out a chip = {mb * 1e3 / us:.1f} "
        f"GB/s a chip, senders a chip {sorted(set(ns.values()))}, router payload {ttnn.get_tt_fabric_max_payload_size_bytes()}"
    )
