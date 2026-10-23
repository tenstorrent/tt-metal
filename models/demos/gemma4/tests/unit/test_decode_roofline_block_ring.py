# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Traced decode step of one 8-chip engine on the (2,4) block with the mesh opened on the RING fabric.

Same body as ``test_decode_roofline`` (prefill, traced warm-up, ``GEMMA4_ROOFLINE_STEPS`` timed steps, WALL ms/step
and tok/s), but the mesh is opened with ``FABRIC_1D_RING`` (``ROOFLINE_FABRIC`` overrides the name), which the decode
MoE kernel path (``GEMMA4_MOE_DECODE_KERNEL=1``: all_to_all dispatch along mesh axis 0, moe_compute, reduce-scatter
over the columns) requires; ``parametrize_mesh_with_fabric`` hard-wires FABRIC_1D. The model's own collectives keep
``GEMMA4_CCL_TOPOLOGY`` (linear on the serving entries).

    GEMMA4_MOE_DECODE_KERNEL=1 GEMMA4_ROOFLINE_TRACE=1 GEMMA4_ROOFLINE_STEPS=20 GEMMA4_ROOFLINE_DISTINCT_ROWS=1 \\
    TT_VISIBLE_DEVICES=0,1,4,5,8,9,12,13 MESH_DEVICE=P150x8 HF_MODEL=google/gemma-4-26B-A4B-it \\
    pytest models/demos/gemma4/tests/unit/test_decode_roofline_block_ring.py -k "2x4_ring and batch_32" -s
"""
import os

import pytest

import ttnn
from models.demos.gemma4.tests.test_factory import GEMMA4_TEST_L1_SMALL_SIZE
from models.demos.gemma4.tests.unit.test_decode_roofline import model_path  # noqa: F401  (fixture)
from models.demos.gemma4.tests.unit.test_decode_roofline import test_decode_roofline as _roofline_body

_FABRIC = getattr(ttnn.FabricConfig, os.environ.get("ROOFLINE_FABRIC", "FABRIC_1D_RING"))
MESHES = [
    pytest.param(
        (2, 4),
        {
            "fabric_config": _FABRIC,
            "trace_region_size": 512_000_000,
            "l1_small_size": int(os.environ.get("ROOFLINE_L1_SMALL", str(GEMMA4_TEST_L1_SMALL_SIZE))),
        },
        id="2x4_ring",
    ),
]


@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=True)
@pytest.mark.parametrize("batch_size", [1, 32], ids=["batch_1", "batch_32"])
def test_decode_roofline_block_ring(mesh_device, model_path, batch_size):
    _roofline_body(mesh_device, model_path, batch_size)
