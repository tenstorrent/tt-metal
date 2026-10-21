# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""G3: DINOv3 ViT-L/16 (TRELLIS.2's condition model) on one device against the CPU fp32 golden cond.

Input: the golden's preprocessed image through the extractor's own preprocessing (512 px). Gate: cond error
(1 - PCC) at most 1.5x the CPU bf16 error of the same input, TRELLIS2_DINO_FLOOR=<pcc> (biome logo S03: 0.9999772);
without it, PCC >= 0.9999.

TRELLIS2_RESULTS_DIR, if set, receives a JSON table.
"""

from __future__ import annotations

import os
import time

import pytest
from loguru import logger
from PIL import Image

from models.experimental.trellis2.reference import dinov3_host as host
from models.experimental.trellis2.tests import goldens
from models.experimental.trellis2.tests.test_structure_flow_pcc import FLOOR_RATIO, _save, precision_from_env
from models.experimental.trellis2.tt.dinov3 import TtDinoV3

PCC = 0.9999

pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


def _threshold() -> float:
    floor = os.environ.get("TRELLIS2_DINO_FLOOR")
    return 1 - FLOOR_RATIO * (1 - float(floor)) if floor else PCC


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dinov3_cond(mesh_device):
    model = TtDinoV3(mesh_device=mesh_device, precision=precision_from_env())
    model.load_torch_state_dict(host.load_state_dict(f"{goldens.weights_dir()}/{host.WEIGHTS}"))
    pixels = host.preprocess(Image.open(f"{goldens.golden_dir()}/preprocessed.png"), 512)
    gold = goldens.conditions()["cond"]
    out = model(pixels)  # first call compiles
    start = time.time()
    out = model(pixels)
    seconds = time.time() - start
    result = {
        "pcc": goldens.pcc(out, gold),
        "max_abs_err": (out - gold).abs().max().item(),
        "threshold": _threshold(),
        "tokens": out.shape[1],
        "seconds_warm": seconds,
    }
    _save("dinov3_cond", result)
    logger.info(f"dinov3 cond: pcc {result['pcc']:.6f}, max abs err {result['max_abs_err']:.3e}, {seconds:.3f} s warm")
    assert out.shape == gold.shape
    assert result["pcc"] >= result["threshold"]
