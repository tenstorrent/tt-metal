# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""G4a: the TRELLIS.2 structure decoder (dense conv3d, 16^3 -> 64^3) on one device against the CPU fp32 golden.

The decoder output is thresholded (logits > 0, then a 64^3 -> 32^3 max pool) into the coords the shape stage uses, so
the gate is on occupancy, not only PCC: voxel flips at 64^3 and the IoU of the 32^3 coords against the golden, held
to 1.5x the error of the reference's own bf16 mode on the same input (TRELLIS2_SSDEC_FLOOR=<flips64>,<iou32>;
S09 CPU floor: T.png 3, 1.0; biome logo 28, 0.999522). Without the variable: PCC >= 0.9999 and IoU >= 0.999.

TRELLIS2_RESULTS_DIR, if set, receives a JSON table.
"""

from __future__ import annotations

import os
import time

import pytest
from loguru import logger

from models.experimental.trellis2.reference import ss_decoder_host as host
from models.experimental.trellis2.tests import goldens
from models.experimental.trellis2.tests.test_structure_flow_pcc import FLOOR_RATIO, _save
from models.experimental.trellis2.tt.ss_decoder import TtSSDecoder

PCC = 0.9999
IOU = 0.999

pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


def _thresholds() -> tuple[float, float]:
    """(max flips at 64^3, min IoU at 32^3)."""
    floor = os.environ.get("TRELLIS2_SSDEC_FLOOR")
    if not floor:
        return float("inf"), IOU
    flips, iou = (float(v) for v in floor.split(","))
    return FLOOR_RATIO * max(flips, 1.0), 1 - FLOOR_RATIO * (1 - iou)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_ss_decoder(mesh_device):
    decoder = TtSSDecoder(host.load_state_dict(f"{goldens.weights_dir()}/{host.CKPT}"), mesh_device=mesh_device)
    z, gold, gold_coords = goldens.ss_decoder_call()
    out = decoder(z)  # first call compiles
    start = time.time()
    out = decoder(z)
    seconds = time.time() - start

    flips = ((out > 0) != (gold > 0)).sum().item()
    g = set(map(tuple, gold_coords.tolist()))
    t = set(map(tuple, host.occupancy_coords(out).tolist()))
    max_flips, min_iou = _thresholds()
    result = {
        "pcc": goldens.pcc(out, gold),
        "max_abs_err": (out - gold).abs().max().item(),
        "flips64": flips,
        "voxels32": len(t),
        "golden_voxels32": len(g),
        "iou32": len(g & t) / len(g | t),
        "thresholds": {"pcc": PCC, "flips64": max_flips, "iou32": min_iou},
        "seconds_warm": seconds,
    }
    _save("ss_decoder", result)
    logger.info(
        f"ss decoder: pcc {result['pcc']:.6f}, flips64 {flips}, voxels32 {len(t)} vs {len(g)}, "
        f"iou32 {result['iou32']:.6f}, {seconds:.3f} s warm"
    )
    assert result["pcc"] >= PCC
    assert flips <= max_flips
    assert result["iou32"] >= min_iou
