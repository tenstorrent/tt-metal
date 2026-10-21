# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""G5: the TRELLIS.2 SC-VAE decoders on one device against the CPU fp32 golden decode (DESIGN.md D08).

- shape decoder, free-running: per level, the device `to_subdiv` logits vs the golden subs (PCC and sign flips; a flip
  changes the topology, and even an fp32 summation reorder flips about one near-zero logit per input, S09);
- shape decoder on the golden topology: output vs the CPU mirror on the same topology (PCC >= 0.999);
- texture decoder on the golden topology: output * 0.5 + 0.5 vs the golden tex voxels (PCC >= 0.999).
Levels after the first one with a flip are reported as diverged: a flip changes which voxels exist, so equal voxel
counts no longer mean the same voxels. Gate: TRELLIS2_SCVAE_FLOOR=<level-0 flips>,<tex PCC> holds level-0 flips and
the texture error (1 - PCC) to 1.5x the reference's bf16 mode on the same input (S10 CPU floor, torch 2.11: T.png 1,
0.9999800; biome logo 1, 0.9999884); without it, the provisional thresholds below.

TRELLIS2_RESULTS_DIR, if set, receives a JSON table.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

from models.experimental.trellis2.reference import scvae_host as host
from models.experimental.trellis2.tests import goldens
from models.experimental.trellis2.tests.test_structure_flow_pcc import FLOOR_RATIO, _save, precision_from_env
from models.experimental.trellis2.tt.scvae import TtSCVAEDecoder

PCC = 0.999
MAX_FLIP_FRACTION = 1e-3


def _thresholds(golden_voxels0: int) -> tuple[float, float]:
    """(max level-0 flips, min tex PCC)."""
    floor = os.environ.get("TRELLIS2_SCVAE_FLOOR")
    if not floor:
        return MAX_FLIP_FRACTION * golden_voxels0 * 8, PCC
    flips, tex = (float(v) for v in floor.split(","))
    return FLOOR_RATIO * max(flips, 1.0), 1 - FLOOR_RATIO * (1 - tex)


pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


def _golden():
    gdir = goldens.golden_dir()
    load = lambda name: torch.load(f"{gdir}/{name}.pt", map_location="cpu", weights_only=False)
    return load("shape")["slat"], load("tex")["slat"], load("decode")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_scvae_decoders(mesh_device):
    slat, tex_slat, dec = _golden()
    guide = [g["feats"] > 0 for g in dec["subs"]]
    result = {"levels": []}

    shape_state = host.load_state_dict(f"{goldens.weights_dir()}/{host.SHAPE_CKPT}")
    shape = TtSCVAEDecoder(shape_state, mesh_device=mesh_device, precision=precision_from_env())
    start = time.time()
    _, coords, subs = shape(slat["feats"], slat["coords"])
    result["shape_free_seconds"] = time.time() - start
    result["shape_free_voxels"] = coords.shape[0]
    diverged = False
    for i, (m, g) in enumerate(zip(subs, dec["subs"])):
        row = {"level": i, "voxels": m.shape[0], "golden_voxels": g["feats"].shape[0]}
        if diverged or m.shape != g["feats"].shape:
            row["diverged"] = True
        else:
            row.update(pcc=goldens.pcc(m, g["feats"]), flips=int(((m > 0) != (g["feats"] > 0)).sum()))
            diverged = row["flips"] > 0
        result["levels"].append(row)
        logger.info(f"shape level {i}: {row}")

    out, _, _ = shape(slat["feats"], slat["coords"], guide_keep=guide)
    with torch.no_grad():
        ref, _, _ = host.functional_forward(shape_state, slat["feats"], slat["coords"], guide_keep=guide)
    result["shape_guided_pcc"] = goldens.pcc(out, ref)

    tex = TtSCVAEDecoder(host.load_state_dict(f"{goldens.weights_dir()}/{host.TEX_CKPT}"), mesh_device=mesh_device)
    start = time.time()
    tout, tcoords, _ = tex(tex_slat["feats"], tex_slat["coords"], guide_keep=guide)
    result["tex_seconds"] = time.time() - start
    result["tex_pcc"] = goldens.pcc(tout * 0.5 + 0.5, dec["tex_voxels"]["feats"])
    result["tex_coords_identical"] = bool(torch.equal(tcoords, dec["tex_voxels"]["coords"]))
    max_flips, min_tex = _thresholds(result["levels"][0]["golden_voxels"])
    result["thresholds"] = {"level0_flips": max_flips, "tex_pcc": min_tex, "shape_guided_pcc": PCC}
    _save("scvae_decoders", result)
    logger.info(f"scvae: {result}")

    assert result["shape_guided_pcc"] >= PCC
    assert result["tex_pcc"] >= min_tex
    assert result["tex_coords_identical"]
    assert result["levels"][0]["flips"] <= max_flips
