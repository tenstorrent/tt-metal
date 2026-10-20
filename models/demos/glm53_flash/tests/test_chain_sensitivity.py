# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only control for the multi-layer chain PCC: the HF bf16 chain's own sensitivity to a small
relative perturbation of the layer-0 input (the size of a device bf16/bfp8 error), per layer. Compare
``bringup/references/glm53_flash/L45_S128`` with ``L45_S128_perturb1e-3`` (produced by
``reference/hf_layer_reference.py --perturb 1e-3``).

    pytest models/demos/glm53_flash/tests/test_chain_sensitivity.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.tests.utils import pcc

BASE = Path(os.environ.get("GLM53_CHAIN_BASE", "bringup/references/glm53_flash/L45_S128"))
PERT = Path(os.environ.get("GLM53_CHAIN_PERT", "bringup/references/glm53_flash/L45_S128_perturb1e-3"))


@pytest.mark.skipif(not (PERT / "layer44_prefill.pt").exists(), reason="perturbed dump not produced yet")
def test_hf_chain_sensitivity():
    rows = []
    for i in range(45):
        a = torch.load(BASE / f"layer{i}_prefill.pt")["x_out"][0]
        b = torch.load(PERT / f"layer{i}_prefill.pt")["x_out"][0]
        p_all = pcc(a, b)
        p_rows = [pcc(a[t], b[t]) for t in range(a.shape[0])]
        da = torch.load(BASE / f"layer{i}_decode1.pt")["x_out"][0]
        db = torch.load(PERT / f"layer{i}_decode1.pt")["x_out"][0]
        rows.append((i, p_all, min(p_rows), p_rows[-1], pcc(da, db)))
        logger.info(
            f"HF chain sensitivity layer {i:2d}: prefill rows pcc {p_all:.6f} (per-row min {min(p_rows):.4f}, "
            f"last row {p_rows[-1]:.4f}); decode step 1 row pcc {pcc(da, db):.4f}"
        )
    x0a = torch.load(BASE / "prefill_input.pt")["x0"]
    x0b = torch.load(PERT / "prefill_input.pt")["x0"]
    logger.info(
        f"layer-0 input perturbation: pcc {pcc(x0a, x0b):.6f}, rel rms {((x0a.float() - x0b.float()).norm() / x0a.float().norm()).item():.2e}"
    )
