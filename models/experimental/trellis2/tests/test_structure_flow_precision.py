# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Diagnostic, not a gate: single-pass PCC of a few structure flow calls under each numerics switch, to find where
the TT error over the CPU bf16 floor comes from. Runs only with TRELLIS2_PRECISION_PROBE=1."""

from __future__ import annotations

import os

import pytest
from loguru import logger

from models.experimental.trellis2.reference import structure_flow_host as host
from models.experimental.trellis2.tests import goldens
from models.experimental.trellis2.tests.test_structure_flow_pcc import _save
from models.experimental.trellis2.tt.structure_flow import Precision, TtStructureFlow

CALLS = [0, 1, 20, 21]
_FAST = dict(sdpa_hifi4_fp32_acc=False, linear_hifi4=False, fp32_residual=False)
VARIANTS = {
    "base": Precision(**_FAST),
    "sdpa": Precision(**{**_FAST, "sdpa_hifi4_fp32_acc": True}),
    "linear": Precision(**{**_FAST, "linear_hifi4": True}),
    "residual": Precision(**{**_FAST, "fp32_residual": True}),
    "all": Precision(),
}

pytestmark = pytest.mark.skipif(
    goldens.missing_reason() is not None or os.environ.get("TRELLIS2_PRECISION_PROBE") != "1",
    reason="set TRELLIS2_PRECISION_PROBE=1, TRELLIS2_GOLDEN_DIR and TRELLIS2_WEIGHTS",
)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_structure_flow_precision(mesh_device, variant):
    model = TtStructureFlow(mesh_device=mesh_device, precision=VARIANTS[variant])
    model.load_torch_state_dict(host.load_state_dict(f"{goldens.weights_dir()}/{goldens.SS_CKPT}"))
    rows = []
    for i in CALLS:
        x, t, cond, gold, branch = goldens.call_inputs(i)
        p = goldens.pcc(model(x, t, cond), gold)
        rows.append({"call": i, "branch": branch, "pcc": p})
        logger.info(f"variant {variant} call {i:2d} ({branch}) pcc={p:.6f}")
    _save(f"precision_{variant}", {"variant": variant, "precision": vars(VARIANTS[variant]), "rows": rows})
