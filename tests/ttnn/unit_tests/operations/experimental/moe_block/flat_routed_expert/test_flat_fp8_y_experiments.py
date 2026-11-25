# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP8_Y_PLAN.md M0 / M1 experiments on one Blackhole chip: the GLM cases of test_flat_routed_expert_checks.py with
row-major bf16 y and down_fp32 (the baseline fp8 y replaces), under the host's experiment env:

  TT_FLAT_EXP_DOWN_HALF_SYNC=1    the fp32 down compute in half sync (passes of 3 columns for GLM)
  TT_FLAT_EXP_DOWN_PACK_DUMMY=N   N dummy SFPU sweeps (load / x1.0 / store) over the packer's half a pass

The env is read when the program is built (not in the program hash): one setting a process.
"""

import os

import pytest
import torch
from loguru import logger

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "flat_checks", Path(__file__).resolve().parent / "test_flat_routed_expert_checks.py"
)
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)
from models.common.utility_functions import comp_pcc

GLM = [c for c in chk.CASES if c.id in ("glm53-8x4-5120-uniform", "glm53-4x4-5120-uniform")]
KW = dict(y_row_major=True, down_fp32=True)


def _env():
    return f"half_sync={os.environ.get('TT_FLAT_EXP_DOWN_HALF_SYNC', '0')} dummy={os.environ.get('TT_FLAT_EXP_DOWN_PACK_DUMMY', '0')}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", chk.DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("case", chk._params(GLM))
def test_fp8_exp_accuracy(device, case):
    chk._skip_unless_supported(device)
    md = chk.cases.MODELS[case.model]
    op, args, (x, spans), wsets = chk._setup(case, device)
    y = chk.ttnn.to_torch(op(*args, **KW)).float()
    q = lambda w: chk.ttnn.to_torch(
        chk.ttnn.from_torch(w, dtype=chk.ttnn.bfloat4_b, layout=chk.ttnn.TILE_LAYOUT)
    ).float()  # noqa: E731
    wq = [tuple(q(w) for w in ws) for ws in wsets]
    worst = 1.0
    for e, (o, c) in enumerate(spans):
        if not c:
            continue
        xe = x[o : o + c].to(torch.bfloat16).float()
        wg, wu, wd = wq[e % chk.cases.N_W]
        ref = chk.weights_py.act_ref(xe @ wg, xe @ wu, md["act"]) @ wd
        ok, pcc = comp_pcc(ref, y[o : o + c], chk.PCC)
        worst = min(worst, float(pcc))
        assert ok, f"{case.id}: expert {e}: pcc {pcc}"
    logger.info(f"EXP ACCURACY {case.id} [{_env()}]: worst pcc {worst:.5f}")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", chk.DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("case", chk._params(GLM))
def test_fp8_exp_perf(device, case):
    chk._skip_unless_supported(device)
    op, args, _, _ = chk._setup(case, device)
    chk.ttnn.deallocate(op(*args, **KW))

    def run():
        for _ in range(chk.PERF_RUNS):
            chk.ttnn.deallocate(op(*args, **KW))

    t = chk.op_device_times(device, run, chk.FLAT_OPS)["flat_routed_expert"][-chk.PERF_RUNS :]
    us = chk.median_us(t)
    logger.info(f"EXP PERF {case.id} [{_env()}]: median {us:.1f} us (min {min(t) / 1e3:.1f}, max {max(t) / 1e3:.1f})")
