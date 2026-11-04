# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The port's in-program fabric all-reduce (tt/fabric_ar.py FabricAllReduce, "cb" mode: the lane CBs, the 2-round
schedule over mesh axis 0 then 1) checked on the whole mesh: every chip's output = the sum of all chips' partials.
The first device check of the generalized round 1 (N1 = 8 on the 8x4: slot q at creation position q).

    GLM53_MESH_SHAPE=8,4 GLM53_FABRIC=1d GLM53_MK_B=4 pytest .../test_mk_fabric_ar.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_ar as FAR
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
B = int(os.environ.get("GLM53_MK_B", "4"))
ITERS = int(os.environ.get("GLM53_MK_ITERS", "3"))
WORKER_L1 = int(os.environ.get("GLM53_MK_WORKER_L1", "1264512"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, worker_l1_size=WORKER_L1)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_fabric_ar(mesh_device):
    n = MESH[0] * MESH[1]
    far = FAR.FabricAllReduce(mesh_device, B, mode="cb", iters=ITERS)
    torch.manual_seed(0)
    worst = 1.0
    for call in range(2):  # two launches: the persistent counters carry over
        x = torch.randn(n, 1, B, 4096) * (1.0 + torch.arange(n).reshape(n, 1, 1, 1).float() / n)
        xt = far.input_from_torch(x)
        y = far.output()
        far(xt, y)
        ttnn.synchronize_device(mesh_device)
        ref = x.to(torch.bfloat16).float().sum(0)  # [1, B, 4096]
        got = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()  # [n, 1, B, 4096]
        per_chip = []
        for d in range(n):
            o = got[d, 0, :B].reshape(-1)
            r = ref[0, :B].reshape(-1)
            p = float(torch.corrcoef(torch.stack([o.nan_to_num(0, 1e9, -1e9), r]))[0, 1])
            p = p if p == p else 0.0  # NaN / inf outputs count as a miss
            err = float((o.nan_to_num(0, 1e9, -1e9) - r).abs().max() / (r.abs().max() + 1e-6))
            bad_vals = int((~torch.isfinite(o)).sum())
            if bad_vals:
                err = float("inf")
            per_chip.append((round(p, 6), round(err, 4)))
            worst = min(worst, p)
        bad = [(d, v) for d, v in enumerate(per_chip) if v[0] < 0.9999]
        logger.info(
            f"MK_FABRIC_AR: mesh {MESH} B {B} iters {ITERS} call {call}: worst pcc {min(v[0] for v in per_chip):.6f}, "
            f"max rel err {max(v[1] for v in per_chip):.4f}; chips below 0.9999: {bad[:8]}"
        )
        ttnn.deallocate(xt)
        ttnn.deallocate(y)
    far.release()
    assert worst > 0.9999, f"all-reduce pcc {worst}"
