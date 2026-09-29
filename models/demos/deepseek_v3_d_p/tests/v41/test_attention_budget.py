# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""V4.1 attention error budget (dev-spec D-I): attention-only chain over the prototype layers, per variant.

Runs TtV41Attention for prototype layers 0..4 in execution order on the oracle's attention inputs and
reports PCC against the oracle's attention outputs for weight dtype x matmul fidelity variants.
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41 import prototype_oracle as po
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.attention import TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.host_fallback import HostCompressedAttention
from tests.ttnn.utils_for_testing import comp_pcc

VARIANTS = {
    "bfp8-default": (ttnn.bfloat8_b, None),
    "bfp8-hifi4": (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4),
    "bf16-hifi4": (ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
}


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)
def test_v41_attention_budget(mesh_device, device_params, variant):
    cfg, args = po.PrototypeScheduleConfig, po.model_args()
    reference = po.build_reference(args)
    rec = po.oracle(reference, args)
    host = HostCompressedAttention(reference)
    topology = per_axis_topology(device_params["fabric_config"])[1]
    shape = tuple(mesh_device.shape)
    wdtype, fidelity = VARIANTS[variant]
    ckc = None
    if fidelity is not None:
        ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=fidelity, fp32_dest_acc_en=True, packer_l1_acc=True
        )
    out = {}
    for layer in range(cfg.NUM_LAYERS):
        w = po.device_weights(reference, layer)["attn"]
        attn = TtV41Attention(
            mesh_device, cfg, layer, w, po.SEQ, host, topology=topology, weights_dtype=wdtype, compute_kernel_config=ckc
        )
        x = ttnn.from_torch(
            rec[f"block{layer}.attn_in"][None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
        )
        y = ttnn.to_torch(attn(x), mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))[0, 0]
        out[layer] = comp_pcc(rec[f"block{layer}.attn_out"].float(), y.float(), 0.0)[1]
        logger.info(f"{variant} layer {layer}: attention PCC {out[layer]:.5f}")
    report = os.environ.get("TT_V41_BUDGET_REPORT")
    if report:
        with open(f"{report}.{variant}.json", "w") as f:
            json.dump(out, f)
    assert all(torch.isfinite(torch.tensor(list(out.values()))))
