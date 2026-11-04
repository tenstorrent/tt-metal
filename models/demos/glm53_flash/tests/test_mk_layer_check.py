# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Numerics of the persistent-stack port against our HF layer reference: ONE layer's decode step at position 0 (fresh
state) on one chip (the full layer: vtp = 1, no fabric) from the HF golden's input residual, PCC of the output
residual against the golden (reference/hf_layer_reference.py --layers 0-4 --seq-len 1 --num-decode 0 --stream).

    GLM53_MK_GOLDEN=<dir> GLM53_MK_LAYER=0 pytest .../test_mk_layer_check.py -s   # dense KDA layer
    GLM53_MK_GOLDEN=<dir> GLM53_MK_LAYER=4 ...                                    # sparse KDA layer (288 experts)
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_KDA, load_hf_text_config
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoderConfig
from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import layer_kind, set_knobs
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import TPDenseKDALayer
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.test_mk_layer_probe import RawHFSource
from models.demos.glm53_flash.tt.mesh import mesh_device_params

LAYER = int(os.environ.get("GLM53_MK_LAYER", "0"))
GOLDEN = os.environ.get("GLM53_MK_GOLDEN", "/home/ttuser/ashai/glm53/_logs/mesh/hf_golden_t1")
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
VTP = int(os.environ.get("GLM53_MK_VTP", "0")) or MESH[0] * MESH[1]  # the TP the shards are sized for (1 chip: 1)
STEPS = int(os.environ.get("GLM53_MK_STEPS", "1"))  # golden steps: the first token, then decode0 .. (state carry)
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "256"))
WORKER_L1 = int(os.environ.get("GLM53_MK_WORKER_L1", "1264512"))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20, worker_l1_size=WORKER_L1)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_layer_check(mesh_device):
    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    ocfg = OptimizedDecoderConfig()
    kind = layer_kind(cfg, LAYER)
    golds = [torch.load(os.path.join(GOLDEN, f"layer{LAYER}_prefill.pt"))] + [
        torch.load(os.path.join(GOLDEN, f"layer{LAYER}_decode{k}.pt")) for k in range(STEPS - 1)
    ]
    gold = golds[0]
    x_in, x_out = gold["x_in"].float(), gold["x_out"].float()  # [1, T, 4, H]
    assert x_in.shape[1] == 1, f"golden T = {x_in.shape[1]}: this check decodes one token at position 0"
    H = cfg.hidden_size
    if os.environ.get("GLM53_MK_KNOBS", "1") == "1":
        set_knobs(kind == SPARSE_KDA, VTP, 1)
    source = RawHFSource(snapshot, [LAYER])
    t0 = time.perf_counter()
    if kind == DENSE_KDA:
        layer = TPDenseKDALayer(mesh_device, cfg, LAYER, source, ocfg, b=1, probe=False, vtp=VTP)
    elif kind == SPARSE_KDA:
        from models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse import TPSparseKDALayer, build_moe

        moe = build_moe(mesh_device, cfg, source, LAYER, ocfg)
        layer = TPSparseKDALayer(mesh_device, cfg, LAYER, source, ocfg, b=1, moe=moe, probe=False, vtp=VTP)
    else:
        pytest.skip(f"layer {LAYER} is {kind}: checked through the backend")
    logger.info(f"{kind} layer {LAYER} built in {time.perf_counter() - t0:.0f} s (vtp {VTP})")
    state = layer.attn.allocate_state(1)
    active = layer.masks(torch.ones(1))
    worst = 1.0
    for step, gold in enumerate(golds):  # the same state object through the steps (the KDA conv history / state)
        x_in, x_out = gold["x_in"].float(), gold["x_out"].float()
        # the residual row of the token: the 4 streams side by side [1, 1, 1, 4 H] fp32
        x = layer.replicated(x_in[0, 0].reshape(1, 1, 1, 4 * H).contiguous(), ttnn.float32)
        s1, out, details, _ = layer(x, state.C, active, state.S, t=1)
        ttnn.synchronize_device(mesh_device)
        o = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(4, H)
        g = x_out[0, 0].reshape(4, H)
        p = pcc(o, g)
        per_stream = [round(pcc(o[s_], g[s_]), 5) for s_ in range(4)]
        delta = pcc(o - x_in[0, 0].reshape(4, H), g - x_in[0, 0].reshape(4, H))
        logger.info(
            f"MK_LAYER_CHECK: {kind} layer {LAYER} mesh {MESH} vtp {VTP} step {step}: pcc {p:.6f} per stream "
            f"{per_stream}; delta (out - in) pcc {delta:.6f}; golden x_out std {x_out.std():.4f}"
        )
        worst = min(worst, p)
    logger.info(
        f"MK_LAYER_CHECK_SUMMARY: {kind} layer {LAYER} mesh {MESH} vtp {VTP} steps {len(golds)}: worst pcc {worst:.6f}"
    )
    assert worst > 0.98, f"layer {LAYER} output pcc {worst}"
