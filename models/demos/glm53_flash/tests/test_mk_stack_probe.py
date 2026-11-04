# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0, step 3 of the persistent-stack port: N dense KDA layers as ONE persistent program (the reference's MkStack
on one chip / TPStack on a TP mesh: the layer loop inside the kernels, per-layer argument table, the weight stream of
layer l + 1 starting under layer l's tail), built from OUR HF weights and timed.

    GLM53_MESH_SHAPE=1,1 GLM53_MK_LAYERS=0,1,2 GLM53_MK_VTP=8 pytest .../test_mk_stack_probe.py -s

Prints the traced time per launch and per layer, and every layer's per-phase probe marks.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, load_hf_text_config
from models.autoports.zai_org_glm_5_3_flash.tt.mk_stack import MkStack
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoderConfig
from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import layer_kind
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import TPDenseKDALayer
from models.autoports.zai_org_glm_5_3_flash.tt.tp_stack import TPStack
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.test_mk_layer_probe import RawHFSource
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
LAYERS = [int(v) for v in os.environ.get("GLM53_MK_LAYERS", "0,1,2").split(",")]
VTP = int(os.environ.get("GLM53_MK_VTP", "8"))
B = int(os.environ.get("GLM53_MK_B", "1"))
T = int(os.environ.get("GLM53_MK_T", "1"))
N_TRACE = int(os.environ.get("GLM53_MK_N", "4"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_stack_probe(mesh_device):
    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    ocfg = OptimizedDecoderConfig()
    kinds = {L: layer_kind(cfg, L) for L in LAYERS}
    assert all(k == DENSE_KDA for k in kinds.values()), f"this probe stacks dense KDA layers only: {kinds}"
    source = RawHFSource(snapshot, LAYERS)
    t0 = time.perf_counter()
    tps = []
    for L in LAYERS:
        tps.append(
            TPDenseKDALayer(
                mesh_device, cfg, L, source, ocfg, b=B, probe=False, vtp=VTP, fabric=tps[0].fabric if tps else None
            )
        )
    logger.info(f"{len(tps)} dense KDA layers built in {time.perf_counter() - t0:.1f} s: mesh {MESH}, vtp {VTP}")
    one_chip = MESH[0] * MESH[1] == 1
    mks = [tp.mk for tp in tps]
    stack = (
        MkStack(mesh_device, mks, t=T, probe=True) if one_chip else TPStack(mesh_device, mks, tps[0], t=T, probe=True)
    )
    states = [tp.attn.allocate_state(B) for tp in tps]
    active = tps[0].masks(torch.ones(B))
    hists = [s.C for s in states]
    ss = [s.S for s in states]
    res = stack.residual()
    torch.manual_seed(0)
    x = tps[0].replicated(
        (torch.randn(*tuple(res.shape)) * 0.1).float(), ttnn.float32, memory_config=res.memory_config()
    )
    ttnn.copy(x, res)
    t0 = time.perf_counter()
    stack(hists, active, ss)  # builds the stack program (eager launch)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"first stack launch (build + run) {time.perf_counter() - t0:.1f} s")
    for L in range(len(tps)):
        _, marks = stack.probe_marks(L)
        logger.info(f"MK_STACK_PROBE marks layer {LAYERS[L]}: {marks}")
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(N_TRACE):
        stack(hists, active, ss)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    per_launch = (time.perf_counter() - t0) / (reps * N_TRACE) * 1e6
    ttnn.release_trace(mesh_device, tid)
    logger.info(
        f"MK_STACK_PROBE: mesh {MESH} layers {LAYERS} vtp {VTP} t {T}: traced {per_launch:.1f} us per launch = "
        f"{per_launch / len(tps):.1f} us per layer"
    )
    assert per_launch > 0
