# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-STAGE cost for a pipeline-parallel prefill: one chip owning whole layers.

PP's appeal is that a stage owns a layer outright, so the intra-layer all-gather /
reduce-scatter disappear -- and those are ~48% of the critical path at SP=4 x TP=8. Pipeline
latency is (M + S - 1) * t_stage with t_stage = T_layer_full / M, so it floors at T_layer_full:
the time for ONE chip to run ONE layer over the whole sequence. That single number decides
whether PP is worth building, so measure it before writing any pipeline.

Differences two layer counts of the SAME type, which cancels embedding + final norm + LM head.

    QWEN_PP_TYPE=gdn|fa   QWEN_PP_ISL=4096
"""

import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

GDN_LAYERS = [0, 1, 2, 4, 5, 6]
FA_LAYERS = [3, 7, 11, 15]
BLOCK = 64


def _time_layers(sub, idxs, T, iters=3):
    args, model, _sd = create_tt_model(
        sub,
        max_batch_size=1,
        max_seq_len=T + 256,
        hf_model=model_path(),
        sequence_parallel=True,
        layer_indices=idxs,
    )
    nb = (T // BLOCK) + 8
    model.allocate_kv_caches((nb, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat16, batch_size=1)
    tokens = torch.randint(0, args.vocab_size, (1, T), dtype=torch.long)
    model.reset_tp()
    model.prefill_tp(tokens, valid_len=T)  # warmup
    ttnn.synchronize_device(sub)
    best = float("inf")
    for _ in range(iters):
        model.reset_tp()
        t0 = time.time()
        model.prefill_tp(tokens, valid_len=T)
        ttnn.synchronize_device(sub)
        best = min(best, (time.time() - t0) * 1e3)
    return best


def test_pp_stage_cost():
    kind = os.environ.get("QWEN_PP_TYPE", "gdn")
    T = int(os.environ.get("QWEN_PP_ISL", "4096"))
    pool = GDN_LAYERS if kind == "gdn" else FA_LAYERS
    lo, hi = (1, 3) if kind == "fa" else (2, 5)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    system = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    parent = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(4, 8) if system.mesh_size() == 32 else system,
        trace_region_size=64 * 1024 * 1024,
        l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
    )
    try:
        sub = parent.create_submeshes(ttnn.MeshShape(1, 1))[0]
        res = {}
        for n in (lo, hi):
            res[n] = _time_layers(sub, pool[:n], T)
            logger.info(f"[pp-stage] {kind} {n} layer(s), ISL {T}: {res[n]:.2f} ms")
        per = (res[hi] - res[lo]) / (hi - lo)
        logger.info(f"[pp-stage] RESULT one {kind.upper()} layer, whole {T}-token sequence, ONE chip: {per:.2f} ms")
        logger.info(
            f"[pp-stage] PP floor with 24 such stages -> {per:.1f} ms; at M=48 microbatches "
            f"-> {(48 + 23) * per / 48:.1f} ms"
        )
    finally:
        for s in parent.get_submeshes():
            ttnn.close_mesh_device(s)
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
