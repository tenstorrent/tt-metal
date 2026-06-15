# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A10 polish: decode wall-clock dense vs SPARSE MoE at B=8/32 (2-layer). Sparse computes only the top-4
routed experts/token (all_to_all dispatch/combine) vs all 16 local experts dense — quantifies the
throughput trade (dispatch/combine overhead vs expert-compute saving). Relative per-step timing at a
fixed layer count; not a full-depth TTFT number."""
import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.mistral4.tests.m4_text_reference import load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2
STEPS = 10


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_sparse_perf(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    rope = cfg.qk_rope_head_dim
    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sdm = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(sdm)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    sdid = ttnn.SubDeviceId(0)

    for B in (8, 32):
        torch.manual_seed(0)
        hd = _repl(torch.randn(B, 1, cfg.hidden_size) * 0.02, mesh_device)
        cd = _repl(torch.ones(B, 1, 1, rope), mesh_device)
        sd_ = _repl(torch.zeros(B, 1, 1, rope), mesh_device)
        pos = ttnn.from_torch(
            torch.zeros(B, dtype=torch.int32), device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
        )

        def run(sparse):
            kv = tt.init_kv_caches(B, 64)
            o = tt.forward_decode(hd, pos, cd, sd_, kv, use_sparse=sparse, sub_device_id=sdid if sparse else None)
            ttnn.synchronize_device(mesh_device)
            return o

        for sparse in (False, True):
            run(sparse)  # warmup/compile
            t0 = time.perf_counter()
            for _ in range(STEPS):
                run(sparse)
            dt = (time.perf_counter() - t0) / STEPS
            tag = "SPARSE" if sparse else "dense "
            logger.info(f"A10 perf B={B:2d} {tag} MoE: {dt*1000:7.1f} ms/step ({B/dt:7.1f} tok/s agg, {N_LAYERS}L)")
