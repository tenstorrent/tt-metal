# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A10 polish: is the SPARSE decode step (all_to_all dispatch/combine + mesh_partition + all_gather)
trace-capturable? Capture forward_decode(use_sparse=True) as a trace and replay; the traced logits must
match eager. If the CCL ops can't be traced, this fails/errors and sparse is documented as an untraced
throughput path (the dense decode trace already satisfies the trace criterion). 2-layer, B=8, 1x8."""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2
B = 8


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 50000000}], indirect=True
)
def test_m4_sparse_trace(mesh_device, reset_seeds):
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

    torch.manual_seed(0)
    x = torch.randn(B, 1, cfg.hidden_size) * 0.02
    tt_x = _repl(x, mesh_device)
    tt_c = _repl(torch.ones(B, 1, 1, rope), mesh_device)
    tt_s = _repl(torch.zeros(B, 1, 1, rope), mesh_device)
    tt_pos = ttnn.from_torch(
        torch.zeros(B, dtype=torch.int32), device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    kv = tt.init_kv_caches(B, 64)

    # eager reference
    eager = ttnn.to_torch(
        tt.forward_decode(tt_x, tt_pos, tt_c, tt_s, kv, use_sparse=True, sub_device_id=sdid),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    ).float()[:B]

    # capture + replay
    kv2 = tt.init_kv_caches(B, 64)
    tt.forward_decode(tt_x, tt_pos, tt_c, tt_s, kv2, use_sparse=True, sub_device_id=sdid)  # warmup
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = tt.forward_decode(tt_x, tt_pos, tt_c, tt_s, kv2, use_sparse=True, sub_device_id=sdid)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    traced = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]

    passing, msg = comp_pcc(eager, traced, 0.99)
    logger.info(f"A10 sparse-decode TRACE vs eager (B={B}): {msg}")
    assert passing, f"sparse-decode trace mismatch: {msg}"
