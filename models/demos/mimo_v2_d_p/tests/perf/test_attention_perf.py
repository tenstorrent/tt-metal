# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Attention-layer device perf (run under ``run_safe_pytest.sh --profile``; analyze with analyze_attention.py).

One GA layer (0) and one SWA layer (1), real weights, a single chunk placed at ``kv_actual`` deep in a long
context (cache contents are irrelevant to timing). Each measured iteration is bracketed by signposts
``{tag}_start`` / ``{tag}_end`` with tag = ``{kind}_C{chunk_local}_ctx{kv_actual+chunk}``.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import layer_state
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.attention.attention import TtAttention, cache_v_dim
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mimo_v2_d_p.tt.ccl import CCLManager, default_num_links
from models.demos.mimo_v2_d_p.tt.rope import build_indexed_rope, build_transformation_mat

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

CTX = [int(c) for c in os.environ.get("MIMO_PERF_CTX", "8192,32768").split(",")]


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["GA", "SWA"])
@pytest.mark.parametrize("chunk_local", [int(c) for c in os.environ.get("MIMO_PERF_CHUNK_LOCAL", "640,2048").split(",")])
def test_attention_perf(mesh_device, device_params, layer_idx, chunk_local):
    cfg = MiMoTextConfig.from_json()
    sd = layer_state(layer_idx, cfg, experts=False)
    attn_sd = {k[len("self_attn.") :]: v for k, v in sd.items() if k.startswith("self_attn.")}
    spec = cfg.layer_attn(layer_idx)
    sp, tp = tuple(mesh_device.shape)
    chunk = chunk_local * sp
    max_seq = max((c + chunk - 1) // chunk * chunk for c in CTX)
    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    ccl = CCLManager(mesh_device, num_links=default_num_links(), topology=sp_topo)
    attn = TtAttention(mesh_device, cfg, layer_idx, attn_sd, ccl)
    kv = allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=max_seq, n_kv_local=attn.nkv_l, k_dim=spec.head_dim, v_dim=cache_v_dim(spec))
    rope = build_indexed_rope(mesh_device, spec, max_seq_len=max_seq, chunk_size=chunk)
    trans = build_transformation_mat(mesh_device)
    x = ttnn.from_torch(
        torch.randn(1, 1, chunk, cfg.hidden_size) * 0.5, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, None)),
    )
    kind = "GA" if spec.window is None else "SWA"
    for ctx in CTX:
        kv_actual = (ctx + chunk - 1) // chunk * chunk - chunk
        for it in range(3):  # 1 warmup + 2 measured
            tag = f"{kind}_C{chunk_local}_ctx{kv_actual + chunk}"
            if it:
                signpost(f"{tag}_start")
            out = attn(x, rope, trans, kv, cache_layer=0, kv_actual=kv_actual)
            ttnn.synchronize_device(mesh_device)
            if it:
                signpost(f"{tag}_end")
            out.deallocate(True)
        logger.info(f"ran {tag}")
