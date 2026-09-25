# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full decoder-layer device perf (attention + FFN), real weights, input = real-token embeddings (realistic
routing). Signposts ``L{idx}_{kind}_C{chunk_local}_ctx{ctx}``; analyze with analyze_tags.py --ops."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.reference import hf
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.attention.attention import cache_v_dim
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mimo_v2_d_p.tt.ccl import CCLManager, default_num_links
from models.demos.mimo_v2_d_p.tt.decoder import TtDecoderLayer
from models.demos.mimo_v2_d_p.tt.rope import build_indexed_rope, build_transformation_mat

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

CTX = int(os.environ.get("MIMO_PERF_CTX", "32768"))


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("layer_idx", [int(x) for x in os.environ.get("MIMO_PERF_LAYERS", "0,1,5").split(",")])
@pytest.mark.parametrize("chunk_local", [int(c) for c in os.environ.get("MIMO_PERF_CHUNK_LOCAL", "640,2048").split(",")])
def test_layer_perf(mesh_device, device_params, layer_idx, chunk_local):
    cfg = MiMoTextConfig.from_json()
    spec = cfg.layer_attn(layer_idx)
    sp, tp = tuple(mesh_device.shape)
    chunk = chunk_local * sp
    max_seq = (CTX + chunk - 1) // chunk * chunk
    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    ccl = CCLManager(mesh_device, num_links=default_num_links(), topology=sp_topo)
    layer = TtDecoderLayer(mesh_device, cfg, layer_idx, layer_state(layer_idx, cfg), ccl=ccl, sp_topology=sp_topo, seq_len_per_chip=chunk_local,
                           num_links=default_num_links())
    kv = allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=max_seq, n_kv_local=layer.attn.nkv_l, k_dim=spec.head_dim, v_dim=cache_v_dim(spec))
    rope = build_indexed_rope(mesh_device, spec, max_seq_len=max_seq, chunk_size=chunk)
    trans = build_transformation_mat(mesh_device)
    ids = hf.tokenize_prompt(chunk)
    x_host = global_state()["embed_tokens.weight"][ids].float()[None, None]
    kv_actual = max_seq - chunk
    kind = "GA" if spec.window is None else "SWA"
    tag = f"L{layer_idx}_{kind}_C{chunk_local}_ctx{max_seq}"
    for it in range(3):
        x = ttnn.from_torch(x_host, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, None)))
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_start")
        out = layer(x, rope, trans, kv, cache_layer=0, kv_actual=kv_actual)
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_end")
        out.deallocate(True)
        x.deallocate(True)
    logger.info(f"ran {tag}")
