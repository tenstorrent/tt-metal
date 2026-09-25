# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One full MiMo-V2 decoder layer (attention + FFN, real weights), chunked SP prefill vs HF MiMoV2DecoderLayer.

L0 = GA + dense MLP, L1 = SWA + MoE, L5 = GA + MoE. Input = real-token embeddings (the true layer-0 input).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.reference import hf
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.common import bc_index, from_mesh_seq, to_mesh_seq
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id, sp_tp
from models.demos.mimo_v2_d_p.tt.attention.attention import cache_v_dim
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mimo_v2_d_p.tt.ccl import CCLManager, default_num_links
from models.demos.mimo_v2_d_p.tt.decoder import TtDecoderLayer
from models.demos.mimo_v2_d_p.tt.rope import build_indexed_rope, build_transformation_mat


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("layer_idx", [0, 1, 5], ids=["L0-GA-dense", "L1-SWA-moe", "L5-GA-moe"])
@pytest.mark.parametrize("n_chunks,chunk", [(2, 2048)], ids=["2x2k"])
def test_decoder_layer(mesh_device, device_params, layer_idx, n_chunks, chunk):
    cfg = MiMoTextConfig.from_json()
    t_sd = time.perf_counter()
    sd = layer_state(layer_idx, cfg)
    logger.info(f"HOST_DEQUANT L{layer_idx}: {time.perf_counter() - t_sd:.2f}s")
    spec = cfg.layer_attn(layer_idx)
    sp, tp, _, _ = sp_tp(mesh_device)
    C, S = chunk // sp, n_chunks * chunk
    is_swa = spec.window is not None

    torch.manual_seed(0)
    x = global_state()["embed_tokens.weight"][torch.randint(0, 150000, (1, S))].float()
    ref_layer = hf.decoder_layer(layer_idx, sd, dtype=torch.float32)
    ref_out = hf.run_layer(ref_layer, x, is_swa, window=spec.window)
    del ref_layer

    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    ccl = CCLManager(mesh_device, num_links=default_num_links(), topology=sp_topo)
    _orig, _acc = ttnn.from_torch, [0.0, 0, 0]

    def _timed(t, *a, **k):
        t1 = time.perf_counter()
        r = _orig(t, *a, **k)
        if k.get("device") is not None:
            ttnn.synchronize_device(mesh_device)
            _acc[0] += time.perf_counter() - t1
            _acc[1] += 1
            _acc[2] += t.numel() * t.element_size()
        return r

    ttnn.from_torch = _timed
    t0 = time.perf_counter()
    layer = TtDecoderLayer(mesh_device, cfg, layer_idx, sd, ccl=ccl, sp_topology=sp_topo, seq_len_per_chip=C)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"WEIGHT_LOAD L{layer_idx}: {time.perf_counter() - t0:.2f}s (pin={os.environ.get('TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES')} cache={os.environ.get('MIMO_TTNN_CACHE')})")
    ttnn.from_torch = _orig
    logger.info(f"FROM_TORCH L{layer_idx}: {_acc[0]:.2f}s over {_acc[1]} tensors, {_acc[2] / 1e9:.2f} GB torch input")
    del sd
    kv = allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=S, n_kv_local=layer.attn.nkv_l, k_dim=spec.head_dim, v_dim=cache_v_dim(spec))
    rope = build_indexed_rope(mesh_device, spec, max_seq_len=S, chunk_size=chunk)
    trans = build_transformation_mat(mesh_device)

    got = torch.zeros_like(ref_out)
    for c in range(n_chunks):
        kv_actual = c * chunk
        idx = bc_index(kv_actual, sp, C)
        out = layer(to_mesh_seq(x, mesh_device, idx), rope, trans, kv, cache_layer=0, kv_actual=kv_actual)
        got[0, idx] = from_mesh_seq(out, mesh_device)
        out.deallocate(True)

    ok, pcc = comp_pcc(ref_out, got, 0.99)
    d_ok, d_pcc = comp_pcc(ref_out - x, got - x, 0.98)  # the layer's own contribution (residual removed)
    logger.info(f"decoder L{layer_idx} ({spec.kind}, moe={cfg.is_moe(layer_idx)}) mesh={mesh_id(mesh_device)}: PCC {pcc} delta-PCC {d_pcc}")
    assert ok and d_ok, (pcc, d_pcc)
