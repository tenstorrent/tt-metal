# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 attention block (real weights) in chunked SP prefill vs the HF MiMoV2Attention (fp32, one-shot).

Layer 0 = global attention (4 KV heads, DV 128), layer 1 = SWA (8 KV heads, window 128, sink).
"""

import os

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
from models.demos.mimo_v2_d_p.tt.attention.attention import TtAttention, cache_v_dim
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mimo_v2_d_p.tt.ccl import CCLManager, default_num_links
from models.demos.mimo_v2_d_p.tt.rope import build_indexed_rope, build_transformation_mat


@MESH_PARAMS
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["L0-GA", "L1-SWA"])
@pytest.mark.parametrize("n_chunks,chunk", [(3, 2048), (3, 2560)], ids=["3x2k", "3x2.5k-kvpad"])  # 2.5k: KV shard 3840 % 1024 != 0
def test_attention_block(mesh_device, device_params, layer_idx, n_chunks, chunk):
    cfg = MiMoTextConfig.from_json()
    sd = layer_state(layer_idx, cfg, experts=False)
    attn_sd = {k[len("self_attn.") :]: v for k, v in sd.items() if k.startswith("self_attn.")}
    spec = cfg.layer_attn(layer_idx)
    sp, tp, sp_axis, _ = sp_tp(mesh_device)
    C, S = chunk // sp, n_chunks * chunk

    hcfg = hf.hf_config()
    _, mod = hf.hf_modules()
    is_swa = spec.window is not None
    ref = mod.MiMoV2Attention(hcfg, is_swa, layer_idx, projection_layout="fused_qkv")
    ref.load_state_dict({k: v.float() for k, v in attn_sd.items()})
    ref = ref.float().eval()
    torch.manual_seed(0)
    emb = global_state()["embed_tokens.weight"]
    x = emb[torch.randint(0, 150000, (1, S))].float()
    norm = mod.MiMoV2RMSNorm(cfg.hidden_size, cfg.layernorm_epsilon)
    norm.weight.data = sd["input_layernorm.weight"].float()
    with torch.no_grad():
        x = norm(x).bfloat16().float()
        pos = torch.arange(S)[None]
        cos, sin = hf.rotary(is_swa, hcfg)(x, pos)
        ref_out, _ = ref(x, (cos, sin), hf.mask(pos[0], S, spec.window))

    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    ccl = CCLManager(mesh_device, num_links=default_num_links(), topology=sp_topo)
    tt_attn = TtAttention(mesh_device, cfg, layer_idx, attn_sd, ccl)
    kv = allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=S, n_kv_local=tt_attn.nkv_l, k_dim=spec.head_dim, v_dim=cache_v_dim(spec))
    rope = build_indexed_rope(mesh_device, spec, max_seq_len=S, chunk_size=chunk)
    trans = build_transformation_mat(mesh_device)

    got = torch.zeros_like(ref_out)
    for c in range(n_chunks):
        kv_actual = c * chunk
        idx = bc_index(kv_actual, sp, C)
        out = tt_attn(to_mesh_seq(x, mesh_device, idx), rope, trans, kv, cache_layer=0, kv_actual=kv_actual)
        got[0, idx] = from_mesh_seq(out, mesh_device)
        out.deallocate(True)

    ok, pcc = comp_pcc(ref_out, got, 0.99)
    per_chunk = [round(comp_pcc(ref_out[:, i * chunk : (i + 1) * chunk], got[:, i * chunk : (i + 1) * chunk])[1], 5) for i in range(n_chunks)]
    logger.info(f"attention L{layer_idx} ({spec.kind}) mesh={mesh_id(mesh_device)} fid={os.environ.get('MIMO_SDPA_FIDELITY','HiFi2')}: PCC {pcc} per-chunk {per_chunk}")
    assert ok, pcc
