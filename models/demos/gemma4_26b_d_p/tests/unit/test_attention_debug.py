# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug: per-op intermediates of the attention stem (qkv -> heads -> norms -> rope) vs torch, 1 chip."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_d_p.reference.blocks import Attention, RMSNorm
from models.demos.gemma4_26b_d_p.reference.config import Gemma4TextConfig
from models.demos.gemma4_26b_d_p.reference.weights import CheckpointReader
from models.demos.gemma4_26b_d_p.tt.attention.attention import TtAttention
from models.demos.gemma4_26b_d_p.tt.rope import build_indexed_rope, build_transformation_mat, meta_cos_sin
from models.tt_transformers.tt.load_checkpoints import reverse_permute


@pytest.mark.parametrize("mesh_device, device_params", [pytest.param((1, 1), {}, id="1x1")], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5])
def test_attention_stem(mesh_device, device_params, layer_idx):
    cfg = Gemma4TextConfig.from_json()
    r = CheckpointReader()
    sd = r.substate(f"layers.{layer_idx}.self_attn")
    S = 1024
    lt = cfg.layer_types[layer_idx]
    spec = cfg.rope_spec(lt)
    n_q, n_kv, hd = 16, cfg.layer_kv_heads(layer_idx), cfg.layer_head_dim(layer_idx)
    torch.manual_seed(0)
    norm = RMSNorm(cfg.hidden_size)
    norm.weight.data = r.get(f"layers.{layer_idx}.input_layernorm.weight").float()
    x = norm(torch.randn(1, S, cfg.hidden_size)).bfloat16().float()
    ref = Attention(cfg, layer_idx).float()
    ref.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)

    # torch "meta-space" goldens
    perm = lambda t, n: t.view(1, n, S, hd // 2 * 0 + hd)  # noqa
    with torch.no_grad():
        q = ref.q_proj(x).view(1, S, n_q, hd).transpose(1, 2)
        kr = ref.k_proj(x).view(1, S, n_kv, hd).transpose(1, 2)
        vr = kr if ref.v_proj is None else ref.v_proj(x).view(1, S, n_kv, hd).transpose(1, 2)
        idx = torch.arange(hd).view(2, hd // 2).T.reshape(-1)  # meta order: [0, D/2, 1, D/2+1, ...]
        qn, kn, vn = ref.q_norm(q), ref.k_norm(kr), ref.v_norm(vr)
        cos, sin = meta_cos_sin(S, spec)
        def rope_meta(t):
            t = t[..., idx]
            x1, x2 = t[..., 0::2], t[..., 1::2]
            rot = torch.stack([-x2, x1], -1).flatten(-2)
            return t * cos + rot * sin
        g = {"q_raw": q[..., idx], "k_raw": kr[..., idx], "v": vr, "q_norm": qn[..., idx], "k_norm": kn[..., idx], "v_norm": vn, "q_rope": rope_meta(qn), "k_rope": rope_meta(kn)}

    tt = TtAttention(mesh_device, cfg, layer_idx, sd, None, None)
    rope = build_indexed_rope(mesh_device, spec, max_seq_len=S, chunk_size=S)
    trans = build_transformation_mat(mesh_device)
    tx = ttnn.from_torch(x[None], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    xqkv = ttnn.linear(tx, tt.w.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=tt.compute_cfg)
    tq, tk, tv = ttnn.experimental.nlp_create_qkv_heads(xqkv, num_heads=tt.nq_l, num_kv_heads=tt.nkv_l, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tqn = ttnn.rms_norm(tq, epsilon=1e-6, weight=tt.w.q_norm)
    tkn = ttnn.rms_norm(tk, epsilon=1e-6, weight=tt.w.k_norm)
    tvn = ttnn.rms_norm(tv, epsilon=1e-6)
    tqr = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(tqn, rope[0], rope[1], trans, kv_actual_global=0, cluster_axis=0)
    tkr = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(tkn, rope[0], rope[1], trans, kv_actual_global=0, cluster_axis=0)
    t2 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    got = {"q_raw": tq, "k_raw": tk, "v": tv, "q_norm": tqn, "k_norm": tkn, "v_norm": tvn, "q_rope": tqr, "k_rope": tkr}
    for name, t in got.items():
        a = t2(t)
        logger.info(f"L{layer_idx} {name}: shape {list(a.shape)} vs {list(g[name].shape)} PCC {comp_pcc(g[name], a)[1]}")
