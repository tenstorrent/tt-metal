# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A6 op-only isolation: feed the paged flash-MLA op a torch-built compressed cache (all positions)
+ torch-absorbed q (last position) and check it reproduces the golden MLA output. Isolates the op
contract from forward_decode_mla's wiring (cache write + absorption matmuls), since the compressed
decode is PCC 0.03 while the cache-write round-trips (1.0) and the absorption math is CPU-1.0."""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_text_reference

B = 32


def _rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_flash_mla_correctness(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    H, nope, rope, vd, kvl = (
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        cfg.v_head_dim,
        cfg.kv_lora_rank,
    )
    d = kvl + rope
    model, _, _ = load_m4_text_reference(ckpt, n_layers=1)
    model = model.float()
    sa = model.model.layers[0].self_attn
    eps = cfg.rms_norm_eps

    torch.manual_seed(0)
    S = 32
    x = torch.randn(1, S, cfg.hidden_size) * 0.02
    cosS, sinS = model.model.rotary_emb(x, torch.arange(S)[None])
    cosS, sinS = cosS[0].float(), sinS[0].float()  # [S, rope]

    def rope_apply(t, cos, sin):  # t [H,S,rope] interleaved
        t1, t2 = t[..., 0::2], t[..., 1::2]
        td = torch.cat([t1, t2], dim=-1)
        rh = torch.cat([-td[..., rope // 2 :], td[..., : rope // 2]], dim=-1)
        return td * cos + rh * sin

    with torch.no_grad():
        q = (
            sa.q_b_proj(_rms(sa.q_a_proj(x), sa.q_a_layernorm.weight.float(), eps))
            .view(1, S, H, nope + rope)
            .transpose(1, 2)
        )
        q_nope, q_rope = q[0, :, :, :nope], rope_apply(q[0, :, :, nope:], cosS, sinS)  # [H,S,nope],[H,S,rope]
        kv_a = sa.kv_a_proj_with_mqa(x)[0]  # [S, kvl+rope]
        kv_pass = _rms(kv_a[:, :kvl], sa.kv_a_layernorm.weight.float(), eps)  # [S,kvl]
        k_rot = rope_apply(kv_a[:, kvl:][None].expand(1, S, rope), cosS, sinS)[0]  # [S,rope]
        kvb = sa.kv_b_proj.weight.float().view(H, nope + vd, kvl)
        Wk, Wv = kvb[:, :nope, :], kvb[:, nope:, :]  # [H,nope,kvl],[H,vd,kvl]
        # golden (standard) last-position MLA output
        k_nope = torch.einsum("sk,hdk->hsd", kv_pass, Wk)
        kfull = torch.cat([k_nope, k_rot[None].expand(H, S, rope)], dim=-1)
        v = torch.einsum("sk,hdk->hsd", kv_pass, Wv)
        qx = torch.cat([q_nope, q_rope], dim=-1)
        sc = (qx[:, -1:, :] @ kfull.transpose(1, 2)) * ((nope + rope) ** -0.5)  # [H,1,S]
        golden = (torch.softmax(sc, -1) @ v)[:, 0, :]  # [H,vd]
        # absorbed q (last pos) + latent cache
        q_lat = torch.cat([torch.einsum("hn,hnk->hk", q_nope[:, -1], Wk), q_rope[:, -1]], dim=-1)  # [H,kvl+rope]
        latent = torch.cat([kv_pass, k_rot], dim=-1)  # [S, kvl+rope]

    # cache [B,1,32,d] (1 block/user, positions 0..S-1), page_table identity
    block = 32
    cache_t = torch.zeros(B, 1, block, d, dtype=torch.bfloat16)
    cache_t[:, 0, :S, :] = latent.to(torch.bfloat16)[None]
    tt_cache = ttnn.from_torch(
        cache_t, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tt_pt = ttnn.from_torch(
        torch.arange(B, dtype=torch.int32).reshape(B, 1),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_pos = ttnn.from_torch(
        torch.full((B,), S - 1, dtype=torch.int32),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    qlat_t = q_lat[None, None].expand(B, 1, H, d).reshape(1, B, H, d).to(torch.bfloat16)  # [1,B,H,d]
    cores = ttnn.num_cores_to_corerangeset(B * H // 32, mesh_device.compute_with_storage_grid_size(), row_wise=True)
    qmem = ttnn.create_sharded_memory_config(
        shape=[32, d],
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_q = ttnn.to_memory_config(
        ttnn.from_torch(
            qlat_t, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
        ),
        qmem,
    )
    omem = ttnn.create_sharded_memory_config(
        shape=[32, kvl],
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    attn = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        tt_q,
        tt_cache,
        page_table_tensor=tt_pt,
        cur_pos_tensor=tt_pos,
        head_dim_v=kvl,
        scale=(nope + rope) ** -0.5,
        program_config=prog,
        compute_kernel_config=ck,
        memory_config=omem,
    )  # [1,B,H,kvl]
    ctx = ttnn.to_torch(
        ttnn.to_memory_config(attn, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[
        0, 0
    ].float()  # [H,kvl] user0
    out = torch.einsum("hk,hdk->hd", ctx, Wv)  # [H,vd]
    passing, msg = comp_pcc(golden, out, 0.99)
    logger.info(f"flash-MLA op-only correctness (vs golden last-pos): {msg}")
    assert passing, f"op-only flash-MLA PCC below 0.99: {msg}"
