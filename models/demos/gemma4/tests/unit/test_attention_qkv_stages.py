# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-wise diagnostic of the fused QKV decode path (custom_ops/qkv_norm_rope) inside the attention module:
on the same projected QKV, the composite path (DRAM per-head norms + RoPE composite) and the fused program are
compared tensor by tensor (q, k, v: PCC per column tile), then the decode SDPA is run with both q on the same
filled paged cache, and both k / v are written into two caches through paged_update_cache and the written rows
compared. Pins down which stage diverges when the end-to-end attention test fails for the fused path."""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.custom_ops.qkv_norm_rope import QkvNormRope
from models.demos.gemma4.tests.unit.test_attention import _kv_fill_to_tt, _skip_if_l1_overflow
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.decode import _gamma_tiles, _sdpa_max_cores_per_head_kwargs
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.attention.operations import (
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    apply_rope_decode_peruser,
    effective_block_size,
    split_qkv_heads_decode,
)
from models.tt_transformers.tt.common import PagedAttentionConfig

from ...tests.test_factory import TestFactory, mesh_tp_and_axis, parametrize_mesh_with_fabric


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _dev0(t, mesh_device):
    if hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return ttnn.to_torch(t).float()


def _report(name, ref, got, heads):
    cols = ref.shape[-1] // 32
    col_pcc = [round(_pcc(ref[..., c * 32 : (c + 1) * 32], got[..., c * 32 : (c + 1) * 32]), 4) for c in range(cols)]
    head_pcc = [round(_pcc(ref[:, :, h], got[:, :, h]), 4) for h in range(heads)]
    p = _pcc(ref, got)
    print(
        f"[stage {name}] pcc {p:.6f} max|d| {(ref - got).abs().max():.4f} per column tile {col_pcc} per head {head_pcc}"
    )
    return p


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="diagnostic; not a CI gate")
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("batch", [1, 32], ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("cache_len", [512, 1500], ids=lambda c: f"cache{c}")
def test_qkv_fused_stages(layer_idx, batch, cache_len, mesh_device, reset_seeds, request):
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    _skip_if_l1_overflow(config, mesh_device)
    tp, tp_axis = mesh_tp_and_axis(mesh_device)
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    is_mesh = num_devices > 1
    rep = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp), tp_axis=tp_axis)
    state_dict = {k: v.clone() for k, v in hf_layer.self_attn.state_dict().items() if not k.startswith("v_norm")}
    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )
    weights = tt_attn.weights
    eps = config.rms_norm_eps
    positions = [cache_len - b for b in range(batch)]
    nkv_local = 1 if weights.kv_replicated else config.num_key_value_heads // tp
    nq_local = config.num_attention_heads // tp

    # the projected QKV in L1 (GEMMA4_QKV_L1_OUT=1 form), split twice: the fused program consumes its shards in place
    x = (torch.randn(1, 1, batch, config.hidden_size) * 0.5).to(torch.bfloat16)
    x_tt = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep)
    xqkv = apply_qkv_projection(x_tt, weights, memory_config=ttnn.L1_MEMORY_CONFIG)
    cos2d, sin2d = TestFactory.create_tt_rope_cache_2d(mesh_device, hf_text_config, cache_len + 64, layer_idx)
    pos_padded = torch.zeros(1, 32, dtype=torch.int32)
    pos_padded[0, :batch] = torch.tensor(positions, dtype=torch.int32)
    position_idx = ttnn.from_torch(
        pos_padded, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=rep
    )
    cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos2d, layout=ttnn.TILE_LAYOUT))
    sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin2d, layout=ttnn.TILE_LAYOUT))

    # --- composite path (tt/attention/decode.py, GEMMA4_HEAD_NORM_4D=1) ---
    os.environ["GEMMA4_HEAD_NORM_4D"] = "1"
    q, k, v = split_qkv_heads_decode(xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated)
    q_sharded_mem = q.memory_config()
    qc = apply_per_head_norm(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), weights.q_norm_weight, eps)
    kc = apply_per_head_norm(ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG), weights.k_norm_weight, eps)
    vc = apply_per_head_norm(ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG), None, eps, with_scale=False)
    if batch == 1:
        qc = apply_rope(qc, cos_pos, sin_pos, token_index=0)
        kc = apply_rope(kc, cos_pos, sin_pos, token_index=0)
    else:
        cos_b = ttnn.transpose(cos_pos, 1, 2)[:, :batch, :, :]
        sin_b = ttnn.transpose(sin_pos, 1, 2)[:, :batch, :, :]
        qc = apply_rope_decode_peruser(qc, cos_b, sin_b)
        kc = apply_rope_decode_peruser(kc, cos_b, sin_b)
    kc_sh = ttnn.to_memory_config(kc, q_sharded_mem)
    vc_sh = ttnn.to_memory_config(vc, q_sharded_mem)

    # --- fused path ---
    q2, k2, v2 = split_qkv_heads_decode(xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated)
    gq, gk = _gamma_tiles(weights, config.head_dim)
    nq, nk, nv = QkvNormRope.op(q2, k2, v2, gq, gk, cos_pos, sin_pos, eps)
    # the shards the two paths consumed are dead: free their L1 like the model does (at 32 users on the head_dim-512
    # layers the decode SDPA program's static buffers need the room)
    for t in (q, k, v, q2, k2, v2):
        t.deallocate(True)

    # stage 1: the three head tensors
    pq = _report(
        "q", _dev0(qc, mesh_device)[:, :batch, :nq_local], _dev0(nq, mesh_device)[:, :batch, :nq_local], nq_local
    )
    pk = _report(
        "k", _dev0(kc, mesh_device)[:, :batch, :nkv_local], _dev0(nk, mesh_device)[:, :batch, :nkv_local], nkv_local
    )
    pv = _report(
        "v", _dev0(vc, mesh_device)[:, :batch, :nkv_local], _dev0(nv, mesh_device)[:, :batch, :nkv_local], nkv_local
    )

    # stage 2: the decode SDPA on the same filled paged cache with the composite q (DRAM) and the fused q (sharded)
    block_size = 64
    blocks_per_user = (cache_len + block_size) // block_size + 1
    paged = PagedAttentionConfig(block_size=block_size, max_num_blocks=blocks_per_user * batch)
    page_table = torch.zeros(batch, blocks_per_user * batch, dtype=torch.int32)
    for b in range(batch):
        page_table[b, :blocks_per_user] = torch.arange(
            b * blocks_per_user, (b + 1) * blocks_per_user, dtype=torch.int32
        )
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=rep
    )
    cache_pos = ttnn.from_torch(
        torch.tensor(positions, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=rep,
    )
    caches = []
    for _ in range(2):
        kcache, vcache = init_kv_cache(
            mesh_device=mesh_device, config=config, paged_attention_config=paged, cache_dtype=ttnn.bfloat16
        )
        for b in range(batch):
            kd = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
            vd = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
            torch.manual_seed(1000 + b)
            kf = _kv_fill_to_tt(
                kd,
                mesh_device,
                num_kv_heads=config.num_key_value_heads,
                num_attention_heads=config.num_attention_heads,
                tp=tp,
                num_devices=num_devices,
            )
            vf = _kv_fill_to_tt(
                vd,
                mesh_device,
                num_kv_heads=config.num_key_value_heads,
                num_attention_heads=config.num_attention_heads,
                tp=tp,
                num_devices=num_devices,
            )
            ttnn.experimental.paged_fill_cache(kcache, kf, page_table_tt, batch_idx=b)
            ttnn.experimental.paged_fill_cache(vcache, vf, page_table_tt, batch_idx=b)
        caches.append((kcache, vcache))
    # the same fill for both caches: re-seed per user above makes the two fills identical
    eff_bs = effective_block_size(caches[0][0], config.head_dim, nkv_local)
    # stage 3: cache writes (composite k/v into cache 0, fused k/v into cache 1), then the written rows
    for (kcache, vcache), (kk, vv) in zip(caches, ((kc_sh, vc_sh), (nk, nv))):
        ttnn.experimental.paged_update_cache(
            kcache,
            kk,
            update_idxs_tensor=cache_pos,
            page_table=page_table_tt,
            block_size=eff_bs,
            num_kv_heads=nkv_local,
        )
        ttnn.experimental.paged_update_cache(
            vcache,
            vv,
            update_idxs_tensor=cache_pos,
            page_table=page_table_tt,
            block_size=eff_bs,
            num_kv_heads=nkv_local,
        )
    for t in (kc_sh, vc_sh, nk, nv):  # consumed by the cache update
        t.deallocate(True)
    kcs = [_dev0(c[0], mesh_device) for c in caches]
    vcs = [_dev0(c[1], mesh_device) for c in caches]
    rows_ref, rows_got = [], []
    for b in range(batch):
        blk = page_table[b, positions[b] // block_size].item()
        off = positions[b] % block_size
        rows_ref.append(kcs[0][blk, :, off, :])
        rows_got.append(kcs[1][blk, :, off, :])
    rows_ref = torch.stack(rows_ref)
    rows_got = torch.stack(rows_got)
    pkc = _pcc(rows_ref, rows_got)
    vrows_ref = torch.stack(
        [
            vcs[0][page_table[b, positions[b] // block_size].item(), :, positions[b] % block_size, :]
            for b in range(batch)
        ]
    )
    vrows_got = torch.stack(
        [
            vcs[1][page_table[b, positions[b] // block_size].item(), :, positions[b] % block_size, :]
            for b in range(batch)
        ]
    )
    pvc = _pcc(vrows_ref, vrows_got)
    print(
        f"[stage k/v-cache rows] k pcc {pkc:.6f} max|d| {(rows_ref - rows_got).abs().max():.4f}; v pcc {pvc:.6f} "
        f"max|d| {(vrows_ref - vrows_got).abs().max():.4f} (the two caches hold different random fills elsewhere)"
    )
    pkc = min(pkc, pvc)
    # stage 2 proper: SDPA with each q on cache 0 (composite k/v written)
    device_grid = mesh_device.compute_with_storage_grid_size()
    sdpa_grid = (
        ttnn.CoreCoord(8, 4)
        if (config.head_dim >= 512 and batch == 1)
        else ttnn.CoreCoord(device_grid.x, device_grid.y)
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_grid,
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        **_sdpa_max_cores_per_head_kwargs(),
    )
    outs = []
    nq_dram = ttnn.to_memory_config(nq, ttnn.DRAM_MEMORY_CONFIG)  # the form the fused branch now hands SDPA
    for qq in (qc, nq, nq_dram):
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qq,
            caches[0][0],
            caches[0][1],
            cur_pos_tensor=cache_pos,
            page_table_tensor=page_table_tt,
            scale=1.0,
            sliding_window_size=config.sliding_window if config.is_sliding else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            paged_cache_geometry=ttnn.PagedCacheGeometryOverride(block_size=eff_bs, num_kv_heads=nkv_local),
        )
        outs.append(_dev0(o, mesh_device)[:, :batch, :nq_local])
    ps_sh = _report("sdpa(q composite vs q fused SHARDED, same cache)", outs[0], outs[1], nq_local)
    ps = _report("sdpa(q composite vs q fused -> DRAM, same cache)", outs[0], outs[2], nq_local)
    print(
        f"STAGES layer={layer_idx} batch={batch} cache={cache_len} tp={tp}: q {pq:.5f} k {pk:.5f} v {pv:.5f} "
        f"kcache_rows {pkc:.5f} sdpa_sharded_q {ps_sh:.5f} sdpa_dram_q {ps:.5f}"
    )
    assert min(pq, pk, pv) > 0.999, "the fused q/k/v differ from the composite's"
    assert pkc > 0.999, "the fused k / v rows written to the cache differ from the composite's"
    # SDPA over a random-filled cache is chaotic (logit std ~16: a bf16 rounding difference in q moves the output
    # by 0.1-1 %), so composite-q vs fused-q is only bounded loosely; sharded q and its DRAM copy must agree.
    assert abs(ps - ps_sh) < 1e-4, "SDPA on the sharded fused q differs from the same q in DRAM"
    assert min(ps, ps_sh) > 0.98, "SDPA on the fused q is far from the composite q"
