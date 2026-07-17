# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP validation for Qwen3.5/3.6 full attention on a Blackhole mesh.

One file per component (decode / prefill / paged-KV), all sharing the loaders and
mesh parametrization from ``test_factory``:

* ``test_attention_tp``         — decode PCC @ pos0 (attention over a single key
  reduces to V, so an exact torch reference covers the q/gate split, V proj, GQA
  mapping, sigmoid gate, output proj, sharding, reduce-scatter) + a second decode
  step for shape/NaN.
* ``test_attention_tp_prefill`` — causal GQA prefill over a short sequence (head
  layout transpose/reshape, partial RoPE, causal SDPA, gate, row-parallel output).
* ``test_attention_tp_paged``   — paged-KV path (vLLM contract) must match the
  concat-KV oracle at both the prefill and decode steps.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py -v -s
"""
import os

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    get_pcc_threshold,
    load_attn_layer,
    model_path,
    parametrize_mesh_only,
    parametrize_mesh_tp,
    replicate_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode, rot_mats_prefill
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


def _rope_torch(x, rope_dim, theta):  # x: [S, H, HD]
    S = x.shape[0]
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    emb = torch.cat([torch.outer(torch.arange(S).float(), inv)] * 2, dim=-1)  # [S, rope_dim]
    cos = emb.cos()[:, None, :]
    sin = emb.sin()[:, None, :]
    xr, xp = x[..., :rope_dim], x[..., rope_dim:]
    r1, r2 = xr[..., : rope_dim // 2], xr[..., rope_dim // 2 :]
    xrot = torch.cat([-r2, r1], dim=-1)
    return torch.cat([xr * cos + xrot * sin, xp], dim=-1)


@torch.no_grad()
@parametrize_mesh_tp()
def test_attention_tp(mesh_device, reset_seeds, ensure_gc, request):
    os.environ.setdefault("HF_MODEL", model_path())
    B = 32
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    logger.info(f"devices={nd} full-attn layer={li} NH={args.n_local_heads} NKV={args.n_local_kv_heads}")

    # args.CKPT_DIR is the resolved local snapshot dir (Qwen36ModelArgs downloads the hub id).
    sd = load_attn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_attention_weights_tp(mesh_device, sd, args)
    attn = TPAttention(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
    x_tt = replicate_to_device(mesh_device, x)
    cur = torch.zeros(B, dtype=torch.int32)
    cur_tt = ttnn.from_torch(
        cur, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos, sin = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, cur)

    out = attn.forward_decode(x_tt, cur_tt, cos, sin)
    out_t = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].float()
    assert out_t.shape[-1] == args.dim, out_t.shape
    assert not torch.isnan(out_t).any() and out_t.abs().max() > 0

    # torch reference @ pos0: attn_out[h] = V[h // group]; out = o_proj(gated)
    NH, NKV, HD = args.n_heads, args.n_kv_heads, args.head_dim
    grp = NH // NKV
    xf = x[0, 0].float()  # [B, dim]
    qg = (xf @ sd["q_proj.weight"].float().T).reshape(B, NH, 2 * HD)
    gate = qg[:, :, HD:]
    vv = (xf @ sd["v_proj.weight"].float().T).reshape(B, NKV, HD)
    attn_ref = vv[:, torch.arange(NH) // grp, :]  # [B, NH, HD]
    gated = attn_ref * torch.sigmoid(gate)
    ref = gated.reshape(B, NH * HD) @ sd["o_proj.weight"].float().T  # [B, dim]

    passing, pcc = comp_pcc(ref, out_t, get_pcc_threshold(request))
    logger.info(f"ATTENTION TP PCC (pos0) = {pcc}")
    assert passing, f"attention TP PCC too low: {pcc}"

    # second decode step @ pos1: real 2-key attention; shape/NaN only
    cur1 = torch.ones(B, dtype=torch.int32)
    cur1_tt = ttnn.from_torch(
        cur1, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos1, sin1 = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, cur1)
    x2 = replicate_to_device(mesh_device, torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16))
    out2 = attn.forward_decode(x2, cur1_tt, cos1, sin1)
    out2_t = ttnn.to_torch(out2, mesh_composer=tp_composer(mesh_device))
    assert not torch.isnan(out2_t).any() and out2_t.abs().max() > 0
    logger.info("PASSED: attention TP decode (pos0 PCC + pos1 shape/NaN)")


@torch.no_grad()
@parametrize_mesh_tp()
def test_attention_tp_prefill(mesh_device, reset_seeds, ensure_gc, request):
    os.environ.setdefault("HF_MODEL", model_path())
    S = 64
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    logger.info(f"devices={nd} full-attn layer={li} S={S}")

    sd = load_attn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_attention_weights_tp(mesh_device, sd, args)
    attn = TPAttention(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, S, args.dim, dtype=torch.bfloat16)
    x_tt = replicate_to_device(mesh_device, x)
    cos, sin = rot_mats_prefill(mesh_device, args.rope_head_dim, S, args.rope_theta)
    out = attn.forward_prefill(x_tt, cos, sin)
    out_t = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].float()

    # ---- torch reference: causal GQA attention ----
    NH, NKV, HD = args.n_heads, args.n_kv_heads, args.head_dim
    grp, rd, scale = NH // NKV, args.rope_head_dim, HD**-0.5
    xf = x[0, 0].float()
    qg = (xf @ sd["q_proj.weight"].float().T).reshape(S, NH, 2 * HD)
    q, gate = qg[..., :HD], qg[..., HD:]
    k = (xf @ sd["k_proj.weight"].float().T).reshape(S, NKV, HD)
    v = (xf @ sd["v_proj.weight"].float().T).reshape(S, NKV, HD)
    qn = sd["q_norm.weight"].float() + 1.0
    kn = sd["k_norm.weight"].float() + 1.0
    q = q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * qn
    k = k / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * kn
    q = _rope_torch(q, rd, args.rope_theta)
    k = _rope_torch(k, rd, args.rope_theta)
    k = k[:, torch.arange(NH) // grp, :]  # expand to NH
    v = v[:, torch.arange(NH) // grp, :]
    # scores [NH,S,S]
    qh, kh, vh = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    cmask = torch.full((S, S), float("-inf")).triu(1)
    attn_w = torch.softmax(scores + cmask, dim=-1)
    ao = torch.matmul(attn_w, vh).permute(1, 0, 2)  # [S,NH,HD]
    gated = ao * torch.sigmoid(gate)
    ref = gated.reshape(S, NH * HD) @ sd["o_proj.weight"].float().T  # [S, dim]

    passing, pcc = comp_pcc(ref, out_t, get_pcc_threshold(request))
    logger.info(f"ATTENTION TP PREFILL PCC (S={S}) = {pcc}")
    assert passing, f"attention TP prefill PCC too low: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_attention_tp_paged(mesh_device, reset_seeds, ensure_gc, request):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    NKV, HD = args.n_local_kv_heads, args.head_dim
    block_size, S, num_blocks = 64, 64, 4
    logger.info(f"devices={nd} layer={li} NKV_local={NKV} HD={HD} S={S} num_blocks={num_blocks}")

    sd = load_attn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_attention_weights_tp(mesh_device, sd, args)

    def to_dev(t):
        return replicate_to_device(mesh_device, t)

    def rm_pt(rows):
        return ttnn.from_torch(
            torch.tensor(rows, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )

    def mk_cache():
        return ttnn.from_torch(
            torch.zeros(num_blocks, NKV, block_size, HD, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    xp = torch.randn(1, 1, S, args.dim, dtype=torch.bfloat16)
    xd = torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16)
    cos_p, sin_p = rot_mats_prefill(mesh_device, args.rope_head_dim, S, args.rope_theta)
    cos_d, sin_d = rot_mats_decode(
        mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, torch.tensor([S], dtype=torch.int32)
    )
    cur_tt = ttnn.from_torch(
        torch.tensor([S], dtype=torch.int32),
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    comp = tp_composer(mesh_device)

    # ---- concat reference (the oracle) ----
    a_ref = TPAttention(mesh_device, args, tw, tt_ccl)
    a_ref.reset_state()
    pre_ref = ttnn.to_torch(a_ref.forward_prefill(to_dev(xp), cos_p, sin_p), mesh_composer=comp).float()
    dec_ref = ttnn.to_torch(a_ref.forward_decode(to_dev(xd), cur_tt, cos_d, sin_d), mesh_composer=comp).float()

    # ---- paged path ----
    a_pag = TPAttention(mesh_device, args, tw, tt_ccl)
    a_pag.set_paged_kv_cache(mk_cache(), mk_cache())
    pre_pag = ttnn.to_torch(
        a_pag.forward_prefill_paged(
            to_dev(xp), cos_p, sin_p, rm_pt([[0]]), chunk_page_table=rm_pt([[0]]), chunk_start_idx=0
        ),
        mesh_composer=comp,
    ).float()
    dec_pag = ttnn.to_torch(
        a_pag.forward_decode(to_dev(xd), cur_tt, cos_d, sin_d, page_table=rm_pt([list(range(num_blocks))])),
        mesh_composer=comp,
    ).float()

    thr = get_pcc_threshold(request)
    ok_p, pcc_p = comp_pcc(pre_ref, pre_pag, thr)
    ok_d, pcc_d = comp_pcc(dec_ref, dec_pag, thr)
    logger.info(f"PREFILL paged-vs-concat PCC = {pcc_p}")
    logger.info(f"DECODE  paged-vs-concat PCC = {pcc_d}")
    assert ok_p, f"prefill paged PCC too low: {pcc_p}"
    assert ok_d, f"decode paged PCC too low: {pcc_d}"
    logger.info("PASSED: TPAttention paged path matches concat path (B=1)")


@torch.no_grad()
@parametrize_mesh_only()
def test_attention_tp_qknorm_offset(mesh_device):
    """Regression: load_attention_weights_tp must add +1 to q_norm/k_norm (the 64k-retrieval fix).

    HF Qwen3_5RMSNorm computes output*(1+weight) and the checkpoints store the raw zero-centered
    weights (means ~0.32-0.58), so the TP loader must add +1 to q_norm/k_norm. Without it, Q·K
    logits are ~14x too small, long-context attention goes UNIFORM, and 64k retrieval collapses.
    Builds a tiny synthetic state_dict and checks the loaded q_norm/k_norm equal raw weights + 1.
    """
    from models.demos.blackhole.qwen36.tt.attention.tp import load_attention_weights_tp

    nd = mesh_device.get_num_devices()
    HD = 128
    NH = 8 * nd  # arbitrary; sharded by dim=-1 across the mesh
    NKV = nd
    DIM = 1024

    torch.manual_seed(0)
    # Distinctive q_norm/k_norm values centered well away from 0 (like the FP8 ckpt's ~0.75),
    # so a stray +1 would be unmistakable.
    q_norm_w = torch.full((HD,), 0.75) + 0.01 * torch.randn(HD)
    k_norm_w = torch.full((HD,), 0.60) + 0.01 * torch.randn(HD)
    state_dict = {
        "q_proj.weight": torch.randn(DIM, NH * HD * 2) * 0.02,  # fused [Q,gate], column-parallel
        "k_proj.weight": torch.randn(DIM, NKV * HD) * 0.02,
        "v_proj.weight": torch.randn(DIM, NKV * HD) * 0.02,
        "o_proj.weight": torch.randn(NH * HD, DIM) * 0.02,
        "q_norm.weight": q_norm_w,
        "k_norm.weight": k_norm_w,
    }

    class _Args:
        n_local_heads = NH // nd
        n_local_kv_heads = max(1, NKV // nd)
        head_dim = HD
        rope_head_dim = 64
        max_batch_size = 1
        max_seq_len = 128

        def ccl_topology(self):
            return ttnn.Topology.Linear

    tw = load_attention_weights_tp(mesh_device, state_dict, _Args(), cache_dir=None)

    # Gather a single replica and compare against the RAW input (no +1).
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if nd > 1 else None
    q_loaded = ttnn.to_torch(tw["q_norm"], mesh_composer=comp).float().reshape(-1)[:HD]
    k_loaded = ttnn.to_torch(tw["k_norm"], mesh_composer=comp).float().reshape(-1)[:HD]

    q_err_raw = (q_loaded - q_norm_w).abs().max().item()
    q_err_plus1 = (q_loaded - (q_norm_w + 1.0)).abs().max().item()
    k_err_raw = (k_loaded - k_norm_w).abs().max().item()
    logger.info(f"q_norm: |loaded-raw|={q_err_raw:.4f}  |loaded-(raw+1)|={q_err_plus1:.4f}")
    logger.info(f"k_norm: |loaded-raw|={k_err_raw:.4f}")

    # bf16 round-trip tolerance ~0.01; a stray +1 would be ~1.0 off.
    assert (
        q_err_raw > 0.5
    ), f"q_norm must load WITH +1 (uniform-attention fix), but |loaded-raw|={q_err_raw:.4f} (regressed +1?)"
    assert (
        k_err_raw > 0.5
    ), f"k_norm must load WITH +1 (uniform-attention fix), but |loaded-raw|={k_err_raw:.4f} (regressed +1?)"
    assert q_err_plus1 < 0.05, "sanity: loaded q_norm must equal raw+1"
    logger.info("PASSED: 27B-TP q_norm/k_norm loaded WITH the +1 offset (uniform-attention fix)")
