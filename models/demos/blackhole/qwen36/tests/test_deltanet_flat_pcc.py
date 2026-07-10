# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Flattened-head (B>1) fused deltanet_decode_full probe — fast repro for the device hang.

Packs B lanes into the head axis (num_heads = B*Nv, qkv = [all_q|all_k|all_v]) in ONE kernel
call and compares the raw output + new_state to a per-lane torch reference. If the single call
hangs, this reproduces it in ~seconds (no 27B). TEST_B selects the batch (default 2).

Run: MESH_DEVICE=P150x4 TEST_B=2 pytest .../test_deltanet_flat_pcc.py -s
"""
import os
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@parametrize_mesh_tp()
def test_deltanet_flat(mesh_device, ensure_gc):
    from loguru import logger

    B = int(os.environ.get("TEST_B", "2"))
    Nk, Nv, Dk, Dv, K = 4, 12, 128, 128, 4
    rf = Nv // Nk
    scale = Dk**-0.5
    kd, vd = Nk * Dk, Nv * Dv
    qkv_dim = 2 * kd + vd
    H = B * Nv
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    torch.manual_seed(0)

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    def l2(x):
        return x / (x.pow(2).sum(-1, keepdim=True).sqrt() + 1e-12)

    q = torch.randn(B, Nk, Dk) * 0.3
    k = torch.randn(B, Nk, Dk) * 0.3
    v = torch.randn(B, Nv, Dv) * 0.3
    beta = torch.rand(B, Nv)
    g = -torch.rand(B, Nv) * 0.1
    rec = torch.randn(B, Nv, Dk, Dv) * 0.1

    # Host-fold L2norm+scale into q,k (un-expanded), pack [all_q | all_k | all_v] batch-major.
    qn = l2(q) * scale  # [B,Nk,Dk]
    kn = l2(k)
    PACK = os.environ.get("PACK", "host")
    if PACK == "host":  # known-good: pack on host, single dev()
        qkv_proj = dev(
            torch.cat([qn.reshape(1, 1, 1, B * kd), kn.reshape(1, 1, 1, B * kd), v.reshape(1, 1, 1, B * vd)], dim=-1)
        )
    else:  # build [B,Nk,Dk] device tensors, pack ON DEVICE (mimics gdn/tp.py)
        qn_d, kn_d, v_d = dev(qn), dev(kn), dev(v)
        if PACK == "device":  # naive device reshape (flattens tiled dims) — expected to mangle
            q_f = ttnn.reshape(qn_d, (1, 1, B * kd))
            k_f = ttnn.reshape(kn_d, (1, 1, B * kd))
            v_f = ttnn.reshape(v_d, (1, 1, B * vd))
            qkv_proj = ttnn.reshape(ttnn.concat([q_f, k_f, v_f], dim=-1), (1, 1, 1, B * qkv_dim))
        elif PACK == "rm":  # reshape-safe: ROW_MAJOR detour, then back to TILE

            def flat_rm(t, w):
                return ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, w))

            qkv_rm = ttnn.concat([flat_rm(qn_d, B * kd), flat_rm(kn_d, B * kd), flat_rm(v_d, B * vd)], dim=-1)
            qkv_proj = ttnn.to_layout(qkv_rm, ttnn.TILE_LAYOUT)
    z_p = dev(torch.randn(1, 1, 1, B * vd) * 0.3)
    b_p = dev(beta.reshape(1, 1, 1, H))
    a_p = dev(torch.exp(g).reshape(1, 1, 1, H))  # decay
    dummy = dev(torch.zeros(1, 1, B * qkv_dim, 32))
    zh = dev(torch.zeros(1, 1, 1, H))
    nw = dev(torch.ones(1, 1, 1, Dv))
    ms = os.environ.get("MODEL_STATE", "0")
    if ms == "1":  # fp32 [B,Nv,Dk,Dv] + device reshape (exact gdn path)
        rec_bnv = ttnn.from_torch(rec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        rec_flat = ttnn.reshape(rec_bnv, (1, H, Dk, Dv))
    elif ms == "2":  # bf16 [B,Nv,Dk,Dv] + device reshape (isolate reshape vs dtype)
        rec_bnv = ttnn.from_torch(rec, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        rec_flat = ttnn.reshape(rec_bnv, (1, H, Dk, Dv))
    else:  # bf16, built directly as [1,H,Dk,Dv] on host (known-good)
        rec_flat = dev(rec.reshape(1, H, Dk, Dv))

    logger.info(f"calling deltanet_decode_full flattened: B={B} H={H} num_k_heads={B*Nk} conv_dim={B*qkv_dim}")
    out = ttnn.experimental.deltanet_decode_full(
        qkv_proj, z_p, b_p, a_p, dummy, rec_flat, dummy, zh, zh, nw,
        num_heads=H, num_k_heads=B * Nk, k_head_dim=Dk, v_head_dim=Dv,
        conv_dim=B * qkv_dim, conv_kernel_size=K, head_expand_ratio=rf,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("kernel returned (no hang)")
    k_out = ttnn.to_torch(out[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].reshape(H, Dv)
    k_rec = ttnn.to_torch(out[1], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].reshape(H, Dk, Dv)

    # Per-lane torch reference (raw q@S_new, no norm/gate).
    outs, recs = [], []
    for bi in range(B):
        qT = (l2(q[bi]) * scale).repeat_interleave(rf, dim=0)  # [Nv,Dk] block-expand
        kT = l2(k[bi]).repeat_interleave(rf, dim=0)
        hT = rec[bi] * torch.exp(g[bi]).reshape(Nv, 1, 1)
        vr = torch.einsum("hk,hkv->hv", kT, hT)
        hT = hT + beta[bi].reshape(Nv, 1, 1) * torch.einsum("hk,hv->hkv", kT, vr.mul(-1).add(v[bi]))
        outs.append(torch.einsum("hk,hkv->hv", qT, hT))
        recs.append(hT)
    torch_out = torch.cat(outs, dim=0)  # [H,Dv]
    torch_rec = torch.cat(recs, dim=0)  # [H,Dk,Dv]
    logger.info(f"FLAT PCC: raw={_pcc(k_out, torch_out):.5f}  new_state={_pcc(k_rec, torch_rec):.5f}")
    # Tail reshape check: the shared gdn tail reshapes o [1,1,1,B*Nv*Dv] -> (B,Nv,Dv). Verify that
    # device reshape (splitting the tiled row) preserves data at B>1 (vs a ROW_MAJOR detour).
    tail_dev = ttnn.reshape(out[0], (B, Nv, Dv))
    tail_t = ttnn.to_torch(tail_dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].reshape(H, Dv)
    tail_rm = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(out[0], ttnn.ROW_MAJOR_LAYOUT), (B, Nv, Dv)), ttnn.TILE_LAYOUT)
    tail_rm_t = ttnn.to_torch(tail_rm, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].reshape(H, Dv)
    logger.info(f"TAIL reshape vs raw: device={_pcc(tail_t, k_out):.5f}  rowmajor={_pcc(tail_rm_t, k_out):.5f}")
    assert _pcc(k_out, torch_out) > 0.99 and _pcc(k_rec, torch_rec) > 0.99
    logger.info("PASSED: flattened-head fused matches per-lane torch")
