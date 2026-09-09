# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-05 op probe: partial RoPE via rotary_embedding_llama with a rotate-half trans_mat, SDPA with a
padded-key mask, nlp_create_qkv_heads for 32 x 64 heads at batch 2, and layer_norm with weight+bias.
Run: with_hw_lock timeout 600 $MM3_PY $MM3_MODEL_DIR/scripts/probe_dit_ops.py
"""
import torch

import ttnn
from models.common.utility_functions import comp_pcc

torch.manual_seed(0)
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    B, H, S, D, RD = 2, 32, 768, 64, 32
    S_real = 690
    x = torch.randn(B, H, S, D)
    # reference partial rope on [B,H,S,D]
    inv = 1.0 / (10000 ** (torch.arange(0, RD, 2).float() / RD))
    fr = torch.outer(torch.arange(S).float(), inv)
    fr = torch.cat([fr, fr], -1)
    cos, sin = fr.cos(), fr.sin()
    r = x[..., :RD]
    h1, h2 = r.chunk(2, -1)
    rot = torch.cat([-h2, h1], -1)
    ref = torch.cat([r * cos + rot * sin, x[..., RD:]], -1)
    # device: cos/sin padded to D with (1, 0); trans_mat = rotate-half within a 32-tile
    cos_d = torch.cat([cos, torch.ones(S, D - RD)], -1).reshape(1, 1, S, D)
    sin_d = torch.cat([sin, torch.zeros(S, D - RD)], -1).reshape(1, 1, S, D)
    tm = torch.zeros(1, 1, 32, 32)
    for j in range(16):
        tm[..., j + 16, j] = -1.0  # out[j]      = -x[j+16]
        tm[..., j, j + 16] = 1.0  # out[j + 16] =  x[j]
    to = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    xd = to(x.reshape(1, B * H, S, D))
    y = ttnn.experimental.rotary_embedding_llama(xd, to(cos_d), to(sin_d), to(tm), is_decode_mode=False)
    y = ttnn.to_torch(y).reshape(B, H, S, D)
    print("rope pcc", comp_pcc(ref, y, 0.99), "maxerr", (ref - y).abs().max().item())
    # SDPA with a padded key mask
    q, k, v = torch.randn(3, B, H, S, D)
    mask = torch.zeros(1, 1, S, S)
    mask[..., S_real:] = -1e9
    ref_attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(dev.compute_with_storage_grid_size().x, dev.compute_with_storage_grid_size().y),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    for mdt in (ttnn.bfloat16, ttnn.bfloat4_b):
        try:
            out = ttnn.transformer.scaled_dot_product_attention(
                to(q),
                to(k),
                to(v),
                attn_mask=to(mask, mdt),
                is_causal=False,
                program_config=pc,
                compute_kernel_config=ck,
            )
            out = ttnn.to_torch(out)
            print("sdpa mask", mdt, comp_pcc(ref_attn[..., :S_real, :], out[..., :S_real, :], 0.99))
        except Exception as e:
            print("sdpa mask", mdt, "FAILED", str(e)[:300])
    # qkv heads split
    qkv = torch.randn(B, 1, S, 3 * H * D)
    qd, kd, vd = ttnn.experimental.nlp_create_qkv_heads(to(qkv), num_heads=H, num_kv_heads=H, transpose_k_heads=False)
    print("qkv shapes", qd.shape, kd.shape, vd.shape)
    qq = ttnn.to_torch(qd).float()
    print(
        "q head split ok",
        torch.allclose(qq, qkv[..., : H * D].reshape(B, S, H, D).permute(0, 2, 1, 3).bfloat16().float(), atol=1e-2),
    )
    # layer norm w/ bias on [1,1,2S,2048]
    xx = torch.randn(1, 1, 2 * S, 2048)
    w = torch.randn(2048)
    b = torch.randn(2048)
    refln = torch.nn.functional.layer_norm(xx, (2048,), w, b, 1e-5)
    ln = ttnn.layer_norm(
        to(xx),
        weight=to(w.reshape(1, 1, 1, -1)),
        bias=to(b.reshape(1, 1, 1, -1)),
        epsilon=1e-5,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    print("ln pcc", comp_pcc(refln, ttnn.to_torch(ln), 0.99))
    # matmul with bias + fused silu, batched input [1,1,2S,2048] @ [2048,8192]
    a = torch.randn(1, 1, 2 * S, 2048)
    wm = torch.randn(2048, 8192) / 45
    bm = torch.randn(8192)
    refm = torch.nn.functional.silu(a @ wm + bm)
    mm = ttnn.linear(
        to(a),
        to(wm),
        bias=to(bm.reshape(1, -1)),
        activation="silu",
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    print("linear silu pcc", comp_pcc(refm, ttnn.to_torch(mm), 0.99))
    print("grid", dev.compute_with_storage_grid_size())
finally:
    ttnn.close_mesh_device(dev)
