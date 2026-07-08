# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP validation for Qwen3.5/3.6 Gated DeltaNet on a Blackhole mesh.

One file per component (decode / chunk-prefill), sharing the loaders and mesh
parametrization from ``test_factory``:

* ``test_gdn_tp``         — decode PCC @ pos0 (recurrent state starts at zero, so
  o = beta*(q̂·k̂)*v); the torch reference covers the sharded QKV/Z/AB reorder,
  per-channel conv, GQA head expansion, L2 norm, gated RMSNorm, Z-gate, output
  projection, and reduce-scatter. Plus a second decode step for shape/NaN.
* ``test_gdn_tp_prefill`` — chunk-prefill (FIR conv + shared chunk kernel) must
  agree with step-by-step decode over the same tokens (zero init state). An
  internal-consistency check across two code paths; no hand-written reference.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
"""
import os

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    get_pcc_threshold,
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    shard_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_tp(mesh_device, reset_seeds, ensure_gc, request):
    """Validate TP decode output against a hand-written PyTorch reference at pos0 (batch=32).

    Checks PCC for the full GDN forward pass (QKV proj, conv tap, L2 norm, beta gating,
    gated RMSNorm, output proj) and runs a second decode step to catch shape/NaN regressions.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    B = 32
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} Nk_tp={args.gdn_nk_tp} Nv_tp={args.gdn_nv_tp}")

    # args.CKPT_DIR is the resolved local snapshot dir (Qwen36ModelArgs downloads the hub id).
    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
    x_tt = replicate_to_device(mesh_device, x)
    out = gdn.forward_decode(x_tt)
    out_t = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].float()
    assert out_t.shape[-1] == args.dim and not torch.isnan(out_t).any() and out_t.abs().max() > 0

    # ---- torch reference @ pos0 (full, unsharded) ----
    Nk, Nv, Dk, Dv = args.gdn_nk, args.gdn_nv, args.gdn_dk, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    xf = x[0, 0].float()
    qkv = xf @ sd["linear_attn.in_proj_qkv.weight"].float().T  # [B, 2*key_dim+value_dim]
    z = xf @ sd["linear_attn.in_proj_z.weight"].float().T
    b = xf @ sd["linear_attn.in_proj_b.weight"].float().T  # [B, Nv]
    tap3 = sd["linear_attn.conv1d.weight"].float()[:, 0, 3]  # [qkv_dim], newest-token tap
    conv = F.silu(qkv * tap3)
    q = conv[:, :key_dim].reshape(B, Nk, Dk)
    k = conv[:, key_dim : 2 * key_dim].reshape(B, Nk, Dk)
    v = conv[:, 2 * key_dim :].reshape(B, Nv, Dv)
    rf = Nv // Nk
    q = q.repeat_interleave(rf, dim=1)
    k = k.repeat_interleave(rf, dim=1)
    q = F.normalize(q, dim=-1) * (Dk**-0.5)
    k = F.normalize(k, dim=-1)
    beta = torch.sigmoid(b)  # [B, Nv]
    qk = (q * k).sum(-1)  # [B, Nv]
    o = beta[..., None] * qk[..., None] * v  # [B, Nv, Dv]
    # gated RMSNorm over Dv (weight only, NO +1)
    o_n = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * sd["linear_attn.norm.weight"].float()
    gated = (o_n * F.silu(z.reshape(B, Nv, Dv))).reshape(B, value_dim)
    ref = gated @ sd["linear_attn.out_proj.weight"].float().T  # [B, dim]

    passing, pcc = comp_pcc(ref, out_t, get_pcc_threshold(request))
    logger.info(f"GDN TP PCC (pos0) = {pcc}")
    assert passing, f"GDN TP PCC too low: {pcc}"

    x2 = replicate_to_device(mesh_device, torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16))
    out2 = gdn.forward_decode(x2)
    out2_t = ttnn.to_torch(out2, mesh_composer=tp_composer(mesh_device))
    assert not torch.isnan(out2_t).any() and out2_t.abs().max() > 0
    logger.info("PASSED: GDN TP decode (pos0 PCC + pos1 shape/NaN)")


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_tp_prefill(mesh_device, reset_seeds, ensure_gc, request):
    """Check that chunk-prefill and step-by-step decode agree on the same T=128 tokens.

    Both paths start from zero state. No hand-written reference — this is a
    self-consistency check between forward_prefill and forward_decode.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    T = 128
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} T={T}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    # Prefill input is K-sharded (the model's prefill norm skips its AG; the fused in-proj gathers).
    x_tt = shard_to_device(mesh_device, x, dim=-1)
    composer = tp_composer(mesh_device)

    # ---- Prefill ----
    gdn.reset_state()
    out_pf = gdn.forward_prefill(x_tt, chunk_size=128)
    pf = ttnn.to_torch(out_pf, mesh_composer=composer)[0, 0].float()  # [T, dim]

    # ---- Decode the same tokens one at a time ----
    gdn.reset_state()
    dec_rows = []
    for t in range(T):
        xt = replicate_to_device(mesh_device, x[:, :, t : t + 1, :])
        ot = gdn.forward_decode(xt)
        dec_rows.append(ttnn.to_torch(ot, mesh_composer=composer)[0, 0, 0].float())  # [dim]
    dec = torch.stack(dec_rows, dim=0)  # [T, dim]

    passing, pcc = comp_pcc(dec, pf, get_pcc_threshold(request))
    logger.info(f"GDN TP PREFILL vs DECODE PCC (T={T}) = {pcc}")
    assert passing, f"GDN prefill/decode mismatch PCC: {pcc}"
