# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 validation for Qwen3.5 Gated DeltaNet decode on a (1,4) Blackhole mesh.

PCC at position 0 (recurrent state starts at zero, so o = beta*(q̂·k̂)*v); the
torch reference covers the sharded QKV/Z/AB reorder, per-channel conv, GQA head
expansion, L2 norm, gated RMSNorm, Z-gate, output projection, and reduce-scatter.
Plus a second decode step for shape/NaN.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=/home/ttuser/atupe/Qwen27b \
      pytest models/demos/blackhole/qwen3_5_9b/tests/test_gdn_tp.py -v -s
"""
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.tp_common import dequant_fp8_block


def _mp():
    return os.path.expanduser(os.environ.get("HF_MODEL", "/home/ttuser/atupe/Qwen27b"))


def _load_gdn_layer(model_path, li):
    from safetensors import safe_open

    model_path = Path(model_path)
    wm = json.load(open(model_path / "model.safetensors.index.json"))["weight_map"]
    names = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "conv1d", "A_log", "dt_bias", "norm"]
    out = {}
    for n in names:
        key = "weight" if n != "A_log" and n != "dt_bias" else None
        suffix = f"linear_attn.{n}.weight" if key else f"linear_attn.{n}"
        base = next(k for k in wm if k.endswith(f"layers.{li}.{suffix}"))
        with safe_open(str(model_path / wm[base]), framework="pt") as sf:
            w = sf.get_tensor(base)
            sk = base + "_scale_inv"
            if wm.get(sk):
                with safe_open(str(model_path / wm[sk]), framework="pt") as sf2:
                    w = dequant_fp8_block(w, sf2.get_tensor(sk))
            else:
                w = w.to(torch.bfloat16)
        out[f"linear_attn.{n}" + (".weight" if key else "")] = w
    return out


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_gdn_tp(mesh_device, reset_seeds, ensure_gc):
    mp = _mp()
    os.environ.setdefault("HF_MODEL", mp)
    B = 32
    args = Qwen35ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} Nk_tp={args.gdn_nk_tp} Nv_tp={args.gdn_nv_tp}")

    sd = _load_gdn_layer(mp, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = gdn.forward_decode(x_tt)
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0))[0, 0].float()
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

    from models.common.utility_functions import comp_pcc

    passing, pcc = comp_pcc(ref, out_t, 0.92)
    logger.info(f"GDN TP PCC (pos0) = {pcc}")
    assert passing, f"GDN TP PCC too low: {pcc}"

    x2 = ttnn.from_torch(
        torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out2 = gdn.forward_decode(x2)
    out2_t = ttnn.to_torch(out2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0))
    assert not torch.isnan(out2_t).any() and out2_t.abs().max() > 0
    logger.info("PASSED: GDN TP decode (pos0 PCC + pos1 shape/NaN)")
