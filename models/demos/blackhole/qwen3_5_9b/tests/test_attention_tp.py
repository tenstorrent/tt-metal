# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 validation for Qwen3.5 full attention on a (1,4) Blackhole mesh.

PCC check at position 0 (attention over a single key reduces to V, so an exact
torch reference covers q/gate split, V proj, GQA mapping, sigmoid gate, output
proj, sharding, and reduce-scatter) + a second decode step for shape/NaN.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=/home/ttuser/atupe/Qwen27b \
      pytest models/demos/blackhole/qwen3_5_9b/tests/test_attention_tp.py -v -s
"""
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_decode
from models.demos.blackhole.qwen3_5_9b.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.tp_common import dequant_fp8_block


def _mp():
    return os.path.expanduser(os.environ.get("HF_MODEL", "/home/ttuser/atupe/Qwen27b"))


def _load_attn_layer(model_path, layer_idx):
    from safetensors import safe_open

    model_path = Path(model_path)
    wm = json.load(open(model_path / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        base = next(k for k in wm if k.endswith(f"layers.{layer_idx}.self_attn.{name}.weight"))
        with safe_open(str(model_path / wm[base]), framework="pt") as sf:
            w = sf.get_tensor(base)
            sk = base + "_scale_inv"
            if wm.get(sk):
                with safe_open(str(model_path / wm[sk]), framework="pt") as sf2:
                    w = dequant_fp8_block(w, sf2.get_tensor(sk))
            else:
                w = w.to(torch.bfloat16)
        out[f"{name}.weight"] = w
    return out


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attention_tp(mesh_device, reset_seeds, ensure_gc):
    mp = _mp()
    os.environ.setdefault("HF_MODEL", mp)
    B = 32
    args = Qwen35ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    logger.info(f"devices={nd} full-attn layer={li} NH={args.n_local_heads} NKV={args.n_local_kv_heads}")

    sd = _load_attn_layer(mp, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_attention_weights_tp(mesh_device, sd, args)
    attn = TPAttention(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cur = torch.zeros(B, dtype=torch.int32)
    cur_tt = ttnn.from_torch(
        cur, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos, sin = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, cur)

    out = attn.forward_decode(x_tt, cur_tt, cos, sin)
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0))[0, 0].float()
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

    from models.common.utility_functions import comp_pcc

    passing, pcc = comp_pcc(ref, out_t, 0.97)
    logger.info(f"ATTENTION TP PCC (pos0) = {pcc}")
    assert passing, f"attention TP PCC too low: {pcc}"

    # second decode step @ pos1: real 2-key attention; shape/NaN only
    cur1 = torch.ones(B, dtype=torch.int32)
    cur1_tt = ttnn.from_torch(
        cur1, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos1, sin1 = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, cur1)
    x2 = ttnn.from_torch(
        torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out2 = attn.forward_decode(x2, cur1_tt, cos1, sin1)
    out2_t = ttnn.to_torch(out2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0))
    assert not torch.isnan(out2_t).any() and out2_t.abs().max() > 0
    logger.info("PASSED: attention TP decode (pos0 PCC + pos1 shape/NaN)")
