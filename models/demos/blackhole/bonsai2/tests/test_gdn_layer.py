# Ternary GDN projections (layer 0) on the device against the dense HF checkpoint (which applied the same v-head
# reorders): in_proj qkv and z, and out_proj.
import json
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.gdn import TernaryGDN, gdn_dims, pack_gdn
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import BonsaiGGUF
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation

GGUF = Path("/local/ttuser/jerrywang/bonsai/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
HF = Path("/local/ttuser/jerrywang/bonsai/out")


def _dense(name):
    wm = json.load(open(HF / "model.safetensors.index.json"))["weight_map"]
    with safe_open(str(HF / wm[name]), "pt") as f:
        return f.get_tensor(name).float()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_gdn_layer0(device):
    g = BonsaiGGUF(GGUF)
    nv, nk, hd, hk = gdn_dims(g)
    rot_in = DecodeRotation(device, g.signs[5120])
    rot_out = DecodeRotation(device, g.signs[nv * hd])
    p = "model.language_model.layers.0.linear_attn."
    wab = torch.cat([_dense(p + "in_proj_a.weight"), _dense(p + "in_proj_b.weight")], 0).T.contiguous()
    wab_t = ttnn.from_torch(wab, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    gdn = TernaryGDN(device, pack_gdn(g, 0), rot_in, rot_out, wab_t)
    x = torch.randn(1, 1, 5120) * 0.3
    xt = ttnn.from_torch(
        x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    qkv, a, b, z = (ttnn.to_torch(t).float().reshape(-1) for t in gdn.in_proj(xt))
    xb = x.bfloat16().float().reshape(-1)
    r_qkv, r_z = _dense(p + "in_proj_qkv.weight") @ xb, _dense(p + "in_proj_z.weight") @ xb
    r_ab = wab.T @ xb
    print(f"GDN a rel {_rel(a[:48], r_ab[:48]):.5f} b rel {_rel(b[:48], r_ab[48:]):.5f}")
    print(
        f"GDN qkv rel {_rel(qkv[:10240], r_qkv):.5f}  v-part rel {_rel(qkv[4096:10240], r_qkv[4096:]):.5f}  z rel {_rel(z[:6144], r_z):.5f}"
    )
    o = torch.randn(1, 1, nv * hd) * 0.3
    ot = ttnn.from_torch(o.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(gdn.out_proj(ot)).float().reshape(-1)[:5120]
    r_y = _dense(p + "out_proj.weight") @ o.bfloat16().float().reshape(-1)
    print(f"GDN out rel {_rel(y, r_y):.5f}")
    assert _rel(qkv[:10240], r_qkv) < 0.02 and _rel(z[:6144], r_z) < 0.02 and _rel(y, r_y) < 0.02
