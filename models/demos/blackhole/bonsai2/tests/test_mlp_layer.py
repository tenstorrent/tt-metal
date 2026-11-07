# TernaryMLP (layer 0) on the device against the dense HF MLP: down(silu(gate x) * up x).
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import pack_records_inline
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import BonsaiGGUF
from models.demos.blackhole.bonsai2.tt.ternary.mlp import STEMS, TernaryMLP
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation

GGUF = Path("/local/ttuser/jerrywang/bonsai/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
HF = Path("/local/ttuser/jerrywang/bonsai/out")


def _dense(name):
    wm = json.load(open(HF / "model.safetensors.index.json"))["weight_map"]
    with safe_open(str(HF / wm[name]), "pt") as f:
        return f.get_tensor(name).float()


def test_mlp_layer0(device):
    g = BonsaiGGUF(GGUF)
    packed = {}
    for stem in STEMS:
        t, s = g.ternary(f"blk.0.{stem}.weight")
        rec, ppr = pack_records_inline(t, s)
        packed[stem] = (rec, ppr, t.shape[0], t.shape[1])
    rots = {w: DecodeRotation(device, s) for w, s in g.signs.items()}
    mlp = TernaryMLP(device, packed, rots)
    x = torch.randn(1, 1, 5120) * 0.3
    xt = ttnn.from_torch(
        x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    y = ttnn.to_torch(mlp.forward_decode(xt)).float().reshape(-1)[:5120]
    p = "model.language_model.layers.0.mlp."
    wg, wu, wd = _dense(p + "gate_proj.weight"), _dense(p + "up_proj.weight"), _dense(p + "down_proj.weight")
    xb = x.bfloat16().float().reshape(-1)
    ref = wd @ (torch.nn.functional.silu(wg @ xb) * (wu @ xb))
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"MLP0 rel {rel:.5f} y[:4]={y[:4].tolist()} ref[:4]={ref[:4].tolist()}")
    assert rel < 0.03


def test_mlp_stages(device):
    g = BonsaiGGUF(GGUF)
    rot = DecodeRotation(device, g.signs[5120])
    x = torch.randn(1, 1, 5120) * 0.3
    xt = ttnn.from_torch(
        x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    xr = ttnn.to_torch(rot(xt)).float()
    href = torch.from_numpy(g.rotate(x.bfloat16().float().reshape(-1).numpy().astype(np.float64)).astype(np.float32))
    print(
        "STAGE rot5120 rel",
        ((xr[0] - href).norm() / href.norm()).item(),
        "copy3 rel",
        ((xr[3] - href).norm() / href.norm()).item(),
    )
    # 3D input as the model feeds it; and the same values from a 4D tensor
    x4 = ttnn.from_torch(
        x.bfloat16().reshape(1, 1, 1, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    xr4 = ttnn.to_torch(rot(x4)).float()
    print("STAGE rot5120 4D-input rel", ((xr4[0] - href).norm() / href.norm()).item())
    print("STAGE x tensor shape", list(xt.shape), "padded", list(xt.padded_shape))
