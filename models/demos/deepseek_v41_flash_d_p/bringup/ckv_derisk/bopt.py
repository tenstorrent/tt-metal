"""B with a per-group best shared exponent (E_max or E_max + 1, the smaller group L2 error); still plain bfloat4_b."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec
from safetensors import safe_open

G = "/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4/s10240_c5120"


def q_at(xg, E):  # RNE to m/4 2^(E-127), m <= 7
    step = torch.pow(2.0, (E - 127 - 2).float())
    m = torch.round(xg.abs() / step).clamp(max=7)  # torch.round = half to even
    return torch.copysign(m * step, xg)


for L in (2, 8, 14, 20):
    with safe_open(f"{G}/kv_cache/layer_{L}.safetensors", "pt") as h:
        x = h.get_tensor(f"compress_kv_cache_layer_{L}").float()
    xg = x.view(-1, 32, 16)
    E = ((xg.view(torch.int32) >> 23) & 0xFF).amax(-1, keepdim=True).long()
    a, b = q_at(xg, E), q_at(xg, E + 1)
    ea, eb = ((a - xg) ** 2).sum(-1, keepdim=True), ((b - xg) ** 2).sum(-1, keepdim=True)
    best = torch.where(eb < ea, b, a).view_as(x)
    std = codec.decode_b(codec.encode_b(x))
    print(
        L,
        "B",
        float((std - x).norm() / x.norm()),
        "B_rne_check",
        float((a.view_as(x) - std).abs().max()),
        "B_opt",
        float((best - x).norm() / x.norm()),
        "E+1 picked",
        float((eb < ea).float().mean()),
    )
