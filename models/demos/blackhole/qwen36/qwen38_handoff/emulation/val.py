import time

import torch
from bfp import bfp_quantize
from safetensors import safe_open

import ttnn


def tt(x, dt):
    return ttnn.to_torch(ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT)).float()


torch.manual_seed(0)
tests = {
    "randn": torch.randn(256, 512),
    "randn_wide": torch.randn(256, 512) * torch.exp(torch.randn(256, 512) * 2),
    "sparse": torch.randn(64, 128) * (torch.rand(64, 128) > 0.5),
}
import json

p = "/home/ttuser/atupe/models/Qwen3.8-27B"
idx = json.load(open(p + "/model.safetensors.index.json"))["weight_map"]
for n in [
    "model.language_model.layers.3.mlp.down_proj.weight",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
]:
    if n in idx:
        with safe_open(p + "/" + idx[n], "pt") as f:
            tests[n.split(".")[-3] + n.split(".")[-2]] = f.get_tensor(n).T.contiguous()[:1024, :1024]
for name, x in tests.items():
    xb = x.bfloat16()
    for bits, dt in [(7, ttnn.bfloat8_b), (3, ttnn.bfloat4_b)]:
        ref = tt(xb, dt)
        em = bfp_quantize(xb.float(), bits)
        print(
            name,
            tuple(x.shape),
            bits,
            "maxabs",
            (ref - em).abs().max().item(),
            "exact frac",
            (ref == em).float().mean().item(),
            flush=True,
        )
x = torch.randn(5120, 17408).bfloat16()
for dt in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
    t = time.time()
    tt(x, dt)
    print("ttnn", dt, time.time() - t)
t = time.time()
bfp_quantize(x.float(), 7)
print("torch emul", time.time() - t)
