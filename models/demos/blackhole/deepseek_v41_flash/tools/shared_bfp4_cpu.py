"""shared_bfp4_cpu.py [layers=0,3,10,20,30,39] [tokens=128]: CPU, bit-exact bfp emulation (reference/bfp_emulation.py) of the shared expert weights (w1 gate, w3 up, w2 down; stored [in, out], 16-blocks along
the out dim as in a tile face row) at bfp8 (today) vs bfp4: relative error and PCC of the shared-expert output vs fp32, per layer, on random inputs (gaussian and heavy tailed).
Prints SHB lines. DSV41_CKPT selects the checkpoint (default 1017)."""
import os
import sys

import torch

from models.demos.blackhole.deepseek_v41_flash.reference.bfp_emulation import bfp_roundtrip
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _fp8_linear, _Shards

LIMIT = 10.0  # swiglu_limit


def shared_ffn(x, w1, w3, w2):
    g = (x @ w1).clamp(max=LIMIT)
    u = (x @ w3).clamp(-LIMIT, LIMIT)
    return (torch.nn.functional.silu(g) * u) @ w2


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])


def main(argv):
    layers = [int(x) for x in (argv[1] if len(argv) > 1 else "0,3,10,20,30,39").split(",")]
    n = int(argv[2]) if len(argv) > 2 else 128
    sh = _Shards(os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash-1017"))
    g = torch.Generator().manual_seed(0)
    xs = {
        "gauss": torch.randn(n, 5120, generator=g),
        "heavy": torch.randn(n, 5120, generator=g) * torch.exp(torch.randn(1, 5120, generator=g)),
    }
    for L in layers:
        sp = f"layers.{L}.ffn.shared_experts."
        w = {
            k: _fp8_linear(sh, sp + k, torch.bfloat16).float().contiguous() for k in ("w1", "w3", "w2")
        }  # already [in, out]
        q8 = {k: bfp_roundtrip(v, 7) for k, v in w.items()}
        q4 = {k: bfp_roundtrip(v, 3) for k, v in w.items()}
        q48 = {**q8, "w2": q4["w2"]}  # only the down projection at bfp4
        for name, x in xs.items():
            ref = shared_ffn(x, w["w1"], w["w3"], w["w2"])
            res = []
            for tag, q in (("bfp8", q8), ("bfp4", q4), ("w2_bfp4", q48)):
                y = shared_ffn(x, q["w1"], q["w3"], q["w2"])
                res.append(f"{tag} pcc {pcc(ref, y):.5f} relrms {float((y - ref).norm() / ref.norm()):.4f}")
            print(f"SHB L{L} {name}: " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main(sys.argv)
