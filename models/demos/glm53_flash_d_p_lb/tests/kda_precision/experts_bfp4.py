"""Offline: the routed experts' output error from bfp4 weights alone, last chunk, golden ffn_norm + golden routing:
rel of (bfp4 experts + golden shared) vs the golden mlp_out (the device block's output is routed + shared)."""
import os
import sys
import time

import torch

sys.argv = [sys.argv[0], "bfp4"] + sys.argv[1:]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_lastchunk.py")).read()
exec(src[src.index("def bfp(") : src.index("def quant_w")])
from models.demos.common.bringup.reference.golden import Golden, hf_path, load_spec
from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig, PackedExpert, experts_forward
from models.demos.glm53_flash_d_p.reference.weights import WeightLoader

torch.set_num_threads(16)
spec = load_spec()
path = hf_path(spec)
cfg = GlmConfig.from_json(f"{path}/config.json")
wl = WeightLoader(path)
g = Golden.for_rung(spec, "s56320")


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


for i in [int(v) for v in sys.argv[2].split(",")]:
    t = time.time()
    gl = g.layer(10, i)
    x, r = gl["ffn_norm"].float(), gl["router"].float()
    pe = [PackedExpert(wl, i, e) for e in range(cfg.n_routed_experts)]
    ex = experts_forward(x, r, lambda e: pe[e].weights(torch.float32), cfg.swiglu_limit).float()
    q = experts_forward(
        x, r, lambda e: tuple(bfp(w.t(), 3).t().contiguous() for w in pe[e].weights(torch.float32)), cfg.swiglu_limit
    ).float()
    sh, mo, eo = gl["shared_out"].float(), gl["mlp_out"].float(), gl["experts_out"].float()
    print(
        f"[xb] L{i}: exact experts vs golden experts_out {rel(ex, eo):.4f}; bfp4 experts vs exact {rel(q, ex):.4f}; "
        f"(bfp4 + shared) vs golden mlp_out {rel(q + sh, mo):.4f}; gain {float((q*ex).sum()/(ex*ex).sum()):.4f} ({time.time()-t:.0f}s)",
        flush=True,
    )
