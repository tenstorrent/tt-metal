# Every bfloat16 bit pattern through the scalar and ternary forms the bf16 rounding change reaches (binop with a scalar,
# lerp, addcdiv), bfloat16 and float32; a sha256 per case. usage: python exh_sct.py <out.json>
import hashlib
import json
import sys

import torch
import ttnn

pat = torch.arange(65536, dtype=torch.int32)
allbf = (pat - (pat >= 32768).to(torch.int32) * 65536).to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
g = torch.Generator().manual_seed(58207)
other = (torch.randn(1, 1, 256, 256, generator=g) * 8).to(torch.bfloat16)
third = (torch.rand(1, 1, 256, 256, generator=g) * 2 + 0.25).to(torch.bfloat16)
SCALARS = [0.0, -0.0, 1.0, -1.0, 0.5, 3.0, -2.75, 1e-3, 1e30, float("inf")]
dev = ttnn.open_device(device_id=0)
res = {}


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:20]


def tt(x, d):
    return ttnn.from_torch(x.to(torch.float32 if d == ttnn.float32 else torch.bfloat16), dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)


try:
    for dn, d in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
        a, b, c = tt(allbf, d), tt(other, d), tt(third, d)
        for s in SCALARS:
            for op in ("add", "subtract", "multiply", "rsub", "divide"):
                for mode, kw in (("", {}), ("|sfpu", {"fast_and_approximate_mode": False})):
                    try:
                        res[f"{op}_s|{dn}|{s}{mode}"] = h(ttnn.to_torch(getattr(ttnn, op)(a, s, **kw)))
                    except Exception as e:  # noqa: BLE001
                        res[f"{op}_s|{dn}|{s}{mode}"] = "skip: " + str(e).splitlines()[0][:80]
        for w in (0.25, 0.5, 0.75):
            res[f"lerp_s|{dn}|{w}"] = h(ttnn.to_torch(ttnn.lerp(a, b, w)))
            res[f"lerp_s2|{dn}|{w}"] = h(ttnn.to_torch(ttnn.lerp(b, a, w)))
        res[f"lerp_t|{dn}"] = h(ttnn.to_torch(ttnn.lerp(a, b, c)))
        for v in (0.5, 1.0, -3.0):
            res[f"addcdiv|{dn}|{v}"] = h(ttnn.to_torch(ttnn.addcdiv(a, b, c, value=v)))
            res[f"addcdiv2|{dn}|{v}"] = h(ttnn.to_torch(ttnn.addcdiv(b, a, c, value=v)))
        print(dn, "done", flush=True)
finally:
    ttnn.close_device(dev)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("saved", sys.argv[1], len(res))
