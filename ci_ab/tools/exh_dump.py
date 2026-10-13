# sfpu_binary verdict rework, 07's exhaustive bit identity: every bfloat16 bit pattern through the ops this branch changes,
# as one operand against a set of special and ordinary values for the other, in both orders (binary ops), and alone (lgamma);
# bfloat16 tensors and the same patterns widened to float32 tensors. Writes a sha256 of every output (and the input bits it
# used) per case, so the farms t_main and t_c can be compared case by case. usage: python exh_dump.py <out.json> [op,...]
import os as _os
import sys as _sys
_mock = _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")
if not _os.environ.get("HWLOCK_HELD") and not (_mock and _os.path.isfile(_mock)):
    _sys.exit("not under hwlock")
import hashlib
import json
import sys

import torch
import ttnn

OUT = sys.argv[1]
BIN = ["divide", "maximum", "minimum", "add", "subtract", "multiply", "rsub", "pow", "eq", "ne", "lt", "gt", "le", "ge",
       "atan2", "fmod", "remainder", "logaddexp", "logaddexp2", "xlogy", "isclose"]
OPS = sys.argv[2].split(",") if len(sys.argv) > 2 else BIN + ["lgamma"]
OTHER = [0x0000, 0x8000, 0x0001, 0x8001, 0x0040, 0x8040, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x4040, 0xC040, 0x3F40, 0x7F7F,
         0xFF7F, 0x7E80, 0x0100, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x4049, 0x3F7F, 0x3F81, 0xBF7F, 0x4120, 0xC2C8]
pat = torch.arange(65536, dtype=torch.int32)
allbf = (pat - (pat >= 32768).to(torch.int32) * 65536).to(torch.int16).view(torch.bfloat16)
_g = torch.Generator().manual_seed(60072)
OTHER = OTHER + [int(v) for v in torch.randint(0, 65536, (36,), generator=_g)]
k = len(OTHER)
full = allbf.repeat(k)                                                     # every pattern, once per other value
oth = torch.tensor(OTHER, dtype=torch.int32)
oth = (oth - (oth >= 32768).to(torch.int32) * 65536).to(torch.int16).view(torch.bfloat16).repeat_interleave(65536)
shape = (1, 1, k * 256, 256)
res = {"lanes_per_case": int(full.numel()), "other": OTHER}
dev = ttnn.open_device(device_id=0)


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:20]


try:
    for op in OPS:
        for dname, d in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
            if op == "lgamma":
                x = allbf.to(torch.float32 if dname == "fp32" else torch.bfloat16).reshape(1, 1, 256, 256)
                y = ttnn.to_torch(ttnn.lgamma(ttnn.from_torch(x, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)))
                res[f"{op}|{dname}|all"] = h(y)
                continue
            fn = getattr(ttnn, op)
            for order, (a, b) in (("all_first", (full, oth)), ("all_second", (oth, full))):
                ta = a.to(torch.float32 if dname == "fp32" else torch.bfloat16).reshape(shape)
                tb = b.to(torch.float32 if dname == "fp32" else torch.bfloat16).reshape(shape)
                try:
                    y = ttnn.to_torch(fn(ttnn.from_torch(ta, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev),
                                         ttnn.from_torch(tb, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)))
                    res[f"{op}|{dname}|{order}"] = h(y)
                except Exception as e:  # an unsupported dtype only skips its case
                    res[f"{op}|{dname}|{order}"] = "skip: " + str(e).splitlines()[0][:80]
        print(op, "done", flush=True)
finally:
    ttnn.close_device(dev)
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
