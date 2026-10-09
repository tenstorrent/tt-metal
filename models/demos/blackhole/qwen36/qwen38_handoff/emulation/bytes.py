import glob
import json
import re
import struct

D = "/home/ttuser/atupe/models/Qwen3.8-27B/"
sh = {}
for f in glob.glob(D + "*.safetensors"):
    fh = open(f, "rb")
    n = struct.unpack("<Q", fh.read(8))[0]
    h = json.loads(fh.read(n))
    for k, v in h.items():
        if k != "__metadata__":
            sh[k] = v["shape"]
cat = {}


def add(c, n):
    cat[c] = cat.get(c, 0) + n


for k, s in sh.items():
    if "visual" in k or k.startswith("mtp"):
        continue
    n = s[0] * (s[1] if len(s) > 1 else 1)
    if k.endswith("gate_proj.weight"):
        add("gate", n)
    elif k.endswith("up_proj.weight"):
        add("up", n)
    elif k.endswith("down_proj.weight"):
        add("down", n)
    elif re.search("in_proj_(qkv|z)", k):
        add("gdn_in", n)
    elif re.search("in_proj_[ab]\.", k):
        add("gdn_ab", s[0] // 4 * 0 + (32 * 4) * s[1] * 0 + n)
        add("gdn_ab_pad", 32 * 4 * s[1] * 2 // 2 - 0) if False else None
    elif k.endswith("out_proj.weight"):
        add("gdn_out", n)
    elif re.search("self_attn.[qkvo]_proj", k):
        add("attn", n)
    elif k == "lm_head.weight":
        add("lm", n)
# a/b: fused pad per device: a_d(12)+b_d(12)->32 cols; per layer 4 devs*32*K elems total vs 2*48*K
L = 48
K = 5120
ab_padded = L * 4 * 32 * K
cat["gdn_ab"] = ab_padded
print({k: v / 1e9 for k, v in cat.items()})
B8, B4, B16 = 1.0625, 0.5625, 2.0
cfgs = {  # (gate,up,down,gdn_in(incl ab),gdn_out,attn,lm)
    "TT-default(Q8)": (4, 4, 8, 8, 8, 8, 8),
    "Q4": (4, 4, 4, 8, 8, 4, 8),
    "Q4all": (4, 4, 4, 4, 8, 4, 8),
    "E7/Q8noGU": (8, 8, 8, 8, 8, 8, 8),
    "E1": (8, 8, 4, 8, 8, 8, 8),
    "E2": (8, 8, 8, 8, 8, 4, 8),
    "E3": (8, 8, 4, 8, 8, 4, 8),
    "E4": (8, 8, 8, 4, 8, 8, 8),
    "E5": (8, 8, 8, 8, 4, 8, 8),
    "E6": (8, 8, 8, 8, 8, 8, 4),
    "E8": (4, 8, 8, 8, 8, 8, 8),
    "E9/Q0 bf16": (16,) * 7,
}
by = {4: B4, 8: B8, 16: B16}
keys = ["gate", "up", "down", "gdn_in", "gdn_out", "attn", "lm"]
res = {}
for n, c in cfgs.items():
    tot = 0
    for k, b in zip(keys, c):
        e = cat[k] + (cat["gdn_ab"] if k == "gdn_in" else 0)
        tot += e * by[b]
    res[n] = tot / 4
base = res["TT-default(Q8)"]
json.dump({"GB_per_dev": {k: v / 1e9 for k, v in res.items()}, "elems": cat}, open("bytes.json", "w"))
for n, v in res.items():
    print(
        f"{n:16s} {v/1e9:7.3f} GB/token/device  {4*v/1e9:7.2f} GB total  delta vs default {(v-base)/1e9:+.3f} ({100*(v-base)/base:+.1f}%)"
    )
