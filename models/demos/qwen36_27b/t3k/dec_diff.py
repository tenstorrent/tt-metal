# SPDX-License-Identifier: Apache-2.0
"""Per-layer PCC of TT decode-step-0 hidden vs HF 6-token position-5 reference.
Finds the first layer where the decode path diverges -> localizes the decode bug
(DeltaNet decode kernel vs full-attention KV-cache/SDPA vs state handoff)."""
import torch

tt = torch.load("/home/yito/work/tt_dec_acts.pt", map_location="cpu")
hf = torch.load("/home/yito/work/hf_dec_ref.pt", map_location="cpu")


def pcc(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    if a.shape != b.shape:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


tta = tt["acts"]          # [65,1,5120]; [0]=embed, [i+1]=layer i
names = tt["names"]
hfl = hf["layers"]        # [64,1,5120]
print(f"embed PCC={pcc(tta[0], hf['embed']):.5f}")
prev = pcc(tta[0], hf["embed"])
for i in range(min(tta.shape[0] - 1, hfl.shape[0])):
    p = pcc(tta[i + 1], hfl[i])
    drop = prev - p
    mark = "  <== BIG DROP" if drop > 0.05 else ""
    print(f"{names[i+1]:>22}  PCC={p:.5f}  d={drop:+.5f}{mark}")
    prev = p
