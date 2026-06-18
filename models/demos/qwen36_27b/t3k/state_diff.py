# SPDX-License-Identifier: Apache-2.0
"""Compare TT DeltaNet final recurrent state vs HF oracle, to localize the bug:
state matches -> recurrence is correct, bug is in output path (q@S / gated-norm / out_proj);
state mismatches -> the multi-token recurrence itself is wrong.
Run in the image: /opt/venv/bin/python3 .../state_diff.py"""
import torch

tt = torch.load("/home/yito/work/tt_acts_st.pt", map_location="cpu")["states"]
hf = torch.load("/home/yito/work/hf_ref_states.pt", map_location="cpu")


def pcc(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    if a.shape != b.shape:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


print("=== Recurrent state PCC (TT vs HF), [1,48,128,128] = [b,v_head,k_dim,v_dim] ===")
for li in [0, 1, 2, 4, 5, 6]:
    k = f"recur{li}"
    if k not in tt or k not in hf:
        print(f"  {k}: missing (tt={k in tt} hf={k in hf})"); continue
    t = tt[k]; h = hf[k]
    overall = pcc(t, h)
    # per-head PCC (detect head-ordering / expansion issues)
    nh = t.shape[1]
    ph = [pcc(t[0, hd], h[0, hd]) for hd in range(nh)]
    worst = sorted(range(nh), key=lambda x: ph[x])[:5]
    nratio = float(t.float().norm() / h.float().norm())
    print(f"  {k}: overall PCC={overall:.5f}  norm(TT/HF)={nratio:.4f}  "
          f"mean/min head PCC={sum(ph)/nh:.4f}/{min(ph):.4f}  worst heads={worst}")

# transposed check: maybe TT stored [v_dim,k_dim] swapped
print("\n=== transpose sanity (swap last two axes of TT) ===")
for li in [0]:
    k = f"recur{li}"
    if k in tt and k in hf:
        print(f"  {k}: PCC(TT^T vs HF)={pcc(tt[k].transpose(-1,-2), hf[k]):.5f}  (vs untransposed {pcc(tt[k],hf[k]):.5f})")
