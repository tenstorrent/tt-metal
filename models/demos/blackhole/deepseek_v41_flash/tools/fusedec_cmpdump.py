"""python3 -I fusedec_cmpdump.py <dump_a.pt> <dump_b.pt> -- bit equality and PCC of two FD_DUMP files (pack, logits of chip 0, compressor state of every owner)"""
import sys

import torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])


def pcc(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    if float(x.std()) == 0 or float(y.std()) == 0:
        return 1.0 if torch.equal(x, y) else 0.0
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


for i, (x, y) in enumerate(zip(a["pack"], b["pack"])):
    if torch.is_tensor(x):
        print(f"pack[{i}] equal {torch.equal(x, y)}")
print(
    f"logits chip0: equal {torch.equal(a['logits0'], b['logits0'])} pcc {pcc(a['logits0'], b['logits0']):.6f} maxabs {float((a['logits0'] - b['logits0']).abs().max()):.4g}"
)
la, lb = a["logits0"], b["logits0"]
print(
    "argmax rows equal:",
    bool(torch.equal(la.reshape(-1, la.shape[-1]).argmax(-1), lb.reshape(-1, lb.shape[-1]).argmax(-1))),
)
for lid in a["prev_cs"]:
    eq = all(torch.equal(x, y) for x, y in zip(a["prev_cs"][lid], b["prev_cs"][lid]))
    print(f"prev_cs layer {lid}: equal {eq}")
