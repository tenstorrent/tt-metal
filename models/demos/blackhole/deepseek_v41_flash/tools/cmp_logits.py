import sys

import torch


def pcc(x, y):
    x = x.flatten().double()
    y = y.flatten().double()
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()))


ld = lambda p: torch.load(p).float()
a, b = ld(sys.argv[1]), ld(sys.argv[2])
print(
    f"{sys.argv[1].split('/')[-1]} vs {sys.argv[2].split('/')[-1]}: pcc {pcc(a,b):.6f} argmax match {(a.argmax(-1)==b.argmax(-1)).sum().item()}/{a.shape[0]} maxabs {(a-b).abs().max().item():.3f} (scale {a.abs().max().item():.1f})"
)
