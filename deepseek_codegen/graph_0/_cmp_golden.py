# Compare a candidate run's saved live-outs against (1) a HEAD-relative golden tag and
# (2) the absolute torch golden logits (../golden_logits.pt, ~0.90 bf4 floor).
# Usage: python _cmp_golden.py <head_tag> <cand_tag>
import os
import sys

import torch

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_io")
GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden_logits.pt")


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    nrm = a.norm() * b.norm()
    return 1.0 if float(nrm) == 0.0 else float((a * b).sum() / (nrm + 1e-12))


g, c = torch.load(f"{d}/main_outs_{sys.argv[1]}.pt"), torch.load(f"{d}/main_outs_{sys.argv[2]}.pt")
ok = True
for n in g:
    a, b = g[n].double(), c[n].double()
    bit = torch.equal(g[n], c[n])
    rel = float((a - b).abs().max()) / (float(a.abs().max()) + 1e-12)
    p = pcc(a, b)
    good = bit or (p >= 0.99 and rel < 1e-2)
    ok &= good
    extra = f"  ARGMAX={'100%' if bit else 'DIFF'}" if "typecast_106" in n else ""
    print(f"  {n:16s} bit_ident={bit} cos64={p:.8f} relmax={rel:.2e}{extra}  {'ok' if good else 'FAIL'}")
print(f"=== HEAD-rel vs {sys.argv[1]}: {'PASS' if ok else 'FAIL'} (bit-identical or cos>=0.99 & relmax<1e-2) ===")

# Absolute golden PCC on the final logits (to_layout_267), held floor ~0.8989 (bf4 moe_compute).
if os.path.exists(GOLDEN) and "to_layout_267" in c:
    gl = torch.load(GOLDEN).float().reshape(-1, torch.load(GOLDEN).shape[-1])
    lg = c["to_layout_267"].float().reshape(-1, c["to_layout_267"].shape[-1])
    nrows = min(gl.shape[0], lg.shape[0])
    gp = pcc(lg[:nrows], gl[:nrows])
    am = float((lg[:nrows].argmax(-1) == gl[:nrows].argmax(-1)).float().mean()) * 100
    print(f"=== ABSOLUTE golden PCC = {gp:.4f}  argmax_agree={am:.1f}%  (floor ~0.8989; FAIL if < 0.895) ===")
