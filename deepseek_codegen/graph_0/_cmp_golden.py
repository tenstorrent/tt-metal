# Compare a candidate run's saved live-outs against a golden tag.
# Usage: python _cmp_golden.py <golden_tag> <cand_tag>
import sys, torch, os

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_io")
g, c = torch.load(f"{d}/main_outs_{sys.argv[1]}.pt"), torch.load(f"{d}/main_outs_{sys.argv[2]}.pt")


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    nrm = a.norm() * b.norm()
    return (
        1.0 if float(nrm) == 0.0 else float((a * b).sum() / (nrm + 1e-12))
    )  # constant tensors: identical-by-construction


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
print(f"=== {'PASS' if ok else 'FAIL'} (bit-identical or cos>=0.99 & relmax<1e-2) ===")
