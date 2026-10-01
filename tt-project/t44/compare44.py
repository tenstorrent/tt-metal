# #44 A/B check: both arms' yuv outputs must be identical; prints the timing lines.
import glob, re, sys, torch

V, D = "/var/tmp/fasth3/t44", "/home/smarton/fasth3/t44"
a, b = torch.load(f"{V}/yuv_fold0.pt"), torch.load(f"{V}/yuv_fold1.pt")
same = a.shape == b.shape and torch.equal(a, b)
diff = (a.int() - b.int()).abs().max().item() if a.shape == b.shape else "shape"
print(f"T44 outputs identical={same} max_abs_diff={diff} shape={tuple(a.shape)}")
for f in sorted(glob.glob(f"{D}/run44_fold*.log")):
    print(f, *[l.strip() for l in open(f) if l.startswith("AB44") or "T44_EXIT" in l], sep="\n  ")
sys.exit(0 if same else 1)
