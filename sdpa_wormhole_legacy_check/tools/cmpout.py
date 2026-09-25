import sys, os, torch
a, b = sys.argv[1:3]
n = same = 0
for f in sorted(set(os.listdir(a)) | set(os.listdir(b))):
    if f not in os.listdir(a) or f not in os.listdir(b):
        print("MISSING", f); continue
    x, y = torch.load(os.path.join(a, f)), torch.load(os.path.join(b, f))
    eq = x.shape == y.shape and torch.equal(x, y)
    n += 1; same += eq
    print(("BITWISE-EQUAL " if eq else "DIFFERENT     ") + f + ("" if eq else f" maxabs={(x-y).abs().max().item() if x.shape==y.shape else 'shape'}"))
print(f"SUMMARY {same}/{n} bitwise equal")
