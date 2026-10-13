# Scratch (not for merge): allpat_cmp.py <x.pt> <y.pt> <x> <y>: per case, rows whose outputs differ and what differs.
import sys
import torch

X, Y = torch.load(sys.argv[1]), torch.load(sys.argv[2])
nx, ny = sys.argv[3], sys.argv[4]


def bits(v):
    return v.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF


def cls(b):
    e = (b >> 7) & 0xFF
    m = b & 0x7F
    if e == 0xFF and m:
        return "nan"
    if e == 0 and m == 0:
        return "zero"
    if e == 0:
        return "denorm"
    if e == 0xFF:
        return "inf"
    return "normal"


tot = {"same": 0, "diff": 0}
for name in sorted(X):
    vx, ix = X[name]
    vy, iy = Y[name]
    bx, by = bits(vx).reshape(32, -1), bits(vy).reshape(32, -1)
    jx, jy = ix.to(torch.int64).reshape(32, -1), iy.to(torch.int64).reshape(32, -1)
    rows = [r for r in range(32) if not (torch.equal(bx[r], by[r]) and torch.equal(jx[r], jy[r]))]
    if not rows:
        tot["same"] += 1
        continue
    tot["diff"] += 1
    kinds = set()
    for r in rows:
        dv = (bx[r] != by[r]).nonzero().flatten().tolist()
        di = (jx[r] != jy[r]).nonzero().flatten().tolist()
        setx, sety = sorted(bx[r].tolist()), sorted(by[r].tolist())
        sel = "same set" if setx == sety else "other set"
        cx = sorted({cls(bx[r][p].item()) for p in dv} | {cls(by[r][p].item()) for p in dv})
        kinds.add(f"row {r}: {len(dv)} values ({','.join(cx) or '-'}), {len(di)} indices, {sel}")
    print("ALLPAT", nx, ny, name, "rows", rows[:8], "|", " ; ".join(sorted(kinds))[:600])
print("ALLPAT TOTAL", nx, ny, tot)
