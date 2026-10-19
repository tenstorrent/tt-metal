"""ELF guard for the tilize all-configuration A/B: per side build dir, a hash of each thread's loaded ELF sections per
variant dir (same params hash the same dir name on every side). GUARD ok when the A/A copy (M2) matches M on every
variant and each head side differs from M on at least one variant; prints per variant which threads differ.
usage: elfguard.py M=<dir> F=<dir> B=<dir> M2=<dir> [--out file]"""
import glob, hashlib, os, re, struct, sys


def image(p):
    d = open(p, "rb").read()
    shoff, = struct.unpack_from("<I", d, 0x20)
    shentsize, shnum, _ = struct.unpack_from("<HHH", d, 0x2E)
    h = hashlib.sha1()
    for i in range(shnum):
        nm, typ, flags, addr, off, size = struct.unpack_from("<IIIIII", d, shoff + i * shentsize)
        if flags & 2:
            h.update(struct.pack("<II", addr, size))
            if typ != 8:
                h.update(d[off:off + size])
    return h.hexdigest()[:10]


args = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
must = sys.argv[sys.argv.index("--must") + 1].split(",") if "--must" in sys.argv else None
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
sides = dict(a.split("=", 1) for a in args)
H = {}
for s, b in sides.items():
    for vd in glob.glob(f"{b}/tt-llk-build/sources/*/*/"):
        if not os.path.exists(f"{vd}/elf/pack.elf"):
            continue
        key = "/".join(vd.rstrip("/").split("/")[-2:])
        H[(s, key)] = tuple(image(f"{vd}/elf/{t}.elf") if os.path.exists(f"{vd}/elf/{t}.elf") else "-" for t in ("unpack", "math", "pack"))
keys = sorted({k for _, k in H})
lines, aa_bad, diff = [], 0, {s: 0 for s in sides if s not in ("M", "M2")}
for k in keys:
    m = H.get(("M", k))
    row = [k]
    if ("M2", k) in H and H[("M2", k)] != m:
        aa_bad += 1
        row.append("AA_DIFF")
    for s in diff:
        h = H.get((s, k))
        if h is None or m is None:
            row.append(f"{s}:absent")
            continue
        d = "".join(t for t, a, b in zip("UMP", m, h) if a != b) or "="
        diff[s] += d != "="
        row.append(f"{s}:{d}")
    lines.append(" ".join(row))
if out:
    open(out, "w").write("\n".join(lines) + "\n")
ok = aa_bad == 0 and all(v > 0 for s, v in diff.items() if must is None or s in must) and keys
print(f"GUARD {'ok' if ok else 'FAIL'} variants={len(keys)} aa_diff={aa_bad} " + " ".join(f"{s}_differs={v}" for s, v in diff.items()))
sys.exit(0 if ok else 1)
