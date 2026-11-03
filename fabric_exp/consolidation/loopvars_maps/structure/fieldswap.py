#!/usr/bin/env python3
"""fieldswap.py BASE NEW: classify .text differences that come from swapping two 4-byte struct fields (v4rxg2's
DownstreamSenderChannelProgress {next_slot_addr, free_slots} -> {free_slots, next_slot_addr}). Per differing function
(branch / call targets normalized as layoutcmp.py does):
  swapped offsets   same instructions in the same order, differing only in memory offsets, each changed one moved by
                    4 bytes on the same base register (one field's offset to the other's);
  swapped, reordered  the same once offsets are mapped that way, as a multiset (stores issued in address order);
  OTHER             anything else (printed)."""
import collections, glob, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layoutcmp as LC
L = LC.L
CB = LC.CB
OFF = re.compile(r"(-?\d+)\((\w+)\)")


def tgt(seq):
    return [re.sub(r"[0-9a-f]+$", "T", x) for x in seq]


def classify(fb, fn):
    res = {}
    for name in sorted(set(fb) | set(fn)):
        if name not in fb or name not in fn:
            res[name] = "OTHER (only in one build)"
            continue
        b, n = tgt(LC.norm(fb[name])), tgt(LC.norm(fn[name]))
        if b == n:
            continue
        mapping, ok = {}, len(b) == len(n)
        for x, y in zip(b, n) if ok else []:
            if OFF.sub("M", x) != OFF.sub("M", y):
                ok = False
                break
            for (ox, rx), (oy, ry) in zip(OFF.findall(x), OFF.findall(y)):
                ox, oy = int(ox), int(oy)
                # per instruction: the same base register, the offset unchanged or moved by one 4-byte field (a
                # register can reach other objects elsewhere in the function, so no function-wide mapping is required)
                if rx != ry or (ox != oy and abs(ox - oy) != 4):
                    ok = False
                elif ox != oy:
                    mapping[(rx, ox)] = oy
        if ok:
            res[name] = "swapped offsets " + ",".join(f"{a[1]}:{a[0]}->{c}" for a, c in sorted(mapping.items()) if a[0] != c)
            continue
        # As a multiset, with offsets mapped by every x <-> x+4 swap the aligned part shows.
        cb = collections.Counter(OFF.sub("M", x) for x in b)
        cn = collections.Counter(OFF.sub("M", x) for x in n)
        if cb == cn:
            ob = collections.Counter(int(o) for x in b for o, _ in OFF.findall(x))
            on = collections.Counter(int(o) for x in n for o, _ in OFF.findall(x))
            if ob == on:
                res[name] = "swapped, reordered (same offsets, other order)"
                continue
        d = [l for l in __import__("difflib").unified_diff(b, n, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        res[name] = "OTHER: " + "; ".join(d[:6])
    return res


def job(rel):
    return rel, classify(L.functions(f"{CB}/{sys.argv[1]}/{rel}"), L.functions(f"{CB}/{sys.argv[2]}/{rel}"))


if __name__ == "__main__":
    import concurrent.futures
    rels = []
    for p in glob.glob(f"{CB}/{sys.argv[2]}/*/*/*.elf"):
        rel = os.path.relpath(p, f"{CB}/{sys.argv[2]}")
        q = f"{CB}/{sys.argv[1]}/{rel}"
        if os.path.exists(q):
            t = lambda f: subprocess.run([L.B + "objcopy", "-O", "binary", "-j", ".text", f, "/dev/stdout"], capture_output=True).stdout
            if t(p) != t(q):
                rels.append(rel)
    with concurrent.futures.ProcessPoolExecutor(16) as ex:
        res = list(ex.map(job, rels))
    by = collections.Counter(); other = []
    for rel, r in res:
        for name, c in r.items():
            by[(L.base(re.sub(r"^_Z\w*?(receiver|sender|single|fabric_router)", r"\1", name))[:40], re.sub(r" -?\d.*", "", c.split(":")[0]))] += 1
            if c.startswith("OTHER"):
                other.append((rel, name, c))
    print(f"{sys.argv[1]} -> {sys.argv[2]}: {len(rels)} images differ in .text")
    for (f, c), k in sorted(by.items()):
        print(f"  {k:5d}  {c:40s} {f}")
    print(f"OTHER: {len(other)}")
    for rel, name, c in other[:int(os.environ.get("SHOW", "20"))]:
        print("  ", rel, name[:60], c[:300])
    sys.exit(1 if other else 0)
