"""layoutcmp.py BASE NEW: every ELF that differs between two coverage builds, classified per function:
  identical                function bytes equal after normalizing branch/call targets (code moved)
  (a) offsets              equal after also normalizing stack/frame offsets: sp-relative memory offsets, `addi rd,sp,N`,
                           and offsets on registers holding a frame address (set from sp, tracked in order)
  (b) init stores          receiver::run only: outside the loop, the new code adds only stores into the frame and the
                           constants they store (lui/addi/li), nothing removed
  (c) frame size           counted with (a): the prologue's `addi sp,sp,-N` and its save offsets
  OTHER                    anything else: printed, the caller stops on it."""
import sys, os, re, glob, difflib, collections, concurrent.futures
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
MEM = re.compile(r"^(\w+),(-?\d+)\((\w+)\)$")

def global_frame(body):
    """Registers every definition of which, over the whole function, is a frame address (flow-insensitive fixpoint)."""
    defs = collections.defaultdict(list)
    saves = {m.group(2) for a, op, raw in body[:40] for m in [re.fullmatch(r"(ra|s\d+),(\d+)\(sp\)", raw)] if op == "sw" and m}
    for a, op, raw in body:
        m = re.fullmatch(r"(ra|s\d+),(\d+)\(sp\)", raw)
        if op == "lw" and m and m.group(2) in saves:
            continue  # the epilogue's restore of a callee-saved register
        d, u, call = L.defuse(op, raw)
        for x in d:
            defs[x].append((op, raw.split(",")))
    g = {"sp"}
    changed = True
    while changed:
        changed = False
        for r, ds in defs.items():
            if r in g:
                continue
            ok = all((op in ("addi", "mv") and len(rs) >= 2 and rs[1] in g) or
                     (op in ("add", "sh1add", "sh2add", "sh3add") and len(rs) == 3 and (rs[1] in g or rs[2] in g))
                     for op, rs in ds)
            if ok and ds:
                g.add(r)
                changed = True
    return g

def norm(body):
    gframe = global_frame(body)
    frame = set(gframe)
    out = []
    for a, op, raw in body:
        r = raw
        if op in L.BRANCHES or op in ("j", "jal"):
            r = re.sub(r"[0-9a-f]+$", "T", r)
        m = MEM.match(r)
        if m and m.group(3) in frame:
            r = f"{m.group(1)},N({m.group(3)})"
        regs = raw.split(",")
        derived = False
        if op == "addi" and len(regs) == 3 and regs[1] in frame:
            r = f"{regs[0]},{regs[1]},N"
            derived = True
        elif op in ("add", "sh1add", "sh2add", "sh3add") and len(regs) == 3:
            if op == "add":
                r = ",".join([regs[0]] + sorted(regs[1:]))
            derived = regs[1] in frame or regs[2] in frame
        elif op == "mv" and len(regs) == 2 and regs[1] in frame:
            derived = True
        d, u, call = L.defuse(op, raw)
        for x in d:
            if x not in gframe:
                frame.discard(x)
        if derived:
            frame.add(regs[0])
        out.append(f"{op} {r}")
    return out

def classify(path_b, path_n):
    fb, fn = L.functions(path_b), L.functions(path_n)
    res = {}
    for name in sorted(set(fb) | set(fn)):
        if name not in fb or name not in fn:
            res[name] = "OTHER (only in one build)"
            continue
        b, n = fb[name], fn[name]
        if [x[1:] for x in b] == [x[1:] for x in n]:
            continue
        nb, nn = norm(b), norm(n)
        if [re.sub(r"[0-9a-f]+$", "T", x) for x in nb] == [re.sub(r"[0-9a-f]+$", "T", x) for x in nn]:
            res[name] = "identical (targets)"
            continue
        if nb == nn:
            res[name] = "(a) offsets"
            continue
        if "8receiver3run" in name:
            ab = [a for a, *_ in b]; an = [a for a, *_ in n]
            sb, _ = L.cfg(b); sn, _ = L.cfg(n)
            lb, ln = L.largest_scc(ab, sb), L.largest_scc(an, sn)
            loop_b = [x for (a, *_), x in zip(b, nb) if a in lb]
            loop_n = [x for (a, *_), x in zip(n, nn) if a in ln]
            if loop_b != loop_n and len(loop_b) == len(loop_n):
                # Address arithmetic whose immediate carries a frame offset into an index register before the frame
                # base is added (`addi a5,s3,80; addi a4,sp,48; add a3,a5,a4`): equal with immediates of addi and
                # memory operands abstracted, and the changed immediates shift by a few constants (the layout).
                imm = re.compile(r"(?<=,)(-?\d+)(?=\(|$)")
                deltas, same = set(), True
                for x, y in zip(loop_b, loop_n):
                    if x == y:
                        continue
                    ox, oy = x.split(" ", 1)[0], y.split(" ", 1)[0]
                    if ox != oy or ox not in ("addi", "lw", "sw", "lbu", "sb", "lh", "lhu", "sh", "lb") or \
                            imm.sub("K", x) != imm.sub("K", y):
                        same = False
                        break
                    deltas |= {int(b_) - int(a_) for a_, b_ in zip(imm.findall(x), imm.findall(y)) if a_ != b_}
                if same:
                    res[name] = f"(a) offsets via an index register (shifts {sorted(deltas)})"
                    loop_b = loop_n  # treated as equal
            if loop_b != loop_n:
                regs_re = re.compile(r"\b(t[0-6]|s1[01]|s[0-9]|a[0-7])\b")
                mapping, ok = {}, len(loop_b) == len(loop_n)
                for x, y in zip(loop_b, loop_n) if ok else []:
                    rx, ry = regs_re.findall(x), regs_re.findall(y)
                    if regs_re.sub("R", x) != regs_re.sub("R", y) or len(rx) != len(ry):
                        ok = False
                        break
                    for u, v in zip(rx, ry):
                        if mapping.setdefault(u, v) != v:
                            ok = False
                    if not ok:
                        break
                d = [l for l in difflib.unified_diff(loop_b, loop_n, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
                abstract = lambda seq: collections.Counter(regs_re.sub("R", x) for x in seq)
                if ok and len(set(mapping.values())) == len(mapping):
                    res[name] = "OTHER (loop register renaming): " + ", ".join(f"{u}->{v}" for u, v in sorted(mapping.items()) if u != v)
                elif abstract(loop_b) == abstract(loop_n):
                    res[name] = "OTHER (loop scheduling: same instructions, registers abstracted; order or registers differ)"
                else:
                    res[name] = "OTHER (loop): " + "; ".join(d[:8])
                continue
            # Outside the loop (setup before it, exits after it): compare as multisets with register names
            # abstracted (setup code's register choices and order follow the added stores). Allowed: added stores
            # into the frame, added or shared constant materializations and register moves.
            regs_re = re.compile(r"\b(zero|ra|sp|gp|tp|t[0-6]|s1[01]|s[0-9]|a[0-7])\b")
            ab_ = lambda x: regs_re.sub("R", x)
            note = res.get(name, "")
            out_b = collections.Counter(ab_(x) for (a, *_), x in zip(b, nb) if a not in lb)
            out_n = collections.Counter(ab_(x) for (a, *_), x in zip(n, nn) if a not in ln)
            removed, added = out_b - out_n, out_n - out_b
            const = re.compile(r"^(lui|li|addi|mv) ")
            bad = ["-" + x for x in removed if not const.match(x)] + \
                  ["+" + x for x in added if not (re.match(r"^s[whb] R,N\(R\)$", x) or const.match(x))]
            res[name] = ("(b) setup: init stores" + (f"; {note}" if note else "")) if not bad else \
                "OTHER (outside loop): " + "; ".join(bad[:8])
            continue
        d = [l for l in difflib.unified_diff(nb, nn, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        res[name] = "OTHER: " + "; ".join(d[:6])
    return res

def job(rel):
    return rel, classify(f"{BASE}/{rel}", f"{NEW}/{rel}")

if __name__ == "__main__":
    BASE, NEW = f"{CB}/{sys.argv[1]}", f"{CB}/{sys.argv[2]}"
    import hashlib, subprocess
    def h(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    rels = []
    for p in glob.glob(f"{NEW}/*/*/*.elf"):
        rel = os.path.relpath(p, NEW)
        q = f"{BASE}/{rel}"
        if os.path.exists(q):
            a = subprocess.run([L.B + "objcopy", "-O", "binary", "-j", ".text", p, "/dev/stdout"], capture_output=True).stdout
            b = subprocess.run([L.B + "objcopy", "-O", "binary", "-j", ".text", q, "/dev/stdout"], capture_output=True).stdout
            if a != b:
                rels.append(rel)
    with concurrent.futures.ProcessPoolExecutor(16) as ex:
        res = list(ex.map(job, rels))
    by = collections.Counter(); other = []
    for rel, r in res:
        for name, c in r.items():
            by[(L.base(re.sub(r"^_Z\w*?(receiver|sender|single)", r"\1", name))[:40], c.split(":")[0])] += 1
            if c.startswith("OTHER"):
                other.append((rel, name, c))
    total = len(glob.glob(f"{NEW}/*/*/*.elf"))
    print(f"{sys.argv[1]} -> {sys.argv[2]}: {len(rels)} of {total} images differ in .text")
    for (f, c), n in sorted(by.items()):
        print(f"  {n:5d}  {c:22s} {f}")
    print(f"OTHER: {len(other)}")
    for rel, name, c in other[:int(os.environ.get("SHOW", "20"))]:
        print("  ", rel, name[:60], c[:300])
    sys.exit(1 if other else 0)
