"""diff_src.py A B [PER_CASE]: for differing images between builds A and B (cov_m/builds; bytecmp's notion: the .elf's
loadable sections), diff the disassembly with addresses and branch targets stripped and map each changed instruction in
B to its innermost source line (addr2line). Prints per case the set of source lines touched, over up to PER_CASE
differing images per case (default 1; 0 = all)."""
import os, re, subprocess, sys, collections, difflib
from concurrent.futures import ThreadPoolExecutor
B = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
BIN = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
a, b = sys.argv[1], sys.argv[2]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 1
def elfs(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".elf"):
                out[os.path.relpath(os.path.join(d, f), root)] = os.path.join(d, f)
    return out
def image(p):
    r = subprocess.run([BIN + "objcopy", "-O", "binary", "--only-section=.text", p, "/dev/stdout"], capture_output=True)
    return r.stdout
def dis(p):
    t = subprocess.run([BIN + "objdump", "-d", "--no-show-raw-insn", p], capture_output=True, text=True).stdout
    out = []
    for l in t.splitlines():
        m = re.match(r"^\s*([0-9a-f]+):\s+(.*)$", l)
        if not m: continue
        ins = re.sub(r"<[^>]*>", "", m.group(2)).strip()
        ins = re.sub(r"\b[0-9a-f]{4,}\b", "ADDR", ins)
        out.append((int(m.group(1), 16), ins))
    return out
ea, eb = elfs(f"{B}/{a}"), elfs(f"{B}/{b}")
keys = sorted(set(ea) & set(eb))
bycase = collections.defaultdict(list)
for k in keys: bycase[k.split("__")[0]].append(k)
def work(k):
    if image(ea[k]) == image(eb[k]): return k, None
    da, db = dis(ea[k]), dis(eb[k])
    sm = difflib.SequenceMatcher(None, [x[1] for x in da], [x[1] for x in db], autojunk=False)
    addrs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            addrs += [db[j][0] for j in range(j1, j2)] or ([db[j1][0]] if j1 < len(db) else [])
    if not addrs: return k, {"(no insn change)"}
    r = subprocess.run([BIN + "addr2line", "-e", eb[k]] + [hex(x) for x in addrs[:200]], capture_output=True, text=True).stdout
    lines = set(re.sub(r"^.*/(overlay|tt-metal[^/]*)/", "", l).split(" ")[0] for l in r.splitlines())
    return k, lines
todo = []
for c, ks in bycase.items():
    todo += ks if per == 0 else ks  # check all, report per case
res = collections.defaultdict(lambda: [0, 0, collections.Counter()])
with ThreadPoolExecutor(24) as ex:
    for k, lines in ex.map(work, todo if per == 0 else [k for c in bycase for k in bycase[c]]):
        c = k.split("__")[0]
        res[c][0] += 1
        if lines is not None:
            res[c][1] += 1
            for l in lines: res[c][2][l] += 1
allsrc = collections.Counter()
for c in sorted(res):
    n, d, cnt = res[c]
    allsrc.update(cnt)
    print(f"{c:30s} images {n:4d} differ {d:4d}  " + "; ".join(f"{l} x{v}" for l, v in cnt.most_common(4)))
print("ALL source lines touched:", dict(allsrc))
