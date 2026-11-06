#!/usr/bin/env python3
"""decode_check2.py: the decode commit (6dad5fec638) against the route-decode A/B's decode-on images. Every
fabric_erisc_router build the A/B's decode-on mock captures logged (dec_on/port/<config>, the port at 22cdb1e55c3 with
the decode behind TT_FABRIC_EXP_DECODE_BOTH, the define set) is rebuilt from its own logged compile and link commands
against the bisect worktree at 6dad5fec638 (the logged tree root swapped, the PCH include dropped, generated headers
copied as cov.py's alt-tree builds do), then its image (loadable bytes) is compared with the capture's own JIT ELF.
Expected: identical except where 22cdb1e55c3 and e5b75d1fbcd differ outside the decode (the cleanup batch: VC0-crossover
receivers, layout); for those, the decode site's function bodies are compared."""
import collections, concurrent.futures, glob, os, re, sys

S = "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad"
os.environ["TT_COV_DIR"] = "/home/aagarwal/.tt-metal-mainline-cov-decode-check2"
sys.path.insert(0, "/home/aagarwal/tt-metal-mainline/.scratch")
sys.path.insert(0, S)
_argv, sys.argv = sys.argv, sys.argv[:1]
import cov  # noqa: E402
import fndiff  # noqa: E402

cov.ROOT = "/home/aagarwal/tt-metal-port-slots"
cov.ALT_ROOT = "/home/aagarwal/tt-metal-mainline-bisect"
PCH = re.compile(r" -include \S+/pch\.h")
OUT = os.path.join(cov.COV, "builds", "dec_on_vs_6dad5fec638")
os.makedirs(OUT, exist_ok=True)


def jobs():
    out = []
    for log in sorted(glob.glob(f"{S}/dec_on/port/*/*/*.log")):
        cfg = log.split("/dec_on/port/")[1].split("/")[0]
        db = {}
        for line in open(log, errors="replace"):
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = cov.CMD_RE.search(line)
            if not m or "/fabric_erisc_router/" not in m.group(2):
                continue
            kind, cmd = m.group(1), m.group(2)
            if not cov.wellformed(kind, cmd):
                continue
            k = cov.build_key(cmd)
            if k is None:
                continue
            db.setdefault(k, {}).setdefault(kind, [PCH.sub("", cmd) if kind == "compile" else cmd])
        for k, v in db.items():
            if "link" in v and "compile" in v:
                out.append((cfg, k, v))
    return out


def jit_elf(cmds, key):
    lm = re.search(r"\s-o\s+(\S+\.elf)", cmds["link"][0])
    # The JIT links to a temporary name and renames it to <processor>.elf.
    return os.path.join(os.path.dirname(lm.group(1)), os.path.basename(lm.group(1)).split(".")[0] + ".elf")


def one(job):
    cfg, key, cmds = job
    wd, err = cov.build_one(cfg, key, cmds, OUT)
    if err:
        return cfg, key, None, err.splitlines()[-1][:200]
    new = glob.glob(os.path.join(wd, "*.elf"))
    old = jit_elf(cmds, key)
    if not new or not os.path.exists(old):
        return cfg, key, None, f"missing elf (new {new}, old {old})"
    return cfg, key, (cov.image(new[0]) == cov.image(old), new[0], old), None


js = jobs()
print(f"{len(js)} router builds over {len({j[0] for j in js})} configs")
with concurrent.futures.ThreadPoolExecutor(32) as ex:
    res = list(ex.map(one, js))
tab = collections.Counter()
diffs = []
for cfg, key, r, err in res:
    erisc = key.rstrip("/").split("/")[-1]
    if err:
        tab[(cfg, erisc, "ERROR")] += 1
        print("ERROR", cfg, key, err)
        continue
    same, new, old = r
    tab[(cfg, erisc, "identical" if same else "differ")] += 1
    if not same:
        diffs.append((cfg, key, new, old))
for k, n in sorted(tab.items()):
    print(f"   {k[0]:22s} {k[1]:26s} {k[2]:9s} {n}")
# Differing pairs: which functions differ (branch / call targets normalized), and whether the decode site's function
# (receiver::run / single::receiver_pass, where the decode inlines) matches apart from the rest.
short = lambda n: re.sub(r"\(.*", "", n)
fn_tab = collections.Counter()
for cfg, key, new, old in diffs:
    fa = {short(n): t for n, t in fndiff.funcs(old).items()}
    fb = {short(n): t for n, t in fndiff.funcs(new).items()}
    ch = tuple(sorted(n for n in set(fa) | set(fb) if fa.get(n) != fb.get(n)))
    fn_tab[(cfg, ch)] += 1
for (cfg, ch), n in sorted(fn_tab.items()):
    print(f"   differ {cfg}: x{n}: {'; '.join(ch)[:400]}")
