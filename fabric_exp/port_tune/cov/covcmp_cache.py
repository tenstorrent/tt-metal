#!/usr/bin/env python3
"""covcmp_cache.py COV_A COV_B CASE [CASE ...] [--all]: per case, the multiset of loadable-byte digests of every router
ELF the JIT itself built during the mock (COV/cases/CASE/cache), independent of the logged commands (multi-rank tt-run
logs can interleave and drop commands from cmds.json). --all: every kernel ELF, not routers only."""
import collections, hashlib, importlib.util, os, subprocess, sys
args = [a for a in sys.argv[1:] if not a.startswith("--")]
everything = "--all" in sys.argv
A, B, cases = args[0], args[1], args[2:]
os.environ.setdefault("TT_COV_TREE", "/home/aagarwal/tt-metal-gate-tip")
_argv, sys.argv = sys.argv, ["cov"]
_spec = importlib.util.spec_from_file_location("cov", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cov.py"))
_cov = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cov)
sys.argv = _argv
image = _cov.image  # per-section loadable bytes (WH firmware-style ELFs place sections far apart)
import concurrent.futures
POOL = concurrent.futures.ThreadPoolExecutor(int(os.environ.get("CMP_JOBS", "16")))
def ms(cov, case):
    """Counter of (kernel name, image digest). Only each build's final ELF (<proc>/<proc>.elf): skips .xip.elf copies
    and link temporaries (<proc>.<hash>_<pid>.elf)."""
    elfs = []
    for dp, _, fs in os.walk(os.path.join(cov, "cases", case, "cache")):
        if "/kernels/" not in dp + "/" or (not everything and "fabric_erisc_router" not in dp):
            continue
        elfs += [os.path.join(dp, f) for f in fs if f == os.path.basename(dp) + ".elf"]
    kname = lambda e: e.split("/kernels/")[1].split("/")[0]
    return collections.Counter(POOL.map(lambda e: (kname(e), hashlib.md5(image(e)).hexdigest()), elfs))
for case in cases:
    a, b = ms(A, case), ms(B, case)
    print(f"{case:24s} {'IDENTICAL' if a == b else 'DIFFER':9s} elfs {sum(a.values())}/{sum(b.values())} distinct "
          f"{len(a)}/{len(b)}" + ("" if a == b else f" only in A {sum((a - b).values())}, only in B {sum((b - a).values())}"))
    if a != b:
        ka, kb = collections.Counter(k for k, _ in (a - b).elements()), collections.Counter(k for k, _ in (b - a).elements())
        for k in sorted(set(ka) | set(kb)):
            print(f"     {k}: only in A {ka[k]}, only in B {kb[k]}")
