#!/usr/bin/env python3
"""img_check.py V: the slack study's offline image check (offline_check.sh's mock builds under V/<router>_<arm>/<test>/cache).

Per router (main, port), test and patched arm (z, tx<D>, rx<D>), each router image paired with the plain root's by its
named compile-time args (unique per image) and processor (active_erisc = E0, subordinate_active_erisc = E1):
  - loadable bytes (PT_LOAD segments) identical or not;
  - for pairs that differ, the functions whose code differs with addresses and immediates masked (objdump), with the
    instruction count change, and functions only in one image;
  - loadable sizes: the largest E0 and E1 image of the arm, and their sum against the 26,624 B ERISC kernel config buffer
    (MEM_ERISC_KERNEL_CONFIG_SIZE, an upper bound per router as size_report.py's).
Expected: z identical to plain everywhere; tx<D> changes only E0 images, rx<D> only E1 images, each in the function(s)
holding the send / receive step and nothing else. Exit 1 when an expectation fails."""
import collections, concurrent.futures, glob, hashlib, os, re, struct, subprocess, sys

V = sys.argv[1]
OBJDUMP = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
KCFG = 26 * 1024
PROCS = {"active_erisc": "E0", "subordinate_active_erisc": "E1"}
MASK = re.compile(r"-?\b0x[0-9a-f]+\b|-?\b[0-9a-f]{4,}\b|-?\b\d+\b")


def load(path):
    b = open(path, "rb").read()
    phoff, = struct.unpack_from("<I", b, 28)
    pe, pn = struct.unpack_from("<HH", b, 42)
    h, size = hashlib.sha256(), 0
    for i in range(pn):
        t, off, va, _pa, fsz, msz, _fl, _al = struct.unpack_from("<8I", b, phoff + i * pe)
        if t == 1:
            h.update(struct.pack("<III", va, fsz, msz)); h.update(b[off:off + fsz]); size += fsz
    return h.hexdigest(), size


def builds(cache):
    out = {}
    for d in glob.glob(f"{cache}/*/kernels/fabric_erisc_router/*/"):
        m = os.path.join(d, "named_ct_arg_map_generated.h")
        if not os.path.exists(m):
            continue
        key = hashlib.md5(open(m).read().encode()).hexdigest()
        for proc in PROCS:
            elfs = [p for p in glob.glob(os.path.join(d, proc, "*.elf")) if not p.endswith(".xip.elf")]
            if elfs:
                assert (proc, key) not in out, (cache, proc, key)
                out[(proc, key)] = elfs[0]
    return out


_fn_cache = {}


def functions(elf):
    if elf in _fn_cache:
        return _fn_cache[elf]
    txt = subprocess.run([OBJDUMP, "-d", "-C", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout
    fns, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            if not m.group(1).startswith(".L"):
                cur = m.group(1); fns.setdefault(cur, [])
            continue
        m = re.match(r"\s+[0-9a-f]+:\s+(.*)$", line)
        if m and cur:
            fns[cur].append(MASK.sub("N", re.sub(r" <[^>]*>|#.*$", "", m.group(1))).strip())
    _fn_cache[elf] = {k: tuple(v) for k, v in fns.items()}
    return _fn_cache[elf]


def short(n):
    n = re.sub(r"\(.*", "", n)
    return n if len(n) <= 70 else n[:67] + "..."


def fdiff(a, b):
    fa, fb = functions(a), functions(b)
    d = [(short(k), len(fb[k]) - len(fa[k])) for k in set(fa) & set(fb) if fa[k] != fb[k]]
    return sorted(d), sorted(short(k) for k in set(fa) - set(fb)), sorted(short(k) for k in set(fb) - set(fa))


def main():
    ok = True
    core = lambda a: a[1:] if a.startswith("m") else a  # mcycle arms: mz, mtx<D>, mrx<D>
    arms = sorted({os.path.basename(p).split("_", 1)[1] for p in glob.glob(f"{V}/*_*") if os.path.isdir(p)},
                  key=lambda a: (a.startswith("m"), core(a) != "plain", core(a) != "z", core(a)[:2],
                                 int(core(a)[2:]) if core(a)[2:].isdigit() else 0))
    tests = sorted({os.path.basename(p) for p in glob.glob(f"{V}/*_plain/*") if os.path.isdir(p)})
    for router in ("main", "port"):
        for test in tests:
            plain = builds(f"{V}/{router}_plain/{test}/cache")
            if not plain:
                continue
            psz = {k: load(e)[1] for k, e in plain.items()}
            pmax = {p: max(s for (q, _), s in psz.items() if q == p) for p in PROCS}
            print(f"## {router} {test}: {len(plain)} router images; plain largest E0 {pmax['active_erisc']} B, "
                  f"E1 {pmax['subordinate_active_erisc']} B, sum {pmax['active_erisc'] + pmax['subordinate_active_erisc']}"
                  f" of {KCFG}")
            for arm in arms:
                if arm == "plain" or not os.path.isdir(f"{V}/{router}_{arm}/{test}/cache"):
                    continue
                other = builds(f"{V}/{router}_{arm}/{test}/cache")
                if set(other) != set(plain):
                    print(f"  {arm}: !!! image sets differ ({len(other)} vs {len(plain)}; unpaired "
                          f"{len(set(other) ^ set(plain))})"); ok = False; continue
                per = collections.defaultdict(lambda: [0, 0, collections.Counter(), set(), 0])
                jobs = {}
                with concurrent.futures.ThreadPoolExecutor(16) as ex:
                    for k in plain:
                        ha, sa = load(plain[k]); hb, sb = load(other[k])
                        p = per[k[0]]
                        p[0] += 1; p[4] = max(p[4], sb)
                        if ha != hb:
                            p[1] += 1
                            jobs[k] = ex.submit(fdiff, plain[k], other[k])
                    for k, f in jobs.items():
                        d, oa, ob = f.result()
                        per[k[0]][2].update(f"{n} ({c:+d} insns)" for n, c in d)
                        per[k[0]][3].update([f"only plain: {n}" for n in oa] + [f"only {arm}: {n}" for n in ob])
                c = core(arm)
                want_changed = {"z": set(), "tx": {"active_erisc"}, "rx": {"subordinate_active_erisc"}}[c[:2] if c != "z" else "z"]
                line = []
                for proc in PROCS:
                    n, nd, fns, extra, mx = per[proc]
                    exp_ok = (nd == 0) if proc not in want_changed else (nd == n and not extra)
                    ok &= exp_ok
                    s = f"{PROCS[proc]} {nd}/{n} differ"
                    if nd:
                        s += ": " + "; ".join(f"{f} x{c}" for f, c in sorted(fns.items()))
                    if extra:
                        s += " | " + "; ".join(sorted(extra))
                    s += f", largest {mx} B" + ("" if exp_ok else "  !!! unexpected")
                    line.append(s)
                tot = per["active_erisc"][4] + per["subordinate_active_erisc"][4]
                fit = tot <= KCFG
                ok &= fit
                print(f"  {arm:6s} " + " || ".join(line) + f" || E0+E1 {tot} B {'fits' if fit else '!!! OVER'}")
    print("IMG_CHECK_OK" if ok else "IMG_CHECK_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
