#!/usr/bin/env python3
"""fnsplit.py BUILD_A BUILD_B LIST [--show N] [--renames FILE]: per image in LIST (relative ELF paths, one per line), the functions whose
code differs between two cov_m builds, after normalising branch and call targets to symbol+offset (code that moved is
equal). Prints, per set of differing functions, how many images have it. With --show N, prints the normalised diff of
the first N images' differing functions. Symbols present in one build only are listed as +name / -name. --renames
FILE: lines "REGEX<TAB>REPLACEMENT", applied (in order) to every function name and instruction of both builds, so a
renamed function and calls to it compare equal."""
import collections, concurrent.futures as cf, difflib, re, subprocess, sys
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"


RENAMES = []


def canon(x):
    for r, n in RENAMES:
        x = r.sub(n, x)
    return x


def funcs(elf):
    out = subprocess.run([B + "objdump", "-d", "-C", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout
    res, cur = {}, None
    for l in out.splitlines():
        m = re.match(r"^([0-9a-f]+) <(.*)>:$", l)
        if m:
            if m.group(2).startswith(".L"):
                continue
            cur = canon(m.group(2))
            res[cur] = []
            continue
        m = re.match(r"^\s+[0-9a-f]+:\s+(\S+)\s*(.*)$", l)
        if m and cur:
            ops = re.sub(r"\s#.*", "", m.group(2))  # objdump's comments (a lambda's name holds '#' without a space)
            ops = re.sub(r"\b[0-9a-f]{3,8} <(.*)>\s*$", r"<\1>", ops)  # targets by symbol (names may hold '>')
            ops = re.sub(r"<\.L[^>]*>", "<L>", ops)
            res[cur].append(canon(m.group(1) + " " + ops.strip()))
    return res


def one(rel):
    a, b = funcs(f"{CB}/{A}/{rel}"), funcs(f"{CB}/{Bn}/{rel}")
    ch = []
    for n in sorted(set(a) | set(b)):
        if n not in a:
            ch.append("+" + n)
        elif n not in b:
            ch.append("-" + n)
        elif a[n] != b[n]:
            ch.append(n)
    return rel, tuple(ch), a, b


A, Bn, lst = sys.argv[1:4]
if "--renames" in sys.argv:
    for l in open(sys.argv[sys.argv.index("--renames") + 1]).read().splitlines():
        if l.strip() and not l.startswith("#"):
            r, n = l.split("\t")
            RENAMES.append((re.compile(r), n))
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
rels = open(lst).read().split()
cnt = collections.Counter()
shown = 0
with cf.ProcessPoolExecutor(16) as ex:
    for rel, ch, a, b in ex.map(one, rels):
        cnt[ch] += 1
        if shown < show and ch:
            shown += 1
            print(f"=== {rel}")
            for n in ch:
                if n[0] in "+-":
                    print("   ", n)
                    continue
                print(f"--- {n}")
                for d in difflib.unified_diff(a[n], b[n], lineterm="", n=1):
                    print("   ", d)
print(f"{len(rels)} images")
for ch, n in cnt.most_common():
    names = "; ".join(re.sub(r"\(.*", "", c) for c in ch) or "(no function differs)"
    print(f"{n:5d}  {names}")
