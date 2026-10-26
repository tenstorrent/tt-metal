#!/usr/bin/env python3
"""fncmp.py A.elf B.elf [--fn REGEX] [--show NAME]: compare the code of the functions two ELFs have in common.

Each function's instructions are normalized before comparing: branch and jump targets inside the function become
offsets from its start, calls become the callee's name (clone suffixes such as .isra.0/.constprop.0 kept), and
objdump's comments are dropped. So a function that only moved compares equal; its start address is reported apart.
Prints one line per function: same / moved / DIFF (instructions, differing lines) / only-A / only-B.
For _start (kernel_main with the inlined main loop) it also compares the receiver main loop region alone: the
strongly connected part of the control-flow graph that contains the inner loop's back edge."""
import difflib, re, subprocess, sys

OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
CB = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez", "blez", "bgtz"}


import functools


@functools.lru_cache(maxsize=None)
def base_name(sym):
    """A callee's name without template arguments, parameters and clone suffixes, so that a helper that only changed
    from a template to a plain function (or got another clone number) compares equal."""
    d = subprocess.run(["c++filt", sym], capture_output=True, text=True).stdout.strip()
    clone = ".".join(p for p in d.split(" ")[-1].split(".")[1:] if not p.isdigit()) if "." in d else ""
    out, depth = "", 0
    for ch in d:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif depth == 0:
            out += ch
    return out.split(" ")[-1].split(".")[0] + (f".{clone}" if clone else "")


def funcs(elf):
    syms = {}
    for l in subprocess.run([OD, "-t", elf], capture_output=True, text=True).stdout.splitlines():
        p = l.split()
        if len(p) >= 6 and p[2] == "F" and p[3] == ".text":
            syms[p[-1]] = (int(p[0], 16), int(p[4], 16))
    return syms


def disasm(elf, lo, hi):
    out = subprocess.run(
        [OD, "-d", "--no-show-raw-insn", f"--start-address={lo:#x}", f"--stop-address={hi:#x}", elf],
        capture_output=True,
        text=True,
    ).stdout
    ins = []
    for l in out.splitlines():
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", l)
        if m:
            arg = re.sub(r"\s*#.*$", "", m.group(3))
            arg = re.sub(r"\s*<[^>]*>", "", arg)
            ins.append((int(m.group(1), 16), m.group(2), arg.strip()))
    return ins


def normalize(ins, start, size, names):
    out = []
    for ad, op, arg in ins:
        parts = arg.split(",")
        if op in CB or op in ("j", "jal"):
            try:
                t = int(parts[-1], 16)
            except ValueError:
                out.append(f"{op} {arg}")
                continue
            if start <= t < start + size:
                tgt = f"+{t - start:#x}"
            else:
                tgt = base_name(names[t]) if t in names else "ext"
            out.append(f"{op} {','.join(parts[:-1] + [tgt])}")
        else:
            out.append(f"{op} {arg}")
    return out


def loop_region(ins, start, size):
    """Instructions of the largest strongly connected region (by instruction count) of the function's CFG."""
    addrs = [a for a, _, _ in ins]
    idx = {a: i for i, a in enumerate(addrs)}
    succ = {}
    for i, (a, op, arg) in enumerate(ins):
        nxt = addrs[i + 1] if i + 1 < len(addrs) else None
        s = []
        if op in CB:
            t = int(arg.split(",")[-1], 16)
            s = [t, nxt]
        elif op == "j":
            s = [int(arg, 16)]
        elif op in ("ret", "jr") or (op == "jalr" and arg.startswith("zero")):
            s = []
        else:
            s = [nxt]
        succ[a] = [x for x in s if x is not None and x in idx]
    # Tarjan (iterative)
    index = {}
    low = {}
    on = set()
    stack = []
    sccs = []
    counter = [0]
    for root in addrs:
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work.pop()
            if pi == 0:
                index[v] = low[v] = counter[0]
                counter[0] += 1
                stack.append(v)
                on.add(v)
            recurse = False
            ss = succ[v]
            for k in range(pi, len(ss)):
                w = ss[k]
                if w not in index:
                    work.append((v, k + 1))
                    work.append((w, 0))
                    recurse = True
                    break
                elif w in on:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    big = max(sccs, key=len)
    keep = set(big)
    return [x for x in ins if x[0] in keep]


def main():
    a, b = sys.argv[1], sys.argv[2]
    fre = None
    show = None
    if "--fn" in sys.argv:
        fre = re.compile(sys.argv[sys.argv.index("--fn") + 1])
    if "--show" in sys.argv:
        show = sys.argv[sys.argv.index("--show") + 1]
    fa, fb = funcs(a), funcs(b)
    na = {v[0]: k for k, v in fa.items()}
    nb = {v[0]: k for k, v in fb.items()}
    names = sorted(set(fa) | set(fb), key=lambda n: fb.get(n, fa.get(n))[0])
    for n in names:
        if fre and not fre.search(n):
            continue
        if n not in fb:
            print(f"only-A  {n}")
            continue
        if n not in fa:
            print(f"only-B  {n}")
            continue
        (sa, za), (sb, zb) = fa[n], fb[n]
        ia, ib = disasm(a, sa, sa + za), disasm(b, sb, sb + zb)
        xa, xb = normalize(ia, sa, za, na), normalize(ib, sb, zb, nb)
        tag = "same" if xa == xb and sa == sb else ("moved" if xa == xb else "DIFF")
        extra = ""
        if tag == "DIFF":
            d = sum(1 for l in difflib.unified_diff(xa, xb, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---"))
            extra = f" ({len(xa)} -> {len(xb)} instructions, {d} diff lines)"
        if tag == "moved":
            extra = f" ({sa:#x} -> {sb:#x})"
        line = f"{tag:6s} {n}{extra}"
        print(line)
        if show == n and tag == "DIFF":
            for l in difflib.unified_diff(xa, xb, lineterm="", n=2):
                print("   ", l)
    # The main loop: the largest strongly connected region of any function (receiver_erisc's loop, wherever inlined).
    def biggest(elf, fs, names_):
        best = None
        for n, (st, sz) in fs.items():
            if sz < 400:
                continue
            ins = disasm(elf, st, st + sz)
            reg = loop_region(ins, st, sz)
            if best is None or len(reg) > len(best[1]):
                best = (n, reg, normalize(reg, st, sz, names_))
        return best
    ba, bb = biggest(a, fa, na), biggest(b, fb, nb)
    ra_, rb_ = _rebase(ba[1], ba[2]), _rebase(bb[1], bb[2])
    print(f"main-loop region: {ba[0]} -> {bb[0]}, {len(ra_)} -> {len(rb_)} instructions: {'same' if ra_ == rb_ else 'DIFF'}"
          f", starts {ba[1][0][0]:#x} -> {bb[1][0][0]:#x} (mod 64: {ba[1][0][0] % 64} -> {bb[1][0][0] % 64})")
    if ra_ != rb_ and "--loopdiff" in sys.argv:
        for l in difflib.unified_diff(ra_, rb_, lineterm="", n=2):
            print("   ", l)


def _rebase(region, norm):
    """Branch targets of a region listing as offsets from the region's first instruction (function-relative offsets
    shift when code before the region changes size)."""
    base_off = None
    out = []
    first = region[0][0]
    # function start = first - (offset of first); recover from the normalized form is not possible, so use addresses
    for (ad, op, arg), x in zip(region, norm):
        m = re.search(r"\+0x([0-9a-f]+)$", x)
        if m:
            # function-relative target; express it relative to the region start
            parts = arg.split(",")
            try:
                t = int(parts[-1], 16)
                x = x[: m.start()] + f"@{t - first:#x}"
            except ValueError:
                pass
        out.append(x)
    return out


if __name__ == "__main__":
    main()
