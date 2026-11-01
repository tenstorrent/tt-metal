#!/usr/bin/env python3
"""rxcmp.py BASE NEW [--cases=a,b] [--list]: two coverage builds (COV/builds/<name>), per router ELF they share.

Per ELF:
  class     same (.text byte-identical), layout (every function's instructions equal as a multiset once branch and
            call targets are stripped and commutative operands sorted: blocks moved or re-laid out), rename (the same
            with register names also stripped; sp kept), differ
  receiver  the receiver loop function (two ERISCs: receiver::run, "receiver3run"; one ERISC: single::receiver_pass,
            "receiver_pass"): its instructions and sp-relative loads and stores, the prologue's and epilogues'
            callee-saved register saves and restores not counted; and the same inside its loop (the largest strongly
            connected region of its control-flow graph: the outer loop with the passes, without the setup before it)
Prints per case: the class counts, the base build's receiver range, and for the ELFs whose receiver function differs
the distribution of (instruction delta, stack-access delta) of the whole function as "dI/dS xcount" and of its loop
as "loop dI/dS". --list prints every differing ELF."""
import collections, concurrent.futures, glob, os, re, subprocess, sys

COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m") + "/builds"
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
RX_KEYS = ("receiver3run", "receiver_pass")
COMMUTATIVE = {"add", "and", "or", "xor", "mul", "minu", "maxu", "min", "max"}
CB = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez", "blez", "bgtz"}


def text(e):
    return subprocess.run([B + "objcopy", "-O", "binary", "--only-section=.text", e, "/dev/stdout"],
                          capture_output=True).stdout


def functions(e):
    """{name without clone suffix: [(address, mnemonic, normalized operands, raw operands)]}"""
    syms = []
    for l in subprocess.run([B + "nm", "-S", e], capture_output=True, text=True).stdout.splitlines():
        p = l.split()
        if len(p) == 4 and p[2] in "tT":
            syms.append((int(p[0], 16), int(p[1], 16), p[3]))
    d = subprocess.run([B + "objdump", "-d", "--no-show-raw-insn", "--section=.text", e], capture_output=True,
                       text=True).stdout
    ins = []
    for l in d.splitlines():
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", l)
        if m:
            raw = re.sub(r"\s*#.*$", "", m.group(3))
            raw = re.sub(r"\s*<[^>]*>", "", raw).strip()
            op, arg = m.group(2), raw
            if op in CB or op in ("j", "jal"):
                arg = ",".join(arg.split(",")[:-1])
            elif op in COMMUTATIVE:
                dst, *src = arg.split(",")
                arg = ",".join([dst] + sorted(src))
            ins.append((int(m.group(1), 16), op, arg, raw))
    out = {}
    for s, sz, name in syms:
        out[re.sub(r"\.(isra|constprop|part)\.\d+", "", name)] = [x for x in ins if s <= x[0] < s + sz]
    return out


REG = re.compile(r"\b(a[0-7]|t[0-6]|s[0-9]|s1[01]|ra)\b")
SAVED = re.compile(r"^(ra|s\d+|fp),")


def frame_saves(body):
    """Addresses of the prologue's callee-saved stores (before the first branch) and of the callee-saved loads in the
    run of sp loads before each ret (the epilogues)."""
    idx = set()
    for a, op, arg, _ in body:
        if op in CB or op in ("j", "ret", "jr", "jalr"):
            break
        if op == "sw" and "(sp)" in arg and SAVED.match(arg):
            idx.add(a)
    for i, (_, op, arg, _) in enumerate(body):
        if op == "ret" or (op == "jr" and arg == "ra"):
            k = i - 1
            while k >= 0 and ((body[k][1] == "lw" and "(sp)" in body[k][2]) or
                              (body[k][1] == "addi" and body[k][2].startswith("sp,"))):
                if body[k][1] == "lw" and SAVED.match(body[k][2]):
                    idx.add(body[k][0])
                k -= 1
    return idx


def loop_region(body):
    """Addresses of the largest strongly connected region of the function's control-flow graph."""
    addrs = [x[0] for x in body]
    known = set(addrs)
    succ = {}
    for i, (a, op, _, raw) in enumerate(body):
        nxt = addrs[i + 1] if i + 1 < len(addrs) else None
        if op in CB:
            s = [int(raw.split(",")[-1], 16), nxt]
        elif op == "j":
            s = [int(raw, 16)]
        elif op in ("ret", "jr") or (op == "jalr" and raw.startswith("zero")):
            s = []
        else:
            s = [nxt]
        succ[a] = [x for x in s if x is not None and x in known]
    index, low, on, stack, sccs, counter = {}, {}, set(), [], [], [0]
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
            for k in range(pi, len(succ[v])):
                w = succ[v][k]
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
    return set(max(sccs, key=len))


def sp_accesses(body, keep=None):
    fs = frame_saves(body)
    loads = sum(1 for a, op, arg, _ in body if op[0] == "l" and "(sp)" in arg and a not in fs and (keep is None or a in keep))
    stores = sum(1 for a, op, arg, _ in body if op[0] == "s" and "(sp)" in arg and a not in fs and (keep is None or a in keep))
    return loads, stores


def rx_body(fns):
    for name, body in fns.items():
        if any(k in name for k in RX_KEYS):
            return body
    return None


def rx_stats(fns):
    """(instructions, sp loads, sp stores, loop instructions, loop sp loads, loop sp stores) of the receiver loop
    function, or None."""
    body = rx_body(fns)
    if body is None:
        return None
    loop = loop_region(body)
    return (len(body), *sp_accesses(body), len(loop), *sp_accesses(body, loop))


def compare(pair):
    case, rel, a, b = pair
    if text(a) == text(b):
        s = rx_stats(functions(a))
        return case, rel, "same", s, s
    fa, fb = functions(a), functions(b)
    norm = lambda body: collections.Counter(f"{op} {arg}" for _, op, arg, _ in body)
    layout = set(fa) == set(fb) and all(norm(fa[k]) == norm(fb[k]) for k in fa)
    if layout:
        return case, rel, "layout", rx_stats(fa), rx_stats(fb)
    anon = lambda body: collections.Counter(REG.sub("r", f"{op} {arg}") for _, op, arg, _ in body)
    rename = set(fa) == set(fb) and all(anon(fa[k]) == anon(fb[k]) for k in fa)
    return case, rel, "rename" if rename else "differ", rx_stats(fa), rx_stats(fb)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    opts = dict((x[2:].split("=", 1) + [""])[:2] for x in sys.argv[1:] if x.startswith("--"))
    cases = set(opts["cases"].split(",")) if opts.get("cases") else None
    a, b = args
    pairs = []
    for e in sorted(glob.glob(f"{COV}/{a}/*/*/*.elf")):
        rel = os.path.relpath(e, f"{COV}/{a}")
        case = rel.split("__")[0]
        if cases and case not in cases:
            continue
        if os.path.exists(f"{COV}/{b}/{rel}"):
            pairs.append((case, rel, e, f"{COV}/{b}/{rel}"))
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        res = list(ex.map(compare, pairs))
    by = collections.defaultdict(list)
    for r in res:
        by[r[0]].append(r)
    tot = collections.Counter()
    for case in sorted(by):
        cls = collections.Counter(r[2] for r in by[case])
        tot.update(cls)
        deltas, ldeltas, base = collections.Counter(), collections.Counter(), []
        for _, rel, c, sa, sb in by[case]:
            if sa and sb and sa != sb:
                deltas[(sb[0] - sa[0], (sb[1] + sb[2]) - (sa[1] + sa[2]))] += 1
                ldeltas[(sb[3] - sa[3], (sb[4] + sb[5]) - (sa[4] + sa[5]))] += 1
            if sa:
                base.append(sa)
        fmt = lambda c: " ".join(f"{di:+d}/{ds:+d}x{n}" for (di, ds), n in sorted(c.items()))
        rng = ""
        if base:
            rng = (f"  base rx {min(s[0] for s in base)}-{max(s[0] for s in base)} insns "
                   f"{min(s[1] + s[2] for s in base)}-{max(s[1] + s[2] for s in base)} sp; loop "
                   f"{min(s[3] for s in base)}-{max(s[3] for s in base)} insns "
                   f"{min(s[4] + s[5] for s in base)}-{max(s[4] + s[5] for s in base)} sp")
        print(f"{case:26s} same {cls['same']:4d} layout {cls['layout']:4d} rename {cls['rename']:4d} differ "
              f"{cls['differ']:4d}{rng}")
        if deltas:
            print(f"{'':26s}   fn {fmt(deltas)}")
            print(f"{'':26s}   loop {fmt(ldeltas)}")
        if "list" in opts:
            for _, rel, c, sa, sb in by[case]:
                if c != "same":
                    print(f"    {rel} {c} {sa} -> {sb}")
    print(f"{'total':26s} same {tot['same']:4d} layout {tot['layout']:4d} rename {tot['rename']:4d} differ "
          f"{tot['differ']:4d}")


if __name__ == "__main__":
    main()
