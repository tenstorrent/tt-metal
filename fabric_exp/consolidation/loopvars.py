#!/usr/bin/env python3
"""loopvars.py: which values a router loop keeps on the stack, from two independent sources, cross-checked.
Study tooling only (fabric_exp; V4_RECEIVER_OWNERSHIP.md §10): not CI, nothing in tt-metal's test tree.

Usage:
  loopvars.py ELF... [--role receiver|sender] [--json OUT]   router images (E1: receiver, E0: sender)
  loopvars.py --compare LABEL=CACHE ... [--json OUT]           every router image of each config in kernel caches
  loopvars.py --live ELF... [--role R] [--at HEX,...] [--json OUT]  live registers and slots in the loop (liveness)
  loopvars.py --headroom BASE NEW [--role R] [--cases C,...] [--json OUT]  hot values NEW moves to the stack (images or
                                                              builds; sender: E0's sender::run, one ERISC's sender_pass)
  loopvars.py --self-test                                     known answers + recorded outputs (loopvars_golden/)
  loopvars.py --record-golden                                 rewrite the recorded outputs (deliberate updates only)

The loop: two ERISCs, receiver::run / sender::run, the largest strongly connected region of its control-flow graph;
one ERISC, single::receiver_pass / single::sender_pass, the whole function.
  A. Instructions: every sp-relative load and store in the loop, with its addr2line inline chain and source line.
  B. DWARF: the location of every variable and parameter of the function's concrete instance and of each inlined
     callee over the loop's instructions (DW_AT_location: an expression or a .debug_loclists list).
Cross-check (each access of A against B):
  slot      B places a variable at the accessed slot at the access or the next instruction;
  register  the stored register holds a variable at the store or the next instruction (B), or the loaded register
            holds one at the next instruction.
  Reported: values stored or loaded in the loop with their lines and the method that named them (a load can name
  both the slot's value and the local it is copied into). Disagreements: accesses neither method attributes
  ("unknown", never guessed); a named value whose scope is not in the access's inline chain; values B places on the
  stack in the loop that no access touches ("DWARF only").

Limits:
  - GCC's -O3/-Os location lists are incomplete and can lag an instruction: unknown accesses and DWARF-only values
    are expected. The one-instruction window above is the only allowance.
  - Slots are sp offsets. The frame comes from the prologue's addi sp,sp,-N; DW_OP_fbreg is converted with it (frame
    base = CFA = sp + frame). A loop that moves sp or jumps indirectly is flagged and its result is not trusted.
  - A location computed from a slot (DW_OP_fbreg/breg2 + deref ... stack_value) counts as that slot (marked ~).
    Other computed values count as their register (marked ~) or as a constant (no storage). Anything else is
    "unknown". Aggregates with DW_OP_piece are split into members by the DWARF type.
  - Only sp-relative accesses are stack. A record reached through another register (the one-ERISC state record,
    a struct in L1) is memory, not stack, and is not reported here.
  - --headroom: values NEW stores to the stack in its loop that BASE does not, classed hot unless they are members of
    an object the source keeps in memory, invariant pointers/coordinates (or the probe's synthetic state), or stored
    only next to an out-of-line call (save/restore around a cold path). It compares DWARF names, so BASE and NEW must
    share the source's naming (a variant and that variant plus synthetic state); across renames it over-reports.
  - --live: register liveness from the instructions (backward dataflow over the function's control-flow graph; a call
    clobbers the caller-saved registers and reads a0-a7; slots by sp offset; the prologue's save slots left out),
    named by DWARF where it places a variable there. Unnamed values are shown with their defining instruction and line
    (an invariant set before the pass loop, or a value set in the pass: GCC's promoted copies of memory, counters).
  - Parsing depends on binutils' text output. The self-test pins the tool's hash and the toolchain binaries; a
    normal run warns until --self-test has passed for the current pair.
"""
import collections, datetime, hashlib, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
GOLDEN = os.path.join(HERE, "loopvars_golden")
CB = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez", "blez", "bgtz"}
LOADS = {"lw": 4, "lh": 2, "lhu": 2, "lb": 1, "lbu": 1}
STORES = {"sw": 4, "sh": 2, "sb": 1}
ROLE_FUNCS = {"receiver": ("8receiver3run", "6single13receiver_pass"), "sender": ("6sender3run", "6single11sender_pass")}
CALLEE_SAVED = {"s%d" % i for i in range(12)}


def sh(args, inp=None):
    return subprocess.run(args, capture_output=True, text=True, input=inp).stdout


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def fingerprint():
    return {"tool": sha(os.path.abspath(__file__)),
            "binutils": sh([B + "readelf", "--version"]).splitlines()[0],
            "binaries": {t: sha(B + t) for t in ("readelf", "objdump", "addr2line", "nm", "gcc")}}


def short(name):
    """'step<0, fabric_router::X>' -> 'step<0>'; drops namespaces and type arguments, keeps integer arguments."""
    name = name.split(": ")[-1].strip()
    m = re.match(r"([^<]*)<(.*)>$", name)
    if not m:
        return name.split("::")[-1]
    args, depth, cur = [], 0, ""
    for c in m.group(2):
        depth += c == "<"
        depth -= c == ">"
        if c == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += c
    args.append(cur.strip())
    nums = [a.rstrip("u") for a in args if re.fullmatch(r"-?\d+u?", a)]
    base = m.group(1).split("::")[-1]
    return f"{base}<{','.join(nums)}>" if nums else base


def base(name):
    return re.sub(r"<.*", "", name)


# ---------------------------------------------------------------------------------------------------------------------
# A. Instructions
# ---------------------------------------------------------------------------------------------------------------------

def functions(elf):
    """{symbol: [(address, mnemonic, operands)]} for the sized text symbols."""
    syms = []
    for l in sh([B + "nm", "-S", elf]).splitlines():
        p = l.split()
        if len(p) == 4 and p[2] in "tT":
            syms.append((int(p[0], 16), int(p[1], 16), p[3]))
    ins = []
    for l in sh([B + "objdump", "-d", "--no-show-raw-insn", "--section=.text", elf]).splitlines():
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", l)
        if m:
            raw = re.sub(r"\s*#.*$", "", m.group(3))
            ins.append((int(m.group(1), 16), m.group(2), re.sub(r"\s*<[^>]*>", "", raw).strip()))
    out = {}
    for lo, size, name in syms:
        body = [x for x in ins if lo <= x[0] < lo + size]
        if body:
            out[name] = body
    return out


def pick(fns, role):
    for key in ROLE_FUNCS[role]:
        names = sorted(n for n in fns if key in n)
        if names:
            return names[0], key.endswith("run")
    sys.exit(f"no {role} loop function in the image (looked for {ROLE_FUNCS[role]})")


def cfg(body):
    """({address: successors within the function}, indirect jumps)."""
    addrs = [x[0] for x in body]
    known, succ, indirect = set(addrs), {}, 0
    for i, (a, op, raw) in enumerate(body):
        nxt = addrs[i + 1] if i + 1 < len(addrs) else None
        if op in CB:
            s = [int(raw.split(",")[-1], 16), nxt]
        elif op == "j":
            s = [int(raw, 16)]
        elif op == "ret":
            s = []
        elif op == "jr" or (op == "jalr" and raw.startswith("zero")):
            s, indirect = [], indirect + 1
        else:
            s = [nxt]
        succ[a] = [x for x in s if x is not None and x in known]
    return succ, indirect


def largest_scc(addrs, succ):
    index, low, on, stack, best, counter = {}, {}, set(), [], set(), [0]
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
            for k in range(pi, len(succ[v])):
                w = succ[v][k]
                if w not in index:
                    work.append((v, k + 1))
                    work.append((w, 0))
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            else:
                if low[v] == index[v]:
                    comp = set()
                    while True:
                        w = stack.pop()
                        on.discard(w)
                        comp.add(w)
                        if w == v:
                            break
                    if len(comp) > len(best):
                        best = comp
                if work:
                    u = work[-1][0]
                    low[u] = min(low[u], low[v])
    return best


def chains(elf, addrs):
    """{address: [(function, file:line)]}, innermost first (addr2line -a -f -i -C)."""
    out = sh([B + "addr2line", "-a", "-f", "-i", "-C", "-e", elf], "\n".join(hex(a) for a in addrs)).splitlines()
    res, cur, i = {}, None, 0
    while i < len(out):
        if re.fullmatch(r"0x[0-9a-f]+", out[i]):
            cur = int(out[i], 16)
            res[cur] = []
            i += 1
            continue
        fn = re.sub(r"\(.*", "", out[i])
        fn = re.sub(r"<.*", "", fn).split("::")[-1].split()[-1] if fn.strip() else "?"
        m = re.match(r"(.*):(\d+)", out[i + 1] if i + 1 < len(out) else "")
        res[cur].append((fn, (os.path.basename(m.group(1)) + ":" + m.group(2)) if m else "?"))
        i += 2
    return res


# ---------------------------------------------------------------------------------------------------------------------
# B. DWARF
# ---------------------------------------------------------------------------------------------------------------------

def parse_info(elf):
    dies, cur, stack = {}, None, []
    for l in sh([B + "readelf", "--debug-dump=info", elf]).splitlines():
        m = re.match(r"\s*<(\d+)><([0-9a-f]+)>: Abbrev Number: (\d+)(?: \((DW_TAG_\w+)\))?", l)
        if m:
            depth, off, tag = int(m.group(1)), int(m.group(2), 16), m.group(4)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            parent = stack[-1][1] if stack else None
            if tag is None:
                cur = None
                continue
            cur = {"tag": tag, "a": {}, "kids": []}
            dies[off] = cur
            if parent is not None:
                dies[parent]["kids"].append(off)
            stack.append((depth, off))
            continue
        m = re.match(r"\s*<[0-9a-f]+>\s+DW_AT_(\w+)\s*:\s*(.*)$", l)
        if m and cur is not None:
            cur["a"][m.group(1)] = m.group(2).strip()
    return dies


def ref(v):
    m = re.search(r"<0x([0-9a-f]+)>", v or "")
    return int(m.group(1), 16) if m else None


def follow(dies, off, attr, seen=0):
    d = dies.get(off)
    if d is None or seen > 8:
        return None
    if attr in d["a"]:
        return d["a"][attr]
    for k in ("abstract_origin", "specification"):
        if k in d["a"]:
            return follow(dies, ref(d["a"][k]), attr, seen + 1)
    return None


def name_of(dies, off):
    n = follow(dies, off, "name")
    return n.split(": ")[-1].strip() if n else "?"


def size_of(d):
    v = d["a"].get("byte_size")
    return int(v.split()[0]) if v else None


def leaves(dies, toff, prefix="", at=0, depth=0):
    """[(byte offset, size or None, member path)] of a type's scalar leaves (structs, classes and arrays expanded)."""
    d = dies.get(toff)
    if d is None or depth > 12:
        return [(at, None, prefix)]
    if d["tag"] in ("DW_TAG_typedef", "DW_TAG_const_type", "DW_TAG_volatile_type"):
        return leaves(dies, ref(d["a"].get("type")), prefix, at, depth + 1)
    if d["tag"] in ("DW_TAG_structure_type", "DW_TAG_class_type"):
        out = []
        for k in d["kids"]:
            m = dies[k]
            if m["tag"] != "DW_TAG_member" or "data_member_location" not in m["a"]:
                continue
            mo = int(re.match(r"(\d+)", m["a"]["data_member_location"]).group(1))
            nm = m["a"].get("name", "?").split(": ")[-1]
            out += leaves(dies, ref(m["a"].get("type")), (prefix + "." if prefix else "") + nm, at + mo, depth + 1)
        return out or [(at, size_of(d), prefix)]
    if d["tag"] == "DW_TAG_array_type":
        sub = [dies[k] for k in d["kids"] if dies[k]["tag"] == "DW_TAG_subrange_type"]
        n = None
        if len(sub) == 1:
            if "count" in sub[0]["a"]:
                n = int(sub[0]["a"]["count"].split()[0], 0)
            elif "upper_bound" in sub[0]["a"]:
                n = int(sub[0]["a"]["upper_bound"].split()[0], 0) + 1
        el = leaves(dies, ref(d["a"].get("type")), "", 0, depth + 1)
        esz = type_size(dies, ref(d["a"].get("type")))
        if n is None or esz is None or n > 256:
            return [(at, None, prefix)]
        return [(at + i * esz + o, s, f"{prefix}[{i}]" + ("." + p if p else "")) for i in range(n) for o, s, p in el]
    return [(at, size_of(d), prefix)]


def type_size(dies, toff, depth=0):
    d = dies.get(toff)
    if d is None or depth > 12:
        return None
    if "byte_size" in d["a"]:
        return size_of(d)
    if d["tag"] in ("DW_TAG_typedef", "DW_TAG_const_type", "DW_TAG_volatile_type"):
        return type_size(dies, ref(d["a"].get("type")), depth + 1)
    if d["tag"] == "DW_TAG_array_type":
        lf = leaves(dies, toff)
        return max((o + (s or 0) for o, s, _ in lf), default=None) or None
    return None


def parse_loclists(elf):
    """{list offset: [(lo, hi, expression)]} from .debug_loclists (DWARF 5 with location views)."""
    lists, cur, pending = {}, None, False
    for l in sh([B + "readelf", "--debug-dump=loc", elf]).splitlines():
        if "<End of list>" in l:
            cur = None
            continue
        m = re.match(r"\s+([0-9a-f]{8}) ([0-9a-f]{8}) \(base address\)", l)
        if m:
            if cur is None:
                cur = int(m.group(1), 16)
                lists[cur] = []
            continue
        m = re.match(r"\s+([0-9a-f]{8}) v[0-9a-f]+ v[0-9a-f]+ views at [0-9a-f]+ for:", l)
        if m:
            if cur is None:
                cur = int(m.group(1), 16)
                lists[cur] = []
            pending = True
            continue
        m = re.match(r"\s+([0-9a-f]{8}) ([0-9a-f]{8}) \((.*)\)\s*$", l)
        if m and pending and cur is not None:
            lists[cur].append((int(m.group(1), 16), int(m.group(2), 16), m.group(3)))
            pending = False
    return lists


def decode(expr, frame):
    """[(size or None, kind, where, computed)]: kind 'reg' (where = name), 'slot' (where = sp offset), 'const', 'unknown'.
    A piece's size comes from DW_OP_piece."""
    parts = re.split(r";?\s*DW_OP_piece: (\d+)", expr)
    if len(parts) == 1:
        return [(None,) + one(expr, frame)]
    return [(int(parts[i + 1]),) + one(parts[i], frame) for i in range(0, len(parts) - 1, 2)]


def one(e, frame):
    toks = [t.strip() for t in e.strip("; ").split(";") if t.strip()]
    if not toks:
        return ("unknown", None, False)
    computed = toks[-1] == "DW_OP_stack_value"
    m = re.fullmatch(r"DW_OP_reg\d+ \((\w+)\)", toks[0])
    if m and len(toks) == 1:
        return ("reg", m.group(1), False)
    slot = None
    m = re.fullmatch(r"DW_OP_breg2 \(sp\): (-?\d+)", toks[0])
    if m:
        slot = int(m.group(1))
    m = re.fullmatch(r"DW_OP_fbreg: (-?\d+)", toks[0])
    if m:
        slot = frame + int(m.group(1))
    if slot is not None:
        if len(toks) == 1:
            return ("slot", slot, False)
        if computed and toks[1].startswith("DW_OP_deref"):
            return ("slot", slot, True)
        return ("unknown", None, False)
    m = re.fullmatch(r"DW_OP_breg\d+ \((\w+)\): -?\d+", toks[0])
    if m and computed and not any("entry_value" in t or "deref" in t for t in toks):
        return ("reg", m.group(1), True)
    if computed and all(re.fullmatch(r"DW_OP_(lit\d+|const\w+: -?\d+|stack_value|neg|not|plus_uconst: \d+)", t)
                        for t in toks):
        return ("const", None, False)
    return ("unknown", None, False)


def locations(elf, dies, root, frame):
    """[(label, scope, kind, where, size, computed, lo, hi, leaves of a whole aggregate in a slot or None)] for every
    variable and parameter under the concrete DIE."""
    lists = parse_loclists(elf)
    out = []

    def walk(off, scope, chan):
        for k in dies[off]["kids"]:
            kd = dies[k]
            if kd["tag"] == "DW_TAG_inlined_subroutine":
                s = short(name_of(dies, k))
                walk(k, s, s if "<" in s else chan)
            elif kd["tag"] == "DW_TAG_lexical_block":
                walk(k, scope, chan)
            elif kd["tag"] in ("DW_TAG_variable", "DW_TAG_formal_parameter") and "location" in kd["a"]:
                lv = kd["a"]["location"]
                m = re.match(r"0x([0-9a-f]+) \(location list\)", lv)
                if m:
                    ranges = lists.get(int(m.group(1), 16), [])
                else:
                    ranges = [(0, 1 << 32, re.sub(r"^\d+ byte block: [0-9a-f ]+\t?", "", lv).strip("() "))]
                toff = ref(follow(dies, k, "type"))
                lf = leaves(dies, toff) if toff else [(0, None, "")]
                label = scope + (f"@{chan}" if chan and chan != scope and "<" not in scope else "") + ":" + \
                    name_of(dies, k)
                for rlo, rhi, expr in ranges:
                    off_b = 0
                    for size, kind, where, computed in decode(expr, frame):
                        member = ""
                        if size is not None:
                            mm = [p for o, s, p in lf if o == off_b]
                            member = "." + mm[0] if mm and mm[0] else f".@{off_b}"
                        elif len(lf) == 1:
                            size = lf[0][1]
                        elif kind == "slot":
                            size = type_size(dies, toff)
                        if kind == "slot" and computed:
                            sz = re.search(r"DW_OP_deref_size: (\d+)", expr)
                            size = int(sz.group(1)) if sz else 4
                        whole = kind == "slot" and not computed and not member and len(lf) > 1
                        out.append((label + member, scope, kind, where, size, computed, rlo, rhi, lf if whole else None))
                        off_b += size or 0
    walk(root, short(name_of(dies, root)), None)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The map and the cross-check
# ---------------------------------------------------------------------------------------------------------------------

def analyse(elf, role="receiver", detail=False):
    fns = functions(elf)
    name, is_run = pick(fns, role)
    body = fns[name]
    addrs = [a for a, *_ in body]
    nxt = {addrs[i]: addrs[i + 1] for i in range(len(addrs) - 1)}
    succ, indirect = cfg(body)
    loop = largest_scc(addrs, succ) if is_run else set(addrs)
    frame, sp_moves = 0, 0
    for i, (a, op, raw) in enumerate(body):
        m = re.fullmatch(r"sp,sp,(-?\d+)", raw)
        if op == "addi" and m:
            if i < 40 and frame == 0 and int(m.group(1)) < 0:
                frame = -int(m.group(1))
            elif a in loop:
                sp_moves += 1
    dies = parse_info(elf)
    roots = [o for o, d in dies.items() if d["tag"] == "DW_TAG_subprogram" and
             d["a"].get("low_pc", "").startswith("0x") and int(d["a"]["low_pc"], 16) == addrs[0]]
    if not roots:
        return {"error": f"no concrete DWARF instance for {name}"}
    locs = locations(elf, dies, roots[0], frame)
    in_loop = sorted(loop)
    regs_at = collections.defaultdict(set)   # (address, register) -> labels
    slots = []                               # (lo, hi, slot, size, label, leaves of an aggregate)
    for label, scope, kind, where, size, computed, lo, hi, lf in locs:
        if kind == "reg":
            for a in in_loop:
                if lo <= a < hi:
                    regs_at[(a, where)].add(label + ("~" if computed else ""))
        elif kind == "slot":
            slots.append((lo, hi, where, size, label + ("~" if computed else ""), lf))

    def member(label, lf, off):
        hit = [p for o, s, p in (lf or []) if o <= off < o + (s or 1)]
        return f"{label}.{hit[0]}" if hit and hit[0] else label

    def at_slot(a, slot, width):
        return {member(label, lf, slot - x) for lo, hi, x, size, label, lf in slots
                if (lo <= a < hi or lo <= nxt.get(a, a) < hi) and x < slot + width and slot < x + (size or 1)}

    ch = chains(elf, in_loop)
    acc = []
    for a, op, raw in body:
        m = re.fullmatch(r"(\w+),(-?\d+)\(sp\)", raw)
        if a not in loop or not m or not (op in LOADS or op in STORES):
            continue
        r, slot = m.group(1), int(m.group(2))
        kind = "L" if op in LOADS else "S"
        by_slot = at_slot(a, slot, LOADS.get(op) or STORES[op])
        if kind == "S":
            by_reg = regs_at.get((a, r), set()) | regs_at.get((nxt.get(a), r), set())
        else:
            by_reg = regs_at.get((nxt.get(a), r), set())
        chain = ch.get(a) or [("?", "?")]
        acc.append({"addr": a, "kind": kind, "reg": r, "slot": slot, "fn": chain[0][0], "line": chain[0][1],
                    "chain": [f for f, _ in chain], "slot_vars": sorted(by_slot), "reg_vars": sorted(by_reg)})
    values = collections.defaultdict(lambda: {"L": 0, "S": 0, "slots": set(), "lines": collections.Counter(),
                                              "methods": set()})
    unknown, mismatched = collections.Counter(), collections.Counter()
    for x in acc:
        where = f"{x['kind']} sp+{x['slot']} {x['fn']} {x['line']}"
        both = set(x["slot_vars"]) | set(x["reg_vars"])
        if not both:
            unknown[where] += 1
            continue
        chain_bases = {base(f) for f in x["chain"]}
        if not any(base(v.split(":")[0].split("@")[0]) in chain_bases for v in both):
            mismatched[f"{where}: {sorted(both)} (chain {' < '.join(x['chain'])})"] += 1
        for v in both:
            d = values[v]
            d[x["kind"]] += 1
            d["slots"].add(x["slot"])
            d["lines"][f"{x['kind']} {x['fn']} {x['line']}"] += 1
            d["methods"].add("slot" if v in x["slot_vars"] else "")
            d["methods"].add("register" if v in x["reg_vars"] else "")
    dwarf_only = collections.defaultdict(set)
    for lo, hi, x, size, label, lf in slots:
        if not any(v == label or v.startswith(label + ".") for v in values) and any(lo <= a < hi for a in in_loop):
            dwarf_only[label].add(x)
    regmap = collections.defaultdict(collections.Counter)
    for (a, r), labels in regs_at.items():
        if r in CALLEE_SAVED:
            for v in labels:
                regmap[r][v] += 1
    m = re.search(r"GCC: \(([^)]*)\) (\S+)", sh([B + "readelf", "-p", ".comment", elf]))
    return {
        "elf": os.path.basename(elf), "sha": sha(elf), "compiler": f"{m.group(1)} {m.group(2)}" if m else "?",
        "role": role, "function": name,
        "loop": "largest strongly connected region of run()" if is_run else "whole pass function (one ERISC)",
        "insns": len(body), "loop_insns": len(loop), "frame": frame,
        "untrusted": [w for w, n in (("sp moves in the loop", sp_moves), ("indirect jumps", indirect)) if n],
        "accesses": {"total": len(acc), "loads": sum(x["kind"] == "L" for x in acc),
                     "stores": sum(x["kind"] == "S" for x in acc), "unknown": sum(unknown.values())},
        "stored": sorted(v for v, d in values.items() if d["S"]),
        "values": {v: {"loads": d["L"], "stores": d["S"], "slots": sorted(d["slots"]),
                       "methods": sorted(m for m in d["methods"] if m), "lines": dict(sorted(d["lines"].items()))}
                   for v, d in sorted(values.items())},
        "unknown": dict(sorted(unknown.items())),
        "scope_mismatches": dict(sorted(mismatched.items())),
        "dwarf_only": {v: sorted(s) for v, s in sorted(dwarf_only.items())},
        "registers": {r: dict(sorted(c.items(), key=lambda t: (-t[1], t[0]))[:3])
                      for r, c in sorted(regmap.items(), key=lambda t: int(t[0][1:]))},
        **({"access_list": acc, "body": body, "loop_addrs": sorted(loop)} if detail else {}),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Register liveness (--live): what the loop holds at each instruction, from the instructions alone, named by DWARF
# ---------------------------------------------------------------------------------------------------------------------

REG_RE = re.compile(r"\b(zero|ra|sp|gp|tp|fp|t[0-6]|s1[01]|s[0-9]|a[0-7])\b")
CALLER_SAVED = {"t%d" % i for i in range(7)} | {"a%d" % i for i in range(8)}
ARG_REGS = {"a%d" % i for i in range(8)}
NOT_COUNTED = {"zero", "sp", "gp", "tp", "ra"}
BRANCHES = CB | {"bgt", "ble", "bgtu", "bleu"}


def defuse(op, raw):
    """(registers written, registers read, is a call) of one instruction (RV32 with the B extensions; pseudo-ops)."""
    regs = ["s0" if t == "fp" else t for t in REG_RE.findall(raw)]
    if op in STORES or op in BRANCHES or op in ("j", "jr") or op.startswith(("csrw", "csrs", "csrc")) and len(regs) <= 1:
        return set(), set(regs), False
    if op in ("ret", "fence", "fence.i", "nop", "ecall", "ebreak", "wfi"):
        return set(), set(), False
    if op == "jal" or (op == "jalr" and not raw.startswith("zero")):
        return CALLER_SAVED | {"ra"}, ARG_REGS | set(r for r in regs if r != "ra"), True
    if op == "jalr":
        return set(), set(regs[1:]), False
    return ({regs[0]} if regs else set()), set(regs[1:]), False


def liveness(elf, role="receiver", at=()):
    """Live registers and stack slots at each instruction of the role's loop (backward dataflow over the function's
    control-flow graph; a call clobbers the caller-saved registers and reads the argument registers; a stack slot is
    written by a store and read by a load at its sp offset; the prologue's callee-saved save slots are left out).
    Names from DWARF where it places a variable in the register (or slot) at that instruction."""
    fns = functions(elf)
    name, is_run = pick(fns, role)
    body = fns[name]
    addrs = [a for a, *_ in body]
    succ, _ = cfg(body)
    syms = {}
    for l in sh([B + "nm", elf]).splitlines():
        p = l.split()
        if len(p) == 3:
            syms[int(p[0], 16)] = p[2]
    rounds = largest_scc(addrs, succ) if is_run else set(addrs)
    cut = {a for a, op, raw in body if op == "jal" and "end_round" in syms.get(int(raw.split(",")[-1], 16), "")}
    if is_run:
        succ2 = {a: ([] if a in cut else [x for x in s if x not in cut]) for a, s in succ.items()}
        passes = largest_scc([a for a in addrs if a not in cut], succ2)
    else:
        passes = set(addrs)
    frame, saves = 0, set()
    for i, (a, op, raw) in enumerate(body[:40]):
        m = re.fullmatch(r"sp,sp,(-?\d+)", raw)
        if op == "addi" and m and frame == 0:
            frame = -int(m.group(1))
        m = re.fullmatch(r"(ra|s\d+),(\d+)\(sp\)", raw)
        if op == "sw" and m:
            saves.add(int(m.group(2)))
    du = {}
    for a, op, raw in body:
        d, u, call = defuse(op, raw)
        m = re.fullmatch(r"(\w+),(-?\d+)\(sp\)", raw)
        if m and int(m.group(2)) not in saves:
            slot = "sp+" + m.group(2)
            if op in STORES:
                d = d | {slot}
            elif op in LOADS:
                u = u | {slot}
        du[a] = (d, u, call)
    live_in = {a: set() for a in addrs}
    changed = True
    while changed:
        changed = False
        for a in reversed(addrs):
            out = set().union(*(live_in[s] for s in succ[a])) if succ[a] else set()
            d, u, _ = du[a]
            new = u | (out - d)
            if new != live_in[a]:
                live_in[a] = new
                changed = True
    # names
    dies = parse_info(elf)
    roots = [o for o, dd in dies.items() if dd["tag"] == "DW_TAG_subprogram" and
             dd["a"].get("low_pc", "").startswith("0x") and int(dd["a"]["low_pc"], 16) == addrs[0]]
    locs = locations(elf, dies, roots[0], frame) if roots else []
    names = collections.defaultdict(set)
    slot_locs = []
    for label, scope, kind, where, size, computed, lo, hi, lf in locs:
        if kind == "reg":
            for a in addrs:
                if lo <= a < hi:
                    names[(a, where)].add(label + ("~" if computed else ""))
        elif kind == "slot":
            slot_locs.append((lo, hi, where, size, label + ("~" if computed else ""), lf))

    def slot_names(a, off):
        out = set()
        for lo, hi, x, size, label, lf in slot_locs:
            if lo <= a < hi and x <= off < x + (size or 1):
                hit = [p for o, sz, p in (lf or []) if o <= off - x < o + (sz or 1)]
                out.add(f"{label}.{hit[0]}" if hit and hit[0] else label)
        return out
    text_of = {a: f"{op} {raw}" for a, op, raw in body}
    defined_in_pass = set()
    for a in passes:
        defined_in_pass |= {x for x in du[a][0] if not du[a][2]}
    ch = chains(elf, sorted(passes) + [x for x in at if x not in passes])
    head = min((a for a in passes if any(p not in passes for p in addrs if a in succ.get(p, []))), default=min(passes))

    def defs_of(v, inside):
        return [a for a in addrs if v in du[a][0] and not du[a][2] and ((a in passes) == inside) and
                not re.fullmatch(r"(ra|s\d+),(\d+)\(sp\)", text_of[a].split(" ", 1)[-1])]
    def_chains = chains(elf, sorted({a for a in addrs if du[a][0] and not du[a][2]}))

    def where(a):
        c = def_chains.get(a) or [("?", "?")]
        return f"{c[0][0]} {c[0][1]}"

    def describe(a):
        vals = []
        for v in sorted(live_in[a] - NOT_COUNTED, key=lambda x: (x.startswith("sp+"), x)):
            n = sorted(slot_names(a, int(v[3:])) if v.startswith("sp+") else names.get((a, v), ()))
            kind = "slot" if v.startswith("sp+") else ("callee" if v.startswith("s") else "caller")
            origin = ""
            if not n:
                inside = defs_of(v, True)
                if not inside:
                    before = defs_of(v, False)
                    origin = "invariant: " + ("; ".join(sorted({f"{text_of[x]} ({where(x)})" for x in before})[:2])
                                              if before else "?")
                else:
                    origin = "set in the pass: " + "; ".join(sorted({f"{text_of[x]} ({where(x)})"
                                                                      for x in inside})[:2])
            vals.append({"loc": v, "kind": kind, "names": n[:3], "origin": origin})
        return {"addr": a, "line": "%s %s" % (ch.get(a, [("?", "?")])[0]), "live": len(vals),
                "regs": sum(x["kind"] != "slot" for x in vals), "slots": sum(x["kind"] == "slot" for x in vals),
                "values": vals}
    counts = {a: len(live_in[a] - NOT_COUNTED) for a in passes}
    peak = max(passes, key=lambda a: (counts[a], -a))
    return {"elf": os.path.basename(elf), "function": name, "pass_insns": len(passes), "round_insns": len(rounds),
            "head": describe(head), "peak": describe(peak),
            "mean_live": round(sum(counts.values()) / len(counts), 1),
            "at": [describe(a) for a in at if a in live_in]}


def live_text(r):
    out = [f"{r['elf']}: {r['function'][:70]}", f"  pass loop {r['pass_insns']} instructions (round loop "
           f"{r['round_insns']}); live values per instruction: mean {r['mean_live']}, peak {r['peak']['live']}"]
    for title, d in [("carried (live at the pass loop's head)", r["head"]), ("peak", r["peak"])] + \
            [("at", x) for x in r["at"]]:
        out.append(f"  {title} {d['addr']:x} {d['line']}: {d['live']} ({d['regs']} registers, {d['slots']} slots)")
        for v in d["values"]:
            out.append(f"    {v['loc']:7s} {v['kind']:6s} {', '.join(v['names']) or v['origin']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------
# Headroom (--headroom): what a change (synthetic state, a variant) moves to the stack in the loop
# ---------------------------------------------------------------------------------------------------------------------

# Values whose stack placement is by design or cheap to rebuild, not "hot": memory objects the source keeps in memory
# (their members), invariant pointers and coordinates (`me`, renamed `my_chip` in v4rxf1), and the headroom probe's own
# synthetic state. Per role. The sender's (V4_SENDER_OWNERSHIP.md §7.4): the channel states and forwarding records kept
# in memory (the loop works on its loop-local copy), the round counters; invariant: the worker interfaces and their
# connection copies, the upstream routers' credit addresses.
DESIGNED_MEMORY = {
    "receiver": re.compile(r"run:(states|memory|tables|sources)\b|sources\b|records\b"),
    "sender": re.compile(r"run:(states|forwarding|later|round)\b|forwarding\.|records\b|\bround\."),
}
INVARIANT = {
    "receiver": re.compile(r"downstreams|routers\b|\bme\.|:me\b|\bmy_chip\b|coords|credits\b|synthetic"),
    "sender": re.compile(r"interface|connections\b|upstream_free_slots_addr|upstream_credit|upstream\b|credits\b|"
                         r"synthetic"),
}


def calls_near(r, window=8):
    """Addresses within `window` instructions of a call in the loop (other than the round's calls): where GCC saves and
    restores live values around an out-of-line call (deliver_other and other cold paths)."""
    body = r["body"]
    loop = set(r["loop_addrs"])
    idx = {a: i for i, (a, _, _) in enumerate(body)}
    near = set()
    for i, (a, op, raw) in enumerate(body):
        if a in loop and (op == "jal" or (op == "jalr" and not raw.startswith("zero"))):
            for j in range(max(0, i - window), min(len(body), i + window + 1)):
                near.add(body[j][0])
    return near


def stack_classes(r, role="receiver"):
    """{label: class} for every value stored to the stack in the loop: 'hot', 'designed memory', 'invariant', or
    'around a call' (stored only next to an out-of-line call)."""
    near = calls_near(r)
    designed, invariant = DESIGNED_MEMORY[role], INVARIANT[role]
    out = {}
    for label in r["stored"]:
        stores = [x for x in r["access_list"] if x["kind"] == "S" and label in set(x["slot_vars"]) | set(x["reg_vars"])]
        if designed.search(label) or (stores and all(
                any(designed.search(v) for v in x["slot_vars"]) for x in stores)):
            out[label] = "designed memory"  # a member of an object kept in memory, or a store into one
        elif invariant.search(label):
            out[label] = "invariant"
        elif stores and all(x["addr"] in near for x in stores):
            out[label] = "around a call"
        else:
            out[label] = "hot"
    return out


def headroom(base_elf, new_elf, role="receiver"):
    """The hot values `new_elf` stores to the stack in its loop that `base_elf` does not (with their store lines), the
    values it sheds that are not hot, and both images' carried and peak live values (liveness)."""
    rb, rn = analyse(base_elf, role, detail=True), analyse(new_elf, role, detail=True)
    if "error" in rb or "error" in rn:
        return {"error": rb.get("error") or rn.get("error")}
    cb, cn = stack_classes(rb, role), stack_classes(rn, role)
    hot_b = {v for v, c in cb.items() if c == "hot"}
    new_hot = {v: [k for k in rn["values"][v]["lines"] if k.startswith("S ")] for v, c in sorted(cn.items())
               if c == "hot" and v not in hot_b}
    shed = {v: c for v, c in sorted(cn.items()) if c != "hot" and v not in cb}
    lb, ln = liveness(base_elf, role), liveness(new_elf, role)
    return {"base": os.path.basename(base_elf), "new": os.path.basename(new_elf), "new_hot": new_hot, "shed": shed,
            "carried": [lb["head"]["live"], ln["head"]["live"]], "carried_regs": [lb["head"]["regs"], ln["head"]["regs"]],
            "peak": [lb["peak"]["live"], ln["peak"]["live"]], "loop_insns": [rb["loop_insns"], rn["loop_insns"]],
            "sp_accesses": [rb["accesses"]["total"], rn["accesses"]["total"]], "passes": not new_hot}


# The role's loop per image kind: the receiver's on E1 images (receiver::run, or one ERISC's single::receiver_pass); the
# sender's on two-ERISC E0 images (sender::run) and on one-ERISC images (single::sender_pass, in the
# subordinate_active_erisc image).
ROLE_IMAGES = {
    "receiver": {"subordinate_active_erisc": ("8receiver3run", "6single13receiver_pass")},
    "sender": {"active_erisc": ("6sender3run",), "subordinate_active_erisc": ("6single11sender_pass",)},
}


def headroom_job(job):
    base, new, key, role = job
    try:
        names = sh([B + "nm", new])
        proc = os.path.basename(new)[:-len(".elf")]
        if not any(f in names for f in ROLE_IMAGES[role].get(proc, ())):
            return key, None
        return key, headroom(base, new, role)
    except Exception as e:  # report, do not guess
        return key, {"error": str(e)}


def headroom_builds(base_dir, new_dir, cases=None, out_json=None, role="receiver"):
    """Every image of the role present in both coverage builds (or kernel caches): per config family, how many configs
    pass (no new hot value on the stack), which hot values move, and the carried / peak changes."""
    import glob, multiprocessing
    def images(d):
        out = {}
        for proc in ROLE_IMAGES[role]:
            for p in glob.glob(f"{d}/**/{proc}.elf", recursive=True):
                rel = os.path.relpath(p, d)
                key = rel.split("/")[-3] if "/kernels/" in rel else rel.split("/")[0]
                if not cases or key.split("__")[0] in cases:
                    out[key if role == "receiver" else f"{key}/{proc}"] = p
        return out
    ib, inew = images(base_dir), images(new_dir)
    jobs = [(ib[k], inew[k], k, role) for k in sorted(set(ib) & set(inew))]
    with multiprocessing.Pool(min(16, os.cpu_count() or 1)) as pool:
        res = [(k, r) for k, r in pool.map(headroom_job, jobs) if r]
    fam = collections.defaultdict(list)
    for k, r in res:
        fam[k.split("__")[0] if "__" in k else "config"].append((k, r))
    total_fail = 0
    for f, rs in sorted(fam.items()):
        ok = [r for k, r in rs if "error" not in r]
        fails = [(k, r) for k, r in rs if "error" not in r and not r["passes"]]
        total_fail += len(fails)
        moved = collections.Counter(v for k, r in fails for v in r["new_hot"])
        shed = collections.Counter(f"{v} ({c})" for k, r in rs if "error" not in r for v, c in r["shed"].items())
        d = lambda key: sorted({r[key][1] - r[key][0] for r in ok})
        print(f"  {f:24s} {len(ok) - len(fails)}/{len(rs)} pass; carried {d('carried')}, peak {d('peak')}, sp accesses "
              f"{d('sp_accesses')}" + (f"; HOT MOVED {dict(moved.most_common(6))}" if moved else "") +
              (f"; shed {dict(shed.most_common(4))}" if shed else ""))
        for k, r in rs:
            if "error" in r:
                print(f"    error {k}: {r['error']}")
    if out_json:
        json.dump(dict(res), open(out_json, "w"), indent=1, sort_keys=True)
    return total_fail


def text(r):
    out = [f"{r['elf']} ({r['sha']}, {r['compiler']}): {r['role']} {r['function'][:60]}",
           f"  loop: {r['loop']}: {r['loop_insns']} of {r['insns']} instructions, frame {r['frame']}"]
    if r["untrusted"]:
        out.append(f"  NOT TRUSTED: {', '.join(r['untrusted'])}")
    a = r["accesses"]
    out.append(f"  sp accesses in the loop: {a['total']} ({a['loads']} loads, {a['stores']} stores), unknown {a['unknown']}")
    fmt = lambda v, d: (f"    {v:44s} sp+{','.join(map(str, d['slots'])):6s} S{d['stores']} L{d['loads']} "
                        f"[{'+'.join(d['methods'])}]  " + "; ".join(f"{k} x{n}" for k, n in d["lines"].items()))
    out.append("  stored to the stack in the loop (spills):")
    out += [fmt(v, r["values"][v]) for v in r["stored"]]
    out.append("  loaded only (on the stack before the loop):")
    out += [fmt(v, d) for v, d in r["values"].items() if not d["stores"]]
    for key, title in (("unknown", "unknown (no DWARF variable by slot or register)"),
                       ("scope_mismatches", "DISAGREE (value's scope not in the access's inline chain)")):
        if r[key]:
            out.append(f"  {title}:")
            out += [f"    {n:3d}  {k}" for k, n in r[key].items()]
    if r["dwarf_only"]:
        out.append("  DWARF only (placed on the stack in the loop, no access found): " +
                   ", ".join(f"{v} sp+{','.join(map(str, s))}" for v, s in r["dwarf_only"].items()))
    out.append("  callee-saved registers (values, loop instructions):")
    out += [f"    {reg:4s} {m}" for reg, m in r["registers"].items()]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------------------------------------------------

# E1 images of one torus-Y + VC2 router config (hash 14894759742743696472, the router PC-sampled on torusns_scatter in
# V4_RECEIVER_OWNERSHIP.md §8): the lead (v4rxv2i3), S2 (v4rxo4) and S3 (B2, v4rxo5). The samples found three freshly
# loaded VC0 values spilled in S3, each store waiting on its load: the TX-queue status (sw a5,12(sp)), the scatter
# header's chunk count (sw a5,24(sp), router_local_delivery.hpp:140) and its encoding (sw ..,28(sp), :141). The lead
# and S2 keep the status and the count in registers. They spill the encoding byte too (S2: sw a2,8(sp) at :141, 4.3
# cycles/packet in the samples), so the brief's "none on the same paths" holds for two of the three values.
# Each known answer: the value's store (slot and the store's own source line) in the loop, or None for no store.
# Exactly what S3 stores in the loop that S2 does not (S3_OVER_S2): the status and the count; the encoding after each
# chunk shift (:162, 1.1 cycles/packet in the samples); a NoC command word (:1485, 1.0); channel 1's encoding (:141).
GOLDEN_IMAGES = ["lead_torusy_vc2_e1.elf", "s2_torusy_vc2_e1.elf", "s3_torusy_vc2_e1.elf"]
WATCHED = ("pass:txq_status", "deliver<0>:chunks", "deliver<0>:encoding")
KNOWN = {
    "s3_torusy_vc2_e1.elf": {"pass:txq_status": (12, "eth_txq_reg_read tt_eth_api.h:31"),
                             "deliver<0>:chunks": (24, "deliver router_local_delivery.hpp:140"),
                             "deliver<0>:encoding": (28, "deliver router_local_delivery.hpp:141")},
    "s2_torusy_vc2_e1.elf": {"pass:txq_status": None, "deliver<0>:chunks": None,
                             "deliver<0>:encoding": (8, "deliver router_local_delivery.hpp:141")},
    "lead_torusy_vc2_e1.elf": {"pass:txq_status": None, "deliver<0>:chunks": None,
                               "deliver<0>:encoding": (8, "deliver router_local_delivery.hpp:102")},
}


S3_OVER_S2 = ["NOC_CMD_BUF_WRITE_REG@deliver<0>:val", "NOC_CMD_BUF_WRITE_REG@deliver<1>:val", "deliver<0>:chunks",
              "deliver<0>:encoding~", "deliver<1>:encoding", "pass:txq_status"]


def stores_of(r, w):
    """[(slot, store line)] of a watched value in the loop; members and computed forms (w.x, w~) count as w."""
    out = []
    for v, d in r["values"].items():
        if re.fullmatch(re.escape(w) + r"([.~].*)?", v) and d["stores"]:
            out += [(slot, k[2:]) for slot in d["slots"] for k in d["lines"] if k.startswith("S ")]
    return out


def self_test(record=False):
    ok, results = True, {}
    for img in GOLDEN_IMAGES:
        r = analyse(os.path.join(GOLDEN, img))
        results[img] = r
        for w, want in KNOWN[img].items():
            got = stores_of(r, w)
            if (want is None and got) or (want is not None and want not in got):
                print(f"FAIL {img}: {w}: expected store {want}, found {got}")
                ok = False
        if r["untrusted"]:
            print(f"FAIL {img}: {r['untrusted']}")
            ok = False
        print(f"{img}: " + "; ".join(f"{w} {stores_of(r, w) or 'not stored'}" for w in WATCHED))
    hr = headroom(os.path.join(GOLDEN, "s2_torusy_vc2_e1.elf"), os.path.join(GOLDEN, "s3_torusy_vc2_e1.elf"))
    if sorted(hr["new_hot"]) != S3_OVER_S2:
        print(f"FAIL headroom S2 -> S3: expected the hot values {S3_OVER_S2}, found {sorted(hr['new_hot'])}")
        ok = False
    if not headroom(os.path.join(GOLDEN, "s2_torusy_vc2_e1.elf"), os.path.join(GOLDEN, "s2_torusy_vc2_e1.elf"))["passes"]:
        print("FAIL headroom S2 -> S2: expected no hot value moved")
        ok = False
    added = sorted(set(results["s3_torusy_vc2_e1.elf"]["stored"]) - set(results["s2_torusy_vc2_e1.elf"]["stored"]))
    if added != S3_OVER_S2:
        print(f"FAIL S3 stores over S2: expected {S3_OVER_S2}, found {added}")
        ok = False
    golden, fp = os.path.join(GOLDEN, "golden.json"), fingerprint()
    if record:
        json.dump({"fingerprint": fp, "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
                   "results": results}, open(golden, "w"), indent=1, sort_keys=True)
        print(f"recorded {golden}")
    elif os.path.exists(golden):
        g = json.load(open(golden))
        for img, r in results.items():
            if json.loads(json.dumps(r, sort_keys=True)) != g["results"].get(img):
                print(f"FAIL {img}: output differs from the recorded output")
                ok = False
        if g["fingerprint"] != fp:
            print("NOTE tool or toolchain differs from the recording:", g["fingerprint"], "->", fp)
    else:
        print("FAIL no recorded outputs (--record-golden)")
        ok = False
    if ok:
        json.dump({"fingerprint": fp, "passed": datetime.datetime.now().isoformat(timespec="seconds")},
                  open(os.path.join(GOLDEN, "selftest_pass.json"), "w"), indent=1)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return ok


def roles_of(elf):
    names = sh([B + "nm", elf])
    return [r for r, keys in ROLE_FUNCS.items() if any(k in names for k in keys)]


def one_image(job):
    label, cfg_hash, img, role = job
    r = analyse(img, role)
    return label, cfg_hash, os.path.basename(os.path.dirname(img)), role, r


def compare(pairs, out_json=None):
    """Per config and image: each router's loop stack accesses and spilled values; spills that differ from the first."""
    import glob, multiprocessing
    jobs = []
    for label, root in pairs:
        for img in sorted(glob.glob(f"{root}/*/kernels/fabric_erisc_router_v2/*/*/*.elf")):
            cfg_hash = img.split("/")[-3]
            for role in roles_of(img):
                jobs.append((label, cfg_hash, img, role))
    with multiprocessing.Pool(min(16, os.cpu_count() or 1)) as pool:
        res = pool.map(one_image, jobs)
    table = collections.defaultdict(dict)
    for label, cfg_hash, kind, role, r in res:
        table[(cfg_hash, kind, role)][label] = r
    labels = [l for l, _ in pairs]
    ref = labels[0]
    summary = collections.defaultdict(collections.Counter)
    for key in sorted(table):
        row = table[key]
        print(f"{key[0]} {key[1]} {key[2]}")
        for l in labels:
            r = row.get(l)
            if r is None or "error" in r:
                print(f"  {l:10s} {'missing' if r is None else r['error']}")
                continue
            a = r["accesses"]
            extra = sorted(set(r["stored"]) - set(row[ref]["stored"])) if l != ref and ref in row else []
            gone = sorted(set(row[ref]["stored"]) - set(r["stored"])) if l != ref and ref in row else []
            print(f"  {l:10s} loop {r['loop_insns']:4d}  sp S{a['stores']:<3d} L{a['loads']:<3d} unknown {a['unknown']:<3d}"
                  + (f" NOT TRUSTED {r['untrusted']}" if r["untrusted"] else "")
                  + (f"  +{extra}" if extra else "") + (f"  -{gone}" if gone else ""))
            for v in r["stored"]:
                summary[(l, key[2])][v] += 1
    print("\nspilled values, configs per router and role:")
    for (l, role), c in sorted(summary.items()):
        print(f"  {l} {role}: " + ", ".join(f"{v} x{n}" for v, n in sorted(c.items())))
    if out_json:
        json.dump({f"{k[0]}/{k[1]}/{k[2]}": v for k, v in table.items()}, open(out_json, "w"), indent=1, sort_keys=True)


def main():
    if "--headroom" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        cases = set(sys.argv[sys.argv.index("--cases") + 1].split(",")) if "--cases" in sys.argv else None
        if cases:
            args.remove(sys.argv[sys.argv.index("--cases") + 1])
        out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
        if out:
            args.remove(out)
        role = sys.argv[sys.argv.index("--role") + 1] if "--role" in sys.argv else "receiver"
        if "--role" in sys.argv:
            args.remove(role)
        if os.path.isdir(args[0]):
            sys.exit(1 if headroom_builds(args[0], args[1], cases, out, role) else 0)
        r = headroom(args[0], args[1], role)
        print(json.dumps(r, indent=1))
        sys.exit(0 if r.get("passes") else 1)
    if "--live" in sys.argv:
        role = sys.argv[sys.argv.index("--role") + 1] if "--role" in sys.argv else "receiver"
        at = [int(x, 16) for x in sys.argv[sys.argv.index("--at") + 1].split(",")] if "--at" in sys.argv else []
        skip = {sys.argv.index(f) + 1 for f in ("--role", "--at", "--json") if f in sys.argv}
        elfs = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and i not in skip]
        rs = [liveness(e, role, at) for e in elfs]
        if "--json" in sys.argv:
            json.dump(rs, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
        for r in rs:
            print(live_text(r))
        return
    if "--compare" in sys.argv:
        pairs = [tuple(a.split("=", 1)) for a in sys.argv[1:] if "=" in a]
        compare(pairs, sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None)
        return
    if "--self-test" in sys.argv or "--record-golden" in sys.argv:
        sys.exit(0 if self_test(record="--record-golden" in sys.argv) else 1)
    passed = os.path.join(GOLDEN, "selftest_pass.json")
    if not os.path.exists(passed) or json.load(open(passed))["fingerprint"] != fingerprint():
        print("WARNING: the self-test has not passed for this tool and toolchain (loopvars.py --self-test)")
    role = sys.argv[sys.argv.index("--role") + 1] if "--role" in sys.argv else "receiver"
    skip = {sys.argv.index(f) + 1 for f in ("--role", "--json") if f in sys.argv}
    elfs = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and i not in skip]
    rs = [analyse(e, role) for e in elfs]
    if "--json" in sys.argv:
        json.dump(rs, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, sort_keys=True)
    for r in rs:
        print(text(r) if "error" not in r else r["error"])


if __name__ == "__main__":
    main()
