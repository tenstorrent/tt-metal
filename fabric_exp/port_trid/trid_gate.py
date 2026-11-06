#!/usr/bin/env python3
"""trid_gate.py: the per-channel transaction-ID port's offline gate (TRID_ALLOCATION.md section 7), over mock
captures of the port-trid build (38bce02a657 + trid_uniform_switch.patch) in two arms: RULE (the host's per-channel
rule, the commit) and UNIFORM (TT_FABRIC_EXP_TRID_UNIFORM=1: every channel the earlier 16 / 8 / 4).

  trid_gate.py table RULE_DIR UNIFORM_DIR [CASE ...]   per case: the receiver ID rows each arm logged (the host's
        "receiver transaction IDs" lines, TT_FABRIC_EXP_LOG_SLOTS=1), one row per distinct router: per receiver
        channel slots, first-level acks, G, groups, ids@start ("!": shared by mask), with the router count
  trid_gate.py images RULE_DIR UNIFORM_DIR [CASE ...]  per case and (ERISC, direction): router builds paired by their
        named args other than RX_CH_<c>_NUM_TRIDS / _TRID_START, and whether their loadable images are identical
  trid_gate.py loads RULE_DIR UNIFORM_DIR CASE DIRS    receiver::run of the paired E1 builds facing DIRS (e.g. NS):
        every stack load (lw/lh/lb.. off sp) by its innermost source line, both arms, and the ones only one arm has;
        the decode site (fabric_common.h, decode_action_y_first) and the proof site (router_receiver.hpp,
        trid_writes_sent) listed apart
A DIR holds one directory per case (rc_capture.sh's RC/port, or cov.py's COV/cases)."""
import collections
import glob
import hashlib
import os
import re
import struct
import subprocess
import sys

TOOL = "/home/aagarwal/tt-metal-port-trid/runtime/sfpi/compiler/bin/riscv-tt-elf-"
CLEAN = re.compile(r"\x1b\[[0-9;]*m")
DIRS = {0: "E", 1: "W", 2: "N", 3: "S", 4: "Z"}
LINE = re.compile(r"receiver transaction IDs \((uniform rule|rule)\): RX_CH_(\d) G (\d+) ids (\d+) start (\d+): (\d+) slots "
                  r"in groups of \d+( \(first-level acks\))?: (\d+) groups")
TRID_ARGS = re.compile(r"RX_CH_\d_(NUM_TRIDS|TRID_START)$")


def cases(a, b, want):
    names = sorted({os.path.basename(p) for d in (a, b) for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)})
    return [n for n in names if not want or n in want]


def id_rows(case_dir):
    """Distinct routers: the consecutive RX_CH_0.. lines of one router's args, as a row; with their count."""
    rows = collections.Counter()
    for log in glob.glob(os.path.join(case_dir, "**", "*.log"), recursive=True):
        cur = []
        for line in open(log, errors="replace"):
            m = LINE.search(CLEAN.sub("", line))
            if not m:
                continue
            _, c, g, ids, start, slots, fla, groups = m.groups()
            if c == "0" and cur:
                rows[tuple(cur)] += 1
                cur = []
            cur.append((int(slots), bool(fla), int(g), int(groups), int(ids), int(start)))
        if cur:
            rows[tuple(cur)] += 1
    return rows


def fmt_row(row):
    return " | ".join(f"{s}{'a' if fla else ''}: G{g} {groups}gr {ids}@{start}{'!' if ids < groups else ''}"
                      for s, fla, g, groups, ids, start in row)


def table(a, b, want):
    print("| case | rule (routers) | uniform (routers) |")
    print("| --- | --- | --- |")
    for c in cases(a, b, want):
        ra, rb = id_rows(os.path.join(a, c)), id_rows(os.path.join(b, c))
        cell = lambda r: "; ".join(f"{fmt_row(k)} (x{v})" for k, v in sorted(r.items())) or "-"  # noqa: E731
        print(f"| {c} | {cell(ra)} | {cell(rb)} |")


def named(h):
    return dict((k, int(v)) for k, v in re.findall(r'\{"([A-Z0-9_]+)",(\d+)\}', open(h).read()))


def image(elf):
    b = open(elf, "rb").read()
    phoff, = struct.unpack_from("<I", b, 0x1C)
    phentsize, phnum = struct.unpack_from("<HH", b, 0x2A)
    h = hashlib.sha1()
    for i in range(phnum):
        p_type, off, vaddr, _, filesz, memsz, _, _ = struct.unpack_from("<8I", b, phoff + i * phentsize)
        if p_type == 1:
            h.update(struct.pack("<II", vaddr, memsz))
            h.update(b[off: off + filesz])
    return h.hexdigest()


def builds(case_dir):
    out = collections.defaultdict(list)  # key -> [elf]
    for elf in glob.glob(os.path.join(case_dir, "**", "kernels", "fabric_erisc_router", "*", "*", "*.elf"), recursive=True):
        if elf.endswith(".xip.elf"):
            continue
        proc = os.path.basename(os.path.dirname(elf))
        n = named(os.path.join(os.path.dirname(os.path.dirname(elf)), "named_ct_arg_map_generated.h"))
        key = (proc,) + tuple(sorted((k, v) for k, v in n.items() if not TRID_ARGS.match(k)))
        out[key].append(elf)
    return out


def erisc(proc):
    return "E0" if proc == "active_erisc" else "E1"


def direction(key):
    return DIRS.get(dict(key[1:]).get("MY_DIRECTION"), "?")


def images(a, b, want):
    for c in cases(a, b, want):
        ba, bb = builds(os.path.join(a, c)), builds(os.path.join(b, c))
        g = collections.defaultdict(collections.Counter)
        for key in set(ba) | set(bb):
            cell = g[(erisc(key[0]), direction(key))]
            if key not in ba or key not in bb:
                cell["unpaired"] += 1
                continue
            cell["paired"] += 1
            same = sorted(map(image, ba[key])) == sorted(map(image, bb[key]))
            cell["identical" if same else "differ"] += 1
        print(f"{c}: " + ", ".join(f"{e}/{d}: {v['paired']} paired, {v['differ']} differ"
                                    + (f", {v['unpaired']} unpaired" if v["unpaired"] else "")
                                    for (e, d), v in sorted(g.items())))


def sh(args, inp=None):
    return subprocess.run(args, capture_output=True, text=True, input=inp).stdout


def stack_loads(elf, fn=r"receiver::run[<(]"):
    """(source line, function, instruction) of every stack load in the function, innermost inline frame first."""
    for line in sh([TOOL + "nm", "-C", "-S", elf]).splitlines():
        p = line.split(None, 3)
        if len(p) == 4 and re.search(fn, p[3]):
            lo, size = int(p[0], 16), int(p[1], 16)
            break
    else:
        return []
    dis = sh([TOOL + "objdump", "-d", "--no-show-raw-insn", f"--start-address={lo}", f"--stop-address={lo + size}", elf])
    ins = []
    for line in dis.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+(.*)$", line)
        if m:
            t = re.sub(r"\s*#.*$", "", re.sub(r" <[^>]*>", "", m.group(2))).strip()
            if re.match(r"(lw|lh|lhu|lb|lbu|c\.lwsp)\s", t) and "(sp)" in t or t.startswith("c.lwsp"):
                ins.append((int(m.group(1), 16), t))
    if not ins:
        return []
    out = sh([TOOL + "addr2line", "-f", "-i", "-C", "-a", "-e", elf], "\n".join(hex(a) for a, _ in ins) + "\n")
    frames, cur, res = {}, None, []
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("0x"):
            cur = int(lines[i], 16)
            frames[cur] = []
            i += 1
            continue
        frames[cur].append((lines[i], lines[i + 1] if i + 1 < len(lines) else "?"))
        i += 2
    for a, t in ins:
        f = frames.get(a, [("?", "?")])
        func, loc = f[0]
        loc = re.sub(r" \(discriminator \d+\)", "", loc)
        res.append((os.path.basename(loc), func.split("(")[0][-60:], re.sub(r"\b(a[0-7]|t[0-6]|s[0-9]|s1[01]|ra)\b", "R", t),
                    " < ".join(f"{fn.split('(')[0][-40:]}@{os.path.basename(l)}" for fn, l in f[:4])))
    return res


def loads(a, b, case, dirs):
    ba, bb = builds(os.path.join(a, case)), builds(os.path.join(b, case))
    pairs = [k for k in ba if k in bb and erisc(k[0]) == "E1" and direction(k) in dirs]
    print(f"{case}: {len(pairs)} paired E1 builds facing {dirs}")
    for k in sorted(pairs, key=str):
        la, lb = stack_loads(ba[k][0]), stack_loads(bb[k][0])
        ca = collections.Counter((s, f, t) for s, f, t, _ in la)
        cb = collections.Counter((s, f, t) for s, f, t, _ in lb)
        only_a, only_b = ca - cb, cb - ca
        n = dict(k[1:])
        print(f"\n## {direction(k)} receiver, MY_ETH_CHANNEL {n.get('MY_ETH_CHANNEL')}, deadlock avoidance "
              f"{n.get('ENABLE_DEADLOCK_AVOIDANCE')}: stack loads in receiver::run rule {len(la)}, uniform {len(lb)}")
        for name, only in (("rule only", only_a), ("uniform only", only_b)):
            for (s, f, t), v in sorted(only.items()):
                print(f"   {name}: {t:24} {s:40} {f} (x{v})")
        for name, lst in (("rule", la), ("uniform", lb)):
            for s, f, t, chain in lst:
                if "fabric_common.h" in chain or "decode_action" in chain:
                    print(f"   decode site, {name}: {t:24} {chain}")
                if "trid_writes_sent" in chain or "proof" in chain.lower():
                    print(f"   proof site, {name}: {t:24} {chain}")


def main():
    mode, a, b = sys.argv[1:4]
    rest = sys.argv[4:]
    if mode == "table":
        table(a, b, rest)
    elif mode == "images":
        images(a, b, rest)
    elif mode == "loads":
        loads(a, b, rest[0], rest[1] if len(rest) > 1 else "NS")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
