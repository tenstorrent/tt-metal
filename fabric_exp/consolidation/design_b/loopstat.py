#!/usr/bin/env python3
"""loopstat.py A B [--cases=c1,c2] [--detail]: per case of two coverage builds, the receiver loop function, B against A.

The receiver loop function: the symbol containing "receiver3run" (two ERISCs, E1) or "receiver_pass" (one ERISC).
Per image: instructions; stack accesses (sp-relative loads and stores) split into
  saves   callee-saved register saves and restores (sw/lw of ra, s0-s11 at the frame's top),
  init    zero stores to the frame (sw/sh/sb zero,X(sp): state initialisation before the loop),
  body    every other sp-relative access (spills, reloads, frame objects addressed directly);
the frame size, callee-saved registers used, and mul count. Comparison class of the function's text:
  identical  same bytes
  layout     same instructions after normalizing branch targets, addresses and sp offsets
  regs       same mnemonic sequence (registers or immediates differ)
  differ     otherwise.
Prints per case the classes and the distribution of (d_insns, d_body) over its receiver images."""
import collections, concurrent.futures, glob, os, re, subprocess, sys

COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m") + "/builds"
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
KEYS = ("receiver3run", "receiver_pass")
SAVED = re.compile(r"^(sw|lw)\s+(ra|s\d+),-?\d+\(sp\)$")
ZERO = re.compile(r"^(sw|sh|sb)\s+zero,-?\d+\(sp\)$")
ARGS = ("MY_DIRECTION", "NUM_RECEIVER_CHANNELS", "ACTUAL_VC1_SENDER_CHANNELS", "ACTUAL_VC2_SENDER_CHANNELS",
        "ENABLE_FIRST_LEVEL_ACK_VC0", "ENABLE_SPEEDY_VC0", "IS_RECEIVER_CHANNEL_0_SERVICED", "IS_RECEIVER_CHANNEL_1_SERVICED",
        "IS_RECEIVER_CHANNEL_2_SERVICED", "DISABLE_RX_CH0_FORWARDING", "DISABLE_RX_CH1_FORWARDING",
        "IS_INTERMESH_ROUTER", "IS_INTRAMESH_ROUTER_ON_EDGE", "NUM_ACTIVE_ERISCS")
SHORT = ("dir", "nrx", "vc1", "vc2", "fla", "spd", "s0", "s1", "s2", "d0", "d1", "im", "edge", "ne")


def config(elf):
    """Key compile-time args of an image (named_ct_arg_map_generated.h next to its proc dir)."""
    h = os.path.join(os.path.dirname(os.path.dirname(elf)), "named_ct_arg_map_generated.h")
    try:
        t = open(h).read()
    except OSError:
        return ""
    v = dict(re.findall(r'\{"(\w+)",(\d+)\}', t))
    return " ".join(f"{s}={v.get(a, '?')}" for a, s in zip(ARGS, SHORT))


def loop_fn(elf):
    for l in subprocess.run([B + "nm", "-S", elf], capture_output=True, text=True).stdout.splitlines():
        p = l.split()
        if len(p) == 4 and any(k in p[3] for k in KEYS):
            return int(p[0], 16), int(p[1], 16)
    return None


def stats(elf):
    f = loop_fn(elf)
    if f is None:
        return None
    s, sz = f
    d = subprocess.run([B + "objdump", "-d", f"--start-address={s}", f"--stop-address={s + sz}", elf],
                       capture_output=True, text=True).stdout
    raw, ins = [], []
    for l in d.splitlines():
        m = re.match(r"\s*[0-9a-f]+:\s+([0-9a-f]+)\s+(.*)$", l)
        if m:
            raw.append(m.group(1))
            ins.append(re.sub(r"\s+", " ", re.sub(r"\s*#.*$", "", m.group(2).strip())).replace(", ", ","))
    st = {"insns": len(ins), "mul": sum(1 for t in ins if re.match(r"mul\b", t))}
    stack = [t for t in ins if "(sp)" in t and not t.startswith("addi")]
    st["saves"] = sum(1 for t in stack if SAVED.match(t))
    st["init"] = sum(1 for t in stack if ZERO.match(t))
    st["body"] = len(stack) - st["saves"] - st["init"]
    st["sregs"] = len({SAVED.match(t).group(2) for t in stack if SAVED.match(t)})
    fr = [int(m.group(1)) for m in (re.match(r"addi sp,sp,-(\d+)$", t) for t in ins) if m]
    st["frame"] = fr[0] if fr else 0
    st["raw"] = tuple(raw)
    lay = [re.sub(r"[0-9a-f]+ <[^>]*>", "T", t) for t in ins]
    lay = [re.sub(r"-?\d+\(sp\)", "N(sp)", t) for t in lay]
    lay = [re.sub(r"addi (\w+),sp,-?\d+", r"addi \1,sp,N", t) for t in lay]
    st["layout"] = tuple(lay)
    st["mnem"] = tuple(t.split(" ")[0] for t in ins)
    return st


def cls(sa, sb):
    if sa["raw"] == sb["raw"]:
        return "identical"
    if sa["layout"] == sb["layout"]:
        return "layout"
    if sa["mnem"] == sb["mnem"]:
        return "regs"
    return "differ"


def main():
    a, b = sys.argv[1], sys.argv[2]
    cases = None
    for x in sys.argv[3:]:
        if x.startswith("--cases"):
            cases = set(x.split("=", 1)[1].split(","))
    detail = "--detail" in sys.argv
    pairs = []
    for e in glob.glob(f"{COV}/{a}/*/*/*.elf"):
        rel = os.path.relpath(e, f"{COV}/{a}")
        case = rel.split("__")[0]
        if cases and case not in cases:
            continue
        if os.path.exists(f"{COV}/{b}/{rel}"):
            pairs.append((case, rel, e, f"{COV}/{b}/{rel}"))
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        res = list(ex.map(lambda p: (p[0], p[1], stats(p[2]), stats(p[3])), pairs))
    per = collections.defaultdict(list)
    for case, rel, sa, sb in res:
        if sa is None or sb is None:
            continue
        per[case].append((rel, sa, sb))
    print(f"{'case':24s} {'n':>3s} {'ident/lay/regs/diff':>19s}  (d_insns,d_body) x count | base: insns body sregs frame")
    for case in sorted(per):
        rows = per[case]
        c = collections.Counter(cls(sa, sb) for _, sa, sb in rows)
        dist = collections.Counter((sb["insns"] - sa["insns"], sb["body"] - sa["body"]) for _, sa, sb in rows)
        ds = " ".join(f"({di:+d},{dk:+d})x{n}" for (di, dk), n in sorted(dist.items()))
        rng = lambda k, s: f"{min(x[s][k] for x in [(None, r[1], r[2]) for r in rows])}-{max(x[s][k] for x in [(None, r[1], r[2]) for r in rows])}"
        print(f"{case:24s} {len(rows):3d} {c['identical']:4d}/{c['layout']:3d}/{c['regs']:3d}/{c['differ']:4d}  {ds:44s}| "
              f"{rng('insns', 1)} {rng('body', 1)} {rng('sregs', 1)} {rng('frame', 1)}")
        if detail:
            for rel, sa, sb in sorted(rows):
                print(f"    {rel.split('/')[0]:46s} {config(f'{COV}/{a}/{rel}')}")
                print(f"        {cls(sa, sb):9s} insns {sa['insns']}->{sb['insns']} ({sb['insns'] - sa['insns']:+d})  "
                      f"body {sa['body']}->{sb['body']}  saves {sa['saves']}->{sb['saves']}  init {sa['init']}->{sb['init']}  "
                      f"sregs {sa['sregs']}->{sb['sregs']}  frame {sa['frame']}->{sb['frame']}  mul {sa['mul']}->{sb['mul']}")


if __name__ == "__main__":
    main()
