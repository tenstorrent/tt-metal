#!/usr/bin/env python3
"""regpress.py BUILD [BUILD ...] [--cases a,b]: static register-pressure symptoms of each router build's hot functions,
per case and ERISC (V4_PROFILE.md). BUILD is a coverage build name under COV/builds (e.g. v4trm_O3, v3qm_O3).

Hot functions: v4's sender::run, receiver::run, single::run and the out-of-line single::sender_pass / receiver_pass;
v3's sender_erisc::run_main_loop, receiver_erisc::run_main_loop, or kernel_main (its generic loop). Per function:
frame bytes, callee-saved registers saved, instructions, and, by loop depth (backward branches nest: depth 1 = the run
loop, deeper = rounds, passes, per-channel waits), sp-relative loads and stores (spills and reloads; prologue and
epilogue saves left out), all loads and stores (mem: registers, L1 state, spills), lui/li (constants rematerialized) and calls. The out-of-line pass functions run once per
pass, so their whole body counts as pass code (reported at depth 0 of their own).
Static symptoms only: in this study static counts have not predicted cycles/packet (codegen_diff.py), so this explains a
profile, it does not rank changes."""
import collections, json, os, re, statistics, subprocess, sys

COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m") + "/builds"
BIN = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
SAVED = re.compile(r"^(s([0-9]|1[01])|ra)$")
BRANCH = ("beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "blez", "bgez", "bltz", "bgtz", "j", "c.j",
          "c.beqz", "c.bnez", "bgt", "ble", "bgtu", "bleu")


def functions(elf):
    out = subprocess.run([BIN + "nm", "-C", "-S", "--defined-only", elf], capture_output=True, text=True).stdout
    f = {}
    for line in out.splitlines():
        m = re.match(r"([0-9a-f]+) ([0-9a-f]+) [tT] (.*)$", line)
        if m:
            s = int(m.group(1), 16)
            f[m.group(3)] = (s, s + int(m.group(2), 16))
    return f


def disasm(elf, start, end):
    out = subprocess.run([BIN + "objdump", "-d", "--no-show-raw-insn", f"--start-address={start}",
                          f"--stop-address={end}", elf], capture_output=True, text=True).stdout
    ins = []
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m:
            ins.append((int(m.group(1), 16), m.group(2), m.group(3).split("#")[0].strip()))
    return ins


def scan(elf, start, end):
    ins = disasm(elf, start, end)
    loops = []  # (target, branch address): a backward branch closes a loop
    for a, mn, ops in ins:
        if mn in BRANCH:
            m = re.search(r"\b([0-9a-f]+) <", ops) or re.search(r"(?:^|,)\s*([0-9a-f]+)\s*$", ops)
            if m:
                t = int(m.group(1), 16)
                if start <= t <= a:
                    loops.append((t, a))
    depth = lambda a: sum(1 for t, b in loops if t <= a <= b)
    frame = 0
    for _, mn, ops in ins[:4]:
        m = re.match(r"sp,sp,-(\d+)", ops.replace(" ", ""))
        if mn in ("addi", "c.addi16sp") and m:
            frame = int(m.group(1))
            break
    saves = {ops.split(",")[0] for _, mn, ops in ins[:40] if mn in ("sw", "c.swsp") and "(sp)" in ops
             and SAVED.match(ops.split(",")[0])}
    by = collections.defaultdict(lambda: collections.Counter())
    for a, mn, ops in ins:
        d = min(depth(a), 4)
        reg = ops.split(",")[0]
        if "(sp)" in ops and reg not in saves:
            if mn in ("lw", "lh", "lhu", "lb", "lbu", "c.lwsp"):
                by[d]["ld"] += 1
            elif mn in ("sw", "sh", "sb", "c.swsp"):
                by[d]["st"] += 1
        if mn in ("lw", "lh", "lhu", "lb", "lbu", "c.lw", "c.lwsp"):
            by[d]["mld"] += 1
        elif mn in ("sw", "sh", "sb", "c.sw", "c.swsp"):
            by[d]["mst"] += 1
        if mn in ("lui", "li", "c.lui", "c.li"):
            by[d]["const"] += 1
        if mn in ("jal", "jalr", "call", "c.jal", "c.jalr") and not ops.startswith("zero"):
            by[d]["call"] += 1
        by[d]["ins"] += 1
    return {"frame": frame, "saved": len(saves), "insns": len(ins), "max_depth": max((depth(a) for a, _, _ in ins), default=0),
            "by_depth": {d: dict(c) for d, c in sorted(by.items())}}


HOT = [("v4 sender", "void fabric_router::sender::run<"), ("v4 receiver", "void fabric_router::receiver::run<"),
       ("v4 receiver", "fabric_router::receiver::run("),
       ("v4 single", "void fabric_router::single::run<"), ("v4 single.sender_pass", "void fabric_router::single::sender_pass<"),
       ("v4 single.receiver_pass", "void fabric_router::single::receiver_pass<"),
       ("v4 idle", "fabric_router::run_idle("),
       ("v3 sender", "sender_erisc::run_main_loop"), ("v3 receiver", "receiver_erisc::run_main_loop")]


def one(path):
    f = functions(path)
    res = {}
    for tag, pref in HOT:
        for k in f:
            if pref in k and tag not in res:
                res[tag] = scan(path, *f[k])
    if not any(t.startswith(("v4 sender", "v4 receiver", "v4 single", "v4 idle", "v3 sender", "v3 receiver")) for t in res):
        if "kernel_main()" in f:
            res["kernel_main (generic loop)"] = scan(path, *f["kernel_main()"])
    return res


def summarize(r, d0):
    """loads/stores/consts/calls at depth >= d0."""
    tot = collections.Counter()
    for d, c in r["by_depth"].items():
        if int(d) >= d0:
            tot.update(c)
    return tot


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cases = None
    for a in sys.argv[1:]:
        if a.startswith("--cases="):
            cases = set(a.split("=", 1)[1].split(","))
    for b in args:
        d = os.path.join(COV, b)
        res = {}
        for key in sorted(os.listdir(d)):
            case = key.split("__")[0]
            if cases and case not in cases:
                continue
            for proc in ("active_erisc", "subordinate_active_erisc"):
                e = os.path.join(d, key, proc, f"{proc}.elf")
                if os.path.exists(e):
                    res[f"{key}/{proc}"] = {"case": case, "erisc": 1 if proc.startswith("sub") else 0, "fns": one(e)}
        json.dump(res, open(os.path.join(d, "regpress.json"), "w"), indent=0)
        print(f"\n## {b}: {len(res)} ELFs  (medians per case / ERISC / function; 'pass' = loop depth >= 3, or the whole"
              f" body of an out-of-line pass function)")
        print(f"{'case':20}{'E':>2} {'function':26}{'n':>4}{'frame':>6}{'saved':>6}{'insns':>6}{'pass ins':>9}"
              f"{'pass ld':>8}{'pass st':>8}{'mem ld':>7}{'mem st':>7}{'consts':>7}{'calls':>6}{'loop ld/st':>11}")
        by = collections.defaultdict(list)
        for r in res.values():
            for fn, m in r["fns"].items():
                by[(r["case"], r["erisc"], fn)].append(m)
        for (case, erisc, fn), ms in sorted(by.items()):
            med = lambda f: statistics.median(f(m) for m in ms)
            p0 = 0 if "pass" in fn else 3
            pass_ = lambda m, k: summarize(m, p0)[k]
            loop = lambda m: summarize(m, 1)["ld"] + summarize(m, 1)["st"]
            print(f"{case:20}{erisc:>2} {fn:26}{len(ms):4}{med(lambda m: m['frame']):6.0f}{med(lambda m: m['saved']):6.0f}"
                  f"{med(lambda m: m['insns']):6.0f}{med(lambda m: pass_(m, 'ins')):9.0f}{med(lambda m: pass_(m, 'ld')):8.0f}"
                  f"{med(lambda m: pass_(m, 'st')):8.0f}{med(lambda m: pass_(m, 'mld')):7.0f}{med(lambda m: pass_(m, 'mst')):7.0f}"
                  f"{med(lambda m: pass_(m, 'const')):7.0f}"
                  f"{med(lambda m: pass_(m, 'call')):6.0f}{med(loop):11.0f}")


if __name__ == "__main__":
    main()
