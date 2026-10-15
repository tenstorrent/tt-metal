#!/usr/bin/env python3
"""Replay scan of unpack ELFs (tt-llk harness unpack.elf, ttnn trisc0.elf) for the datacopy family's replay bodies.
The block calls replay a per tile body recorded into slots (0, 4), (4, 4) (tilize and one-face unpack A halves), (0, 6),
(6, 6) (two faces) or (0, 10), (10, 10) (four faces). For every replay (TTREPLAY without the record bit) the scan takes the
records over its slots that can precede it in the same function (a call to a recording function counts as a record at the
call site) and flags: (S) the nearest earlier record over its slots records a different (start, length) than the replay
plays, with no record of the played program between; (L) a replay inside a loop whose body, after the replay, records a
different program over its slots, with the played program not recorded again in the body before the replay.
The block calls' run time guard (block_replay_body) is not modelled: a flagged ELF is read by hand.
usage: replay_scan_udc.py [--selftest] <ELF or dir>...   (objdump: sfpi 7.86.0)"""
import glob, os, re, subprocess, sys
OBJ = os.environ.get("REPLAY_OBJDUMP", "/proj_sw/user_dev/mvlahovic/reviewing_prs/sfpi-7.86.0/sfpi/compiler/bin/riscv-tt-elf-objdump")
BR = re.compile(r"\t(beq|bne|blt|bge|bltu|bgeu|beqz|bnez|blez|bgez|bltz|bgtz|bgt|ble|bgtu|bleu|j)\t(?:.*,)?([0-9a-f]+) <")
def parse(text):
    funcs, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if m: cur = m.group(2); funcs[cur] = []; continue
        if cur is None: continue
        m = re.match(r"^\s*([0-9a-f]+):\t[0-9a-f]+\s+\t?(\S+)\t?(.*)$", line)
        if not m: continue
        a, op, args = int(m.group(1), 16), m.group(2), m.group(3)
        if op == "ttreplay":
            s, l, e, r = (int(x) for x in args.split(",")[:4])
            funcs[cur].append((a, "REC" if r else "PLAY", s, l))
        elif op in ("jal", "call") and "<" in args:
            funcs[cur].append((a, "CALL", args.split("<")[1].rstrip(">").split("+")[0], 0))
        else:
            b = BR.search(line)
            if b and int(b.group(2), 16) <= a: funcs[cur].append((a, "LOOP", int(b.group(2), 16), 0))
    return funcs
def sl(s, l): return frozenset(range(s, s + l))
def scan(funcs):
    crec = {f: [(s, l) for _, k, s, l in ev if k == "REC"] for f, ev in funcs.items()}
    probs, nplay, nrec = [], 0, 0
    for f, ev in funcs.items():
        recs = [(a, (s, l)) for a, k, s, l in ev if k == "REC"]
        recs += [(a, p) for a, k, c, _ in ev if k == "CALL" and c in crec and c != f for p in crec[c]]
        recs.sort(); nrec += len(recs)
        loops = [(t, a) for a, k, t, _ in ev if k == "LOOP"]
        for a, k, s, l in ev:
            if k != "PLAY": continue
            nplay += 1; ps = sl(s, l)
            prior = [r for r in recs if r[0] < a and sl(*r[1]) & ps]
            if prior and prior[-1][1] != (s, l):
                probs.append(f"{f}: (S) replay {s},{l} at {a:x}: nearest record over its slots is {prior[-1][1]} at {prior[-1][0]:x}")
            for t, b in loops:
                if not (t <= a <= b): continue
                later = [r for r in recs if a < r[0] <= b and sl(*r[1]) & ps and r[1] != (s, l)]
                again = [r for r in recs if t <= r[0] < a and r[1] == (s, l)]
                if later and not again:
                    probs.append(f"{f}: (L) replay {s},{l} at {a:x} in loop {t:x}..{b:x}: record {later[0][1]} at {later[0][0]:x} in the body")
    return nplay, nrec, probs
SELF = {"straight other program": ("00001000 <k>:\n    1000:\t0\tttreplay\t0,4,0,1\n    1004:\t0\tttreplay\t0,10,0,1\n    1008:\t0\tttreplay\t0,4,0,0\n", True),
        "loop record of other program after play": ("00001000 <k>:\n    1000:\t0\tttreplay\t0,10,0,1\n    1004:\t0\tttreplay\t0,10,0,0\n    1008:\t0\tttreplay\t0,4,0,1\n    100c:\t0\tbnez\ta5,1004 <k+0x4>\n", True),
        "clean": ("00001000 <k>:\n    1000:\t0\tttreplay\t0,10,0,1\n    1004:\t0\tttreplay\t10,10,0,1\n    1008:\t0\tttreplay\t0,10,0,0\n    100c:\t0\tttreplay\t10,10,0,0\n    1010:\t0\tbnez\ta5,1008 <k+0x8>\n", False)}
def main():
    a = sys.argv[1:]
    if a and a[0] == "--selftest":
        ok = True
        for n, (t, want) in SELF.items():
            p = scan(parse(t))[2]; good = bool(p) == want; ok &= good
            print(f"selftest {'flagged' if p else 'clean'} {'ok' if good else 'WRONG'}: {n}")
        print("# selftest", "passed" if ok else "FAILED"); return
    elfs = []
    for x in a:
        elfs += (glob.glob(x + "/**/elf/unpack.elf", recursive=True) + glob.glob(x + "/**/trisc0/trisc0.elf", recursive=True)) if os.path.isdir(x) else [x]
    n = bad = withplay = 0
    for e in sorted(elfs):
        np_, nr, p = scan(parse(subprocess.run([OBJ, "-d", e], capture_output=True, text=True).stdout))
        n += 1; withplay += np_ > 0; bad += bool(p)
        for q in p: print("BAD", e, q)
    print(f"# {n} ELFs, {withplay} with replays, {bad} flagged")
main()
