#!/usr/bin/env python3
"""stack_scan.py NAME [NAME ...]: which program each router config runs, and the stack traffic in its main loop.

For every ELF of coverage builds COV/builds/<NAME>_<opt>/ (coverage.py build), per ERISC:
  - program: "sender" (sender_erisc::run_main_loop is present), "receiver" (receiver_erisc::run_main_loop), else
    "generic" (the generic loop, inlined in kernel_main);
  - for the loop function (run_main_loop, or kernel_main for the generic loop): frame size, callee-saved registers
    saved, and static counts of stack loads and stores (sp-relative lw/lh/lb/sw/sh/sb), prologue and epilogue
    excluded (the callee-saved saves and restores).
Prints, per case and level, the program mix and the median loop counts per program, and writes a JSON per build."""
import collections
import json
import os
import re
import statistics
import subprocess
import sys

COV = "/home/aagarwal/.tt-metal-v3r-builds/cov/builds"
BIN = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
LOADS = ("lw", "lh", "lhu", "lb", "lbu")
STORES = ("sw", "sh", "sb")
SAVED = re.compile(r"^(s([0-9]|1[01])|ra)$")


def functions(elf):
    """{demangled name: (start, end)} for text symbols."""
    out = subprocess.run([BIN + "nm", "-C", "-S", "--defined-only", elf], capture_output=True, text=True).stdout
    f = {}
    for line in out.splitlines():
        m = re.match(r"([0-9a-f]+) ([0-9a-f]+) [tT] (.*)$", line)
        if m:
            s = int(m.group(1), 16)
            f[m.group(3)] = (s, s + int(m.group(2), 16))
    return f


def scan(elf, start, end):
    out = subprocess.run([BIN + "objdump", "-d", "--no-show-raw-insn", f"--start-address={start}",
                          f"--stop-address={end}", elf], capture_output=True, text=True).stdout
    ins = []
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m:
            ins.append((m.group(2), m.group(3).split("#")[0].strip()))
    frame = 0
    for mn, ops in ins[:4]:
        m = re.match(r"sp,sp,-(\d+)", ops.replace(" ", ""))
        if mn in ("addi", "c.addi16sp") and m:
            frame = int(m.group(1))
            break
    saves = {ops.split(",")[0] for mn, ops in ins[:40] if mn in ("sw", "c.swsp") and "(sp)" in ops
             and SAVED.match(ops.split(",")[0])}
    loads = stores = 0
    for mn, ops in ins:
        if "(sp)" not in ops:
            continue
        reg = ops.split(",")[0]
        if reg in saves:  # callee-saved save/restore (prologue, epilogue)
            continue
        if mn in LOADS or mn == "c.lwsp":
            loads += 1
        elif mn in STORES or mn == "c.swsp":
            stores += 1
    return {"frame": frame, "saved": len(saves), "stack_loads": loads, "stack_stores": stores, "insns": len(ins)}


def one(path):
    f = functions(path)
    snd = [k for k in f if "sender_erisc::run_main_loop" in k or k.startswith("void fabric_router::sender::run<")]
    rcv = [k for k in f if "receiver_erisc::run_main_loop" in k or k.startswith("fabric_router::receiver::run(")]
    prog, fn = ("sender", snd[0]) if snd else ("receiver", rcv[0]) if rcv else ("generic", "kernel_main()")
    if fn not in f:
        return {"program": prog, "error": "no loop function"}
    return {"program": prog, **scan(path, *f[fn])}


def main():
    for name in sys.argv[1:]:
        for opt in ("O3", "O2", "Os"):
            d = os.path.join(COV, f"{name}_{opt}")
            if not os.path.isdir(d):
                continue
            res = {}
            for key in sorted(os.listdir(d)):
                for proc in ("active_erisc", "subordinate_active_erisc", "erisc"):  # erisc: Wormhole's one ERISC
                    e = os.path.join(d, key, proc, f"{proc}.elf")
                    if os.path.exists(e):
                        res[f"{key}/{proc}"] = {"case": key.split("__")[0],
                                                "erisc": 1 if proc == "subordinate_active_erisc" else 0, **one(e)}
            json.dump(res, open(os.path.join(d, "stack_scan.json"), "w"), indent=0)
            print(f"\n## {name} -{opt}: {len(res)} ELFs")
            by = collections.defaultdict(list)
            for r in res.values():
                by[(r["case"], r["erisc"])].append(r)
            print(f"{'case':20} ERISC  programs                       loop: frame  saved  stack ld  stack st  (medians)")
            for (case, erisc), rs in sorted(by.items()):
                mix = collections.Counter(r["program"] for r in rs)
                ok = [r for r in rs if "error" not in r]
                med = lambda k: statistics.median(r[k] for r in ok) if ok else float("nan")
                print(f"{case:20} E{erisc}     {', '.join(f'{p} {n}' for p, n in sorted(mix.items())):30} "
                      f"{med('frame'):5.0f} {med('saved'):6.0f} {med('stack_loads'):9.0f} {med('stack_stores'):9.0f}")


if __name__ == "__main__":
    main()
