#!/usr/bin/env python3
"""Count TT vs TTI Tensix instructions in compiled kernels using elfray.

TTI = Tensix instruction embedded directly in the code (objdump mnemonic starting with "tt").
TT  = Tensix instruction built at run time and stored into the instruction buffer; elfray marks
      these store sites with "; tensix issue <OPCODE>".
elfray only marks a store when it can see that the base register points at the instruction buffer.
When the pointer was loaded earlier (e.g. `lui a4,0xffe40` at the top of the function and many
`sw x,0(a4)` later), the store is not marked. So "TT (elfray)" is a lower bound, and "TT (estimate)"
adds a linear register-tracking pass (see count_buffer_stores) that ignores control flow.
Counts are static (instructions in the binary), not dynamic (how many times they execute).

Usage: elfray_tt_tti.py --elfray E --objdump O --artefacts DIR --label NAME --md OUT.md --json OUT.json
                        [--orig eltwise_binary_test.cpp] [--hal eltwise_binary_hal_test.cpp]
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

TESTS = {"original": "eltwise_binary_test.cpp", "hal": "eltwise_binary_hal_test.cpp"}
KERNELS = ["unpack", "math", "pack"]
LINE_RE = re.compile(r"^\s+0x[0-9a-f]+\s+(\S+)")
ISSUE_RE = re.compile(r"; tensix issue ([A-Z0-9_?]+)")


def count_buffer_stores(lines):
    """Stores into the instruction buffer, found by tracking registers that hold 0xffe40000."""
    ptr = set()
    n = 0
    for line in lines:
        if not LINE_RE.match(line):
            if line.strip().endswith(":"):
                ptr = set()
            continue
        body = line.split(";", 1)[0]
        parts = body.split(None, 2)
        mnem = parts[1]
        ops = [o.strip() for o in (parts[2].split("#", 1)[0] if len(parts) > 2 else "").split(",")]
        if mnem in ("sw", "sh", "sb"):
            m = re.match(r"(-?\w*)\((\w+)\)", ops[-1])
            if (m and m.group(1) in ("", "0") and m.group(2) in ptr) or "<__instrn_buffer>" in line:
                n += 1
            continue
        if not ops or not ops[0]:
            continue
        dst = ops[0]
        if mnem == "lui" and len(ops) > 1 and ops[1] == "0xffe40":
            ptr.add(dst)
        elif mnem == "mv" and len(ops) > 1 and ops[1] in ptr:
            ptr.add(dst)
        elif "<__instrn_buffer>" in line and mnem in ("addi", "lw"):
            ptr.add(dst)
        elif not mnem.startswith(("tt", "b", "j")) and mnem not in ("ret", "fence"):
            ptr.discard(dst)
    return n


def analyse(elfray, elf):
    out = subprocess.run([elfray, "--section", ".text", "--no-color", elf],
                         capture_output=True, text=True, check=True).stdout
    total = 0
    tti = collections.Counter()
    tt = collections.Counter()
    for line in out.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        total += 1
        mnem = m.group(1)
        if mnem.startswith("tt"):
            tti[mnem] += 1
        i = ISSUE_RE.search(line)
        if i:
            tt[i.group(1)] += 1
    return total, tti, tt, count_buffer_stores(out.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elfray", required=True)
    ap.add_argument("--objdump", required=True)
    ap.add_argument("--artefacts", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--orig", default=TESTS["original"])
    ap.add_argument("--hal", default=TESTS["hal"])
    a = ap.parse_args()
    TESTS["original"], TESTS["hal"] = a.orig, a.hal
    os.environ["ELFRAY_OBJDUMP"] = a.objdump

    res = {}
    for tname, tdir in TESTS.items():
        base = os.path.join(a.artefacts, "sources", tdir)
        variants = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        res[tname] = {"variants": len(variants)}
        for k in KERNELS:
            tot = 0
            tti = collections.Counter()
            tt = collections.Counter()
            est = 0
            n = 0
            for v in variants:
                elf = os.path.join(base, v, "elf", f"{k}.elf")
                if not os.path.exists(elf):
                    continue
                t, i, s, e = analyse(a.elfray, elf)
                est += e
                tot += t
                tti.update(i)
                tt.update(s)
                n += 1
            res[tname][k] = {
                "elfs": n,
                "insns_avg": tot / n,
                "tti_avg": sum(tti.values()) / n,
                "tt_avg": sum(tt.values()) / n,
                "tt_estimate_avg": est / n,
                "tti_by_mnemonic_avg": {m: c / n for m, c in tti.most_common()},
                "tt_by_opcode_avg": {m: c / n for m, c in tt.most_common()},
            }

    with open(a.json, "w") as f:
        json.dump({"label": a.label, "results": res}, f, indent=2)

    def fmt(x):
        return f"{x:.1f}".rstrip("0").rstrip(".")

    L = [f"# TT / TTI (elfray) — {a.label}", "",
         f"Variants: original {res['original']['variants']}, HAL {res['hal']['variants']}. "
         "Static count (in the binary file), average per variant. TT elfray = stores that elfray marks "
         "(`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register "
         "that holds the buffer address (0xffe40000). Per opcode (below) only the stores that elfray marks are listed.", "",
         "| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |",
         "|---|---|---|---|---|---|"]
    for k in KERNELS:
        o, h = res["original"][k], res["hal"][k]
        L.append(f"| {k} | {fmt(o['insns_avg'])} / {fmt(h['insns_avg'])} | {fmt(o['tti_avg'])} / {fmt(h['tti_avg'])} | "
                 f"{fmt(o['tt_avg'])} / {fmt(h['tt_avg'])} | {fmt(o['tt_estimate_avg'])} / {fmt(h['tt_estimate_avg'])} | "
                 f"{fmt(o['tti_avg'] + o['tt_estimate_avg'])} / {fmt(h['tti_avg'] + h['tt_estimate_avg'])} |")
    for k in KERNELS:
        for tname in TESTS:
            r = res[tname][k]
            L += ["", f"**{k}, {tname}** — TTI: " + (", ".join(f"`{m}` {fmt(c)}" for m, c in r["tti_by_mnemonic_avg"].items()) or "—"),
                  "", f"TT: " + (", ".join(f"`{m}` {fmt(c)}" for m, c in r["tt_by_opcode_avg"].items()) or "—")]
    with open(a.md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:9]))


if __name__ == "__main__":
    sys.exit(main())
