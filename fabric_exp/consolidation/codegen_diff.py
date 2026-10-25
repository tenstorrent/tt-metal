#!/usr/bin/env python3
"""codegen_diff.py BASE_BUILD_DIR NAME=BUILD_DIR...: how much each build's router code differs from the base build.

Builds come from offline_build.py (or any dir of <hash>/<proc>/<proc>.elf). Per build and ERISC: how many configs'
.text differ from the base (same instruction stream at the same addresses counts as equal), and the median and
max change in .text instructions and in the largest stack frame. Then, for the hot configs (E/W E0 sending, middle
West E1 forwarding), the instruction count, frame, and how many instructions differ in a line diff against the base.
Static numbers only: RESULTS shows they do not predict cycles/packet, so this says *whether* and *where* code moved,
and hardware says by how much it matters.
"""
import concurrent.futures, difflib, glob, os, re, statistics, subprocess, sys

OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
HOT = {"E0 E/W": "10733501891889981293/active_erisc", "E1 mid-W": "10856749301170935304/subordinate_active_erisc"}


def text(elf):
    out = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    ins = [re.sub(r"\s*#.*$", "", l.split("\t", 2)[-1]) for l in out.splitlines() if re.match(r"\s+[0-9a-f]+:\t", l)]
    ins = [re.sub(r" <[^>]*>", "", x) for x in ins]
    frames = sorted((int(m) for m in re.findall(r"addi\s+sp,sp,-(\d+)", out)), reverse=True) + [0]
    return ins, frames[0]


def load(d):
    elfs = {os.path.relpath(e, d)[: -len(".elf")].rsplit("/", 1)[0]: e for e in glob.glob(f"{d}/*/*/*.elf")}
    with concurrent.futures.ThreadPoolExecutor(48) as ex:
        return dict(zip(elfs, ex.map(text, elfs.values())))


base = load(sys.argv[1])
print(f"{'build':24s} {'ERISC':5s} {'changed':>9s} {'d insns med/max':>16s} {'d frame med/max':>16s}")
hot_rows = []
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1)
    b = load(d)
    for proc, tag in (("active_erisc", "E0"), ("subordinate_active_erisc", "E1")):
        keys = [k for k in base if k.endswith("/" + proc) and k in b]
        ch = [k for k in keys if b[k][0] != base[k][0]]
        di = [len(b[k][0]) - len(base[k][0]) for k in ch] or [0]
        df = [b[k][1] - base[k][1] for k in ch] or [0]
        print(f"{name:24s} {tag:5s} {len(ch):4d}/{len(keys):<4d} {statistics.median(di):+7.0f}/{max(di, key=abs):+5d}"
              f"      {statistics.median(df):+6.0f}/{max(df, key=abs):+5d}")
    for hn, k in HOT.items():
        if k in b:
            sm = difflib.SequenceMatcher(None, base[k][0], b[k][0], autojunk=False)
            moved = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
            hot_rows.append(f"{name:24s} {hn:9s} {len(b[k][0]):5d} insns ({len(b[k][0]) - len(base[k][0]):+d}), "
                            f"frame {b[k][1]} ({b[k][1] - base[k][1]:+d}), {moved} instructions differ")
print("\nhot configs vs base:")
for hn, k in HOT.items():
    print(f"{'base':24s} {hn:9s} {len(base[k][0]):5d} insns, frame {base[k][1]}")
print("\n".join(hot_rows))
