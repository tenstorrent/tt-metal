#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router bring-up tools (temporary: drop before merge; see README.md).

codegen_diff.py [--kernel NAME] [--hot HASH/IMAGE ...] BASE_DIR NAME=DIR [NAME=DIR ...]: how much each build's router
code differs from the base build, config by config.

Builds: offline_build.py outputs of the same cmds.json (so configs pair by hash), or kernel caches with the same
configs. Per build and ERISC image (Wormhole: erisc; Blackhole: active_erisc, subordinate_active_erisc): how many
configs' .text differ from the base (the same instruction stream counts as equal; disassembler comments ignored), and
the median and largest change in .text instructions and in the largest stack frame. For each --hot config (pick the
configs of the rows you care about), the instruction count, the largest frame and how many instructions differ in a
line diff against the base. Static numbers only: they say whether and where code moved, not what it costs.
"""
import argparse
import concurrent.futures
import difflib
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_tools_common import ERISC_IMAGES, ERISC_TAG, DEFAULT_KERNEL, router_images, toolchain_prefix  # noqa: E402

OD = toolchain_prefix() + "objdump"


def text(elf):
    out = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    ins = [re.sub(r"\s*#.*$", "", l.split("\t", 2)[-1]) for l in out.splitlines() if re.match(r"\s+[0-9a-f]+:\t", l)]
    ins = [re.sub(r" <[^>]*>", "", x) for x in ins]
    frames = sorted((int(m) for m in re.findall(r"addi\s+sp,sp,-(\d+)", out)), reverse=True) + [0]
    return ins, frames[0]


def load(d, kernel):
    elfs = {f"{h}/{p}": e for h, p, e in router_images(d, kernel)}
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        return dict(zip(elfs, ex.map(text, elfs.values())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("builds", nargs="+", help="NAME=DIR")
    ap.add_argument("--kernel", default=DEFAULT_KERNEL)
    ap.add_argument("--hot", action="append", default=[], help="HASH/IMAGE, e.g. 1234/erisc")
    a = ap.parse_args()
    base = load(a.base, a.kernel)
    if not base:
        sys.exit(f"no {a.kernel} images under {a.base}")
    print(f"{'build':24s} {'image':6s} {'changed':>9s} {'d insns med/max':>16s} {'d frame med/max':>16s}")
    hot_rows = []
    for arg in a.builds:
        name, d = arg.split("=", 1)
        b = load(d, a.kernel)
        for proc in ERISC_IMAGES:
            keys = [k for k in base if k.endswith("/" + proc) and k in b]
            if not keys:
                continue
            ch = [k for k in keys if b[k][0] != base[k][0]]
            di = [len(b[k][0]) - len(base[k][0]) for k in ch] or [0]
            df = [b[k][1] - base[k][1] for k in ch] or [0]
            print(
                f"{name:24s} {ERISC_TAG[proc]:6s} {len(ch):4d}/{len(keys):<4d} "
                f"{statistics.median(di):+7.0f}/{max(di, key=abs):+5d}      {statistics.median(df):+6.0f}/{max(df, key=abs):+5d}"
            )
        for k in a.hot:
            if k in b and k in base:
                sm = difflib.SequenceMatcher(None, base[k][0], b[k][0], autojunk=False)
                moved = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
                hot_rows.append(
                    f"{name:24s} {k}: {len(b[k][0])} insns ({len(b[k][0]) - len(base[k][0]):+d}), "
                    f"frame {b[k][1]} ({b[k][1] - base[k][1]:+d}), {moved} instructions differ"
                )
    if a.hot:
        print("\nhot configs vs base:")
        for k in a.hot:
            if k in base:
                print(f"{'base':24s} {k}: {len(base[k][0])} insns, frame {base[k][1]}")
            else:
                print(f"{'base':24s} {k}: not in the base build")
        print("\n".join(hot_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
