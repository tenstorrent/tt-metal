#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checks that perf builds with counters off and on are the same program, so a code change moves both alike.

Pairs the variants of two --compile-producer outputs (RUNNER_TEMP dirs) by build.h and requires identical code in
every TRISC ELF (with its Wormhole layout copies) and the BRISC ELF, and loaded data that differs only in the counter
command words. The counters off build must hold 0 in arm_cmd: every piece of counter work branches on it (counters.h).
usage: single_code_path.py <counters off RUNNER_TEMP> <counters on RUNNER_TEMP>
"""
import hashlib
import os
import struct
import sys
from pathlib import Path

ARM_CMD = b"_ZN8llk_perf6detail7arm_cmdE"
ALLOWED_SYMBOLS = {
    ARM_CMD,
    b"_ZN8llk_perf6detail13l1_client_cmdE",
}
ARM_CMD_VALUES = (0, 1)  # counters off: no command, so no counter work runs; on: START
ELFS = ("unpack", "math", "pack", "sfpu")
SHF_ALLOC, SHT_NOBITS, SHT_SYMTAB = 2, 8, 2


def read_elf(path):
    """Loaded sections {name: (addr, bytes)} and the address ranges of the allowed symbols (ELF32 little endian)."""
    data = Path(path).read_bytes()
    (shoff,) = struct.unpack_from("<I", data, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x2E)
    heads = [
        struct.unpack_from("<IIIIIIIIII", data, shoff + i * shentsize)
        for i in range(shnum)
    ]
    name = lambda off, tab: data[tab[4] + off : data.index(b"\0", tab[4] + off)]
    sections, allowed, arm_cmd = {}, [], None
    for h in heads:
        n = name(h[0], heads[shstrndx])
        if h[2] & SHF_ALLOC and h[1] != SHT_NOBITS and n != b".profiler_meta":
            sections[n] = (h[3], data[h[4] : h[4] + h[5]])
        if h[1] == SHT_SYMTAB:
            for off in range(h[4], h[4] + h[5], 16):
                st_name, st_value, st_size = struct.unpack_from("<III", data, off)
                if name(st_name, heads[h[6]]) in ALLOWED_SYMBOLS:
                    allowed.append((st_value, max(st_size, 4)))
                if name(st_name, heads[h[6]]) == ARM_CMD:
                    arm_cmd = st_value
    for addr, body in sections.values():
        if arm_cmd is not None and addr <= arm_cmd < addr + len(body):
            arm_cmd = struct.unpack_from("<I", body, arm_cmd - addr)[0]
            break
    else:
        arm_cmd = None
    return sections, allowed, arm_cmd


def compare(fa, fb):
    if not (os.path.isfile(fa) and os.path.isfile(fb)):
        return "missing in one build"
    (sa, allowed, arm_a), (sb, _, arm_b) = read_elf(fa), read_elf(fb)
    if (arm_a, arm_b) not in ((None, None), ARM_CMD_VALUES):
        return f"arm_cmd is {arm_a} off / {arm_b} on, not {ARM_CMD_VALUES[0]} / {ARM_CMD_VALUES[1]}"
    for n in sorted(set(sa) | set(sb)):
        (xa, da), (xb, db) = sa.get(n, (None, b"")), sb.get(n, (None, b""))
        if xa != xb or len(da) != len(db):
            return f"section {n.decode()} placement differs"
        if da == db:
            continue
        if n in (b".init", b".text"):
            return f"code in {n.decode()} differs"
        for i in range(0, len(da), 4):
            if da[i : i + 4] != db[i : i + 4] and not any(
                a <= xa + i < a + s for a, s in allowed
            ):
                return f"{n.decode()} word at {xa + i:#x} differs"
    return None


def variants(root):
    sources = Path(root) / "tt-llk-build" / "sources"
    return {
        (
            str(b.parent.parent.relative_to(sources)),
            hashlib.sha256(b.read_bytes()).hexdigest(),
        ): b.parent
        for b in sources.rglob("build.h")
        # the measured variants: a run without --speed-of-light keeps its INIT measurement builds apart, each built by
        # the first test of its sharing key (test_config.py _init_build_dir), and none of them is compared
        if b.parent.parent.name != "init_sol"
    }


def main(off, on):
    a, b = variants(off), variants(on)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("no paired variants: are these producer outputs of the same tests?")
    bad = []
    for k in common:
        # the Wormhole layout copies (perf/layout.py) must pick the same pads in both builds
        layouts = {d.name for d in a[k].glob("layout_*")} | {
            d.name for d in b[k].glob("layout_*")
        }
        for sub in ["elf"] + [f"{lay}/elf" for lay in sorted(layouts)]:
            for elf in ELFS:
                fa, fb = a[k] / sub / f"{elf}.elf", b[k] / sub / f"{elf}.elf"
                if fa.is_file() or fb.is_file():
                    problem = compare(fa, fb)
                    if problem:
                        bad.append(f"{k[0]} {a[k].name[:12]} {sub} {elf}: {problem}")
    brisc = [
        Path(d) / "tt-llk-build" / "shared" / "elf" / "brisc.elf" for d in (off, on)
    ]
    if any(f.is_file() for f in brisc):
        problem = compare(*brisc)
        if problem:
            bad.append(f"brisc: {problem}")
    for line in bad:
        print(line)
    print(
        f"{len(common)} variants paired, {len(bad)} ELFs differ beyond the counter command words"
    )
    if len(common) < max(len(a), len(b)):
        print(
            f"unpaired variants: {len(a) - len(common)} counters off, {len(b) - len(common)} counters on"
        )
    return 1 if bad or len(common) < max(len(a), len(b)) else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
