#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checks that perf builds with counters off and on are the same program, so a code change moves both alike.

Pairs the variants of two --compile-producer outputs (RUNNER_TEMP dirs) by build.h and requires identical code in
every TRISC ELF and the BRISC ELF, and loaded data that differs only in the counter command words.
usage: single_code_path.py <counters off RUNNER_TEMP> <counters on RUNNER_TEMP>
"""
import hashlib
import os
import struct
import sys
from pathlib import Path

ALLOWED_SYMBOLS = {
    b"_ZN8llk_perf6detail7arm_cmdE",
    b"_ZN8llk_perf6detail13l1_client_cmdE",
}
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
    sections, allowed = {}, []
    for h in heads:
        n = name(h[0], heads[shstrndx])
        if h[2] & SHF_ALLOC and h[1] != SHT_NOBITS and n != b".profiler_meta":
            sections[n] = (h[3], data[h[4] : h[4] + h[5]])
        if h[1] == SHT_SYMTAB:
            for off in range(h[4], h[4] + h[5], 16):
                st_name, st_value, st_size = struct.unpack_from("<III", data, off)
                if name(st_name, heads[h[6]]) in ALLOWED_SYMBOLS:
                    allowed.append((st_value, max(st_size, 4)))
    return sections, allowed


def compare(fa, fb):
    if not (os.path.isfile(fa) and os.path.isfile(fb)):
        return "missing in one build"
    (sa, allowed), (sb, _) = read_elf(fa), read_elf(fb)
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
    }


def main(off, on):
    a, b = variants(off), variants(on)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("no paired variants: are these producer outputs of the same tests?")
    bad = []
    for k in common:
        for elf in ELFS:
            fa, fb = a[k] / "elf" / f"{elf}.elf", b[k] / "elf" / f"{elf}.elf"
            if fa.is_file() or fb.is_file():
                problem = compare(fa, fb)
                if problem:
                    bad.append(f"{k[0]} {a[k].name[:12]} {elf}: {problem}")
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
