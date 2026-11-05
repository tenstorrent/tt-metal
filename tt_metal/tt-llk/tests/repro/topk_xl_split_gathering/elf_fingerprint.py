#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fingerprint the kernel ELFs the topk_xl split repro builds.

The failure depends on where the PACK code sits in memory, so a run only counts if its ELFs are byte-identical
(executable sections) to the ones validated on silicon. Usage:

    elf_fingerprint.py <build-root>

<build-root> is the harness build tree: $RUNNER_TEMP/tt-llk-build, or /tmp/tt-llk-build when RUNNER_TEMP is
unset. The gathering-off build lives under <build-root>/no-gathering. Prints one line per variant of
topk_xl_split_test.cpp: the tree, the variant's ASCENDING / TOPK_XL_TAIL_ELEMENTS values and a sha256 of
each thread ELF's executable sections.
"""

import hashlib
import re
import struct
import sys
from pathlib import Path


def exec_sections_digest(path: Path) -> str:
    data = path.read_bytes()
    (e_shoff,) = struct.unpack_from("<I", data, 0x20)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x2E)
    h = hashlib.sha256()
    for i in range(e_shnum):
        _, sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from("<IIIIII", data, e_shoff + i * e_shentsize)
        if sh_type == 1 and sh_flags & 0x4:  # PROGBITS + EXECINSTR
            h.update(struct.pack("<II", sh_addr, sh_size))
            h.update(data[sh_offset : sh_offset + sh_size])
    return h.hexdigest()[:16]


def main(root: str) -> int:
    found = 0
    for tree in ("", "no-gathering"):
        base = Path(root) / tree / "sources" / "topk_xl_split_test.cpp"
        for variant in sorted(base.glob("*/")):
            build_h = (variant / "build.h").read_text() if (variant / "build.h").is_file() else ""
            params = {k: v for k, v in re.findall(r"(ASCENDING|TOPK_XL_TAIL_ELEMENTS)\s*=\s*([A-Za-z0-9_]+)", build_h)}
            digests = " ".join(
                f"{t}={exec_sections_digest(variant / 'elf' / f'{t}.elf')}"
                for t in ("unpack", "math", "pack")
                if (variant / "elf" / f"{t}.elf").is_file()
            )
            print(f"{tree or 'gathering'} {params} {digests}")
            found += 1
    if not found:
        print(f"no topk_xl_split_test.cpp variants under {root}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
