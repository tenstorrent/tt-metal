# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-role binary sizes of the gated_delta_net kernels in the JIT cache, against the kernel-config budget.

The descriptor tags every kernel with the named compile-time arg gdn_role (named_ct_arg_map_generated.h), so each
cached build maps to a role. A binary's size is the sum of its ELF LOAD segments (what the kernel config buffer holds).
One line per build, then per role the largest binary per RISC over all builds; their sum bounds the role's kernel text
(the tightest real program, RTAs and CBs included, is in TT_METAL_LOG_JIT_TELEMETRY=1's summary). Measure one build
mode (plain, profiler, NoC trace) per TT_METAL_CACHE directory.

    python3 tests/ttnn/unit_tests/operations/gated_delta_net/kernel_text_sizes.py --minutes 30
"""

import argparse
import os
import re
import struct
import time
from collections import defaultdict
from pathlib import Path

ROLE_NAMES = {0: "local", 1: "prep", 2: "state", 3: "qk", 4: "prep_qk"}
RISCS = {"reader": ("ncrisc",), "writer": ("brisc",), "compute": ("trisc0", "trisc1", "trisc2")}
CONFIG_BUDGET = 70688


def load_bytes(elf):
    data = elf.read_bytes()
    phoff, phentsize, phnum = struct.unpack_from("<I", data, 28)[0], *struct.unpack_from("<HH", data, 42)
    total = 0
    for idx in range(phnum):
        p_type, _, _, _, p_filesz = struct.unpack_from("<IIIII", data, phoff + idx * phentsize)
        if p_type == 1:  # PT_LOAD
            total += p_filesz
    return total


def role_of(build_dir):
    arg_map = build_dir / "named_ct_arg_map_generated.h"
    if not arg_map.exists():
        return None
    match = re.search(r'\{"gdn_role",(\d+)\}', arg_map.read_text())
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=60, help="only builds newer than this")
    parser.add_argument("--cache", default=os.environ.get("TT_METAL_CACHE", str(Path.home() / ".cache/tt-metal-cache")))
    args = parser.parse_args()
    since = time.time() - args.minutes * 60
    sizes = defaultdict(dict)  # (key, build hash set) -> {risc: bytes}
    rows = []
    for kernel_dir in Path(args.cache).glob("*/kernels/gated_delta_net_*"):
        kind = next((k for k in RISCS if k in kernel_dir.name), None)
        for build in kernel_dir.iterdir():
            if not build.is_dir() or build.stat().st_mtime < since:
                continue
            role = role_of(build)
            if role is None:
                continue
            per_risc = {}
            for risc in RISCS[kind]:
                elf = build / risc / f"{risc}.elf.xip.elf"
                if not elf.exists():
                    elf = build / risc / f"{risc}.elf"
                if elf.exists():
                    per_risc[risc] = load_bytes(elf)
            key = kernel_dir.parent.parent.name
            rows.append((key, ROLE_NAMES.get(role, role), kernel_dir.name, build.name, per_risc))
    for key, role, name, build, per_risc in sorted(rows):
        cells = " ".join(f"{risc} {size:6d}" for risc, size in per_risc.items())
        print(f"{key[:8]} {role:8s} {name:36s} {build:22s} {cells}")
        # Worst case per (key, role, risc) across shapes / dtypes.
        for risc, size in per_risc.items():
            sizes[key, role][risc] = max(sizes[key, role].get(risc, 0), size)
    print()
    for (key, role), per_risc in sorted(sizes.items()):
        total = sum(per_risc.values())
        cells = " ".join(f"{risc} {per_risc.get(risc, 0):6d}" for kind in RISCS.values() for risc in kind)
        print(f"{key[:8]} {role:8s} max per RISC: {cells}  sum {total:6d} ({CONFIG_BUDGET - total:+d} B to budget)")


if __name__ == "__main__":
    main()
