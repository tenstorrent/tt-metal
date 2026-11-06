#!/usr/bin/env python3
"""decode_gate.py A_DIR B_DIR [CONFIG ...]: the route-decode A/B's offline gate (SLOT_ALLOCATOR.md §12.13).
A_DIR / B_DIR: rc_capture.sh outputs (RC/port/<config>) of the port tree under two settings. Per config, every
fabric_erisc_router build of each side is paired by its processor and its named compile-time args (the decode switch is
a compile define, so a build's named args are the same on both sides); the pair's images (every PT_LOAD segment) are
compared. Prints per config and per (ERISC, direction): builds paired, identical, differing, unpaired."""
import collections
import glob
import hashlib
import os
import re
import struct
import sys

DIRS = {0: "E", 1: "W", 2: "N", 3: "S", 4: "Z"}


def image(elf):
    b = open(elf, "rb").read()
    phoff, = struct.unpack_from("<I", b, 0x1C)
    phentsize, phnum = struct.unpack_from("<HH", b, 0x2A)
    h = hashlib.sha1()
    for i in range(phnum):
        p_type, off, vaddr, _, filesz, memsz, _, _ = struct.unpack_from("<8I", b, phoff + i * phentsize)
        if p_type == 1:
            h.update(struct.pack("<II", vaddr, memsz))
            h.update(b[off : off + filesz])
    return h.hexdigest()


def builds(run_dir):
    out = collections.defaultdict(list)  # (proc, named text) -> [(image, elf)]
    for elf in glob.glob(os.path.join(run_dir, "**", "kernels", "fabric_erisc_router", "*", "*", "*.elf"), recursive=True):
        if elf.endswith(".xip.elf"):
            continue
        proc = os.path.basename(os.path.dirname(elf))
        named = open(os.path.join(os.path.dirname(os.path.dirname(elf)), "named_ct_arg_map_generated.h")).read()
        out[(proc, named)].append((image(elf), elf))
    return out


def direction(named):
    m = re.search(r'\{"MY_DIRECTION",(\d+)\}', named)
    return DIRS.get(int(m.group(1)), "?") if m else "?"


def main():
    a_dir, b_dir = sys.argv[1:3]
    configs = sys.argv[3:] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(a_dir, "*")) if os.path.isdir(p)
                                     and os.path.isdir(os.path.join(b_dir, os.path.basename(p))))
    for cfg in configs:
        a, b = builds(os.path.join(a_dir, cfg)), builds(os.path.join(b_dir, cfg))
        groups = collections.defaultdict(lambda: collections.Counter())
        for key in set(a) | set(b):
            g = groups[("E0" if key[0] == "active_erisc" else "E1", direction(key[1]))]
            if key not in a or key not in b:
                g["unpaired"] += 1
                continue
            ia, ib = sorted(x for x, _ in a[key]), sorted(x for x, _ in b[key])
            g["paired"] += 1
            g["identical" if ia == ib else "differ"] += 1
        cells = ", ".join(f"{e}/{d}: {g['paired']} paired, {g['differ']} differ"
                          + (f", {g['unpaired']} unpaired" if g["unpaired"] else "")
                          for (e, d), g in sorted(groups.items()))
        print(f"{cfg}: {cells}")


if __name__ == "__main__":
    main()
