#!/usr/bin/env python3
"""cg_gate.py OLD_DIR NEW_DIR: the default-identical gate of a host rebuild (SLOT_ALLOCATOR.md section 12.10): two
trid_vet.sh outputs (V=OLD_DIR on the old host's tree, V=NEW_DIR on the new one; same configs, same switches, the
new switch unset) must build the same router images.

Each run's kernel cache holds its fabric_erisc_router_v2 builds (the build hash differs between trees, so builds are
not paired by it). A build is its processor, its named compile-time args (named_ct_arg_map_generated.h) and its image:
every PT_LOAD segment of its .elf (address, memory size, file bytes). Per run, the two hosts' builds must be the same
multiset. (The runs' logs are not used: a two-rank run can drop compile-command lines.) Exit 1 otherwise.
"""
import collections
import glob
import hashlib
import os
import struct
import sys


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
    out = collections.Counter()
    for elf in glob.glob(os.path.join(run_dir, "cache_*", "*", "kernels", "fabric_erisc_router_v2", "*", "*", "*.elf")):
        if elf.endswith(".xip.elf"):
            continue
        proc = os.path.basename(os.path.dirname(elf))
        named = open(os.path.join(os.path.dirname(os.path.dirname(elf)), "named_ct_arg_map_generated.h")).read()
        out[(proc, hashlib.sha1(named.encode()).hexdigest()[:16], image(elf))] += 1
    return out


def main():
    old_dir, new_dir = sys.argv[1:3]
    bad = 0
    runs = sorted(os.path.relpath(d, old_dir) for d in glob.glob(os.path.join(old_dir, "*", "*", "*")) if os.path.isdir(d))
    if not runs:
        print(f"no runs under {old_dir}")
        sys.exit(1)
    for rel in runs:
        a, b = builds(os.path.join(old_dir, rel)), builds(os.path.join(new_dir, rel))
        same = sum((a & b).values())
        ok = a and a == b
        bad += not ok
        print(f"{rel}: old {sum(a.values())} builds, new {sum(b.values())}, identical (named args and image) {same}: "
              f"{'ok' if ok else 'DIFFERS'}")
    print("CG_GATE_OK" if bad == 0 else "CG_GATE_DIFFERS")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
