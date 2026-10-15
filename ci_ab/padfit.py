#!/usr/bin/env python3
"""Scratch, not for merge. Packed sizes of the topk_route_prep_untilize_clamp TRISC images in two probe caches (one build
each, the qwen case) and the NOP pads that make them equal: the smaller side of each TRISC gets the difference.
usage: padfit.py <cache_m> <cache_s> <m pads u,m,p> <s pads u,m,p>; prints one "RAB PADFIT" line, last field the next pads."""
import glob
import os
import struct
import sys


def packed(elf):
    b = open(elf, "rb").read()
    phoff, = struct.unpack_from("<I", b, 0x1C)
    phentsize, phnum = struct.unpack_from("<HH", b, 0x2A)
    n = 0
    for i in range(phnum):
        p_type, _, _, _, p_filesz = struct.unpack_from("<IIIII", b, phoff + i * phentsize)
        if p_type == 1:
            n += (p_filesz + 3) // 4 * 4
    return n


def sizes(cache):
    builds = sorted(glob.glob(os.path.join(cache, "**/kernels/topk_route_prep_untilize_clamp/*"), recursive=True))
    if len(builds) != 1:
        sys.exit(f"RAB PADFIT expected one route prep build in {cache}, found {len(builds)}")
    return [packed(glob.glob(os.path.join(builds[0], f"trisc{t}", "*.elf"))[0]) for t in range(3)]


cm, cs = sys.argv[1], sys.argv[2]
pm = [int(x) for x in sys.argv[3].split(",")]
ps = [int(x) for x in sys.argv[4].split(",")]
sm, ss = sizes(cm), sizes(cs)
for t in range(3):
    d = ss[t] - sm[t]
    if d > 0:
        pm[t] += d // 4
    elif d < 0:
        ps[t] += -d // 4
match = int(sm == ss)
print(f"RAB PADFIT sizes_m {','.join(map(str, sm))} sizes_s {','.join(map(str, ss))} match {match} "
      f"next {','.join(map(str, pm))} {','.join(map(str, ps))}")
