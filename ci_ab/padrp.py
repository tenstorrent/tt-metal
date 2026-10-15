#!/usr/bin/env python3
"""Scratch, not for merge. Layout-only pad for topk_route_prep_untilize_clamp.cpp: before the closing brace of kernel_main
(the last line of the file that is a lone "}"), three lines that put nu, nm and np NOPs into the unpack, math and pack
TRISC images (each runs once per kernel call; a count of 0 gives an empty statement, so every variant has the same line
numbers). usage: padrp.py <in> <out> <nu> <nm> <np>"""
import sys

src, dst, nu, nm, np_ = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6])
lines = open(src).read().split("\n")
end = max(i for i, l in enumerate(lines) if l.rstrip() == "}")


def pad(mac, n):
    if n <= 0:
        return f"    {mac}((void)0);"
    return f'    {mac}(asm volatile(".rept {n}\\n\\tnop\\n\\t.endr"));'


lines[end:end] = [pad("UNPACK", nu), pad("MATH", nm), pad("PACK", np_)]
open(dst, "w").write("\n".join(lines))
