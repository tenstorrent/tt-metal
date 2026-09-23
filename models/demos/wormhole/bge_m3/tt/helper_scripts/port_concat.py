"""Splice the p150 concat-heads variants into main's op.py."""

import re

P = "/local/ttuser/gtobar/p150_reference/models/demos/wormhole/bge_m3/tt/custom_ops/fused_concat_heads/op.py"
M = "/local/ttuser/gtobar/tt-metal/models/demos/wormhole/bge_m3/tt/custom_ops/fused_concat_heads/op.py"

p_src = open(P).read()
m_src = open(M).read()


def grab(src, start_marker, end_marker):
    i = src.index(start_marker)
    j = src.index(end_marker, i + len(start_marker))
    return src[i:j].rstrip() + "\n"


tracka_paths = grab(p_src, "TRACKA_READER_KERNEL_REL_PATH = (", "HEADSPLIT_READER_KERNEL_REL_PATH = (")
fn_stock = grab(p_src, "def bge_concat_heads_stock(", "def _split_work_to_cores(")
fn_tracka = grab(p_src, "def bge_concat_heads_tracka(", "def bge_concat_heads_headsplit(")

anchor = "HEADSPLIT_READER_KERNEL_REL_PATH = ("
assert m_src.count(anchor) == 1
m_src = m_src.replace(anchor, tracka_paths + "\n" + anchor, 1)
m_src = m_src.rstrip() + "\n\n\n" + fn_stock + "\n\n" + fn_tracka

open(M, "w").write(m_src)
print("  functions now in main concat op.py:")
for n in re.findall(r"^def (bge_\w+)", m_src, re.M):
    print("    " + n)
print("  TRACKA paths added:", "TRACKA_READER_KERNEL_REL_PATH" in m_src)
