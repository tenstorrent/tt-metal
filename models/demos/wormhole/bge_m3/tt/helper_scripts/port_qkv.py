"""Splice the p150 QKV head variants into main's op.py.

Main kept only bge_qkv_heads_headsplit. The p150 perf sweeps also drive the
stock baseline, the Track A batched-barrier variant, and the scatter stub, so
port all three plus the kernel path constants they name.
"""
import re

P = "/local/ttuser/gtobar/p150_reference/models/demos/wormhole/bge_m3/tt/custom_ops/fused_qkv_heads/op.py"
M = "/local/ttuser/gtobar/tt-metal/models/demos/wormhole/bge_m3/tt/custom_ops/fused_qkv_heads/op.py"

p_src = open(P).read()
m_src = open(M).read()


def grab(src, start_marker, end_marker):
    """Return the text from start_marker up to (not including) end_marker."""
    i = src.index(start_marker)
    j = src.index(end_marker, i + len(start_marker))
    return src[i:j].rstrip() + "\n"


# 1. Kernel path constants for the Track A batched kernels.
tracka_paths = grab(p_src, "# Track A optimized kernels", "# Head-split kernels")

# 2. The three function bodies.
fn_stock = grab(p_src, "def bge_qkv_heads_stock(", "def _split_work_to_cores(")
fn_tracka = grab(p_src, "def bge_qkv_heads_tracka(", "def bge_qkv_heads_headsplit(")
fn_scatter = p_src[p_src.index("def bge_qkv_heads_scatter(") :].rstrip() + "\n"

# Insert the constants just before main's head-split path block.
anchor = "# Head-split kernels"
assert m_src.count(anchor) == 1, "expected one head-split kernel comment in main"
m_src = m_src.replace(anchor, tracka_paths + "\n" + anchor, 1)

# Append the variants after main's head-split implementation.
m_src = m_src.rstrip() + "\n\n\n" + fn_stock + "\n\n" + fn_tracka + "\n\n" + fn_scatter

open(M, "w").write(m_src)

names = re.findall(r"^def (bge_\w+)", m_src, re.M)
print("  functions now in main op.py:")
for n in names:
    print("    " + n)
print("  TRACKA paths added:", "TRACKA_READER_KERNEL_REL_PATH" in m_src)
print("  SCATTER path added:", "SCATTER_WRITER_KERNEL_REL_PATH" in m_src)
