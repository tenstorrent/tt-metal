"""Static analysis for the SFPU optimization tutorial.

- fma_count: analytical fused-multiply-add count per element for the predicated
  piecewise Horner cascade (every segment is evaluated under predication, so the
  cost is the sum over segments). Parity halves each segment's Horner length.
- count_sfpu_insns: best-effort real SFPU instruction count from the JIT'd TRISC
  object via riscv-tt-elf-objdump (counts mnemonics starting 'sfp').
"""
import subprocess
import os
import math
import re

SFPI_OBJDUMP = "/localdev/nkapre/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"


def fma_count(seg_degrees, parity=False):
    """Sum of Horner FMAs across all segments (predicated cascade evaluates all)."""

    def per_seg(d):
        return math.ceil(d / 2) if parity else d

    return int(sum(per_seg(int(d)) for d in seg_degrees))


def count_sfpu_insns(obj_path):
    if not obj_path or not os.path.exists(SFPI_OBJDUMP) or not os.path.exists(obj_path):
        return None
    try:
        out = subprocess.run([SFPI_OBJDUMP, "-d", obj_path], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    return len(re.findall(r"\b(sfp[a-z0-9_]+)\b", out))
