#!/usr/bin/env python3
"""Hash of the matmul_expert kernel ELFs per RISC in one JIT cache."""
import glob, hashlib, os, sys
from collections import defaultdict
arm, cache = sys.argv[1:3]
h = defaultdict(list)
for p in glob.glob(os.path.join(cache, "**", "*.elf"), recursive=True):
    if "matmul_expert_kernel" not in p:
        continue
    risc = os.path.basename(os.path.dirname(p))
    h[risc].append(hashlib.md5(open(p, "rb").read()).hexdigest())
print(f"ELF {arm} " + " ".join(f"{r}:{hashlib.md5(''.join(sorted(v)).encode()).hexdigest()[:10]}({len(v)})" for r, v in sorted(h.items())))
