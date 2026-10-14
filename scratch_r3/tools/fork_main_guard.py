#!/usr/bin/env python3
"""#58817 fork audit: the head's fork builds per FORK_VARIANT against main's build of the same configuration (main's
source ignores the define), per thread. Expected: variant 0 and 4 equal main everywhere.
usage: fork_main_guard.py <head cache> <main cache>"""
import os, sys
os.environ.setdefault("GUARD_KERNEL", "bmm_large_block_zm_fused_bias_activation_metal2")
os.environ.setdefault("GUARD_DEFINE", "FORK_VARIANT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alt_guard import collect  # noqa: E402

h, m = collect(sys.argv[1]), collect(sys.argv[2])
A, B = os.environ.get("GUARD_NAMES", "head,main").split(",")
keys = sorted(set(h) & set(m))
print(f"== {A} against {B}: {len(keys)} shared configurations ({len(h)} {A}, {len(m)} {B})")
same0 = 0
for k in keys:
    ref = next(iter(m[k].values()))
    row = []
    for v in sorted(h[k]):
        diff = [t for t in ref if h[k][v].get(t) != ref[t]]
        row.append(f"v{v} {'= ' + B if not diff else 'differs in ' + ','.join(diff)}")
        same0 += v == 0 and not diff
    print(f"   {k}: " + "; ".join(row))
print(f"== variant 0 equals {B} in {same0} of {len(keys)} configurations")
