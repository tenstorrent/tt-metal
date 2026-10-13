#!/usr/bin/env python3
"""bits_cmp.py <dirA> <dirB>: every .pt in A equal word for word to the one in B (dicts compare the 'device' entry)."""
import os, sys, torch
a, b = sys.argv[1:3]
fs = sorted(f for f in os.listdir(a) if f.endswith('.pt'))
same = words = 0; diff = []
for f in fs:
    x = torch.load(os.path.join(a, f)); p = os.path.join(b, f)
    if not os.path.exists(p):
        diff.append(f + ' (missing)'); continue
    y = torch.load(p)
    if isinstance(x, dict):
        x, y = x['device'], y['device']
    words += x.numel()
    if torch.equal(x, y):
        same += 1
    else:
        diff.append(f'{f} ({int((x != y).sum())} of {x.numel()} words)')
print(f'BITS {a} vs {b}: {same} of {len(fs)} cases identical, {words} words compared')
for d in diff[:20]:
    print('  differ:', d)
