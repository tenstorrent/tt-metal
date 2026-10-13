#!/usr/bin/env python3
"""Scratch, not for merge. Per case of an ab_dump results.pt: sha256 of every output's dtype, shape and bytes, printed as
"TBITS <arm> <dump> <case> <sha16>"."""
import hashlib
import sys

import torch

d, arm, dump = sys.argv[1:4]
res = torch.load(d + "/results.pt", weights_only=True)
for k in sorted(res):
    h = hashlib.sha256()
    for x in res[k]:
        x = x.contiguous()
        h.update(f"{x.dtype}|{tuple(x.shape)}|".encode())
        h.update(x.view(torch.uint8).numpy().tobytes() if x.numel() else b"")
    print(f"TBITS {arm} {dump} {k} {h.hexdigest()[:16]}")
