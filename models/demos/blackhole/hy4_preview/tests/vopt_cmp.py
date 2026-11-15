# SPDX-License-Identifier: Apache-2.0
"""verify-opt: compare two saved logits files (run_rowdep.py with SAVE_LOGITS=1, variant true): per (round, user, row) PCC and argmax agreement. usage: python vopt_cmp.py A.pt B.pt"""
import sys

import torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])  # [NR, R, S, V]
pc = []
ag = 0
n = 0
for c in range(a.shape[0]):
    for u in range(a.shape[1]):
        for j in range(a.shape[2]):
            x, y = a[c, u, j].double(), b[c, u, j].double()
            pc.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
            ag += int(x.argmax() == y.argmax())
            n += 1
pc = torch.tensor(pc)
print(f"RES pcc min {pc.min():.5f} mean {pc.mean():.5f} argmax_agree {ag}/{n}")
