# SPDX-License-Identifier: Apache-2.0
"""Check whether the routed output I dumped (routed_moe) is actually what got added into add_27.
If (add27_moe - add27_sparse) == (routed_moe - routed_sparse), the dumped routed IS what's added."""
import torch
from pathlib import Path

d = Path(__file__).resolve().parent / "moe_io"
sadd = torch.load(d / "ttnn_add_27_sparse.pt").float()
madd = torch.load(d / "ttnn_add_27_moe.pt").float()
rs = torch.load(d / "routed_sparse.pt").float()
rm = torch.load(d / "routed_moe.pt").float()


def pcc(a, b):
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


print("shapes  add27:", tuple(sadd.shape), tuple(madd.shape), " routed:", tuple(rs.shape), tuple(rm.shape))
print("add27   sparse_norm=%.3f moe_norm=%.3f  PCC(moe,sparse)=%.5f" % (sadd.norm(), madd.norm(), pcc(madd, sadd)))
da = (madd - sadd).flatten()
dr = (rm - rs).flatten()
print(
    "delta_add norm=%.4f   delta_routed norm=%.4f   ratio=%.4f"
    % (da.norm(), dr.norm(), float(da.norm() / (dr.norm() + 1e-12)))
)
print("PCC(delta_add, delta_routed) = %.5f" % pcc(da, dr))
print(" -> if ~1.0 & ratio~1.0: dumped routed IS what's added; else the added tensor differs from the dump")
