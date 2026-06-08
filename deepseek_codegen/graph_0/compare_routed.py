# SPDX-License-Identifier: Apache-2.0
"""Compare full-mesh routed output (ConcatMeshToTensor dim=0 -> [32 devices, 32 tokens, 896]) between
the sparse path (golden) and the moe_compute path, with a per-device breakdown to localize the bug."""
import torch

d = __import__("pathlib").Path(__file__).resolve().parent / "moe_io"
s = torch.load(d / "routed_sparse.pt").float()
m = torch.load(d / "routed_moe.pt").float()
print("sparse", tuple(s.shape), "norm %.3f" % s.norm(), "| moe", tuple(m.shape), "norm %.3f" % m.norm())


def pcc(a, b):
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


print("OVERALL routed PCC(moe vs sparse) = %.6f  norm ratio = %.4f" % (pcc(m, s), float(m.norm() / (s.norm() + 1e-12))))
# per-device (dim0) breakdown: mesh (4,8) row-major -> dev = m0*8 + m1
print("per-device: dev (m0,m1)  PCC     normratio   s_norm   m_norm")
for dev in range(s.shape[0]):
    m0, m1 = dev // 8, dev % 8
    sd, md = s[dev], m[dev]
    nr = float(md.norm() / (sd.norm() + 1e-12))
    print(
        "  dev%2d (%d,%d)  %.4f   %.4f   %.3f   %.3f"
        % (dev, m0, m1, pcc(md, sd), nr, float(sd.norm()), float(md.norm()))
    )
