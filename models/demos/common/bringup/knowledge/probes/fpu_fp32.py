# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Run: scripts/tt-probe.sh bringup_knowledge < models/demos/common/bringup/knowledge/probes/<this file>
# Known issue "The FPU reads fp32 operands at about TF32": fp32 matmul vs fp64, fp32 all_reduce vs exact and vs SFPU adds.
import torch

import ttnn

torch.manual_seed(0)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
rep = ttnn.ReplicateTensorToMesh(mesh)
dev0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()


def trunc(x, keep):  # keep `keep` mantissa bits (round toward zero), as a narrower register would
    i = x.contiguous().view(torch.int32) & ~((1 << (23 - keep)) - 1)
    return i.view(torch.float32)


def ulp(v, bits):
    return torch.exp2(torch.floor(torch.log2(v.abs().clamp_min(1e-30))) - bits)


# (a) matmul: fp32 inputs at HiFi4 + fp32 DEST
a, b = torch.randn(512, 2048), torch.randn(2048, 512)
ref = a.double() @ b.double()
ck = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)
out = dev0(
    ttnn.matmul(
        ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep),
        ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep),
        compute_kernel_config=ck,
    )
)
for name, m in (
    ("device", out),
    ("model tf32 inputs", trunc(a, 10).double() @ trunc(b, 10).double()),
    ("model fp32 inputs", (a @ b).double()),
):
    print(
        f"MATMUL {name:18s} rel {float((m - ref).norm() / ref.norm()):.3e}  norm ratio {float(m.norm() / ref.norm()):.7f}"
    )
# shrinking: mean of |out| / |ref| over the larger outputs
big = ref.abs() > ref.abs().median()
print(f"MATMUL mean |device|/|ref| on large outputs {float((out.abs()[big] / ref.abs()[big]).mean()):.7f}")

# (b) all_reduce over the 4 chips, fp32
x = torch.randn(4, 1, 64, 1024) * torch.tensor([1.0, 3.0, 0.3, 7.0]).view(4, 1, 1, 1)
tx = ttnn.from_torch(
    x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
)
exact = x.double().sum(0, keepdim=True)
red = dev0(ttnn.all_reduce(tx, cluster_axis=1))
g = ttnn.all_gather(tx, dim=0, cluster_axis=1)
acc = ttnn.slice(g, [0, 0, 0, 0], [1, 1, 64, 1024])
for i in range(1, 4):
    acc = ttnn.add(acc, ttnn.slice(g, [i, 0, 0, 0], [i + 1, 1, 64, 1024]))
sfpu = dev0(acc)
for name, r in (("all_reduce", red), ("all_gather + add", sfpu), ("torch fp32 sum", x.sum(0, keepdim=True).double())):
    err = (r - exact).abs()
    print(
        f"REDUCE {name:16s} max err {float(err.max()):.3e}  in fp32 ulps {float((err / ulp(exact, 23)).max()):.1f}"
        f"  in tf32 ulps {float((err / ulp(exact, 10)).max()):.3f}  mean fp32 ulps {float((err / ulp(exact, 23)).mean()):.2f}"
    )
ttnn.close_mesh_device(mesh)
