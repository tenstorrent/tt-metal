# SPDX-License-Identifier: Apache-2.0
"""Numerics of the column reduce variants vs the exact fp32 sum of the 8 per-column partials (scale alpha = <dev,exact>/<exact,exact>, rel L2): all_gather+sum (default compute config),
all_gather+sum with fp32 accumulation (HiFi4 fp32_dest_acc_en), all_gather + fp32 typecast + sum, and ttnn.all_reduce. Data: per-device bf16 partials, normal(0, 0.5) plus heavy-tailed rows."""
import json
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import close_mesh, open_mesh_full  # noqa: E402


def main():
    R, D = 32, 4096
    mesh = open_mesh_full()
    try:
        g = torch.Generator().manual_seed(3)
        h = torch.randn(4, 8, R, D, generator=g) * 0.5
        h[:, :, :, :64] *= 40.0  # a few massive-activation channels
        h = h.bfloat16()
        x = ttnn.from_torch(
            h,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, 1)),
        )
        x = ttnn.reshape(x, [1, 1, R, D])
        exact = h.float().sum(1)  # [4, R, D]
        comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

        def ag(x):
            return ttnn.all_gather(x, dim=0, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        variants = {
            "ag_sum_default": lambda x: ttnn.sum(ag(x), dim=0, keepdim=True),
            "ag_sum_fp32cfg": lambda x: ttnn.sum(ag(x), dim=0, keepdim=True, compute_kernel_config=ckc),
            "ag_fp32_sum": lambda x: ttnn.typecast(
                ttnn.sum(ttnn.typecast(ag(x), ttnn.float32), dim=0, keepdim=True, compute_kernel_config=ckc),
                ttnn.bfloat16,
            ),
            "all_reduce": lambda x: ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        }
        for name, fn in variants.items():
            o = ttnn.to_torch(fn(x), mesh_composer=comp).float().reshape(4, 8, R, D)[:, 0]
            a = ((o * exact).sum() / (exact * exact).sum()).item()
            rel = ((o - exact).norm() / exact.norm()).item()
            # the same on the non-massive channels only
            o2, e2 = o[..., 64:], exact[..., 64:]
            a2 = ((o2 * e2).sum() / (e2 * e2).sum()).item()
            s = json.dumps(dict(case="colreduce_numerics", variant=name, alpha=a, rel_l2=rel, alpha_normal_channels=a2))
            print("CN_RESULT " + s, flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
