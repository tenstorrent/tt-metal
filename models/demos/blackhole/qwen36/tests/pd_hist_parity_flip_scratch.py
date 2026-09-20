# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-side parity flip of packed conv-history tiles: for the flipped slots, even and odd tile rows swap
(a 32x32 row permutation applied by one HiFi4 matmul over the whole [B, Nv*4, 32, 32] tensor); other slots get the
identity. Checks bitwise against torch and times it.  TT_VISIBLE_DEVICES=0,1,6,7 python <this file>"""
import time

import torch
from loguru import logger

import ttnn

B, NV, K = 32, 12, 4


def parity_perm(flipped, B):
    P = torch.eye(32, dtype=torch.bfloat16).repeat(B, 1, 1, 1)  # [B,1,32,32]
    swap = torch.zeros(32, 32, dtype=torch.bfloat16)
    for c in range(16):
        swap[2 * c, 2 * c + 1] = 1.0
        swap[2 * c + 1, 2 * c] = 1.0
    for b in flipped:
        P[b, 0] = swap
    return P


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
    mesh.enable_program_cache()
    torch.manual_seed(0)
    n_dev = 4
    host = torch.randn(n_dev * B, NV, K, 32, 32).to(torch.bfloat16)
    hist = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    exact = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    for trial, flipped in enumerate([[1, 0, 6], list(range(0, B, 3)), list(range(B)), [5]]):
        P = parity_perm(flipped, B).expand(B, NV * K, 32, 32).contiguous()
        Pt = ttnn.from_torch(
            P,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for rep in range(2):
            t0 = time.perf_counter()
            h4 = ttnn.reshape(hist, (B, NV * K, 32, 32))
            out = ttnn.matmul(Pt, h4, compute_kernel_config=exact)
            out5 = ttnn.reshape(out, (B, NV, K, 32, 32))
            ttnn.copy(out5, hist)
            ttnn.deallocate(out)  # h4/out5 are views (reshape shares the buffer): never deallocate them
            ttnn.synchronize_device(mesh)
            dt = time.perf_counter() - t0
        # reference: two applications == identity for flipped slots; apply twice on host too
        ref = host.view(n_dev, B, NV, K, 32, 32).clone()
        # (rep loop applied the permutation twice -> identity; do the same to reference: no-op) -> compare equality
        got = ttnn.to_torch(hist, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).view(n_dev, B, NV, K, 32, 32)
        ok2 = torch.equal(got, ref)
        # single application check
        h4 = ttnn.reshape(hist, (B, NV * K, 32, 32))
        out = ttnn.matmul(Pt, h4, compute_kernel_config=exact)
        got1 = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).view(n_dev, B, NV, K, 32, 32)
        ref1 = ref.clone()
        for b in flipped:
            r = ref1[:, b].clone()
            ref1[:, b, :, :, 0::2, :] = r[:, :, :, 1::2, :]
            ref1[:, b, :, :, 1::2, :] = r[:, :, :, 0::2, :]
        ok1 = torch.equal(got1, ref1)
        ttnn.deallocate(out)
        ttnn.deallocate(Pt)
        logger.info(
            f"trial {trial} flipped={len(flipped)} slots: single flip {'OK' if ok1 else 'BAD'}, double flip = identity {'OK' if ok2 else 'BAD'}, {1e3 * dt:.1f} ms per flip (reshape+matmul+reshape+copy, warm)"
        )
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
