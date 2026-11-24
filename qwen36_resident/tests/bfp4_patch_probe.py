# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S0-8: does patch_bfp4_row (kernels/bfp4_rtn.hpp) re-encode rows of a DEVICE-packed bfp4_b tile to nearest? A bf16
tensor is packed to bfp4 on the device (ttnn.typecast: the packer truncates), the kernel re-encodes rows r0 .. r1 from
the bf16 source, and the result is read back and compared with the host packer's round-to-nearest and with the
truncated original. usage: python -m qwen36_resident.tests.bfp4_patch_probe"""
import torch

import ttnn
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

KDIR = __file__.rsplit("/tests/", 1)[0] + "/kernels/"


def run_patch(mesh, xq, x, n, r0, r1):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [xq.buffer_address(), x.buffer_address(), n, r0, r1]
    cbs = [
        ttnn.CBDescriptor(
            total_size=576,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat4_b, page_size=576)],
        ),
        ttnn.CBDescriptor(
            total_size=2048,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=2048)],
        ),
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "bfp4_patch_test.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
    mesh_pd = ttnn.MeshProgramDescriptor()
    n_dev = mesh.get_num_devices()
    mesh_pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, n_dev - 1))] = program
    ttnn.generic_op([x, xq], mesh_pd)


def main():
    mesh = open_mesh()
    try:
        torch.manual_seed(0)
        x = (torch.randn(32, 256) * torch.linspace(0.1, 4, 256)).bfloat16()
        rep = ttnn.ReplicateTensorToMesh(mesh)
        xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        xq = ttnn.typecast(xd, ttnn.bfloat4_b)  # (the device packer: truncation)
        comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        trunc = ttnn.to_torch(xq, mesh_composer=comp)[:32].float()
        rtn_host = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
        r0, r1 = 0, 32
        run_patch(mesh, xq, xd, 8, r0, r1)
        patched = ttnn.to_torch(xq, mesh_composer=comp)[:32].float()
        xf = x.float()
        err = lambda a: ((a - xf).norm() / xf.norm()).item()  # noqa: E731
        print(
            f"rel err vs bf16: device truncated {err(trunc):.4f}, host RTN {err(rtn_host):.4f}, patched {err(patched):.4f}"
        )
        same = (patched == rtn_host).float().mean().item()
        print(
            f"patched == host RTN: {100 * same:.2f}% of elements; patched == truncated: "
            f"{100 * (patched == trunc).float().mean().item():.2f}%"
        )
        bad = (patched != rtn_host).nonzero()[:8].tolist()
        for r, c in bad:
            print(
                f"  [{r},{c}] x {xf[r, c]:+.5f} host RTN {rtn_host[r, c]:+.5f} patched {patched[r, c]:+.5f} trunc {trunc[r, c]:+.5f}"
            )
        print("PASS" if same > 0.99 else "FAIL")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
