# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""O-1c: the byte layout of a bfp8_b batch x 32 (8 x 32) tile — device packer (ttnn.typecast) vs host (from_torch) — and
whether 4 of them gathered as [4 x 16 B exponents | 4 x 256 B mantissas] decode as one bfp8 32 x 32 tile whose face F
row R holds tile F's face R // 8 row R % 8. usage: python -m qwen36_resident.tests.bfp8_tiny_probe"""
import torch

import ttnn
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

KDIR = __file__.rsplit("/tests/", 1)[0] + "/kernels/"


def raw(mesh, t, page, n):
    """the first n pages of t's buffer (chip 0) as bytes"""
    out = ttnn.from_torch(
        torch.zeros(1, n * page // 4, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    # (a row-major uint32 tensor of one row: its single page is n * page bytes; the kernel writes it as n pages)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address(), page, out.buffer_address(), n]
    cbs = [
        ttnn.CBDescriptor(
            total_size=page,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=page)],
        )
    ]
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + "raw_dump_test.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    pd = ttnn.MeshProgramDescriptor()
    pd[
        ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, mesh.get_num_devices() - 1))
    ] = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs)
    ttnn.generic_op([t, out], pd)
    b = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0].to(torch.int32).numpy().tobytes()
    return b


def pack_probe(mesh, x, page):
    """x [8, 32 n] bf16 through the streamers' pack path (fp32 DST -> a bfp8 batch x 32 CB with pages of `page` B);
    returns CB 1's raw bytes"""
    n = x.shape[1] // 32
    rep = ttnn.ReplicateTensorToMesh(mesh)
    # bf16 batch x 32 tiles, host-tilized into one row-major uint32 page (face f row r col c at f * 256 + r * 32 + 2 c)
    t = torch.zeros(n, 2, 8, 16, dtype=torch.bfloat16)
    for i in range(n):
        for f in range(2):
            t[i, f] = x[:, 32 * i + 16 * f : 32 * i + 16 * f + 16]
    words = t.contiguous().view(torch.int32).reshape(1, -1)
    src = ttnn.from_torch(words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
    dst = ttnn.from_torch(
        torch.zeros(1, n * page // 4, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=rep,
    )
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    tile = ttnn.TileDescriptor(8, 32)
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * 512,
            core_ranges=crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=512, tile=tile)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat8_b, page_size=page, tile=tile)
            ],
        ),
    ]

    def k(src_, rt_, cfg, ct=()):
        rt = ttnn.RuntimeArgs()
        rt[0][0] = rt_
        return ttnn.KernelDescriptor(
            kernel_source=KDIR + src_,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=list(ct),
            runtime_args=rt,
            config=cfg,
        )

    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = True
    cc.dst_full_sync_en = True
    kernels = [
        k(
            "pack_probe_reader.cpp",
            [src.buffer_address(), n],
            ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
            ),
        ),
        k(
            "pack_probe_writer.cpp",
            [dst.buffer_address(), n, page],
            ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        ),
        k("pack_probe_compute.cpp", [], cc, ct=[n]),
    ]
    pd = ttnn.MeshProgramDescriptor()
    pd[
        ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, mesh.get_num_devices() - 1))
    ] = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([src, dst], pd)
    return ttnn.to_torch(dst, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0].to(torch.int32).numpy().tobytes()


def decode_tiny(b, moff=16):
    """assumed layout: 16 exponent bytes (face f row r at 8 f + r), then 2 faces x 8 rows x 16 sign-magnitude bytes"""
    out = torch.zeros(8, 32)
    for f in range(2):
        for r in range(8):
            e = b[8 * f + r]
            for c in range(16):
                m = b[moff + f * 128 + r * 16 + c]
                v = (m & 0x7F) * 2.0 ** (e - 127 - 6)
                out[r, 16 * f + c] = -v if m & 0x80 else v
    return out


def main():
    mesh = open_mesh()
    try:
        torch.manual_seed(0)
        n = 4
        x = (torch.randn(8, 32 * n) * torch.linspace(0.1, 4, 32 * n)).bfloat16()
        tile = ttnn.Tile([8, 32])
        rep = ttnn.ReplicateTensorToMesh(mesh)
        xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=tile, device=mesh, mesh_mapper=rep)
        q_dev = ttnn.typecast(xd, ttnn.bfloat8_b)
        q_host = ttnn.from_torch(
            x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, tile=tile, device=mesh, mesh_mapper=rep
        )
        print("tile size", tile.get_tile_size(ttnn.bfloat8_b))
        host_ref = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, tile=tile)).float()
        for page in (320, 272):
            b = pack_probe(mesh, x, page)
            for moff in (16, 32, 64):
                dec = torch.cat([decode_tiny(b[page * t : page * (t + 1)], moff) for t in range(n)], 1)
                fin = torch.isfinite(dec)
                err = ((dec - x.float())[fin].norm() / x.float().norm()).item()
                print(
                    f"pack page {page} mantissas at {moff}: finite {fin.float().mean().item():.3f}, rel err vs bf16 {err:.4f}, "
                    f"== host bfp8 {torch.equal(dec, host_ref)}"
                )
            print("   tile 0 bytes 0..80:", b[:80].hex())
        for name, q in (("device", q_dev), ("host", q_host)):
            b = raw(mesh, q, 272, n)
            ref = ttnn.to_torch(q, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:8].float()
            dec = torch.cat([decode_tiny(b[272 * t : 272 * (t + 1)]) for t in range(n)], 1)
            ok = torch.equal(dec, ref)
            print(
                f"{name}: assumed-layout decode == ttnn decode: {ok}; max |diff| {(dec - ref).abs().max().item():.4g}"
            )
            for o in range(0, 272, 32):
                print(f"   {o:3d}: {b[o:o+32].hex()}")
            print("   ref row0", [round(v, 3) for v in ref[0, :16].tolist()])
            print("   ref row1", [round(v, 3) for v in ref[1, :16].tolist()])
            print("   ref row0 c16+", [round(v, 3) for v in ref[0, 16:32].tolist()])
            # the 32 x 32 view: gather exponents then mantissas
            v = b"".join(b[272 * t : 272 * t + 16] for t in range(4)) + b"".join(
                b[272 * t + 16 : 272 * (t + 1)] for t in range(4)
            )
            vt = ttnn.from_torch(
                torch.zeros(32, 32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
            )
            # decode the view on the host with the 32x32 layout: face F row R exp at 16 F + R, mantissa at 64 + 256 F + 16 R
            view = torch.zeros(32, 32)
            for F in range(4):
                for R in range(16):
                    e = v[16 * F + R]
                    for c in range(16):
                        m = v[64 + 256 * F + 16 * R + c]
                        val = (m & 0x7F) * 2.0 ** (e - 127 - 6)
                        row, col = (F // 2) * 16 + R, (F % 2) * 16 + c
                        view[row, col] = -val if m & 0x80 else val
            # expected: face F row R col c = tile F, face R // 8, row R % 8, col 16 (R // 8) + c
            exp_ = torch.zeros(32, 32)
            for F in range(4):
                for R in range(16):
                    row, cols = (F // 2) * 16 + R, slice((F % 2) * 16, (F % 2) * 16 + 16)
                    exp_[row, cols] = ref[R % 8, 32 * F + 16 * (R // 8) : 32 * F + 16 * (R // 8) + 16]
            print(f"  view decode == expected mapping: {torch.equal(view, exp_)}")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
