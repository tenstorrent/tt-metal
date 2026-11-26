# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S0-9: the Bfp2_b 32 x 32 tile on device (a CB declared Bfp2_b through CBFormatDescriptor.data_format_as_uint8 = 15):
(1) pack — bf16 tiles through DST into a Bfp2_b CB, the raw bytes decoded under candidate layouts (64 exponent bytes, face
F row R at 16 F + R; then 4 B a face row of 16 elements x 2 bits; element order within a byte and the sign / magnitude
bit tried); (2) unpack — host-encoded Bfp2_b bytes copied through DST to bf16, compared with the host decode.
usage: python -m qwen36_resident.tests.bfp2_probe"""
import itertools

import torch

import ttnn
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

KDIR = __file__.rsplit("/tests/", 1)[0] + "/kernels/"
BFP2_B = 15  # tt::DataFormat::Bfp2_b
PAGE = 64 + 1024 * 2 // 8  # 320 B


def faces(x):
    """a 32 x 32 tile's elements in face order: [4 faces, 16 rows, 16 cols]"""
    return torch.stack([x[16 * (f // 2) : 16 * (f // 2) + 16, 16 * (f % 2) : 16 * (f % 2) + 16] for f in range(4)])


def unfaces(fc):
    x = torch.zeros(32, 32)
    for f in range(4):
        x[16 * (f // 2) : 16 * (f // 2) + 16, 16 * (f % 2) : 16 * (f % 2) + 16] = fc[f]
    return x


def decode(b, order, sign_hi):
    """b: 320 bytes; order: element j of a byte at bits order[j] * 2 .. +1; sign_hi: the sign the higher bit of the pair"""
    fc = torch.zeros(4, 16, 16)
    for f in range(4):
        for r in range(16):
            e = b[16 * f + r]
            row = b[64 + 64 * f + 4 * r : 64 + 64 * f + 4 * r + 4]
            for c in range(16):
                byte = row[c // 4]
                two = (byte >> (2 * order[c % 4])) & 3
                s, m = ((two >> 1) & 1, two & 1) if sign_hi else (two & 1, (two >> 1) & 1)
                v = m * 2.0 ** (e - 127)
                fc[f, r, c] = -v if s else v
    return unfaces(fc)


def encode(x, order=(0, 1, 2, 3), sign_hi=True):
    """host Bfp2_b: per face row the shared exponent of its max |x|, magnitude bit = |x| >= 2^(e - 1) (round to nearest
    between 0 and 2^e), sign bit"""
    fc = faces(x.float())
    b = bytearray(PAGE)
    for f in range(4):
        for r in range(16):
            v = fc[f, r]
            amax = v.abs().max().item()
            e = 0 if amax == 0 else int(torch.floor(torch.log2(torch.tensor(amax))).item()) + 127
            e = max(0, min(255, e))
            b[16 * f + r] = e
            for c in range(16):
                m = 1 if amax > 0 and abs(v[c].item()) >= 2.0 ** (e - 127 - 1) else 0
                s = 1 if v[c].item() < 0 and m else 0
                two = (s << 1 | m) if sign_hi else (m << 1 | s)
                b[64 + 64 * f + 4 * r + c // 4] |= two << (2 * order[c % 4])
    return bytes(b)


def run(mesh, data, n, in_fmt, in_page, out_fmt, out_page):
    rep = ttnn.ReplicateTensorToMesh(mesh)
    words = torch.frombuffer(bytearray(data), dtype=torch.int32).reshape(1, -1)
    src = ttnn.from_torch(words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
    dst = ttnn.from_torch(
        torch.zeros(1, n * out_page // 4, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=rep,
    )
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def fmt(idx, f, page):
        fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat8_b, page_size=page)
        fd.data_format_as_uint8 = f
        return ttnn.CBDescriptor(total_size=n * page, core_ranges=crs, format_descriptors=[fd])

    cbs = [fmt(0, in_fmt, in_page), fmt(1, out_fmt, out_page)]

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
    dm = lambda p, noc: ttnn.DataMovementConfigDescriptor(processor=p, noc=noc)  # noqa: E731
    kernels = [
        k(
            "bfp2_probe_reader.cpp",
            [src.buffer_address(), n, in_page],
            dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
        ),
        k(
            "bfp2_probe_writer.cpp",
            [dst.buffer_address(), n, out_page],
            dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
        ),
        k("pack_probe_compute.cpp", [], cc, ct=[n]),
    ]
    pd = ttnn.MeshProgramDescriptor()
    pd[
        ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, mesh.get_num_devices() - 1))
    ] = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([src, dst], pd)
    return ttnn.to_torch(dst, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0].to(torch.int32).numpy().tobytes()


def bf16_tiles(xs):
    """bf16 32 x 32 tiles in face order, bytes"""
    return b"".join(faces(x).bfloat16().contiguous().view(torch.int16).numpy().tobytes() for x in xs)


def from_bf16(b, n):
    t = torch.frombuffer(bytearray(b), dtype=torch.bfloat16).reshape(n, 4, 16, 16).float()
    return [unfaces(t[i]) for i in range(n)]


def main():
    mesh = open_mesh()
    try:
        torch.manual_seed(0)
        n = 2
        xs = [torch.randn(32, 32) * torch.linspace(0.05, 3, 32)[:, None] for _ in range(n)]
        bf = ttnn.DataType.BFLOAT16 if hasattr(ttnn.DataType, "BFLOAT16") else None
        BF16 = 5  # tt::DataFormat::Float16_b
        # (1) pack: bf16 -> Bfp2_b
        raw = run(mesh, bf16_tiles(xs), n, BF16, 2048, BFP2_B, PAGE)
        print("tile 0 bytes 0..96:", raw[:96].hex())
        best = None
        for order in itertools.permutations(range(4)):
            for sign_hi in (True, False):
                dec = [decode(raw[PAGE * i : PAGE * (i + 1)], order, sign_hi) for i in range(n)]
                # a sound decode keeps every sign of the nonzero outputs equal to the input's
                agree = sum(((d != 0) & (torch.sign(d) != torch.sign(x))).sum().item() for d, x in zip(dec, xs))
                err = sum((d - x).norm().item() for d, x in zip(dec, xs)) / sum(x.norm().item() for x in xs)
                if best is None or (agree, err) < best[0]:
                    best = ((agree, err), order, sign_hi, dec)
        (bad, err), order, sign_hi, dec = best
        print(f"pack: best layout order {order} sign_hi {sign_hi}: sign mismatches {bad}, rel err {err:.4f}")
        host = [decode(encode(x, order, sign_hi), order, sign_hi) for x in xs]
        print(
            f"pack: device == host encode (round to nearest): {all(torch.equal(a, b) for a, b in zip(dec, host))}; "
            f"rel diff {sum((a - b).norm().item() for a, b in zip(dec, host)) / sum(b.norm().item() for b in host):.4f}"
        )
        # (2) unpack: host Bfp2_b -> bf16
        enc = b"".join(encode(x, order, sign_hi) for x in xs)
        back = from_bf16(run(mesh, enc, n, BFP2_B, PAGE, BF16, 2048), n)
        ref = [decode(encode(x, order, sign_hi), order, sign_hi) for x in xs]
        print(
            f"unpack: device == host decode: {all(torch.equal(a, b) for a, b in zip(back, ref))}; "
            f"max |diff| {max((a - b).abs().max().item() for a, b in zip(back, ref)):.4g}"
        )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
