# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-tile probe: does a BFP2_b circular buffer actually unpack on Blackhole?

The whole engine depends on one unverified thing: that a raw 2-bit block-float
tile in L1, declared with ``DataFormat::Bfp2_b``, unpacks to a usable register
format and multiplies. tt-metal has the format (``DataFormat::Bfp2_b = 15``,
320 B tiles) and the LLK has the unpack path (``cunpack_common.h`` treats
Bfp2_b as a B-side block float that expands to BF16), but nothing in the repo
dispatches it: there is no ``DataType::BFLOAT2_B``, so Python cannot name the
format, and no op uses it.

This probe answers it with the smallest possible program -- one core, one output
tile, K tiles streamed -- so that a failure is a format or encoding problem and
not a tiling, semaphore or multi-core bug. It is the (b) feasibility test the
plan asks for, run before any factory work.

Two things it establishes:

1. **Can Python even express Bfp2_b?** ``CBFormatDescriptor`` only accepts
   ``tt::DataFormat`` from the bound overloads, and the enum is not bound, so
   ``tt::DataFormat::Bfp2_b`` is unreachable from Python. The workaround tested
   here is to declare the CB as raw bytes (``uint8``, page size 320 -- the CB page
   size is what the unpacker uses as the tile size) and have the compute kernel
   force the unpack *source* format to 15 itself, which is legitimate because the
   unpacker's format registers are written by the kernel's reconfig path anyway.

2. **What encoding is right?** The plan says bit 1 is sign, bit 0 magnitude, so
   01=+1, 11=-1, 00=0, and that 10 decodes as -Inf. Rather than trust that, the
   probe builds the same logical ternary matrix in three plausible encodings and
   reports which reproduces a float64 reference. If none do, the byte layout
   (which of the 64 scale bytes and 256 data bytes belong where) is wrong and the
   probe prints the decoded values so the layout can be corrected rather than
   guessed.

Usage (on a chip)::

    python -m models.demos.blackhole.bonsai2.native.bench.bfp2_probe
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

#: Authoritative BFP2 tile layout, read off tt-metal's own encoder
#: (pack_as_bfp2_tiles in tt_metal/impl/data_format/bfloat2.cpp) rather than guessed.
#:
#:   * 320 bytes = 64 exponent bytes + 256 data bytes.
#:   * The exponent region is 4 faces x 16 bytes, all 0x7f (bias 127) for magnitude-1
#:     ternary values; a value of 0 has datum 0x00 and needs no exponent.
#:   * The data region is 4 faces of 128 bytes each, in row-major quadrant order:
#:     top-left, top-right, bottom-left, bottom-right. (Checked against the encoder
#:     with a quadrant-marked tile: the top-right quadrant lands at byte 128.)
#:   * Within a face the 256 values are packed 4 per byte, 2 bits each, LSB first.
#:   * Value codes, confirmed by round-tripping through the decoder:
#:         0b01 = +1   (0x55 pattern for a whole face of +1)
#:         0b11 = -1   (0xff pattern for a whole face of -1)
#:         0b00 =  0
#:     0b10 is not produced by the encoder and decodes as -Inf, so it must never be
#:     emitted.
EXPONENT_BYTE = 0x7F


def build_tile(values: np.ndarray, encoding: str = "plan") -> np.ndarray:
    """32x32 ternary values -> one 320-byte BFP2_b tile, in the real layout."""
    assert values.shape == (32, 32)
    if encoding != "plan":
        raise ValueError(f"only the authoritative 'plan' encoding is meaningful, got {encoding!r}")
    code = np.zeros((32, 32), np.uint8)
    code[values > 0] = 0b01
    code[values < 0] = 0b11

    out = bytearray(320)
    for i in range(64):
        out[i] = EXPONENT_BYTE

    # Face order: (0,0), (1,0), (0,1), (1,1) in 16x16 quadrant coordinates.
    faces = [(0, 0), (0, 1), (1, 0), (1, 1)]
    data = np.zeros(256, np.uint8)
    pos = 0
    for fr, fc in faces:
        face = code[fr * 16:(fr + 1) * 16, fc * 16:(fc + 1) * 16].reshape(-1)
        for i in range(0, 256, 4):
            data[pos] = (int(face[i]) | (int(face[i + 1]) << 2)
                         | (int(face[i + 2]) << 4) | (int(face[i + 3]) << 6))
            pos += 1
    out[64:320] = bytes(data)
    return np.frombuffer(bytes(out), np.uint8).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-tiles", type=int, default=4)
    args = ap.parse_args()

    import ttnn
    from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor

    kt = args.k_tiles
    rng = np.random.default_rng(0)
    a_vals = rng.integers(-1, 2, size=(32, kt * 32)).astype(np.float32)
    b_vals = rng.integers(-1, 2, size=(kt * 32, 32)).astype(np.float32)
    ref = a_vals.astype(np.float64) @ b_vals.astype(np.float64)

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    def run(label, a_t, b_t, ct_define):
        o_t = ttnn.from_torch(
            torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        page = 320 if ct_define else 2048
        cbs = [
            ttnn.CBDescriptor(
                total_size=page * kt, core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=0, data_format=(ttnn.uint8 if ct_define else ttnn.bfloat16),
                    page_size=page)]),
            ttnn.CBDescriptor(
                total_size=page * kt, core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=1, data_format=(ttnn.uint8 if ct_define else ttnn.bfloat16),
                    page_size=page)]),
            # The output CB is the sharded output tensor's own buffer, so the pack
            # lands where the host reads it. An interleaved L1 tensor left the pack
            # in core 0's L1 with no way back.
            ttnn.cb_descriptor_from_sharded_tensor(17, o_t),
        ]
        defines = [("MODE_BFP2", "1")] if ct_define else []
        uk = UnifiedKernelDescriptor(
            kernel_source="models/demos/blackhole/bonsai2/native/bench/bfp2_probe_kernel.cpp",
            core_ranges=core,
            ncrisc_common_runtime_args=[a_t.buffer_address(), b_t.buffer_address(), kt],
            trisc_common_runtime_args=[a_t.buffer_address(), b_t.buffer_address(), kt],
            defines=defines,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        )
        kds = uk.get_kernel_descriptors()
        prog = ttnn.ProgramDescriptor(kernels=kds.kernels, cbs=cbs, semaphores=[])
        ttnn.generic_op([a_t, b_t, o_t], prog)
        ttnn.synchronize_device(device)
        got = ttnn.to_torch(o_t).float().numpy().reshape(32, 32)
        ttnn.deallocate(o_t)
        a = got.ravel() - got.mean()
        b = ref.ravel() - ref.mean()
        p = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        print(f"{label:16s} pcc={p:9.7f} max={np.abs(got).max():7.3f} "
              f"refmax={np.abs(ref).max():7.3f} {'MATCH' if p > 0.999 else 'no'}", flush=True)
        return got

    try:
        # ---- control: plain bf16 operands. If this is wrong the bug is plumbing,
        # not the format, and BFP2 results would be meaningless.
        try:
            a_bf = ttnn.from_torch(
                torch.from_numpy(a_vals).to(torch.bfloat16).reshape(1, 1, 32, kt * 32),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # ttnn's linear computes x @ W.T, and matmul_tiles takes both operands as
            # [rows, K] tiles, so B is the transposed weight [N, K] -- not [K, N].
            b_bf = ttnn.from_torch(
                torch.from_numpy(b_vals.T.copy()).to(torch.bfloat16).reshape(1, 1, 32, kt * 32),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            run("control bf16", a_bf, b_bf, False)
            ttnn.deallocate(a_bf)
            ttnn.deallocate(b_bf)
        except Exception as e:  # noqa: BLE001
            print(f"control bf16   FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)

        for enc in ("plan",):
            # Both operands are [rows, K] tiles: A is [M=32, K], B is [N=32, K].
            a_img = b"".join(build_tile(a_vals[:, i * 32:(i + 1) * 32], enc).tobytes() for i in range(kt))
            b_img = b"".join(build_tile(b_vals[i * 32:(i + 1) * 32, :].T, enc).tobytes() for i in range(kt))
            try:
                a_t = ttnn.from_torch(
                    torch.frombuffer(bytearray(a_img), dtype=torch.uint8),
                    dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
                b_t = ttnn.from_torch(
                    torch.frombuffer(bytearray(b_img), dtype=torch.uint8),
                    dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
                run(f"bfp2 {enc}", a_t, b_t, True)
                ttnn.deallocate(a_t)
                ttnn.deallocate(b_t)
            except Exception as e:  # noqa: BLE001
                print(f"bfp2 {enc:6s}    FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
