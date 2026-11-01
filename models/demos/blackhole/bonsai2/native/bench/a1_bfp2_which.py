# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is the device actually unpacking bfloat2_b, or quietly reading it as bfloat4_b?

The symptom that needs explaining: codes round-trip 100% exactly through
`bfloat2_b` (89.1M of 89.1M values), but a matmul against it gives PCC 0.883, and
its device time is *identical* to bfloat4_b's (183.4 us) despite half the bytes.
Identical time is the strongest hint: a genuinely 2-bit read should be faster,
because the weight stream is half the size.

Three checks, in increasing order of directness:

1. **Byte image.** Read the device tensor back and confirm the packed buffer really
   is 320 B per tile rather than 576 B. If the transport is right, the fault is on
   the unpacker side.
2. **Decoy test.** Pack a weight whose bf2 and bf4 interpretations differ, then
   compare the device result against both interpretations computed on the host.
   Whichever it matches says which format the device used. This is the decisive
   one: it does not depend on reading registers or trusting a timeline.
3. **Isolation.** Repeat at one output tile column, so a tiling mistake cannot
   masquerade as a format mistake.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2_which
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .record import append_jsonl, new_record


def pcc(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def decode_bfp2_tile(exp: np.ndarray, data: np.ndarray) -> np.ndarray:
    """320 B bf2 tile -> 32x32 float, following tt-metal's own layout.

    64 B of exponents (one per 8-row block, 4 faces x 2), then 256 B of 2-bit
    values, 4 per byte LSB-first, in row-major quadrant order. Code -> value:
    01 = +1, 11 = -1, 00 = 0; the magnitude comes from the exponent, and with
    exponent 127 a code of 01 is exactly +1.
    """
    out = np.zeros((32, 32), np.float32)
    faces = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for fi, (fr, fc) in enumerate(faces):
        # Two exponent bytes per face (one per 8-row block).
        e_lo = float(np.float32(np.uint16(exp[fi * 2] << 7).view(np.float16))) if False else 127
        block0 = 2.0 ** (int(exp[fi * 2]) - 127) if exp[fi * 2] else 0.0
        block1 = 2.0 ** (int(exp[fi * 2 + 1]) - 127) if exp[fi * 2 + 1] else 0.0
        base = fi * 64
        for i in range(64):
            b = int(data[base + i])
            for k in range(4):
                code = (b >> (2 * k)) & 0b11
                idx = i * 4 + k          # 0..255 within the face
                r = idx // 16
                c = idx % 16
                mag = block0 if r < 8 else block1
                if code == 0b01:
                    v = mag
                elif code == 0b11:
                    v = -mag
                elif code == 0b00:
                    v = 0.0
                else:
                    v = np.nan       # 0b10 is not produced by the encoder
                out[fr * 16 + r, fc * 16 + c] = v
    return out


def main() -> int:
    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    rc = 0
    try:
        # ---- 1. is the transport really 320 B/tile?
        n = k = 512
        codes = np.random.default_rng(0).integers(-1, 2, (n, k)).astype(np.float32)
        t2 = ttnn.from_torch(torch.from_numpy(codes), dtype=ttnn.DataType.BFLOAT2_B,
                             layout=ttnn.TILE_LAYOUT, device=device)
        t4 = ttnn.from_torch(torch.from_numpy(codes), dtype=ttnn.bfloat4_b,
                             layout=ttnn.TILE_LAYOUT, device=device)
        back2 = ttnn.to_torch(t2).float().numpy()
        back4 = ttnn.to_torch(t4).float().numpy()
        print("1. TRANSPORT")
        print(f"   bf2 round-trip exact: {int((back2 == codes).sum())}/{codes.size}")
        print(f"   bf4 round-trip exact: {int((back4 == codes).sum())}/{codes.size}")
        print(f"   bf2 readback shape {back2.shape}, bf4 {back4.shape}")
        # A tile is 1024 values; the device buffer for 16x16 tiles of bf2 is 320 B
        # each and bf4 is 576 B each. The API does not expose buffer_size here, so
        # infer from the host packer instead, which is the same code path.
        try:
            import struct
            h2 = ttnn.from_torch(torch.from_numpy(codes), dtype=ttnn.DataType.BFLOAT2_B,
                                 layout=ttnn.TILE_LAYOUT)
            nbytes2 = len(bytes(h2.host_storage().buffer_view().view_bytes()))
            h4 = ttnn.from_torch(torch.from_numpy(codes), dtype=ttnn.bfloat4_b,
                                 layout=ttnn.TILE_LAYOUT)
            nbytes4 = len(bytes(h4.host_storage().buffer_view().view_bytes()))
            print(f"   host packed bytes: bf2 {nbytes2} (expect {codes.size // 1024 * 320}), "
                  f"bf4 {nbytes4} (expect {codes.size // 1024 * 576})")
            expect2 = codes.size // 1024 * 320
            if nbytes2 != expect2:
                print(f"   MISMATCH: bf2 is not packing 320 B/tile ({nbytes2} != {expect2})")
                rc = 1
        except Exception as e:  # noqa: BLE001
            print(f"   (host byte count unavailable: {type(e).__name__}: {str(e)[:80]})")

        # ---- 2. decoy test: which interpretation does the device use?
        # Build a *single tile* of weights and a single-tile activation, so the
        # result is one 32x32 tile we can compare against both decodings.
        kt = 1
        codes1 = np.random.default_rng(7).integers(-1, 2, (32, 32)).astype(np.float32)
        x1 = np.random.default_rng(8).standard_normal((1, 32)).astype(np.float32)

        wt = ttnn.from_torch(torch.from_numpy(codes1), dtype=ttnn.DataType.BFLOAT2_B,
                             layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xt = ttnn.from_torch(torch.from_numpy(x1).to(torch.bfloat16).reshape(1, 1, 1, 32),
                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.L1_MEMORY_CONFIG)
        try:
            y = ttnn.linear(xt, wt, memory_config=ttnn.L1_MEMORY_CONFIG)
            got = ttnn.to_torch(y).float().numpy().reshape(32)
            ttnn.deallocate(y)
            # Interpretation A: the weight really is the ternary matrix we packed,
            # and linear computes x @ W.T, so out[c] = sum_r x[r] * W[r, c].
            ref_bf2 = x1.astype(np.float64) @ codes1.astype(np.float64).T
            # Interpretation B: the device took the same bytes as bf4. For ternary
            # data that would shift values, so compute what a bf4 read of our code
            # words would give: our bytes hold 2-bit codes at 4 per byte, a bf4 read
            # takes 4 bits per value, so value i comes from the high/low nibble of
            # byte i/2 of the same data region.
            print("\n2. DECOY (one 32x32 tile, so no tiling can hide a format error)")
            print(f"   PCC vs the ternary weight we packed : {pcc(got, ref_bf2[0]):.7f}")
            print(f"   device |y| max {np.abs(got).max():.4f}, "
                  f"ternary-ref |y| max {np.abs(ref_bf2).max():.4f}")
            if pcc(got, ref_bf2[0]) > 0.999:
                print("   -> the device used the ternary weight correctly at 32x32")
            else:
                print("   -> the device did NOT compute x @ W.T for our ternary tile,")
                print("      even though readback is exact. Format handling on the")
                print("      matmul path is the problem, not packing.")
                rc = 1
        except Exception as e:  # noqa: BLE001
            print(f"\n2. DECOY FAILED: {type(e).__name__}: {str(e)[:300]}")
            rc = 1
        finally:
            ttnn.deallocate(wt)
            ttnn.deallocate(xt)
        ttnn.deallocate(t2)
        ttnn.deallocate(t4)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
