// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// A's BONSAI_SCALE16 scale tiles are fp16 (2 KB). The prefill kernels convert the 8 rows a unit uses to fp32 in
// L1 on the reader, so compute keeps its fp32 SrcB path. (Reprogramming SrcB to Float16 in compute decoded the
// bits with an 8-bit-exponent bias: every prefill output came out ~1e-33 too small, see C 01:3x.)
// The fp16 tile is read into the upper half of the 4 KB fp32 slot; element e (face order, same index in both
// formats) moves from slot + 2048 + 2e to slot + 4e.
#pragma once
#include <cstdint>

inline uint32_t fp16_to_fp32_bits(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    if (exp == 0) {
        if (mant == 0) {
            return sign;  // +-0
        }
        while (!(mant & 0x400u)) {  // subnormal: normalize
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3ffu;
    } else if (exp == 31) {
        return sign | 0x7f800000u | (mant << 13);  // inf / nan
    }
    return sign | ((exp + 112u) << 23) | (mant << 13);
}

// Convert rows row0 .. row0+nrows-1 (nrows <= 8) of the fp16 tile at slot+2048 into the fp32 tile at slot. The
// selected fp16 values are copied out first: row order isn't monotonic in element index across faces, so an
// in-place pass could overwrite a source it hasn't read yet.
inline void fp16_rows_to_fp32(uint32_t slot, uint32_t row0, uint32_t nrows) {
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot + 2048);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
    uint16_t buf[8 * 32];
    uint32_t idx[8 * 32];
    uint32_t n = 0;
    for (uint32_t r = row0; r < row0 + nrows; ++r) {
        const uint32_t base = (r / 16) * 2 * 256 + (r % 16) * 16;  // faces (r/16, 0) and (r/16, 1)
        for (uint32_t f = 0; f < 2; ++f) {
            for (uint32_t c = 0; c < 16; ++c, ++n) {
                idx[n] = base + f * 256 + c;
                buf[n] = src[idx[n]];
            }
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        dst[idx[i]] = fp16_to_fp32_bits(buf[i]);
    }
}
