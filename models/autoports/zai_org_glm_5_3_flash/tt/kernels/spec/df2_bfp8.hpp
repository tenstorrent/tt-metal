// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 17 DFlash2 at 1M (tt/dflash2_1m.py, GLM53_DF2_KV_DTYPE=bfp8): the drafter's K / V cache tiles as bfloat8_b,
// written row by row from bf16 rows by a data-movement RISC. A bfp8_b tile (1,088 B) = 64 shared-exponent bytes (face
// f, face row i at byte f 16 + i) + 4 faces x 16 rows x 16 one-byte codes (sign << 7 | 7-bit mantissa) at 64 + (f 16 +
// i) 16 + c. One token row of a 32 x 32 tile is two face rows (faces (r / 16) 2 and (r / 16) 2 + 1, face row r % 16),
// each encoded on its own: the shared exponent is the max bf16 exponent of the 16 values, every mantissa (with its
// hidden 1) is shifted right by the exponent difference (truncating), rounded to nearest-even to 7 bits, saturated at
// 127, zero / denormal inputs code 0 -- byte for byte tt-metal's host packer (tt_metal/impl/data_format/
// bfp_fast_pack.cpp pack_row, is_exp_a = false; the host mirror tt/dflash2_1m.py bfp8_rows is bitwise against
// ttnn.from_torch(bfloat8_b)). Integer arithmetic only.
#pragma once

#include <cstdint>

namespace df2bfp8 {

constexpr uint32_t TILE_BYTES = 1088;
constexpr uint32_t EXP_BYTES = 64;

// 16 bf16 values at src (32 B, L1) -> the exponent byte and the 16 codes of face row (face, row) of the bfp8 tile at
// `tile` (L1)
FORCE_INLINE void encode_row16(uint32_t src, uint32_t tile, uint32_t face, uint32_t row) {
    volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
    volatile tt_l1_ptr uint8_t* exps = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile);
    volatile tt_l1_ptr uint8_t* codes =
        reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile + EXP_BYTES + (face * 16u + row) * 16u);
    uint32_t b[16];
    uint32_t e[16];
    uint32_t shared = 0;
    for (uint32_t j = 0; j < 16; ++j) {
        b[j] = s[j];
        e[j] = (b[j] >> 7) & 0xffu;
        shared = e[j] > shared ? e[j] : shared;
    }
    exps[face * 16u + row] = static_cast<uint8_t>(shared);
    for (uint32_t j = 0; j < 16; ++j) {
        uint32_t d = shared - e[j];
        d = d > 31u ? 31u : d;
        const uint32_t mant = ((1u << 23) | ((b[j] & 0x7fu) << 16)) >> d;
        const uint32_t round_value = mant & 0x1ffffu;  // the 17 bits below the 7 kept
        uint32_t q = mant >> 17;
        q += (round_value > 0x10000u) | ((round_value == 0x10000u) & q & 1u);
        q = q > 127u ? 127u : q;
        q = e[j] == 0 ? 0 : q;
        const uint32_t sign = q == 0 ? 0 : (b[j] >> 15);
        codes[j] = static_cast<uint8_t>((sign << 7) | q);
    }
}

// a token row (32 bf16: the left face row at src, the right one at src + 512 of a bf16 tile) -> row r of the bfp8
// tile at dst_tile
FORCE_INLINE void patch_row(uint32_t src, uint32_t dst_tile, uint32_t r) {
    const uint32_t f0 = (r >> 4) * 2u, i = r & 15u;
    encode_row16(src, dst_tile, f0, i);
    encode_row16(src + 512u, dst_tile, f0 + 1u, i);
}

}  // namespace df2bfp8
