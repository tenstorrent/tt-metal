// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Row-granular DEST copies, the bodies of copy_dest_rows / copy_dest_row (api/compute/copy_dest_rows.h).
//
// DEST geometry as sfpi sees it: a tile slot is 32 sfpi vectors (dst_reg[32 * slot + v]); face f (0..3) is vectors
// 8f..8f+7; within a face, vector 2b + g holds the face's rows 4b..4b+3 at lane 8j + k = row 4b + j, column 2k + g
// (g = 0 the even columns, g = 1 the odd ones). The 4-row band `band` (0..7) of the whole tile is therefore the two
// faces of its half (bands 0..3: faces 0 and 1, bands 4..7: faces 2 and 3) at vectors 2 * (band % 4) and the next, four
// vectors in all, and row 4 * band + j is lane group j (lanes 8j..8j+7) of each of those four. vConstTileId = 2 * lane,
// so the lane group is vConstTileId >> 4. The band kernel of ckernel_sfpu_binary_bcast.h walks the same geometry.
//
// Every load and store is a raw one (sfpi DataLayout U32 on an fp32 DEST, U16 on a 16-bit one): the SFPU's float
// load / store modes flush denormals, the raw modes move the datum's bits untouched, so the copies are exact for every
// bit pattern, NaN payloads, the sign of zero and denormals included.
namespace copy_dest_rows_detail {

constexpr std::uint32_t kSlotVecs = 32;
constexpr std::uint32_t kFaceVecs = 8;

struct BandVecs {
    std::uint32_t v[4];
};

// The four sfpi vectors of band `band` of the tile in slot `slot`: left face even / odd columns, right face even / odd.
inline BandVecs band_vecs(std::uint32_t slot, std::uint32_t band) {
    const std::uint32_t base = slot * kSlotVecs + (band / 4) * 2 * kFaceVecs + 2 * (band % 4);
    return {{base, base + 1, base + kFaceVecs, base + kFaceVecs + 1}};
}

}  // namespace copy_dest_rows_detail

// Rows 4 * out_band .. +3 of DEST[dst_index_out] <- rows 4 * in_band .. +3 of DEST[dst_index_in], all 32 columns, as
// four raw vector copies. Called once under VectorMode::None by the binary SFPU params wrapper.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void copy_dest_band(
    std::uint32_t dst_index_in,
    std::uint32_t dst_index_out,
    std::uint32_t /*unused*/,
    std::uint32_t in_band,
    std::uint32_t out_band) {
    using namespace copy_dest_rows_detail;
    constexpr sfpi::DataLayout L = is_fp32_dest_acc_en ? sfpi::DataLayout::U32 : sfpi::DataLayout::U16;
    const BandVecs s = band_vecs(dst_index_in, in_band);
    const BandVecs d = band_vecs(dst_index_out, out_band);
#pragma GCC unroll 4
    for (std::uint32_t i = 0; i < 4; i++) {
        const sfpi::vUInt x = sfpi::dst_reg[s.v[i]].mode<L>();
        sfpi::dst_reg[d.v[i]].mode<L>() = x;
    }
}

// Row to_row of DEST[dst_index_out] <- row from_row of DEST[dst_index_in], all 32 columns; the target's other rows
// keep their bits. The rows live in lane group from_row % 4 of the source band's vectors and lane group to_row % 4 of
// the target band's. Same lane group: a lane-masked move. Different lane groups: SFPTRANSP, which maps subvector k of
// input m to subvector m of output k, so the source vector placed at input position to_row % 4 (the other inputs
// zero) has its subvector from_row % 4 at subvector to_row % 4 of output from_row % 4; that output is then merged into
// the target under the lane mask. Two source vectors ride one transpose (its two independent 4-register groups).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void copy_dest_row(
    std::uint32_t dst_index_in,
    std::uint32_t dst_index_out,
    std::uint32_t /*unused*/,
    std::uint32_t from_row,
    std::uint32_t to_row) {
    using namespace copy_dest_rows_detail;
    constexpr sfpi::DataLayout L = is_fp32_dest_acc_en ? sfpi::DataLayout::U32 : sfpi::DataLayout::U16;
    const BandVecs s = band_vecs(dst_index_in, from_row / 4);
    const BandVecs d = band_vecs(dst_index_out, to_row / 4);
    const std::uint32_t fi = from_row % 4;
    const std::uint32_t ti = to_row % 4;
    const int ti_i = static_cast<int>(ti);
    if (fi == ti) {
#pragma GCC unroll 4
        for (std::uint32_t i = 0; i < 4; i++) {
            const sfpi::vUInt x = sfpi::dst_reg[s.v[i]].mode<L>();
            sfpi::vUInt y = sfpi::dst_reg[d.v[i]].mode<L>();
            const sfpi::vInt sub = sfpi::vConstTileId >> 4;
            v_if(sub == ti_i) { y = x; }
            v_endif;
            sfpi::dst_reg[d.v[i]].mode<L>() = y;
        }
        return;
    }
#pragma GCC unroll 2
    for (std::uint32_t p = 0; p < 2; p++) {
        sfpi::vUInt a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
        sfpi::vUInt b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
        switch (ti) {
            case 0:
                a0 = sfpi::dst_reg[s.v[2 * p]].mode<L>();
                b0 = sfpi::dst_reg[s.v[2 * p + 1]].mode<L>();
                break;
            case 1:
                a1 = sfpi::dst_reg[s.v[2 * p]].mode<L>();
                b1 = sfpi::dst_reg[s.v[2 * p + 1]].mode<L>();
                break;
            case 2:
                a2 = sfpi::dst_reg[s.v[2 * p]].mode<L>();
                b2 = sfpi::dst_reg[s.v[2 * p + 1]].mode<L>();
                break;
            default:
                a3 = sfpi::dst_reg[s.v[2 * p]].mode<L>();
                b3 = sfpi::dst_reg[s.v[2 * p + 1]].mode<L>();
                break;
        }
        sfpi::subvec_transp(a0, a1, a2, a3, b0, b1, b2, b3);
        sfpi::vUInt ra, rb;
        switch (fi) {
            case 0:
                ra = a0;
                rb = b0;
                break;
            case 1:
                ra = a1;
                rb = b1;
                break;
            case 2:
                ra = a2;
                rb = b2;
                break;
            default:
                ra = a3;
                rb = b3;
                break;
        }
        sfpi::vUInt y0 = sfpi::dst_reg[d.v[2 * p]].mode<L>();
        sfpi::vUInt y1 = sfpi::dst_reg[d.v[2 * p + 1]].mode<L>();
        const sfpi::vInt sub = sfpi::vConstTileId >> 4;
        v_if(sub == ti_i) {
            y0 = ra;
            y1 = rb;
        }
        v_endif;
        sfpi::dst_reg[d.v[2 * p]].mode<L>() = y0;
        sfpi::dst_reg[d.v[2 * p + 1]].mode<L>() = y1;
    }
}

}  // namespace sfpu
}  // namespace ckernel
