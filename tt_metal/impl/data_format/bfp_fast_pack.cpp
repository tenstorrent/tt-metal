// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fast host conversion of row-major bfloat16 data into Bfp8_b / Bfp4_b tiles (32x32, 16x16 faces), byte-identical to
// encode_tensor_data + pack_as_bfp_tiles (blockfloat_common.cpp) with is_exp_a = false: same shared exponent (max fp32
// exponent field per 16-value face row), same truncating shift, same round-to-nearest-even and saturation, same zero
// padding of the last two dims to multiples of 32. One pass straight from row-major bf16 (no fp32 copy, no separate
// tilize), every tile written at its fixed offset, threads over 32-row bands, AVX2 clone picked at runtime.

#include <tt-metalium/bfp_fast_pack.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <cstdlib>
#include <memory>
#include <sys/mman.h>
#include <vector>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::bfp_fast_pack {
namespace {

constexpr uint32_t kTile = 32;
constexpr uint32_t kFace = 16;
constexpr uint32_t kExpBytes = 64;  // 4 faces x 16 rows, one shared exponent byte each

// One bf16 value -> sign|mantissa field of width MANT_W + 1. Mirrors convert_u32_to_bfp<Bfp*_b, false>.
template <uint32_t MANT_W>
inline uint32_t to_bfp(uint16_t b, uint32_t shared_exp) {
    const uint32_t exp = (b >> 7) & 0xffu;
    if (exp == 0) {
        return 0;  // +/- zero or denormal
    }
    uint32_t sign = b >> 15;
    uint32_t mant = (1u << 23) | (static_cast<uint32_t>(b & 0x7fu) << 16);
    if (shared_exp > exp) {
        int d = static_cast<int>(shared_exp - exp);
        while (d > 31) {
            mant >>= 31;
            d -= 31;
        }
        mant >>= d;
    }
    constexpr uint32_t SHIFT = 24 - MANT_W;
    constexpr uint32_t ROUND_MASK = (1u << SHIFT) - 1;
    constexpr uint32_t TIE = 1u << (SHIFT - 1);
    constexpr uint32_t MAX_VAL = (1u << MANT_W) - 1;
    const uint32_t round_value = mant & ROUND_MASK;
    mant >>= SHIFT;
    if (round_value > TIE || (round_value == TIE && (mant & 1u))) {
        mant += 1;
    }
    mant = std::min(mant, MAX_VAL);
    if (mant == 0) {
        sign = 0;
    }
    return (sign << MANT_W) | mant;
}

// One 16-value face row -> shared exponent + 16 sign|mantissa codes. Branchless form of to_bfp (kept above as the
// readable reference): the shared exponent is the row max so the shift is never negative, a shift of >= 24 already
// clears the 24-bit mantissa (so clamping it to 31 matches the original's repeated >> 31), and zero/denormal inputs
// are masked to 0 afterwards. Written so the compiler can vectorize the 16 lanes.
template <uint32_t MANT_W>
inline uint32_t pack_row(const uint16_t* row, uint8_t* codes) {
    constexpr uint32_t SHIFT = 24 - MANT_W;
    constexpr uint32_t ROUND_MASK = (1u << SHIFT) - 1;
    constexpr uint32_t TIE = 1u << (SHIFT - 1);
    constexpr uint32_t MAX_VAL = (1u << MANT_W) - 1;
    uint32_t e[kFace];
    uint32_t shared = 0;
    for (uint32_t j = 0; j < kFace; ++j) {
        e[j] = (row[j] >> 7) & 0xffu;
        shared = e[j] > shared ? e[j] : shared;
    }
    for (uint32_t j = 0; j < kFace; ++j) {
        const uint32_t b = row[j];
        uint32_t d = shared - e[j];
        d = d > 31 ? 31 : d;
        const uint32_t mant = ((1u << 23) | ((b & 0x7fu) << 16)) >> d;
        const uint32_t round_value = mant & ROUND_MASK;
        uint32_t q = mant >> SHIFT;
        q += (round_value > TIE) | ((round_value == TIE) & q & 1u);
        q = q > MAX_VAL ? MAX_VAL : q;
        q = e[j] == 0 ? 0 : q;
        const uint32_t sign = q == 0 ? 0 : (b >> 15);
        codes[j] = static_cast<uint8_t>((sign << MANT_W) | q);
    }
    return shared;
}

// Pack one 32x32 tile whose top-left element is `src` (row stride `stride` elements) into `dst`.
template <uint32_t MANT_W>
inline void pack_tile(const uint16_t* src, size_t stride, uint32_t* dst) {
    constexpr uint32_t BYTES_PER_ROW = kFace * (MANT_W + 1) / 8;  // 16 for bfp8, 8 for bfp4
    auto* exps = reinterpret_cast<uint8_t*>(dst);
    auto* mant_out = reinterpret_cast<uint8_t*>(dst) + kExpBytes;
    for (uint32_t tr = 0; tr < 2; ++tr) {
        for (uint32_t tc = 0; tc < 2; ++tc) {
            const uint32_t face = tr * 2 + tc;
            for (uint32_t i = 0; i < kFace; ++i) {
                const uint16_t* row = src + (tr * kFace + i) * stride + tc * kFace;
                uint8_t codes[kFace];
                exps[face * kFace + i] = static_cast<uint8_t>(pack_row<MANT_W>(row, codes));
                uint8_t* out = mant_out + (face * kFace + i) * BYTES_PER_ROW;
                if constexpr (MANT_W == 7) {
                    std::memcpy(out, codes, kFace);  // little-endian: value k is byte k of the row's 4 dwords
                } else {
                    for (uint32_t k = 0; k < kFace / 2; ++k) {  // value 2k in the low nibble
                        out[k] = static_cast<uint8_t>(codes[2 * k] | (codes[2 * k + 1] << 4));
                    }
                }
            }
        }
    }
}

constexpr uint32_t tile_dwords(uint32_t bits) { return (kExpBytes + kTile * kTile * bits / 8) / 4; }

constexpr size_t round_up_tile(size_t n) { return (n + kTile - 1) / kTile * kTile; }

// Copy the logical 32x32 tile at (row0, col0) of matrix `m` into `tile` (row-major, stride 32), zero outside the
// logical H x W. With transpose the logical matrix is the input's last two dims swapped: logical(i, j) = in(j, i).
inline size_t src_row_stride(const Geometry& g) { return g.row_stride != 0 ? g.row_stride : g.cols; }

inline size_t src_batch_stride(const Geometry& g) {
    if (g.batch_stride != 0) {
        return g.batch_stride;
    }
    return g.transpose ? g.rows * g.cols : g.rows * src_row_stride(g);
}

inline void gather_tile(const Geometry& g, const uint16_t* m, size_t row0, size_t col0, uint16_t* tile) {
    if (!g.transpose) {
        const size_t stride = src_row_stride(g);
        for (uint32_t i = 0; i < kTile; ++i) {
            const size_t r = row0 + i;
            for (uint32_t j = 0; j < kTile; ++j) {
                const size_t c = col0 + j;
                tile[i * kTile + j] = (r < g.rows && c < g.cols) ? m[r * stride + c] : 0;
            }
        }
        return;
    }
    // Input is cols x rows (in_rows = g.cols, in_cols = g.rows); input row c holds logical column c, so read each
    // input row segment contiguously and scatter it down a column of the tile.
    for (uint32_t j = 0; j < kTile; ++j) {
        const size_t c = col0 + j;
        for (uint32_t i = 0; i < kTile; ++i) {
            const size_t r = row0 + i;
            tile[i * kTile + j] = (r < g.rows && c < g.cols) ? m[c * g.rows + r] : 0;
        }
    }
}

// Work units = (matrix, 32-row band); units [u0, u1).
template <uint32_t MANT_W>
inline void pack_units(const Geometry& g, const uint16_t* src, uint32_t* dst, size_t u0, size_t u1) {
    const size_t bands = round_up_tile(g.rows) / kTile;
    const size_t tiles_per_band = round_up_tile(g.cols) / kTile;
    const size_t tdw = tile_dwords(MANT_W + 1);
    const bool aligned = g.rows % kTile == 0 && g.cols % kTile == 0;
    uint16_t tile[kTile * kTile];
    for (size_t u = u0; u < u1; ++u) {
        const size_t mat = u / bands;
        const size_t band = u % bands;
        const uint16_t* m = src + mat * src_batch_stride(g);
        uint32_t* out = dst + u * tiles_per_band * tdw;
        for (size_t t = 0; t < tiles_per_band; ++t) {
            const size_t row0 = band * kTile;
            const size_t col0 = t * kTile;
            if (!g.transpose && (aligned || (row0 + kTile <= g.rows && col0 + kTile <= g.cols))) {
                pack_tile<MANT_W>(m + row0 * src_row_stride(g) + col0, src_row_stride(g), out + t * tdw);
            } else {
                gather_tile(g, m, row0, col0, tile);
                pack_tile<MANT_W>(tile, kTile, out + t * tdw);
            }
        }
    }
}

__attribute__((target_clones("avx2", "default"))) void pack_units_bfp8(
    const Geometry& g, const uint16_t* src, uint32_t* dst, size_t u0, size_t u1) {
    pack_units<7>(g, src, dst, u0, u1);
}

__attribute__((target_clones("avx2", "default"))) void pack_units_bfp4(
    const Geometry& g, const uint16_t* src, uint32_t* dst, size_t u0, size_t u1) {
    pack_units<3>(g, src, dst, u0, u1);
}

}  // namespace

size_t packed_words(const Geometry& g, int bits) {
    TT_FATAL(bits == 8 || bits == 4, "fast_bfp: bits must be 8 or 4, got {}", bits);
    return g.batch * (round_up_tile(g.rows) / kTile) * (round_up_tile(g.cols) / kTile) * tile_dwords(bits);
}

void pack_bf16(const Geometry& g, const uint16_t* src, int bits, uint32_t* dst, unsigned nthreads) {
    TT_FATAL(bits == 8 || bits == 4, "fast_bfp: bits must be 8 or 4, got {}", bits);
    TT_FATAL(
        !g.transpose || (g.row_stride == 0 && g.batch_stride == 0), "fast_bfp: strides need a non-transposed input");
    if (nthreads == 0) {
        nthreads = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    }
    const size_t units = g.batch * (round_up_tile(g.rows) / kTile);
    auto work = [&](size_t u0, size_t u1) {
        if (bits == 8) {
            pack_units_bfp8(g, src, dst, u0, u1);
        } else {
            pack_units_bfp4(g, src, dst, u0, u1);
        }
    };
    nthreads = static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(nthreads, units)));
    if (nthreads == 1) {
        work(0, units);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned i = 0; i < nthreads; ++i) {
        pool.emplace_back(work, units * i / nthreads, units * (i + 1) / nthreads);
    }
    for (auto& th : pool) {
        th.join();
    }
}

HostBuffer pack_bf16_to_host_buffer(const Geometry& g, const uint16_t* src, int bits) {
    constexpr size_t kHuge = size_t{2} << 20;
    const size_t words = packed_words(g, bits);
    const size_t bytes = std::max<size_t>(kHuge, (words * sizeof(uint32_t) + kHuge - 1) / kHuge * kHuge);
    auto* mem = static_cast<uint32_t*>(std::aligned_alloc(kHuge, bytes));
    TT_FATAL(mem != nullptr, "fast_bfp: allocation of {} bytes failed", bytes);
    if (bytes >= (size_t{8} << 20)) {
        madvise(mem, bytes, MADV_HUGEPAGE);  // first touch happens in the packing threads
    }
    std::shared_ptr<void> owner(mem, [](void* p) { std::free(p); });
    pack_bf16(g, src, bits, mem);
    return HostBuffer(ttsl::Span<uint32_t>(mem, words), MemoryPin(std::move(owner)));
}

std::optional<int> fast_bfp_bits(const TensorSpec& spec) {
    static const bool disabled = [] {
        const char* v = std::getenv("TT_METAL_DISABLE_FAST_BFP_PACK");
        return v != nullptr && v[0] == '1';
    }();
    const DataType dtype = spec.data_type();
    if (disabled || (dtype != DataType::BFLOAT8_B && dtype != DataType::BFLOAT4_B) || spec.layout() != Layout::TILE) {
        return std::nullopt;
    }
    const Tile& tile = spec.tile();
    if (tile.get_tile_shape()[0] != 32 || tile.get_tile_shape()[1] != 32 || tile.get_face_shape()[0] != 16 ||
        tile.get_face_shape()[1] != 16 || tile.get_transpose_within_face() || tile.get_transpose_of_faces()) {
        return std::nullopt;
    }
    const Shape& logical = spec.logical_shape();
    if (logical.rank() < 2) {
        return std::nullopt;
    }
    size_t batch = 1;
    for (size_t d = 0; d + 2 < logical.rank(); ++d) {
        batch *= logical[d];
    }
    const auto round32 = [](size_t n) { return (n + 31) / 32 * 32; };
    const auto physical = spec.physical_shape();
    if (physical.height() != batch * round32(logical[logical.rank() - 2]) ||
        physical.width() != round32(logical[logical.rank() - 1])) {
        return std::nullopt;
    }
    return dtype == DataType::BFLOAT8_B ? 8 : 4;
}

}  // namespace tt::tt_metal::bfp_fast_pack
