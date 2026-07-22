// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/float8.hpp>

namespace tt::test_utils {

namespace detail {
// Element-wise static_cast of any vector of float-convertible values into a
// flat std::vector<float>. Shared body of fp8_to_floats / bf16_to_floats so
// the two public helpers differ only by which unpacker they invoke.
template <typename T>
inline std::vector<float> to_floats(const std::vector<T>& vec) {
    std::vector<float> floats;
    floats.reserve(vec.size());
    for (const auto& v : vec) {
        floats.push_back(static_cast<float>(v));
    }
    return floats;
}
}  // namespace detail

// Unpack a packed uint32 vector (4 fp8 bytes per word) into a flat float vector.
inline std::vector<float> fp8_to_floats(const std::vector<uint32_t>& packed) {
    return detail::to_floats(unpack_uint32_vec_into_float8_e4m3_vec(packed));
}

// Unpack a packed uint32 vector (2 bf16 per word) into a flat float vector.
inline std::vector<float> bf16_to_floats(const std::vector<uint32_t>& packed) {
    return detail::to_floats(unpack_uint32_vec_into_bfloat16_vec(packed));
}

// Linear byte index within a TILE_HEIGHT x TILE_WIDTH tile of 1-byte elements
// stored as 4 FACE_HEIGHT x FACE_WIDTH faces in (TL, TR, BL, BR) order,
// addressed by logical (col, row). Applies to any 8-bit tile format
// (Fp8_e4m3, Lf8, Int8, UInt8).
inline int byte_tile_face_major_index(int col, int row) {
    constexpr int kFaceH = static_cast<int>(tt::constants::FACE_HEIGHT);
    constexpr int kFaceW = static_cast<int>(tt::constants::FACE_WIDTH);
    constexpr int kFaceHW = kFaceH * kFaceW;     // bytes per face (TL, TR, or BL)
    constexpr int kTopRowFacesHW = 2 * kFaceHW;  // TL + TR — bytes per top tile-row
    const int offset = ((col < kFaceW) ? 0 : kFaceHW) + ((row < kFaceH) ? 0 : kTopRowFacesHW);
    return offset + ((row % kFaceH) * kFaceW) + (col % kFaceW);
}

}  // namespace tt::test_utils
