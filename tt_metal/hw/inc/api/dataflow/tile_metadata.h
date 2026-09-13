// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr inline std::int32_t get_tile_size(const std::int32_t operand) {
    std::uint32_t input = operand;

    // L1 16B words
    std::uint32_t num_words = (uint)unpack_tile_size[input];

    // return bytes
    return num_words;
}

constexpr inline uint32_t get_tile_hw(const std::int32_t operand) {
    std::uint32_t input = operand;
    return (uint32_t)unpack_tile_r_dim[input] * (uint32_t)unpack_tile_c_dim[input];
}

constexpr inline uint32_t get_tile_num_faces(const std::int32_t operand) {
    std::uint32_t input = operand;
    return (uint32_t)unpack_tile_num_faces[input];
}

constexpr inline DataFormat get_dataformat(const std::int32_t operand) {
    return static_cast<DataFormat>((uint)unpack_src_format[operand]);
}
