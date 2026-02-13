// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "internal/risc_attribs.h"
#include "internal/firmware_common.h"
#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"
#include "tt_metal/hw/firmware/src/trisc.cc"

// Those values come from generated chlkc_descriptors.h, so hardcode values here for fake JIT builds

// chlkc_descriptors.h (scalars)
constexpr bool DST_ACCUM_MODE = false;
#define DST_SYNC_MODE DstSync::SyncHalf
constexpr bool APPROX = true;
constexpr std::int32_t MATH_FIDELITY = 255;

// chlkc_descriptors.h (pack data format arrays)
constexpr unsigned char pack_src_format[32] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
constexpr unsigned char pack_dst_format[32] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

// chlkc_descriptors.h (pack tile dim arrays)
constexpr uint8_t pack_tile_num_faces[32] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
};
constexpr uint8_t pack_partial_face[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
constexpr uint8_t pack_tile_face_r_dim[32] = {
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
};
constexpr uint8_t pack_narrow_tile[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
constexpr uint8_t pack_tile_r_dim[32] = {
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
};
constexpr uint8_t pack_tile_c_dim[32] = {
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
};
constexpr uint16_t pack_tile_size[32] = {
    1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088,
    1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088,
};

// chlkc_descriptors.h (unpack data format arrays)
constexpr std::int32_t unpack_src_format[32] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
constexpr std::int32_t unpack_dst_format[32] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

// chlkc_descriptors.h (unpack tile dim arrays)
constexpr uint8_t unpack_tile_num_faces[32] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
};
constexpr uint8_t unpack_partial_face[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
constexpr uint8_t unpack_tile_face_r_dim[32] = {
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
};
constexpr uint8_t unpack_narrow_tile[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
constexpr uint8_t unpack_tile_r_dim[32] = {
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
};
constexpr uint8_t unpack_tile_c_dim[32] = {
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
};
constexpr uint16_t unpack_tile_size[32] = {
    1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088,
    1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088,
};
