// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "cce_test_status_abi.h"

// cce_addrgen kernel ABI. The kernel (cce_addrgen.cpp) and the host (dram/test_mimir_dm_addrgen.cpp) both include it.
// Args, modes, patterns and codes are append-only: a later stage adds fields after the last one and never renumbers.
namespace cce_addrgen_abi {

using cce_test_status_abi::kLineBytes;

// Runtime args. kArgStatusOffset and kArgResultOffset are 64 B aligned offsets in the kernel's own TL1, on separate
// lines. kArgSrcStrideEnable / kArgDestStrideEnable select whether that side programs its loops (upstream's
// src_stride_en / dst_stride_en); a side without loops stays at its base.
inline constexpr uint32_t kArgMode = 0, kArgStatusOffset = 1, kArgResultOffset = 2, kArgPattern = 3,
                          kArgSrcStrideEnable = 4, kArgDestStrideEnable = 5, kArgNumAddresses = 6, kNumArgs = 7;

// kModePattern: program address generator 0 with kPatterns[kArgPattern], pop kArgNumAddresses source and destination
// addresses and store each pair at kArgResultOffset through the uncached alias.
inline constexpr uint32_t kModePattern = 0;

// Result entry i: source address, destination address, as 64-bit values.
inline constexpr uint32_t kResultSrc = 0, kResultDest = 1, kResultEntryWords64 = 2;
inline constexpr uint32_t kMaxAddresses = 256;

struct LoopSpec {
    uint64_t stride;
    uint64_t end;
};

struct SideSpec {
    uint64_t base;
    LoopSpec inner;
    LoopSpec outer;
    uint64_t face_size;
};

// Software patch loop (upstream im2col): before each group of patch_pops addresses the kernel rewrites the source base
// to src.base + patch_row * patch_row_stride + patch_column * patch_column_stride, patch_row < patch_row_count,
// patch_column < patch_column_count (patch_column innermost).
struct PatternSpec {
    SideSpec src;
    SideSpec dest;
    uint32_t patch_pops;
    uint32_t patch_row_count;
    uint64_t patch_row_stride;
    uint32_t patch_column_count;
    uint64_t patch_column_stride;
};

// The upstream quasar_addrgen / quasar_im2col example kernels' configurations.
inline constexpr uint32_t kPattern1d = 0, kPattern2d = 1, kPatternFace = 2, kPatternIm2col = 3,
                          kPatternIm2colDilation1 = 4, kNumPatterns = 5;

// addrgen_im2col_example.cpp: H 6, W 7, k 3, dilation (1, 2), stride 1, 8 B elements.
inline constexpr uint64_t kIm2colElem = 8, kIm2colRowBytes = 7 * kIm2colElem;
// addrgen_im2col_dilation1_example.cpp: H 4, W 5, k 3x3, 128 B elements.
inline constexpr uint64_t kIm2colD1Elem = 128, kIm2colD1RowBytes = 5 * kIm2colD1Elem;

inline constexpr PatternSpec kPatterns[kNumPatterns] = {
    // addrgen_1d_example.cpp, 10 addresses.
    {.src = {.base = 0x10000, .inner = {4096, 10 * 4096}, .outer = {0, 0}, .face_size = 0},
     .dest = {.base = 0x20000, .inner = {4096, 10 * 4096}, .outer = {0, 0}, .face_size = 0}},
    // addrgen_2d_example.cpp: 4 columns 128 B apart, 4 rows 1024 B apart.
    {.src = {.base = 0x30000, .inner = {128, 4 * 128}, .outer = {1024, 4 * 1024}, .face_size = 0},
     .dest = {.base = 0x40000, .inner = {128, 4 * 128}, .outer = {1024, 4 * 1024}, .face_size = 0}},
    // addrgen_face_example.cpp: the 2D pattern, base advanced by one 4096 B face after each outer loop.
    {.src = {.base = 0x10000, .inner = {128, 4 * 128}, .outer = {1024, 4 * 1024}, .face_size = 4 * 1024},
     .dest = {.base = 0x20000, .inner = {128, 4 * 128}, .outer = {1024, 4 * 1024}, .face_size = 4 * 1024}},
    // addrgen_im2col_example.cpp: one 3x3 patch per software step, 3 x 3 patch positions, 81 addresses.
    {.src =
         {.base = 0x10000,
          .inner = {kIm2colElem * 2, 3 * 2 * kIm2colElem},
          .outer = {kIm2colRowBytes, 3 * kIm2colRowBytes},
          .face_size = 0},
     .dest = {.base = 0x20000, .inner = {kIm2colElem, 81 * kIm2colElem}, .outer = {0, 0}, .face_size = 0},
     .patch_pops = 9,
     .patch_row_count = 3,
     .patch_row_stride = kIm2colRowBytes,
     .patch_column_count = 3,
     .patch_column_stride = kIm2colElem},
    // addrgen_im2col_dilation1_example.cpp: kernel rows inner, kernel columns outer, face = one image row; 180
    // addresses. The loop ends are one past the last value, not a multiple of the stride.
    {.src =
         {.base = 0x10000,
          .inner = {kIm2colD1RowBytes, kIm2colD1RowBytes * 2 + 1},
          .outer = {kIm2colD1Elem, kIm2colD1RowBytes - 3 * kIm2colD1Elem + 1},
          .face_size = kIm2colD1RowBytes},
     .dest = {.base = 0x20000, .inner = {kIm2colD1Elem, 180 * kIm2colD1Elem}, .outer = {0, 0}, .face_size = 0}},
};

// Status line, written once at exit: every field, `fence w, w`, then the magic. kStatusPopCycles is the rdcycle count
// of the pops.
using cce_test_status_abi::kStatusCode;
using cce_test_status_abi::kStatusMagic;
inline constexpr uint32_t kStatusPopCycles = 2, kStatusWords = 3;
static_assert(kStatusWords * sizeof(uint32_t) <= kLineBytes);
inline constexpr uint32_t kDoneMagic = 0xADD6E0DEu;
using cce_test_status_abi::kStatusPoison;

inline constexpr uint32_t kCodeDone = 1, kCodeBadArgs = 2;

}  // namespace cce_addrgen_abi
