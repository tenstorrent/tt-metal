// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Logical CB limits stay inside the allocated L1 inputs; failed bounds checks are recorded, not trapped.

#ifdef LLK_TRISC_UNPACK

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Minimal Metal operand metadata for two full BF16 input CBs.
constexpr std::uint8_t unpack_src_format[32]      = {10, 10};
constexpr std::uint8_t unpack_dst_format[32]      = {10, 10};
constexpr std::uint8_t unpack_tile_num_faces[32]  = {4, 4};
constexpr std::uint8_t unpack_partial_face[32]    = {};
constexpr std::uint8_t unpack_tile_face_r_dim[32] = {16, 16};
constexpr std::uint8_t unpack_narrow_tile[32]     = {};
constexpr std::uint8_t unpack_tile_r_dim[32]      = {32, 32};
constexpr std::uint8_t unpack_tile_c_dim[32]      = {32, 32};
constexpr std::uint8_t unpack_num_faces_r_dim[32] = {2, 2};
constexpr std::uint8_t unpack_num_faces_c_dim[32] = {2, 2};
#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

// Override only this wrapper's assertions; its dependencies retain normal asserts.
std::uint32_t bounds_failures = 0;
#pragma push_macro("LLK_ASSERT")
#undef LLK_ASSERT
#define LLK_ASSERT(condition, ...) bounds_failures += !(condition)
#include "experimental/llk_unpack_AB_sub_bcast_col_custom_api.h"
#pragma pop_macro("LLK_ASSERT")
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];
#include "llk_unpack_common.h"
#include "params.h"
#include "tensor_shape.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        params.TEST_FACE_R_DIM,
        params.TEST_FACE_R_DIM,
        params.num_faces,
        params.num_faces);
    // CB addresses use 16-byte units, while params.buffer_A/B contain byte addresses.
    auto& a          = get_local_cb_interface(0);
    auto& b          = get_local_cb_interface(1);
    a.fifo_rd_ptr    = params.buffer_A[0] >> 4;
    b.fifo_rd_ptr    = params.buffer_B[0] >> 4;
    a.fifo_page_size = b.fifo_page_size = 2048 >> 4;
    a.fifo_limit                        = a.fifo_rd_ptr + BOUNDS_A_TILES * a.fifo_page_size;
    b.fifo_limit                        = b.fifo_rd_ptr + BOUNDS_B_TILES * b.fifo_page_size;
    bounds_failures                     = 0;
    llk_unpack_AB_sub_bcast_col_init_custom(0, 1);
    llk_unpack_AB_sub_bcast_col_custom(0, 1, BOUNDS_A_START, BOUNDS_B_START, params.NUM_TILES_IN_BLOCK);
    // The fourth allocated SrcA tile is never read; use its first word for the verdict.
    *reinterpret_cast<volatile std::uint32_t*>(params.buffer_A[0] + 3 * 2048) = bounds_failures;
}

#else

// Reuse the existing custom-broadcast math and pack drivers. Only the unpack
// side needs CB metadata to exercise the Metal wrapper's bounds assertions.
#include "multiple_tiles_eltwise_custom_test.cpp"

#endif
