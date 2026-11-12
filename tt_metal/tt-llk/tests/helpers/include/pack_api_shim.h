// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The compute API's pack side for tt-llk tests (JIT descriptor arrays, metal CB interface and accessors), so pack_block
// runs production's code: llk_pack_block on the PR's LLK, main's per-tile loop on main's. Needs compile-time formats.

#if defined(ARCH_BLACKHOLE) && defined(LLK_TRISC_PACK)

#include <cstdint>

#include "internal/circular_buffer_interface.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

#if defined(RUNTIME_FORMATS)
#error "pack_api_shim.h needs compile-time formats: the compute API's descriptor arrays are constants"
#endif

// The output CB and its tile geometry: a test sets them before the include where they differ from a 32x32 tile in CB 16.
#ifndef PACK_API_SHIM_OUT
#define PACK_API_SHIM_OUT 16
#endif
#ifndef PACK_API_SHIM_NUM_FACES
#define PACK_API_SHIM_NUM_FACES 4
#endif
#ifndef PACK_API_SHIM_FACE_R_DIM
#define PACK_API_SHIM_FACE_R_DIM 16
#endif
// 1 for an output of partial faces, as the JIT sets pack_partial_face from the CB's tile.
#ifndef PACK_API_SHIM_PARTIAL_FACE
#define PACK_API_SHIM_PARTIAL_FACE 0
#endif

#ifndef DST_SYNC_MODE
#define DST_SYNC_MODE dest_sync
#endif
// The JIT's pack-thread define (genfiles.cpp): some CB of the kernel unpacks to DEST. MATH_FIDELITY comes from the test.
#ifndef LLK_UNPACK_TO_DEST_ANY
#define LLK_UNPACK_TO_DEST_ANY (unpack_to_dest ? 1 : 0)
#endif
#ifndef SAN_HOOK
#define SAN_HOOK(...)
#endif

// The JIT's arrays (genfiles.cpp): 255 for an unused CB, the output's values at PACK_API_SHIM_OUT, CB 0 and 1 the inputs.
#define PACK_API_SHIM_ROW(f)                                                                                                                               \
    f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10), f(11), f(12), f(13), f(14), f(15), f(16), f(17), f(18), f(19), f(20), f(21), f(22), \
        f(23), f(24), f(25), f(26), f(27), f(28), f(29), f(30), f(31), f(32), f(33), f(34), f(35), f(36), f(37), f(38), f(39), f(40), f(41), f(42), f(43), \
        f(44), f(45), f(46), f(47), f(48), f(49), f(50), f(51), f(52), f(53), f(54), f(55), f(56), f(57), f(58), f(59), f(60), f(61), f(62), f(63)
#define PACK_API_SHIM_OUT_OR(i, out, other) ((i) == PACK_API_SHIM_OUT ? (out) : (other))
#define PACK_API_SHIM_SRC(i)                PACK_API_SHIM_OUT_OR(i, PACK_IN, 255)
#define PACK_API_SHIM_DST(i)                PACK_API_SHIM_OUT_OR(i, PACK_OUT, 255)
#define PACK_API_SHIM_UNPACK(i)             ((i) == 0 ? UNPACK_A_IN : ((i) == 1 ? UNPACK_B_IN : PACK_API_SHIM_OUT_OR(i, PACK_OUT, 255)))
#define PACK_API_SHIM_FACES(i)              PACK_API_SHIM_OUT_OR(i, PACK_API_SHIM_NUM_FACES, 4)
#define PACK_API_SHIM_FACE_R(i)             PACK_API_SHIM_OUT_OR(i, PACK_API_SHIM_FACE_R_DIM, 16)
#define PACK_API_SHIM_ZERO(i)               0
#define PACK_API_SHIM_PARTIAL(i)            PACK_API_SHIM_OUT_OR(i, PACK_API_SHIM_PARTIAL_FACE, 0)
#define PACK_API_SHIM_TILE_R(i)             PACK_API_SHIM_OUT_OR(i, PACK_API_SHIM_FACE_R_DIM*(PACK_API_SHIM_NUM_FACES > 2 ? 2 : 1), 32)
#define PACK_API_SHIM_TILE_C(i)             PACK_API_SHIM_OUT_OR(i, PACK_API_SHIM_NUM_FACES == 1 ? 16 : 32, 32)
#define PACK_API_SHIM_FACES_R(i)            PACK_API_SHIM_OUT_OR(i, PACK_API_SHIM_NUM_FACES > 2 ? 2 : 1, 2)
#define PACK_API_SHIM_FACES_C(i)            PACK_API_SHIM_OUT_OR(i, PACK_API_SHIM_NUM_FACES == 1 ? 1 : 2, 2)

constexpr unsigned char pack_src_format[64]     = {PACK_API_SHIM_ROW(PACK_API_SHIM_SRC)};
constexpr unsigned char pack_dst_format[64]     = {PACK_API_SHIM_ROW(PACK_API_SHIM_DST)};
constexpr std::uint8_t pack_tile_num_faces[64]  = {PACK_API_SHIM_ROW(PACK_API_SHIM_FACES)};
constexpr std::uint8_t pack_partial_face[64]    = {PACK_API_SHIM_ROW(PACK_API_SHIM_PARTIAL)};
constexpr std::uint8_t pack_tile_face_r_dim[64] = {PACK_API_SHIM_ROW(PACK_API_SHIM_FACE_R)};
constexpr std::uint8_t pack_narrow_tile[64]     = {PACK_API_SHIM_ROW(PACK_API_SHIM_ZERO)};
constexpr std::uint8_t pack_tile_r_dim[64]      = {PACK_API_SHIM_ROW(PACK_API_SHIM_TILE_R)};
constexpr std::uint8_t pack_tile_c_dim[64]      = {PACK_API_SHIM_ROW(PACK_API_SHIM_TILE_C)};
constexpr std::uint8_t pack_num_faces_r_dim[64] = {PACK_API_SHIM_ROW(PACK_API_SHIM_FACES_R)};
constexpr std::uint8_t pack_num_faces_c_dim[64] = {PACK_API_SHIM_ROW(PACK_API_SHIM_FACES_C)};
constexpr std::uint8_t unpack_src_format[64]    = {PACK_API_SHIM_ROW(PACK_API_SHIM_UNPACK)};

// The firmware's cb_interface, sized to the CBs up to the output: the metal header's 64 entries would take 2 KB of the
// kernel's local data, which production leaves to the firmware.
extern "C" CBInterface pack_api_shim_cb_interface[PACK_API_SHIM_OUT + 1];
__attribute__((used)) CBInterface pack_api_shim_cb_interface[PACK_API_SHIM_OUT + 1];
asm(".globl cb_interface\n.set cb_interface, pack_api_shim_cb_interface");

#include "../llk_io/llk_outputs.h"

// The metal API's address of the next tile of an output (llk_pack_common_api.h).
template <bool out_of_order_output, PackMode pack_addr_mode = PackMode::Default>
inline std::uint32_t get_output_tile_address(std::uint8_t output_id, std::uint32_t output_tile_index)
{
    static_assert(
        pack_addr_mode == PackMode::Default || pack_addr_mode == PackMode::Untilize,
        "Pack tile address helper supports PackMode::Default and PackMode::Untilize only");
    std::uint32_t pack_tile_addr;
    if constexpr (out_of_order_output)
    {
        pack_tile_addr =
            get_local_cb_interface(output_id).fifo_wr_ptr + (std::uint32_t)(get_local_cb_interface(output_id).fifo_page_size) * output_tile_index - 1;
    }
    else
    {
        if constexpr (pack_addr_mode == PackMode::Untilize)
        {
            static_assert(pack_addr_mode != PackMode::Untilize, "Use llk_pack_untilize APIs for pack-untilize.");
        }
        else
        {
            pack_tile_addr = get_local_cb_interface(output_id).fifo_wr_ptr + get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1;
            get_local_cb_interface(output_id).fifo_wr_tile_ptr += get_local_cb_interface(output_id).fifo_page_size;
        }
    }
    return pack_tile_addr;
}

// The kernel's compute API traits (pack_math_context.h): a matmul API header, a pack-thread SFPU init header.
#if defined(PACK_API_SHIM_MATMUL)
namespace ckernel::packer
{
template <>
struct llk_pack_matmul_api<1>
{
    static constexpr bool value = true;
};
} // namespace ckernel::packer
#endif
#if defined(PACK_API_SHIM_SFPU)
namespace ckernel::packer
{
template <>
struct llk_pack_sfpu_api<1>
{
    static constexpr bool value = true;
};
} // namespace ckernel::packer
#endif

#if __has_include("llk_pack_block.h")
#define PACK_API_SHIM_PR 1
#include "llk_pack_block.h"
#else
#define PACK_API_SHIM_PR 0
#endif

namespace pack_api_shim
{

// A CB of num_pages pages of page_words over L1 from wr_ptr, empty, as the firmware sets one up; in 16 byte words, as
// the metal CB pointers.
inline void cb_bind(const std::uint32_t cb, const std::uint32_t wr_ptr, const std::uint32_t page_words, const std::uint32_t num_pages)
{
    LocalCBInterface& c = get_local_cb_interface(cb);
    c.fifo_page_size    = page_words;
    c.fifo_num_pages    = num_pages;
    c.fifo_size         = page_words * num_pages;
    c.fifo_wr_ptr       = wr_ptr;
    c.fifo_rd_ptr       = wr_ptr;
    c.fifo_limit        = wr_ptr + c.fifo_size;
    c.fifo_wr_tile_ptr  = 0;
}

// cb_push_back's pointer moves on the pack thread (llk_push_tiles), without the consumer's notification.
inline void cb_push_back(const std::uint32_t cb, const std::uint32_t num_tiles)
{
    LocalCBInterface& c           = get_local_cb_interface(cb);
    const std::uint32_t num_words = num_tiles * c.fifo_page_size;
    c.fifo_wr_ptr += num_words;
    c.fifo_wr_tile_ptr = 0;
    if (c.fifo_wr_ptr >= c.fifo_limit)
    {
        c.fifo_wr_ptr -= c.fifo_size;
    }
    // A compute kernel's block loop holds opaque code (CB waits, noinline reconfigures), so the pack state lives in memory.
    asm volatile("" ::: "memory");
}

// Main's pack_block: llk_matmul_pack's per-tile loop. On the PR's LLK the per-tile pack keeps main's destination write.
template <bool is_fp32_dest_acc_en>
inline void main_pack_block(const std::uint32_t start_tile_index, const std::uint32_t output, const std::uint32_t ntiles)
{
    const std::uint8_t output_id = get_output_id(output);
    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++)
    {
        const std::uint32_t pack_tile_addr = get_output_tile_address<false, PackMode::Default>(output_id, 0);
#if PACK_API_SHIM_PR
        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default, false, false>(tile_index, pack_tile_addr);
#else
        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default>(tile_index, pack_tile_addr);
#endif
    }
}

#if PACK_API_SHIM_PR
// The PR's pack_block<ntiles, acc, kt>: the form tables, or arm_form ('T', 'B', 'W') as a constant output that always takes it.
template <std::uint32_t ntiles, bool is_fp32_dest_acc_en, std::uint32_t matmul_kt = 0, char arm_form = 0>
__attribute__((always_inline)) inline void pack_block(const std::uint32_t start_tile_index, const std::uint32_t output)
{
    llk_pack_block<ntiles, is_fp32_dest_acc_en, false, matmul_kt, arm_form>(start_tile_index, output);
}
#endif

} // namespace pack_api_shim

#endif
