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
// An optional second output of the same tile geometry and its destination format, from the same DEST format.
#ifndef PACK_API_SHIM_OUT2
#define PACK_API_SHIM_OUT2 64
#endif
#ifndef PACK_API_SHIM_OUT2_FORMAT
#define PACK_API_SHIM_OUT2_FORMAT PACK_OUT
#endif
// Output B's packer input: the JIT's per-CB pack_src_format, which follows B's format (fp32 DEST); the tests pass it.
#ifndef PACK_API_SHIM_OUT2_SRC
#define PACK_API_SHIM_OUT2_SRC PACK_IN
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

// The JIT's arrays (genfiles.cpp): 255 for an unused CB, the outputs' values at PACK_API_SHIM_OUT(2), CB 0 and 1 the inputs.
#define PACK_API_SHIM_ROW(f)                                                                                                                               \
    f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10), f(11), f(12), f(13), f(14), f(15), f(16), f(17), f(18), f(19), f(20), f(21), f(22), \
        f(23), f(24), f(25), f(26), f(27), f(28), f(29), f(30), f(31), f(32), f(33), f(34), f(35), f(36), f(37), f(38), f(39), f(40), f(41), f(42), f(43), \
        f(44), f(45), f(46), f(47), f(48), f(49), f(50), f(51), f(52), f(53), f(54), f(55), f(56), f(57), f(58), f(59), f(60), f(61), f(62), f(63)
#define PACK_API_SHIM_OUT_OR(i, out, other) ((i) == PACK_API_SHIM_OUT || (i) == PACK_API_SHIM_OUT2 ? (out) : (other))
#define PACK_API_SHIM_OUT_FMT(i)            ((i) == PACK_API_SHIM_OUT2 ? PACK_API_SHIM_OUT2_FORMAT : PACK_OUT)
#define PACK_API_SHIM_SRC(i)                ((i) == PACK_API_SHIM_OUT2 ? PACK_API_SHIM_OUT2_SRC : PACK_API_SHIM_OUT_OR(i, PACK_IN, 255))
#define PACK_API_SHIM_DST(i)                PACK_API_SHIM_OUT_OR(i, PACK_API_SHIM_OUT_FMT(i), 255)
#define PACK_API_SHIM_UNPACK(i)             ((i) == 0 ? UNPACK_A_IN : ((i) == 1 ? UNPACK_B_IN : PACK_API_SHIM_OUT_OR(i, PACK_API_SHIM_OUT_FMT(i), 255)))
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

// The firmware's cb_interface, sized to the CBs up to the last output: the metal header's 64 entries would take 2 KB of the
// kernel's local data, which production leaves to the firmware.
#define PACK_API_SHIM_LAST_CB (PACK_API_SHIM_OUT2 < 64 && PACK_API_SHIM_OUT2 > PACK_API_SHIM_OUT ? PACK_API_SHIM_OUT2 : PACK_API_SHIM_OUT)
extern "C" CBInterface pack_api_shim_cb_interface[PACK_API_SHIM_LAST_CB + 1];
__attribute__((used)) CBInterface pack_api_shim_cb_interface[PACK_API_SHIM_LAST_CB + 1];
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

#if __has_include("llk_pack_block.h")
#define PACK_API_SHIM_PR 1
#else
#define PACK_API_SHIM_PR 0
#endif

// The kernel's compute API traits (pack_math_context.h): a matmul API header, a pack-thread SFPU init header. Main's LLK has
// no pack contexts, so on main they set nothing.
#if PACK_API_SHIM_PR && defined(PACK_API_SHIM_MATMUL)
namespace ckernel::packer
{
template <>
struct llk_pack_matmul_api<1>
{
    static constexpr bool value = true;
};

template <>
struct llk_pack_lofi_api<1>
{
    static constexpr bool value = MATH_FIDELITY == MathFidelity::LoFi;
};
} // namespace ckernel::packer
#endif
#if PACK_API_SHIM_PR && defined(PACK_API_SHIM_SFPU)
namespace ckernel::packer
{
template <>
struct llk_pack_sfpu_api<1>
{
    static constexpr bool value = true;
};
} // namespace ckernel::packer
#endif

#if PACK_API_SHIM_PR
#include "llk_pack_block.h"
#endif

namespace pack_api_shim
{

#ifndef PACK_API_SHIM_CB_COUNTS
#define PACK_API_SHIM_CB_COUNTS

// metal's get_cb_tiles_received_ptr / get_cb_tiles_acked_ptr (stream_io_map.h): CB n's counts in overlay stream n's
// REMOTE_DEST_BUF_SIZE (10) and REMOTE_DEST_BUF_START (8) registers; the harness keeps noc_overlay_parameters.h out.
inline volatile std::uint32_t* cb_tiles_received_ptr(const std::uint32_t cb)
{
    return reinterpret_cast<volatile std::uint32_t*>(0xFFB40000 + cb * 0x1000 + 10 * 4);
}

inline volatile std::uint32_t* cb_tiles_acked_ptr(const std::uint32_t cb)
{
    return reinterpret_cast<volatile std::uint32_t*>(0xFFB40000 + cb * 0x1000 + 8 * 4);
}
#endif

// A CB of num_pages pages of page_words over L1 from wr_ptr, empty, as the firmware sets one up; in 16 byte words, as
// the metal CB pointers. notify: the CB's counts as metal keeps them (llk_io), in the CB's stream scratch registers,
// which the producer clears; its consumer waits no earlier than the next zone boundary, a four-thread rendezvous.
template <bool notify = false>
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
    if constexpr (notify)
    {
        c.tiles_acked_received_init = 0;
        *cb_tiles_received_ptr(cb)  = 0;
        *cb_tiles_acked_ptr(cb)     = 0;
    }
}

// The isolate run types have no peer thread to wait on, so their waits are dropped and the pointer moves stay.
#if defined(LLK_PROFILER)
inline constexpr bool cb_waits = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;
#else
inline constexpr bool cb_waits = true;
#endif

// cb_reserve_back: llk_wait_for_free_tiles, with notify.
template <bool notify = false>
inline void cb_reserve_back(const std::uint32_t cb, const std::uint32_t num_tiles)
{
    if constexpr (notify && cb_waits)
    {
        volatile std::uint32_t* tiles_acked_ptr = cb_tiles_acked_ptr(cb);
        const std::uint16_t tiles_received      = get_local_cb_interface(cb).tiles_received;
        std::int32_t free_tiles;
        do
        {
            const std::uint16_t tiles_acked     = (std::uint16_t)ckernel::reg_read((std::uint32_t)tiles_acked_ptr);
            const std::uint16_t free_tiles_wrap = get_local_cb_interface(cb).fifo_num_pages - (tiles_received - tiles_acked);
            free_tiles                          = (std::int32_t)free_tiles_wrap;
        } while (free_tiles < (std::int32_t)num_tiles);
    }
}

// cb_push_back's pointer moves on the pack thread (llk_push_tiles); with notify also its consumer notification
// (llk_push_to_brisc): the count stored once the packer is done.
template <bool notify = false>
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
    if constexpr (notify)
    {
        const std::uint32_t tiles_received_addr_tensix =
            static_cast<std::uint32_t>((reinterpret_cast<std::uintptr_t>(cb_tiles_received_ptr(cb)) >> 2) & 0x3ffff);
        c.tiles_received += num_tiles;
        const std::uint16_t tiles_received_new = c.tiles_received;
        TT_SETDMAREG(0, tiles_received_new, 0, LO_16(ckernel::p_gpr_pack::NUM_MSGS_RECEIVED));
        TTI_STALLWAIT(ckernel::p_stall::STALL_THCON, ckernel::p_stall::PACK);
        TT_STOREREG(ckernel::p_gpr_pack::NUM_MSGS_RECEIVED, tiles_received_addr_tensix);
    }
    // A compute kernel's block loop holds opaque code (CB waits, noinline reconfigures), so the pack state lives in memory.
    asm volatile("" ::: "memory");
}

// Main's pack_block: llk_matmul_pack's per-tile loop, with main's destination write on the PR's LLK too.
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

// Main's llk_pack_init (llk_pack_tile_api.h, both overloads; its llk_pack_init_impl is _llk_pack_init_ alone).
template <PackMode pack_mode = PackMode::Default, bool zero_output = false, bool skip_addrmod_config = false, bool skip_packer_strides = false>
inline void main_pack_init(const std::uint32_t pack_output, const std::uint32_t num_tiles = 1, [[maybe_unused]] const std::uint32_t input_operand = 0)
{
    const std::uint32_t output_id = get_output_id(pack_output);
    if constexpr (!skip_addrmod_config)
    {
        LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    }
    bool is_input_8bit_format = false;
    if constexpr (pack_mode == PackMode::Tilize)
    {
        is_input_8bit_format = IS_8BIT_FORMAT(static_cast<std::uint32_t>(unpack_src_format[input_operand]));
    }
    _llk_pack_init_<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_src_format[output_id],
        get_output_face_r_dim(output_id),
        get_output_tile_c_dim(output_id),
        get_output_num_faces(output_id),
        num_tiles,
        is_input_8bit_format);
}

// The compute API's llk_pack_init overloads (llk_pack_tile_api.h): on the PR's LLK, llk_pack_init_of_output (llk_pack_block_tile.h);
// counted: an overload that passes a tile count.
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides, bool counted>
inline void pack_init_body(const std::uint32_t pack_output, const std::uint32_t num_tiles, [[maybe_unused]] const std::uint32_t input_operand)
{
#if PACK_API_SHIM_PR
    const std::uint32_t output_id = get_output_id(pack_output);
    if constexpr (!skip_addrmod_config)
    {
        LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    }
    bool is_input_8bit_format = false;
    if constexpr (pack_mode == PackMode::Tilize)
    {
        is_input_8bit_format = IS_8BIT_FORMAT(static_cast<std::uint32_t>(unpack_src_format[input_operand]));
    }
    llk_pack_init_of_output<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides, counted, is_fp32_dest_acc_en>(
        output_id, num_tiles, is_input_8bit_format);
#else
    main_pack_init<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(pack_output, num_tiles, input_operand);
#endif
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false, bool skip_addrmod_config = false, bool skip_packer_strides = false>
inline void pack_init(const std::uint32_t pack_output, const std::uint32_t num_tiles, const std::uint32_t input_operand)
{
    pack_init_body<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides, true>(pack_output, num_tiles, input_operand);
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false, bool skip_addrmod_config = false, bool skip_packer_strides = false>
inline void pack_init(const std::uint32_t pack_output, const std::uint32_t num_tiles)
{
    pack_init_body<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides, true>(pack_output, num_tiles, 0);
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false, bool skip_addrmod_config = false, bool skip_packer_strides = false>
inline void pack_init(const std::uint32_t pack_output)
{
    pack_init_body<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides, false>(pack_output, 1, 0);
}

#if PACK_API_SHIM_PR
// The compute API's llk_pack_hw_configure (llk_pack_common_api.h) on the PR's LLK, the output's page in 16 byte words.
template <bool is_fp32_dest_acc_en>
inline void pack_hw_configure(const std::uint32_t pack_output, const std::uint32_t tile_size)
{
    const std::uint32_t output_id  = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const std::uint32_t num_faces  = get_output_num_faces(output_id);
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        pack_src_format[output_id], pack_dst_format[output_id], tile_size, face_r_dim, tile_c_dim, num_faces, get_output_partial_face(output_id), 0);
    llk_pack_hw_configure_live<is_fp32_dest_acc_en>(face_r_dim, tile_c_dim, num_faces);
}
#endif

// The compute API's llk_pack_reconfig_data_format(new_output) (llk_pack_common_api.h): on the PR's LLK with its skip mask.
template <bool is_fp32_dest_acc_en, bool gate = true, bool inl = false>
inline void pack_reconfig_data_format_body(const std::uint32_t new_output)
{
    const std::uint32_t output_id  = get_output_id(new_output);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const std::uint32_t num_faces  = get_output_num_faces(output_id);
    LLK_ASSERT_FP32_DEST_ACC_EN_MATCHES(is_fp32_dest_acc_en);
#if PACK_API_SHIM_PR
    if constexpr (!gate)
    {
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::UNPACK0);
    }
    constexpr std::uint32_t skip = ckernel::packer::llk_pack_reconfig_skip_all<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
        pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face);
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, skip, inl>(
#else
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
#endif
        pack_src_format[output_id],
        pack_dst_format[output_id],
        get_local_cb_interface(output_id).fifo_page_size,
        tile_c_dim,
        num_faces,
        false /* partial_face */);
}

template <bool is_fp32_dest_acc_en>
inline void pack_reconfig_data_format(const std::uint32_t new_output)
{
#if PACK_API_SHIM_PR
    if constexpr (ckernel::packer::llk_pack_reconfig_u2d_half<is_fp32_dest_acc_en, DST_SYNC_MODE>())
    {
        constexpr int kind = ckernel::packer::llk_pack_out32_kind(pack_dst_format);
        if constexpr (kind == 0)
        {
            pack_reconfig_data_format_body<is_fp32_dest_acc_en, false>(new_output);
            return;
        }
        else if constexpr (kind == 2)
        {
            if (!ckernel::packer::llk_pack_out32(pack_dst_format[get_output_id(new_output)]))
            {
                pack_reconfig_data_format_body<is_fp32_dest_acc_en, false>(new_output);
                return;
            }
        }
        if (ckernel::packer::pack_short_block)
        {
            pack_reconfig_data_format_body<is_fp32_dest_acc_en, false, true>(new_output);
            return;
        }
    }
#endif
    pack_reconfig_data_format_body<is_fp32_dest_acc_en, true>(new_output);
}

#if PACK_API_SHIM_PR
// The compute API's pack_reconfig_data_format(cb) with a constant cb (llk_pack_reconfig_data_format_ct): the immediate reconfigure.
template <bool is_fp32_dest_acc_en, std::uint32_t new_output, bool gate, bool inl = false>
__attribute__((always_inline)) inline void pack_reconfig_data_format_gate()
{
    constexpr ckernel::packer::llk_pack_reconfig_imm_t imm =
        ckernel::packer::llk_pack_reconfig_imm<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
            pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face);
    constexpr bool skip_all = (ckernel::packer::llk_pack_reconfig_skip_all<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
                                   pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face) &
                               ckernel::packer::LLK_PACK_RECONFIG_SKIP_ALL) != 0;
    if constexpr (
        !imm.ok || skip_all || new_output >= sizeof(pack_dst_format) || pack_src_format[new_output] == static_cast<unsigned char>(DataFormat::Invalid) ||
        pack_dst_format[new_output] == static_cast<unsigned char>(DataFormat::Invalid))
    {
        pack_reconfig_data_format_body<is_fp32_dest_acc_en, gate, inl>(new_output);
    }
    else
    {
        LLK_ASSERT_FP32_DEST_ACC_EN_MATCHES(is_fp32_dest_acc_en);
        if constexpr (!gate)
        {
            TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::UNPACK0);
        }
        ckernel::packer::reconfig_packer_data_format_imm_and_state<
            is_fp32_dest_acc_en,
            imm.strides_full,
            imm.format,
            imm.section,
            imm.lf8,
            imm.threshold,
            imm.dstacc,
            imm.ystride,
            imm.zwstride,
            pack_src_format[new_output],
            pack_dst_format[new_output],
            pack_tile_num_faces[new_output]>();
    }
}

template <bool is_fp32_dest_acc_en, std::uint32_t new_output>
__attribute__((always_inline)) inline void pack_reconfig_data_format()
{
    if constexpr (ckernel::packer::llk_pack_reconfig_u2d_half<is_fp32_dest_acc_en, DST_SYNC_MODE>() && new_output < sizeof(pack_dst_format))
    {
        if constexpr (ckernel::packer::llk_pack_out32(pack_dst_format[new_output]))
        {
            if (ckernel::packer::pack_short_block)
            {
                pack_reconfig_data_format_gate<is_fp32_dest_acc_en, new_output, false, true>();
            }
            else
            {
                pack_reconfig_data_format_gate<is_fp32_dest_acc_en, new_output, true>();
            }
        }
        else
        {
            pack_reconfig_data_format_gate<is_fp32_dest_acc_en, new_output, false>();
        }
    }
    else
    {
        pack_reconfig_data_format_gate<is_fp32_dest_acc_en, new_output, true>();
    }
}

// The PR's pack_block<ntiles, acc, kt>: the form tables, or arm_form ('T', 'B', 'W') as a constant output that always takes it.
template <std::uint32_t ntiles, bool is_fp32_dest_acc_en, std::uint32_t matmul_kt = 0, char arm_form = 0>
__attribute__((always_inline)) inline void pack_block(const std::uint32_t start_tile_index, const std::uint32_t output)
{
    llk_pack_block<ntiles, is_fp32_dest_acc_en, false, matmul_kt, arm_form>(start_tile_index, output);
}
#endif

} // namespace pack_api_shim

#endif

// The unpack thread's side of a CB the pack thread fills, for a test that defines PACK_API_SHIM_UNPACK_SIDE: its own
// cb_interface over the same L1, and metal's llk_io waits and pops (llk_wait_tiles, llk_pop_tiles).
#if defined(ARCH_BLACKHOLE) && defined(LLK_TRISC_UNPACK) && defined(PACK_API_SHIM_UNPACK_SIDE)

#include <cstdint>

#include "internal/circular_buffer_interface.h"

// The CBs the unpack thread binds: ids below this.
#ifndef PACK_API_SHIM_UNPACK_CBS
#define PACK_API_SHIM_UNPACK_CBS 25
#endif
extern "C" CBInterface pack_api_shim_cb_interface[PACK_API_SHIM_UNPACK_CBS];
__attribute__((used)) CBInterface pack_api_shim_cb_interface[PACK_API_SHIM_UNPACK_CBS];
asm(".globl cb_interface\n.set cb_interface, pack_api_shim_cb_interface");

namespace pack_api_shim
{

#ifndef PACK_API_SHIM_CB_COUNTS
#define PACK_API_SHIM_CB_COUNTS

// metal's get_cb_tiles_received_ptr / get_cb_tiles_acked_ptr (stream_io_map.h): CB n's counts in overlay stream n's
// REMOTE_DEST_BUF_SIZE (10) and REMOTE_DEST_BUF_START (8) registers; the harness keeps noc_overlay_parameters.h out.
inline volatile std::uint32_t* cb_tiles_received_ptr(const std::uint32_t cb)
{
    return reinterpret_cast<volatile std::uint32_t*>(0xFFB40000 + cb * 0x1000 + 10 * 4);
}

inline volatile std::uint32_t* cb_tiles_acked_ptr(const std::uint32_t cb)
{
    return reinterpret_cast<volatile std::uint32_t*>(0xFFB40000 + cb * 0x1000 + 8 * 4);
}
#endif

// The consumer's binding: the producer clears the counts (pack thread's cb_bind<true>); this one starts its own at 0.
inline void cb_bind(const std::uint32_t cb, const std::uint32_t rd_ptr, const std::uint32_t page_words, const std::uint32_t num_pages)
{
    LocalCBInterface& c         = get_local_cb_interface(cb);
    c.fifo_page_size            = page_words;
    c.fifo_num_pages            = num_pages;
    c.fifo_size                 = page_words * num_pages;
    c.fifo_wr_ptr               = rd_ptr;
    c.fifo_rd_ptr               = rd_ptr;
    c.fifo_limit                = rd_ptr + c.fifo_size;
    c.fifo_wr_tile_ptr          = 0;
    c.tiles_acked_received_init = 0;
}

#if defined(LLK_PROFILER)
inline constexpr bool cb_waits = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;
#else
inline constexpr bool cb_waits = true;
#endif

// cb_wait_front: llk_wait_tiles.
inline void cb_wait_front(const std::uint32_t cb, const std::uint32_t num_tiles)
{
    if constexpr (cb_waits)
    {
        volatile std::uint32_t* tiles_received_ptr = cb_tiles_received_ptr(cb);
        const std::uint16_t num_tiles_u            = (std::uint16_t)num_tiles;
        std::uint16_t num_tiles_recv;
        do
        {
            const std::uint16_t tiles_received = (std::uint16_t)ckernel::reg_read((std::uint32_t)tiles_received_ptr);
            num_tiles_recv                     = tiles_received - get_local_cb_interface(cb).tiles_acked;
        } while (num_tiles_recv < num_tiles_u);
    }
}

// cb_pop_front: llk_pop_tiles, the ack stored once the unpacker is done.
inline void cb_pop_front(const std::uint32_t cb, const std::uint32_t num_tiles)
{
    const std::uint32_t tiles_acked_addr_tensix = static_cast<std::uint32_t>((reinterpret_cast<std::uintptr_t>(cb_tiles_acked_ptr(cb)) >> 2) & 0x3ffff);
    LocalCBInterface& c                         = get_local_cb_interface(cb);
    const std::uint32_t num_words               = num_tiles * c.fifo_page_size;
    c.tiles_acked += num_tiles;
    TT_SETDMAREG(0, c.tiles_acked, 0, LO_16(4));
    TTI_STALLWAIT(ckernel::p_stall::STALL_THCON, ckernel::p_stall::UNPACK);
    TT_STOREREG(4, tiles_acked_addr_tensix);
    c.fifo_rd_ptr += num_words;
    if (c.fifo_rd_ptr >= c.fifo_limit)
    {
        c.fifo_rd_ptr -= c.fifo_size;
    }
}

} // namespace pack_api_shim

#endif
