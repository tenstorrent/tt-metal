// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Blackhole hardware allocation experiment. Per-row values provide an
// address-sensitive oracle. An intervening typed load/store must not
// change the raw value. SCHEME: 0 unannotated, 1 read/write pairs, 2 effects,
// 3 explicitly thread the producer value to the consumer; 4 also pins the
// capture so its point clobber survives when the result is unused; 5 additionally
// threads a partial write's old destination across a preceding typed gap.
#include <array>
#include <cstdint>
#include <utility>
#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context = 0;
std::uint32_t pack_sync_tile_dst_ptr = 0;
std::uint32_t math_sync_tile_dst_index = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}
#endif

#ifdef LLK_TRISC_MATH
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"
using namespace ckernel;
#undef __builtin_rvtt_sfpload
#undef __builtin_rvtt_sfpstore

template <unsigned Word> __attribute__((always_inline)) inline void issue_word()
{
    if constexpr (USE_MMIO) instrn_buffer[0] = Word;
    else asm volatile (".ttinsn %0" : : "n"(Word));
}

template <unsigned Row> __attribute__((always_inline)) inline void probe_row()
{
    __xtt_vector old_destination;
    if constexpr (PARTIAL) {
        issue_word<static_cast<unsigned>(TT_OP_SFPENCC(3, 0, 0, 10))>();
        issue_word<TT_OP_SFPLOADI(LREG, 0, 0x4040)>(); // inactive lanes: 3.0
        if constexpr (SCHEME == 2)
            __builtin_rvtt_sfprawlreg_effect(1u << LREG, 1u << LREG);
        if constexpr (SCHEME == 5) {
            old_destination = __builtin_rvtt_sfpreadlreg(LREG);
            __builtin_rvtt_sfpwritelreg(old_destination, LREG);
        }
        if constexpr (OLD_DESTINATION_GAP) {
            auto gap = __builtin_rvtt_sfpload(nullptr, 2 * Row, 0, 0, 0, 7);
            __builtin_rvtt_sfpstore(nullptr, gap, 2 * Row, 0, 0, 0, 7);
        }
        constexpr unsigned predicate_reg = (LREG + 1) % 8;
        issue_word<TT_OP_SFPLOAD(predicate_reg, 0, 7, 2 * Row)>();
        if constexpr (SCHEME == 2)
            __builtin_rvtt_sfprawlreg_effect(1u << predicate_reg, 1u << predicate_reg);
        issue_word<TT_OP_SFPSETCC(0, predicate_reg, 0, 0)>();
        if constexpr (SCHEME == 2)
            __builtin_rvtt_sfprawlreg_effect(1u << predicate_reg, 0);
        if constexpr (SCHEME == 5)
            __builtin_rvtt_sfpwritelreg(old_destination, LREG);
    }
    __xtt_vector temporary;
    if constexpr (LIVE_BEFORE)
        temporary = __builtin_rvtt_sfpload(nullptr, 2 * Row, 0, 0, 0, 7);
    issue_word<TT_OP_SFPLOADI(LREG, 0, 0x3f80 + Row)>();
    if constexpr (SCHEME == 1)
        __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpreadlreg(LREG), LREG);
    else if constexpr (SCHEME == 2)
        __builtin_rvtt_sfprawlreg_effect(1u << LREG, 1u << LREG);
    // A real C++ use-def edge, unlike two independent identity pairs.
    // Reading is deliberately after the raw producer, not before it.
    __xtt_vector saved;
    if constexpr (SCHEME == 3 && !DEAD_OUTPUT) saved = __builtin_rvtt_sfpreadlreg(LREG);
    else if constexpr (SCHEME == 3) (void)__builtin_rvtt_sfpreadlreg(LREG);
    else if constexpr (SCHEME == 4 || SCHEME == 5) {
        saved = __builtin_rvtt_sfpreadlreg(LREG);
        // Retain the producer clobber even if no consumer uses saved.
        // When saved is used, its C++ lifetime also spans the typed work.
        __builtin_rvtt_sfpwritelreg(saved, LREG);
    }
    if constexpr (!LIVE_BEFORE)
        temporary = __builtin_rvtt_sfpload(nullptr, 2 * Row, 0, 0, 0, 7);
    __builtin_rvtt_sfpstore(nullptr, temporary, 2 * Row, 0, 0, 0, 7);
    if constexpr (FORCE_MOVE)
        __builtin_rvtt_sfpwritelreg(temporary, LREG);
    if constexpr (SCHEME == 1)
        __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpreadlreg(LREG), LREG);
    else if constexpr ((SCHEME == 3 || SCHEME == 4 || SCHEME == 5) && !DEAD_OUTPUT)
        __builtin_rvtt_sfpwritelreg(saved, LREG);
    if constexpr (PARTIAL) issue_word<static_cast<unsigned>(TT_OP_SFPENCC(3, 0, 0, 10))>();
    if constexpr (!DEAD_OUTPUT) {
        issue_word<TT_OP_SFPSTORE(LREG, 0, 7, 64 + 2 * Row)>();
        if constexpr (SCHEME == 2) __builtin_rvtt_sfprawlreg_effect(1u << LREG, 0);
    }
}

template <std::size_t... Row> inline void probe_rows(std::index_sequence<Row...>)
{
    (probe_row<Row>(), ...);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_init_once_();
    math::reset_counters(p_setrwc::SET_ABD_F);
    _llk_math_welfords_sfpu_params_(+[]() {
        // Each partial-predicate row establishes its own mask below.
        TTI_SFPENCC(3, 0, 0, 10);
        probe_rows(std::make_index_sequence<32>{});
    }, 0);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    if constexpr (!DEAD_OUTPUT)
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif
