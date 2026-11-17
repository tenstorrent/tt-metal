// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>
#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "api/compute/matmul_fe_mark.h"
#ifdef ARCH_BLACKHOLE
#include "llk_size_mode.h"
#endif
#include "llk_assert.h"
#include "sanitizer/api.h"
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_common_api.h"
#endif
PACK_MATMUL_API(1)
// defines the default throttle level for matmul kernels (default 0)
#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif
namespace ckernel {

// reuse_dir of matmul_block_init and matmul_block when the kernel names none: on Blackhole the direction matmul_api_reuse_dir picks
constexpr std::uint32_t MATMUL_REUSE_DIR_AUTO = 0xFF;

#ifdef ARCH_BLACKHOLE
// The forms key of a kernel's matmul block: 0, main's form, where the block (block_rt, block_ct, block_kt: optional keys of the caller) is one of the cells
// llk_matmul_forms.h keeps on main's form (the lookup the tt-llk harness runs); the caller's k steps per DEST section otherwise.
template <std::uint32_t kt_per_section, std::uint32_t block_rt, std::uint32_t block_ct, std::uint32_t block_kt, DataFormat in0, DataFormat in1, DataFormat out_format, int throttle_level>
constexpr std::uint32_t llk_matmul_form_key() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    if constexpr (block_rt != 0 && block_ct != 0 && block_kt != 0 && in0 != DataFormat::Invalid && in1 != DataFormat::Invalid) {
#ifdef PACKER_L1_ACC
        constexpr bool l1_acc = true;
#else
        constexpr bool l1_acc = false;
#endif
        return matmul_form_k<
            MATH_FIDELITY,
            DST_SYNC_MODE == DstSync::SyncHalf ? 0 : 1,
            DST_ACCUM_MODE,
            kt_per_section,
            block_kt,
            in0,
            in1,
            out_format,
            block_ct,
            block_rt,
            throttle_level,
            l1_acc>();
    } else {
        return kt_per_section;
    }
#else
    return kt_per_section;  // the pack thread has no MATH_FIDELITY and never reads this
#endif
}

// The forced form (llk_matmul_forms.h matmul_row_mop_form) that replaces the row MOP a caller opts into for its block, 0 where the row MOP stays or the block keys are not given.
template <bool row_mop, std::uint32_t kt_per_section, std::uint32_t block_rt, std::uint32_t block_ct, std::uint32_t block_kt, DataFormat in0, DataFormat in1, DataFormat out_format>
constexpr char llk_matmul_row_mop_form() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    if constexpr (row_mop && block_rt != 0 && block_ct != 0 && block_kt != 0 && in0 != DataFormat::Invalid && in1 != DataFormat::Invalid) {
        return matmul_row_mop_form<MATH_FIDELITY, DST_ACCUM_MODE, in0, in1, out_format, DST_SYNC_MODE == DstSync::SyncHalf ? 0 : 1, block_rt, block_ct, block_kt, kt_per_section>();
    }
#endif
    return 0;
}

// Whether the caller's block is one of the cells llk_matmul_forms.h keeps on main's form (block keys given): an opt-in
// (row_mop, unpack_auto_ttsync) on such a block runs main's default.
template <
    std::uint32_t block_rt,
    std::uint32_t block_ct,
    std::uint32_t block_kt,
    DataFormat in0,
    DataFormat in1,
    DataFormat out_format>
constexpr bool llk_matmul_main_cell() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    if constexpr (
        block_rt != 0 && block_ct != 0 && block_kt != 0 && in0 != DataFormat::Invalid && in1 != DataFormat::Invalid) {
#ifdef PACKER_L1_ACC
        constexpr bool l1_acc = true;
#else
        constexpr bool l1_acc = false;
#endif
        return matmul_grid_main_form(
            MATH_FIDELITY,
            DST_SYNC_MODE == DstSync::SyncHalf,
            DST_ACCUM_MODE,
            in1,
            in0,
            out_format,
            block_rt,
            block_ct,
            block_kt,
            MM_THROTTLE,
            l1_acc);
    }
#endif
    return false;
}

// A block that keeps its tile MOP though the caller opts into the row MOP (llk_matmul_forms.h matmul_row_mop_off).
template <
    std::uint32_t block_rt,
    std::uint32_t block_ct,
    std::uint32_t block_kt,
    DataFormat in0,
    DataFormat in1,
    DataFormat out_format>
constexpr bool llk_matmul_row_mop_off() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    if constexpr (block_rt != 0 && block_ct != 0 && block_kt != 0 && in1 != DataFormat::Invalid) {
        return matmul_row_mop_off(
            MATH_FIDELITY,
            DST_SYNC_MODE == DstSync::SyncHalf,
            DST_ACCUM_MODE,
            in1,
            in0,
            out_format,
            block_rt,
            block_ct,
            block_kt,
            MM_THROTTLE);
    }
#endif
    return false;
}

// The unpack_auto_ttsync the unpack calls run: the caller's, except on a block (keys given) that is a main form cell or
// where the ttsync unpack measured slower than the default unpack (matmul_unpack_ttsync_cell). Decided at compile time,
// so the unpack code of every other block is unchanged.
template <
    std::uint32_t unpack_auto_ttsync,
    std::uint32_t block_rt,
    std::uint32_t block_ct,
    std::uint32_t block_kt,
    DataFormat in0,
    DataFormat in1,
    DataFormat out_format>
constexpr std::uint32_t llk_matmul_unpack_ttsync() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    // LoFi class rule: the ttsync unpack gains on one-tile blocks (the poll binds) and loses where a call unpacks two
    // or more tiles of a row (its config writes then sit on the busy unpack thread); a block the front end cannot see
    // keeps the poll. A cell pick (3) stands
    if constexpr (MATH_FIDELITY == MathFidelity::LoFi && unpack_auto_ttsync == 1 && !(block_rt == 1 && block_ct == 1)) {
        return 0;
    }
    if constexpr (
        unpack_auto_ttsync != 0 && block_rt != 0 && block_ct != 0 && block_kt != 0 && in0 != DataFormat::Invalid &&
        in1 != DataFormat::Invalid) {
        if (llk_matmul_main_cell<block_rt, block_ct, block_kt, in0, in1, out_format>()) {
            return 0;
        }
        return matmul_unpack_ttsync_cell(
                   MATH_FIDELITY,
                   DST_SYNC_MODE == DstSync::SyncHalf,
                   DST_ACCUM_MODE,
                   in1,
                   in0,
                   out_format,
                   block_rt,
                   block_ct,
                   block_kt)
                   ? unpack_auto_ttsync
                   : 0;
    }
#endif
    return unpack_auto_ttsync;
}

// True when no CB of the kernel holds a 32-bit format. The JIT's unpack_src_format holds every CB's buffer format,
// outputs included (genfiles.cpp generate_unpack_data_formats, equalized with pack_dst_format), so then no section
// packs to Float32.
constexpr bool llk_matmul_kernel_no_32bit_cb() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    for (const std::uint8_t f : unpack_src_format) {
        if (matmul_format_is_32bit(static_cast<DataFormat>(f))) {
            return false;
        }
    }
    return true;
#else
    return false;
#endif
}

// The fidelity whose phases a matmul runs for operand formats known at compile time (matmul_phase_fidelity); the requested
// one when they are Invalid, mixed, or the output and DEST width are a cell that measured slower with the phases dropped.
// Only the math calls take it; the unpack and math forms stay keyed on the requested fidelity (MATH_FIDELITY).
template <DataFormat in0_format, DataFormat in1_format, DataFormat out_format>
constexpr MathFidelity matmul_elided_fidelity() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    // an unknown output of a kernel without 32-bit CBs is not Float32, which is all the keep rule asks
    constexpr DataFormat keep_out =
        out_format == DataFormat::Invalid && llk_matmul_kernel_no_32bit_cb() ? DataFormat::Float16_b : out_format;
    return matmul_phase_fidelity<MATH_FIDELITY, in0_format, in1_format, keep_out, DST_ACCUM_MODE, MM_THROTTLE, true>();
#else
    return MathFidelity::LoFi;  // the pack thread has no MATH_FIDELITY and never reads this
#endif
}

// True when the kernel's unpack formats hold a 32-bit one and one that unpacks twice as fast (matmul_format_streams_fast): only such a
// kernel can hold in1 where main holds in0 (or the reverse), so the others fold to main's code.
template <bool throttled>
constexpr bool llk_matmul_kernel_mixes_32bit() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    bool wide = false, fast = false;
    for (const std::uint8_t f : unpack_src_format) {
        wide = wide || matmul_format_is_32bit(static_cast<DataFormat>(f));
        fast = fast || matmul_format_streams_fast(static_cast<DataFormat>(f), throttled);
    }
    return wide && fast;
#else
    return false;
#endif
}

// True when every CB of the kernel with a format holds whole 32x32 tiles of four faces (the two-row schemes need them)
constexpr bool llk_matmul_kernel_full_tiles() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    for (std::uint32_t cb = 0; cb < sizeof(unpack_src_format); cb++) {
        if (unpack_src_format[cb] != static_cast<std::uint8_t>(DataFormat::Invalid) &&
            (unpack_tile_r_dim[cb] != 32 || unpack_tile_c_dim[cb] != 32 || unpack_tile_num_faces[cb] != 4 || unpack_partial_face[cb] != 0)) {
            return false;
        }
    }
    return true;
#else
    return false;
#endif
}

// The reuse direction a matmul call runs. MATMUL_REUSE_DIR_AUTO: with the operand formats and the block (block_rt, block_ct) as template
// arguments matmul_reuse_dir picks it at compile time, the two-row schemes included (whole tiles, no static throttle, and not a half sync
// Float32 output over one k step, which is pack bound); a kernel that passes neither runs 3, the init picking the held operand from the CB
// formats and its ct_dim and rt_dim (matmul_api_set_reuse_a), when it is LoFi, may pair a 32-bit operand with a fast one, keeps main's
// form (no row MOP, k key, Auto TTSync or stream_z opt-in) and has a known non-Float32 output at half sync. 3 has no two-row scheme: that
// is another unpack and math body, which a run-time choice would branch to on every call. A direction other than 0 and 3 runs main's tile
// form and per-row unpack (the forms the cells were measured on).
template <
    std::uint32_t reuse_dir,
    DataFormat in0,
    DataFormat in1,
    DataFormat out_format,
    std::uint32_t block_rt,
    std::uint32_t block_ct,
    std::uint32_t kt_per_section,
    bool row_mop,
    std::uint32_t unpack_auto_ttsync,
    std::uint32_t stream_z>
constexpr std::uint32_t matmul_api_reuse_dir() {
    if constexpr (reuse_dir != MATMUL_REUSE_DIR_AUTO) {
        return reuse_dir;
    } else {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
        constexpr bool half_f32 =
            DST_SYNC_MODE == DstSync::SyncHalf && (out_format == DataFormat::Float32 || out_format == DataFormat::Invalid);
        if constexpr (MATH_FIDELITY != MathFidelity::LoFi) {
            return 0;
        } else if constexpr (in0 != DataFormat::Invalid && in1 != DataFormat::Invalid && block_rt != 0 && block_ct != 0) {
            constexpr bool two_row_off = MM_THROTTLE != 0 || !llk_matmul_kernel_full_tiles() || (half_f32 && kt_per_section <= 1);
            return matmul_reuse_dir(MATH_FIDELITY, in0, in1, block_ct, block_rt, half_f32, two_row_off, MM_THROTTLE != 0);
        } else if constexpr (row_mop || unpack_auto_ttsync != 0 || stream_z != 0 || kt_per_section != 0) {
            return 0;
        } else if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
            // half sync: the run-time direction where a 32-bit operand sits beside a fast one (matmul_kernel_lofi_mix)
            return !LLK_SIZE_MODE && out_format != DataFormat::Float32 && matmul_kernel_lofi_mix(unpack_src_format) ? 3
                                                                                                                    : 0;
        } else {
            return llk_matmul_kernel_mixes_32bit<MM_THROTTLE != 0>() ? 3 : 0;
        }
#else
        return 0;
#endif
    }
}

// reuse_dir 3: the direction matmul_reuse_dir picks for the operand formats of the CBs and the block
ALWI std::uint32_t matmul_api_runtime_dir(
    const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id, const std::uint32_t ct_dim, const std::uint32_t rt_dim) {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    return matmul_reuse_dir(
        MathFidelity::LoFi,
        static_cast<DataFormat>(unpack_src_format[get_operand_id(in0_cb_id)]),
        static_cast<DataFormat>(unpack_src_format[get_operand_id(in1_cb_id)]),
        ct_dim,
        rt_dim,
        false,
        true,
        MM_THROTTLE != 0);
#else
    return 0;
#endif
}
#endif

#ifdef ARCH_BLACKHOLE
// defines the FW-controlled throttle level for block matmul kernels on Blackhole
#define MM_THROTTLE_MAX 5
// 4-byte word at MEM_L1_ARC_FW_SCRATCH written by FW - even means no throttle, odd means throttle
volatile tt_l1_ptr std::uint32_t* throttle_ptr =
    reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(MEM_L1_ARC_FW_SCRATCH);
// tracks the state of the currently programmed matmul MOP (0: default throttle level, 1: max throttle level)
static std::uint32_t throttled_mop_status = 0;

// clang-format off
/**
 * Performs matmul block operation with dynamic throttling.
 * This function is only available on Blackhole architecture and implements
 * firmware-controlled dynamic throttling for block matmul operations.
 * The throttle level is controlled by firmware via MEM_L1_ARC_FW_SCRATCH.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The column dimension for the output block.                              | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
// clang-format on
template <
    bool row_mop = false,
    bool tile_mop = true,
    int in1_order = 0,
    DataFormat in0_format = DataFormat::Invalid,
    DataFormat in1_format = DataFormat::Invalid,
    DataFormat out_format = DataFormat::Invalid,
    std::uint32_t reuse_dir = 0,
    bool row_end_fold = false,
    bool half_end_fold = false,
    std::uint32_t key_ct = 0,
    std::uint32_t key_rt = 0>
ALWI void matmul_block_math_dynamic_throttle(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    const std::uint32_t held_a = matmul_held_a_init) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    // Dynamic throttling is only available on Blackhole architecture
    // Check firmware-controlled throttle enable flag (even = no throttle, odd = throttle)
    const std::uint32_t mm_throttle_en = *(throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (throttled_mop_status != 1) {
            // the held operand of a reuse_dir 3 MOP
            if constexpr (reuse_dir == 3) {
                if (held_a != matmul_held_a_init) {
                    matmul_runtime_reuse_a = held_a != 0;
                }
            }
            MATH((llk_math_matmul_init_switch<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE_MAX, false, true, in1_order, reuse_dir, false, false, key_ct, key_rt>(
                in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 1;
        }
        MATH((llk_math_matmul<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE_MAX, /*num_faces=*/4, false, true, reuse_dir>(idst, ct_dim, rt_dim, held_a)));
    } else {
        if (throttled_mop_status != 0) {
            // the held operand of a reuse_dir 3 MOP
            if constexpr (reuse_dir == 3) {
                if (held_a != matmul_held_a_init) {
                    matmul_runtime_reuse_a = held_a != 0;
                }
            }
            MATH((llk_math_matmul_init_switch<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE, row_mop, tile_mop, in1_order, reuse_dir, row_end_fold, half_end_fold, key_ct, key_rt>(
                in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 0;
        }
        MATH((llk_math_matmul<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE, /*num_faces=*/4, row_mop, tile_mop, reuse_dir, row_end_fold>(idst, ct_dim, rt_dim, held_a)));
    }
#endif
}
#endif

#ifdef ARCH_BLACKHOLE
// HiFi3 half sync without a Bfp8_b CB, and with a 32-bit DEST also a 32-bit CB: in a kernel whose plain calls all have
// a run-time block (minimal_matmul) the ttsync unpack reads slower (224 2x1 bf16 +117 ns, 244 1x4 +52 ns); with a
// constant block (bmm) it is 6-16 cy faster
constexpr bool matmul_fe_hifi3_wide() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    bool wide = false, bfp8 = false;
    for (const std::uint8_t f : unpack_src_format) {
        wide = wide || f == to_underlying(DataFormat::Float32) || f == to_underlying(DataFormat::Tf32);
        bfp8 = bfp8 || f == to_underlying(DataFormat::Bfp8_b);
    }
    return MATH_FIDELITY == MathFidelity::HiFi3 && DST_SYNC_MODE == DstSync::SyncHalf && (!DST_ACCUM_MODE || wide) && !bfp8;
#else
    return false;
#endif
}

}  // namespace ckernel
// the TRISC wrapper's reading of the cut (matmul.h defines it, so a kernel without the matmul API skips the wrapper's ttsync)
#define MATMUL_FE_TTSYNC_CUT() (::ckernel::matmul_fe_hifi3_wide() && !::ckernel::matmul_fe_mark::is_used<8>(0))
namespace ckernel {

#ifdef TRISC_MATH
}  // namespace ckernel
#include "llk_math_matmul_main.h"
namespace ckernel {
#endif

#ifdef TRISC_MATH
// llk_math_matmul_tile_mop, and the row form at HiFi2 half sync with a 32-bit DEST in a kernel with a constant block
// (matmul_hifi2_half_wide); the matmul_fe_hifi3_wide class keeps the row form (minimal_matmul 424/444/888 bf16 HiFi3
// -2.5 to -3.2 %)
template <bool row_mop_on, std::uint32_t kt_key, DataFormat form_out>
constexpr bool matmul_fe_tile_mop() {
    using K = std::integral_constant<std::uint32_t, kt_key>;
    if constexpr (
        !row_mop_on && kt_key == 0 &&
        matmul_hifi2_half_wide(
            MATH_FIDELITY, DST_SYNC_MODE == DstSync::SyncHalf, DST_ACCUM_MODE, llk_math_matmul_kernel_has_32bit_cb()) &&
        matmul_fe_mark::is_used_d<8, K>(0) && !matmul_fe_mark::is_used_d<15, K>(0)) {
        return false;
    } else if constexpr (
        !row_mop_on && kt_key == 0 && MATH_FIDELITY == MathFidelity::LoFi && DST_SYNC_MODE == DstSync::SyncHalf &&
        matmul_fe_mark::is_used_d<8, K>(0) && !matmul_fe_mark::is_used_d<13, K>(0) &&
        !matmul_fe_mark::is_used_d<15, K>(0) && !matmul_kernel_lofi_mix(unpack_src_format)) {
        // LoFi half sync: the row form in a kernel with a constant block and no constant one-tile or kt-1 block, whose
        // call it slows (bmm's reload init passes its constant block through a helper's parameters, so tag 7 does not
        // tell a run-time-block kernel); a kernel pairing a 32-bit operand with a fast one keeps the tile MOP (the row form hangs
        // the 2D-mcast bmm of Float32 x Float16_b blocks of two or more rows)
        return false;
    } else {
        return llk_math_matmul_tile_mop<row_mop_on, kt_key, form_out>();
    }
}
#endif

#ifdef TRISC_MATH
// a plain call that takes none of the matmul forms in a kernel with no constant block (minimal_matmul's default cells), and every
// plain call in a kernel bound by the kernel config buffer: main's math code, so the kernel's math TRISC is main's
template <std::uint32_t llk_dir, bool row_mop_on, std::uint32_t kt_key, DataFormat form_out, int in1_order, bool row_end_fold, DataFormat in0_format,
          DataFormat in1_format, DataFormat out_format, std::uint32_t block_rt, std::uint32_t block_ct>
constexpr bool matmul_fe_main_math() {
    return llk_dir == 0 && !row_mop_on && kt_key == 0 && form_out == DataFormat::Invalid && in1_order == 0 && !row_end_fold &&
           in0_format == DataFormat::Invalid && in1_format == DataFormat::Invalid && out_format == DataFormat::Invalid && block_rt == 0 &&
           block_ct == 0 &&
           (LLK_SIZE_MODE || (matmul_fe_tile_mop<false, 0, DataFormat::Invalid>() &&
                              !matmul_fe_mark::is_used_d<8, std::integral_constant<std::uint32_t, block_rt + kt_key>>(0)));
}
#endif

#ifdef TRISC_MATH
// The smallest kt_dim from which a plain call that runs main's math because the front end sees no constant block takes the row
// form (0: never), outside size mode: LoFi half sync from kt 3 (census 65c1c: 40 of 76 keys faster, 0 slower; kt <= 2 loses to the
// row path's per-call cost), HiFi2 half sync with a 32-bit DEST from kt 2 (a kt-1 block loses: ccl F32 4x1x1 +1.05). The init and the
// calls receive the same kt_dim.
template <std::uint32_t llk_dir, bool row_mop_on, std::uint32_t kt_key>
constexpr std::uint32_t matmul_fe_rowform_min_kt() {
    if constexpr (
        llk_dir != 0 || row_mop_on || kt_key != 0 || DST_SYNC_MODE != DstSync::SyncHalf || LLK_SIZE_MODE ||
        matmul_fe_mark::is_used_d<8, std::integral_constant<std::uint32_t, kt_key>>(0)) {
        return 0;
    } else if constexpr (MATH_FIDELITY == MathFidelity::LoFi) {
        return 3;
    } else if constexpr (matmul_hifi2_half_wide(MATH_FIDELITY, true, DST_ACCUM_MODE, llk_math_matmul_kernel_has_32bit_cb())) {
        return 2;
    } else {
        return 0;
    }
}
#endif

// The unpack and math inits of matmul_block_init for the reuse direction the LLK runs
template <
    std::uint32_t llk_dir,
    std::uint32_t unpack_ttsync,
    std::uint32_t stream_z,
    bool row_mop_on,
    std::uint32_t kt_key,
    DataFormat form_out,
    int in1_order,
    bool row_end_fold,
    DataFormat in0_format,
    DataFormat in1_format,
    DataFormat out_format,
    std::uint32_t block_rt,
    std::uint32_t block_ct,
    std::uint32_t block_kt>
ALWI void matmul_block_init_llk(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_matmul_init<unpack_ttsync, llk_dir, stream_z>(
        in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
#ifdef TRISC_MATH
    if constexpr (matmul_fe_main_math<llk_dir, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct>()) {
        if constexpr (matmul_fe_rowform_min_kt<llk_dir, row_mop_on, kt_key>() != 0) {
            if (kt_dim >= matmul_fe_rowform_min_kt<llk_dir, row_mop_on, kt_key>()) {
                llk_math_matmul_init<
                    matmul_elided_fidelity<in0_format, in1_format, out_format>(),
                    MM_THROTTLE,
                    false,
                    false,
                    in1_order,
                    0,
                    row_end_fold,
                    llk_math_matmul_half_end_fold<false, matmul_elided_fidelity<in0_format, in1_format, out_format>(), kt_key, out_format, in0_format, in1_format, block_rt, block_ct, block_kt>(),
                    block_ct,
                    block_rt>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim);
                return;
            }
        }
        mm_main::llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim);
        return;
    }
#endif
    MATH((llk_math_matmul_init<
          matmul_elided_fidelity<in0_format, in1_format, out_format>(),
          MM_THROTTLE,
          row_mop_on,
          llk_dir != 0 || matmul_fe_tile_mop<row_mop_on, kt_key, form_out>(),
          in1_order,
          llk_dir,
          row_end_fold,
          llk_math_matmul_half_end_fold<row_mop_on, matmul_elided_fidelity<in0_format, in1_format, out_format>(), kt_key, out_format, in0_format, in1_format, block_rt, block_ct, block_kt>(),
          block_ct,
          block_rt>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
}

// reuse_dir 3 at the init: keeps the held operand for the calls; where the CBs and the block are compile-time constants the direction
// folds and the init runs as 0, 1 or 2
template <
    std::uint32_t unpack_ttsync,
    std::uint32_t stream_z,
    bool row_mop_on,
    std::uint32_t kt_key,
    DataFormat form_out,
    int in1_order,
    bool row_end_fold,
    DataFormat in0_format,
    DataFormat in1_format,
    DataFormat out_format,
    std::uint32_t block_rt,
    std::uint32_t block_ct,
    std::uint32_t block_kt>
ALWI void matmul_block_init_llk_runtime(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    const std::uint32_t d = matmul_api_runtime_dir(in0_cb_id, in1_cb_id, ct_dim, rt_dim);
    if (__builtin_constant_p(d)) {
        if (d == 1) {
            matmul_block_init_llk<1, unpack_ttsync, stream_z, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct, block_kt>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
        } else if (d == 2) {
            matmul_block_init_llk<2, unpack_ttsync, stream_z, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct, block_kt>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
        } else {
            matmul_block_init_llk<0, unpack_ttsync, stream_z, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct, block_kt>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
        }
    } else {
        matmul_runtime_reuse_a = d == 0 ? ct_dim >= rt_dim : d == 1;
        matmul_block_init_llk<3, unpack_ttsync, stream_z, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct, block_kt>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
    }
#endif
}

// The unpack and math calls of matmul_block for the reuse direction the LLK runs
template <
    std::uint32_t llk_dir,
    std::uint32_t unpack_ttsync,
    std::uint32_t stream_z,
    bool row_mop_on,
    std::uint32_t kt_key,
    DataFormat form_out,
    int in1_order,
    bool row_end_fold,
    DataFormat in0_format,
    DataFormat in1_format,
    DataFormat out_format,
    std::uint32_t block_rt,
    std::uint32_t block_ct,
    std::uint32_t block_kt>
ALWI void matmul_block_llk(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t idst,
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim,
    const std::uint32_t held_a = matmul_held_a_init) {
    UNPACK((llk_unpack_AB_matmul<unpack_ttsync, kt_key, form_out, llk_dir, stream_z>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim, held_a)));
#ifdef TRISC_MATH
    if constexpr (matmul_fe_main_math<llk_dir, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct>()) {
        if constexpr (matmul_fe_rowform_min_kt<llk_dir, row_mop_on, kt_key>() != 0) {
            if (kt_dim >= matmul_fe_rowform_min_kt<llk_dir, row_mop_on, kt_key>()) {
                matmul_block_math_dynamic_throttle<
                    false,
                    false,
                    in1_order,
                    in0_format,
                    in1_format,
                    out_format,
                    0,
                    row_end_fold,
                    llk_math_matmul_half_end_fold<false, matmul_elided_fidelity<in0_format, in1_format, out_format>(), kt_key, out_format, in0_format, in1_format, block_rt, block_ct, block_kt>(),
                    block_ct,
                    block_rt>(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim, held_a);
                return;
            }
        }
        mm_main::matmul_block_math_dynamic_throttle(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim);
        return;
    }
#endif
    // Dynamic throttling is only available on Blackhole architecture
    MATH((matmul_block_math_dynamic_throttle<
          row_mop_on,
          llk_dir != 0 || matmul_fe_tile_mop<row_mop_on, kt_key, form_out>(),
          in1_order,
          in0_format,
          in1_format,
          out_format,
          llk_dir,
          row_end_fold,
          llk_math_matmul_half_end_fold<row_mop_on, matmul_elided_fidelity<in0_format, in1_format, out_format>(), kt_key, out_format, in0_format, in1_format, block_rt, block_ct, block_kt>(),
          block_ct,
          block_rt>(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim, held_a)));
}

// reuse_dir 3 at a call: the held operand the init kept; where the CBs and the block are compile-time constants the direction folds
// and the call runs as 0, 1 or 2
template <
    std::uint32_t unpack_ttsync,
    std::uint32_t stream_z,
    bool row_mop_on,
    std::uint32_t kt_key,
    DataFormat form_out,
    int in1_order,
    bool row_end_fold,
    DataFormat in0_format,
    DataFormat in1_format,
    DataFormat out_format,
    std::uint32_t block_rt,
    std::uint32_t block_ct,
    std::uint32_t block_kt>
ALWI void matmul_block_llk_runtime(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t idst,
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    const std::uint32_t d = matmul_api_runtime_dir(in0_cb_id, in1_cb_id, ct_dim, rt_dim);
    const bool reuse_a = d == 0 ? ct_dim >= rt_dim : d == 1;
    matmul_block_llk<3, unpack_ttsync, stream_z, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct, block_kt>(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, idst, transpose, ct_dim, rt_dim, kt_dim, reuse_a);
#endif
}
#endif

// clang-format off
/**
 * Short init for matmul_tiles. Configures the unpacker and math engine to matmul mode.
 *
 * Must be called before matmul_tiles. The one-time HW configuration must already have been
 * performed via compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) at the start of MAIN.
 * Matmul maps in0 -> SrcB and in1 -> SrcA (the reverse of other ops), which is why
 * compute_kernel_hw_startup must use SrcOrder::Reverse.
 *
 * NOTE (known gap, #46769): if a preceding op left SrcA/SrcB with asymmetric tile sizes (i.e. different
 * data formats per source) and the following matmul uses the same formats, matmul_init cannot fix the
 * per-source tile sizes on its own. It does not re-program the tile descriptor, and a reconfig_data_format
 * is inappropriate when the data formats did not change. No current kernel hits this; tracked in #46769.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                       | Required |
 * |----------------|---------------------------------------------------------------|----------|---------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                           | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                           | True     |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate transpose is set | False    |
 */
// clang-format on
template <
    DataFormat in0_format = DataFormat::Invalid,
    DataFormat in1_format = DataFormat::Invalid,
    DataFormat out_format = DataFormat::Invalid>
ALWI void matmul_init(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    const std::uint32_t transpose = 0,
    std::uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
#ifdef ARCH_BLACKHOLE
    MATH((llk_math_matmul_init<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
#else
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
#endif
    PACK_MATH_CONTEXT(true);
#else
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id)));
#endif
}

// clang-format off
/**
 * (Quasar) Undo the automatic MxFp4 -> MxFp4_2x_B src-format selection applied by matmul_init.
 *
 * matmul_init overrides an MxFp4 operand's unpacker OUT_DATA_FORMAT and ALU format to the 2x-packed
 * MxFp4_2x_B, diverging from the op-agnostic unpack_dst_format[] table. That override PERSISTS: the
 * non-matmul unpack inits never reprogram OUT_DATA_FORMAT, and reconfig_data_format is silently
 * skipped for a same-format operand. So a kernel that feeds the SAME MxFp4 buffer to matmul and then
 * to a non-matmul op (datacopy/SFPU/eltwise) MUST call mm_uninit(in0, in1) after the matmuls and
 * before the next op, or that op will keep unpacking the buffer as MxFp4_2x_B and produce garbage.
 * A no-op on non-MxFp4 operands and on non-Quasar architectures.
 *
 * | Argument  | Description                                            | Type     | Valid Range | Required |
 * |-----------|--------------------------------------------------------|----------|-------------|----------|
 * | in0_cb_id | First input CB used in the matmul (same as matmul_init)| uint32_t | 0 to 31     | True     |
 * | in1_cb_id | Second input CB used in the matmul                     | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void mm_uninit(std::uint32_t in0_cb_id, std::uint32_t in1_cb_id) {
#ifdef ARCH_QUASAR
    UNPACK((llk_unpack_AB_matmul_uninit(in0_cb_id, in1_cb_id)));
    MATH((llk_math_matmul_uninit(in0_cb_id, in1_cb_id)));
#endif
}

// clang-format off
/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles in two
 * specified input CBs and accumulates the result to DST (DST += C). The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile A from the first input CB                         | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile B from the second input CB                        | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <
    DataFormat in0_format = DataFormat::Invalid,
    DataFormat in1_format = DataFormat::Invalid,
    DataFormat out_format = DataFormat::Invalid>
ALWI void matmul_tiles(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst) {
    LLK_SAN_FUNCTION();
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index)));
#ifndef ARCH_QUASAR
#ifdef ARCH_BLACKHOLE
    MATH((llk_math_matmul<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE>(idst)));
#else
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)));
#endif
#else
    MATH((llk_math_matmul_tile(idst)));
#endif
}

// clang-format off
/**
 * Short init for matmul_block. Configures the unpacker and math engine to matmul mode.
 *
 * Must be called before matmul_block. The one-time HW configuration must already have been
 * performed via compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) at the start of MAIN.
 * Matmul maps in0 -> SrcB and in1 -> SrcA (the reverse of other ops), which is why
 * compute_kernel_hw_startup must use SrcOrder::Reverse.
 *
 * NOTE (known gap, #46769): if a preceding op left SrcA/SrcB with asymmetric tile sizes (i.e. different
 * data formats per source) and the following matmul uses the same formats, matmul_block_init cannot fix
 * the per-source tile sizes on its own. It does not re-program the tile descriptor, and a
 * reconfig_data_format is inappropriate when the data formats did not change. No current kernel hits this;
 * tracked in #46769.
 *
 * Template parameter row_mop (ignored outside Blackhole): one math MOP per reuse row, the same value in matmul_block.
 * It sets the math ADDR_MOD_3, 5 and 6 (ADDR_MOD_7 stays the SFPU invariant {0,0,0}): the SFPU ops whose init sets ADDR_MOD_6 (reciprocal, typecast,
 * unary min and max, the comparisons with zero, topk local sort) need their init redone after this one, and this init after them.
 *
 * Template parameter unpack_auto_ttsync (ignored outside Blackhole): row base addresses under Auto TTSync instead of polling
 * for a free context, the same value in matmul_block; the firmware turns Auto TTSync off after the kernel.
 *
 * Template parameters kt_per_section and out_format (ignored outside Blackhole): the matmul_block calls per DEST section
 * and the packed output format, which pick the unpack and math forms (matmul_default_form); 0 and Invalid keep the
 * forms main runs. The same values in matmul_block.
 *
 * Return value: None
 *
 * | Argument  | Description                                                | Type     | Valid Range                                                                                    | Required |
 * |-----------|------------------------------------------------------------|----------|------------------------------------------------------------------------------------------------|----------|
 * | in0_cb_id | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                                                                        | True     |
 * | in1_cb_id | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                                                                        | True     |
 * | transpose | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set                                              | False    |
 * | ct_dim    | The column dimension for the output block.                 | uint32_t | Must be equal to block B column dimension; 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | rt_dim    | The row dimension for the output block.                    | uint32_t | Must be equal to block A row dimension; 1 to 8 in half-sync mode, 1 to 16 in full-sync mode    | False    |
 * | kt_dim    | The inner dimension.                                       | uint32_t | Must be equal to block A column dimension                                                      | False    |
 */
// clang-format on
template <
    bool row_mop = false,
    std::uint32_t unpack_auto_ttsync = false,
    std::uint32_t kt_per_section = 0,
    DataFormat out_format = DataFormat::Invalid,
    int in1_order = 0,
    DataFormat in0_format = DataFormat::Invalid,
    DataFormat in1_format = DataFormat::Invalid,
    std::uint32_t reuse_dir = MATMUL_REUSE_DIR_AUTO,
    bool row_end_fold = false,
    std::uint32_t stream_z = 0,
    std::uint32_t block_rt = 0,
    std::uint32_t block_ct = 0,
    std::uint32_t block_kt = 0,
    DataFormat form_out_key = DataFormat::Invalid>
ALWI void matmul_block_init(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    const std::uint32_t transpose = 0,
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1,
    std::uint32_t call_line = __builtin_LINE()) {
#ifdef ARCH_BLACKHOLE
    constexpr std::uint32_t dir = matmul_api_reuse_dir<
        reuse_dir,
        in0_format,
        in1_format,
        out_format,
        block_rt,
        block_ct,
        kt_per_section,
        row_mop,
        unpack_auto_ttsync,
        stream_z>();
    // a held operand other than main's runs main's tile form and per-row unpack with the default stream
    constexpr bool keyed = dir == 0 || dir == 3;
    constexpr std::uint32_t kt_key =
        keyed ? llk_matmul_form_key<kt_per_section, block_rt, block_ct, block_kt, in0_format, in1_format, out_format, MM_THROTTLE>() : 0;
    constexpr char row_mop_form =
        keyed ? llk_matmul_row_mop_form<row_mop, kt_per_section, block_rt, block_ct, block_kt, in0_format, in1_format, out_format>() : 0;
    constexpr bool row_mop_on =
        keyed && row_mop && row_mop_form == 0 &&
        !llk_matmul_main_cell<block_rt, block_ct, block_kt, in0_format, in1_format, out_format>() &&
        !llk_matmul_row_mop_off<block_rt, block_ct, block_kt, in0_format, in1_format, out_format>();
    constexpr std::uint32_t unpack_ttsync =
        keyed ? llk_matmul_unpack_ttsync<unpack_auto_ttsync, block_rt, block_ct, block_kt, in0_format, in1_format, out_format>() : 0;
    constexpr std::uint32_t stream_z_on = keyed ? stream_z : 0;
    constexpr DataFormat form_out = row_mop_form != 0 ? matmul_forced_form_key(row_mop_form)
                                    : keyed ? (out_format != DataFormat::Invalid ? out_format : form_out_key)
                                            : DataFormat::Invalid;
#else
    constexpr std::uint32_t kt_key = kt_per_section;
#endif
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
#ifdef ARCH_BLACKHOLE
    if constexpr (dir == 3) {
        matmul_block_init_llk_runtime<unpack_ttsync, stream_z_on, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct, block_kt>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        matmul_block_init_llk<dir, unpack_ttsync, stream_z_on, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct, block_kt>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
    }
    // Dynamic throttling is only available on Blackhole architecture
    MATH((throttled_mop_status = 0));
#else
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
#endif
    PACK_MATH_CONTEXT(true);
#else
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, ct_dim, rt_dim)));
#endif
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and accumulates the result to DST (DST += C). The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * A block is a rectangle of tiles: A is rt_dim x kt_dim tiles, B is kt_dim x ct_dim tiles, and the
 * output C is rt_dim x ct_dim tiles. So a block is just ct_dim * rt_dim output tiles produced in one
 * call (with kt_dim tiles along the shared inner dimension). The output must fit in DST, so the block
 * size is limited by DST size and sync mode (see matmul_block_init for the valid ct_dim/rt_dim ranges).
 * A call may use smaller ct_dim and rt_dim than matmul_block_init but not flip ct_dim >= rt_dim, which fixes the held
 * operand; row_mop, unpack_auto_ttsync and in1_order must be the values matmul_block_init was called with.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile in block A from the first input CB                | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile in block B from the second input CB               | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The column dimension for the output block.                              | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
// clang-format on
template <
    bool row_mop = false,
    std::uint32_t unpack_auto_ttsync = false,
    std::uint32_t kt_per_section = 0,
    DataFormat out_format = DataFormat::Invalid,
    int in1_order = 0,
    DataFormat in0_format = DataFormat::Invalid,
    DataFormat in1_format = DataFormat::Invalid,
    std::uint32_t reuse_dir = MATMUL_REUSE_DIR_AUTO,
    bool row_end_fold = false,
    std::uint32_t stream_z = 0,
    std::uint32_t block_rt = 0,
    std::uint32_t block_ct = 0,
    std::uint32_t block_kt = 0,
    DataFormat form_out_key = DataFormat::Invalid>
ALWI void matmul_block(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    std::uint32_t call_line = __builtin_LINE()) {
#ifdef ARCH_BLACKHOLE
    constexpr std::uint32_t dir = matmul_api_reuse_dir<
        reuse_dir,
        in0_format,
        in1_format,
        out_format,
        block_rt,
        block_ct,
        kt_per_section,
        row_mop,
        unpack_auto_ttsync,
        stream_z>();
    // a held operand other than main's runs main's tile form and per-row unpack with the default stream
    constexpr bool keyed = dir == 0 || dir == 3;
    constexpr std::uint32_t kt_key =
        keyed ? llk_matmul_form_key<kt_per_section, block_rt, block_ct, block_kt, in0_format, in1_format, out_format, MM_THROTTLE>() : 0;
    constexpr char row_mop_form =
        keyed ? llk_matmul_row_mop_form<row_mop, kt_per_section, block_rt, block_ct, block_kt, in0_format, in1_format, out_format>() : 0;
    constexpr bool row_mop_on =
        keyed && row_mop && row_mop_form == 0 &&
        !llk_matmul_main_cell<block_rt, block_ct, block_kt, in0_format, in1_format, out_format>() &&
        !llk_matmul_row_mop_off<block_rt, block_ct, block_kt, in0_format, in1_format, out_format>();
    constexpr std::uint32_t unpack_ttsync =
        keyed ? llk_matmul_unpack_ttsync<unpack_auto_ttsync, block_rt, block_ct, block_kt, in0_format, in1_format, out_format>() : 0;
    constexpr std::uint32_t stream_z_on = keyed ? stream_z : 0;
    constexpr DataFormat form_out = row_mop_form != 0 ? matmul_forced_form_key(row_mop_form)
                                    : keyed ? (out_format != DataFormat::Invalid ? out_format : form_out_key)
                                            : DataFormat::Invalid;
#else
    constexpr std::uint32_t kt_key = kt_per_section;
#endif
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
#ifdef ARCH_BLACKHOLE
    if constexpr (dir == 3) {
        matmul_block_llk_runtime<unpack_ttsync, stream_z_on, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct, block_kt>(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, idst, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        matmul_block_llk<dir, unpack_ttsync, stream_z_on, row_mop_on, kt_key, form_out, in1_order, row_end_fold, in0_format, in1_format, out_format, block_rt, block_ct, block_kt>(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, idst, transpose, ct_dim, rt_dim, kt_dim);
    }
#else
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
#endif
#else
    LLK_ASSERT(idst == 0, "non-default idst not supported on Quasar");
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_block(ct_dim, rt_dim)));
#endif
}

#ifdef ARCH_BLACKHOLE
// A plain call goes through the macros at the end of this file with the cb ids that are constant where it is written;
// each choice is an if constexpr at instantiation, so a call it does not apply to keeps its instantiation.

// a block dim by value (MATMUL_FE_SHAPE)
constexpr std::uint32_t matmul_fe_dim(const std::uint32_t dim) { return dim; }

// a cb id as a template argument; 0xFF for an argument that is not one
template <typename T>
constexpr std::uint32_t matmul_fe_cb(const T& cb) {
    if constexpr (std::is_convertible_v<T, std::uint32_t>) {
        return static_cast<std::uint32_t>(cb);
    } else {
        return 0xFFu;
    }
}

// The format of equal Bfp8_b or Bfp4_b operands whose dropped fidelity phases add exactly 0 (matmul_phase_fidelity,
// with the output of matmul_elided_fidelity); Invalid where the call keeps the requested fidelity or a cb id is not a
// constant
template <std::uint32_t in0_cb, std::uint32_t in1_cb>
constexpr DataFormat matmul_fe_phase_format() {
#if defined(UCK_CHLKC_MATH)
    if constexpr (in0_cb < sizeof(unpack_src_format) && in1_cb < sizeof(unpack_src_format)) {
        constexpr auto f = static_cast<DataFormat>(unpack_src_format[in0_cb]);
        if constexpr (
            f == static_cast<DataFormat>(unpack_src_format[in1_cb]) &&
            (f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b) &&
            matmul_elided_fidelity<f, f, DataFormat::Invalid>() != MATH_FIDELITY) {
            return f;
        }
    }
#endif
    return DataFormat::Invalid;
}

// the block keys of the init (key_ct, key_rt: the end fold at full sync, no bias write) where ct and rt are constant; in size
// mode none, so every call shares main's init instead of one instantiation per keyed block
template <std::uint32_t ct, std::uint32_t rt>
constexpr std::uint32_t matmul_fe_key(const std::uint32_t dim) {
    return (ct != 0xFFu && rt != 0xFFu && !LLK_SIZE_MODE) ? dim : 0;
}

// Whether a plain call of constant cbs and block (ct, rt, kt) runs the ttsync unpack in a kernel whose class rule polls
// (llk_matmul_forms.h matmul_kernel_ttsync_block); a run-time key (0xFF) keeps the class rule
template <std::uint32_t in0_cb, std::uint32_t in1_cb, std::uint32_t ct, std::uint32_t rt, std::uint32_t kt>
constexpr bool matmul_fe_ttsync_cell() {
#if defined(UCK_CHLKC_UNPACK)
    if constexpr (
        in0_cb < sizeof(unpack_src_format) && in1_cb < sizeof(unpack_src_format) && ct != 0xFFu && rt != 0xFFu &&
        kt != 0xFFu) {
#ifdef PACKER_L1_ACC
        constexpr bool l1_acc = true;
#else
        constexpr bool l1_acc = false;
#endif
        return matmul_kernel_ttsync_block(
            MATH_FIDELITY,
            DST_SYNC_MODE == DstSync::SyncFull,
            DST_ACCUM_MODE,
            unpack_src_format,
            static_cast<DataFormat>(unpack_src_format[in0_cb]),
            static_cast<DataFormat>(unpack_src_format[in1_cb]),
            rt,
            ct,
            kt,
            MM_THROTTLE,
            l1_acc);
    }
#endif
    return false;
}

// matmul_fe_ttsync_cell of a one-tile block for a kt given as an argument (constant cbs)
template <std::uint32_t in0_cb, std::uint32_t in1_cb>
constexpr bool matmul_fe_ttsync_cell_k(const std::uint32_t kt) {
#if defined(UCK_CHLKC_UNPACK)
    if constexpr (in0_cb < sizeof(unpack_src_format) && in1_cb < sizeof(unpack_src_format)) {
#ifdef PACKER_L1_ACC
        constexpr bool l1_acc = true;
#else
        constexpr bool l1_acc = false;
#endif
        return matmul_kernel_ttsync_block(
            MATH_FIDELITY,
            DST_SYNC_MODE == DstSync::SyncFull,
            DST_ACCUM_MODE,
            unpack_src_format,
            static_cast<DataFormat>(unpack_src_format[in0_cb]),
            static_cast<DataFormat>(unpack_src_format[in1_cb]),
            1,
            1,
            kt,
            MM_THROTTLE,
            l1_acc);
    }
#endif
    return false;
}

// A call of constant cbs and run-time ct or rt: the least kt from which a one-tile block of every kt up to 64 (or of
// the constant kt) is a ttsync cell (matmul_fe_ttsync_cell), so the call can pick the ttsync unpack on such a block at
// run time; 0 for none. Not in a kernel bound by the kernel config buffer (both unpack bodies), nor where the polling
// call holds its operand by run time (reuse_dir 3, which the ttsync call does not run).
template <std::uint32_t in0_cb, std::uint32_t in1_cb, std::uint32_t kt>
constexpr std::uint32_t matmul_fe_rt_ttsync_kt() {
#if defined(UCK_CHLKC_UNPACK)
    if constexpr (
        !LLK_SIZE_MODE && in0_cb < sizeof(unpack_src_format) && in1_cb < sizeof(unpack_src_format) &&
        (MATH_FIDELITY != MathFidelity::LoFi ||
         (DST_SYNC_MODE == DstSync::SyncHalf ? !matmul_kernel_lofi_mix(unpack_src_format)
                                             : !llk_matmul_kernel_mixes_32bit<MM_THROTTLE != 0>()))) {
        if constexpr (kt != 0xFFu) {
            return matmul_fe_ttsync_cell<in0_cb, in1_cb, 1, 1, kt>() ? kt : 0;
        } else {
            std::uint32_t least = 0;
            for (std::uint32_t k = 64; k >= 2; k--) {
                if (!matmul_fe_ttsync_cell_k<in0_cb, in1_cb>(k)) {
                    break;
                }
                least = k;
            }
            return least;
        }
    }
#endif
    return 0;
}

// MATMUL_FE_CELL_MARK's value: 3 and tag 5 where the call's cell runs the ttsync unpack, 4 and tag 5 where a call of
// run-time ct or rt may run it on a one-tile block (matmul_fe_rt_ttsync_kt), else 1 (tag 6, unread)
template <std::uint32_t in0_cb, std::uint32_t in1_cb, std::uint32_t ct, std::uint32_t rt, std::uint32_t kt, bool ok>
struct matmul_fe_cell_mark {
    static constexpr bool cell = ok && matmul_fe_ttsync_cell<in0_cb, in1_cb, ct, rt, kt>();
    static constexpr bool rt_cell =
        ok && !cell && (ct == 0xFFu || rt == 0xFFu) && matmul_fe_rt_ttsync_kt<in0_cb, in1_cb, kt>() != 0;
    static constexpr std::size_t value =
        sizeof(matmul_fe_mark::set < (cell || rt_cell) ? 5 : 6 >) + (cell ? 2 : (rt_cell ? 3 : 0));
};

// the unpack_auto_ttsync of a plain call: the kernel's class rule (MATMUL_FE_MARK 1 or 2, one value per kernel, and tag 8 read at the
// end of the unit), or 3 where the call's cell runs the ttsync unpack in a polling class (MATMUL_FE_CELL_MARK)
template <std::size_t mark>
constexpr std::uint32_t matmul_fe_unpack_ttsync() {
#if defined(UCK_CHLKC_UNPACK)
    constexpr bool runtime_cut =
        matmul_fe_hifi3_wide() && !matmul_fe_mark::is_used_d<8, std::integral_constant<std::size_t, mark>>(0);
    return mark == 3 ? 3 : ((mark == 1 && llk_unpack_AB_matmul_kernel_ttsync() && !runtime_cut) ? 1 : 0);
#else
    return 0;
#endif
}

// Whether an init can key the forms (MATMUL_FE_KDEC's want): run-time ct or rt, constant cbs and kt, and a forms key (llk_matmul_forms.h
// matmul_fe_form_out over the kernel's CB formats). kt is a function argument: no class is instantiated per kt value
template <std::uint32_t in0_cb, std::uint32_t in1_cb, std::uint32_t ct, std::uint32_t rt>
constexpr bool matmul_fe_keyable(const std::uint32_t kt) {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_UNPACK)
    if constexpr (in0_cb < sizeof(unpack_src_format) && in1_cb < sizeof(unpack_src_format) && (ct == 0xFFu || rt == 0xFFu)) {
        return kt != 0xFFu && matmul_fe_form_out(
                                  MATH_FIDELITY,
                                  DST_SYNC_MODE == DstSync::SyncFull,
                                  DST_ACCUM_MODE,
                                  kt,
                                  static_cast<DataFormat>(unpack_src_format[in0_cb]),
                                  static_cast<DataFormat>(unpack_src_format[in1_cb]),
                                  unpack_src_format,
                                  MM_THROTTLE) != DataFormat::Invalid;
    }
#endif
    return false;
}

// the keyed form: the forms table's class of 5 k steps or more into Float16_b (matmul_fe_form_out), for the init and every call
template <bool fkey>
constexpr std::uint32_t matmul_fe_kt_key() {
    return fkey ? 5 : 0;
}
template <bool fkey>
constexpr DataFormat matmul_fe_out_key() {
    return fkey ? DataFormat::Float16_b : DataFormat::Invalid;
}

// a call that names no cbs (the experimental 2_0 call, whose init is not a front-end init) never runs the keyed forms
template <bool fkey>
constexpr bool matmul_fe_unkeyed() {
    static_assert(!fkey, "a front-end init keyed the matmul forms of a kernel whose 2_0 calls cannot follow it");
    return false;
}

// mark: MATMUL_FE_MARK() from the unpack and pack macros; fkey: the kernel's forms key decision (MATMUL_FE_KDEC)
template <std::uint32_t in0_cb, std::uint32_t in1_cb, std::uint32_t ct, std::uint32_t rt, std::size_t mark = 0, bool fkey = false>
ALWI void matmul_block_init_fe(
    const std::uint32_t call_line,
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    const std::uint32_t transpose = 0,
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1) {
    // a kernel bound by the kernel config buffer runs main's direction and math
    constexpr DataFormat f = LLK_SIZE_MODE ? DataFormat::Invalid : matmul_fe_phase_format<in0_cb, in1_cb>();
    constexpr std::uint32_t AUTO = LLK_SIZE_MODE ? 0 : MATMUL_REUSE_DIR_AUTO;
    matmul_block_init<
        false,
        matmul_fe_unpack_ttsync<mark>(),
        matmul_fe_kt_key<fkey>(),
        DataFormat::Invalid,
        0,
        f,
        f,
        AUTO,
        false,
        0,
        matmul_fe_key<ct, rt>(rt),
        matmul_fe_key<ct, rt>(ct),
        0,
        matmul_fe_out_key<fkey>()>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim, call_line);
}

template <
    std::uint32_t in0_cb,
    std::uint32_t in1_cb,
    std::uint32_t ct,
    std::uint32_t rt,
    std::size_t mark = 0,
    bool fkey = false,
    std::uint32_t kt = 0xFFu>
ALWI void matmul_block_fe(
    const std::uint32_t call_line,
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim) {
    // a kernel bound by the kernel config buffer runs main's direction and math
    constexpr DataFormat f = LLK_SIZE_MODE ? DataFormat::Invalid : matmul_fe_phase_format<in0_cb, in1_cb>();
    constexpr std::uint32_t AUTO = LLK_SIZE_MODE ? 0 : MATMUL_REUSE_DIR_AUTO;
#if defined(UCK_CHLKC_UNPACK)
    if constexpr (mark == 4) {
        if (ct_dim == 1 && rt_dim == 1 && kt_dim >= matmul_fe_rt_ttsync_kt<in0_cb, in1_cb, kt>()) {
            matmul_block<
                false,
                3,
                matmul_fe_kt_key<fkey>(),
                DataFormat::Invalid,
                0,
                f,
                f,
                AUTO,
                false,
                0,
                matmul_fe_key<ct, rt>(rt),
                matmul_fe_key<ct, rt>(ct),
                0,
                matmul_fe_out_key<fkey>()>(
                in0_cb_id,
                in1_cb_id,
                in0_tile_index,
                in1_tile_index,
                idst,
                transpose,
                ct_dim,
                rt_dim,
                kt_dim,
                call_line);
            return;
        }
    }
#endif
    matmul_block<
        false,
        matmul_fe_unpack_ttsync<mark>(),
        matmul_fe_kt_key<fkey>(),
        DataFormat::Invalid,
        0,
        f,
        f,
        AUTO,
        false,
        0,
        matmul_fe_key<ct, rt>(rt),
        matmul_fe_key<ct, rt>(ct),
        0,
        matmul_fe_out_key<fkey>()>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, idst, transpose, ct_dim, rt_dim, kt_dim, call_line);
}
#endif

}  // namespace ckernel

#ifdef ARCH_BLACKHOLE

// the cb ids, ct and rt of a plain call where they are constant expressions (pk-state's front-end dispatch); a call
// with template arguments is not a macro call
#define MATMUL_FE_C(x) (__builtin_constant_p(::ckernel::matmul_fe_cb(x)) ? ::ckernel::matmul_fe_cb(x) : 0xFFu)
#define MATMUL_FE_NARG(...) MATMUL_FE_NARG_(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define MATMUL_FE_NARG_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define MATMUL_FE_CAT(a, b) MATMUL_FE_CAT_(a, b)
#define MATMUL_FE_CAT_(a, b) a##b
// ct and rt key only the math init; elsewhere naming them in the dispatch moved the register allocation of run-time
// dims
#if defined(UCK_CHLKC_MATH)
// every math init and call carries the kernel's forms key decision (MATMUL_FE_KDEC); an init of run-time ct or rt and constant kt may key it
#define matmul_block_init(...) MATMUL_FE_CAT(MATMUL_FE_INIT_, MATMUL_FE_NARG(__VA_ARGS__))(__VA_ARGS__)
#define MATMUL_FE_INIT_2(a, b) \
    matmul_block_init_fe<MATMUL_FE_C(a), MATMUL_FE_C(b), 1, 1, MATMUL_FE_SHAPE(1, 1), MATMUL_FE_KDEC(a, false)>(__builtin_LINE(), a, b)
#define MATMUL_FE_INIT_3(a, b, tr) \
    matmul_block_init_fe<MATMUL_FE_C(a), MATMUL_FE_C(b), 1, 1, MATMUL_FE_SHAPE(1, 1), MATMUL_FE_KDEC(a, false)>(__builtin_LINE(), a, b, tr)
#define MATMUL_FE_INIT_4(a, b, tr, ct)                                                                         \
    matmul_block_init_fe<MATMUL_FE_C(a), MATMUL_FE_C(b), MATMUL_FE_C(ct), 1, MATMUL_FE_SHAPE(ct, 1), MATMUL_FE_KDEC(a, false)>( \
        __builtin_LINE(), a, b, tr, ct)
#define MATMUL_FE_INIT_5(a, b, tr, ct, rt)                                                                              \
    matmul_block_init_fe<MATMUL_FE_C(a), MATMUL_FE_C(b), MATMUL_FE_C(ct), MATMUL_FE_C(rt), MATMUL_FE_SHAPE(ct, rt), MATMUL_FE_KDEC(a, false)>( \
        __builtin_LINE(), a, b, tr, ct, rt)
#define MATMUL_FE_INIT_6(a, b, tr, ct, rt, kt)                                                               \
    matmul_block_init_fe<                                                                                    \
        MATMUL_FE_C(a),                                                                                      \
        MATMUL_FE_C(b),                                                                                      \
        MATMUL_FE_C(ct),                                                                                     \
        MATMUL_FE_C(rt),                                                                                     \
        MATMUL_FE_SHAPE(ct, rt) + MATMUL_FE_KT1(kt),                                                         \
        MATMUL_FE_KDEC(                                                                                      \
            a,                                                                                               \
            (::ckernel::matmul_fe_keyable<MATMUL_FE_C(a), MATMUL_FE_C(b), MATMUL_FE_C(ct), MATMUL_FE_C(rt)>( \
                MATMUL_FE_CK(kt))))>(__builtin_LINE(), a, b, tr, ct, rt, kt)
#define MATMUL_FE_INIT_7(a, b, tr, ct, rt, kt, line)                                                         \
    matmul_block_init_fe<                                                                                    \
        MATMUL_FE_C(a),                                                                                      \
        MATMUL_FE_C(b),                                                                                      \
        MATMUL_FE_C(ct),                                                                                     \
        MATMUL_FE_C(rt),                                                                                     \
        MATMUL_FE_SHAPE(ct, rt) + MATMUL_FE_KT1(kt),                                                         \
        MATMUL_FE_KDEC(                                                                                      \
            a,                                                                                               \
            (::ckernel::matmul_fe_keyable<MATMUL_FE_C(a), MATMUL_FE_C(b), MATMUL_FE_C(ct), MATMUL_FE_C(rt)>( \
                MATMUL_FE_CK(kt))))>(line, a, b, tr, ct, rt, kt)
#define matmul_block(...) MATMUL_FE_CAT(MATMUL_FE_BLOCK_, MATMUL_FE_NARG(__VA_ARGS__))(__VA_ARGS__)
// the experimental 2_0 call (no transpose argument)
#define MATMUL_FE_BLOCK_8(a, ...) \
    matmul_block_fe<0xFFu, 0xFFu, 0xFFu, 0xFFu, 0, ::ckernel::matmul_fe_unkeyed<MATMUL_FE_KDEC(a, false)>()>(__builtin_LINE(), a, __VA_ARGS__)
#define MATMUL_FE_BLOCK_9(a, b, i0, i1, d, tr, ct, rt, kt) \
    matmul_block_fe<                                       \
        MATMUL_FE_C(a),                                    \
        MATMUL_FE_C(b),                                    \
        MATMUL_FE_C(ct),                                   \
        MATMUL_FE_C(rt),                                   \
        MATMUL_FE_SHAPE(ct, rt) + MATMUL_FE_KT1(kt),       \
        MATMUL_FE_KDEC(a, false)>(__builtin_LINE(), a, b, i0, i1, d, tr, ct, rt, kt)
#define MATMUL_FE_BLOCK_10(a, b, i0, i1, d, tr, ct, rt, kt, line) \
    matmul_block_fe<                                              \
        MATMUL_FE_C(a),                                           \
        MATMUL_FE_C(b),                                           \
        MATMUL_FE_C(ct),                                          \
        MATMUL_FE_C(rt),                                          \
        MATMUL_FE_SHAPE(ct, rt) + MATMUL_FE_KT1(kt),              \
        MATMUL_FE_KDEC(a, false)>(line, a, b, i0, i1, d, tr, ct, rt, kt)
#else
// the unpack and pack macros: the class rule's mark and the shape mark, and on a call that names ct, rt and kt the cell mark; every
// init and call carries the kernel's forms key decision as the math macros do
#define matmul_block_init(...) MATMUL_FE_CAT(MATMUL_FE_UINIT_, MATMUL_FE_NARG(__VA_ARGS__))(__VA_ARGS__)
#define MATMUL_FE_UINIT(a, b, ct, rt) \
    matmul_block_init_fe<MATMUL_FE_C(a), MATMUL_FE_C(b), 0xFFu, 0xFFu, MATMUL_FE_MARK() + MATMUL_FE_SHAPE(ct, rt) + MATMUL_FE_PACK_SHAPE(ct, rt), \
                         MATMUL_FE_KDEC(a, false)>
#if defined(UCK_CHLKC_PACK)
// a dim that is a constant expression, 0xFF otherwise (by value, as MATMUL_FE_SHAPE)
#define MATMUL_FE_V(x)                                                                   \
    (__builtin_constant_p(::ckernel::matmul_fe_cb(::ckernel::matmul_fe_dim(x))) ? ::ckernel::matmul_fe_cb(::ckernel::matmul_fe_dim(x)) \
                                                                                 : 0xFFu)
// 0, marking the init's block for the pack (cpack_common.h llk_pack_shape_set): 5 constant and not tall, 6 constant and tall,
// 7 (read by nothing) run time
#define MATMUL_FE_PACK_SHAPE(ct, rt)                                                                   \
    (0 * sizeof(::ckernel::packer::llk_pack_shape_set<                                                \
             (MATMUL_FE_V(ct) == 0xFFu || MATMUL_FE_V(rt) == 0xFFu) ? 7                                \
             : (MATMUL_FE_V(rt) >= 4 && MATMUL_FE_V(ct) <= 2)       ? 6                                \
                                                                    : 5>) +                            \
     0 * sizeof(::ckernel::packer::llk_pack_shape_set<(MATMUL_FE_V(rt) == 1u && MATMUL_FE_V(ct) >= 2u && MATMUL_FE_V(ct) != 0xFFu) ? 9 : 10>))
// 0, marking a call with a constant kt of 1 for the pack (11)
#define MATMUL_FE_PACK_KT1(kt) \
    (0 * sizeof(::ckernel::packer::llk_pack_shape_set < (MATMUL_FE_CK(kt) == 1u) ? 11 : 12 >))
#else
#define MATMUL_FE_PACK_SHAPE(ct, rt) 0
#define MATMUL_FE_PACK_KT1(kt) 0
#endif
#define MATMUL_FE_UCELL(a, b, ct, rt, kt)                                                                      \
    matmul_block_init_fe<                                                                                      \
        MATMUL_FE_C(a),                                                                                        \
        MATMUL_FE_C(b),                                                                                        \
        0xFFu,                                                                                                 \
        0xFFu,                                                                                                 \
        MATMUL_FE_CELL_MARK(a, b, ct, rt, kt) + MATMUL_FE_SHAPE(ct, rt) + MATMUL_FE_PACK_SHAPE(ct, rt) +       \
            MATMUL_FE_PACK_KT1(kt),                                                                            \
        MATMUL_FE_KDEC(                                                                                        \
            a,                                                                                                 \
            (::ckernel::matmul_fe_keyable<MATMUL_FE_C(a), MATMUL_FE_C(b), MATMUL_FE_CK(ct), MATMUL_FE_CK(rt)>( \
                MATMUL_FE_CK(kt))))>
#define MATMUL_FE_UINIT_2(a, b) MATMUL_FE_UINIT(a, b, 1, 1)(__builtin_LINE(), a, b)
#define MATMUL_FE_UINIT_3(a, b, tr) MATMUL_FE_UINIT(a, b, 1, 1)(__builtin_LINE(), a, b, tr)
#define MATMUL_FE_UINIT_4(a, b, tr, ct) MATMUL_FE_UINIT(a, b, ct, 1)(__builtin_LINE(), a, b, tr, ct)
#define MATMUL_FE_UINIT_5(a, b, tr, ct, rt) MATMUL_FE_UINIT(a, b, ct, rt)(__builtin_LINE(), a, b, tr, ct, rt)
#define MATMUL_FE_UINIT_6(a, b, tr, ct, rt, kt) MATMUL_FE_UCELL(a, b, ct, rt, kt)(__builtin_LINE(), a, b, tr, ct, rt, kt)
#define MATMUL_FE_UINIT_7(a, b, tr, ct, rt, kt, line) MATMUL_FE_UCELL(a, b, ct, rt, kt)(line, a, b, tr, ct, rt, kt)
#define matmul_block(...) MATMUL_FE_CAT(MATMUL_FE_UBLOCK_, MATMUL_FE_NARG(__VA_ARGS__))(__VA_ARGS__)
#define MATMUL_FE_UBCELL(a, b, ct, rt, kt)                                                        \
    matmul_block_fe<                                                                              \
        MATMUL_FE_C(a),                                                                           \
        MATMUL_FE_C(b),                                                                           \
        0xFFu,                                                                                    \
        0xFFu,                                                                                    \
        MATMUL_FE_CELL_MARK(a, b, ct, rt, kt) + MATMUL_FE_SHAPE(ct, rt) + MATMUL_FE_PACK_KT1(kt), \
        MATMUL_FE_KDEC(a, false),                                                                 \
        MATMUL_FE_CK(kt)>
// the experimental 2_0 call (no transpose argument)
#define MATMUL_FE_UBLOCK_8(a, b, i0, i1, d, ct, rt, kt)                      \
    matmul_block_fe<                                                         \
        MATMUL_FE_C(a),                                                      \
        MATMUL_FE_C(b),                                                      \
        0xFFu,                                                               \
        0xFFu,                                                               \
        MATMUL_FE_MARK() + MATMUL_FE_SHAPE(ct, rt) + MATMUL_FE_PACK_KT1(kt), \
        ::ckernel::matmul_fe_unkeyed<MATMUL_FE_KDEC(a, false)>()>(__builtin_LINE(), a, b, i0, i1, d, ct, rt, kt)
#define MATMUL_FE_UBLOCK_9(a, b, i0, i1, d, tr, ct, rt, kt) \
    MATMUL_FE_UBCELL(a, b, ct, rt, kt)(__builtin_LINE(), a, b, i0, i1, d, tr, ct, rt, kt)
#define MATMUL_FE_UBLOCK_10(a, b, i0, i1, d, tr, ct, rt, kt, line) \
    MATMUL_FE_UBCELL(a, b, ct, rt, kt)(line, a, b, i0, i1, d, tr, ct, rt, kt)
#endif
#endif
