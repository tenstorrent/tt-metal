// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>
#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
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
        } else if constexpr (half_f32 || row_mop || unpack_auto_ttsync != 0 || stream_z != 0 || kt_per_section != 0) {
            return 0;
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
    bool multi_k = false,
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
    std::uint32_t num_k = 1,
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
        if constexpr (multi_k) {
            for (std::uint32_t k = 0; k < num_k; k++) {
                MATH((llk_math_matmul<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE_MAX, /*num_faces=*/4, false, true, reuse_dir>(idst, ct_dim, rt_dim, held_a)));
            }
        } else {
            MATH((llk_math_matmul<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE_MAX, /*num_faces=*/4, false, true, reuse_dir>(idst, ct_dim, rt_dim, held_a)));
        }
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
        if constexpr (multi_k) {
            for (std::uint32_t k = 0; k < num_k; k++) {
                MATH((llk_math_matmul<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE, /*num_faces=*/4, row_mop, tile_mop, reuse_dir, row_end_fold>(
                    idst, ct_dim, rt_dim, held_a)));
            }
        } else {
            MATH((llk_math_matmul<matmul_elided_fidelity<in0_format, in1_format, out_format>(), MM_THROTTLE, /*num_faces=*/4, row_mop, tile_mop, reuse_dir, row_end_fold>(idst, ct_dim, rt_dim, held_a)));
        }
    }
#endif
}
#endif

#ifdef ARCH_BLACKHOLE
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
    UNPACK((llk_unpack_AB_matmul_init<unpack_ttsync, llk_dir, stream_z>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<
          matmul_elided_fidelity<in0_format, in1_format, out_format>(),
          MM_THROTTLE,
          row_mop_on,
          llk_math_matmul_tile_mop<row_mop_on, kt_key, form_out>(),
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
    // Dynamic throttling is only available on Blackhole architecture
    MATH((matmul_block_math_dynamic_throttle<
          row_mop_on,
          llk_math_matmul_tile_mop<row_mop_on, kt_key, form_out>(),
          in1_order,
          false,
          in0_format,
          in1_format,
          out_format,
          llk_dir,
          row_end_fold,
          llk_math_matmul_half_end_fold<row_mop_on, matmul_elided_fidelity<in0_format, in1_format, out_format>(), kt_key, out_format, in0_format, in1_format, block_rt, block_ct, block_kt>(),
          block_ct,
          block_rt>(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim, 1, held_a)));
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
 * for a free context, the same value in matmul_block; the kernel must define MATMUL_UNPACK_TTSYNC to a constant that
 * is true when any of its calls passes true.
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
    std::uint32_t block_kt = 0>
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
    constexpr DataFormat form_out = row_mop_form != 0 ? matmul_forced_form_key(row_mop_form) : (keyed ? out_format : DataFormat::Invalid);
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
    std::uint32_t block_kt = 0>
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
    constexpr DataFormat form_out = row_mop_form != 0 ? matmul_forced_form_key(row_mop_form) : (keyed ? out_format : DataFormat::Invalid);
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

// clang-format off
/**
 * matmul_block for num_k consecutive inner-dimension steps in one call: step k multiplies the rt_dim x 1 tiles of in0 at
 * in0_tile_index + k (rows kt_dim apart) by the 1 x ct_dim tiles of in1 at in1_tile_index + k * in1_k_stride and
 * accumulates into DST, as num_k matmul_block calls with the indices advanced by 1 and in1_k_stride. Blackhole runs the
 * k loop inside the unpack and math calls; other architectures call matmul_block per step.
 * The template parameters are those of matmul_block_init and matmul_block; unpack_auto_ttsync must be false. The unpack
 * rows are always the hoisted form.
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
    std::uint32_t block_kt = 0>
ALWI void matmul_block_k(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    std::uint32_t num_k,
    std::uint32_t in1_k_stride,
    std::uint32_t call_line = __builtin_LINE()) {
#ifdef ARCH_BLACKHOLE
    constexpr std::uint32_t kt_key = llk_matmul_form_key<kt_per_section, block_rt, block_ct, block_kt, in0_format, in1_format, out_format, MM_THROTTLE>();
    constexpr char row_mop_form = llk_matmul_row_mop_form<row_mop, kt_per_section, block_rt, block_ct, block_kt, in0_format, in1_format, out_format>();
    constexpr bool row_mop_on =
        row_mop && row_mop_form == 0 &&
        !llk_matmul_main_cell<block_rt, block_ct, block_kt, in0_format, in1_format, out_format>() &&
        !llk_matmul_row_mop_off<block_rt, block_ct, block_kt, in0_format, in1_format, out_format>();
    constexpr DataFormat form_out = row_mop_form != 0 ? matmul_forced_form_key(row_mop_form) : out_format;
#else
    constexpr std::uint32_t kt_key = kt_per_section;
#endif
#ifdef ARCH_BLACKHOLE
    static_assert(!unpack_auto_ttsync, "matmul_block_k: Auto TTSync rows are not supported");
    // the k loop runs main's direction, so the init with the same template arguments must resolve to it
    static_assert(
        matmul_api_reuse_dir<reuse_dir, in0_format, in1_format, out_format, block_rt, block_ct, kt_per_section, row_mop, unpack_auto_ttsync, stream_z>() == 0,
        "matmul_block_k: init and call with reuse_dir 0");
    LLK_SAN_FUNCTION();
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul_k<stream_z>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim, num_k, in1_k_stride)));
    MATH((matmul_block_math_dynamic_throttle<
        row_mop_on,
        llk_math_matmul_tile_mop<row_mop_on, kt_key, form_out>(),
        in1_order,
        true,
        in0_format,
        in1_format,
        out_format,
        0,
        row_end_fold,
        llk_math_matmul_half_end_fold<row_mop_on, matmul_elided_fidelity<in0_format, in1_format, out_format>(), kt_key, out_format, in0_format, in1_format, block_rt, block_ct, block_kt>(), block_ct, block_rt>(
        in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim, num_k)));
#else
    for (std::uint32_t k = 0; k < num_k; k++) {
        matmul_block<row_mop, unpack_auto_ttsync, kt_per_section, out_format, in1_order, in0_format, in1_format, 0, false, 0, block_rt, block_ct, block_kt>(
            in0_cb_id,
            in1_cb_id,
            in0_tile_index + k,
            in1_tile_index + k * in1_k_stride,
            idst,
            transpose,
            ct_dim,
            rt_dim,
            kt_dim,
            call_line);
    }
#endif
}

#ifdef ARCH_BLACKHOLE
// A plain call goes through the macros at the end of this file with the cb ids that are constant where it is written;
// each choice is an if constexpr at instantiation, so a call it does not apply to keeps its instantiation.

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

template <std::uint32_t in0_cb, std::uint32_t in1_cb>
ALWI void matmul_block_init_fe(
    const std::uint32_t call_line,
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    const std::uint32_t transpose = 0,
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1) {
    constexpr DataFormat f = matmul_fe_phase_format<in0_cb, in1_cb>();
    matmul_block_init<false, 0, 0, DataFormat::Invalid, 0, f, f>(
        in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim, call_line);
}

template <std::uint32_t in0_cb, std::uint32_t in1_cb>
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
    constexpr DataFormat f = matmul_fe_phase_format<in0_cb, in1_cb>();
    matmul_block<false, 0, 0, DataFormat::Invalid, 0, f, f>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, idst, transpose, ct_dim, rt_dim, kt_dim, call_line);
}
#endif

}  // namespace ckernel

#ifdef ARCH_BLACKHOLE
// the cb ids of a plain call where they are constant expressions (pk-state's front-end dispatch); a call with template
// arguments is not a macro call
#define MATMUL_FE_CB(cb) (__builtin_constant_p(cb) ? ::ckernel::matmul_fe_cb(cb) : 0xFFu)
#define matmul_block_init(in0_cb_id, in1_cb_id, ...)                        \
    matmul_block_init_fe<MATMUL_FE_CB(in0_cb_id), MATMUL_FE_CB(in1_cb_id)>( \
        __builtin_LINE(), in0_cb_id, in1_cb_id, ##__VA_ARGS__)
#define matmul_block(in0_cb_id, in1_cb_id, ...)                        \
    matmul_block_fe<MATMUL_FE_CB(in0_cb_id), MATMUL_FE_CB(in1_cb_id)>( \
        __builtin_LINE(), in0_cb_id, in1_cb_id, __VA_ARGS__)
#endif
