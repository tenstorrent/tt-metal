// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "data_format_derive.h"
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

// Id-free init helpers. Each 2.0 op init calls the helpers for the threads it uses, then its own
// MOP / addrmod setup. noinline: the inits are ALWI, so an inline body would be copied into every
// init call site.
//
// Each thread tracks which config it last programmed: the address of the noinline body, unique per
// template instantiation. A config is skipped when the tracker already holds its body. nullptr means
// unknown; .bss starts zeroed, so the first config after launch always runs. Anything that writes
// tracked registers outside these bodies (hw_startup, reconfigs, uninits, tilize's unpack init,
// custom_mm full init) must reset that thread's tracker.

namespace ckernel::experimental::detail {
using ConfigFn = void (*)();
inline ConfigFn unpack_config_state = nullptr;
inline ConfigFn math_config_state = nullptr;
inline ConfigFn pack_config_state = nullptr;

template <ConfigFn body>
inline __attribute__((always_inline)) void config_if_changed(ConfigFn& state) {
    if (state != body) {
        body();
        state = body;
    }
}
}  // namespace ckernel::experimental::detail

#ifdef TRISC_UNPACK
#include "llk_unpack_tilize_api.h"  // _llk_unpack_tilize_uninit_ + unpack reconfig

// SrcA/SrcB L1 and register formats via the 2-operand rule, geometry, then clear tilize bits.
// Register formats use infer_unpack_dst_format_2op (Float32 rebias / mixed-width compile error).
// FACE_ROW_MAJOR refreshes Tile_x_dim; the 2.0 reconfig path passes IGNORE and never does.
// Tilize teardown runs first so its canonical X/Z write does not clobber the geometry reconfig,
// and so tileize_mode/haloize/shift_amount are cleared. The following field RMWs leave those bits 0.
template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor DESC_A,
    ckernel::experimental::LLKMemDescriptor DESC_B>
__attribute__((noinline)) void llk_unpack_config_body() {
    constexpr DataFormat A = DESC_A.format;
    constexpr DataFormat B = DESC_B.format;
    constexpr DataFormat reg_a = ckernel::infer_unpack_dst_format_2op<A, B>(is_fp32_dest_acc_en);
    constexpr DataFormat reg_b = ckernel::infer_unpack_dst_format_2op<B, A>(is_fp32_dest_acc_en);
    constexpr std::uint32_t tile_size_a = ckernel::experimental::tile_stride_words(A, DESC_A.shape);
    constexpr std::uint32_t tile_size_b = ckernel::experimental::tile_stride_words(B, DESC_B.shape);

    _llk_unpack_tilize_uninit_(static_cast<std::uint32_t>(reg_a), DESC_A.shape);
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, false>(
        static_cast<std::uint32_t>(A),
        static_cast<std::uint32_t>(reg_a),
        tile_size_a,
        DESC_A.shape.face_r_dim,
        DESC_A.shape.total_num_faces());
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, false>(
        static_cast<std::uint32_t>(B),
        static_cast<std::uint32_t>(reg_b),
        tile_size_b,
        DESC_B.shape.face_r_dim,
        DESC_B.shape.total_num_faces());
}

template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor DESC_A,
    ckernel::experimental::LLKMemDescriptor DESC_B>
inline __attribute__((always_inline)) void llk_unpack_config() {
    using namespace ckernel::experimental::detail;
    config_if_changed<&llk_unpack_config_body<is_fp32_dest_acc_en, DESC_A, DESC_B>>(unpack_config_state);
}
#endif

#ifdef TRISC_MATH
#include "llk_math_common_api.h"

// INT8 enable and both zero-flag format caches, via the combined math reconfig (the srca-then-srcb
// pair leaves INT8 = is_int(srcB)). Tracked.
template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor DESC_A,
    ckernel::experimental::LLKMemDescriptor DESC_B>
__attribute__((noinline)) void llk_math_config_body() {
    constexpr DataFormat reg_a =
        ckernel::infer_unpack_dst_format_2op<DESC_A.format, DESC_B.format>(is_fp32_dest_acc_en);
    constexpr DataFormat reg_b =
        ckernel::infer_unpack_dst_format_2op<DESC_B.format, DESC_A.format>(is_fp32_dest_acc_en);
    _llk_math_reconfig_data_format_<is_fp32_dest_acc_en, false>(
        static_cast<std::uint32_t>(reg_a), static_cast<std::uint32_t>(reg_b));
}

// The zero flag is applied on every init, not tracked here: execute paths (B2D bcast, UInt16 fp32 copy,
// transpose) rewrite it and rely on the next init to re-apply. tt-llk tracks the flag itself, so an
// unchanged value costs one inline compare. Datacopy ops pass datacopy_zero_flag so the copy rule wins:
// preserve bf16 -0 / 16b ints, flush fp8.
template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor DESC_A,
    ckernel::experimental::LLKMemDescriptor DESC_B,
    bool datacopy_zero_flag = false>
inline __attribute__((always_inline)) void llk_math_config() {
    using namespace ckernel::experimental::detail;
    config_if_changed<&llk_math_config_body<is_fp32_dest_acc_en, DESC_A, DESC_B>>(math_config_state);
    if constexpr (datacopy_zero_flag) {
        constexpr DataFormat reg_a =
            ckernel::infer_unpack_dst_format_2op<DESC_A.format, DESC_B.format>(is_fp32_dest_acc_en);
        ckernel::math::_configure_copy_zero_flag_state_(static_cast<std::uint32_t>(reg_a));
    } else {
        ckernel::math::_configure_default_zero_flag_state_();
    }
}
#endif

#ifdef TRISC_PACK
#include "llk_pack_reduce_api.h"                 // _llk_pack_reduce_mask_clear_
#include "experimental/2_0/llk_pack_tile.h"      // llk_pack_reconfig_data_format / llk_pack_init
#include "experimental/2_0/llk_pack_untilize.h"  // llk_pack_untilize_init

// Clear the reduce edge mask, then program Default-mode formats, strides, and MOP.
// reduce_init sets the mask again after this so it survives. Every other init leaves it clear.
// Also undoes pack_untilize: Default strides, addrmods, MOP, and X count overwrite every untilize write.
// The last pack config wins. Relu is never touched: only hw_configure sets it (off), and 2.0 never enables it.
template <bool is_fp32_dest_acc_en, ckernel::experimental::LLKMemDescriptor OUT_DESC>
__attribute__((noinline)) void llk_pack_config_body() {
    _llk_pack_reduce_mask_clear_();
    llk_pack_reconfig_data_format<OUT_DESC, is_fp32_dest_acc_en>();
    llk_pack_init<OUT_DESC, is_fp32_dest_acc_en>();
}

// Mode configs: Default config, then the mode's own pack setup. Tracked as one config, so a repeated
// init of the same mode is skipped and any other pack config switches the packer back.
template <bool is_fp32_dest_acc_en, ckernel::experimental::LLKMemDescriptor OUT_DESC, ReduceDim reduce_dim>
__attribute__((noinline)) void llk_pack_reduce_config_body() {
    llk_pack_config_body<is_fp32_dest_acc_en, OUT_DESC>();
    llk_pack_reduce_mask_config_impl<reduce_dim, PackMode::Default>(OUT_DESC.shape.face_r_dim);
}

template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor OUT_DESC,
    ckernel::experimental::LLKMemDescriptor IN_DESC>
__attribute__((noinline)) void llk_pack_tilize_config_body() {
    llk_pack_config_body<is_fp32_dest_acc_en, OUT_DESC>();
    llk_pack_init<OUT_DESC, IN_DESC, is_fp32_dest_acc_en, PackMode::Tilize>(1 /* num_tiles */);
}

template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor OUT_DESC,
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim,
    bool narrow_row,
    std::uint32_t row_num_datums,
    bool dense>
__attribute__((noinline)) void llk_pack_untilize_config_body() {
    llk_pack_config_body<is_fp32_dest_acc_en, OUT_DESC>();
    llk_pack_untilize_init<
        OUT_DESC,
        is_fp32_dest_acc_en,
        block_ct_dim,
        full_ct_dim,
        narrow_row,
        row_num_datums,
        dense>();
}

template <bool is_fp32_dest_acc_en, ckernel::experimental::LLKMemDescriptor OUT_DESC>
inline __attribute__((always_inline)) void llk_pack_config() {
    using namespace ckernel::experimental::detail;
    config_if_changed<&llk_pack_config_body<is_fp32_dest_acc_en, OUT_DESC>>(pack_config_state);
}

template <bool is_fp32_dest_acc_en, ckernel::experimental::LLKMemDescriptor OUT_DESC, ReduceDim reduce_dim>
inline __attribute__((always_inline)) void llk_pack_reduce_config() {
    using namespace ckernel::experimental::detail;
    config_if_changed<&llk_pack_reduce_config_body<is_fp32_dest_acc_en, OUT_DESC, reduce_dim>>(pack_config_state);
}

template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor OUT_DESC,
    ckernel::experimental::LLKMemDescriptor IN_DESC>
inline __attribute__((always_inline)) void llk_pack_tilize_config() {
    using namespace ckernel::experimental::detail;
    config_if_changed<&llk_pack_tilize_config_body<is_fp32_dest_acc_en, OUT_DESC, IN_DESC>>(pack_config_state);
}

template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor OUT_DESC,
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
inline __attribute__((always_inline)) void llk_pack_untilize_config() {
    using namespace ckernel::experimental::detail;
    config_if_changed<&llk_pack_untilize_config_body<
        is_fp32_dest_acc_en,
        OUT_DESC,
        block_ct_dim,
        full_ct_dim,
        narrow_row,
        row_num_datums,
        dense>>(pack_config_state);
}
#endif
