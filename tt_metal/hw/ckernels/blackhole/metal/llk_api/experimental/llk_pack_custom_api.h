// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_pack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK CUSTOM (ADC-mutexed single-tile pack)
 *
 * Same single-tile pack as llk_pack_init / llk_pack (PackMode::Default only),
 * with every SETADC* the pack thread issues taken under mutex::THREAD2_ADC.
 * Needed when another
 * thread borrows the pack thread's address counters (ADCs): two SETADC*
 * reaching the MISC unit in the same cycle corrupt each other's counters.
 *
 * Usage:
 *   1. llk_pack_init_mutex_ADC - once, in place of llk_pack_init
 *   2. llk_pack_mutex_ADC      - every tile, in place of llk_pack
 *
 * The two are a mandatory pair: the init guards one SETADC, each pack guards
 * two more. llk_matmul_pack / llk_pack_block / llk_pack_untilize /
 * llk_pack_rows are NOT covered - they issue their own unguarded SETADC*, so
 * a kernel on this path must pack exclusively through llk_pack_mutex_ADC.
 *
 * Uses llk_pack.h (_llk_pack_init_ / _llk_pack_) as the low-level implementation.
 *************************************************************************/

// Program the packer for single-tile packing, taking the SETADCXX under the mutex.
// Pair with llk_pack_mutex_ADC, never with plain llk_pack.
template <bool zero_output = false, bool skip_addrmod_config = false, bool skip_packer_strides = false>
inline void llk_pack_init_mutex_ADC(
    const std::uint32_t pack_output, std::uint32_t num_tiles = 1, const std::uint32_t input_operand = 0) {
    SAN_HOOK(unsupported());
    const std::uint32_t output_id = get_output_id(pack_output);
    // 8-bit datums (Int8, UInt8, Fp8_e4m3, Lf8) do not require the Blackhole tilize workaround.
    const bool is_input_8bit_format = IS_8BIT_FORMAT(static_cast<std::uint32_t>(unpack_src_format[input_operand]));

    _llk_pack_init_<PackMode::Default, zero_output, skip_addrmod_config, skip_packer_strides, true /*mutex_ADC*/>(
        pack_src_format[output_id],
        get_output_face_r_dim(output_id),
        get_output_tile_c_dim(output_id),
        get_output_num_faces(output_id),
        num_tiles,
        is_input_8bit_format);
}

// Pack one tile from dest, taking both per-tile SETADCs (dest-write address and
// Z-counter reset) under the mutex. Requires llk_pack_init_mutex_ADC.
template <bool is_fp32_dest_acc_en, bool out_of_order_output = false>
inline void llk_pack_mutex_ADC(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    SAN_HOOK(unsupported());
    const std::uint8_t output_id = get_output_id(output);

    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default, true /*mutex_ADC*/>(
        tile_index, get_output_tile_address<out_of_order_output, PackMode::Default>(output_id, output_tile_index));
}
