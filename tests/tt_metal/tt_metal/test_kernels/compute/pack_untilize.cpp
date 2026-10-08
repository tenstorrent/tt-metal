// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#ifdef FAST_UNTILIZE
#include "api/compute/experimental/fast_untilize.h"
#endif
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Helper constexpr function to compute num_blocks_per_col
constexpr uint32_t compute_num_blocks_per_col(uint32_t per_core_block_tile_cnt) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;

    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }

    return 1;
}

#ifdef FAST_UNTILIZE_AT_ADDRESS
// fast_untilize_block, with the strided pack issued through llk_pack_fast_untilize_block_strided_at_address.
// Keep in sync with the strided loop in fast_untilize_block (api/compute/experimental/fast_untilize.h).
template <std::uint32_t full_ct_dim>
void fast_untilize_block_at_address(uint32_t icb, uint32_t ocb) {
    static_assert(full_ct_dim > FAST_UNTILIZE_MAX_UNIT_DIM, "the strided pack needs more than one unit");
    std::uint32_t tiles_done = 0;
    constexpr std::uint32_t first_unpack_unit_dim = fast_untilize_next_unit_dim(full_ct_dim);
    [[maybe_unused]] std::uint32_t prev_unpack_unit_dim = first_unpack_unit_dim;
    [[maybe_unused]] std::uint32_t prev_pack_unit_dim = 0;
    while (tiles_done < full_ct_dim) {
        const std::uint32_t unit_dim = fast_untilize_next_unit_dim(full_ct_dim - tiles_done);
        MATH((_llk_math_wait_for_dest_available_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>()));
        UNPACK((llk_unpack_fast_untilize_block<DST_ACCUM_MODE>(icb, tiles_done, unit_dim, prev_unpack_unit_dim)));
        MATH((llk_math_fast_untilize_block<DST_ACCUM_MODE>(/*dst_index=*/0, unit_dim)));
        MATH((_llk_math_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));

        PACK((llk_packer_wait_for_math_done()));
        PACK(({
            const std::uint32_t output_id = get_output_id(ocb);
            constexpr std::uint32_t bytes_per_16B_unit = 16;
            const std::uint32_t address =
                get_output_tile_address</*out_of_order_output=*/true, PackMode::Default>(
                    output_id, /*output_tile_index=*/0) +
                SCALE_DATUM_SIZE(pack_dst_format[output_id], tiles_done * TILE_C_DIM) / bytes_per_16B_unit;
            const std::uint32_t output_row_stride_16B =
                SCALE_DATUM_SIZE(pack_dst_format[output_id], full_ct_dim * TILE_C_DIM) / bytes_per_16B_unit;
            llk_pack_fast_untilize_block_strided_at_address<FAST_UNTILIZE_MAX_UNIT_DIM, full_ct_dim>(
                address, unit_dim, prev_pack_unit_dim, output_row_stride_16B);
        }));
        PACK((_llk_pack_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));
        tiles_done += unit_dim;
    }
    UNPACK(
        (llk_unpack_fast_untilize_restore_unit_dim<DST_ACCUM_MODE>(icb, first_unpack_unit_dim, prev_unpack_unit_dim)));
}
#endif

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    DataflowBuffer dfb_in0(dfb::in);
    DataflowBuffer dfb_out0(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);

#ifndef FAST_UNTILIZE
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(per_core_block_tile_cnt);
    constexpr uint32_t block_ct_dim = per_core_block_tile_cnt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;

    pack_untilize_init<block_ct_dim, full_ct_dim>(dfb::in, dfb::out);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        dfb_out0.reserve_back(full_ct_dim);

        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            dfb_in0.wait_front(block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(dfb::in, 1, dfb::out, b);
            dfb_in0.pop_front(block_ct_dim);
        }
        dfb_out0.push_back(full_ct_dim);
    }

    pack_untilize_uninit(dfb::out);
#else
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;

    fast_untilize_init<full_ct_dim>(dfb::in, dfb::out);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        dfb_in0.wait_front(full_ct_dim);
        dfb_out0.reserve_back(full_ct_dim);

#ifdef FAST_UNTILIZE_AT_ADDRESS
        fast_untilize_block_at_address<full_ct_dim>(dfb::in, dfb::out);
#else
        fast_untilize_block<full_ct_dim>(dfb::in, dfb::out);
#endif

        dfb_in0.pop_front(full_ct_dim);
        dfb_out0.push_back(full_ct_dim);
    }

    fast_untilize_uninit<full_ct_dim>(dfb::out);
#endif
}
