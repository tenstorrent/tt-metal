// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    // out tensor args
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_next_subblock_stride_w = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_next_subblock_stride_h = get_arg_val<uint32_t>(5);

    // out subblock args
    uint32_t out_subblock_w = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(10);

    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    experimental::Noc noc;
    constexpr uint32_t cb_id_out0 = 16;
    experimental::CircularBuffer cb_out0(cb_id_out0);

    // single-tile
    uint32_t single_tile_size_bytes = cb_out0.get_tile_size();

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(out_args, out_tensor_addr, single_tile_size_bytes);

    bool one_time_profile = true;
    uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
    for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
        for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
            uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

            cb_out0.wait_front(out_subblock_tile_count);

            for (uint32_t h = 0; h < out_subblock_h; h++) {
                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                for (uint32_t w = 0; w < out_subblock_w; w++) {
                    noc.async_write(
                        cb_out0,
                        s,
                        single_tile_size_bytes,
                        {.offset_bytes = ((h * out_subblock_w + w) * single_tile_size_bytes)},
                        {.page_id = out_tensor_tile_id}
                    );

                    out_tensor_tile_id += out_tensor_stride_w;
                }
                out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
            }

            noc.async_write_barrier();
            cb_out0.pop_front(out_subblock_tile_count);
            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
        }
        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
    }
}
