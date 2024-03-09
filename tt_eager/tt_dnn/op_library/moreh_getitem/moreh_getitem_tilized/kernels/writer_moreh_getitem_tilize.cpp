// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/moreh_getitem/moreh_getitem_tilized/kernels/common.hpp"

void kernel_main() {
    uint32_t i = 0;
    // buffers
    uint32_t dst_addr = get_arg_val<uint32_t>(i++);

    // output
    uint32_t output_size_c_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t output_size_h_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t output_size_w_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t output_noc_id_stride_n = get_arg_val<uint32_t>(i++);
    uint32_t output_noc_id_stride_c = get_arg_val<uint32_t>(i++);
    uint32_t output_noc_id_stride_h = get_arg_val<uint32_t>(i++);
    uint32_t output_num_stick_width = get_arg_val<uint32_t>(i++);

    // etc
    uint32_t start_id = get_arg_val<uint32_t>(i++);
    uint32_t num_sticks = get_arg_val<uint32_t>(i++);
    uint32_t stick_size = get_arg_val<uint32_t>(i++);
    uint32_t element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_out = tt::CB::c_in0;

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGen<dst_is_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = 1024 * element_size,
    };

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++ i) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        uint32_t stick_idx = i;

        Idx4d stick_index_4d = get_stick_indices(stick_idx, output_size_c_without_padding, output_size_h_without_padding, output_num_stick_width);
        Idx4d tile_index_4d = get_tile_indices(stick_index_4d);

        uint32_t noc_id =   tile_index_4d.n * output_noc_id_stride_n +
                            tile_index_4d.c * output_noc_id_stride_c +
                            tile_index_4d.h * output_noc_id_stride_h +
                            tile_index_4d.w;

        uint32_t noc_offset = get_noc_offset_in_tile(stick_index_4d.h, stick_index_4d.w, tile_index_4d.h, element_size);

        uint64_t dst_noc_addr = get_noc_addr(noc_id, s0, noc_offset);

        noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
    }
}
