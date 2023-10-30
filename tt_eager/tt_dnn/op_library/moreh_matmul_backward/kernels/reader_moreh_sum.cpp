// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug_print.h"
#include "tt_eager/tt_dnn/op_library/moreh_layernorm_backward/kernels/utils.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_src_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_dst_tiles = get_arg_val<uint32_t>(2);
    const uint32_t read_tile_offset = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);
    const bool b1_batched = (get_arg_val<uint32_t>(5) == 1);
    const uint32_t HtWt = get_arg_val<uint32_t>(6);
    const uint32_t src_B2HtWt = get_arg_val<uint32_t>(7);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_zero = 1;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 0.0f;
    fill_cb_with_value(cb_id_zero, scaler.u);

    uint32_t l1_write_addr_in0;
    uint32_t src_tile_bytes = get_tile_size(cb_id_in0);
    auto src_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src_is_dram> s0 = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};

    for (uint32_t i = start_id; i < start_id + num_dst_tiles; i++) {
        uint32_t read_tile_id = i;
        if (b1_batched) {
            read_tile_id = (i / HtWt * src_B2HtWt) + (i % HtWt);
        }

        for (uint32_t j = 0; j < num_src_tiles; ++j) {
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(read_tile_id, s0, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            read_tile_id += read_tile_offset;
        }
    }
}
