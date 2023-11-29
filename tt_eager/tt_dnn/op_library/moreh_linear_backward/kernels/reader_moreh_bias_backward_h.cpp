// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/moreh_linear_backward/kernels/utils.hpp"

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t B1B2Ht = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t Wt_per_core = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);
    const uint32_t mask_h = get_arg_val<uint32_t>(5);
    const uint32_t mask_w = get_arg_val<uint32_t>(6);
    const bool do_mask_h = get_arg_val<uint32_t>(7) == 1;
    const bool do_mask_w = get_arg_val<uint32_t>(8) == 1;

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_scaler = 1;
    constexpr uint32_t cb_id_mask_h_w = 2;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_scaler, scaler.u);

    if (do_mask_h || do_mask_w) {
        generate_mask_h_w(cb_id_mask_h_w, mask_h, mask_w);
    }

    uint32_t l1_write_addr_in0;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    auto src0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    constexpr uint32_t onetile = 1;
    for (uint32_t wt = 0; wt < Wt_per_core; ++wt) {
        uint32_t read_tile_id = start_id + wt;
        for (uint32_t b1b2ht = 0; b1b2ht < B1B2Ht; ++b1b2ht) {
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(read_tile_id, s0, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            read_tile_id += Wt;
        }
    }
}
