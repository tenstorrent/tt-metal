// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
//#include "tt_eager/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
//#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    const uint32_t out_addr  = get_arg_val<uint32_t>(0);
    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CB::c_out0;

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;
    generate_reduce_scaler(cb_identity_scale_in, 0x40404040); // 3

    uint32_t out_tile_id = 0;
    // Wait for compute to deliver output chunk
    cb_wait_front(cb_out, 1);

    uint32_t l1_read_addr = get_read_ptr(cb_out);
    noc_async_write_tile(0, out_writer, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
