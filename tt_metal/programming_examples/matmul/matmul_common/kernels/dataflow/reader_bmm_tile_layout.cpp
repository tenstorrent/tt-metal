// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_start_tile_id = get_arg(args::in0_tensor_start_tile_id);
    uint32_t in0_tensor_stride_w = get_arg(args::in0_tensor_stride_w);
    uint32_t in0_tensor_stride_h = get_arg(args::in0_tensor_stride_h);
    uint32_t in0_tensor_next_block_stride = get_arg(args::in0_tensor_next_block_stride);

    // in0 block args
    uint32_t in0_block_w = get_arg(args::in0_block_w);
    uint32_t in0_block_h = get_arg(args::in0_block_h);
    uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);

    // in1 tensor args
    uint32_t in1_tensor_start_tile_id = get_arg(args::in1_tensor_start_tile_id);
    uint32_t in1_tensor_stride_w = get_arg(args::in1_tensor_stride_w);
    uint32_t in1_tensor_stride_h = get_arg(args::in1_tensor_stride_h);
    uint32_t in1_tensor_next_block_stride = get_arg(args::in1_tensor_next_block_stride);

    // in1 block args
    uint32_t in1_block_w = get_arg(args::in1_block_w);
    uint32_t in1_block_h = get_arg(args::in1_block_h);
    uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);

    // in0/in1 common args
    uint32_t num_blocks = get_arg(args::num_blocks);

    // batch args
    uint32_t MtKt = get_arg(args::MtKt);  // if 0
    uint32_t KtNt = get_arg(args::KtNt);
    uint32_t batch = get_arg(args::batch);
    uint32_t bcast_B = get_arg(args::bcast_B);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    const uint32_t in0_single_tile_size_bytes = dfb_in0.get_entry_size();
    const uint32_t in1_single_tile_size_bytes = dfb_in1.get_entry_size();

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    // The in0 and in1 tensors are bound to this kernel by name in the host code.
    const auto s0 = TensorAccessor(tensor::in0);
    const auto s1 = TensorAccessor(tensor::in1);

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; block++) {
            dfb_in0.reserve_back(in0_block_num_tiles);
            dfb_in1.reserve_back(in1_block_num_tiles);

            l1_write_addr_in0 = dfb_in0.get_write_ptr();
            l1_write_addr_in1 = dfb_in1.get_write_ptr();

            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; h++) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; w++) {
                    noc_async_read_page(in0_tensor_tile_id, s0, l1_write_addr_in0);
                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; h++) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; w++) {
                    noc_async_read_page(in1_tensor_tile_id, s1, l1_write_addr_in1);
                    l1_write_addr_in1 += in1_single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            noc_async_read_barrier();

            dfb_in0.push_back(in0_block_num_tiles);
            dfb_in1.push_back(in1_block_num_tiles);
        }
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }
        in0_tensor_start_tile_id += MtKt;
    }
}
