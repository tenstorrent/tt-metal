// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // out tensor args
    uint32_t out_tensor_start_tile_id = get_arg(args::out_tensor_start_tile_id);
    uint32_t out_tensor_stride_w = get_arg(args::out_tensor_stride_w);
    uint32_t out_tensor_stride_h = get_arg(args::out_tensor_stride_h);
    uint32_t out_tensor_next_subblock_stride_w = get_arg(args::out_tensor_next_subblock_stride_w);
    uint32_t out_tensor_next_subblock_stride_h = get_arg(args::out_tensor_next_subblock_stride_h);

    // out subblock args
    uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    uint32_t out_subblock_tile_count = get_arg(args::out_subblock_tile_count);
    uint32_t out_num_subblocks_w = get_arg(args::out_num_subblocks_w);
    uint32_t out_num_subblocks_h = get_arg(args::out_num_subblocks_h);

    // batch args
    uint32_t MtNt = get_arg(args::MtNt);  // if 0
    uint32_t batch = get_arg(args::batch);

    DataflowBuffer dfb_out0(dfb::out0);

    // single-tile
    const uint32_t single_tile_size_bytes = dfb_out0.get_entry_size();

    // The output tensor is bound to this kernel by name in the host code.
    const auto s = TensorAccessor(tensor::out);

    bool one_time_profile = true;
    for (uint32_t b = 0; b < batch; b++) {
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                dfb_out0.wait_front(out_subblock_tile_count);
                uint32_t l1_read_addr = dfb_out0.get_read_ptr();

                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        noc_async_write_page(out_tensor_tile_id, s, l1_read_addr);
                        l1_read_addr += single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();  // This will wait until the write is done. As
                                            // an alternative, noc_async_write_flushed()
                                            // can be faster because it waits until the
                                            // write request is sent. In that case, you
                                            // have to use noc_async_write_barrier() at
                                            // least once at the end of data movement kernel
                                            // to make sure all writes are done.
                dfb_out0.pop_front(out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
    }
}
