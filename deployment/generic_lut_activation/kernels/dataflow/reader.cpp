// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
// tt-emule does not force-include the dataflow API (no COMPILE_FOR_* defines);
// standard tt-metal dataflow kernels include it explicitly. No-op on silicon/
// craq-sim (already force-included; guard-protected). Keeps TensorAccessorArgs.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

#ifndef TT_ACT_DATAFLOW_PAGE_BATCH
#define TT_ACT_DATAFLOW_PAGE_BATCH 1
#endif
static_assert(TT_ACT_DATAFLOW_PAGE_BATCH == 1 || TT_ACT_DATAFLOW_PAGE_BATCH == 2);

void kernel_main() {
    uint32_t in_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);  // Tile offset for multi-core

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
#ifdef TT_ACT_RUNTIME_TENSOR_SHAPE
    constexpr auto in_args = TensorAccessorArgs<0, 0>();
#else
    constexpr auto in_args = TensorAccessorArgs<0>();
#endif
    const auto in_accessor = TensorAccessor(in_args, in_addr);
    const uint32_t tile_size_bytes = get_local_cb_interface(cb_in).fifo_page_size;
    Noc noc;
    DataflowBuffer input_dfb(cb_in);

#if defined(FUSE_GRAD_MUL) && !defined(FUSE_GRAD_ON_WRITER)
    // Second stream for the fused unary-backward multiply: the compute kernel
    // parks this in DST tile 1 and forms grad * f'(x) in its epilogue, so no
    // separate ttnn.multiply program (and no DRAM round-trip on the f'
    // intermediate) is needed. Runtime arg 3 is the grad buffer address; the
    // accessor args follow the input's in the compile-time arg block.
    uint32_t grad_addr = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_grad = tt::CBIndex::c_1;
    constexpr auto grad_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto grad_accessor = TensorAccessor(grad_args, grad_addr, tile_size_bytes);
    DataflowBuffer grad_dfb(cb_grad);
#endif

    constexpr uint32_t kPagesPerBatch = TT_ACT_DATAFLOW_PAGE_BATCH;
    for (uint32_t i = 0; i < n_tiles; i += kPagesPerBatch) {
        const uint32_t pages = (n_tiles - i < kPagesPerBatch) ? n_tiles - i : kPagesPerBatch;
        input_dfb.reserve_back(pages);
        for (uint32_t page = 0; page < pages; ++page) {
            noc.async_read(
                in_accessor,
                input_dfb,
                tile_size_bytes,
                {.page_id = start_tile_id + i + page},
                {.offset_bytes = page * tile_size_bytes});
        }
#if defined(FUSE_GRAD_MUL) && !defined(FUSE_GRAD_ON_WRITER)
        // Both reads issued before a single barrier so they overlap on the NoC.
        grad_dfb.reserve_back(pages);
        for (uint32_t page = 0; page < pages; ++page) {
            noc.async_read(
                grad_accessor,
                grad_dfb,
                tile_size_bytes,
                {.page_id = start_tile_id + i + page},
                {.offset_bytes = page * tile_size_bytes});
        }
        noc.async_read_barrier();
        input_dfb.push_back(pages);
        grad_dfb.push_back(pages);
#else
        noc.async_read_barrier();
        input_dfb.push_back(pages);
#endif
    }
}
