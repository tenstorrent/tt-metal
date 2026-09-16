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

#ifndef TT_ACT_OUTPUT_CB
#define TT_ACT_OUTPUT_CB 16
#endif

#if !defined(ARCH_BLACKHOLE) && !defined(ARCH_WORMHOLE)
#error "generic activation writer flush pipeline is certified only on Blackhole and Wormhole"
#endif

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);  // Tile offset for multi-core

    constexpr uint32_t cb_out = TT_ACT_OUTPUT_CB;
#ifdef TT_ACT_RUNTIME_TENSOR_SHAPE
    constexpr auto out_args = TensorAccessorArgs<0, 0>();
#else
    constexpr auto out_args = TensorAccessorArgs<0>();
#endif
    const auto out_accessor = TensorAccessor(out_args, out_addr);
    const uint32_t tile_size_bytes = get_local_cb_interface(cb_out).fifo_page_size;
    Noc noc;
    DataflowBuffer output_dfb(cb_out);

#ifdef FUSE_GRAD_ON_WRITER
    // Split-NoC variant of the fused backward multiply: the grad stream is
    // fetched HERE (RISCV_1 / NOC1) instead of in the reader (RISCV_0 / NOC0),
    // so the two NoCs carry one read each plus this kernel's write, rather than
    // NOC0 carrying both reads while NOC1 sits idle until compute produces
    // output. The reader-side variant measured +1.48 us for the third stream;
    // this exists to test how much of that is NOC0 serialization.
    //
    // No deadlock: the writer PRODUCES cb_grad[i] before it CONSUMES cb_out[i],
    // and compute needs cb_grad[i] to produce cb_out[i] -- so the dependency is
    // acyclic and the 2-deep cb_grad lets this run one tile ahead.
    uint32_t grad_addr = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_grad = tt::CBIndex::c_1;
    constexpr auto grad_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto grad_accessor = TensorAccessor(grad_args, grad_addr, tile_size_bytes);
    DataflowBuffer grad_dfb(cb_grad);
#endif

    constexpr uint32_t kPagesPerBatch = TT_ACT_DATAFLOW_PAGE_BATCH;
    for (uint32_t i = 0; i < n_tiles; i += kPagesPerBatch) {
        const uint32_t pages = (n_tiles - i < kPagesPerBatch) ? n_tiles - i : kPagesPerBatch;
#ifdef FUSE_GRAD_ON_WRITER
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
        grad_dfb.push_back(pages);
#endif
        output_dfb.wait_front(pages);
        for (uint32_t page = 0; page < pages; ++page) {
            noc.async_write(
                output_dfb,
                out_accessor,
                tile_size_bytes,
                {.offset_bytes = page * tile_size_bytes},
                {.page_id = start_tile_id + i + page});
        }
        // The CB page may be released as soon as the non-posted write has left
        // its L1 source.  Waiting for the remote DRAM acknowledgement here
        // serializes every tile and prevents the two-page CB from pipelining.
        // Keep completion ownership at kernel exit below.
        noc.async_writes_flushed();
        output_dfb.pop_front(pages);
    }
    noc.async_write_barrier();
}
