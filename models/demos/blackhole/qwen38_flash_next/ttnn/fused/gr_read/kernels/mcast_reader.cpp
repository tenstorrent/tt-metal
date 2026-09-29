// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// A consumer core's reader: reserves the multicast CB, waits until `senders` producers have raised the semaphore
// (their tiles have landed), pushes the block, optionally generates a zero tile, then streams up to two TILE tensors
// (first + o * outer_stride + i * inner_stride, `batch` at a time) as reader.cpp does.
// Compile-time args: 0 recv cb, 1 recv tiles, 2 senders, 3 semaphore id, 4 num_streams (0..2), 5-6 stream cbs,
//   7 zero-tile cb (0xFF none), 8.. two TensorAccessorArgs sets (unused slots repeat a used tensor's).
// Runtime args: per stream 7 (addr, outer, inner, first, inner_stride, outer_stride, batch).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "../../kernels/zones.h"

constexpr uint32_t RECV_CB = get_compile_time_arg_val(0);
constexpr uint32_t RECV_TILES = get_compile_time_arg_val(1);
constexpr uint32_t SENDERS = get_compile_time_arg_val(2);
constexpr uint32_t SEM_ID = get_compile_time_arg_val(3);
constexpr uint32_t NUM_STREAMS = get_compile_time_arg_val(4);
constexpr uint32_t ZERO_CB = get_compile_time_arg_val(7);
constexpr uint32_t ACCESSOR_BASE = 8;
constexpr uint32_t STREAM_RT_ARGS = 7;

template <typename Args>
FORCE_INLINE void read_stream(const Args& args, uint32_t cb, uint32_t rt) {
    const uint32_t addr = get_arg_val<uint32_t>(rt);
    const uint32_t outer = get_arg_val<uint32_t>(rt + 1);
    const uint32_t inner = get_arg_val<uint32_t>(rt + 2);
    const uint32_t first = get_arg_val<uint32_t>(rt + 3);
    const uint32_t inner_stride = get_arg_val<uint32_t>(rt + 4);
    const uint32_t outer_stride = get_arg_val<uint32_t>(rt + 5);
    const uint32_t batch = get_arg_val<uint32_t>(rt + 6);
    const auto accessor = TensorAccessor(args, addr);
    const uint32_t tile_bytes = get_tile_size(cb);
    Noc noc;
    DataflowBuffer dfb(cb);
    uint32_t pending = 0;
    for (uint32_t o = 0; o < outer; ++o) {
        for (uint32_t i = 0; i < inner; ++i) {
            if (pending == 0) {
                dfb.reserve_back(batch);
            }
            noc.async_read(
                accessor,
                dfb,
                tile_bytes,
                {.page_id = first + o * outer_stride + i * inner_stride},
                {.offset_bytes = pending * tile_bytes});
            if (++pending == batch) {
                noc.async_read_barrier();
                dfb.push_back(batch);
                pending = 0;
            }
        }
    }
}

void kernel_main() {
    DataflowBuffer recv(RECV_CB);
    Semaphore<> sem(SEM_ID);
    recv.reserve_back(RECV_TILES);
    if constexpr (ZERO_CB != 0xFF) {
        dataflow_kernel_lib::prepare_zero_tile<ZERO_CB>();
    }
    // The streams' CBs hold their whole stream, so they prefetch while the producers still compute.
    {
        FUSED_ZONE("fz_gr_mcr_streams");
        constexpr auto args0 = TensorAccessorArgs<ACCESSOR_BASE>();
        constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
        if constexpr (NUM_STREAMS > 0) {
            read_stream(args0, get_compile_time_arg_val(5), 0);
        }
        if constexpr (NUM_STREAMS > 1) {
            read_stream(args1, get_compile_time_arg_val(6), STREAM_RT_ARGS);
        }
    }
    {
        FUSED_ZONE("fz_gr_mcr_wait");
        sem.wait(SENDERS);
        recv.push_back(RECV_TILES);
    }
}
