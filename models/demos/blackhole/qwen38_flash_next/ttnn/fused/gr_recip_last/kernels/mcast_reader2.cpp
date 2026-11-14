// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gr_read/kernels/mcast_reader.cpp with a SECOND multicast receive: the down worker of ttnn/fused/gr_recip_last
// reserves the u-row CB and the recip CB, optionally generates a zero tile, streams up to two TILE tensors (its weight
// column), then pushes the u row once `u_row_senders` producers signalled and the recip tiles once `recip_senders`
// did.  Compile-time args: 0 u-row cb, 1 u-row tiles, 2 u-row senders, 3 u-row semaphore id, 4 recip cb, 5 recip
//   tiles, 6 recip senders, 7 recip semaphore id, 8 num_streams (0..2), 9-10 stream cbs, 11 zero-tile cb (0xFF none),
//   12.. two TensorAccessorArgs sets (unused slots repeat a used tensor's).
// Runtime args: per stream 7 (addr, outer, inner, first, inner_stride, outer_stride, batch).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

constexpr uint32_t u_row_cb = get_compile_time_arg_val(0);
constexpr uint32_t u_row_tiles = get_compile_time_arg_val(1);
constexpr uint32_t u_row_senders = get_compile_time_arg_val(2);
constexpr uint32_t u_row_semaphore_id = get_compile_time_arg_val(3);
constexpr uint32_t recip_cb = get_compile_time_arg_val(4);
constexpr uint32_t recip_tiles = get_compile_time_arg_val(5);
constexpr uint32_t recip_senders = get_compile_time_arg_val(6);
constexpr uint32_t recip_semaphore_id = get_compile_time_arg_val(7);
constexpr uint32_t num_streams = get_compile_time_arg_val(8);
constexpr uint32_t zero_cb = get_compile_time_arg_val(11);
constexpr uint32_t accessor_args_base = 12;
constexpr uint32_t stream_runtime_args = 7;

template <typename Args>
FORCE_INLINE void read_stream(const Args& args, uint32_t cb_id, uint32_t first_arg) {
    const uint32_t addr = get_arg_val<uint32_t>(first_arg);
    const uint32_t outer = get_arg_val<uint32_t>(first_arg + 1);
    const uint32_t inner = get_arg_val<uint32_t>(first_arg + 2);
    const uint32_t first = get_arg_val<uint32_t>(first_arg + 3);
    const uint32_t inner_stride = get_arg_val<uint32_t>(first_arg + 4);
    const uint32_t outer_stride = get_arg_val<uint32_t>(first_arg + 5);
    const uint32_t batch = get_arg_val<uint32_t>(first_arg + 6);
    const auto accessor = TensorAccessor(args, addr);
    const uint32_t tile_bytes = get_tile_size(cb_id);
    const Noc noc;
    DataflowBuffer stream_cb(cb_id);
    uint32_t pending = 0;
    for (uint32_t o = 0; o < outer; ++o) {
        for (uint32_t i = 0; i < inner; ++i) {
            if (pending == 0) {
                stream_cb.reserve_back(batch);
            }
            noc.async_read(
                accessor,
                stream_cb,
                tile_bytes,
                {.page_id = first + o * outer_stride + i * inner_stride},
                {.offset_bytes = pending * tile_bytes});
            if (++pending == batch) {
                noc.async_read_barrier();
                stream_cb.push_back(batch);
                pending = 0;
            }
        }
    }
}

void kernel_main() {
    DataflowBuffer u_row(u_row_cb);
    DataflowBuffer recip(recip_cb);
    Semaphore<> u_row_landed(u_row_semaphore_id);
    Semaphore<> recip_landed(recip_semaphore_id);
    u_row.reserve_back(u_row_tiles);
    recip.reserve_back(recip_tiles);
    if constexpr (zero_cb != 0xFF) {
        dataflow_kernel_lib::prepare_zero_tile<zero_cb>();
    }
    {
        constexpr auto args0 = TensorAccessorArgs<accessor_args_base>();
        constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
        if constexpr (num_streams > 0) {
            read_stream(args0, get_compile_time_arg_val(9), 0);
        }
        if constexpr (num_streams > 1) {
            read_stream(args1, get_compile_time_arg_val(10), stream_runtime_args);
        }
    }
    {
        u_row_landed.wait(u_row_senders);
        u_row.push_back(u_row_tiles);
    }
    {
        recip_landed.wait(recip_senders);
        recip.push_back(recip_tiles);
    }
}
