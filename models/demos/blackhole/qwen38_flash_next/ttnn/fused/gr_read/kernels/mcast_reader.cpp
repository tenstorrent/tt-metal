// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// A consumer core's reader: reserves the multicast CB, waits until `senders` producers have raised the semaphore
// (their tiles have landed), pushes the block, optionally generates a zero tile, then streams up to two TILE tensors
// (first + o * outer_stride + i * inner_stride, `batch` at a time) as reader.cpp does.
// Compile-time args: 0 receive cb, 1 receive tiles, 2 senders, 3 semaphore id, 4 num_streams (0..2), 5-6 stream cbs,
//   7 zero-tile cb (0xFF none), 8.. two TensorAccessorArgs sets (unused slots repeat a used tensor's).
// Runtime args: per stream 7 (addr, outer, inner, first, inner_stride, outer_stride, batch).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

constexpr uint32_t receive_cb = get_compile_time_arg_val(0);
constexpr uint32_t receive_tiles = get_compile_time_arg_val(1);
constexpr uint32_t senders = get_compile_time_arg_val(2);
constexpr uint32_t semaphore_id = get_compile_time_arg_val(3);
constexpr uint32_t num_streams = get_compile_time_arg_val(4);
constexpr uint32_t zero_cb = get_compile_time_arg_val(7);
constexpr uint32_t accessor_args_base = 8;
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
    DataflowBuffer received(receive_cb);
    Semaphore<> landed(semaphore_id);
    received.reserve_back(receive_tiles);
    if constexpr (zero_cb != 0xFF) {
        dataflow_kernel_lib::prepare_zero_tile<zero_cb>();
    }
    // The streams' CBs hold their whole stream, so they prefetch while the producers still compute.
    {
        constexpr auto args0 = TensorAccessorArgs<accessor_args_base>();
        constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
        if constexpr (num_streams > 0) {
            read_stream(args0, get_compile_time_arg_val(5), 0);
        }
        if constexpr (num_streams > 1) {
            read_stream(args1, get_compile_time_arg_val(6), stream_runtime_args);
        }
    }
    {
        landed.wait(senders);
        received.push_back(receive_tiles);
    }
}
