// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writes whole tiles from up to three CBs into TILE tensors: stream s writes its tiles to pages first + t * stride.
// Compile-time args: 0 num_streams, 1-3 cb per stream, 4.. three TensorAccessorArgs sets, chained (unused slots
// repeat a used tensor's).
// Runtime args: per stream 5 (addr, count, first, stride, batch).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t num_streams = get_compile_time_arg_val(0);
constexpr uint32_t accessor_args_base = 4;
constexpr uint32_t stream_runtime_args = 5;

template <typename Args>
FORCE_INLINE void write_stream(const Args& args, uint32_t cb_id, uint32_t first_arg) {
    const uint32_t addr = get_arg_val<uint32_t>(first_arg);
    const uint32_t count = get_arg_val<uint32_t>(first_arg + 1);
    const uint32_t first = get_arg_val<uint32_t>(first_arg + 2);
    const uint32_t stride = get_arg_val<uint32_t>(first_arg + 3);
    const uint32_t batch = get_arg_val<uint32_t>(first_arg + 4);
    const auto accessor = TensorAccessor(args, addr);
    const uint32_t tile_bytes = get_tile_size(cb_id);
    const Noc noc;
    DataflowBuffer stream_cb(cb_id);
    for (uint32_t done = 0; done < count; done += batch) {
        stream_cb.wait_front(batch);
        for (uint32_t t = 0; t < batch; ++t) {
            noc.async_write(
                stream_cb,
                accessor,
                tile_bytes,
                {.offset_bytes = t * tile_bytes},
                {.page_id = first + (done + t) * stride});
        }
        noc.async_write_barrier();
        stream_cb.pop_front(batch);
    }
}

void kernel_main() {
    constexpr auto args0 = TensorAccessorArgs<accessor_args_base>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    {
        write_stream(args0, get_compile_time_arg_val(1), 0);
        if constexpr (num_streams > 1) {
            write_stream(args1, get_compile_time_arg_val(2), stream_runtime_args);
        }
        if constexpr (num_streams > 2) {
            write_stream(args2, get_compile_time_arg_val(3), 2 * stream_runtime_args);
        }
    }
}
