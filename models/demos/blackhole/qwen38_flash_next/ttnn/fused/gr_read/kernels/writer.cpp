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
#include "../../kernels/zones.h"

constexpr uint32_t NUM_STREAMS = get_compile_time_arg_val(0);
constexpr uint32_t ACCESSOR_BASE = 4;
constexpr uint32_t STREAM_RT_ARGS = 5;

template <typename Args>
FORCE_INLINE void write_stream(const Args& args, uint32_t cb, uint32_t rt) {
    const uint32_t addr = get_arg_val<uint32_t>(rt);
    const uint32_t count = get_arg_val<uint32_t>(rt + 1);
    const uint32_t first = get_arg_val<uint32_t>(rt + 2);
    const uint32_t stride = get_arg_val<uint32_t>(rt + 3);
    const uint32_t batch = get_arg_val<uint32_t>(rt + 4);
    const auto accessor = TensorAccessor(args, addr);
    const uint32_t tile_bytes = get_tile_size(cb);
    Noc noc;
    DataflowBuffer dfb(cb);
    for (uint32_t done = 0; done < count; done += batch) {
        dfb.wait_front(batch);
        for (uint32_t t = 0; t < batch; ++t) {
            noc.async_write(
                dfb, accessor, tile_bytes, {.offset_bytes = t * tile_bytes}, {.page_id = first + (done + t) * stride});
        }
        noc.async_write_barrier();
        dfb.pop_front(batch);
    }
}

void kernel_main() {
    constexpr auto args0 = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    {
        FUSED_ZONE("fz_gr_wr_main");
        write_stream(args0, get_compile_time_arg_val(1), 0);
        if constexpr (NUM_STREAMS > 1) {
            write_stream(args1, get_compile_time_arg_val(2), STREAM_RT_ARGS);
        }
        if constexpr (NUM_STREAMS > 2) {
            write_stream(args2, get_compile_time_arg_val(3), 2 * STREAM_RT_ARGS);
        }
    }
}
