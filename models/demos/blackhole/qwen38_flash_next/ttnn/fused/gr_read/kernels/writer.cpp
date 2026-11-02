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

#ifdef GR_LIVE_ROWS
// Verify rows (GR_LIVE_ROWS valid rows of the 32-row tile): the rows past them are overwritten with zeros in the L1
// tile before it is written, so the output page keeps today's 32-row contract (dead rows exact zeros) although the
// program computed only the live rows.  GR_LIVE_ZERO_CB is a CB of a core-local zero tile (>= 1024 B).  A BF16 tile:
// face 0 = rows 0..15 of columns 0..15 (32 B a row) at byte 0, face 1 at byte 512, faces 2 and 3 (rows 16..31) at 1024.
FORCE_INLINE void gr_zero_dead_rows(uint32_t tile_addr) {
    constexpr uint32_t FACE = 512, LIVE = GR_LIVE_ROWS * 32;
    DataflowBuffer zero(GR_LIVE_ZERO_CB);
    const uint32_t z = zero.get_read_ptr();
    const uint64_t base = get_noc_addr(my_x[noc_index], my_y[noc_index], tile_addr);
    noc_async_write(z, base + LIVE, FACE - LIVE);
    noc_async_write(z, base + FACE + LIVE, FACE - LIVE);
    noc_async_write(z, base + 2 * FACE, 2 * FACE);
    noc_async_write_barrier();
}
#endif

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
#ifdef GR_LIVE_ROWS
        for (uint32_t t = 0; t < batch; ++t) {
            gr_zero_dead_rows(dfb.get_read_ptr() + t * tile_bytes);
        }
#endif
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
