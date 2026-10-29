// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The GR gate worker's writer with the gather hand-off (QWEN38_GATE_GATHER=1): gr_read/kernels/writer.cpp's one-stream
// write of the block tile into its DRAM page (unchanged), plus: the tile's valid rows (rows [0, ROWS) of faces 0 and 1)
// are NoC-written into the gather transport core's staging slot (chunk 2 j + k at j * 128 + k * 64, 64 B pitch:
// row_gather2's layout) and the transport core's arrival semaphore is stepped once the chunks have landed. Compile-time
// args as writer.cpp (0 num_streams = 1, 1 cb, 2-3 unused, 4.. three accessor sets), then after them: ROWS, SEM_ARRIVE.
// Runtime args: the stream's 5 (addr, count = 1, first = the tile index j, stride, batch), then the transport core's
// NoC x, y and the staging slot's L1 address.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "../../kernels/zones.h"

constexpr uint32_t ACCESSOR_BASE = 4;

void kernel_main() {
    constexpr auto args0 = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    constexpr uint32_t EXTRA = args2.next_compile_time_args_offset();
    constexpr uint32_t ROWS = get_compile_time_arg_val(EXTRA);
    constexpr uint32_t SEM_ARRIVE = get_compile_time_arg_val(EXTRA + 1);
    constexpr uint32_t CB = get_compile_time_arg_val(1);
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t tx = get_arg_val<uint32_t>(5);
    const uint32_t ty = get_arg_val<uint32_t>(6);
    const uint32_t stage = get_arg_val<uint32_t>(7);
    const auto accessor = TensorAccessor(args0, addr);
    const uint32_t tile_bytes = get_tile_size(CB);
    constexpr uint32_t CHUNK = ROWS * 32;  // bf16 rows [0, ROWS) of a face
    Noc noc;
    DataflowBuffer dfb(CB);
    Semaphore<> arrive(SEM_ARRIVE);
    {
        FUSED_ZONE("fz_rg3_wr");
        dfb.wait_front(1);
        noc.async_write(dfb, accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = first});
        const uint32_t src = dfb.get_read_ptr();
        noc_async_write(src, safe_get_noc_addr(tx, ty, stage + (2 * first) * 64, noc_index), CHUNK);
        noc_async_write(
            src + tile_bytes / 4, safe_get_noc_addr(tx, ty, stage + (2 * first + 1) * 64, noc_index), CHUNK);
        noc.async_write_barrier();
        arrive.up(noc, tx, ty, 1);
        noc.async_atomic_barrier();
        dfb.pop_front(1);
    }
}
