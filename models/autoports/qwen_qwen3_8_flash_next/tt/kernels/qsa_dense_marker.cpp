// SPDX-License-Identifier: Apache-2.0
// One core, after the incremental pooled-key gather and its consumers: record [slot, completed
// rows] for the lane, so the next step's gather rebuilds only rows the buffer does not hold.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0), lane = get_compile_time_arg_val(1);
    constexpr auto pa = TensorAccessorArgs<2>();
    constexpr auto la = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(0));
    const auto slots = TensorAccessor(la, get_arg_val<uint32_t>(1));
    const auto marker = TensorAccessor(ma, get_arg_val<uint32_t>(2));
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    Noc noc;
    noc.async_read(positions, CoreLocalMem<uint32_t>(base), 128, {.page_id = lane}, {});
    noc.async_read(slots, CoreLocalMem<uint32_t>(base + 128), 128, {.page_id = lane}, {});
    noc.async_read_barrier();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto mark = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 256);
    mark[0] = words[32];                 // slot
    mark[1] = (words[0] + S) / 4;        // completed pooled rows
    for (uint32_t i = 2; i < 8; ++i) mark[i] = 0;
    noc.async_write(CoreLocalMem<uint32_t>(base + 256), marker, 32, {}, {.page_id = 0});
    noc.async_write_barrier();
}
