// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Request-slot gather/scatter of complete interleaved state tiles. No numeric
// conversion: FP32 recurrence and BF16 convolution state retain their bits.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t pages = get_compile_time_arg_val(0);
    constexpr uint32_t bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr bool store = get_compile_time_arg_val(3);
    constexpr auto ba = TensorAccessorArgs<4>();
    constexpr auto wa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto bank = TensorAccessor(ba, get_arg_val<uint32_t>(0));
    const auto work = TensorAccessor(wa, get_arg_val<uint32_t>(1));
    const auto slot = TensorAccessor(sa, get_arg_val<uint32_t>(2));
    const auto pos = TensorAccessor(pa, get_arg_val<uint32_t>(3));
    const uint32_t lane = get_arg_val<uint32_t>(4);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    Noc noc;
    noc.async_read(slot, CoreLocalMem<uint32_t>(base + bytes), 128, {.page_id = lane}, {});
    noc.async_read(pos, CoreLocalMem<uint32_t>(base + bytes + 128), 128, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t request = words[bytes / 4];
    const uint32_t position = words[bytes / 4 + 32];
    ASSERT(request < slots);
    if (!store && position == 0) {
        for (uint32_t j = 0; j < bytes / 4; ++j) {
            words[j] = 0;
        }
    }
    for (uint32_t tile = 0; tile < pages; ++tile) {
        if constexpr (store) {
            noc.async_read(work, CoreLocalMem<uint32_t>(base), bytes, {.page_id = lane * pages + tile}, {});
            noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base), bank, bytes, {}, {.page_id = request * pages + tile});
        } else {
            if (position != 0) {
                noc.async_read(bank, CoreLocalMem<uint32_t>(base), bytes, {.page_id = request * pages + tile}, {});
                noc.async_read_barrier();
            }
            noc.async_write(CoreLocalMem<uint32_t>(base), work, bytes, {}, {.page_id = lane * pages + tile});
        }
        noc.async_write_barrier();
    }
}
