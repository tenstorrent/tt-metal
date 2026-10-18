// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t seq = get_compile_time_arg_val(0);
    constexpr auto xa = TensorAccessorArgs<1>();
    constexpr auto va = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto valid = TensorAccessor(va, get_arg_val<uint32_t>(1));
    const uint32_t lane = get_arg_val<uint32_t>(2);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    Noc noc;
    noc.async_read(valid, CoreLocalMem<uint32_t>(base), 4, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t length = words[0];
    ASSERT(length <= seq);
    for (uint32_t j = 0; j < 5120; ++j) {
        words[j] = 0;
    }
    for (uint32_t t = length; t < seq; ++t) {
        noc.async_write(CoreLocalMem<uint32_t>(base), x, 20480, {}, {.page_id = lane * seq + t});
        noc.async_write_barrier();
    }
}
