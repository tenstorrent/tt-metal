// SPDX-License-Identifier: Apache-2.0
// One conservative key-prefix bound for all lanes: max over lanes of
// max(512, ceil32(floor((position[b] + seq) / 4))). Same contract as indexer_prefix.cpp.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0), K = get_compile_time_arg_val(1);
    constexpr auto pa = TensorAccessorArgs<2>();
    constexpr auto oa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1));
    const uint32_t seq = get_arg_val<uint32_t>(2);
    ASSERT(seq > 0 && seq <= K * 4);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    Noc noc;
    uint32_t cap = 512;
    for (uint32_t b = 0; b < B; ++b) {
        noc.async_read(positions, CoreLocalMem<uint32_t>(base), 128, {.page_id = b}, {});
        noc.async_read_barrier();
        const uint32_t start = words[0];
        ASSERT(start <= K * 4 - seq);
        const uint32_t rounded = ((start + seq) / 4 + 31) / 32 * 32;
        cap = rounded > cap ? rounded : cap;
    }
    ASSERT(cap <= K);
    for (uint32_t i = 0; i < 8; ++i) {
        words[32 + i] = i == 0 ? cap : 0;
    }
    noc.async_write(CoreLocalMem<uint32_t>(base + 128), output, 32, {}, {.page_id = 0});
    noc.async_write_barrier();
}
