
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0), S = get_compile_time_arg_val(1),
                       K = get_compile_time_arg_val(2), W = get_compile_time_arg_val(3);
    constexpr auto sa = TensorAccessorArgs<4>();
    constexpr auto pa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto scores = TensorAccessor(sa, get_arg_val<uint32_t>(0));
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(1));
    const auto bound = TensorAccessor(ba, get_arg_val<uint32_t>(2));
    const uint32_t worker = get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    Noc noc;
    auto pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + 128);
    if (worker < B) {
        noc.async_read(positions, CoreLocalMem<uint32_t>(base), 128, {.page_id = worker}, {});
        noc.async_read_barrier();
        invalidate_l1_cache();
        const uint32_t end = (pos[0] + S) / 4;
        const uint32_t last = end > 512 ? end : 512;
        ASSERT(last <= K);
        auto result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 640);
        for (uint32_t i = 0; i < 8; ++i) {
            result[i] = 0;
        }
        result[0] = last;
        noc.async_write(CoreLocalMem<uint32_t>(base + 640), bound, 32, {}, {.page_id = worker});
        noc.async_write_barrier();
    }
    constexpr uint32_t chunks = (K + 255) / 256;
    uint32_t loaded = 0xffffffff, start = 0;
    for (uint32_t job = worker; job < B * S * chunks; job += W) {
        const uint32_t row = job / chunks, b = row / S, col = (job % chunks) * 256;
        if (loaded != b) {
            noc.async_read(positions, CoreLocalMem<uint32_t>(base), 128, {.page_id = b}, {});
            noc.async_read_barrier();
            start = pos[0];
            loaded = b;
        }
        const uint32_t complete = (start + row % S + 1) / 4;
        const uint32_t count = K - col < 256 ? K - col : 256;
        if (col + count <= complete) {
            continue;
        }
        uint32_t first = 0;
        if (col < complete) {
            noc.async_read(
                scores, CoreLocalMem<uint32_t>(base + 128), count * 2, {.page_id = row, .offset_bytes = col * 2}, {});
            noc.async_read_barrier();
            first = complete - col;
        }
        for (uint32_t i = first; i < count; ++i) {
            values[i] = 0xff80;
        }
        noc.async_write(
            CoreLocalMem<uint32_t>(base + 128), scores, count * 2, {}, {.page_id = row, .offset_bytes = col * 2});
        noc.async_write_barrier();
    }
}
