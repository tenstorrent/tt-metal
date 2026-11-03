// SPDX-License-Identifier: Apache-2.0
// Gather every lane's complete pooled-key prefix from the paged pooled cache (eight 256-byte
// BF16 rows per physical P32 page) into dense lane-major [B,1,T,128] tiles, in one program.
// The 32 row reads of a tile are issued together and waited on once. Partial tiles are zeroed;
// tiles beyond a lane's prefix are left untouched (the score mask discards those columns).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

static uint32_t face_index(uint32_t row, uint32_t col) {
    return (row / 16) * 512 + (col / 16) * 256 + (row % 16) * 16 + col % 16;
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0), S = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2), physical_pages = get_compile_time_arg_val(3);
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr uint32_t tablebytes = C / 32 * 4, lane_tiles = T / 32 * 4;
    constexpr auto sa = TensorAccessorArgs<5>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(sa, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1));
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(2));
    const auto table = TensorAccessor(ta, get_arg_val<uint32_t>(3));
    const uint32_t worker = get_arg_val<uint32_t>(4), workers = get_arg_val<uint32_t>(5);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    const uint32_t rowsaddr = base + 2048, posaddr = rowsaddr + 2048, tableaddr = posaddr + 128;
    auto tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    auto rows = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rowsaddr);
    auto pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(posaddr);
    auto pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tableaddr);
    Noc noc;
    for (uint32_t b = 0; b < B; ++b) {
        noc.async_read(positions, CoreLocalMem<uint32_t>(posaddr), 128, {.page_id = b}, {});
        noc.async_read_barrier();
        const uint32_t start = pos[0];
        ASSERT(start < C && S <= C - start);
        const uint32_t complete = (start + S) / 4;
        if (complete == 0) {
            continue;
        }
        const uint32_t jobs = (complete + 31) / 32 * 4;
        if (worker >= jobs) {
            continue;
        }
        // DMA reads begin at the aligned page-row base. Round the live prefix to 32 bytes; never
        // read beyond the full-capacity row.
        const uint32_t live_pages = (start + S + 31) / 32;
        const uint32_t live_bytes = ((live_pages * 4 + 31) / 32) * 32;
        noc.async_read(
            table,
            CoreLocalMem<uint32_t>(tableaddr),
            live_bytes < tablebytes ? live_bytes : tablebytes,
            {.page_id = b},
            {});
        noc.async_read_barrier();
        for (uint32_t job = worker; job < jobs; job += workers) {
            const uint32_t first = (job / 4) * 32, col = job % 4;
            const uint32_t count = complete - first < 32 ? complete - first : 32;
            for (uint32_t r = 0; r < count; ++r) {
                const uint32_t group = first + r;
                const uint32_t physical = pages[group / 8];
                ASSERT(physical < physical_pages);
                noc.async_read(
                    source,
                    CoreLocalMem<uint32_t>(rowsaddr + r * 64),
                    64,
                    {.page_id = physical * 8 + group % 8, .offset_bytes = col * 64},
                    {});
            }
            for (uint32_t i = 0; i < 1024; ++i) {
                tile[i] = 0;
            }
            noc.async_read_barrier();
            for (uint32_t r = 0; r < count; ++r) {
                for (uint32_t c = 0; c < 32; ++c) {
                    tile[face_index(r, c)] = rows[r * 32 + c];
                }
            }
            noc.async_write(CoreLocalMem<uint32_t>(base), output, 2048, {}, {.page_id = b * lane_tiles + job});
            noc.async_write_barrier();
        }
    }
}
