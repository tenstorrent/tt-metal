// SPDX-License-Identifier: Apache-2.0
// Completed BF16 index groups use eight 256-byte rows per physical P32 page.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

static uint32_t face_index(uint32_t row, uint32_t col) {
    return (row / 16) * 512 + (col / 16) * 256 + (row % 16) * 16 + col % 16;
}

void kernel_main() {
    constexpr bool gather = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1), S = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3), physical_pages = get_compile_time_arg_val(4);
    constexpr uint32_t lane = get_compile_time_arg_val(5), St = (S + 31) / 32;
    constexpr uint32_t tablebytes = C / 32 * 4;
    constexpr auto sa = TensorAccessorArgs<6>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(sa, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1));
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(2));
    const auto table = TensorAccessor(ta, get_arg_val<uint32_t>(3));
    const uint32_t begin = get_arg_val<uint32_t>(4), count = get_arg_val<uint32_t>(5);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr(), rowaddr = base + 2048;
    const uint32_t posaddr = rowaddr + 128, tableaddr = posaddr + 128;
    auto tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    auto row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rowaddr);
    auto pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(posaddr);
    auto pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tableaddr);
    Noc noc;
    uint32_t loaded = 0xffffffff, start = 0;
    const uint32_t stride = get_arg_val<uint32_t>(6);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t job = begin + item * stride;
        const uint32_t b = gather ? lane : job / 4;
        const uint32_t col = job % 4;
        ASSERT(b < B);
        if (b != loaded) {
            noc.async_read(positions, CoreLocalMem<uint32_t>(posaddr), 128, {.page_id = b}, {});
            noc.async_read_barrier();
            start = pos[0];
            loaded = b;
            ASSERT(start < C && S <= C - start);
            if constexpr (gather) {
                if ((job / 4) * 32 >= (start + S) / 4) {
                    // Jobs are ordered by logical key tile. All remaining
                    // tiles are outside the completed prefix for this lane.
                    // The score mask overwrites these columns before TopK.
                    break;
                }
            } else {
                if (start / 4 == (start + S) / 4) {
                    // No completed pooled row to commit.
                    continue;
                }
            }
            // DMA reads begin at the aligned page-row base. Round the live
            // prefix to 32 bytes; never read beyond the full-capacity row.
            const uint32_t live_pages = (start + S + 31) / 32;
            const uint32_t live_bytes = ((live_pages * 4 + 31) / 32) * 32;
            noc.async_read(
                table,
                CoreLocalMem<uint32_t>(tableaddr),
                live_bytes < tablebytes ? live_bytes : tablebytes,
                {.page_id = b},
                {});
            noc.async_read_barrier();
        }
        if constexpr (!gather) {
            // Distinct workers own disjoint 64-byte segments of each RM row;
            // there is no tile read/modify/write race between physical pages.
            for (uint32_t s = 0; s < S; ++s) {
                const uint32_t absolute = start + s;
                if (absolute % 4 != 3) {
                    continue;
                }
                const uint32_t physical = pages[absolute / 32];
                ASSERT(physical < physical_pages);
                const uint32_t dest = physical * 8 + (absolute % 32) / 4;
                const uint32_t src = (b * St + s / 32) * 4 + col;
                noc.async_read(source, CoreLocalMem<uint32_t>(base), 2048, {.page_id = src}, {});
                noc.async_read_barrier();
                for (uint32_t c = 0; c < 32; ++c) {
                    row[c] = tile[face_index(s % 32, c)];
                }
                noc.async_write(
                    CoreLocalMem<uint32_t>(rowaddr), output, 64, {}, {.page_id = dest, .offset_bytes = col * 64});
                noc.async_write_barrier();
            }
        } else {
            // A dense tile spans four virtual pages. Define every completed
            // row and zero the partial tile; whole suffix tiles are scratch.
            // Indexer scores are independent across key columns, and the
            // downstream score mask overwrites all incomplete columns with
            // -inf before TopK, including NaNs left by another lane/request.
            const uint32_t first = (job / 4) * 32, complete = (start + S) / 4;
            if (first >= complete) {
                break;
            }
            for (uint32_t i = 0; i < 1024; ++i) {
                tile[i] = 0;
            }
            for (uint32_t r = 0; r < 32 && first + r < complete; ++r) {
                const uint32_t group = first + r;
                const uint32_t physical = pages[group / 8];
                ASSERT(physical < physical_pages);
                const uint32_t src = physical * 8 + group % 8;
                noc.async_read(
                    source, CoreLocalMem<uint32_t>(rowaddr), 64, {.page_id = src, .offset_bytes = col * 64}, {});
                noc.async_read_barrier();
                for (uint32_t c = 0; c < 32; ++c) {
                    tile[face_index(r, c)] = row[c];
                }
            }
            noc.async_write(CoreLocalMem<uint32_t>(base), output, 2048, {}, {.page_id = job});
            noc.async_write_barrier();
        }
    }
}
