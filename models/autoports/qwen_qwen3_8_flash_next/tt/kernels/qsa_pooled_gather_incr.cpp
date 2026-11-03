// SPDX-License-Identifier: Apache-2.0
// INCREMENTAL variant (one-row decode view, per-layer dense buffer): a marker [slot, rows valid]
// written by qsa_dense_marker.cpp after the step says which completed rows the buffer already
// holds for this request slot; only tile rows from the first stale row onward are rebuilt (one
// tile row per step in steady state). A different slot or fewer completed rows than recorded
// rebuilds everything; the generator zeroes the marker on every prefill and reset.
// Decode lane gather (paged_qsa.transfer, seq 1): the lane's completed pooled rows into the
// dense [T/4, 128] BF16 tile buffer. One job is one 32-row tile row: 32 reads of a full
// 256-byte pooled row, then four 32-column tiles assembled with word copies (rows past the
// completed prefix zeroed) and written back. Workers form one GRID_X-wide rectangle and take
// tile rows cyclically. Same output as qsa_pooled_paged.cpp's gather mode.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(2), C = get_compile_time_arg_val(3);
    constexpr uint32_t physical_pages = get_compile_time_arg_val(4), lane = get_compile_time_arg_val(5);
    constexpr uint32_t tablebytes = C / 32 * 4;
    constexpr auto sa = TensorAccessorArgs<6>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto la = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(sa, get_common_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(oa, get_common_arg_val<uint32_t>(1));
    const auto positions = TensorAccessor(pa, get_common_arg_val<uint32_t>(2));
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(3));
    const auto slots = TensorAccessor(la, get_common_arg_val<uint32_t>(4));
    const auto marker = TensorAccessor(ma, get_common_arg_val<uint32_t>(5));
    const uint32_t worker = get_absolute_logical_y() * GRID_X + get_absolute_logical_x();
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr(), staging = base + 4 * 2048;
    const uint32_t posaddr = staging + 32 * 256, slotaddr = posaddr + 128, markaddr = slotaddr + 128;
    const uint32_t tableaddr = markaddr + 64;
    auto pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(posaddr);
    auto pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tableaddr);
    auto tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto staged = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
    Noc noc;
    noc.async_read(positions, CoreLocalMem<uint32_t>(posaddr), 128, {.page_id = lane}, {});
    noc.async_read(slots, CoreLocalMem<uint32_t>(slotaddr), 128, {.page_id = lane}, {});
    noc.async_read(marker, CoreLocalMem<uint32_t>(markaddr), 32, {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t start = pos[0], complete = (start + S) / 4;
    ASSERT(start < C && S <= C - start);
    auto mark = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(markaddr);
    const uint32_t slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slotaddr)[0];
    const uint32_t valid = (mark[0] == slot && mark[1] <= complete) ? mark[1] : 0;
    const uint32_t first_row = valid / 32;
    if ((first_row + worker) * 32 >= complete) {
        return;
    }
    // Only the live prefix of the page table, rounded to 32 bytes.
    const uint32_t live_bytes = (((start + S + 31) / 32 * 4 + 31) / 32) * 32;
    noc.async_read(table, CoreLocalMem<uint32_t>(tableaddr), live_bytes < tablebytes ? live_bytes : tablebytes,
                   {.page_id = lane}, {});
    noc.async_read_barrier();
    for (uint32_t row = first_row + worker; row * 32 < complete; row += WORKERS) {
        const uint32_t first = row * 32, live = complete - first < 32 ? complete - first : 32;
        for (uint32_t r = 0; r < live; ++r) {
            const uint32_t group = first + r;
            const uint32_t physical = pages[group / 8];
            ASSERT(physical < physical_pages);
            noc.async_read(source, CoreLocalMem<uint32_t>(staging + r * 256), 256,
                           {.page_id = physical * 8 + group % 8}, {});
        }
        noc.async_read_barrier();
        // Row r of column tile c: values 32c..32c+31 = staged words [r*64 + 16c, +16), split
        // into the face-row halves (columns 0-15 -> face 0/2, 16-31 -> face 1/3).
        for (uint32_t c = 0; c < 4; ++c) {
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t dst = c * 512 + (r / 16) * 256 + (r % 16) * 8, src = r * 64 + c * 16;
                if (r < live) {
                    for (uint32_t w = 0; w < 8; ++w) {
                        tiles[dst + w] = staged[src + w];
                        tiles[dst + 128 + w] = staged[src + 8 + w];
                    }
                } else {
                    for (uint32_t w = 0; w < 8; ++w) {
                        tiles[dst + w] = 0;
                        tiles[dst + 128 + w] = 0;
                    }
                }
            }
        }
        for (uint32_t c = 0; c < 4; ++c) {
            noc.async_write(CoreLocalMem<uint32_t>(base + c * 2048), output, 2048, {}, {.page_id = row * 4 + c});
        }
        noc.async_write_barrier();
    }
}
