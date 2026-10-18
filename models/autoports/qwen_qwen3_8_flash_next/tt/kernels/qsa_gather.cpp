// SPDX-License-Identifier: Apache-2.0
// Model-local BF16 tile-row transfers. All addressing is device-resident.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

static uint32_t face_index(uint32_t row, uint32_t col) {
    return (row / 16) * 512 + (col / 16) * 256 + (row % 16) * 16 + col % 16;
}

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr uint32_t capacity = get_compile_time_arg_val(6);
    constexpr uint32_t index_bytes = get_compile_time_arg_val(7);
    constexpr bool indices_tiled = get_compile_time_arg_val(8);
    constexpr uint32_t St = (S + 31) / 32;
    constexpr uint32_t page_tiles = page_size / 32;
    constexpr uint32_t table_bytes = capacity / page_size * 4;
    static_assert(get_compile_time_arg_val(9) == 0x51534101, "QSA Python/kernel argument ABI mismatch");
    constexpr auto aa = TensorAccessorArgs<10>();
    constexpr auto oa = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(aa, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1));
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(2));
    const auto table = TensorAccessor(ta, get_arg_val<uint32_t>(3));
    const auto indices = TensorAccessor(ia, get_arg_val<uint32_t>(4));
    const uint32_t job_begin = get_arg_val<uint32_t>(5);
    const uint32_t job_count = get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    const uint32_t input_tile_addr = base;
    const uint32_t output_tile_addr = base + 2048;
    const uint32_t pos_addr = base + 4096;
    const uint32_t table_addr = base + 4224;
    const uint32_t index_addr = table_addr + table_bytes;
    auto in = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_tile_addr);
    auto out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(output_tile_addr);
    auto pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_addr);
    auto pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_addr);
    auto ids16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(index_addr);
    auto ids32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_addr);
    Noc noc;
    for (uint32_t job = job_begin; job < job_begin + job_count; ++job) {
        const uint32_t col = job % Wt;
        const uint32_t head = (job / Wt) % H;
        const uint32_t request_query = job / (Wt * H);
        const uint32_t batch = (mode == 1 || mode == 3) ? request_query / S : request_query;
        const uint32_t query = (mode == 1 || mode == 3) ? request_query % S : 0;
        ASSERT(batch < B);
        noc.async_read(positions, CoreLocalMem<uint32_t>(pos_addr), 128, {.page_id = batch}, {});
        noc.async_read(table, CoreLocalMem<uint32_t>(table_addr), table_bytes, {.page_id = batch}, {});
        noc.async_read_barrier();
        const uint32_t start = pos[0];
        ASSERT(start + S <= capacity);
        if constexpr (mode == 0 || mode == 2) {
            // Assign one (request,head,column) to one worker: no competing RMW
            // of a destination tile, even with non-aligned prefix and tail.
            for (uint32_t s = 0; s < S; ++s) {
                const uint32_t absolute = start + s;
                if constexpr (mode == 2) {
                    if (absolute % 4 != 3) {
                        continue;
                    }
                }
                const uint32_t source_tile = ((batch * H + head) * St + s / 32) * Wt + col;
                uint32_t output_tile, output_row;
                if constexpr (mode == 0) {
                    const uint32_t physical = pages[absolute / page_size];
                    output_tile = ((physical * H + head) * page_tiles + absolute % page_size / 32) * Wt + col;
                    output_row = absolute % 32;
                } else {
                    const uint32_t block = absolute / 4;
                    output_tile = ((batch * H + head) * (capacity / 4 / 32) + block / 32) * Wt + col;
                    output_row = block % 32;
                }
                noc.async_read(source, CoreLocalMem<uint32_t>(input_tile_addr), 2048, {.page_id = source_tile}, {});
                noc.async_read(output, CoreLocalMem<uint32_t>(output_tile_addr), 2048, {.page_id = output_tile}, {});
                noc.async_read_barrier();
                for (uint32_t c = 0; c < 32; ++c) {
                    out[face_index(output_row, c)] = in[face_index(s % 32, c)];
                }
                noc.async_write(CoreLocalMem<uint32_t>(output_tile_addr), output, 2048, {}, {.page_id = output_tile});
                noc.async_write_barrier();
            }
        } else if constexpr (mode == 1) {
            // Four raw index rows, before pooling/norm/RoPE. Unfinished groups
            // may include stale future rows; they are never committed/selected.
            for (uint32_t i = 0; i < 1024; ++i) {
                out[i] = 0;
            }
            const uint32_t group = (start + query) / 4 * 4;
            for (uint32_t r = 0; r < 4; ++r) {
                const uint32_t absolute = group + r;
                const uint32_t physical = pages[absolute / page_size];
                const uint32_t tile = (physical * page_tiles + absolute % page_size / 32) * Wt + col;
                noc.async_read(source, CoreLocalMem<uint32_t>(input_tile_addr), 2048, {.page_id = tile}, {});
                noc.async_read_barrier();
                for (uint32_t c = 0; c < 32; ++c) {
                    out[face_index(r, c)] = in[face_index(absolute % 32, c)];
                }
            }
            noc.async_write(
                CoreLocalMem<uint32_t>(output_tile_addr), output, 2048, {}, {.page_id = request_query * Wt + col});
            noc.async_write_barrier();
        } else if constexpr (mode == 3) {
            if constexpr (indices_tiled) {
                const uint32_t tile_addr = index_addr + 2048;
                auto tile16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
                auto tile32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
                for (uint32_t c_tile = 0; c_tile < 16; ++c_tile) {
                    const uint32_t tile = (batch * St + query / 32) * 16 + c_tile;
                    noc.async_read(
                        indices, CoreLocalMem<uint32_t>(tile_addr), 1024 * index_bytes, {.page_id = tile}, {});
                    noc.async_read_barrier();
                    for (uint32_t c = 0; c < 32; ++c) {
                        if constexpr (index_bytes == 2) {
                            ids16[c_tile * 32 + c] = tile16[face_index(query % 32, c)];
                        } else {
                            ids32[c_tile * 32 + c] = tile32[face_index(query % 32, c)];
                        }
                    }
                }
            } else {
                noc.async_read(
                    indices, CoreLocalMem<uint32_t>(index_addr), 512 * index_bytes, {.page_id = request_query}, {});
                noc.async_read_barrier();
            }
            const uint32_t last = start + query;
            const uint32_t complete = (last + 1) / 4;
            uint32_t previous_tile = 0xffffffff;
            for (uint32_t dest_tile = 0; dest_tile < 65; ++dest_tile) {
                for (uint32_t i = 0; i < 1024; ++i) {
                    out[i] = 0;
                }
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t selected_token = dest_tile * 32 + r;
                    uint32_t absolute;
                    if (selected_token < 2048) {
                        const uint32_t selected_block =
                            index_bytes == 2 ? ids16[selected_token / 4] : ids32[selected_token / 4];
                        if (selected_block >= complete) {
                            continue;
                        }
                        absolute = selected_block * 4 + selected_token % 4;
                    } else {
                        absolute = complete * 4 + selected_token - 2048;
                        if (absolute > last) {
                            continue;
                        }
                    }
                    const uint32_t physical = pages[absolute / page_size];
                    const uint32_t tile = ((physical * H + head) * page_tiles + absolute % page_size / 32) * Wt + col;
                    if (tile != previous_tile) {
                        noc.async_read(source, CoreLocalMem<uint32_t>(input_tile_addr), 2048, {.page_id = tile}, {});
                        noc.async_read_barrier();
                        previous_tile = tile;
                    }
                    for (uint32_t c = 0; c < 32; ++c) {
                        out[face_index(r, c)] = in[face_index(absolute % 32, c)];
                    }
                }
                const uint32_t tile = ((request_query * H + head) * 65 + dest_tile) * Wt + col;
                noc.async_write(CoreLocalMem<uint32_t>(output_tile_addr), output, 2048, {}, {.page_id = tile});
                noc.async_write_barrier();
            }
        }
    }
}
