// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader for conv_spatial_compute.cpp: depthwise K x K conv per frame (zero padding K / 2), K = 1 for a plain copy.
//
// x is `[1, 1, rows, C]` bf16 tiled, frame f's tokens are rows row0 + f stride .. + S - 1 (row-major over a gh x gw grid),
// not tile-aligned. This core handles one 32-channel group g and a range of units u = f RT + rt (output rows
// 32 rt .. 32 rt + 31 of frame f, RT = ceil(S / 32)). Per unit it reads the input tiles of group g that any tap can
// touch into a window, then builds one tile per tap: row r holds the input token at (y + i - K/2, x + j - K/2) for the
// output token p = 32 rt + r at (y, x), or zeros where that falls outside the frame or p >= S. The taps' weights
// (fp32 tiles, each row the group's 32 channel weights) are read once.
//
// Compile-time args: K, gh, gw, channel tiles of x, then the tensor accessor args of x, w.
// Runtime args: x, w addresses, group, first unit, number of units, row0, frame stride (rows from one frame's start to
// the next: S, or the frame slot of the frame-aligned layout).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_w = 0;
constexpr uint32_t cb_window = 1;
constexpr uint32_t cb_shift = 2;
constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t HALF_ROW_WORDS = 8;   // 16 bf16
constexpr uint32_t HALF_ROW_BYTES = 32;  // 16 bf16
constexpr uint32_t FACE_BYTES = 512;     // 16 x 16 bf16
}  // namespace

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t group = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const uint32_t row0 = get_arg_val<uint32_t>(5);
    const uint32_t stride = get_arg_val<uint32_t>(6);

    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t gh = get_compile_time_arg_val(1);
    constexpr uint32_t gw = get_compile_time_arg_val(2);
    constexpr uint32_t c_tiles = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t S = gh * gw;
    constexpr uint32_t RT = (S + TILE_ROWS - 1) / TILE_ROWS;
    constexpr uint32_t pad = K / 2;
    constexpr uint32_t taps = K * K;

    Noc noc;
    DataflowBuffer w_buf(cb_w);
    DataflowBuffer window(cb_window);
    DataflowBuffer shift(cb_shift);
    const uint32_t w_bytes = w_buf.get_entry_size();
    const uint32_t x_bytes = shift.get_entry_size();
    const auto x = TensorAccessor(x_args, x_addr);
    const auto w = TensorAccessor(w_args, w_addr);

    w_buf.reserve_back(taps);
    for (uint32_t t = 0; t < taps; ++t) {
        noc.async_read(w, w_buf, w_bytes, {.page_id = t * c_tiles + group}, {.offset_bytes = t * w_bytes});
    }
    noc.async_read_barrier();
    w_buf.push_back(taps);

    window.reserve_back(1);  // used as scratch, never pushed
    const uint32_t window_base = window.get_write_ptr();

    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t f = u / RT;
        const uint32_t p0 = (u % RT) * TILE_ROWS;
        const uint32_t frame_row = row0 + f * stride;
        // Input rows any tap can reach, clipped to the frame.
        const uint32_t reach = pad * (gw + 1);
        const uint32_t lo = frame_row + (p0 > reach ? p0 - reach : 0);
        const uint32_t last_p = p0 + TILE_ROWS - 1 + reach;
        const uint32_t hi = frame_row + (last_p < S ? last_p : S - 1);
        const uint32_t tile_lo = lo / TILE_ROWS;
        const uint32_t tile_hi = hi / TILE_ROWS;
        for (uint32_t r = tile_lo; r <= tile_hi; ++r) {
            noc.async_read(
                x, window, x_bytes, {.page_id = r * c_tiles + group}, {.offset_bytes = (r - tile_lo) * x_bytes});
        }
        noc.async_read_barrier();

        // Tile layout: row r's two 16-element halves are in faces 2 (r >= 16) and 2 (r >= 16) + 1, 32 bytes each, so
        // rows r .. r + n - 1 within one 16-row half are contiguous in both faces. Runs of consecutive source rows go
        // by local NoC reads (two per run), invalid rows are zeroed; one barrier for all taps of the tile.
        const auto row_offset = [](uint32_t r) { return (r >= 16 ? 2 * FACE_BYTES : 0) + (r % 16) * HALF_ROW_BYTES; };
        const uint32_t y0 = p0 / gw;
        const uint32_t x0 = p0 % gw;
        shift.reserve_back(taps);
        const uint32_t shift_base = shift.get_write_ptr();
        for (uint32_t i = 0; i < K; ++i) {
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t out_base = shift_base + (i * K + j) * x_bytes;
                uint32_t run_src = 0, run_dst = 0, run_len = 0;
                const auto flush = [&]() {
                    if (run_len == 0) {
                        return;
                    }
                    const uint32_t src_addr = window_base + (run_src / TILE_ROWS - tile_lo) * x_bytes +
                                              row_offset(run_src % TILE_ROWS);
                    const uint32_t dst_addr = out_base + row_offset(run_dst);
                    noc_async_read(get_noc_addr(src_addr), dst_addr, run_len * HALF_ROW_BYTES);
                    noc_async_read(get_noc_addr(src_addr + FACE_BYTES), dst_addr + FACE_BYTES, run_len * HALF_ROW_BYTES);
                    run_len = 0;
                };
                uint32_t y = y0, xx = x0;
                for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                    const uint32_t p = p0 + r;
                    const int32_t ys = static_cast<int32_t>(y + i) - static_cast<int32_t>(pad);
                    const int32_t xs = static_cast<int32_t>(xx + j) - static_cast<int32_t>(pad);
                    const bool valid = p < S && ys >= 0 && ys < static_cast<int32_t>(gh) && xs >= 0 &&
                                       xs < static_cast<int32_t>(gw);
                    if (valid) {
                        const uint32_t src = frame_row + static_cast<uint32_t>(ys) * gw + static_cast<uint32_t>(xs);
                        const bool extends = run_len > 0 && src == run_src + run_len && r % 16 != 0 && src % 16 != 0;
                        if (!extends) {
                            flush();
                            run_src = src;
                            run_dst = r;
                        }
                        ++run_len;
                    } else {
                        flush();
                        auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_base + row_offset(r));
                        for (uint32_t k = 0; k < HALF_ROW_WORDS; ++k) {
                            out[k] = 0;
                            out[FACE_BYTES / 4 + k] = 0;
                        }
                    }
                    if (++xx == gw) {
                        xx = 0;
                        ++y;
                    }
                }
                flush();
            }
        }
        noc_async_read_barrier();
        shift.push_back(taps);
    }
}
