// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused Gumbel-max sampling, reduction half.
//
// Consumes the score tiles the compute kernel produces and folds each into a running per-token
// (max, argmax), then writes one token id per row. This replaces the `untilize` + `ttnn::argmax`
// pair at the end of ttnn_fixed::sample -- and because the reduction happens as the scores stream
// past, the [B, 1, tokens, V] score tensor never reaches DRAM at all.
//
// Comparison is done on raw FP32 bit patterns via float32_greater (same helper ttnn's argmax reader
// uses): the data-movement RISCs have no FPU, so an actual float compare would be soft-float.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/numeric/float32.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

namespace {

constexpr uint32_t kTileHeight = 32U;
constexpr uint32_t kTileWidth = 32U;
constexpr uint32_t kFaceHeight = 16U;
constexpr uint32_t kFaceWidth = 16U;
constexpr uint32_t kFaceSize = kFaceHeight * kFaceWidth;

// Bytes reserved per staged output value. A NOC write needs its L1 source aligned, so each of the
// 32 token ids gets its own aligned slot instead of sitting packed 4 bytes apart; that lets all 32
// page writes be issued back to back behind a single barrier.
constexpr uint32_t kOutputSlotBytes = 32U;

}  // namespace

void kernel_main() {
    uint32_t rt_idx = 0U;
    const uint32_t output_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t start_row = get_arg_val<uint32_t>(rt_idx++);

    constexpr uint32_t cb_scores_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_output_staging_idx = tt::CBIndex::c_3;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t logical_vocab = get_compile_time_arg_val(1);
    constexpr uint32_t logical_tokens = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);

    constexpr auto output_args = TensorAccessorArgs<4>();
    const auto output_address_generator = TensorAccessor(output_args, output_address);

    const uint32_t staging_address = get_write_ptr(cb_output_staging_idx);

    uint32_t max_values[kTileHeight];
    uint32_t arg_max[kTileHeight];

    for (uint32_t i = 0U; i < num_rows_to_process; ++i) {
        const uint32_t global_row = start_row + i;

        for (uint32_t h = 0U; h < kTileHeight; ++h) {
            max_values[h] = NEG_INF_FLOAT32;
            arg_max[h] = 0U;
        }

        // ---- streaming argmax over the row's vocab tiles ----
        for (uint32_t wt = 0U; wt < Wt; ++wt) {
            cb_wait_front(cb_scores_idx, onetile);
            auto* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_scores_idx));

            const uint32_t tile_col_base = wt * kTileWidth;

            for (uint32_t face = 0U; face < 4U; ++face) {
                const uint32_t face_row_base = (face >= 2U) ? kFaceHeight : 0U;
                const uint32_t face_col_base = (face & 1U) ? kFaceWidth : 0U;
                const uint32_t global_col_base = tile_col_base + face_col_base;

                // Columns past the logical vocab are tile padding: skip them outright rather than
                // masking them to -inf, so nothing that lands there (including NaN) can ever win.
                if (global_col_base >= logical_vocab) {
                    continue;
                }
                const uint32_t remaining = logical_vocab - global_col_base;
                const uint32_t cols_to_scan = (remaining < kFaceWidth) ? remaining : kFaceWidth;

                const uint32_t face_offset = face * kFaceSize;

                for (uint32_t rr = 0U; rr < kFaceHeight; ++rr) {
                    const uint32_t row_in_tile = face_row_base + rr;

                    uint32_t running_max = max_values[row_in_tile];
                    uint32_t running_arg = arg_max[row_in_tile];

                    const uint32_t row_offset = face_offset + rr * kFaceWidth;
                    for (uint32_t cc = 0U; cc < cols_to_scan; ++cc) {
                        const uint32_t value = tile_ptr[row_offset + cc];
                        // Strict greater, scanning columns in increasing global order, so ties keep
                        // the lowest index -- matching ttnn::argmax's tie-break.
                        if (float32_greater(value, running_max)) {
                            running_max = value;
                            running_arg = global_col_base + cc;
                        }
                    }

                    max_values[row_in_tile] = running_max;
                    arg_max[row_in_tile] = running_arg;
                }
            }

            cb_pop_front(cb_scores_idx, onetile);
        }

        // ---- emit one token id per valid row ----
        // Output is ROW_MAJOR UINT32 [B, 1, tokens, 1], so one page is one token id and page ids run
        // row-major over [B, 1, tokens].
        const uint32_t batch_index = global_row / Ht;
        const uint32_t tile_row_in_batch = global_row % Ht;
        const uint32_t first_token = tile_row_in_batch * kTileHeight;

        // The final tile row of each batch entry is partly padding when tokens % 32 != 0.
        uint32_t valid_rows = 0U;
        if (first_token < logical_tokens) {
            const uint32_t remaining_tokens = logical_tokens - first_token;
            valid_rows = (remaining_tokens < kTileHeight) ? remaining_tokens : kTileHeight;
        }

        const uint32_t page_base = batch_index * logical_tokens + first_token;
        for (uint32_t h = 0U; h < valid_rows; ++h) {
            const uint32_t slot_address = staging_address + h * kOutputSlotBytes;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_address) = arg_max[h];
            noc_async_write_page(page_base + h, output_address_generator, slot_address);
        }
        noc_async_write_barrier();
    }
}
