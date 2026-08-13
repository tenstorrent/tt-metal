// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused Gumbel-max sampling, reduction half -- entirely on device, spread across the whole grid.
//
// The work unit is a TILE, so a token row's vocabulary is spread over many cores and no core owns a
// whole row. That is what fixes decode: with a row-based unit, tokens == 1 gave only B_local work
// units, most of the grid idled, and the fused elementwise half measured ~2.9x slower than the six
// tile-parallel ttnn ops it replaces. The price of tile units is that the argmax becomes a
// cross-core reduction, arranged here so the common case never pays for it:
//
//   * A row whose tiles lie ENTIRELY inside this core's run is reduced and written right here. In
//     prefill nearly every row is like that, so the exchange below is almost never used.
//   * A row this core only partially covers is a BOUNDARY row. A core has at most two (its first and
//     its last), so the exchange is bounded at 2 records per core whatever the shape. Those records
//     are NOC-written into the origin core's L1, which merges by row id and writes those rows.
//     Follows frobenius_normalize: partials to origin's L1, noc_semaphore_inc, origin waits on
//     num_cores - 1.
//
// Comparison is on raw FP32 bit patterns via float32_greater: the data-movement RISCs have no FPU.

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

// A boundary record: [valid, row_id, 32 max bit-patterns, 32 indices], padded to a NOC-friendly
// multiple of 16 bytes. Two per core is the hard bound (the first and last row of a core's run).
constexpr uint32_t kRecordStrideU32 = 72U;  // 66 used, padded to 288 bytes
constexpr uint32_t kRecordBytes = kRecordStrideU32 * sizeof(uint32_t);
constexpr uint32_t kRecordsPerCore = 2U;

// Bytes per staged output value: a NOC write needs its L1 source aligned, so each token id gets its
// own slot rather than sitting packed 4 bytes apart.
constexpr uint32_t kOutputSlotBytes = 32U;

}  // namespace

void kernel_main() {
    uint32_t rt_idx = 0U;
    const uint32_t output_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t core_index = get_arg_val<uint32_t>(rt_idx++);

    constexpr uint32_t cb_scores_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_output_staging_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_records_idx = tt::CBIndex::c_4;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t logical_vocab = get_compile_time_arg_val(1);
    constexpr uint32_t logical_tokens = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t num_cores = get_compile_time_arg_val(4);
    constexpr uint32_t reduction_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t origin_phys_x = get_compile_time_arg_val(6);
    constexpr uint32_t origin_phys_y = get_compile_time_arg_val(7);

#ifdef DO_POSITIONS
    constexpr bool do_positions = true;
#else
    constexpr bool do_positions = false;
#endif

    // Per-entry token positions, appended after the four fixed args (see kWriterPositionsArgBase).
    // Every core holds the full local list, so the origin can re-derive the target row of any entry
    // it merges without that row travelling in the record.
    constexpr uint32_t positions_arg_base = 4U;
    auto target_row_of = [](uint32_t entry) -> uint32_t {
        return get_arg_val<uint32_t>(positions_arg_base + entry) & (kTileHeight - 1U);
    };

    constexpr auto output_args = TensorAccessorArgs<8>();
    const auto output_address_generator = TensorAccessor(output_args, output_address);

    const uint32_t staging_address = get_write_ptr(cb_output_staging_idx);
    const uint32_t records_base = get_write_ptr(cb_records_idx);

    uint32_t max_values[kTileHeight];
    uint32_t arg_max[kTileHeight];

    // How many of a tile row's 32 rows are real tokens. This bounds the SCAN, not just the
    // write-out: decode produces one token per step, so 31 of 32 rows are padding and scanning them
    // anyway costs 32x.
    auto valid_rows_of = [](uint32_t tile_row) -> uint32_t {
        if constexpr (do_positions) {
            // One row per batch entry, always real: the host validated every position against the
            // logical token count.
            return 1U;
        }
        const uint32_t first_token = (tile_row % Ht) * kTileHeight;
        if (first_token >= logical_tokens) {
            return 0U;
        }
        const uint32_t remaining = logical_tokens - first_token;
        return (remaining < kTileHeight) ? remaining : kTileHeight;
    };

    // Emit a group's token ids. Output pages run row-major over [B, 1, tokens] normally, and over
    // [B, 1, 1] -- one page per batch entry -- when positions selected a single row each.
    auto write_row = [&](uint32_t tile_row, uint32_t valid_rows) {
        if constexpr (do_positions) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging_address) = arg_max[target_row_of(tile_row)];
            noc_async_write_page(tile_row, output_address_generator, staging_address);
            noc_async_write_barrier();
            return;
        }
        const uint32_t page_base = (tile_row / Ht) * logical_tokens + (tile_row % Ht) * kTileHeight;
        for (uint32_t h = 0U; h < valid_rows; ++h) {
            const uint32_t slot = staging_address + h * kOutputSlotBytes;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot) = arg_max[h];
            noc_async_write_page(page_base + h, output_address_generator, slot);
        }
        noc_async_write_barrier();
    };

    uint32_t staged_records = 0U;

    // All 32 slots travel verbatim: rows this core never scanned are still NEG_INF from
    // reset_accumulators, so the merge below leaves them alone whichever mode is compiled.
    auto stage_record = [&](uint32_t tile_row, uint32_t /*valid_rows*/) {
        auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            records_base + (core_index * kRecordsPerCore + staged_records) * kRecordBytes);
        rec[0] = 1U;
        rec[1] = tile_row;
        for (uint32_t h = 0U; h < kTileHeight; ++h) {
            rec[2U + h] = max_values[h];
            rec[2U + kTileHeight + h] = arg_max[h];
        }
        ++staged_records;
    };

    auto row_is_owned = [&](uint32_t tile_row) -> bool {
        const uint32_t row_first = tile_row * Wt;
        return (row_first >= start_tile) && (row_first + Wt <= start_tile + num_tiles);
    };

    auto finish_row = [&](uint32_t tile_row, uint32_t valid_rows) {
        if (valid_rows == 0U) {
            return;
        }
        if (row_is_owned(tile_row)) {
            write_row(tile_row, valid_rows);
        } else {
            stage_record(tile_row, valid_rows);
        }
    };

    auto reset_accumulators = [&]() {
        for (uint32_t h = 0U; h < kTileHeight; ++h) {
            max_values[h] = NEG_INF_FLOAT32;
            arg_max[h] = 0U;
        }
    };

    // ---- pass 1: reduce the scores as they stream past ----
    uint32_t current_row = start_tile / Wt;
    uint32_t current_valid = valid_rows_of(current_row);
    reset_accumulators();

    for (uint32_t t = 0U; t < num_tiles; ++t) {
        const uint32_t global_tile = start_tile + t;
        const uint32_t tile_row = global_tile / Wt;

        if (tile_row != current_row) {
            finish_row(current_row, current_valid);
            current_row = tile_row;
            current_valid = valid_rows_of(current_row);
            reset_accumulators();
        }

        cb_wait_front(cb_scores_idx, onetile);
        auto* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_scores_idx));
        const uint32_t tile_col_base = (global_tile % Wt) * kTileWidth;

        // Rows worth scanning in this tile. Normally that is every real token row; with positions
        // it is the single row the entry asked for, and the accumulator it lands in is indexed by
        // that same row so write_row and the merge need no extra bookkeeping.
        const uint32_t row_begin = do_positions ? target_row_of(current_row) : 0U;
        const uint32_t row_end = do_positions ? row_begin + 1U : current_valid;

        for (uint32_t face = 0U; face < 4U; ++face) {
            const uint32_t face_row_base = (face >= 2U) ? kFaceHeight : 0U;
            const uint32_t face_col_base = (face & 1U) ? kFaceWidth : 0U;
            const uint32_t global_col_base = tile_col_base + face_col_base;

            // Columns past the logical vocab are tile padding; so are rows outside [begin, end).
            if (global_col_base >= logical_vocab) {
                continue;
            }
            const uint32_t first_row = (row_begin > face_row_base) ? row_begin : face_row_base;
            const uint32_t face_row_end = face_row_base + kFaceHeight;
            const uint32_t last_row = (row_end < face_row_end) ? row_end : face_row_end;
            if (first_row >= last_row) {
                continue;
            }
            const uint32_t cols_left = logical_vocab - global_col_base;
            const uint32_t cols_to_scan = (cols_left < kFaceWidth) ? cols_left : kFaceWidth;

            const uint32_t face_offset = face * kFaceSize;
            for (uint32_t row_in_tile = first_row; row_in_tile < last_row; ++row_in_tile) {
                uint32_t running_max = max_values[row_in_tile];
                uint32_t running_arg = arg_max[row_in_tile];

                const uint32_t row_offset = face_offset + (row_in_tile - face_row_base) * kFaceWidth;
                for (uint32_t cc = 0U; cc < cols_to_scan; ++cc) {
                    const uint32_t value = tile_ptr[row_offset + cc];
                    // Strict greater, scanning columns in increasing global order, so ties keep the
                    // lowest index -- matching ttnn::argmax's tie-break.
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

    finish_row(current_row, current_valid);

    // ---- pass 2: combine boundary rows on the origin core ----
    if constexpr (num_cores > 1U) {
        const uint32_t local_base = records_base + core_index * kRecordsPerCore * kRecordBytes;
        // Blank the unused slots so the origin can tell live records from stale L1.
        for (uint32_t k = staged_records; k < kRecordsPerCore; ++k) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_base + k * kRecordBytes) = 0U;
        }

        if (core_index != 0U) {
            // One write and one increment per core, so the origin waits on a core count.
            const uint64_t dst = get_noc_addr(origin_phys_x, origin_phys_y, local_base);
            noc_async_write(local_base, dst, kRecordsPerCore * kRecordBytes);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(origin_phys_x, origin_phys_y, get_semaphore(reduction_sem_id)), 1U);
        } else {
            auto* sem_ptr = get_sem_ptr(reduction_sem_id);
            noc_semaphore_wait(sem_ptr, num_cores - 1U);
            noc_semaphore_set(sem_ptr, 0U);

            // Merge by row id: fold every later record carrying the same row into the first one,
            // then write that row once. Records number 2 per core, so the quadratic scan is cheap.
            constexpr uint32_t total_records = num_cores * kRecordsPerCore;
            for (uint32_t a = 0U; a < total_records; ++a) {
                auto* ra = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(records_base + a * kRecordBytes);
                if (ra[0] == 0U) {
                    continue;
                }
                const uint32_t row = ra[1];
                for (uint32_t h = 0U; h < kTileHeight; ++h) {
                    max_values[h] = ra[2U + h];
                    arg_max[h] = ra[2U + kTileHeight + h];
                }
                ra[0] = 0U;

                for (uint32_t b = a + 1U; b < total_records; ++b) {
                    auto* rb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(records_base + b * kRecordBytes);
                    if (rb[0] == 0U || rb[1] != row) {
                        continue;
                    }
                    for (uint32_t h = 0U; h < kTileHeight; ++h) {
                        const uint32_t v = rb[2U + h];
                        const uint32_t i = rb[2U + kTileHeight + h];
                        // Ties keep the lower index, matching the in-row scan.
                        if (float32_greater(v, max_values[h]) || (v == max_values[h] && i < arg_max[h])) {
                            max_values[h] = v;
                            arg_max[h] = i;
                        }
                    }
                    rb[0] = 0U;
                }

                write_row(row, valid_rows_of(row));
            }
        }
    }
}
