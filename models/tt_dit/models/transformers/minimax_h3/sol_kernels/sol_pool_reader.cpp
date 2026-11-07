// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, pool step, reader kernel.
//
// Pooling = the mean of every 64-token block of Q, K and V. A unit is (tensor, head, group): 32 blocks.
// For each tile row of the unit the reader sends compute the HEAD_DIM_TILES data tiles (cb_data) and one pooling tile
// (cb_pool_tiles): a 32x32 tile with 1/len in the block's row, so compute just accumulates pool_tile @ data. Rows past
// the shard's valid end get zero tiles instead of DRAM reads. The valid token count comes from the geo tensor (word 0).
//
// The 32 standard pooling tiles are built once per core (cb_pool_table) and copied per row with a local NoC copy; only
// the rows of a partial last block are built in place. Data is read in batches of 4 rows with two batches in flight
// (transaction ids 1 and 2) while compute works on a third.
//
// Data tiles are in the inputs' dtype (bf16 or bf8), the pooling tiles in bf16. K and V share one dtype. Q has it too
// (Q_OWN_CB = 0: one data CB for all three) or its own (Q_OWN_CB = 1: Q's data goes through cb_q_data in Q's format,
// e.g. bf16 Q with bf8 K/V). A batch is pushed before the next batch starts in the other data CB.
//
// Compile-time args: 0 NUM_HEADS, 1 SHARD_ROW_TILES (tile rows per shard), 2 HEAD_DIM_TILES, 3 SHARD_BLOCK_TILES
// (output tile rows per head), 4 SHARD_ROWS, 5 GRID_X, 6 GRID_Y, 7 NUM_UNITS, 8 Q_OWN_CB, then TensorAccessorArgs of
// q_in, k_in, v_in, geo. Common runtime args: 0..2 the Q, K, V input addresses, 3 geo address.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "sol_pool_units.hpp"

namespace {
constexpr uint32_t cb_data = 0, cb_q_data = 1, cb_pool_tiles = 2, cb_zero = 3, cb_geo = 4, cb_pool_table = 5;
constexpr uint32_t TILE_BYTES = 2048;  // bf16 tiles: pooling, zero and table tiles
constexpr uint32_t BLOCK_TOKENS = sol_pool::BLOCK_TOKENS;
constexpr uint32_t TILE_ROWS_PER_GROUP = sol_pool::TILE_ROWS_PER_GROUP;
constexpr uint32_t GEO_BYTES = 128;
constexpr uint32_t ROWS_PER_BATCH = 4;
constexpr uint32_t BATCHES_IN_CB = 3;  // three batches per CB (SolPrepass._pool_program)

// bf16(1/len) for len = 0..64 (len 0 unused).
constexpr uint16_t RECIP_BF16[65] = {
    0x0000, 0x3f80, 0x3f00, 0x3eab, 0x3e80, 0x3e4d, 0x3e2b, 0x3e12, 0x3e00, 0x3de4, 0x3dcd, 0x3dba, 0x3dab,
    0x3d9e, 0x3d92, 0x3d89, 0x3d80, 0x3d71, 0x3d64, 0x3d58, 0x3d4d, 0x3d43, 0x3d3a, 0x3d32, 0x3d2b, 0x3d24,
    0x3d1e, 0x3d18, 0x3d12, 0x3d0d, 0x3d09, 0x3d04, 0x3d00, 0x3cf8, 0x3cf1, 0x3cea, 0x3ce4, 0x3cdd, 0x3cd8,
    0x3cd2, 0x3ccd, 0x3cc8, 0x3cc3, 0x3cbf, 0x3cba, 0x3cb6, 0x3cb2, 0x3cae, 0x3cab, 0x3ca7, 0x3ca4, 0x3ca1,
    0x3c9e, 0x3c9b, 0x3c98, 0x3c95, 0x3c92, 0x3c90, 0x3c8d, 0x3c8b, 0x3c89, 0x3c86, 0x3c84, 0x3c82, 0x3c80};

// Index of element (row, col) in a 32x32 bf16 tile (four 16x16 faces).
inline uint32_t tile_elem(uint32_t row, uint32_t col) {
    const uint32_t face = ((row >> 4) << 1) | (col >> 4);
    return face * 256 + (row & 15) * 16 + (col & 15);
}

// Set the first num_cols columns of tile row `row` to value (two elements per word).
inline void write_row(uint32_t tile_addr, uint32_t row, uint32_t num_cols, uint16_t value) {
    CoreLocalMem<volatile uint32_t> words(tile_addr);
    const uint32_t value_pair = (static_cast<uint32_t>(value) << 16) | value;
    uint32_t col = 0;
    for (; col + 1 < num_cols; col += 2) {
        words[tile_elem(row, col) >> 1] = value_pair;
    }
    if (col < num_cols) {
        CoreLocalMem<volatile uint16_t> elems(tile_addr);
        elems[tile_elem(row, col)] = value;
    }
}

inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }
}  // namespace

void kernel_main() {
    constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(0);
    constexpr uint32_t SHARD_ROW_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t HEAD_DIM_TILES = get_compile_time_arg_val(2);
    constexpr uint32_t SHARD_BLOCK_TILES = get_compile_time_arg_val(3);
    constexpr uint32_t Q_OWN_CB = get_compile_time_arg_val(8);  // 1: Q's dtype differs from K/V's
    constexpr auto q_in_args = TensorAccessorArgs<9>();
    constexpr auto k_in_args = TensorAccessorArgs<q_in_args.next_compile_time_args_offset()>();
    constexpr auto v_in_args = TensorAccessorArgs<k_in_args.next_compile_time_args_offset()>();
    constexpr auto geo_args = TensorAccessorArgs<v_in_args.next_compile_time_args_offset()>();

    constexpr uint32_t GRID_X = get_compile_time_arg_val(5);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(6);
    constexpr uint32_t NUM_UNITS = get_compile_time_arg_val(7);
    constexpr uint32_t NUM_CORES = GRID_X * GRID_Y;
    using Layout = sol_pool::Layout<NUM_HEADS, SHARD_ROW_TILES, SHARD_BLOCK_TILES>;
    static_assert(Layout::NUM_UNITS == NUM_UNITS, "host unit count must match the kernel layout");
    const uint32_t core_index = static_cast<uint32_t>(get_absolute_logical_y()) * GRID_X + get_absolute_logical_x();
    uint32_t unit_start, unit_count;
    Layout::range(core_index, NUM_CORES, unit_start, unit_count);
    if (unit_count == 0) {
        return;
    }
    const auto q_in = TensorAccessor(q_in_args, get_common_arg_val<uint32_t>(0));
    const auto k_in = TensorAccessor(k_in_args, get_common_arg_val<uint32_t>(1));
    const auto v_in = TensorAccessor(v_in_args, get_common_arg_val<uint32_t>(2));
    const auto geo = TensorAccessor(geo_args, get_common_arg_val<uint32_t>(3));

    Noc noc;
    CircularBuffer data_cb(cb_data), q_data_cb(cb_q_data), pool_tiles_cb(cb_pool_tiles), zero_cb(cb_zero),
        geo_cb(cb_geo), pool_table_cb(cb_pool_table);
    const PrecomposedUnicastEndpoint local;

    // This shard's valid token count.
    geo_cb.reserve_back(1);
    noc.async_read(geo, geo_cb, GEO_BYTES, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t valid_end = CoreLocalMem<volatile uint32_t>(geo_cb.get_write_ptr())[0];

    // The zero tile and the table of the 32 standard pooling tiles (1/64 in row r of tile r). The table is built while
    // the first batch's reads are in flight.
    zero_cb.reserve_back(1);
    const uint32_t zero_tile_addr = zero_cb.get_write_ptr();
    pool_table_cb.reserve_back(32);
    const uint32_t table_addr = pool_table_cb.get_write_ptr();
    bool table_built = false;
    auto build_table = [&]() {
        noc.async_write_zeros(CoreLocalMem<uint32_t>(zero_tile_addr), TILE_BYTES);
        for (uint32_t row = 0; row < 32; ++row) {
            noc.async_write_zeros(CoreLocalMem<uint32_t>(table_addr + row * TILE_BYTES), TILE_BYTES);
        }
        noc.write_zeros_l1_barrier();
        for (uint32_t row = 0; row < 32; ++row) {
            write_row(table_addr + row * TILE_BYTES, row, 32, RECIP_BF16[BLOCK_TOKENS]);
        }
        zero_cb.push_back(1);
        pool_table_cb.push_back(32);  // never popped: resident for the whole program
        table_built = true;
    };

    constexpr uint32_t DATA_PAGES_PER_BATCH = ROWS_PER_BATCH * HEAD_DIM_TILES;
    // Per data CB (0: cb_data, 1: cb_q_data): tile size, ring base and ring size.
    uint32_t data_tile_bytes[2] = {data_cb.get_tile_size(), 0};
    uint32_t data_cb_base[2] = {data_cb.get_write_ptr(), 0};
    if constexpr (Q_OWN_CB) {
        data_tile_bytes[1] = q_data_cb.get_tile_size();
        data_cb_base[1] = q_data_cb.get_write_ptr();
    }
    constexpr uint32_t POOL_TILES_CB_BYTES = BATCHES_IN_CB * ROWS_PER_BATCH * TILE_BYTES;
    const uint32_t pool_tiles_cb_base = pool_tiles_cb.get_write_ptr();

    bool pending = false;  // a batch on the NoC, not yet pushed
    uint32_t pending_trid = 2;
    CircularBuffer* pending_data_cb = &data_cb;
    auto finish_pending = [&]() {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = pending_trid});
        pool_tiles_cb.push_back(ROWS_PER_BATCH);
        pending_data_cb->push_back(DATA_PAGES_PER_BATCH);
        pending = false;
    };
    // Copy one pooling tile from local L1 (the table or the zero tile) into the batch.
    auto copy_pool_tile = [&](uint32_t src_addr, uint32_t dst_addr, uint32_t trid) {
        noc.async_read<NocOptions::TXN_ID>(
            local,
            CoreLocalMem<uint32_t>(dst_addr),
            TILE_BYTES,
            {.noc_addr = get_noc_addr(src_addr)},
            {},
            {.trid = trid});
    };

    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        const sol_pool::Unit coords = Layout::decode(unit);
        const uint32_t first_row = coords.group * TILE_ROWS_PER_GROUP;
        const uint32_t end_row = min_u(first_row + TILE_ROWS_PER_GROUP, SHARD_ROW_TILES);
        const uint32_t ring = (Q_OWN_CB && coords.tensor == 0) ? 1 : 0;  // always 0 when Q_OWN_CB = 0
        CircularBuffer& unit_data_cb = ring ? q_data_cb : data_cb;
        const uint32_t DATA_TILE_BYTES = data_tile_bytes[ring];
        const uint32_t DATA_CB_BYTES = BATCHES_IN_CB * DATA_PAGES_PER_BATCH * DATA_TILE_BYTES;
        if constexpr (Q_OWN_CB) {
            if (pending && pending_data_cb != &unit_data_cb) {
                finish_pending();  // the next batch goes into the other data CB
            }
        }

        auto read_page = [&](uint32_t page, uint32_t dst_addr, uint32_t trid) {
            if (coords.tensor == 0) {
                noc.async_read<NocOptions::TXN_ID>(
                    q_in, CoreLocalMem<uint32_t>(dst_addr), DATA_TILE_BYTES, {.page_id = page}, {}, {.trid = trid});
            } else if (coords.tensor == 1) {
                noc.async_read<NocOptions::TXN_ID>(
                    k_in, CoreLocalMem<uint32_t>(dst_addr), DATA_TILE_BYTES, {.page_id = page}, {}, {.trid = trid});
            } else {
                noc.async_read<NocOptions::TXN_ID>(
                    v_in, CoreLocalMem<uint32_t>(dst_addr), DATA_TILE_BYTES, {.page_id = page}, {}, {.trid = trid});
            }
        };

        for (uint32_t batch_row = first_row; batch_row < end_row; batch_row += ROWS_PER_BATCH) {
            const uint32_t rows_in_batch = min_u(ROWS_PER_BATCH, end_row - batch_row);
            // Reserve two batches so the slot after the pending batch is free.
            unit_data_cb.reserve_back(2 * DATA_PAGES_PER_BATCH);
            pool_tiles_cb.reserve_back(2 * ROWS_PER_BATCH);
            uint32_t data_slot = unit_data_cb.get_write_ptr(), pool_tiles_slot = pool_tiles_cb.get_write_ptr();
            if (pending) {
                data_slot += DATA_PAGES_PER_BATCH * DATA_TILE_BYTES;
                if (data_slot >= data_cb_base[ring] + DATA_CB_BYTES) {
                    data_slot -= DATA_CB_BYTES;
                }
                pool_tiles_slot += ROWS_PER_BATCH * TILE_BYTES;
                if (pool_tiles_slot >= pool_tiles_cb_base + POOL_TILES_CB_BYTES) {
                    pool_tiles_slot -= POOL_TILES_CB_BYTES;
                }
            }
            const uint32_t trid = pending ? 3 - pending_trid : 1;  // alternate 1, 2

            // Pooling tiles: copied from the table, or the zero tile past the valid end; the rows of a partial last
            // block are built in place.
            uint32_t partial_block_rows = 0;  // bit i: row i of the batch needs its pooling tile built in place
            bool zeros_issued = false;
            auto issue_pool_tiles = [&]() {
                for (uint32_t row_in_batch = 0; row_in_batch < rows_in_batch; ++row_in_batch) {
                    const uint32_t tile_row = batch_row + row_in_batch;
                    const uint32_t pool_tile_addr = pool_tiles_slot + row_in_batch * TILE_BYTES;
                    if (32 * tile_row >= valid_end) {
                        copy_pool_tile(zero_tile_addr, pool_tile_addr, trid);
                    } else {
                        const uint32_t block = tile_row >> 1;  // two tile rows per 64-token block
                        const uint32_t block_len = min_u(BLOCK_TOKENS, valid_end - block * BLOCK_TOKENS);
                        const uint32_t valid_cols = min_u(32, valid_end - 32 * tile_row);
                        if (block_len == BLOCK_TOKENS && valid_cols == 32) {
                            copy_pool_tile(table_addr + (block & 31) * TILE_BYTES, pool_tile_addr, trid);
                        } else {
                            noc.async_write_zeros(CoreLocalMem<uint32_t>(pool_tile_addr), TILE_BYTES);
                            zeros_issued = true;
                            partial_block_rows |= 1u << row_in_batch;
                        }
                    }
                }
            };
            // Data pages, zero past the valid end.
            auto issue_data_pages = [&]() {
                uint32_t offset = 0;
                for (uint32_t row_in_batch = 0; row_in_batch < rows_in_batch; ++row_in_batch) {
                    const uint32_t tile_row = batch_row + row_in_batch;
                    for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                        if (32 * tile_row < valid_end) {
                            read_page(
                                (coords.head * SHARD_ROW_TILES + tile_row) * HEAD_DIM_TILES + dim_tile,
                                data_slot + offset,
                                trid);
                        } else {
                            noc.async_write_zeros(CoreLocalMem<uint32_t>(data_slot + offset), DATA_TILE_BYTES);
                            zeros_issued = true;
                        }
                        offset += DATA_TILE_BYTES;
                    }
                }
            };
            if (table_built) {
                issue_pool_tiles();
                issue_data_pages();
            } else {  // first batch: issue the reads, then build the table
                issue_data_pages();
                build_table();
                issue_pool_tiles();
            }
            if (zeros_issued) {
                noc.write_zeros_l1_barrier();
                for (uint32_t row_in_batch = 0; row_in_batch < rows_in_batch; ++row_in_batch) {
                    if (partial_block_rows & (1u << row_in_batch)) {
                        const uint32_t tile_row = batch_row + row_in_batch;
                        const uint32_t block = tile_row >> 1;
                        const uint32_t block_len = min_u(BLOCK_TOKENS, valid_end - block * BLOCK_TOKENS);
                        write_row(
                            pool_tiles_slot + row_in_batch * TILE_BYTES,
                            block & 31,
                            min_u(32, valid_end - 32 * tile_row),
                            RECIP_BF16[block_len]);
                    }
                }
            }
            if (pending) {
                finish_pending();
            }
            pending = true;
            pending_trid = trid;
            pending_data_cb = &unit_data_cb;
        }
    }
    if (pending) {
        finish_pending();
    }
    if (!table_built) {
        // An all-empty core skips the row loop, but compute still waits for cb_zero.
        noc.async_write_zeros(CoreLocalMem<uint32_t>(zero_tile_addr), TILE_BYTES);
        noc.write_zeros_l1_barrier();
        zero_cb.push_back(1);
    }
    // Leave the NoC transaction id at 0 for the next kernel on this core.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
