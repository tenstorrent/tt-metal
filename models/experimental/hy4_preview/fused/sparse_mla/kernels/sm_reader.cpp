// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sparse-MLA row reader (data movement, on every reader / converter core). For every chunk its S-block serves it
// fetches the chunk's row ids from the selection table and reads the rows of its own slabs (sm_common.hpp
// own_slabs) straight out of the KV storage in DRAM, one NoC read per row at the row's own address (kv_row_*):
// only pages that hold a selected row are touched. Rows land in 32-row slabs at the DRAM page stride, the layout
// the converter tilizes in place; each row's four block scales are copied into column 0 of four FP32 broadcast
// tiles. A slab with fewer than 32 rows is zero-filled, a slab with none is not produced.
//
// Per chunk with own slabs the converter gets one control page [own slabs]; CTRL_END ends the stream.
//
// SM_READ_ONLY (read-rate measurement): the rows are read into the slab buffer and dropped, nothing is handed on.
// SM_ROLE_FROM_ARGS: the S-block index is runtime argument 0 (any set of cores, one converter per S-block).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#include "sm_common.hpp"

void kernel_main() {
#ifdef SM_NOP
    return;
#endif
    constexpr uint32_t cb_rows = get_named_compile_time_arg_val("cb_rows");
    constexpr uint32_t cb_scale = get_named_compile_time_arg_val("cb_scale");
    constexpr uint32_t cb_ctrl = get_named_compile_time_arg_val("cb_ctrl");
    constexpr uint32_t cb_idx = get_named_compile_time_arg_val("cb_idx");
    constexpr uint32_t USERS = get_named_compile_time_arg_val("users");
    constexpr uint32_t GROUPS = get_named_compile_time_arg_val("groups");
    constexpr uint32_t NCONV = get_named_compile_time_arg_val("converters");
    constexpr uint32_t CHUNK = get_named_compile_time_arg_val("chunk_rows");
    constexpr uint32_t STRIDE = get_named_compile_time_arg_val("row_stride");
    constexpr uint32_t LATENT = get_named_compile_time_arg_val("latent");
    constexpr uint32_t SCALE_BLOCKS = get_named_compile_time_arg_val("scale_blocks");
    constexpr uint32_t ROWS_PER_SLOT = get_named_compile_time_arg_val("rows_per_slot");
    constexpr uint32_t SLOT_STRIDE = get_named_compile_time_arg_val("slot_stride");
    constexpr uint32_t BANKS = get_named_compile_time_arg_val("banks");
    constexpr uint32_t SHARD_ROWS = get_named_compile_time_arg_val("shard_rows");
    static_assert(USERS <= sm::MAX_USERS && CHUNK % sm::TILE_ROWS == 0 && STRIDE % 64 == 0);

    constexpr auto sel_args = TensorAccessorArgs<0>();

#ifdef SM_ROLE_FROM_ARGS
    const sm::Role role{get_arg_val<uint32_t>(0), 0};
#else
    const sm::Role role =
        sm::role_from_position<get_named_compile_time_arg_val("x0_left"), get_named_compile_time_arg_val("x0_right")>();
#endif
    const uint32_t block = role.block;
    const uint32_t conv = role.slot;
    if (conv >= NCONV) {
        return;  // a core of the reader columns that holds no converter
    }
    const uint32_t kv_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sel_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t layer = get_common_arg_val<uint32_t>(2);

    const auto sel = TensorAccessor(sel_args, sel_addr);
    Noc noc;
    experimental::CB rows_cb(cb_rows);

    // Scratch (one page, reserved once and never pushed): [ids of the current chunk | ids of chunk 0 of user
    // `block` | header]. Chunk 0 of user `block` is this S-block's chunk whenever that user has a row, so its
    // ids are fetched together with the header instead of after it.
    cb_reserve_back(cb_idx, 1);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_idx));
    volatile tt_l1_ptr uint32_t* early_ids = ids + CHUNK;
    volatile tt_l1_ptr uint32_t* header = ids + 2 * CHUNK;
    const bool early = block < USERS;

    noc_async_read(sel.get_noc_addr(0), reinterpret_cast<uint32_t>(header), USERS * sizeof(uint32_t));
    if (early) {
        noc_async_read(sel.get_noc_addr(1 + block), reinterpret_cast<uint32_t>(early_ids), CHUNK * sizeof(uint32_t));
    }
    noc_async_read_barrier();
    uint32_t counts[sm::MAX_USERS];
    for (uint32_t u = 0; u < USERS; ++u) {
        counts[u] = header[u];
    }

    sm::for_each_chunk<USERS, GROUPS, CHUNK>(counts, block, [&](uint32_t u, uint32_t c, uint32_t valid) {
        const uint32_t own = sm::own_slabs<NCONV>(sm::ceil_div(valid, sm::TILE_ROWS), conv);
        if (own == 0) {
            return;
        }
        volatile tt_l1_ptr uint32_t* idx = early_ids;
        if (!(early && u == block && c == 0)) {
            idx = ids;
            noc_async_read(
                sel.get_noc_addr(1 + u, c * CHUNK * sizeof(uint32_t)),
                reinterpret_cast<uint32_t>(idx),
                valid * sizeof(uint32_t));
            noc_async_read_barrier();
        }
#ifndef SM_READ_ONLY
        cb_reserve_back(cb_ctrl, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctrl))[0] = own;
        cb_push_back(cb_ctrl, 1);
#endif

        const uint32_t page_base = (u * SLOT_STRIDE + layer) * ROWS_PER_SLOT;
        for (uint32_t i = 0; i < own; ++i) {
            const uint32_t first = (conv + i * NCONV) * sm::TILE_ROWS;
            const uint32_t rows = (valid - first) < sm::TILE_ROWS ? (valid - first) : sm::TILE_ROWS;

            cb_reserve_back(cb_rows, sm::TILE_ROWS);
            cb_reserve_back(cb_scale, SCALE_BLOCKS);
            const uint32_t dst = get_write_ptr(cb_rows);
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t page = page_base + idx[first + r];
                const uint64_t src = get_noc_addr_from_bank_id<true>(
                    sm::kv_row_bank<BANKS, SHARD_ROWS>(page),
                    kv_addr + sm::kv_row_offset<BANKS, SHARD_ROWS>(page, STRIDE));
                noc_async_read_one_packet(src, dst + r * STRIDE, STRIDE);
            }
            noc_async_read_barrier();
#ifndef SM_READ_ONLY
            volatile tt_l1_ptr uint32_t* packed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
            volatile tt_l1_ptr uint32_t* scales =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scale));
            for (uint32_t r = 0; r < sm::TILE_ROWS; ++r) {
                const uint32_t at = sm::tile_col0_offset(r);
                for (uint32_t blk = 0; blk < SCALE_BLOCKS; ++blk) {
                    scales[blk * 1024 + at] = r < rows ? packed[(r * STRIDE + LATENT) / 4 + blk] : 0;
                }
            }
            if (rows < sm::TILE_ROWS) {
                noc.async_write_zeros(rows_cb, (sm::TILE_ROWS - rows) * STRIDE, {.offset_bytes = rows * STRIDE});
                noc.write_zeros_l1_barrier();
            }
            cb_push_back(cb_rows, sm::TILE_ROWS);
            cb_push_back(cb_scale, SCALE_BLOCKS);
#endif
        }
    });

#ifndef SM_READ_ONLY
    cb_reserve_back(cb_ctrl, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctrl))[0] = sm::CTRL_END;
    cb_push_back(cb_ctrl, 1);
#endif
}
