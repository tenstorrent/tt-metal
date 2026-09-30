// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the V4.1 candidate scoring (candscore_compute.cpp): per query row it collects row 0 of the compute's
// K * 8 / 32 result tiles (32 candidate scores each, candidate rows in block rank order) into the output row
// (bf16 row-major [R, K * 8]), sets -inf where the candidate row is not visible to the query (t > p) or its block
// id is the sentinel (>= nblocks), and writes the row. Two output rows alternate so a row's write overlaps the next
// row's collection.
//
// Query position: row i of this chip is chunk query q = query_first + i (query_first read from the per-chip pin
// tensor, ``V41ChunkTables.query_first()``), absolute position p = start + q; ratio-1 row t is visible iff t <= p.
//
// compile_time_args = [cb_out, cb_scratch, K, TensorAccessorArgs(ids)..., TensorAccessorArgs(out)...,
//                      TensorAccessorArgs(pin)...]
// runtime args      = [ids_addr, out_addr, pin_addr, row_start, row_count, start, nblocks]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t pin_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);
    const uint32_t row_count = get_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(5);
    const uint32_t nblocks = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr auto ids_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto pin_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t BLOCK = 8;
    constexpr uint32_t TILE_ROWS = K * BLOCK / 32;
    constexpr uint32_t ROW_BYTES = K * BLOCK * 2;
    constexpr uint32_t FACE1_U32 = 16 * 16 / 2;  // word offset of face 1 in a bf16 tile
    constexpr uint16_t NEG_INF = 0xFF80;
    constexpr uint32_t NEGINF_PAIR = 0xFF80FF80u;

    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto pin = TensorAccessor(pin_args, pin_addr);
    DataflowBuffer res(cb_out);
    DataflowBuffer scratch(cb_scratch);  // [0, 64): pin page | ids row (K * 4) | two output rows (ROW_BYTES each)

    const uint32_t base_l1 = scratch.get_write_ptr();
    const uint32_t ids_l1 = base_l1 + 64;
    const uint32_t buf_l1[2] = {ids_l1 + K * 4, ids_l1 + K * 4 + ROW_BYTES};
    volatile tt_l1_ptr uint32_t* id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);
    noc_async_read(pin.get_noc_addr(0, 0), base_l1, 32);
    noc_async_read_barrier();
    const uint32_t query_first = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_l1)[0];

    uint32_t slot = 0;
    for (uint32_t row = row_start; row < row_start + row_count; ++row) {
        const uint32_t p = start + query_first + row;
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_l1[slot]);
        volatile tt_l1_ptr uint16_t* dst16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_l1[slot]);
        noc_async_read(ids.get_noc_addr(row, 0), ids_l1, K * 4);
        noc_async_read_barrier();

        for (uint32_t c = 0; c < TILE_ROWS; ++c) {
            res.wait_front(1);
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res.get_read_ptr());
            for (uint32_t i = 0; i < 8; ++i) {
                dst[c * 16 + i] = t[i];
                dst[c * 16 + 8 + i] = t[FACE1_U32 + i];
            }
            res.pop_front(1);
        }
        for (uint32_t r = 0; r < K; ++r) {
            const uint32_t b = id[r];
            if (b >= nblocks) {
                for (uint32_t i = 0; i < BLOCK / 2; ++i) {
                    dst[r * (BLOCK / 2) + i] = NEGINF_PAIR;
                }
            } else if (b * BLOCK + BLOCK - 1 > p) {
                for (uint32_t j = 0; j < BLOCK; ++j) {
                    if (b * BLOCK + j > p) {
                        dst16[r * BLOCK + j] = NEG_INF;
                    }
                }
            }
        }
        noc_async_write_barrier();  // the previous row's write has left its buffer
        noc_async_write(buf_l1[slot], out.get_noc_addr(row, 0), ROW_BYTES);
        slot ^= 1;
    }
    noc_async_write_barrier();
}
