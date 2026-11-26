// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b model input (tt/mk_head.py MkEmbed), one owner core (BRISC): the token of every row t < T (word 0 of row
// t of the device-resident token tensor [4, 64] uint32: the previous step's select-core output, or a host write) ->
// its embedding row's 64 columns of this core (column tiles c0, c0 + 1; the table is bf16 ROW_MAJOR [V, 4096], one
// 8-KB page per vocab id) -> the stacked residual [T, 1, 4, 4096] fp32 tiles (t, c0 + j): rows 0..3 = the 4 identical
// hyper-connection streams (reference embed_to_streams), every other row 0. bf16 -> fp32 is exact (bits << 16).
// Compile-time args: TensorAccessorArgs of tokens, table, res. Runtime args: tokens, table, res (chunk 0) addresses,
// c0, T (<= 32), then the residual addresses of chunks 1 .. (T - 1) / 4 (row t: chunk t / 4, local row t % 4). CBs:
// EMB_CB (4096 + 384 + 64 B: 128 B embedding slice, 256 B token row, one fp32 tile, alignment).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/head/head_common.hpp"

using namespace head;

void kernel_main() {
    constexpr auto tok_args = TensorAccessorArgs<0>();
    constexpr auto emb_args = TensorAccessorArgs<tok_args.next_compile_time_args_offset()>();
    constexpr auto res_args = TensorAccessorArgs<emb_args.next_compile_time_args_offset()>();
    const auto tok = TensorAccessor(tok_args, get_arg_val<uint32_t>(0));
    const auto emb = TensorAccessor(emb_args, get_arg_val<uint32_t>(1));
    const uint32_t c0 = get_arg_val<uint32_t>(3);
    const uint32_t T = get_arg_val<uint32_t>(4);

    const uint32_t base = (get_write_ptr(EMB_CB) + 63u) & ~63u;  // DRAM reads land 64-B aligned
    const uint32_t erow = base;                                  // 128 B: 64 bf16 embedding values
    const uint32_t trow = base + 128;                            // 256 B: the token row
    const uint32_t tile = base + 384;                            // one fp32 tile
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
#if defined(EMB_FLAT)
    // the FLAT residual [1, 1, T, 16384] (T > 4: 7a's native-t layers): tile (s, c0 + j) of the single row block holds
    // token t in row t; the 4 streams are identical -> one tile per column tile j, written to the 4 stream pages
    const auto res = TensorAccessor(res_args, get_arg_val<uint32_t>(2));
    for (uint32_t j = 0; j < COLS; ++j) {
        for (uint32_t off = 0; off < FP32_TILE_BYTES; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, tile + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
        for (uint32_t t = 0; t < T; ++t) {
            noc_async_read_page(t, tok, trow);
            noc_async_read_barrier();
            const uint32_t id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(trow)[0];
            noc_async_read(emb.get_noc_addr(id, (c0 + j) * 64), erow, 64);  // the 32 columns of tile c0 + j
            noc_async_read_barrier();
            volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(erow);
            for (uint32_t c = 0; c < TILE; ++c) {
                d[tile_off(t, c)] = static_cast<uint32_t>(e[c]) << 16;
            }
        }
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            noc_async_write_page(s * COL_TILES + c0 + j, res, tile);
        }
        noc_async_write_barrier();
    }
    return;
#endif
    for (uint32_t t = 0; t < T; ++t) {
        noc_async_read_page(t, tok, trow);
        noc_async_read_barrier();
        const uint32_t id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(trow)[0];
        noc_async_read(emb.get_noc_addr(id, c0 * 64), erow, 128);  // columns 32 c0 .. 32 c0 + 63 (2 B each)
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(erow);
        for (uint32_t j = 0; j < COLS; ++j) {
            for (uint32_t off = 0; off < FP32_TILE_BYTES; off += MEM_ZEROS_SIZE) {
                noc_async_read(zeros, tile + off, MEM_ZEROS_SIZE);
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
            for (uint32_t c = 0; c < TILE; ++c) {
                const uint32_t v = static_cast<uint32_t>(e[j * TILE + c]) << 16;
                for (uint32_t s = 0; s < N_STREAMS; ++s) {
                    d[tile_off(s, c)] = v;
                }
            }
            const uint32_t ch = t / CHUNK_ROWS;
            const auto res = TensorAccessor(res_args, get_arg_val<uint32_t>(ch == 0 ? 2 : 5 + ch - 1));
            noc_async_write_page((t % CHUNK_ROWS) * COL_TILES + c0 + j, res, tile);
            noc_async_write_barrier();
        }
    }
}
