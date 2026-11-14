// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 decode indexer score (task fuse-sparse-merge, B1), reader. Heads are the M rows: q [S,1,32 heads,128] TILE (page s*4+d), w = stock weights tile [1,1,32 rows,32 heads] (page 0, row r = query row r),
// keys [1,1,n,128] bf8 TILE (page kt*4+d, 1088 B). This core streams key row tiles [row0, row0+nrows) in batches of KC rows.
// CB_Q: S*4 q tiles (resident); CB_W: S masked weight tiles (only row r of tile r non-zero); CB_K: batches of KC*4 key tiles.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t KC = get_compile_time_arg_val(1);
    constexpr uint32_t CB_Q = get_compile_time_arg_val(2);
    constexpr uint32_t CB_W = get_compile_time_arg_val(3);
    constexpr uint32_t CB_K = get_compile_time_arg_val(4);
    constexpr uint32_t CB_TMP = get_compile_time_arg_val(5);
    constexpr uint32_t K_TILE_B = get_compile_time_arg_val(6);
    constexpr auto k_args = TensorAccessorArgs<7, 3>();
    constexpr auto q_args = TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr uint32_t TILE_B = 2048;
    const uint32_t k_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t row0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);
    const auto ka = TensorAccessor(k_args, k_addr, K_TILE_B);
    const auto qa = TensorAccessor(q_args, q_addr, TILE_B);
    const auto wa = TensorAccessor(w_args, w_addr, TILE_B);
    CircularBuffer cq(CB_Q), cw(CB_W), ck(CB_K), ct(CB_TMP);
    // keys first: the first batch is issued before the (small) q / w setup so the DRAM pipe starts at once
    uint32_t nb = (nrows + KC - 1) / KC;
    cq.reserve_back(S * 4);
    {
        const uint32_t p = cq.get_write_ptr();
        for (uint32_t i = 0; i < S * 4; ++i) {
            noc_async_read(qa.get_noc_addr(i), p + i * TILE_B, TILE_B);
        }
    }
    ct.reserve_back(1);
    noc_async_read(wa.get_noc_addr(0), ct.get_write_ptr(), TILE_B);
    noc_async_read_barrier();
    cq.push_back(S * 4);
    cw.reserve_back(S);
    {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.get_write_ptr());
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cw.get_write_ptr());
        for (uint32_t i = 0; i < S * TILE_B / 4; ++i) {
            dst[i] = 0;
        }
        for (uint32_t r = 0; r < S; ++r) {
            // row r (< 16) of a 32x32 tile: face 0 (cols 0..15) at word r*8, face 1 (cols 16..31) at word 128 + r*8; 16 bf16 = 8 words
            for (uint32_t i = 0; i < 8; ++i) {
                dst[r * (TILE_B / 4) + r * 8 + i] = src[r * 8 + i];
                dst[r * (TILE_B / 4) + 128 + r * 8 + i] = src[128 + r * 8 + i];
            }
        }
    }
    cw.push_back(S);
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t kc = (b * KC + KC <= nrows) ? KC : (nrows - b * KC);
        ck.reserve_back(kc * 4);
        const uint32_t p = ck.get_write_ptr();
        const uint32_t page0 = (row0 + b * KC) * 4;
        for (uint32_t i = 0; i < kc * 4; ++i) {
            noc_async_read(ka.get_noc_addr(page0 + i), p + i * K_TILE_B, K_TILE_B);
        }
        noc_async_read_barrier();
        ck.push_back(kc * 4);
    }
}
