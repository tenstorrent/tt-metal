// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 decode indexer score, writer. Output tile of key row tile j: row r (r < S) holds the 32 scores of query row r; they are gathered into a per-row strip in L1 (64 B per key row tile:
// face 0 row r + face 1 row r) and written as one contiguous segment of row r of out bf16 RM [1,1,S,n]. Keys at position >= kv_len get -inf.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t KC = get_compile_time_arg_val(1);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
    constexpr uint32_t CB_STRIP = get_compile_time_arg_val(3);
    constexpr uint32_t ROW_B = get_compile_time_arg_val(4);   // output page bytes (n * 2 / PPR)
    constexpr uint32_t MAXR = get_compile_time_arg_val(5);    // max key row tiles per core
    constexpr uint32_t KVT = get_compile_time_arg_val(
        6);  // 1: kv_len is element 0 of a device tensor (trace-safe: per-device count of valid keys), else the runtime
             // arg; 2: element r is the count of query row r (verify rows: causal per row)
    constexpr uint32_t CB_KV = get_compile_time_arg_val(7);
    constexpr uint32_t PPR =
        get_compile_time_arg_val(8);  // output pages per query row (1 = out [1,1,S,n]; B = out [1,1,S*B,n/B],
                                      // HY4_IDX_BLK=1: the blocked layout of the select pipeline, no reshape copies)
    constexpr auto o_args = TensorAccessorArgs<9, 1>();
    constexpr auto k_args = TensorAccessorArgs<o_args.next_compile_time_args_offset(), o_args.next_common_runtime_args_offset()>();
    constexpr uint32_t TILE_B = 2048;
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    uint32_t kv_len = get_common_arg_val<uint32_t>(1);
    uint32_t kv_row[S];
    const uint32_t row0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);
    const auto oa = TensorAccessor(o_args, o_addr, ROW_B);
    CircularBuffer co(CB_OUT), cs(CB_STRIP);
    if constexpr (KVT) {
        CircularBuffer ck(CB_KV);
        ck.reserve_back(1);
        const uint32_t kp = ck.get_write_ptr();
        const auto ka = TensorAccessor(k_args, get_common_arg_val<uint32_t>(2), 64);
        noc_async_read(ka.get_noc_addr(0, 0), kp, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        kv_len = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kp);
        for (uint32_t r = 0; r < S; ++r) {
            kv_row[r] = KVT == 2 ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kp)[r] : kv_len;
        }
    } else {
        for (uint32_t r = 0; r < S; ++r) {
            kv_row[r] = kv_len;
        }
    }
    cs.reserve_back(1);
    const uint32_t strip = cs.get_write_ptr();  // S rows x (MAXR * 64 B)
    constexpr uint32_t SROW = MAXR * 64;
    const uint32_t nb = (nrows + KC - 1) / KC;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t kc = (b * KC + KC <= nrows) ? KC : (nrows - b * KC);
        co.wait_front(kc);
        const uint32_t base = co.get_read_ptr();
        for (uint32_t col = 0; col < kc; ++col) {
            const uint32_t t = base + col * TILE_B;
            for (uint32_t r = 0; r < S; ++r) {
                volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t + r * 32);
                volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t + 512 + r * 32);
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(strip + r * SROW + (b * KC + col) * 64);
                for (uint32_t i = 0; i < 8; ++i) {
                    d[i] = s0[i];
                    d[8 + i] = s1[i];
                }
            }
        }
        co.pop_front(kc);
    }
    // mask: global key index >= kv_len -> -inf (bf16 0xFF80)
    const uint32_t first = row0 * 32;
    for (uint32_t r = 0; r < S; ++r) {
        if (kv_row[r] < first + nrows * 32) {
            volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(strip + r * SROW);
            uint32_t from = kv_row[r] > first ? kv_row[r] - first : 0;
            for (uint32_t e = from; e < nrows * 32; ++e) {
                d[e] = 0xFF80;
            }
        }
    }
    for (uint32_t r = 0; r < S; ++r) {
        uint32_t src = strip + r * SROW, off = first * 2, rem = nrows * 64;
        while (rem) {  // split at output page boundaries (PPR > 1)
            const uint32_t pg = off / ROW_B, po = off % ROW_B;
            const uint32_t nn = (rem < ROW_B - po) ? rem : (ROW_B - po);
            noc_async_write(src, oa.get_noc_addr(r * PPR + pg, po), nn);
            src += nn;
            off += nn;
            rem -= nn;
        }
    }
    noc_async_write_barrier();
}
