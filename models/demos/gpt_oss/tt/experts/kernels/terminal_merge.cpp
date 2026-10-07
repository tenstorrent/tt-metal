// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Per-device candidate merge of the fused decode sampler (tt/decode_terminal.py), one core.
//
// Input: the row-wise top-k of the folded logits (ttnn.topk, sorted): values [1, 1, 32, k] BF16 and positional
// indices [1, 1, 32, k] UINT16, one 32x32 tile each (k = 32). Row r of the folded logits holds the local vocab ids
// [r * row_len, (r + 1) * row_len), so candidate (r, c) is local id r * row_len + index(r, c).
// Output: the device's top-k of the 32 x k candidates, ordered by (value descending, global id ascending), in row 0
// of `vout` (BF16 [1, 1, 32, k] tile) and of `iout` (UINT32 [1, 1, 32, k] row-major, global id = id_offset + local
// id). Rows 1..31 are left as they are (constant filler written at allocation).
//
// Every row is sorted, so the top-k is a k-step merge of the 32 row heads; ties are then ordered by id (topk is not
// stable).
//
// runtime args: [v_addr, i_addr, vout_addr, iout_addr, id_offset]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
inline uint32_t elem(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
// BF16 bits -> unsigned key with the same order as the value.
inline uint32_t key_of(uint16_t b) { return (b & 0x8000) ? (~b & 0xFFFFu) : (b | 0x8000u); }
}  // namespace

void kernel_main() {
    const uint32_t v_addr = get_arg_val<uint32_t>(0);
    const uint32_t i_addr = get_arg_val<uint32_t>(1);
    const uint32_t vout_addr = get_arg_val<uint32_t>(2);
    const uint32_t iout_addr = get_arg_val<uint32_t>(3);
    const uint32_t id_offset = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t row_len = get_compile_time_arg_val(1);
    constexpr uint32_t k = get_compile_time_arg_val(2);
    constexpr auto v_args = TensorAccessorArgs<3>();
    constexpr auto i_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto vo_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto io_args = TensorAccessorArgs<vo_args.next_compile_time_args_offset()>();
    static_assert(k == 32, "one 32x32 candidate tile per row");

    const auto s_v = TensorAccessor(v_args, v_addr, 2048);
    const auto s_i = TensorAccessor(i_args, i_addr, 2048);
    const auto s_vo = TensorAccessor(vo_args, vout_addr, 2048);
    const auto s_io = TensorAccessor(io_args, iout_addr, k * 4);

    cb_reserve_back(cb_scr, 1);
    const uint32_t base = get_write_ptr(cb_scr);  // [v tile 2048 | i tile 2048 | out values 1024 | out ids 128]
    noc_async_read(s_v.get_noc_addr(0), base, 2048);
    noc_async_read(s_i.get_noc_addr(0), base + 2048, 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    volatile tt_l1_ptr uint16_t* ix = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + 2048);
    volatile tt_l1_ptr uint16_t* ov = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + 4096);
    volatile tt_l1_ptr uint32_t* oi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 5120);

    uint32_t head[32];
    uint32_t head_key[32];
    uint32_t head_id[32];
    for (uint32_t r = 0; r < 32; ++r) {
        head[r] = 0;
        const uint32_t e = elem(r, 0);
        head_key[r] = key_of(v[e]);
        head_id[r] = r * row_len + ix[e];
    }
    uint32_t sel_key[k];
    uint32_t sel_id[k];
    for (uint32_t j = 0; j < k; ++j) {
        uint32_t best = 0;
        for (uint32_t r = 1; r < 32; ++r) {
            if (head_key[r] > head_key[best] || (head_key[r] == head_key[best] && head_id[r] < head_id[best])) {
                best = r;
            }
        }
        sel_key[j] = head_key[best];
        sel_id[j] = head_id[best];
        const uint32_t c = ++head[best];
        if (c < k) {
            const uint32_t e = elem(best, c);
            head_key[best] = key_of(v[e]);
            head_id[best] = best * row_len + ix[e];
        } else {
            head_key[best] = 0;  // below every real value
            head_id[best] = 0xFFFFFFFFu;
        }
    }
    // Keys are non-increasing; order equal keys by id.
    for (uint32_t j = 1; j < k; ++j) {
        const uint32_t kk = sel_key[j], id = sel_id[j];
        uint32_t m = j;
        while (m > 0 && sel_key[m - 1] == kk && sel_id[m - 1] > id) {
            sel_key[m] = sel_key[m - 1];
            sel_id[m] = sel_id[m - 1];
            --m;
        }
        sel_key[m] = kk;
        sel_id[m] = id;
    }
    // Row 0 of a 32x32 tile: columns 0..15 at face 0 row 0, 16..31 at face 1 row 0.
    for (uint32_t j = 0; j < k; ++j) {
        const uint32_t kk = sel_key[j];
        ov[j < 16 ? j : 256 + (j - 16)] = (kk & 0x8000) ? (kk & 0x7FFF) : (~kk & 0xFFFF);
        oi[j] = id_offset + sel_id[j];
    }
    const uint64_t vo = s_vo.get_noc_addr(0);
    noc_async_write(base + 4096, vo, 32);
    noc_async_write(base + 4096 + 512, vo + 512, 32);
    noc_async_write(base + 5120, s_io.get_noc_addr(0), k * 4);
    noc_async_write_barrier();
}
