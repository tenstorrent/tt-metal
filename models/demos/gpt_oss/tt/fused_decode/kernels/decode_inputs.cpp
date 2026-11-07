// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode step inputs of the fused decode path (fused_decode/inputs.py), one core (BRISC):
//   - the token's embedding row: row tokens[0] of the row-major BF16 embedding table -> emb (one row-major page);
//   - the RoPE rows of the fused Q/K rotary op: for user u in [0, users) (Q users then K users), row rot[u] of the
//     row-major BF16 cos / sin tables ([.., positions, head_dim], one page per position) -> row 0 of the head_dim / 32
//     tiles of user u's shard of the height-sharded cos / sin outputs (the rotary op broadcasts row 0, rows 1..31 are
//     not read).
//
// EMB_VSHARD = Vd (vocab-sharded table, fused_decode/embed.py): the local row is tokens[0] - d Vd when that is < Vd (d
// Vd: entry 0 of the offsets tensor), else row Vd (zeros); it goes to the flat partial of the first layer boundary's
// all-reduce (value h at byte 2 h, boundary core) instead of emb, a fresh flat residual is zeroed, and the boundary's
// fabric sender (sending_program, site "entry") is notified.
//
// runtime args: [tok_addr, emb_table_addr, emb_addr, rot_addr, cos_table_addr, sin_table_addr, cos_addr, sin_addr,
//                (noc_x, noc_y) x users] (+ EMB_VSHARD: [off_addr, partial_addr, residual_addr])

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t tok_addr = get_arg_val<uint32_t>(0);
    const uint32_t emb_table_addr = get_arg_val<uint32_t>(1);
    const uint32_t emb_addr = get_arg_val<uint32_t>(2);
    const uint32_t rot_addr = get_arg_val<uint32_t>(3);
    const uint32_t cos_table_addr = get_arg_val<uint32_t>(4);
    const uint32_t sin_table_addr = get_arg_val<uint32_t>(5);
    const uint32_t cos_addr = get_arg_val<uint32_t>(6);
    const uint32_t sin_addr = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);  // embedding row
    constexpr uint32_t tok_page = get_compile_time_arg_val(2);
    constexpr uint32_t rot_page = get_compile_time_arg_val(3);
    constexpr uint32_t users = get_compile_time_arg_val(4);
    constexpr uint32_t dim_tiles = get_compile_time_arg_val(5);
    constexpr auto tok_args = TensorAccessorArgs<6>();
    constexpr auto et_args = TensorAccessorArgs<tok_args.next_compile_time_args_offset()>();
    constexpr auto emb_args = TensorAccessorArgs<et_args.next_compile_time_args_offset()>();
    constexpr auto rot_args = TensorAccessorArgs<emb_args.next_compile_time_args_offset()>();
    constexpr auto ct_args = TensorAccessorArgs<rot_args.next_compile_time_args_offset()>();
    constexpr auto st_args = TensorAccessorArgs<ct_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;
#ifdef EMB_VSHARD
    constexpr auto off_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
    constexpr auto part_args = TensorAccessorArgs<off_args.next_compile_time_args_offset()>();
    constexpr auto res_args = TensorAccessorArgs<part_args.next_compile_time_args_offset()>();
    constexpr uint32_t vx = res_args.next_compile_time_args_offset();
    constexpr uint32_t flat_bytes = get_compile_time_arg_val(vx);  // the flat residual's pages
    constexpr uint32_t notify_x = get_compile_time_arg_val(vx + 1);
    constexpr uint32_t notify_y = get_compile_time_arg_val(vx + 2);
    constexpr uint32_t notify_sem = get_compile_time_arg_val(vx + 3);
#endif

    const auto s_tok = TensorAccessor(tok_args, tok_addr, tok_page);
    const auto s_et = TensorAccessor(et_args, emb_table_addr, row_bytes);
    const auto s_emb = TensorAccessor(emb_args, emb_addr, row_bytes);
    const auto s_rot = TensorAccessor(rot_args, rot_addr, rot_page);
    constexpr uint32_t rope_row = dim_tiles * 64;  // head_dim BF16 values
    const auto s_ct = TensorAccessor(ct_args, cos_table_addr, rope_row);
    const auto s_st = TensorAccessor(st_args, sin_table_addr, rope_row);

    // scratch: [tokens page | rot page | embedding row | cos rows | sin rows], 64-byte aligned pieces
    constexpr uint32_t a64 = 64;
    constexpr uint32_t off_rot = (tok_page + a64 - 1) / a64 * a64;
    constexpr uint32_t off_row = off_rot + (rot_page + a64 - 1) / a64 * a64;
    constexpr uint32_t off_cos = off_row + (row_bytes + a64 - 1) / a64 * a64;
    constexpr uint32_t off_sin = off_cos + users * rope_row;
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);

    noc_async_read(s_tok.get_noc_addr(0), scr, tok_page);
    noc_async_read(s_rot.get_noc_addr(0), scr + off_rot, rot_page);
#ifdef EMB_VSHARD
    constexpr uint32_t off_vs = off_sin + users * rope_row;  // offsets page (128 B), then flat_bytes of zeros
    const uint32_t a_users = 8 + 2 * users;
    const auto s_off = TensorAccessor(off_args, get_arg_val<uint32_t>(a_users), 128);
    noc_async_read(s_off.get_noc_addr(0), scr + off_vs, 128);
#endif
    noc_async_read_barrier();
#ifdef EMB_VSHARD
    const uint32_t local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr)[0] -
                           reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + off_vs)[0];
    const uint32_t token = local < EMB_VSHARD ? local : EMB_VSHARD;
#else
    const uint32_t token = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr)[0];
#endif
    volatile tt_l1_ptr uint32_t* rot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + off_rot);

    noc_async_read(s_et.get_noc_addr(token), scr + off_row, row_bytes);
    for (uint32_t u = 0; u < users; ++u) {
        noc_async_read(s_ct.get_noc_addr(rot[u]), scr + off_cos + u * rope_row, rope_row);
        noc_async_read(s_st.get_noc_addr(rot[u]), scr + off_sin + u * rope_row, rope_row);
    }
    noc_async_read_barrier();
#ifdef EMB_VSHARD
    {
        const auto s_part = TensorAccessor(part_args, get_arg_val<uint32_t>(a_users + 1), tile);
        const auto s_res = TensorAccessor(res_args, get_arg_val<uint32_t>(a_users + 2), tile);
        // the flat layouts are one core's contiguous tile pages: value h at byte 2 h from page 0
        noc_async_write(scr + off_row, s_part.get_noc_addr(0), row_bytes);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + off_vs + 128);
        for (uint32_t i = 0; i < flat_bytes / 4; ++i) {
            z[i] = 0;
        }
        noc_async_write(scr + off_vs + 128, s_res.get_noc_addr(0), flat_bytes);
    }
#else
    noc_async_write(scr + off_row, s_emb.get_noc_addr(0), row_bytes);
#endif

    for (uint32_t u = 0; u < users; ++u) {
        const uint32_t x = get_arg_val<uint32_t>(8 + 2 * u);
        const uint32_t y = get_arg_val<uint32_t>(9 + 2 * u);
        // row 0 of tile t: values 32 t .. 32 t + 15 at face 0, 32 t + 16 .. 32 t + 31 at face 1 (byte 512)
        for (uint32_t t = 0; t < dim_tiles; ++t) {
            const uint32_t c = scr + off_cos + u * rope_row + t * 64;
            const uint32_t s = scr + off_sin + u * rope_row + t * 64;
            noc_async_write(c, get_noc_addr(x, y, cos_addr + t * tile), 32);
            noc_async_write(c + 32, get_noc_addr(x, y, cos_addr + t * tile + 512), 32);
            noc_async_write(s, get_noc_addr(x, y, sin_addr + t * tile), 32);
            noc_async_write(s + 32, get_noc_addr(x, y, sin_addr + t * tile + 512), 32);
        }
    }
    noc_async_write_barrier();
#ifdef EMB_VSHARD
    // the partial is written: the boundary's fabric sender (sending_program) may send it
    noc_semaphore_inc(get_noc_addr(notify_x, notify_y, get_semaphore(notify_sem)), 1);
    noc_async_atomic_barrier();
#endif
}
