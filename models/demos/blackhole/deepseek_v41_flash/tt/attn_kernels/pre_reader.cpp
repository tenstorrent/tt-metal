// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Attention pre program, reader. ROPE=0 (tile columns 0..13): assemble the q-head tile of (user t, column n) and write
// it + the cache rows itself.  ROPE=1 (columns 14,15): assemble the tile, build the rotation matrix R (see
// rope_reader.cpp) and hand both to compute; the writer finishes.
#include "pre_common.hpp"

void kernel_main() {
    constexpr uint32_t ROPE = get_compile_time_arg_val(0);
    constexpr uint32_t HAS_LAT = get_compile_time_arg_val(1);
    constexpr uint32_t CACHE_PT = get_compile_time_arg_val(2);
    constexpr uint32_t cb_x = get_compile_time_arg_val(3);
    constexpr uint32_t cb_r = get_compile_time_arg_val(4);
    constexpr uint32_t cb_s = get_compile_time_arg_val(5);
    constexpr auto kv_args = TensorAccessorArgs<6>();
    constexpr auto q_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto l_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto tc_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto ts_args = TensorAccessorArgs<tc_args.next_compile_time_args_offset()>();

    const uint32_t a0 = get_arg_val<uint32_t>(0);  // ROPE=1: work id w (t = w>>1, n = 14 + (w&1)); ROPE=0: first item
    const uint32_t a1 = get_arg_val<uint32_t>(1);  // ROPE=0: item count (item i: t = i / 14, n = i % 14)
    Noc noc;
    const auto kv_acc = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto q_acc = TensorAccessor(q_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto l_acc = TensorAccessor(l_args, get_common_arg_val<uint32_t>(2), 2048);
    const auto c_acc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(3), 2048);
    const auto o_acc = TensorAccessor(o_args, get_common_arg_val<uint32_t>(4), 2048);
    const auto p_acc = TensorAccessor(p_args, get_common_arg_val<uint32_t>(5), 64);
    const auto m_acc = TensorAccessor(m_args, get_common_arg_val<uint32_t>(6), 64);
    experimental::CB cbx(cb_x), cbs(cb_s);
    cbs.reserve_back(1);
    cbx.reserve_back(1);

    if constexpr (ROPE) {
        const auto tc_acc = TensorAccessor(tc_args, get_common_arg_val<uint32_t>(7), 2048);
        const auto ts_acc = TensorAccessor(ts_args, get_common_arg_val<uint32_t>(8), 2048);
        experimental::CB cbr(cb_r);
        cbr.reserve_back(1);
        const uint32_t t = a0 >> 1, n = 14 + (a0 & 1);
        PW32 o = reinterpret_cast<PW32>(cbx.get_write_ptr());
        noc.async_write_zeros(cbx, 2048, {.offset_bytes = 0});
        noc.async_write_zeros(cbr, 2048, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        // table rows -> scratch at PATCH_OFF (4 x 64 B)
        noc.async_read(tc_acc, cbs, 64, {.page_id = t * 16 + n, .offset_bytes = 0}, {.offset_bytes = PATCH_OFF});
        noc.async_read(tc_acc, cbs, 64, {.page_id = t * 16 + n, .offset_bytes = 512}, {.offset_bytes = PATCH_OFF + 64});
        noc.async_read(ts_acc, cbs, 64, {.page_id = t * 16 + n, .offset_bytes = 0}, {.offset_bytes = PATCH_OFF + 128});
        noc.async_read(
            ts_acc, cbs, 64, {.page_id = t * 16 + n, .offset_bytes = 512}, {.offset_bytes = PATCH_OFF + 192});
        noc.async_read_barrier();
        build_tile(noc, kv_acc, q_acc, cbs, o, t, n);
        volatile tt_l1_ptr uint16_t* sc =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbs.get_write_ptr() + PATCH_OFF);
        volatile tt_l1_ptr uint16_t* R = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbr.get_write_ptr());
#define EL(r, q) ((((((r) >> 4) << 1) + ((q) >> 4)) << 8) + (((r) & 15) << 4) + ((q) & 15))
        for (uint32_t e = 0; e < 32; e += 2) {
            const uint32_t half = e >> 4;
            const uint16_t c = sc[half * 32 + (e & 15)];
            const uint16_t s = sc[64 + half * 32 + (e & 15)];
            R[EL(e, e)] = c;
            R[EL(e + 1, e + 1)] = c;
            R[EL(e, e + 1)] = s;
            R[EL(e + 1, e)] = s ^ 0x8000;
        }
#undef EL
        cbr.push_back(1);
        cbx.push_back(1);
    } else {
        PW32 o = reinterpret_cast<PW32>(cbx.get_write_ptr());
        for (uint32_t i = a0; i < a0 + a1; ++i) {
            const uint32_t t = i / 14, n = i - t * 14;
            noc.async_write_zeros(cbx, 2048, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
            build_tile(noc, kv_acc, q_acc, cbs, o, t, n);
            noc.async_write(cbx, o_acc, 2048, {.offset_bytes = 0}, {.page_id = t * 16 + n, .offset_bytes = 0});
            noc.async_write_barrier();
            cache_writes<HAS_LAT, CACHE_PT>(noc, c_acc, l_acc, p_acc, m_acc, cbs, o, t, n);
        }
    }
}
