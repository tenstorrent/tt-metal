// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// qk_heads (lever L06), reader.  One core = (tile column n, NT users t0 .. t0+NT-1): reads the 9 WHOLE source tiles of
// column n (kv tile n and the q tiles (k-1)*16+n, k = 1..8; 2 KB reads, no small-chunk DRAM traffic), builds the NT
// output tiles (row 0 = kv, rows 1..8 = heads, rest zero) in L1.  ROPE = 0 (n < 14): writes them to out page t*16+n.
// ROPE = 1 (n = 14, 15): builds the rotation matrices from the table rows and hands X and R to compute; the writer
// finishes.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

typedef volatile tt_l1_ptr uint32_t* PW32;

// row t of the 9 source tiles (src_base + k * 2048) -> rows 0..8 of the output tile at dst: 2 x 32 B (face halves) per
// row, local NoC writes
FORCE_INLINE void copy_rows(uint32_t src_base, uint32_t dst, uint32_t t) {
    const uint32_t rb = (((t >> 4) << 10) + ((t & 15) << 5));  // byte offset of row t, face half 0 (half 1: + 512)
    const uint32_t mx = my_x[noc_index], my = my_y[noc_index];
    for (uint32_t k = 0; k < 9; ++k) {
        noc_async_write(src_base + k * 2048 + rb, get_noc_addr(mx, my, dst + k * 32), 32);
        noc_async_write(src_base + k * 2048 + 512 + rb, get_noc_addr(mx, my, dst + 512 + k * 32), 32);
    }
}

void kernel_main() {
    constexpr uint32_t ROPE = get_compile_time_arg_val(0);
    constexpr uint32_t cb_src = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = get_compile_time_arg_val(2);
    constexpr uint32_t cb_r = get_compile_time_arg_val(3);
    constexpr uint32_t cb_s = get_compile_time_arg_val(4);
    constexpr auto kv_args = TensorAccessorArgs<5>();
    constexpr auto q_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto tc_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto ts_args = TensorAccessorArgs<tc_args.next_compile_time_args_offset()>();

    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    Noc noc;
    const auto kv_acc = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto q_acc = TensorAccessor(q_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto o_acc = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2), 2048);
    experimental::CB cbsrc(cb_src), cbx(cb_x);

    cbsrc.reserve_back(9);
    cbx.reserve_back(nt);
    noc.async_read(kv_acc, cbsrc, 2048, {.page_id = n, .offset_bytes = 0}, {.offset_bytes = 0});
    for (uint32_t k = 1; k < 9; ++k) {
        noc.async_read(
            q_acc, cbsrc, 2048, {.page_id = (k - 1) * 16 + n, .offset_bytes = 0}, {.offset_bytes = k * 2048});
    }
    noc.async_write_zeros(cbx, nt * 2048, {.offset_bytes = 0});
    if constexpr (ROPE) {
        const auto tc_acc = TensorAccessor(tc_args, get_common_arg_val<uint32_t>(3), 2048);
        const auto ts_acc = TensorAccessor(ts_args, get_common_arg_val<uint32_t>(4), 2048);
        experimental::CB cbr(cb_r), cbs(cb_s);
        cbr.reserve_back(nt);
        cbs.reserve_back(1);
        noc.async_write_zeros(cbr, nt * 2048, {.offset_bytes = 0});
        for (uint32_t i = 0; i < nt;
             ++i) {  // table rows of user t0 + i -> scratch 256 B each (cos f0, cos f1, sin f0, sin f1; 64 B chunks)
            const uint32_t pg = (t0 + i) * 16 + n;
            noc.async_read(tc_acc, cbs, 64, {.page_id = pg, .offset_bytes = 0}, {.offset_bytes = i * 256});
            noc.async_read(tc_acc, cbs, 64, {.page_id = pg, .offset_bytes = 512}, {.offset_bytes = i * 256 + 64});
            noc.async_read(ts_acc, cbs, 64, {.page_id = pg, .offset_bytes = 0}, {.offset_bytes = i * 256 + 128});
            noc.async_read(ts_acc, cbs, 64, {.page_id = pg, .offset_bytes = 512}, {.offset_bytes = i * 256 + 192});
        }
        noc.write_zeros_l1_barrier();
        noc.async_read_barrier();
        const uint32_t src_base = cbsrc.get_write_ptr();
        const uint32_t dst_base = cbx.get_write_ptr();
        for (uint32_t i = 0; i < nt; ++i) {
            const uint32_t t = t0 + i;
            copy_rows(src_base, dst_base + i * 2048, t);
        }
        noc_async_write_barrier();
        volatile tt_l1_ptr uint16_t* sc0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbs.get_write_ptr());
        volatile tt_l1_ptr uint16_t* R0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbr.get_write_ptr());
#define EL(r, q) ((((((r) >> 4) << 1) + ((q) >> 4)) << 8) + (((r) & 15) << 4) + ((q) & 15))
        for (uint32_t i = 0; i < nt; ++i) {
            volatile tt_l1_ptr uint16_t* sc = sc0 + i * 128;
            volatile tt_l1_ptr uint16_t* R = R0 + i * 1024;
            for (uint32_t e = 0; e < 32; e += 2) {
                const uint32_t half = e >> 4;
                const uint16_t c = sc[half * 32 + (e & 15)];
                const uint16_t s = sc[64 + half * 32 + (e & 15)];
                R[EL(e, e)] = c;
                R[EL(e + 1, e + 1)] = c;
                R[EL(e, e + 1)] = s;
                R[EL(e + 1, e)] = s ^ 0x8000;
            }
        }
#undef EL
        cbr.push_back(nt);
        cbx.push_back(nt);
    } else {
        noc.write_zeros_l1_barrier();
        noc.async_read_barrier();
        const uint32_t src_base = cbsrc.get_write_ptr();
        const uint32_t dst_base = cbx.get_write_ptr();
        for (uint32_t i = 0; i < nt; ++i) {
            const uint32_t t = t0 + i;
            copy_rows(src_base, dst_base + i * 2048, t);
        }
        noc_async_write_barrier();  // local row copies done
        for (uint32_t i = 0; i < nt; ++i) {
            noc.async_write(
                cbx, o_acc, 2048, {.offset_bytes = i * 2048}, {.page_id = (t0 + i) * 16 + n, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
}
