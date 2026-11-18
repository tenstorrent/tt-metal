// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill merged PRE program (one core per token tile-row r, the whole hidden row on that core: no cross-core
// exchange), reader. Phases: 0 constants; A stats: x tiles (cb_x) + hc_fn tiles (cb_w) of the 4 streams; B wait for the
// pp tile (compute -> cb_pp), build the 4 pre tiles S_s (column s of pp replicated over the columns, exact bit copies
// on the RISC) into cb_s; C x tiles again (cb_xc, 4 streams per column tile); D the bf16 norm weight tiles (cb_wn). CT:
// 0 cb_x, 1 cb_w, 2 cb_cst, 3 cb_cst2, 4 cb_pp, 5 cb_s, 6 cb_wn, 7 cb_xc, 8 TR, 9 NTD, accessors x, w, cst, wn, st.
// Common: 5 addresses. RT: r.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cst = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cst2 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pp = get_compile_time_arg_val(4);
    constexpr uint32_t cb_s = get_compile_time_arg_val(5);
    constexpr uint32_t cb_wn = get_compile_time_arg_val(6);
    constexpr uint32_t cb_xc = get_compile_time_arg_val(7);
    constexpr uint32_t TR = get_compile_time_arg_val(8);
    constexpr uint32_t NTD = get_compile_time_arg_val(9);
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto st_args = TensorAccessorArgs<n_args.next_compile_time_args_offset()>();
    const uint32_t r = get_arg_val<uint32_t>(0);
    const auto x_acc = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto w_acc = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), 4096);
    const auto c_acc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(2), 4096);
    const auto n_acc = TensorAccessor(n_args, get_common_arg_val<uint32_t>(3), 2048);
    const auto st_acc = TensorAccessor(st_args, get_common_arg_val<uint32_t>(4), 4096);
    Noc noc;
    experimental::CB cbx(cb_x), cbw(cb_w), cbc(cb_cst), cbc2(cb_cst2), cbp(cb_pp), cbs(cb_s), cbn(cb_wn), cbxc(cb_xc);
    cbc.reserve_back(1);
    cbc2.reserve_back(4);
    noc.async_read(c_acc, cbc, 4096, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    for (uint32_t t = 1; t < 5; ++t) {
        noc.async_read(c_acc, cbc2, 4096, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = (t - 1) * 4096});
    }
    noc.async_read_barrier();
    cbc.push_back(1);
    cbc2.push_back(4);
    // ---- A
    for (uint32_t s = 0; s < 4; ++s) {
        for (uint32_t jb = 0; jb < NTD; jb += 4) {
            cbx.reserve_back(4);
            cbw.reserve_back(4);
            for (uint32_t jj = 0; jj < 4; ++jj) {
                const uint32_t c = jb + jj;
                noc.async_read(
                    x_acc,
                    cbx,
                    4096,
                    {.page_id = s * TR * NTD + r * NTD + c, .offset_bytes = 0},
                    {.offset_bytes = jj * 4096});
                noc.async_read(
                    w_acc, cbw, 4096, {.page_id = s * NTD + c, .offset_bytes = 0}, {.offset_bytes = jj * 4096});
            }
            noc.async_read_barrier();
            cbx.push_back(4);
            cbw.push_back(4);
        }
    }
    // ---- B
    cbp.wait_front(1);
    cbs.reserve_back(4);
    {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbp.get_read_ptr());
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
        for (uint32_t k = 0; k < 4; ++k) {
            for (uint32_t t = 0; t < 32; ++t) {
                const uint32_t v = src[((t >> 4) << 9) + ((t & 15) << 4) + k];
                const uint32_t row = ((t >> 4) << 9) + ((t & 15) << 4);
                volatile tt_l1_ptr uint32_t* d = dst + k * 1024 + row;
                for (uint32_t c = 0; c < 16; ++c) {
                    d[c] = v;
                    d[256 + c] = v;
                }
            }
        }
    }
    cbs.push_back(4);
    // the pp tile itself goes to DRAM (page r of `st`) for the expand program
    noc.async_write(cbp, st_acc, 4096, {.offset_bytes = 0}, {.page_id = r, .offset_bytes = 0});
    noc.async_write_barrier();
    cbp.pop_front(1);
    // ---- C
    for (uint32_t jb = 0; jb < NTD; jb += 2) {
        cbxc.reserve_back(8);
        for (uint32_t jj = 0; jj < 2; ++jj) {
            const uint32_t c = jb + jj;
            for (uint32_t s = 0; s < 4; ++s) {
                noc.async_read(
                    x_acc,
                    cbxc,
                    4096,
                    {.page_id = s * TR * NTD + r * NTD + c, .offset_bytes = 0},
                    {.offset_bytes = (jj * 4 + s) * 4096});
            }
        }
        noc.async_read_barrier();
        cbxc.push_back(8);
    }
    // ---- D (x again: the collapse is recomputed in the norm pass) with the norm weight tiles
    for (uint32_t jb = 0; jb < NTD; jb += 2) {
        cbxc.reserve_back(8);
        cbn.reserve_back(2);
        for (uint32_t jj = 0; jj < 2; ++jj) {
            const uint32_t c = jb + jj;
            for (uint32_t s = 0; s < 4; ++s) {
                noc.async_read(
                    x_acc,
                    cbxc,
                    4096,
                    {.page_id = s * TR * NTD + r * NTD + c, .offset_bytes = 0},
                    {.offset_bytes = (jj * 4 + s) * 4096});
            }
            noc.async_read(n_acc, cbn, 2048, {.page_id = c, .offset_bytes = 0}, {.offset_bytes = jj * 2048});
        }
        noc.async_read_barrier();
        cbxc.push_back(8);
        cbn.push_back(2);
    }
}
