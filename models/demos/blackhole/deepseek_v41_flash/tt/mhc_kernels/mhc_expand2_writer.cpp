// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC expand, writer (BRISC).  First builds the per-token A tiles (the reader does the B tiles, in parallel):
//   A_t[j][4*tl + i] = comb[t][i][j],  A_t[j][16 + tl] = post[t][j],  A_t[j][20 + tl] = post[t][j] (y2 term)
// so that  out_t (rows j = new streams) = A_t @ B_(group, col tile).  Then streams the output tiles to DRAM.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sa = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);
    constexpr uint32_t NG = get_compile_time_arg_val(5);
    constexpr uint32_t HAS_Y2 = get_compile_time_arg_val(6);
    constexpr auto o_args = TensorAccessorArgs<7>();
    constexpr auto c_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, TILE);
    const auto c_acc = TensorAccessor(c_args, c_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    experimental::CB out(cb_out), cba(cb_a), csa(cb_sa);

    csa.reserve_back(1);
    cba.reserve_back(T);
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_read(c_acc, csa, 256, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = t * 512});
        noc.async_read(p_acc, csa, 256, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = t * 512 + 256});
    }
    noc.async_write_zeros(cba, T * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* A = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
    volatile tt_l1_ptr uint32_t* SC = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(csa.get_write_ptr());
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t tl = t & 3;
        volatile tt_l1_ptr uint32_t* At = A + t * (TILE / 4);
        volatile tt_l1_ptr uint32_t* sc = SC + t * 128;
        volatile tt_l1_ptr uint32_t* sp = sc + 64;
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t i = 0; i < 4; ++i) {
                At[j * 16 + 4 * tl + i] = sc[i * 16 + j];
            }
            At[256 + j * 16 + tl] = sp[j * 16];
            if (HAS_Y2) {
                At[256 + j * 16 + 4 + tl] = sp[j * 16];
            }
        }
    }
    cba.push_back(T);

    const uint32_t nb = NG * T;
    constexpr uint32_t CH = 4;
    for (uint32_t k = 0; k < nb; k += CH) {
        const uint32_t n = (nb - k) < CH ? (nb - k) : CH;
        out.wait_front(n);
        for (uint32_t q = 0; q < n; ++q) {
            const uint32_t kk = k + q;
            const uint32_t t = kk % T;
            const uint32_t j = j0 + kk / T;
            noc.async_write(out, o_acc, TILE, {.offset_bytes = q * TILE}, {.page_id = t * NT + j, .offset_bytes = 0});
        }
        noc.async_writes_flushed();
        out.pop_front(n);
    }
    noc.async_write_barrier();
}
