// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC post kernel, writer: turns the kernel's token-row tiles (row t: pre[0..3] / post[0..3] / comb[0..15]) into the
// token-major outputs  pre [T,1,1,4], post [T,1,4,1], comb [T,1,4,4]  (one fp32 tile page per token).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// row r of a 32x32 tile: byte offset of its first face half (fp32) and word index
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))
#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))

void kernel_main() {
    constexpr uint32_t cb_pre = get_compile_time_arg_val(0);
    constexpr uint32_t cb_post = get_compile_time_arg_val(1);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);  // 3*T scratch pages
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr auto pre_args = TensorAccessorArgs<5>();
    constexpr auto post_args = TensorAccessorArgs<pre_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t pre_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t post_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t comb_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto pre_acc = TensorAccessor(pre_args, pre_addr, TILE);
    const auto post_acc = TensorAccessor(post_args, post_addr, TILE);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, TILE);
    experimental::CB cpre(cb_pre), cpost(cb_post), ccomb(cb_comb), cout(cb_out);

    constexpr uint32_t TB = T < 8 ? T : 8;  // tokens per scratch round
    cout.reserve_back(3 * TB);
    noc.async_write_zeros(cout, 3 * TB * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    cpre.wait_front(1);
    cpost.wait_front(1);
    ccomb.wait_front(1);
    volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpre.get_read_ptr());
    volatile tt_l1_ptr uint32_t* so = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpost.get_read_ptr());
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ccomb.get_read_ptr());
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cout.get_write_ptr());
    for (uint32_t tb0 = 0; tb0 < T; tb0 += TB) {
        for (uint32_t tl = 0; tl < TB; ++tl) {
            const uint32_t t = tb0 + tl;
            volatile tt_l1_ptr uint32_t* opre = o + (0 * TB + tl) * (TILE / 4);
            volatile tt_l1_ptr uint32_t* opost = o + (1 * TB + tl) * (TILE / 4);
            volatile tt_l1_ptr uint32_t* ocomb = o + (2 * TB + tl) * (TILE / 4);
            for (uint32_t i = 0; i < 4; ++i) {
                opre[i] = sp[RW(t) + i];
                opost[i * 16] = so[RW(t) + i];
                for (uint32_t j = 0; j < 4; ++j) {
                    ocomb[i * 16 + j] = sc[RW(t) + i * 4 + j];
                }
            }
        }
        for (uint32_t tl = 0; tl < TB; ++tl) {
            const uint32_t t = tb0 + tl;
            noc.async_write(
                cout, pre_acc, TILE, {.offset_bytes = (0 * TB + tl) * TILE}, {.page_id = t, .offset_bytes = 0});
            noc.async_write(
                cout, post_acc, TILE, {.offset_bytes = (1 * TB + tl) * TILE}, {.page_id = t, .offset_bytes = 0});
            noc.async_write(
                cout, comb_acc, TILE, {.offset_bytes = (2 * TB + tl) * TILE}, {.page_id = t, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
    noc.async_write_barrier();
    cpre.pop_front(1);
    cpost.pop_front(1);
    ccomb.pop_front(1);
}
