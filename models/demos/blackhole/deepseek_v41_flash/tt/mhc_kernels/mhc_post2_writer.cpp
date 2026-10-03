// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC post kernel v2, writer: turns the kernel's token-row tiles (PP: row t = pre[0..3] | post[0..3] in columns 0..3
// | 4..7; COMB: row t = comb[0..15]) into the token-major outputs  pre [T,1,1,4], post [T,1,4,1], comb [T,1,4,4]  (one
// zero-padded fp32 tile page per token).  pre/post are written while the compute core still runs the Sinkhorn
// iterations.

#include <stdint.h>
#include "tools/profiler/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// row r of a 32x32 tile: word index of its first face half (fp32)
#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))

void kernel_main() {
    constexpr uint32_t cb_pp = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);  // 3*TB scratch pages
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr auto pre_args = TensorAccessorArgs<4>();
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
    experimental::CB cpp(cb_pp), ccomb(cb_comb), cout(cb_out);

    constexpr uint32_t TB = T < 8 ? T : 8;  // tokens per scratch round
    cout.reserve_back(3 * TB);
    noc.async_write_zeros(cout, 3 * TB * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cout.get_write_ptr());

    {
        DeviceZoneScopedN("QW_zero");
    }
    cpp.wait_front(1);
    {
        DeviceZoneScopedN("QW_gotPP");
    }
    volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpp.get_read_ptr());
    for (uint32_t tb0 = 0; tb0 < T; tb0 += TB) {
        for (uint32_t tl = 0; tl < TB; ++tl) {
            const uint32_t t = tb0 + tl;
            volatile tt_l1_ptr uint32_t* opre = o + (0 * TB + tl) * (TILE / 4);
            volatile tt_l1_ptr uint32_t* opost = o + (1 * TB + tl) * (TILE / 4);
            for (uint32_t i = 0; i < 4; ++i) {
                opre[i] = sp[RW(t) + i];
                opost[i * 16] = sp[RW(t) + 4 + i];
            }
        }
        for (uint32_t tl = 0; tl < TB; ++tl) {
            const uint32_t t = tb0 + tl;
            noc.async_write(
                cout, pre_acc, TILE, {.offset_bytes = (0 * TB + tl) * TILE}, {.page_id = t, .offset_bytes = 0});
            noc.async_write(
                cout, post_acc, TILE, {.offset_bytes = (1 * TB + tl) * TILE}, {.page_id = t, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
    cpp.pop_front(1);

    {
        DeviceZoneScopedN("QW_prepost_done");
    }
    ccomb.wait_front(1);
    {
        DeviceZoneScopedN("QW_gotcomb");
    }
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ccomb.get_read_ptr());
    for (uint32_t tb0 = 0; tb0 < T; tb0 += TB) {
        for (uint32_t tl = 0; tl < TB; ++tl) {
            const uint32_t t = tb0 + tl;
            volatile tt_l1_ptr uint32_t* ocomb = o + (2 * TB + tl) * (TILE / 4);
            const uint32_t dw = ((t >> 3) << 4) + ((t & 7) << 1);
            for (uint32_t i = 0; i < 4; ++i) {
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint32_t e = 4 * i + j;  // SFPU layout, see mhc_sinkhorn_sfpu.h
                    ocomb[i * 16 + j] = sc[(e >> 3) * 256 + (((e & 7) >> 1) << 6) + dw + (e & 1)];
                }
            }
        }
        for (uint32_t tl = 0; tl < TB; ++tl) {
            const uint32_t t = tb0 + tl;
            noc.async_write(
                cout, comb_acc, TILE, {.offset_bytes = (2 * TB + tl) * TILE}, {.page_id = t, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
    ccomb.pop_front(1);
}
