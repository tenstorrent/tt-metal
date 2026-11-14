// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4-Preview skinny projection BRISC (NC = 8 DRAM-bank cores): like hsk_ln_writer.cpp, but the output column tiles belong to up to 5 SEGMENT tensors (the fused h-proj
// [q_a | kv_a | gate | idx_wk | idx_wp] of Hy4AttentionOpt: the 5 ttnn.slice ops are folded into the writer): global output tile g -> segment s (seg_end[s-1] <= g < seg_end[s]),
// page g - seg_start of the s-th output tensor.  Core b owns the global tiles [first_b, first_b + valid_b) (padded to PNC local tiles; pad tiles are not written).
// CT args: 0 cb_x, 1 cb_o, 2 NC, 3 KT, 4 PNC, 5 SBW, 6 XS, 7 MT, 8 R, then TensorAccessorArgs of x and of the 5 output tensors (unused segments repeat the last tensor).
// RT args: 0 core index b, 1 first global tile, 2 valid tiles. Common: 0 x addr, 1..5 output addrs, 6..10 seg_end (tiles, exclusive; unused = total), then physical (x, y) of the NC cores. Semaphore 2: x exchange.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_o = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr uint32_t PNC = get_compile_time_arg_val(4);
    constexpr uint32_t SBW = get_compile_time_arg_val(5);
    constexpr bool XS = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t MT = get_compile_time_arg_val(7);
    constexpr uint32_t R = get_compile_time_arg_val(8);
    constexpr uint32_t NS = 5;
    constexpr uint32_t TB = 64 * R;
    constexpr uint32_t FB = 32 * R;
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto o_args0 = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto o_args1 = TensorAccessorArgs<o_args0.next_compile_time_args_offset()>();
    constexpr auto o_args2 = TensorAccessorArgs<o_args1.next_compile_time_args_offset()>();
    constexpr auto o_args3 = TensorAccessorArgs<o_args2.next_compile_time_args_offset()>();
    constexpr auto o_args4 = TensorAccessorArgs<o_args3.next_compile_time_args_offset()>();

    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t valid = get_arg_val<uint32_t>(2);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    auto nx = [&](uint32_t d) { return get_common_arg_val<uint32_t>(1 + 2 * NS + 2 * d); };
    auto ny = [&](uint32_t d) { return get_common_arg_val<uint32_t>(2 + 2 * NS + 2 * d); };
    auto seg_end = [&](uint32_t q) { return get_common_arg_val<uint32_t>(1 + NS + q); };
    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, 2048);
    experimental::CB cbx(cb_x), cbo(cb_o);

    {
        DeviceZoneScopedN("B_XLOAD");
        cbx.reserve_back(KT * MT);
        constexpr uint32_t XPC = KT / NC;
        for (uint32_t j = 0; j < (XS ? XPC : KT); ++j) {
            const uint32_t i = XS ? me * XPC + j : j;
            noc.async_read(x_acc, cbx, FB, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * TB});
            noc.async_read(x_acc, cbx, FB, {.page_id = i, .offset_bytes = 512}, {.offset_bytes = i * TB + FB});
            if constexpr (MT == 2) {
                noc.async_read(x_acc, cbx, FB, {.page_id = i, .offset_bytes = 256}, {.offset_bytes = (KT + i) * TB});
                noc.async_read(x_acc, cbx, FB, {.page_id = i, .offset_bytes = 768}, {.offset_bytes = (KT + i) * TB + FB});
            }
        }
        noc.async_read_barrier();
        if constexpr (XS) {
            const uint32_t xb = cbx.get_write_ptr();
            for (uint32_t kk = 1; kk < NC; ++kk) {
                const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
                for (uint32_t m = 0; m < MT; ++m) {
                    const uint32_t off = (m * KT + me * XPC) * TB;
                    noc_async_write(xb + off, get_noc_addr(nx(d), ny(d), xb + off), XPC * TB);
                }
            }
            noc_async_write_barrier();
            for (uint32_t kk = 1; kk < NC; ++kk) {
                const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
                noc_semaphore_inc(get_noc_addr(nx(d), ny(d), get_semaphore(2)), 1);
            }
            noc_async_atomic_barrier();
            volatile tt_l1_ptr uint32_t* sem_x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
            noc_semaphore_wait(sem_x, NC - 1);
            *sem_x = 0;
        }
        cbx.push_back(KT * MT);
    }
    {
        DeviceZoneScopedN("B_OUT");
        constexpr uint32_t NSB = PNC / SBW;
        for (uint32_t s = 0; s < NSB; ++s) {
            cbo.wait_front((s + 1) * SBW * MT);
            for (uint32_t w = 0; w < SBW; ++w) {
                const uint32_t j = s * SBW + w;
                if (j >= valid) {
                    continue;
                }
                const uint32_t g = first + j;
                uint32_t seg = 0, seg_start = 0;
                for (uint32_t q = 0; q < NS; ++q) {
                    if (g >= seg_end(q)) {
                        seg = q + 1;
                        seg_start = seg_end(q);
                    }
                }
                for (uint32_t m = 0; m < MT; ++m) {
                    const uint32_t t_off = ((s * MT + m) * SBW + w) * 512;
                    const uint32_t pg = g - seg_start;
                    auto put = [&](const auto& o_acc) {
                        noc.async_write(cbo, o_acc, FB, {.offset_bytes = t_off}, {.page_id = pg, .offset_bytes = m * 256});
                        noc.async_write(
                            cbo, o_acc, FB, {.offset_bytes = t_off + 256}, {.page_id = pg, .offset_bytes = 512 + m * 256});
                    };
                    if (seg == 0) {
                        put(TensorAccessor(o_args0, get_common_arg_val<uint32_t>(1), 2048));
                    } else if (seg == 1) {
                        put(TensorAccessor(o_args1, get_common_arg_val<uint32_t>(2), 2048));
                    } else if (seg == 2) {
                        put(TensorAccessor(o_args2, get_common_arg_val<uint32_t>(3), 2048));
                    } else if (seg == 3) {
                        put(TensorAccessor(o_args3, get_common_arg_val<uint32_t>(4), 2048));
                    } else {
                        put(TensorAccessor(o_args4, get_common_arg_val<uint32_t>(5), 2048));
                    }
                }
            }
        }
        noc.async_write_barrier();
        cbo.pop_front(PNC * MT);
    }
}
