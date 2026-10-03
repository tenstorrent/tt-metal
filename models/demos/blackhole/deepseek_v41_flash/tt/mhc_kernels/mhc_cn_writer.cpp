// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused mHC collapse + RMSNorm, writer (BRISC): builds the ones tile, writes the normalised bf16 tiles (and,
// optionally, the bf16 row-major [T,1,1,D] copy).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(1);
    constexpr uint32_t EMIT_RM = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t ROWBYTES = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t GPC = get_compile_time_arg_val(6);
    constexpr auto o_args = TensorAccessorArgs<7>();
    constexpr auto m_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t j0 = k * GPC;
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_common_arg_val<uint32_t>(1);
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    const auto m_acc = TensorAccessor(m_args, m_addr, ROWBYTES);
    experimental::CB out(cb_out), rm(cb_rm), ones(cb_ones);

    ones.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones.get_write_ptr());
        for (uint32_t i = 0; i < 1024; ++i) {
            o[i] = 0x3F800000u;
        }
    }
    ones.push_back(1);

    const uint32_t ng = GPC;
    out.wait_front(ng);
    for (uint32_t g = 0; g < ng; ++g) {
        noc.async_write(out, o_acc, 2048, {.offset_bytes = g * 2048}, {.page_id = j0 + g, .offset_bytes = 0});
    }
    if (EMIT_RM) {
        if constexpr (T <= 8) {
            // bf16 row-major [T,1,1,D]: row t of an output tile = two 32-byte face halves -> 64 contiguous bytes of
            // page t
            for (uint32_t g = 0; g < ng; ++g) {
                for (uint32_t t = 0; t < T; ++t) {
                    const uint32_t src = g * 2048 + t * 32;
                    noc.async_write(
                        out, m_acc, 32, {.offset_bytes = src}, {.page_id = t, .offset_bytes = (j0 + g) * 64});
                    noc.async_write(
                        out,
                        m_acc,
                        32,
                        {.offset_bytes = src + 512},
                        {.page_id = t, .offset_bytes = (j0 + g) * 64 + 32});
                }
            }
        } else {
            rm.reserve_back(1);
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out.get_read_ptr());
            volatile tt_l1_ptr uint32_t* dstp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rm.get_write_ptr());
            for (uint32_t g = 0; g < ng; ++g) {
                for (uint32_t t = 0; t < T; ++t) {
                    volatile tt_l1_ptr uint32_t* tile = src + g * 512;
                    volatile tt_l1_ptr uint32_t* d = dstp + (g * T + t) * 16;
                    for (uint32_t e = 0; e < 8; ++e) {
                        d[e] = tile[((t >> 4) << 8) + ((t & 15) << 3) + e];
                        d[8 + e] = tile[128 + ((t >> 4) << 8) + ((t & 15) << 3) + e];
                    }
                }
            }
            for (uint32_t g = 0; g < ng; ++g) {
                for (uint32_t t = 0; t < T; ++t) {
                    noc.async_write(
                        rm,
                        m_acc,
                        64,
                        {.offset_bytes = (g * T + t) * 64},
                        {.page_id = t, .offset_bytes = (j0 + g) * 64});
                }
            }
        }
    }
    noc.async_write_barrier();
    out.pop_front(ng);
}
