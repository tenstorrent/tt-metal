// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused mHC collapse + RMSNorm, writer v2 (DSV41_MHC_FRONT_FAST: the row-major copy is staged per token and written as
// one chunk per token). (v1 text:) Fused mHC collapse + RMSNorm, writer (BRISC): builds the ones tile, writes the
// normalised bf16 tiles (and, optionally, the bf16 row-major [T,1,1,D] copy).

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
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
    // compact output for the skinny chains (DSV41_SKINNY_HC=1): tiny 8x32 tiles in a RM buffer of 1<<PBSH byte pages,
    // row tile m at tile m * KSTRIDE
    constexpr uint32_t CMP = get_compile_time_arg_val(m_args.next_compile_time_args_offset());
    constexpr uint32_t KSTRIDE = get_compile_time_arg_val(m_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t PBSH = get_compile_time_arg_val(m_args.next_compile_time_args_offset() + 2);
    constexpr auto c_args = TensorAccessorArgs<m_args.next_compile_time_args_offset() + 3>();
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t j0 = k * GPC;
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_common_arg_val<uint32_t>(1);
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    const auto m_acc = TensorAccessor(m_args, m_addr, ROWBYTES);
    const auto c_acc = TensorAccessor(c_args, CMP ? get_common_arg_val<uint32_t>(2) : 0u, 1u << PBSH);
    experimental::CB out(cb_out), rm(cb_rm), ones(cb_ones);

    DeviceZoneScopedN("CW_ALL");
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
    {
        DeviceZoneScopedN("CW_gotout");
    }
    for (uint32_t g = 0; g < ng; ++g) {
        noc.async_write(out, o_acc, 2048, {.offset_bytes = g * 2048}, {.page_id = j0 + g, .offset_bytes = 0});
    }
    if constexpr (CMP != 0) {
        constexpr uint32_t MTC = T > 8 ? 2 : 1;
        for (uint32_t g = 0; g < ng; ++g) {
            for (uint32_t m = 0; m < MTC; ++m) {
                const uint32_t flat = (m * KSTRIDE + j0 + g) * 512;
                noc.async_write(
                    out,
                    c_acc,
                    256,
                    {.offset_bytes = g * 2048 + m * 256},
                    {.page_id = flat >> PBSH, .offset_bytes = flat & ((1u << PBSH) - 1)});
                noc.async_write(
                    out,
                    c_acc,
                    256,
                    {.offset_bytes = g * 2048 + 512 + m * 256},
                    {.page_id = (flat + 256) >> PBSH, .offset_bytes = (flat + 256) & ((1u << PBSH) - 1)});
            }
        }
    }
    if (EMIT_RM) {
        // bf16 row-major [T,1,1,D]: this core's ng tiles of token t are ng * 64 contiguous bytes of page t: stage
        // [t][g] and write ONE chunk per token
        rm.reserve_back(1);
        // gather the face rows with the NoC (L1 -> own L1, 32 B each) instead of word copies on the RISC core
        const uint32_t src = out.get_read_ptr();
        const uint32_t dstb = rm.get_write_ptr();
        const uint32_t mx = my_x[noc_index];
        const uint32_t my = my_y[noc_index];
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t rof =
                ((t >> 4) << 10) + ((t & 15) << 5);  // row t of face 0 (t < 16) or face 2: bytes within the bf16 tile
            for (uint32_t g = 0; g < ng; ++g) {
                const uint32_t d = dstb + (t * ng + g) * 64;
                noc_async_write(src + g * 2048 + rof, get_noc_addr(mx, my, d), 32);
                noc_async_write(src + g * 2048 + rof + 512, get_noc_addr(mx, my, d + 32), 32);
            }
        }
        noc_async_write_barrier();
        for (uint32_t t = 0; t < T; ++t) {
            noc.async_write(rm, m_acc, ng * 64, {.offset_bytes = t * ng * 64}, {.page_id = t, .offset_bytes = j0 * 64});
        }
    }
    noc.async_write_barrier();
    out.pop_front(ng);
}
