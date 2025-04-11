// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    const uint32_t in0_cb = get_compile_time_arg_val(0);
    const uint32_t in1_cb = get_compile_time_arg_val(1);
    const uint32_t imm_cb = get_compile_time_arg_val(2);
    const uint32_t sync_cb = get_compile_time_arg_val(3);
    const uint32_t out_cb = get_compile_time_arg_val(4);
    const uint32_t rt_dim = get_compile_time_arg_val(5);
    const uint32_t ct_dim = get_compile_time_arg_val(6);
    const uint32_t kt_dim = get_compile_time_arg_val(7);
    const uint32_t loop_factor = get_compile_time_arg_val(8);
    const uint32_t reuse_a = get_compile_time_arg_val(9);

#ifdef TILIZE_MATMUL_FUSED
    const uint32_t mm_cb = in0_cb;
#else
    const uint32_t mm_cb = imm_cb;
#endif

    cb_reserve_back(sync_cb, 1);
    volatile std::uint32_t* base_address = (std::uint32_t*)MEM_LLK_DEBUG_BASE;

#ifndef TILIZE_MATMUL_FUSED
    tilize_init(in0_cb, kt_dim, imm_cb);

    cb_reserve_back(imm_cb, rt_dim * kt_dim);
    cb_wait_front(in0_cb, rt_dim * kt_dim);

    tensix_sync();
    {
        DeviceZoneScopedN("B0");
        UNPACK((base_address[1] = 1));
        MATH((base_address[2] = 2));
        PACK((base_address[3] = 3));
        while (base_address[1] != 1) {
            asm("nop");
        }
        while (base_address[2] != 2) {
            asm("nop");
        }
        while (base_address[3] != 3) {
            asm("nop");
        }
        UNPACK((base_address[5] = 5));
        MATH((base_address[6] = 6));
        PACK((base_address[7] = 7));
        while (base_address[5] != 5) {
            asm("nop");
        }
        while (base_address[6] != 6) {
            asm("nop");
        }
        while (base_address[7] != 7) {
            asm("nop");
        }
        UNPACK((base_address[1] = 0));
        MATH((base_address[2] = 0));
        PACK((base_address[3] = 0));
        while (base_address[1] != 0) {
            asm("nop");
        }
        while (base_address[2] != 0) {
            asm("nop");
        }
        while (base_address[3] != 0) {
            asm("nop");
        }
        UNPACK((base_address[5] = 0));
        MATH((base_address[6] = 0));
        PACK((base_address[7] = 0));
    }

    {
        DeviceZoneScopedN("Tilize sync");
        {
            DeviceZoneScopedN("Tilize seq");
            for (uint j = 0; j < rt_dim; j++) {
                for (uint i = 0; i < loop_factor; i++) {
                    tilize_block(in0_cb, kt_dim, imm_cb);
                }
                cb_pop_front(in0_cb, kt_dim);
                cb_push_back(imm_cb, kt_dim);
            }
        }
        tensix_sync();
    }

    tilize_uninit(in0_cb, imm_cb);
#endif

#ifdef TILIZE_MATMUL_FUSED
    mm_block_tilize_A_init(mm_cb, in1_cb, out_cb, ct_dim, rt_dim, kt_dim, reuse_a);
#else
    mm_block_init(mm_cb, in1_cb, out_cb, 0, ct_dim, rt_dim, kt_dim, reuse_a);
#endif

    cb_reserve_back(out_cb, rt_dim * ct_dim);
    cb_wait_front(in1_cb, ct_dim * kt_dim);
    cb_wait_front(mm_cb, rt_dim * kt_dim);

    {
        DeviceZoneScopedN("B1");
        UNPACK((base_address[1] = 1));
        MATH((base_address[2] = 2));
        PACK((base_address[3] = 3));
        while (base_address[1] != 1) {
            asm("nop");
        }
        while (base_address[2] != 2) {
            asm("nop");
        }
        while (base_address[3] != 3) {
            asm("nop");
        }
        UNPACK((base_address[5] = 5));
        MATH((base_address[6] = 6));
        PACK((base_address[7] = 7));
        while (base_address[5] != 5) {
            asm("nop");
        }
        while (base_address[6] != 6) {
            asm("nop");
        }
        while (base_address[7] != 7) {
            asm("nop");
        }
        UNPACK((base_address[1] = 0));
        MATH((base_address[2] = 0));
        PACK((base_address[3] = 0));
        while (base_address[1] != 0) {
            asm("nop");
        }
        while (base_address[2] != 0) {
            asm("nop");
        }
        while (base_address[3] != 0) {
            asm("nop");
        }
        UNPACK((base_address[5] = 0));
        MATH((base_address[6] = 0));
        PACK((base_address[7] = 0));
    }

    {
        DeviceZoneScopedN("Matmul sync");
        {
            DeviceZoneScopedN("Matmul loop");
            for (uint i = 0; i < loop_factor; i++) {
                acquire_dst();
                for (uint j = 0; j < kt_dim; j++) {
#ifdef TILIZE_MATMUL_FUSED
                    matmul_block_tilize_A(mm_cb, in1_cb, j, j * ct_dim, 0, ct_dim, rt_dim, kt_dim, reuse_a);
#else
                    matmul_block(mm_cb, in1_cb, j, j * ct_dim, 0, 0, ct_dim, rt_dim, kt_dim, reuse_a);
#endif
                }
                matmul_pack_tile(0, out_cb, rt_dim * ct_dim);
                release_dst();
            }
        }
        tensix_sync();
    }

    cb_pop_front(mm_cb, rt_dim * kt_dim);
    cb_pop_front(in1_cb, ct_dim * kt_dim);
    cb_push_back(out_cb, rt_dim * ct_dim);

    {
        DeviceZoneScopedN("B2");
        UNPACK((base_address[1] = 1));
        MATH((base_address[2] = 2));
        PACK((base_address[3] = 3));
        while (base_address[1] != 1) {
            asm("nop");
        }
        while (base_address[2] != 2) {
            asm("nop");
        }
        while (base_address[3] != 3) {
            asm("nop");
        }
        UNPACK((base_address[5] = 5));
        MATH((base_address[6] = 6));
        PACK((base_address[7] = 7));
        while (base_address[5] != 5) {
            asm("nop");
        }
        while (base_address[6] != 6) {
            asm("nop");
        }
        while (base_address[7] != 7) {
            asm("nop");
        }
        UNPACK((base_address[1] = 0));
        MATH((base_address[2] = 0));
        PACK((base_address[3] = 0));
        while (base_address[1] != 0) {
            asm("nop");
        }
        while (base_address[2] != 0) {
            asm("nop");
        }
        while (base_address[3] != 0) {
            asm("nop");
        }
        UNPACK((base_address[5] = 0));
        MATH((base_address[6] = 0));
        PACK((base_address[7] = 0));
    }
}
}  // namespace NAMESPACE
