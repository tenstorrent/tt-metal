// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Probe: can a UInt16 tile of fp16 bit patterns, copied into an fp32 DST, be read back as fp32 by the
// SFPU? mode 0: SFPLOAD as F16a; mode 1: LO16 + integer rebias (fp16 -> fp32 bits).
// NCRISC: in (uint16 tile) -> cb0. TRISC: copy to DST 0, SFPU writes slot 1 as fp32, pack -> cb16.
// BRISC: cb16 -> out (fp32 tile).
#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"
#endif
#endif

constexpr uint32_t mode = get_named_compile_time_arg_val("pr_mode");

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
inline void convert() {
#pragma GCC unroll 32
    for (uint32_t u = 0; u < 32; ++u) {
        if constexpr (mode == 0) {
            sfpi::vFloat f = sfpi::dst_reg[u].mode<sfpi::DataLayout::F16a>();
            sfpi::dst_reg[32 + u] = f;
        } else {
            sfpi::vUInt b = sfpi::dst_reg[u].mode<sfpi::DataLayout::LO16>();
            sfpi::vUInt em = b & 0x7fff;
            sfpi::vUInt r = (em << 13) + (112u << 23);
            v_if(em == 0) { r = 0; }
            v_endif;
            r = r | ((b & 0x8000) << 16);
            sfpi::dst_reg[32 + u] = sfpi::as<sfpi::vFloat>(r);
        }
    }
}
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    const InterleavedAddrGen<true> g{.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 2048};
    cb_reserve_back(0, 1);
    noc_async_read(get_noc_addr(0, g), get_write_ptr(0), 2048);
    noc_async_read_barrier();
    cb_push_back(0, 1);
#elif defined(COMPILE_FOR_BRISC)
    const InterleavedAddrGen<true> g{.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 4096};
    cb_wait_front(16, 1);
    noc_async_write(get_read_ptr(16), get_noc_addr(0, g), 4096);
    noc_async_write_barrier();
    cb_pop_front(16, 1);
#elif defined(COMPILE_FOR_TRISC)
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
    cb_wait_front(0, 1);
    tile_regs_acquire();
    copy_tile(0, 0, 0);
    MATH((_llk_math_eltwise_sfpu_start_(0)));
    MATH((convert()));
    MATH((_llk_math_eltwise_sfpu_done_()));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(16, 1);
    pack_tile(1, 16);
    cb_push_back(16, 1);
    tile_regs_release();
    cb_pop_front(0, 1);
#endif
}
