// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Probe: rstd = rsqrt(mean(x^2) + eps) in EVERY lane of a DST tile, all on the Tensix (no RISC scalar reads):
//   S = x^2 (SFPU) -> cb 2;  Z = S @ O (row sums in every column) -> cb 3;  T = O @ Z (the total everywhere);
//   r = rsqrt(T / K + eps) (SFPU) -> cb 16 (fp32). O = all-ones tile (cb 1). x: one 32x32 bf16 tile (cb 0).
#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/experimental/pack_block.h"
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"
#endif
#endif

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
template <typename F>
inline void sfpu_section(F f) {
    _llk_math_eltwise_sfpu_start_(0);
    f();
    _llk_math_eltwise_sfpu_done_();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}
inline void square0() {
#pragma GCC unroll 32
    for (uint32_t i = 0; i < 32; ++i) {
        sfpi::vFloat v = sfpi::dst_reg[i];
        sfpi::dst_reg[i] = v * v;
    }
}
inline void mean_eps0(float inv_k, float eps) {
#pragma GCC unroll 32
    for (uint32_t i = 0; i < 32; ++i) {
        sfpi::vFloat v = sfpi::dst_reg[i];
        sfpi::dst_reg[i] = v * inv_k + eps;
    }
}
#endif

#if defined(COMPILE_FOR_TRISC)
// pr_hifi: HiFi4 for these two matmuls only, whatever the kernel's fidelity (the A3 runs LoFi for custom_mm)
constexpr bool hifi = get_named_compile_time_arg_val("pr_hifi") == 1;
inline void mm_init(uint32_t in0, uint32_t in1) {
    matmul_init(in0, in1);
    if constexpr (hifi) {
        MATH((llk_math_matmul_init<MathFidelity::HiFi4, MM_THROTTLE>(in0, in1, 0)));
    }
}
inline void mm_tile(uint32_t in0, uint32_t in1) {
    if constexpr (hifi) {
        UNPACK((llk_unpack_AB_matmul(in0, in1, 0, 0)));
        MATH((llk_math_matmul<MathFidelity::HiFi4, MM_THROTTLE>(0)));
    } else {
        matmul_tiles(in0, in1, 0, 0, 0);
    }
}
template <uint32_t cb>
inline void pack_one(uint32_t slot) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_block_contiguous_init(cb);
    pack_block_contiguous(slot, cb, 1);
    cb_push_back(cb, 1);
    tile_regs_release();
}
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    const InterleavedAddrGen<true> a{.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 2048};
    const InterleavedAddrGen<true> o{.bank_base_address = get_arg_val<uint32_t>(1), .page_size = 2048};
    cb_reserve_back(0, 1);
    cb_reserve_back(1, 1);
    noc_async_read(get_noc_addr(0, a, 0), get_write_ptr(0), 2048);
    noc_async_read(get_noc_addr(0, o, 0), get_write_ptr(1), 2048);
    noc_async_read_barrier();
    cb_push_back(0, 1);
    cb_push_back(1, 1);
#elif defined(COMPILE_FOR_BRISC)
    const InterleavedAddrGen<true> y{.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 4096};
    cb_wait_front(16, 1);
    noc_async_write(get_read_ptr(16), get_noc_addr(0, y, 0), 4096);
    noc_async_write_barrier();
    cb_pop_front(16, 1);
#elif defined(COMPILE_FOR_TRISC)
    constexpr float inv_k = 1.0f / 1024.0f, eps = 1e-6f;
    deepseek_compute_kernel_init();
    cb_wait_front(0, 1);
    cb_wait_front(1, 1);
    tile_regs_acquire();  // S = x^2
    copy_init(0);
    copy_tile(0, 0, 0);
    MATH((sfpu_section([] { square0(); })));
    pack_one<2>(0);
    cb_wait_front(2, 1);
    tile_regs_acquire();  // Z = S @ O (acquire before the init: an init ahead of it gave zeros in rmsprep)
    mm_init(2, 1);
    mm_tile(2, 1);
    pack_one<3>(0);
    cb_wait_front(3, 1);
    tile_regs_acquire();  // T = O @ Z, then r = rsqrt(T / K + eps)
    mm_init(1, 3);
    mm_tile(1, 3);
    MATH((sfpu_section([] { mean_eps0(inv_k, eps); })));
    rsqrt_tile_init();
    rsqrt_tile(0);
    pack_one<16>(0);
#endif
}
