// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Conv core of the split GDN step (QWEN38_GDN_CONV_SPLIT): one conv tile T of the 40 q/k/v tiles per core.  Reads the
// projection tile P[T], the three older ring slots' tile T and the four taps' tile T in one burst (one barrier),
// pushes them to the conv compute, then writes P[T] into the newest ring slot (once, not once per head).  The tile id
// T is the same in the projection row, the slots, the taps and the newest slot (q 0-7, k 8-15, v 16-39).
// Compile-time args: TensorAccessorArgs of projected, slot0, slot1, slot2, tap0..tap3, newest.  Runtime args: 0
// projected, 1-3 slots, 4-7 taps, 8 newest addresses, 9 T.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_P = 0, CB_S = 1, CB_T = 2;
constexpr uint32_t BF16_TILE = 2048;
}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t s0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t s1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t s2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t newest_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tile = get_arg_val<uint32_t>(arg++);

    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto s0_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    constexpr auto t0_args = TensorAccessorArgs<s2_args.next_compile_time_args_offset()>();
    constexpr auto t1_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
    constexpr auto t2_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();
    constexpr auto t3_args = TensorAccessorArgs<t2_args.next_compile_time_args_offset()>();
    constexpr auto newest_args = TensorAccessorArgs<t3_args.next_compile_time_args_offset()>();

    const auto p = TensorAccessor(p_args, p_addr);
    const auto s0 = TensorAccessor(s0_args, s0_addr);
    const auto s1 = TensorAccessor(s1_args, s1_addr);
    const auto s2 = TensorAccessor(s2_args, s2_addr);
    const auto t0 = TensorAccessor(t0_args, t0_addr);
    const auto t1 = TensorAccessor(t1_args, t1_addr);
    const auto t2 = TensorAccessor(t2_args, t2_addr);
    const auto t3 = TensorAccessor(t3_args, t3_addr);
    const auto newest = TensorAccessor(newest_args, newest_addr);

#ifdef CS_NOWORK  // dispatch-cost probe (QWEN38_GDN_CS_NOWORK=1): the conv cores launch and do nothing
    return;
#endif
    FUSED_ZONE("fz_gcv_r");
    cb_reserve_back(CB_P, 1);
    cb_reserve_back(CB_S, 3);
    cb_reserve_back(CB_T, 4);
    const uint32_t p_l1 = get_write_ptr(CB_P);
    const uint32_t s_l1 = get_write_ptr(CB_S);
    const uint32_t t_l1 = get_write_ptr(CB_T);
    noc_async_read_page(tile, p, p_l1);
    noc_async_read_page(tile, s0, s_l1);
    noc_async_read_page(tile, s1, s_l1 + BF16_TILE);
    noc_async_read_page(tile, s2, s_l1 + 2 * BF16_TILE);
    noc_async_read_page(tile, t0, t_l1);
    noc_async_read_page(tile, t1, t_l1 + BF16_TILE);
    noc_async_read_page(tile, t2, t_l1 + 2 * BF16_TILE);
    noc_async_read_page(tile, t3, t_l1 + 3 * BF16_TILE);
    noc_async_read_barrier();
    cb_push_back(CB_P, 1);
    cb_push_back(CB_S, 3);
    cb_push_back(CB_T, 4);
    noc_async_write_page(tile, newest, p_l1);  // the newest ring slot's tile T: the projection's tile T, written once
    noc_async_write_barrier();
}
