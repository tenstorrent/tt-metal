// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Attention pre program, writer of the RoPE cores (tile columns 14, 15): q_rot page + cache rows.
#include "pre_common.hpp"

void kernel_main() {
    constexpr uint32_t HAS_LAT = get_compile_time_arg_val(0);
    constexpr uint32_t CACHE_PT = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s = get_compile_time_arg_val(3);
    constexpr auto kv_args = TensorAccessorArgs<4>();
    constexpr auto q_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto l_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const uint32_t w = get_arg_val<uint32_t>(0);
    const uint32_t t = w >> 1, n = 14 + (w & 1);
    Noc noc;
    const auto l_acc = TensorAccessor(l_args, get_common_arg_val<uint32_t>(2), 2048);
    const auto c_acc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(3), 2048);
    const auto o_acc = TensorAccessor(o_args, get_common_arg_val<uint32_t>(4), 2048);
    const auto p_acc = TensorAccessor(p_args, get_common_arg_val<uint32_t>(5), 64);
    const auto m_acc = TensorAccessor(m_args, get_common_arg_val<uint32_t>(6), 64);
    experimental::CB out(cb_o), cbs(cb_s);
    cbs.reserve_back(1);
    out.wait_front(1);
    noc.async_write(out, o_acc, 2048, {.offset_bytes = 0}, {.page_id = t * 16 + n, .offset_bytes = 0});
    noc.async_write_barrier();
    cache_writes<HAS_LAT, CACHE_PT>(
        noc, c_acc, l_acc, p_acc, m_acc, cbs, reinterpret_cast<PW32>(out.get_read_ptr()), t, n);
    out.pop_front(1);
}
