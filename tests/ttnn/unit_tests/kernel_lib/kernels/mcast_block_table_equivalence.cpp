// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One topology attached twice to this kernel: "p" with per-core blocks, "t" with block_table. Every core reads
// all words of both and the role/coordinate accessors through both decoders, then writes how many differ.
#include <stdint.h>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t P_CT = get_named_compile_time_arg_val("p_ct_offset");
    constexpr uint32_t P_RT = get_named_compile_time_arg_val("p_rt_offset");
    constexpr uint32_t T_CT = get_named_compile_time_arg_val("t_ct_offset");
    constexpr uint32_t T_RT = get_named_compile_time_arg_val("t_rt_offset");
    constexpr auto p = McastArgs<P_CT, P_RT>();
    constexpr auto t = McastArgs<T_CT, T_RT>();
    using PositionalRuntime = detail::positional_mcast_runtime<P_CT, P_RT>;
    using TableRuntime = detail::positional_mcast_runtime<T_CT, T_RT>;
    static_assert(std::is_same_v<PositionalRuntime, detail::PositionalMcastRuntime<P_RT>>);
    static_assert(std::is_same_v<TableRuntime, detail::TableMcastRuntime<T_CT, T_RT>>);
    constexpr auto metadata = detail::positional_mcast_metadata<P_CT>();
    constexpr uint32_t words = mcast_wire::runtime_words(
        metadata.rotating_span, metadata.rectangle_capacity, mcast_wire::transfer_mode(metadata.flags));
    static_assert(McastArgs<T_CT, T_RT>::next_runtime_args_offset() == T_RT + 1, "a table family holds one pointer");

    uint32_t mismatches = 0;
    for (uint32_t w = 0; w < words; ++w) {
        mismatches += PositionalRuntime::read(w) != TableRuntime::read(w);
    }
    mismatches += p.can_send() != t.can_send();
    mismatches += p.can_receive() != t.can_receive();
    mismatches += p.sender_x() != t.sender_x();
    mismatches += p.sender_y() != t.sender_y();
    const auto p_from = p.sender_coordinates();
    const auto t_from = t.sender_coordinates();
    for (uint32_t w = 0; w < 2 * p.num_senders; ++w) {
        mismatches += p_from[w] != t_from[w];
        mismatches += PositionalRuntime::coordinates(mcast_wire::sender_coords_offset(p.rotating_span))[w] !=
                      TableRuntime::coordinates(mcast_wire::sender_coords_offset(t.rotating_span))[w];
    }

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t row = get_arg_val<uint32_t>(1);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t l1 = cb.get_write_ptr();
    auto* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    result[0] = mismatches;
    result[1] = words;
    result[2] = get_arg_val<uint32_t>(T_RT);  // this core's table pointer
    result[3] = p.can_send();
    noc_async_write(l1, out.get_noc_addr(row), 32);
    noc_async_write_barrier();
}
