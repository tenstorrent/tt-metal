// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One topology attached twice to this kernel: "p" with per-core blocks, "t" with block_table. Every core reads
// every word of its block and the role/coordinate accessors through both decoders, then writes how many differ.
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
    using P = McastArgs<P_CT, P_RT>;
    using T = McastArgs<T_CT, T_RT>;
    constexpr auto p = P();
    constexpr auto t = T();
    using PositionalRuntime = detail::positional_mcast_runtime<P_CT, P_RT>;
    using TableRuntime = detail::positional_mcast_runtime<T_CT, T_RT>;
    static_assert(P::runtime_layout.words == T::runtime_layout.words, "both channels decode the same block layout");
    constexpr uint32_t words = P::runtime_layout.words;
    static_assert(std::is_same_v<PositionalRuntime, detail::PositionalMcastRuntime<P_RT>>);
    // With no per-core words there is nothing to share and the host keeps the positional (empty) block.
    static_assert(
        words == 0 ? std::is_same_v<TableRuntime, detail::PositionalMcastRuntime<T_RT>>
                   : std::is_same_v<TableRuntime, detail::TableMcastRuntime<T_CT, T_RT>>);
    static_assert(T::next_runtime_args_offset() == T_RT + (words == 0 ? 0 : 1), "a table channel holds one pointer");

    uint32_t mismatches = 0;
    for (uint32_t w = 0; w < words; ++w) {
        mismatches += PositionalRuntime::read(w) != TableRuntime::read(w);
    }
    mismatches += p.can_send() != t.can_send();
    mismatches += p.can_receive() != t.can_receive();
    if constexpr (P::runtime_layout.sender_coordinates != mcast_wire::OMITTED) {
        mismatches += p.sender_x() != t.sender_x();
        mismatches += p.sender_y() != t.sender_y();
        const auto p_from = p.sender_coordinates();
        const auto t_from = t.sender_coordinates();
        for (uint32_t w = 0; w < mcast_wire::SENDER_COORD_WORDS * P::num_senders; ++w) {
            mismatches += p_from[w] != t_from[w];
        }
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
