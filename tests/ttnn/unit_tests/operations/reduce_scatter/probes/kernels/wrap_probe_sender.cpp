// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Refinement-1 fabric probe (sender, BRISC): stage ONE local input page in L1, fabric-unicast it
// into the REMOTE device's output tensor page 0 via the CCL egress helper, then inc the remote
// counting semaphore. Minimal single-purpose instrument for the FABRIC_1D wrap-link data-plane
// probe — NOT part of the reduce_scatter op pipeline.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib::ccl;

void kernel_main() {
    constexpr uint32_t cb_page = get_compile_time_arg_val(0);
    constexpr uint32_t alignment = get_compile_time_arg_val(1);
    constexpr auto input_args = TensorAccessorArgs<2>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    uint32_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t num_hops = get_arg_val<uint32_t>(ai++);
    const uint32_t counting_sem_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t target_noc_x = get_arg_val<uint32_t>(ai++);
    const uint32_t target_noc_y = get_arg_val<uint32_t>(ai++);

    // Fabric connection arg block (laid out by the host's has_forward/has_backward idiom); the
    // leading has_forward flag doubles as the send direction.
    size_t conn_arg_idx = ai;
    const bool dst_is_forward = get_arg_val<uint32_t>(conn_arg_idx);
    FabricStreamSender<> sender(conn_arg_idx, dst_is_forward, alignment);

    const auto input = TensorAccessor(input_args, input_addr, page_size);
    const auto output = TensorAccessor(output_args, output_addr, page_size);

    // Stage the single local input page in L1 (reserve-only scratch — one page, one use).
    cb_reserve_back(cb_page, 1);
    const uint32_t l1 = get_write_ptr(cb_page);
    noc_async_read(input.get_noc_addr(0), l1, page_size);
    noc_async_read_barrier();

    auto stream = sender.open(unicast_route(num_hops));
    auto writer = stream.arm_unicast_write(page_size);
    auto counter = stream.arm_inc(1);

    // Uniform mesh buffer address: the LOCAL output accessor's page-0 address, routed num_hops away,
    // lands in the remote device's output tensor (same idiom as the reduce_scatter Phase-A writer).
    writer.write_page(l1, 0, output);
    counter.inc(safe_get_noc_addr(target_noc_x, target_noc_y, counting_sem_addr, 0));

    stream.close();  // drains write + atomic barriers
}
