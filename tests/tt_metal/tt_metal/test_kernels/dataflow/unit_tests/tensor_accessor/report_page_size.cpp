// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel that constructs a TensorAccessor from TensorAccessorArgs WITHOUT
// passing page_size (so it defaults to AlignedPageSize), then writes the
// resulting page_size to a DRAM output buffer for host-side verification.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t input_ta_cta_offset = 0;
    constexpr auto input_ta_args = TensorAccessorArgs<input_ta_cta_offset>();

    constexpr uint32_t output_ta_cta_offset = input_ta_args.next_compile_time_args_offset();
    constexpr auto output_ta_args = TensorAccessorArgs<output_ta_cta_offset>();

    // page_size is intentionally omitted — it should default to AlignedPageSize
    auto input_accessor = TensorAccessor(input_ta_args, input_addr);
    auto output_accessor = TensorAccessor(output_ta_args, output_addr);

    constexpr uint32_t output_cb = 0;
    cb_reserve_back(output_cb, 1);
    uint32_t l1_addr = get_write_ptr(output_cb);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    ptr[0] = input_accessor.get_aligned_page_size();
    ptr[1] = output_accessor.get_aligned_page_size();

    uint64_t dram_noc_addr = output_accessor.get_noc_addr(0);
    noc_async_write(l1_addr, dram_noc_addr, output_accessor.get_aligned_page_size());
    noc_async_write_barrier();
}
