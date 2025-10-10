// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"

void kernel_main() {
    // CTA layout (fixed first, then variable-length TensorAccessorArgs at the end):
    // 0: bank_id
    // 1: l1_src_addr
    // 2: page_size
    // 3: base_offset
    // [4..): TensorAccessorArgs CTAs (variable length)
    const uint32_t bank_id = get_compile_time_arg_val(0);
    const uint32_t l1_src_addr = get_compile_time_arg_val(1);
    const uint32_t page_size_bytes = get_compile_time_arg_val(2);
    const uint32_t base_offset_bytes = get_compile_time_arg_val(3);

    constexpr auto ta = TensorAccessorArgs</*CTA_OFFSET=*/4>();
    auto addrgen = TensorAccessor(ta, base_offset_bytes, page_size_bytes);

    noc_async_write_page(/*page_id=*/0, addrgen, l1_src_addr);
    noc_async_write_barrier();
}


