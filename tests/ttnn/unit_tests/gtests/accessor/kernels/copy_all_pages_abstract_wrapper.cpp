// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Copies all pages from input to output tensor using AbstractTensorAccessorWrapper
 * with the Device 2.0 Noc API (noc.async_read / noc.async_write).
 *
 * Builds a heterogeneous tuple of src/dst TensorAccessors, wraps them via
 * make_abstract_tensor_accessor_wrappers, and selects src/dst at runtime by index.
 */

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    auto args_src = TensorAccessorArgs<0, 0>();
    auto args_dst = TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>();

    const uint32_t cb_id = get_compile_time_arg_val(args_dst.next_compile_time_args_offset());
    const uint32_t page_size = get_compile_time_arg_val(args_dst.next_compile_time_args_offset() + 1);
    const uint32_t tensor_volume = get_compile_time_arg_val(args_dst.next_compile_time_args_offset() + 2);

    const uint32_t input_base_address = get_common_arg_val<uint32_t>(0);
    const uint32_t output_base_address = get_common_arg_val<uint32_t>(1);

    const auto tensor_accessor_src = TensorAccessor(args_src, input_base_address);
    const auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address);
    const auto tensor_accessors_tuple = std::make_tuple(tensor_accessor_src, tensor_accessor_dst);
    const auto abstract_tensor_accessor_wrappers = make_abstract_tensor_accessor_wrappers(tensor_accessors_tuple);

    Noc noc(noc_index);
    CircularBuffer cb(cb_id);

    for (uint32_t page_id = 0; page_id < tensor_volume; ++page_id) {
        cb.reserve_back(1);
        noc.async_read(
            abstract_tensor_accessor_wrappers[0],
            cb,
            page_size,
            {.page_id = page_id},
            {});
        noc.async_read_barrier();
        cb.push_back(1);

        cb.wait_front(1);
        noc.async_write(
            cb,
            abstract_tensor_accessor_wrappers[1],
            page_size,
            {},
            {.page_id = page_id});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
}
