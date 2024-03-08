// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
	constexpr uint32_t num_tensors     = get_compile_time_arg_val(0);
	constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
	constexpr uint32_t output_shard_cb = get_compile_time_arg_val(2);
	const uint32_t dst_addr  = get_arg_val<uint32_t>(0);
	const uint32_t core_id = get_arg_val<uint32_t>(1);
	const uint32_t num_pages_per_core = get_arg_val<uint32_t>(2);
	const uint32_t out_stick_size = get_arg_val<uint32_t>(3);
	const uint32_t num_tensors_times_rows_per_shard = get_arg_val<uint32_t>(4);
	const uint32_t num_pages_per_tensor = get_arg_val<uint32_t>(5);

	uint32_t arg_index = 6;

	const InterleavedAddrGenFast<dst_is_dram> s = {
		.bank_base_address = dst_addr,
		.page_size = out_stick_size,
	};

	cb_reserve_back(output_shard_cb, num_tensors * num_pages_per_tensor);
	uint32_t starting_l1_write_addr = get_write_ptr(output_shard_cb);
	for (uint32_t tensor_id = 0; tensor_id < num_tensors; tensor_id++) {
		uint32_t l1_write_addr = starting_l1_write_addr + tensor_id*out_stick_size;
		const uint32_t stick_size = get_arg_val<uint32_t>(arg_index++);

    const uint32_t input_shard_cb = tensor_id;
    cb_wait_front(input_shard_cb, num_pages_per_tensor);
		uint32_t l1_read_addr = get_read_ptr(input_shard_cb);
		for(uint32_t page_id_input = 0; page_id_input < num_pages_per_tensor; page_id_input++) {
			noc_async_read(get_noc_addr(l1_read_addr), l1_write_addr, stick_size);
			noc_async_read_barrier();
			l1_read_addr += stick_size;
			l1_write_addr += num_tensors*stick_size;
		}
		cb_pop_front(input_shard_cb, num_pages_per_tensor);
		cb_push_back(output_shard_cb, num_pages_per_tensor);
	}
	cb_wait_front(output_shard_cb, num_pages_per_tensor*num_tensors);

}
