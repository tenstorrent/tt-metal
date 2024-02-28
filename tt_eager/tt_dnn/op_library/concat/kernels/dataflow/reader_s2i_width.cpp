// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

void kernel_main() {
	constexpr uint32_t num_tensors = get_compile_time_arg_val(0);

	uint32_t arg_index = 0;
	for (uint32_t i = 0; i < num_tensors; i++) {
		const uint32_t input_shard_cb = get_arg_val<uint32_t>(arg_index++);
		const uint32_t num_pages = get_arg_val<uint32_t>(arg_index++);
		cb_reserve_back(input_shard_cb, num_pages);
		noc_async_read_barrier();
		cb_push_back(input_shard_cb, num_pages);
	}



}
