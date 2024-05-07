// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {
	constexpr uint32_t shard_cb = get_compile_time_arg_val(0);
	constexpr uint32_t num_x_cores = get_compile_time_arg_val(1);
	constexpr uint32_t num_y_cores = get_compile_time_arg_val(2);
	constexpr uint32_t page_size = get_compile_time_arg_val(3);

	uint32_t y_offset = num_x_cores;

	uint32_t arg_index = num_x_cores + num_y_cores;
	const uint32_t input_shard_addr  = get_arg_val<uint32_t>(arg_index++);
	const uint32_t num_output_pages = get_arg_val<uint32_t>(arg_index++);
	const uint32_t num_ranges = get_arg_val<uint32_t>(arg_index++);
	const uint32_t output_page_offset = get_arg_val<uint32_t>(arg_index++);


	uint32_t l1_write_addr = get_write_ptr(shard_cb) + output_page_offset * page_size;
	uint32_t l1_write_addr_start = l1_write_addr;
	//DPRINT << "L1 WRITE ADDR START: " << HEX() << l1_write_addr << DEC() << ENDL();

	uint32_t mask_byte = 0x0ff; //8 bits
	uint32_t mask_short = 0x0ffff; //16 bits

	uint32_t total_num_pages = 0;
	for(uint32_t range_id = 0; range_id <num_ranges; range_id++) {
		const uint32_t core_start_stride = get_arg_val<uint32_t>(arg_index++);
		const uint32_t start_x_index = (core_start_stride >> 24);
		const uint32_t start_y_index = (core_start_stride >> 16) & mask_byte;
		const uint32_t stride_x = (core_start_stride >> 8) & mask_byte;
		const uint32_t stride_y = (core_start_stride) & mask_byte;
		const uint32_t start_x = get_arg_val<uint32_t>(start_x_index);
		const uint32_t start_y = get_arg_val<uint32_t>(y_offset + start_y_index);

		const uint32_t stride_data_offset = get_arg_val<uint32_t>(arg_index++);
		const uint32_t stride_size_num_strides_skip = get_arg_val<uint32_t>(arg_index++);
		const uint32_t num_strides = ((stride_size_num_strides_skip) & mask_short) >> 8;
		const bool skip = (((stride_size_num_strides_skip) & mask_byte)  == 1);


		const uint32_t stride_data = ((stride_data_offset >> 16)) * page_size;
		const uint32_t offset = ((stride_data_offset) & mask_short) * page_size;
		const uint32_t num_pages_per_stride = (stride_size_num_strides_skip >> 16);
		const uint32_t stride_size = num_pages_per_stride * page_size;

		uint32_t addr_offset = offset;
		uint32_t core_id_x_index = start_x_index;
		uint32_t core_id_y_index = start_y_index;
		//DPRINT << "PAGE_SIZE: " << page_size << ENDL();
		//DPRINT << "NUM STRIDES: " << num_strides << ENDL();
		//DPRINT << "STRIDE SIZE: " << stride_size << ENDL();
		//DPRINT << "OFFSET: " << offset << ENDL();
		//DPRINT << "STRIDE_DATA : " << stride_data << ENDL();
		//DPRINT << "OUTPUT PAGE OFFSET : " << output_page_offset << ENDL();
		//DPRINT << "SKIP " << (uint32_t)skip << ENDL();
		//DPRINT << "STRIDE_SIZE_NUM_STRIDES_SKIP " <<  HEX() << stride_size_num_strides_skip << DEC() << ENDL();

		for(uint32_t stride_idx = 0; stride_idx < num_strides; stride_idx++) {
			if(!skip) {
				uint32_t core_id_x = get_arg_val<uint32_t>(core_id_x_index);
				uint32_t core_id_y = get_arg_val<uint32_t>(y_offset + core_id_y_index);
				uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
						input_shard_addr + addr_offset);
				noc_async_read(noc_address, l1_write_addr, stride_size);
				total_num_pages += num_pages_per_stride;
			}
			l1_write_addr+=stride_size;
			if(stride_x == 0 and stride_y == 0) {
				addr_offset += (stride_data + stride_size);
			}
			else {
				addr_offset += (stride_data);
			}
			core_id_x_index += stride_x;
			core_id_y_index += stride_y;
		}


	}
	noc_async_read_barrier();


	uint32_t print_index = 0;
	volatile tt_l1_ptr int* addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(l1_write_addr_start);
	for(uint32_t i=0; i<total_num_pages; i++) {
		for(uint32_t i=0; i<(page_size/4); i++) {
			DPRINT << "OUTPUT PAGE[" << i << "]" << (uint32_t) addr_ptr[print_index++] << ENDL();
		}
	}

}
