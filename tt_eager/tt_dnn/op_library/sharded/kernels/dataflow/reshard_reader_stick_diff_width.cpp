// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"


inline void print_stride(
					uint32_t local_addr,
					uint32_t stride_size,
					uint32_t num_floats_per_row = 0) {

  	volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_addr);
	for(uint32_t i=0; i< stride_size/2; i++) {
		if(num_floats_per_row > 0 and i%num_floats_per_row ==0 and i != 0){
			DPRINT << ENDL();
		}
		uint32_t num = (uint32_t)ptr[0];
		num = num << 16;
 				float * f_ptr = reinterpret_cast<float *>(&num);
		DPRINT << f_ptr[0] << " ";
		ptr++;
	}
	DPRINT << ENDL();
}





inline void read_into_scratchpad(
	uint32_t core_x,
	uint32_t core_y,
	uint32_t input_base_addr,
	uint32_t scratchpad_l1_addr,
	uint32_t offset,
	uint32_t input_size,
	uint32_t input_size_per_row,
	uint32_t alignment_per_row
	) {
	uint64_t noc_address = get_noc_addr(core_x, core_y, input_base_addr);

	if(alignment_per_row != 0) {
		uint32_t begininning_alignment = offset/input_size_per_row * alignment_per_row;
		noc_address += begininning_alignment + offset;

		uint32_t amount_left = input_size;
		uint32_t amount_read;

		uint32_t offset_in_last_row = offset % input_size_per_row;

		if(amount_left > (input_size_per_row - offset_in_last_row)) {
			amount_read = input_size_per_row - offset_in_last_row;
		}
		else {
			amount_read = amount_left;
		}

		while(amount_left > 0 ) {
			noc_async_read(noc_address, scratchpad_l1_addr, amount_read);
			amount_left -= amount_read;
			if(amount_left > (input_size_per_row - offset_in_last_row)) {
				amount_read = input_size_per_row - offset_in_last_row;
			}
			else {
				amount_read = amount_left;
			}
			noc_address += amount_read + alignment_per_row;
			scratchpad_l1_addr += amount_read;
		}
	}
	else {
		noc_address += offset;
		noc_async_read(noc_address, scratchpad_l1_addr, input_size);
	}

}


inline uint32_t write_from_scratchpad(
	uint32_t l1_base_addr,
	uint32_t scratchpad_l1_addr,
	uint32_t offset,
	uint32_t output_size,
	uint32_t output_size_per_row,
	uint32_t alignment_per_row
	) {

	uint64_t noc_address = get_noc_addr(scratchpad_l1_addr);
	uint32_t l1_write_addr;

	if(alignment_per_row != 0) {
		uint32_t begininning_alignment = offset/output_size_per_row * alignment_per_row;
		l1_write_addr = l1_base_addr + begininning_alignment + offset;

		uint32_t amount_left = output_size;
		uint32_t amount_write;

		uint32_t offset_in_last_row = offset % output_size_per_row;

		if(amount_left > (output_size_per_row - offset_in_last_row)) {
			amount_write = output_size_per_row - offset_in_last_row;
		}
		else {
			amount_write = amount_left;
		}

		while(amount_left > 0 ) {
			noc_async_read(noc_address, l1_write_addr, amount_write);
			amount_left -= amount_write;
			if(amount_left > (output_size_per_row - offset_in_last_row)) {
				amount_write = output_size_per_row - offset_in_last_row;
			}
			else {
				amount_write = amount_left;
			}
			l1_write_addr += amount_write + alignment_per_row;
			scratchpad_l1_addr += amount_write;
		}
	}
	else {
		l1_write_addr = l1_base_addr + offset;
		noc_async_read(noc_address, l1_write_addr, output_size);
	}
	return l1_write_addr -  l1_base_addr;
}




void kernel_main() {
	constexpr uint32_t shard_cb = get_compile_time_arg_val(0);
	constexpr uint32_t num_x_cores = get_compile_time_arg_val(1);
	constexpr uint32_t num_y_cores = get_compile_time_arg_val(2);
	constexpr uint32_t page_size = get_compile_time_arg_val(3);
	constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
	constexpr uint32_t output_page_size = get_compile_time_arg_val(5);
	constexpr uint32_t temp_cb_0 = get_compile_time_arg_val(6);
	constexpr uint32_t input_alignment_amount = get_compile_time_arg_val(7);
	constexpr uint32_t output_alignment_amount = get_compile_time_arg_val(8);

	uint32_t y_offset = num_x_cores;

	uint32_t arg_index = num_x_cores + num_y_cores;
	const uint32_t input_shard_addr  = get_arg_val<uint32_t>(arg_index++);
	const uint32_t num_output_pages = get_arg_val<uint32_t>(arg_index++);
	const uint32_t num_ranges = get_arg_val<uint32_t>(arg_index++);
	const uint32_t output_page_offset = get_arg_val<uint32_t>(arg_index++);

	uint32_t l1_write_base_addr = get_write_ptr(shard_cb);
	uint32_t l1_write_addr = l1_write_base_addr + output_page_offset * (page_size);

	uint32_t mask_byte = 0x0ff; //8 bits
	uint32_t mask_short = 0x0ffff; //16 bits

	uint32_t scratch_pad_base_addr = get_write_ptr(temp_cb_0);




	uint32_t scratch_pad_addr = scratch_pad_base_addr;
	uint32_t total_data_read = 0;
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
		const uint32_t input_addr_offset_base = ((stride_data_offset) & mask_short) * (page_size);
		const uint32_t num_pages_per_stride = (stride_size_num_strides_skip >> 16);
		const uint32_t stride_size = num_pages_per_stride * page_size;

		uint32_t core_id_x_index = start_x_index;
		uint32_t core_id_y_index = start_y_index;

		uint32_t input_addr_offset = input_addr_offset_base;
		if(!skip) {
			uint32_t num_input_iterations = num_strides;

			//Reads input stride at a time into scratchpad
			for(uint32_t stride_idx = 0; stride_idx < num_input_iterations ; stride_idx++) {
				uint32_t core_id_x = get_arg_val<uint32_t>(core_id_x_index);
				uint32_t core_id_y = get_arg_val<uint32_t>(y_offset + core_id_y_index);


				read_into_scratchpad(core_id_x, core_id_y, input_shard_addr, scratch_pad_addr, input_addr_offset, stride_size, input_page_size, input_alignment_amount);

				scratch_pad_addr += stride_size;
				if(stride_x == 0 and stride_y == 0) {
					input_addr_offset += (stride_data + stride_size);
				}
				else {
					input_addr_offset += (stride_data);
				}
				core_id_x_index += stride_x;
				core_id_y_index += stride_y;
				total_data_read+=stride_size;
			}

		}

	}

	if(total_data_read > 0) {
		noc_async_read_barrier();
		write_from_scratchpad(l1_write_addr, scratch_pad_base_addr, 0, total_data_read, output_page_size, output_alignment_amount);
		noc_async_read_barrier();
	}
}
