// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"


inline void print_stride(
					uint32_t local_addr,
					uint32_t stride_size) {

  	volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_addr);
	for(uint32_t i=0; i< stride_size/2; i++) {
		uint32_t num = (uint32_t)ptr[0];
		num = num << 16;
 				float * f_ptr = reinterpret_cast<float *>(&num);
		DPRINT << f_ptr[0] << " ";
		ptr++;
	}
	DPRINT << ENDL();
}

inline void read_print_stride(uint32_t input_shard_addr,
					uint32_t scratch_pad_addr,
					uint32_t core_x,
					uint32_t core_y,
					uint32_t addr_offset,
					uint32_t stride_size) {
	uint64_t noc_address = get_noc_addr(core_x, core_y,
						input_shard_addr + addr_offset);
	noc_async_read(noc_address, scratch_pad_addr, stride_size);
	noc_async_read_barrier();
	print_stride(scratch_pad_addr, stride_size);


}



void kernel_main() {
	constexpr uint32_t shard_cb = get_compile_time_arg_val(0);
	constexpr uint32_t num_x_cores = get_compile_time_arg_val(1);
	constexpr uint32_t num_y_cores = get_compile_time_arg_val(2);
	constexpr uint32_t page_size = get_compile_time_arg_val(3);
	constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
	constexpr uint32_t output_page_size = get_compile_time_arg_val(5);
	constexpr uint32_t input_page_allignment = get_compile_time_arg_val(6);
	constexpr uint32_t output_page_allignment = get_compile_time_arg_val(7);
	constexpr uint32_t temp_cb = get_compile_time_arg_val(8);

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

	uint32_t scratch_pad_base_addr = get_write_ptr(temp_cb);


	//DPRINT << "PRINTING SHARD ON CORE (0,0)" << ENDL();
	//read_print_stride(input_shard_addr, scratch_pad_base_addr, 1, 4, 0, 16);


	DPRINT << "PAGE SIZE " << page_size << ENDL();
	DPRINT << "INPUT PAGE SIZE " << input_page_size << ENDL();
	DPRINT << "OUTPUT PAGE SIZE " << output_page_size << ENDL();
	DPRINT << "INPUT PAGE ALLIGNMENT " << input_page_allignment << ENDL();
	DPRINT << "OUTPUT PAGE ALLIGNMENT " << output_page_allignment << ENDL();
	DPRINT << "NUM RANGES " << num_ranges << ENDL();
	DPRINT << "OUTPUT PAGE OFFSET " << output_page_offset << ENDL();
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

		uint32_t input_addr_offset = ((stride_data_offset) & mask_short) * (page_size);
		const uint32_t num_pages_per_stride = (stride_size_num_strides_skip >> 16);
		const uint32_t stride_size = num_pages_per_stride * page_size;
		DPRINT << "NUM PAGES PER STRIDE " << num_pages_per_stride << ENDL();

		uint32_t core_id_x_index = start_x_index;
		uint32_t core_id_y_index = start_y_index;

		uint32_t scratch_pad_addr = scratch_pad_base_addr;
		if(!skip) {
			uint32_t num_input_iterations = num_strides;
			uint32_t total_data_read = 0;

			DPRINT << "TOTAL INPUT ITERATIONS " << num_input_iterations << ENDL();
			//Reads input stride at a time into scratchpad
			for(uint32_t stride_idx = 0; stride_idx < num_input_iterations ; stride_idx++) {
				uint32_t core_id_x = get_arg_val<uint32_t>(core_id_x_index);
				uint32_t core_id_y = get_arg_val<uint32_t>(y_offset + core_id_y_index);
				uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
						input_shard_addr + input_addr_offset);
				DPRINT << "ADDR OFFSET " << HEX() << input_addr_offset << DEC() << ENDL();
				DPRINT << "STRIDE SIZE " << stride_size << ENDL();
				DPRINT << "CORE_ID_X " << core_id_x << ENDL();
				DPRINT << "CORE_ID_Y " << core_id_x << ENDL();
				noc_async_read(noc_address, scratch_pad_addr, stride_size);
				noc_async_read_barrier();
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
			noc_async_read_barrier();


			DPRINT << "TOTAL DATA READ " << total_data_read << ENDL();
			DPRINT << "FLOATS IN SCRATCH PAD INPUT ROW ";
			print_stride(scratch_pad_base_addr, total_data_read);


			//At this point entire shard is in scratchpad

			if(output_page_size < page_size){
				uint32_t num_output_pages_in_range = total_data_read/output_page_size;
				scratch_pad_addr = scratch_pad_base_addr;
				uint32_t num_output_iterations = num_output_pages_in_range;
				//writes output from scratchpad , output row at a time
				for(uint32_t stride_idx = 0; stride_idx < num_output_iterations; stride_idx++) {
					//local copy from scratchpad to output shard
					noc_async_read(get_noc_addr(scratch_pad_addr), l1_write_addr, output_page_size);
					l1_write_addr += (output_page_size);
					scratch_pad_addr += output_page_size;
					DPRINT << "FLOATS IN OUTPUT ROW " << ENDL();
					print_stride(l1_write_addr, output_page_size);
				}
				noc_async_read_barrier();
			}
			else {
				noc_async_read(get_noc_addr(scratch_pad_base_addr), l1_write_addr, total_data_read);
				DPRINT << "FLOATS IN OUTPUT ROW " << ENDL();
				print_stride(l1_write_addr, total_data_read);
			}
		}

	}
	noc_async_read_barrier();

}
