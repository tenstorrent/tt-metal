// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
	constexpr uint32_t shard_cb = get_compile_time_arg_val(0);
	constexpr uint32_t num_x_cores = get_compile_time_arg_val(1);
	constexpr uint32_t num_y_cores = get_compile_time_arg_val(2);
	constexpr uint32_t page_size = get_compile_time_arg_val(3);
	constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
	constexpr uint32_t output_page_size = get_compile_time_arg_val(5);
	constexpr uint32_t temp_cb = get_compile_time_arg_val(6);

	uint32_t y_offset = num_x_cores;

	uint32_t arg_index = num_x_cores + num_y_cores;
	const uint32_t input_shard_addr  = get_arg_val<uint32_t>(arg_index++);
	const uint32_t num_output_pages = get_arg_val<uint32_t>(arg_index++);
	const uint32_t num_ranges = get_arg_val<uint32_t>(arg_index++);
	const uint32_t output_page_offset = get_arg_val<uint32_t>(arg_index++);

	uint32_t input_stride_allignment = 16;

	uint32_t l1_write_base_addr = get_write_ptr(shard_cb);
	uint32_t l1_write_addr = l1_write_base_addr + output_page_offset * page_size;

	uint32_t mask_byte = 0x0ff; //8 bits
	uint32_t mask_short = 0x0ffff; //16 bits

	DPRINT << "IN RESHARD READER STICK DIFF WIDTH with  " <<  num_ranges << " ranges " << ENDL();
	DPRINT << "Num x cores  " <<  num_x_cores << ENDL();
	DPRINT << "Num y cores  " <<  num_x_cores << ENDL();
	DPRINT << "Input shard addr  " << HEX() <<  input_shard_addr << DEC() << ENDL();
	DPRINT << "num output pages " <<  num_output_pages << ENDL();
	DPRINT << "INPUT PAGE SIZE " << input_page_size << ENDL();
	DPRINT << "OUTPUT PAGE SIZE " << output_page_size << ENDL();
	DPRINT << "NUM RANGES " << num_ranges << ENDL();
	uint32_t scratch_pad_base_addr = get_write_ptr(temp_cb);

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
		const uint32_t offset = ((stride_data_offset) & mask_short) * (page_size + input_stride_allignment);
		const uint32_t num_pages_per_stride = (stride_size_num_strides_skip >> 16);
		const uint32_t stride_size = num_pages_per_stride * page_size;

		uint32_t addr_offset = offset;
		uint32_t core_id_x_index = start_x_index;
		uint32_t core_id_y_index = start_y_index;

		DPRINT << "STRIDE SIZE " << stride_size << ENDL();
		DPRINT << "BASE INPUT SHARD ADDR " << HEX() << input_shard_addr << DEC() << ENDL();
		DPRINT << "PAGE SIZE " << page_size << ENDL();
		DPRINT << "SHARD CB " << shard_cb << ENDL();
		DPRINT << "WRITE BASE ADDR " << HEX() << l1_write_base_addr << DEC() << ENDL();
		DPRINT << "WRITE OFFSET " << HEX() << output_page_offset * page_size << DEC() << ENDL();
		DPRINT << "STRIDE DATA " << stride_data << ENDL();
		DPRINT << "STRIDE X " << stride_x << ENDL();
		DPRINT << "STRIDE Y " << stride_y << ENDL();
		uint32_t scratch_pad_addr = scratch_pad_base_addr;
		if(!skip) {
			DPRINT << "FIRST CASE " << ENDL();
			//uint32_t num_input_iterations = num_strides * num_output_pages;
			uint32_t num_input_iterations = num_strides;
			DPRINT << "NUM INPUT ITERATIONS " << num_input_iterations << ENDL();
			uint32_t total_data_read = 0;
			for(uint32_t stride_idx = 0; stride_idx < num_input_iterations ; stride_idx++) {
				uint32_t core_id_x = get_arg_val<uint32_t>(core_id_x_index);
				uint32_t core_id_y = get_arg_val<uint32_t>(y_offset + core_id_y_index);
				uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
						input_shard_addr + addr_offset);
				DPRINT << "READING STRIDE " << stride_idx << ENDL();
				DPRINT << "CORE_ID_X_INDEX " << core_id_x_index << " CORE_ID_Y_INDEX " << core_id_y_index << ENDL();
				DPRINT << "CORE_ID_X " << core_id_x << " CORE_ID_Y " << core_id_y << ENDL();
				DPRINT << "ADDR OFFSET " << HEX() << addr_offset << DEC() << ENDL();
				//need to do this if the input or output page width is diff and has diff alignment
				DPRINT << "READING ADDR " << HEX() << (noc_address) << DEC() << ENDL();
				DPRINT << "SCRATCH ADDR " << HEX() << scratch_pad_addr << DEC() << ENDL();
				noc_async_read(noc_address, scratch_pad_addr, input_page_size);
				noc_async_read_barrier();
   				volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_pad_addr);
				DPRINT << "FLOATS IN SCRATCH PAD INPUT ROW ";
				for(uint32_t i=0; i< input_page_size/2; i++) {
					uint32_t num = (uint32_t)ptr[0];
					num = num << 16;
   					float * f_ptr = reinterpret_cast<float *>(&num);
					DPRINT << f_ptr[0] << " ";
					ptr++;
				}
				DPRINT << ENDL();


				scratch_pad_addr += input_page_size;

				if(stride_x == 0 and stride_y == 0) {
					addr_offset += (stride_data + stride_size + input_stride_allignment);
				}
				else {
					addr_offset += (stride_data);
				}
				core_id_x_index += stride_x;
				core_id_y_index += stride_y;
				total_data_read+=stride_size;
			}
			noc_async_read_barrier();
			uint32_t num_output_pages_in_range = total_data_read/output_page_size;
			scratch_pad_addr = scratch_pad_base_addr;
			uint32_t num_output_iterations = num_output_pages_in_range;
			for(uint32_t stride_idx = 0; stride_idx < num_output_iterations; stride_idx++) {
				//local copy from scratchpad to output shard
				DPRINT << "WRITING ADDR " << HEX() << (l1_write_addr) << DEC() << ENDL();
				DPRINT << "SCRATCH ADDR " << HEX() << scratch_pad_addr << DEC() << ENDL();
				noc_async_read(get_noc_addr(scratch_pad_addr), l1_write_addr, output_page_size);
				DPRINT << "FLOATS IN SCRATCH PAD OUTPUT ROW ";
   				volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_pad_addr);
				for(uint32_t i=0; i< output_page_size/2; i++) {
					uint32_t num = (uint32_t)ptr[0];
					num = num << 16;
   					float * f_ptr = reinterpret_cast<float *>(&num);
					DPRINT << f_ptr[0] << " ";
					ptr++;

				}
				DPRINT << ENDL();
				noc_async_read_barrier();
				DPRINT << "FLOATS IN  OUTPUT ROW ";
   				volatile tt_l1_ptr uint16_t* ptr2= reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
				for(uint32_t i=0; i< output_page_size/2; i++) {
					uint32_t num = (uint32_t)ptr2[0];
					num = num << 16;
   					float * f_ptr = reinterpret_cast<float *>(&num);
					DPRINT << f_ptr[0] << " ";
					ptr++;

				}
				l1_write_addr += output_page_size;
				scratch_pad_addr += output_page_size;
			}
		}

	}
	noc_async_read_barrier();

}
