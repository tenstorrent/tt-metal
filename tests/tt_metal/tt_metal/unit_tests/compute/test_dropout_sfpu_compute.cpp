// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "device_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;

/*
* 1. Host creates one vector of data with constant non-zero values.
* 2. Device eltwise performs a unary SFPU dropout operation on the data.
* 3. Read result back and compare percentage of elements droopped out
*    with probability and if scaling is consistent with scale factor.
* */

namespace unit_tests::compute::sfpu::dropout {

struct DropoutConfig {
	float probability;
	float fill_constant;
	uint32_t seed_0;
	uint32_t seed_1;
};

bool check_dropout(std::vector<bfloat16>& src_vec, std::vector<bfloat16>& result_vec, float probability, float scale_factor) {
   bool pass = true;
   int vec_size = src_vec.size();
   int zero_count = 0;
   for(int i = 0; i < vec_size; i++) {
       auto srcf = src_vec[i].to_float();
       auto resf = result_vec[i].to_float();
       if(resf == 0.0f) {
           zero_count++;
       } else if (!is_close(resf, srcf*scale_factor)) {
           tt::log_error(tt::LogTest, "Invalid scaling for dropout src={}, res={}, scaling={}", srcf, resf, scale_factor);
           pass = false;
           break;
       }
   }

   float dropout_rate = (float) zero_count/(float) vec_size;
   bool rate_ok = is_close(probability, dropout_rate, 0.05f, 0.05f);
   if(!rate_ok) {
       tt::log_error(tt::LogTest, "Dropout rate & probability mismatch probability={}, dropout_rate={}", probability, dropout_rate);
   } else {
        tt::log_info(tt::LogTest, "dropout probability={}, dropout_rate={} ", probability, dropout_rate);
   }

   pass &=rate_ok;
   return pass;
}

bool test_dropout_standalone(tt_metal::Device* device, float probability, uint32_t seed,  float const_bias, std::vector<bfloat16>& res_vec) {
    bool pass = true;
    uint32_t int_probability = probability * (double)INT_MAX;
    float scale_factor_f = 1.0f/(1.0f - probability);
    uint32_t scale_factor;
    std::memcpy(&scale_factor, &scale_factor_f, sizeof(uint32_t));

    try {
        /*
        * Setup program to execute along with its buffers and kernels to use
        */
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};
        constexpr uint32_t single_tile_size = 2 * 1024;
        constexpr uint32_t num_tiles = 128;
        constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

        tt_metal::InterleavedBufferConfig dram_config{
                    .device= device,
                    .size = dram_buffer_size,
                    .page_size = dram_buffer_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

        std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
        const uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

        std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
        const uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        /*
         * Use circular buffers to set input and output buffers that the
         * compute engine will use.
         */
        constexpr uint32_t src0_cb_index = CB::c_in0;
        constexpr uint32_t num_input_tiles = 2;
        CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
        CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        constexpr uint32_t output_cb_index = CB::c_out0;
        constexpr uint32_t num_output_tiles = 2;
        CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}}).set_page_size(output_cb_index, single_tile_size);
        CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        /*
         * Specify data movement kernels for reading/writing data to/from
         * DRAM.
         */
        KernelHandle unary_reader_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        KernelHandle unary_writer_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        /*
         * Set the parameters that the compute kernel will use.
         */
        vector<uint32_t> compute_kernel_args = {
            num_tiles,
            1,
	    seed,
	    int_probability,
	    scale_factor
        };

        constexpr bool math_approx_mode = false;
        const std::map<std::string, std::string> sfpu_defines = {
            {"SFPU_OP_DROPOUT_INCLUDE", "1"},
        };

        KernelHandle eltwise_sfpu_kernel_id = CreateKernel(
            program,
            "/tests/tt_metal/tt_metal/test_kernels/compute/dropout_sfpu.cpp",
            core,
            ComputeConfig{
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args,
                .defines = sfpu_defines,
            }
        );

        /*
         * Create source data and write to DRAM.
         */
        std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(
            dram_buffer_size, const_bias);

        tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);

        /*
         * Configure program and runtime kernel arguments, then execute.
         */
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src0_dram_buffer->address(),
                static_cast<uint32_t>(src0_dram_buffer->noc_coordinates().x),
                static_cast<uint32_t>(src0_dram_buffer->noc_coordinates().y),
                num_tiles,
            }
        );

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_dram_buffer->address(),
                static_cast<uint32_t>(dst_dram_buffer->noc_coordinates().x),
                static_cast<uint32_t>(dst_dram_buffer->noc_coordinates().y),
                num_tiles
            }
        );

        tt_metal::detail::LaunchProgram(device, program);

        /*
         * Read the result and compare to a golden result. Record pass/fail
         * and teardown.
         */
        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        auto transform_identity = [](const bfloat16 &a) {
            return a;
        };

        // Unpack source and result vectors.
        std::vector<bfloat16> src0_vec_bfloat16 = unpack_uint32_vec_into_bfloat16_vec(src0_vec, transform_identity);
        std::vector<bfloat16> result_vec_bfloat16 = unpack_uint32_vec_into_bfloat16_vec(result_vec, transform_identity);
	res_vec = result_vec_bfloat16;
	pass &= check_dropout(src0_vec_bfloat16, result_vec_bfloat16, probability, scale_factor_f);
    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());
        throw;
    }

    return pass;
}

void test_dropout(tt_metal::Device* device, const DropoutConfig& test_config) {
    bool pass = true;
    float probability = test_config.probability;
    float fill_constant = test_config.fill_constant;
    uint32_t seed_0 = test_config.seed_0;
    uint32_t seed_1 = test_config.seed_1;

    std::vector<bfloat16> res_0, res_1, res_2;
    pass &= test_dropout_standalone(device, probability, seed_0, fill_constant, res_0);
    pass &= test_dropout_standalone(device, probability, seed_0, fill_constant, res_1);
    bool repeatable = std::equal(res_0.begin(), res_0.end(), res_1.begin());
    if(!repeatable) {
       tt::log_error(tt::LogTest, "Same parameters gave different results probability={}, seed={}", probability, seed_0);
    } else {
       tt::log_info(tt::LogTest, "Two attempts with same parameters matched");
    }
    pass &= repeatable;

    if(probability != 0.0 && probability != 1.0) {
        pass &= test_dropout_standalone(device, probability, seed_1, fill_constant, res_2);
        bool unique = !std::equal(res_0.begin(), res_0.end(), res_2.begin());
        if(!unique) {
           tt::log_error(tt::LogTest, "Different seed gave same result probability={}, seed_0={}, seed_1={}", probability, seed_0, seed_1);
        } else {
           tt::log_info(tt::LogTest, "Different seed gave different results");
        }
        pass &=unique;
    }

    EXPECT_TRUE(pass);
}

}

TEST_F(DeviceFixture, ComputeDropout) {
    if (this->arch_ != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP();
    }
    srand(0);
    int num_tests = 5;
    float fill_constant = 9.0;
    for(int i = 0; i <= num_tests; i++) {
	float probability = (float)i/(float)num_tests;
        unit_tests::compute::sfpu::dropout::DropoutConfig test_config = {
		.probability = probability,
		.fill_constant = fill_constant,
		.seed_0 = static_cast<uint32_t>(rand()),
		.seed_1 = static_cast<uint32_t>(rand())
        };
        unit_tests::compute::sfpu::dropout::test_dropout(this->devices_.at(0), test_config);
    }
}
