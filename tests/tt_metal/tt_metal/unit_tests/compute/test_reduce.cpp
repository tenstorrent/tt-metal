// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <math.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "test_golden_impls.hpp"
#include "common/test_tiles.hpp"
#include "common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace constants;

namespace unit_tests::compute::reduce {

enum ReduceDim : uint8_t {
    H = 0,
    W = 1,
    HW = 2
};

enum ReduceType : uint8_t {
    SUM = 0,
    AVG = 1,
    MAX = 2
};
struct ReduceConfig {
    bool short_init = false;
    std::vector<uint32_t> shape;
    ReduceDim reduce_dim;
    ReduceType reduce_type = ReduceType::SUM;
    float data_gen_rand_max;
    int data_gen_seed;
    float data_gen_offset;
    float atol;
    float rtol;
    std::function<std::vector<uint16_t>(const std::vector<uint16_t>&, const std::vector<uint32_t>&, float, uint8_t, bool)> golden_function;
    std::vector<uint32_t> result_shape;
    bool math_only_reduce = false;
    bool fp32_dest_acc = false;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
};

void set_math_fid_masks(uint16_t &math_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    auto arch = get_arch_from_string(get_env_arch_name());
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: { break; }
        case MathFidelity::HiFi2:
        case MathFidelity::LoFi: { math_fid_mask = (arch == tt::ARCH::GRAYSKULL) ? 0xFFF8 : 0xFFFE; break; }
        default: { TT_THROW("Unsupported MathFidelity={}", math_fidelity); break; }
    }
}

void add_reader_writer_kernels(tt_metal::Program &program, const CoreCoord &logical_core, const ReduceConfig &test_config, std::shared_ptr<tt_metal::Buffer> src_dram_buffer, std::shared_ptr<tt_metal::Buffer> dst_dram_buffer) {
    uint32_t W = test_config.shape[3], H = test_config.shape[2], NC = test_config.shape[1]*test_config.shape[0];
    uint32_t HW = H*W;
    uint32_t N = test_config.shape[0]*test_config.shape[1];
    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;
    uint32_t num_tensor_tiles = NC*H*W / (TILE_WIDTH*TILE_HEIGHT);
    float scaler = ((test_config.reduce_type == ReduceType::MAX) or (test_config.reduce_dim == ReduceDim::HW)) ? 1.0f : (test_config.reduce_dim == ReduceDim::H ? 1.0f/H : 1.0f/W);

    switch (test_config.reduce_dim) {
        case ReduceDim::H: {
            bfloat16 bfloat_scaler_value = bfloat16(scaler);
            uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
            std::vector<uint32_t> reader_compile_args = {(std::uint32_t) true, packed_scaler_value};
            std::map<string, string> reader_defines = {{"REDUCE_SCALER", "1"}};

            auto unary_reader_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp",
                logical_core,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_args, .defines = reader_defines});

            auto unary_writer_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp", // no need to transpose the output since output Ht=1
                logical_core,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});


            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel,
                logical_core,
                {
                    src_dram_buffer->address(),
                    N, Ht, Wt, Ht*Wt
                }
            );

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel,
                logical_core,
                {
                    dst_dram_buffer->address(),
                    (std::uint32_t)dst_dram_buffer->noc_coordinates().x,
                    (std::uint32_t)dst_dram_buffer->noc_coordinates().y,
                    num_tensor_tiles/Ht
                }
            );

            break;
        }
        case ReduceDim::W:
        case ReduceDim::HW: {
            auto unary_reader_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank_reduce.cpp",
                logical_core,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

            auto unary_writer_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
                logical_core,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel,
                logical_core,
                {
                    src_dram_buffer->address(),
                    (std::uint32_t)src_dram_buffer->noc_coordinates().x,
                    (std::uint32_t)src_dram_buffer->noc_coordinates().y,
                    num_tensor_tiles, NC, Ht, Wt, Ht*Wt,
                    *reinterpret_cast<uint32_t*>(&scaler),
                }
            );

            uint32_t num_tiles = test_config.reduce_dim == ReduceDim::W ? (num_tensor_tiles/Wt) : (num_tensor_tiles/(Wt*Ht));
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel,
                logical_core,
                {
                    dst_dram_buffer->address(),
                    (std::uint32_t)dst_dram_buffer->noc_coordinates().x,
                    (std::uint32_t)dst_dram_buffer->noc_coordinates().y,
                    num_tiles
                }
            );

            break;
        }
        default:
            TT_THROW("Unsupported reduce dim!");
    }
}

std::string get_reduce_dim_define_string(const ReduceDim &reduce_dim) {
    std::string reduce_dim_define_str;
    switch (reduce_dim) {
        case ReduceDim::H: reduce_dim_define_str = "ReduceDim::REDUCE_COL"; break;
        case ReduceDim::W: reduce_dim_define_str = "ReduceDim::REDUCE_ROW"; break;
        case ReduceDim::HW: reduce_dim_define_str = "ReduceDim::REDUCE_SCALAR"; break;
        default:
            TT_THROW("Unsupported reduce dim!");
    }
    return reduce_dim_define_str;
}

std::string get_compute_kernel_name(const ReduceDim &reduce_dim) {
    std::string compute_kernel_name;
    switch (reduce_dim) {
        case ReduceDim::H: compute_kernel_name = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_h.cpp"; break;
        case ReduceDim::W: compute_kernel_name = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_w.cpp"; break;
        case ReduceDim::HW: compute_kernel_name = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_hw.cpp"; break;
        default:
            TT_THROW("Unsupported reduce dim!");
    }
    return compute_kernel_name;
}

void run_single_core_reduce_program(tt_metal::Device* device, const ReduceConfig& test_config) {
    Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t W = test_config.shape[3], H = test_config.shape[2], NC = test_config.shape[1]*test_config.shape[0];
    uint32_t HW = H*W;
    uint32_t N = test_config.shape[0]*test_config.shape[1];
    TT_FATAL(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0, "Error");
    TT_FATAL(H > 0 && W > 0 && NC > 0, "Error");
    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_golden_elements;
    switch (test_config.reduce_dim) {
        case ReduceDim::H: num_golden_elements = NC*W*32/2; break; // expecting one tile in H, and half the elements since the vector packs 2 uint16_ts
        case ReduceDim::W: num_golden_elements = NC*H*TILE_WIDTH/2; break; // expecting one tile in H, and half the elements since the vector packs 2 uint16_ts
        case ReduceDim::HW: num_golden_elements = NC*32*32/2; break; // expecting one tile in H, and half the elements since the vector packs 2 uint16_ts
        default:
            TT_THROW("Unsupported reduce dim!");
    }

    float scaler = ((test_config.reduce_type == ReduceType::MAX) or test_config.reduce_dim == ReduceDim::HW) ? 1.0f : (test_config.reduce_dim == ReduceDim::H ? 1.0f/H : 1.0f/W);
    uint32_t num_tensor_tiles = NC*H*W / (TILE_WIDTH*TILE_HEIGHT);
    uint32_t divisor = test_config.reduce_dim == ReduceDim::W ? Wt : Ht;
    TT_FATAL(num_tensor_tiles%divisor == 0, "Error");

    uint32_t single_tile_bytes = 2 * 1024;
    uint32_t dram_buffer_size = single_tile_bytes * num_tensor_tiles;

    uint32_t src_page_size = single_tile_bytes;
    uint32_t dst_page_size = single_tile_bytes;

    tt_metal::InterleavedBufferConfig src_config{
            .device=device,
            .size = dram_buffer_size,
            .page_size = src_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };


    uint32_t output_size_bytes;
    switch (test_config.reduce_dim) {
        case ReduceDim::H: output_size_bytes = dram_buffer_size / Ht; break;
        case ReduceDim::W: output_size_bytes = dram_buffer_size / Wt; break;
        case ReduceDim::HW: output_size_bytes = dram_buffer_size / (Ht * Wt); break;
        default:
            TT_THROW("Unsupported reduce dim!");
    }

    tt_metal::InterleavedBufferConfig dst_config{
            .device=device,
            .size = output_size_bytes,
            .page_size = dst_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt_metal::Buffer> src_dram_buffer = CreateBuffer(src_config);
    std::shared_ptr<tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dst_config);

    uint32_t src0_cb_index = 0;
    uint32_t num_buffer_tiles = 32;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_bytes);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_buffer_tiles = 32;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_buffer_tiles * single_tile_bytes, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, single_tile_bytes);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    tt_metal::CircularBufferConfig cb_temp_reduce_tile_config = tt_metal::CircularBufferConfig(2 * single_tile_bytes, {{CB::c_in2, tt::DataFormat::Float16_b}})
        .set_page_size(CB::c_in2, single_tile_bytes);
    auto cb_temp_reduce_tile = tt_metal::CreateCircularBuffer(program, core, cb_temp_reduce_tile_config);

    add_reader_writer_kernels(program, core, test_config, src_dram_buffer, dst_dram_buffer);

    vector<uint32_t> compute_kernel_args = {
        uint(Ht),
        uint(Wt),
        uint(NC),
    };

    std::map<string, string> reduce_defines = {
        // {"REDUCE_OP", test_config.do_max ? "PoolType::MAX" : "PoolType::SUM"},
        {"REDUCE_DIM", get_reduce_dim_define_string(test_config.reduce_dim)}
    };
    switch (test_config.reduce_type) {
        case ReduceType::SUM: {reduce_defines["REDUCE_OP"] = "PoolType::SUM"; break;}
        case ReduceType::AVG: {reduce_defines["REDUCE_OP"] = "PoolType::AVG"; break;}
        case ReduceType::MAX: {reduce_defines["REDUCE_OP"] = "PoolType::MAX"; break;}
    }
    if (test_config.short_init)
    {
        reduce_defines["SHORT_INIT"] = "1";
    }

    if (test_config.math_only_reduce) {
        reduce_defines["MATH_ONLY"] = "1";
    } else {
        reduce_defines["MATH_ONLY"] = "0";
    }
    if (test_config.fp32_dest_acc) {
        reduce_defines["DST_ACCUM_MODE"] = "1";
    } else {
        reduce_defines["DST_ACCUM_MODE"] = "0";
    }

    std::string compute_kernel_name = get_compute_kernel_name(test_config.reduce_dim);

    auto reduce_compute_kernel = tt_metal::CreateKernel(
        program,
        compute_kernel_name,
        core,
        tt_metal::ComputeConfig{.math_fidelity = test_config.math_fidelity,
                                .compile_args = compute_kernel_args,
                                .defines = reduce_defines});

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();

    vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, test_config.data_gen_rand_max, test_config.data_gen_seed, test_config.data_gen_offset);

    tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

    tt_metal::detail::LaunchProgram(device, program);

    // The kernel will view the input as TILED32_4FACES
    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    EXPECT_EQ(result_vec.size(), num_golden_elements);

    int argfail = -1;
    auto comparison_function = [&](float a, float b) {
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        auto result = (absdiff <= test_config.atol) || absdiff < test_config.rtol * maxabs;
        return result;
    };

    // recover a linear view of input vector for consumption by gold_ function
    auto u16_src0_vec = u16_from_u32_vector(src_vec);
    std::vector<uint16_t> src_linear = convert_layout<uint16_t>(u16_src0_vec, test_config.shape, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
    std::vector<uint16_t> gold_reduced = test_config.golden_function(src_linear, test_config.shape, scaler, uint8_t(test_config.reduce_type), true); // result is uint16_t untilized

    if (test_config.reduce_type == ReduceType::AVG) {
        uint16_t math_fid_mask = 0xFFFF;
        set_math_fid_masks(math_fid_mask, test_config.math_fidelity);
        for (auto i = 0; i < gold_reduced.size(); i++) {
            gold_reduced[i] = gold_reduced[i] & math_fid_mask;
        }
    }

    // Tilize from row major and convert to pairs (uint32_t)
    auto gold_4f_u32 = u32_from_u16_vector(convert_layout<uint16_t>(gold_reduced, test_config.result_shape, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES));

    bool pass = packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);

    if (!pass)
        log_error(LogTest, "Failure position={}", argfail);

    EXPECT_TRUE(pass);
}

} // namespace unit_tests::compute::reduce

using namespace unit_tests::compute::reduce;

TEST_F(DeviceFixture, ComputeReduceH) {
    if (this->arch_ != tt::ARCH::BLACKHOLE) {
        // (issue #10181: disabling due to sporadic failures in slow dispatch mode)
        GTEST_SKIP();
    }
    std::vector<uint32_t> shape = {1, 3, 19*TILE_HEIGHT, 17*TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], TILE_HEIGHT, shape[3]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) continue;
        tt::log_info(tt::LogTest, "Math Fidelity = {}", math_fid);
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            // Currently AVG is not tested
            if (reduce_type == ReduceType::AVG) continue;
            tt::log_info(tt::LogTest, "Reduce type = {}", reduce_type);
            for (bool fp32_dest_acc : {true, false}) {
                tt::log_info(tt::LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
                ReduceConfig test_config = {
                    .shape = shape,
                    .reduce_dim = ReduceDim::H,
                    .reduce_type = ReduceType(reduce_type),
                    .data_gen_rand_max = 10.0f,
                    .data_gen_seed = 0x1234,
                    .data_gen_offset = -4.5f,
                    .atol = 1e-2f,
                    .rtol = 0.06f,
                    .golden_function = unit_tests::compute::gold_reduce_h,
                    .result_shape = result_shape,
                    .fp32_dest_acc = fp32_dest_acc,
                    .math_fidelity = MathFidelity(math_fid),
                };
                run_single_core_reduce_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeReduceW) {
    std::vector<uint32_t> shape = {1, 3, 17*TILE_HEIGHT, 19*TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) continue;
        tt::log_info(tt::LogTest, "Math Fidelity = {}", math_fid);
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            // Currently AVG is not tested
            if (reduce_type == ReduceType::AVG) continue;
            tt::log_info(tt::LogTest, "Reduce type = {}", reduce_type);
            for (bool fp32_dest_acc : {true, false}) {
                tt::log_info(tt::LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
                ReduceConfig test_config = {
                    .shape = shape,
                    .reduce_dim = ReduceDim::W,
                    .reduce_type = ReduceType(reduce_type),
                    .data_gen_rand_max = 1.0f,
                    .data_gen_seed = 0x1234,
                    .data_gen_offset = 0.0f,
                    .atol = 0.20f,
                    .rtol = 0.10f,
                    .golden_function = unit_tests::compute::gold_reduce_w,
                    .result_shape = result_shape,
                    .fp32_dest_acc = fp32_dest_acc,
                    .math_fidelity = MathFidelity(math_fid),
                };
                run_single_core_reduce_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeReduceHW) {
    std::vector<uint32_t> shape = {1, 2, 7*TILE_HEIGHT, 5*TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], 32, 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) continue;
            tt::log_info(tt::LogTest, "Math Fidelity = {}", math_fid);
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            // Currently AVG is not tested
            if (reduce_type == ReduceType::AVG) continue;
            tt::log_info(tt::LogTest, "Reduce type = {}", reduce_type);
            for (bool fp32_dest_acc : {true, false}) {
                // Currently fp32 dest unsupported with reduce scalar
                if (fp32_dest_acc) continue;
                tt::log_info(tt::LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
                ReduceConfig test_config = {
                    .shape = shape,
                    .reduce_dim = ReduceDim::HW,
                    .reduce_type = ReduceType(reduce_type),
                    .data_gen_rand_max = 1.0f,
                    .data_gen_seed = 0x1234,
                    .data_gen_offset = -0.48f,
                    .atol = 1e-2f,
                    .rtol = 0.06f,
                    .golden_function = unit_tests::compute::gold_reduce_hw,
                    .result_shape = result_shape,
                    .fp32_dest_acc = fp32_dest_acc,
                    .math_fidelity = MathFidelity(math_fid)
                };
                run_single_core_reduce_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeReduceHMathOnly) {
    if (this->arch_ != tt::ARCH::BLACKHOLE) {
        // (issue #10181: disabling due to sporadic failures in slow dispatch mode)
        GTEST_SKIP();
    }
    std::vector<uint32_t> shape = {1, 3, 19*TILE_HEIGHT, 17*TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], TILE_HEIGHT, shape[3]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) continue;
        tt::log_info(tt::LogTest, "Math Fidelity = {}", math_fid);
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            // Currently AVG is not tested
            if (reduce_type == ReduceType::AVG) continue;
            tt::log_info(tt::LogTest, "Reduce type = {}", reduce_type);
            for (bool fp32_dest_acc : {true, false}) {
                tt::log_info(tt::LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
                ReduceConfig test_config = {
                    .shape = shape,
                    .reduce_dim = ReduceDim::H,
                    .reduce_type = ReduceType(reduce_type),
                    .data_gen_rand_max = 10.0f,
                    .data_gen_seed = 0x1234,
                    .data_gen_offset = -4.5f,
                    .atol = 1e-2f,
                    .rtol = 0.06f,
                    .golden_function = unit_tests::compute::gold_reduce_h,
                    .result_shape = result_shape,
                    .math_only_reduce = true,
                    .fp32_dest_acc = fp32_dest_acc,
                    .math_fidelity = MathFidelity(math_fid)
                };
                run_single_core_reduce_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeReduceWMathOnly) {
    std::vector<uint32_t> shape = {1, 3, 17*TILE_HEIGHT, 19*TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) continue;
        tt::log_info(tt::LogTest, "Math Fidelity = {}", math_fid);
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            // Currently AVG is not tested
            if (reduce_type == ReduceType::AVG) continue;
            tt::log_info(tt::LogTest, "Reduce type = {}", reduce_type);
            for (bool fp32_dest_acc : {true, false}) {
                tt::log_info(tt::LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
                ReduceConfig test_config = {
                    .shape = shape,
                    .reduce_dim = ReduceDim::W,
                    .reduce_type = ReduceType(reduce_type),
                    .data_gen_rand_max = 1.0f,
                    .data_gen_seed = 0x1234,
                    .data_gen_offset = 0.0f,
                    .atol = 0.20f,
                    .rtol = 0.10f,
                    .golden_function = unit_tests::compute::gold_reduce_w,
                    .result_shape = result_shape,
                    .math_only_reduce = true,
                    .fp32_dest_acc = fp32_dest_acc,
                    .math_fidelity = MathFidelity(math_fid)
                };
                run_single_core_reduce_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeReduceHWMathOnly) {
    std::vector<uint32_t> shape = {1, 2, 7*TILE_HEIGHT, 5*TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], 32, 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) continue;
            tt::log_info(tt::LogTest, "Math Fidelity = {}", math_fid);
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            // Currently AVG is not tested
            if (reduce_type == ReduceType::AVG) continue;
            tt::log_info(tt::LogTest, "Reduce type = {}", reduce_type);
            for (bool fp32_dest_acc : {true, false}) {
                // Currently fp32 dest unsupported with reduce scalar
                if (fp32_dest_acc) continue;
                tt::log_info(tt::LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
                ReduceConfig test_config = {
                    .shape = shape,
                    .reduce_dim = ReduceDim::HW,
                    .reduce_type = ReduceType(reduce_type),
                    .data_gen_rand_max = 1.0f,
                    .data_gen_seed = 0x1234,
                    .data_gen_offset = -0.48f,
                    .atol = 1e-2f,
                    .rtol = 0.06f,
                    .golden_function = unit_tests::compute::gold_reduce_hw,
                    .result_shape = result_shape,
                    .math_only_reduce = true,
                    .fp32_dest_acc = fp32_dest_acc,
                    .math_fidelity = MathFidelity(math_fid)
                };
                run_single_core_reduce_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeReduceHShortInit) {
    if (this->arch_ != tt::ARCH::BLACKHOLE) {
        // (issue #10181: disabling due to sporadic failures in slow dispatch mode)
        GTEST_SKIP();
    }
    std::vector<uint32_t> shape = {1, 3, 19*TILE_HEIGHT, 17*TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], TILE_HEIGHT, shape[3]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) continue;
        tt::log_info(tt::LogTest, "Math Fidelity = {}", math_fid);
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            // Currently AVG is not tested
            if (reduce_type == ReduceType::AVG) continue;
            tt::log_info(tt::LogTest, "Reduce type = {}", reduce_type);
            for (bool fp32_dest_acc : {true, false}) {
                tt::log_info(tt::LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
                ReduceConfig test_config = {
                    .short_init = true,
                    .shape = shape,
                    .reduce_dim = ReduceDim::H,
                    .reduce_type = ReduceType(reduce_type),
                    .data_gen_rand_max = 10.0f,
                    .data_gen_seed = 0x1234,
                    .data_gen_offset = -4.5f,
                    .atol = 1e-2f,
                    .rtol = 0.06f,
                    .golden_function = unit_tests::compute::gold_reduce_h,
                    .result_shape = result_shape,
                    .fp32_dest_acc = fp32_dest_acc,
                    .math_fidelity = MathFidelity(math_fid)
                };
                run_single_core_reduce_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeReduceWShortInit) {
    std::vector<uint32_t> shape = {1, 3, 17*TILE_HEIGHT, 19*TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) continue;
        tt::log_info(tt::LogTest, "Math Fidelity = {}", math_fid);
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            // Currently AVG is not tested
            if (reduce_type == ReduceType::AVG) continue;
            tt::log_info(tt::LogTest, "Reduce type = {}", reduce_type);
            for (bool fp32_dest_acc : {true, false}) {
                tt::log_info(tt::LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
                ReduceConfig test_config = {
                    .short_init = true,
                    .shape = shape,
                    .reduce_dim = ReduceDim::W,
                    .reduce_type = ReduceType(reduce_type),
                    .data_gen_rand_max = 1.0f,
                    .data_gen_seed = 0x1234,
                    .data_gen_offset = 0.0f,
                    .atol = 0.20f,
                    .rtol = 0.10f,
                    .golden_function = unit_tests::compute::gold_reduce_w,
                    .result_shape = result_shape,
                    .fp32_dest_acc = fp32_dest_acc,
                    .math_fidelity = MathFidelity(math_fid)
                };
                run_single_core_reduce_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeReduceHWShortInit) {
    std::vector<uint32_t> shape = {1, 2, 7*TILE_HEIGHT, 5*TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], 32, 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) continue;
        tt::log_info(tt::LogTest, "Math Fidelity = {}", math_fid);
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            // Currently AVG is not tested
            if (reduce_type == ReduceType::AVG) continue;
            tt::log_info(tt::LogTest, "Reduce type = {}", reduce_type);
            for (bool fp32_dest_acc : {true, false}) {
                // Currently fp32 dest unsupported with reduce scalar
                if (fp32_dest_acc) continue;
                tt::log_info(tt::LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
                ReduceConfig test_config = {
                    .short_init = true,
                    .shape = shape,
                    .reduce_dim = ReduceDim::HW,
                    .reduce_type = ReduceType(reduce_type),
                    .data_gen_rand_max = 1.0f,
                    .data_gen_seed = 0x1234,
                    .data_gen_offset = -0.48f,
                    .atol = 1e-2f,
                    .rtol = 0.06f,
                    .golden_function = unit_tests::compute::gold_reduce_hw,
                    .result_shape = result_shape,
                    .fp32_dest_acc = fp32_dest_acc,
                    .math_fidelity = MathFidelity(math_fid)
                };
                run_single_core_reduce_program(this->devices_.at(0), test_config);
            }
        }
    }
}
