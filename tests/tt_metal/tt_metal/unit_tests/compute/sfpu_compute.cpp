#include <math.h>

#include <algorithm>
#include <functional>
#include <random>

#include "doctest.h"
#include "single_device_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace sfpu_util {

const map<string, string> sfpu_op_to_op_name = {
    // FIXME: #1157
    {"relu", "pack_relu_tile_to_stream(0, CB::c_out0);"},
    {"exponential", "exp_tile_init(); exp_tile(0); pack_tile(0, CB::c_out0);"},
    {"reciprocal", "recip_tile_init(); recip_tile(0); pack_tile(0, CB::c_out0);"},
    {"gelu", "gelu_tile_init(); gelu_tile(0); pack_tile(0, CB::c_out0);"},
    {"sqrt", "sqrt_tile_init(); sqrt_tile(0); pack_tile(0, CB::c_out0);"},
    {"sigmoid", "sigmoid_tile_init(); sigmoid_tile(0); pack_tile(0, CB::c_out0);"},
    {"log", "log_tile_init(); log_tile(0); pack_tile(0, CB::c_out0);"},
    {"tanh", "tanh_tile_init(); tanh_tile(0); pack_tile(0, CB::c_out0);"},
};

bfloat16 sfpu_function(const string& op_name, const bfloat16& input) {
    if (op_name == "relu") {
        return bfloat16(fmaxf(input.to_float(), 0.0f));
    } else if (op_name == "exponential") {
        return bfloat16(std::exp(input.to_float()));
    } else if (op_name == "reciprocal") {
        return bfloat16(1 / input.to_float());
    } else if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x = input.to_float();
        auto x3 = x * x * x;
        float result = x * 0.5 * (1.0 + tanhf(alpha * (x + 0.044715 * x3)));
        return bfloat16(result);
    } else if (op_name == "sqrt") {
        return bfloat16(sqrtf(input.to_float()));
    } else if (op_name == "sigmoid") {
        auto x = input.to_float();
        float result = 1 / (1 + std::exp(-x));
        return bfloat16(result);
    } else if (op_name == "log") {
        return bfloat16(logf(input.to_float()));
    } else if (op_name == "tanh") {
        return bfloat16(std::tanh(input.to_float()));
    } else {
        tt::log_fatal("Unsupported op_name in test");
        return bfloat16(0.0f);
    }
}
vector<u32> generate_packed_sfpu_input(const unsigned int numel, const string& op_name, const int seed) {
    if ((op_name == "sqrt") or (op_name == "log")) {
        return generate_packed_uniform_random_vector<u32, bfloat16>(0.0001f, 4.0f, numel, seed);
    } else if ((op_name == "exponential") or (op_name == "gelu") or (op_name == "reciprocal")) {
        auto possible_values = vector<bfloat16>({-1.0f, -0.5f, 0.5f, 1.0f});
        return generate_packed_random_vector_from_vector<u32, bfloat16>(possible_values, numel, seed);
    } else {
        return generate_packed_uniform_random_vector<u32, bfloat16>(-1.0f, 1.0f, numel, seed);
    }
}

bool is_close_packed_sfpu_output(const vector<u32>& vec_a, const vector<u32>& vec_b, const string& op_name) {
    if (op_name == "tanh") {
        return is_close_packed_vectors<bfloat16, u32>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.175f, 0.1f); });
    } else if ((op_name == "gelu") or (op_name == "relu")) {
        return is_close_packed_vectors<bfloat16, u32>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.15f); });
    } else if ((op_name == "exponential")) {
        return is_close_packed_vectors<bfloat16, u32>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.1f, 0.1f); });
    } else {
        return is_close_packed_vectors<bfloat16, u32>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.06f, 0.006f); });
    }
}

}  // namespace sfpu_util

namespace unit_tests::sfpu_compute {

struct SfpuConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    size_t output_dram_channel = 0;
    size_t output_dram_byte_address = 0;
    size_t input_dram_channel = 0;
    size_t input_dram_byte_address = 0;
    size_t l1_input_byte_address = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    size_t l1_output_byte_address = 0;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores = {{}};
    std::string sfpu_op = "";
};

/// @brief Does Dram --> Reader --> CB --> Sfpu Compute --> CB --> Writer --> Dram. So far, enqueue APIs only added to
/// grayskull
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_all_same_buffer(tt_metal::Device* device, const SfpuConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input_dram_buffer = tt_metal::Buffer(
        device,
        byte_size,
        test_config.input_dram_byte_address,
        test_config.input_dram_channel,
        byte_size,
        tt_metal::BufferType::DRAM);
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = tt_metal::Buffer(
        device,
        byte_size,
        test_config.output_dram_byte_address,
        test_config.output_dram_channel,
        byte_size,
        tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    vector<u32> compute_kernel_args = {
        u32(test_config.num_tiles),  // per_core_block_cnt
        1                            // per_core_block_cnt
    };

    // Input
    std::vector<u32> packed_input = sfpu_util::generate_packed_sfpu_input(
        byte_size / bfloat16::SIZEOF, test_config.sfpu_op, std::chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    auto input = unpack_vector<bfloat16, u32>(packed_input);
    std::vector<bfloat16> golden(input.size());
    std::transform(input.begin(), input.end(), golden.begin(), [&](const bfloat16& val) {
        return sfpu_util::sfpu_function(test_config.sfpu_op, val);
    });
    std::vector<u32> packed_golden = pack_vector<u32, bfloat16>(golden);

    // Same runtime args for every core
    vector<u32> reader_rt_args = {
        (u32)test_config.input_dram_byte_address,
        (u32)input_dram_noc_xy.x,
        (u32)input_dram_noc_xy.y,
        (u32)test_config.num_tiles,
    };

    vector<u32> writer_rt_args = {
        (u32)test_config.output_dram_byte_address,
        (u32)output_dram_noc_xy.x,
        (u32)output_dram_noc_xy.y,
        (u32)test_config.num_tiles,
    };

    for (const CoreRange& core_range : test_config.cores.ranges()) {
        auto l1_input_cb = tt_metal::CreateCircularBuffers(
            program,
            device,
            0,
            core_range,
            test_config.num_tiles,
            byte_size,
            test_config.l1_input_byte_address,
            test_config.l1_input_data_format);

        auto l1_output_cb = tt_metal::CreateCircularBuffers(
            program,
            device,
            16,
            core_range,
            test_config.num_tiles,
            byte_size,
            test_config.l1_output_byte_address,
            test_config.l1_output_data_format);

        auto reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary.cpp",
            test_config.cores,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);

        auto writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            test_config.cores,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        // Enqueue apis only supported on gs so far
        if (device->arch() == tt::ARCH::GRAYSKULL) {
            writer_kernel->add_define("DEVICE_DISPATCH_MODE", "1");
        }

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto sfpu_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            test_config.cores,
            compute_kernel_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode);

        sfpu_kernel->add_define("SFPU_OP_AND_PACK", sfpu_util::sfpu_op_to_op_name.at(test_config.sfpu_op));
        bool is_relu = (test_config.sfpu_op == "relu");
        sfpu_kernel->add_define("INIT_RELU", is_relu ? "pack_relu_config(1);" : "");
        sfpu_kernel->add_define("DEINIT_RELU", is_relu ? "pack_relu_config(0);" : "");

        int chip_id = 0;
        CoresInCoreRangeGenerator cores_in_core_range(
            core_range, device->cluster()->get_soc_desc(chip_id).worker_grid_size);

        bool terminate;

        // TODO(agrebenisan): Clean this up to only use the first path once Enqueue apis supported on WH
        if (device->arch() == tt::ARCH::GRAYSKULL) {
            do {
                auto [core_coord, terminate_] = cores_in_core_range();

                terminate = terminate_;

                SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
                SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
            } while (not terminate);
        } else {
            do {
                auto [core_coord, terminate_] = cores_in_core_range();

                terminate = terminate_;

                tt_metal::WriteRuntimeArgsToDevice(
                    device, writer_kernel, core_coord, writer_rt_args);

                tt_metal::WriteRuntimeArgsToDevice(
                    device, reader_kernel, core_coord, reader_rt_args);

            } while (not terminate);
        }
    }

    tt_metal::CompileProgram(device, program);

    std::vector<u32> dest_buffer_data;
    if (device->arch() == tt::ARCH::GRAYSKULL) {
        CommandQueue cq(device);
        EnqueueWriteBuffer(cq, input_dram_buffer, packed_input, false);

        EnqueueProgram(cq, program, false);

        EnqueueReadBuffer(cq, output_dram_buffer, dest_buffer_data, true);
    } else {
        tt_metal::WriteToBuffer(input_dram_buffer, packed_input);
        tt_metal::ConfigureDeviceWithProgram(device, program);
        tt_metal::LaunchKernels(device, program);
        tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    }

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

}  // namespace unit_tests::sfpu_compute

#define WARN_AND_EXIT(msg)      \
    WARN([]() {                 \
        log_assert(false, msg); \
        return false;           \
    });                         \
    return;

TEST_SUITE("SfpuCompute" * doctest::description("Eltwise unary SFPU tests") * doctest::timeout(90)) {
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "SingleCore") {
        CoreRange core_range = {{0, 0}};
        CoreRangeSet core_range_set({core_range});
        unit_tests::sfpu_compute::SfpuConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .output_dram_channel = 0,
            .output_dram_byte_address = 0,
            .input_dram_channel = 0,
            .input_dram_byte_address = 16 * 32 * 32,
            .l1_input_byte_address = UNRESERVED_BASE,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_byte_address = UNRESERVED_BASE + 16 * 32 * 32,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .cores = core_range_set};

        SUBCASE("SingleTile") {
            test_config.num_tiles = 1;
            SUBCASE("relu") {
                test_config.sfpu_op = "relu";
                if (arch_ == tt::ARCH::WORMHOLE_B0) {
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                } else {
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
            }
            SUBCASE("exponential") {
                test_config.sfpu_op = "exponential";
                if (arch_ == tt::ARCH::WORMHOLE_B0) {
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                } else {
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
            }
            SUBCASE("reciprocal") {
                test_config.sfpu_op = "reciprocal";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
            SUBCASE("gelu") {
                test_config.sfpu_op = "gelu";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
            SUBCASE("sqrt") {
                test_config.sfpu_op = "sqrt";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
            SUBCASE("sigmoid") {
                test_config.sfpu_op = "sigmoid";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
            SUBCASE("log") {
                test_config.sfpu_op = "log";
                if (arch_ == tt::ARCH::WORMHOLE_B0) {
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                } else {
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
            }
            SUBCASE("tanh") {
                test_config.sfpu_op = "tanh";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
        }

        SUBCASE("MultiTile") {
            test_config.num_tiles = 4;
            SUBCASE("relu") {
                test_config.sfpu_op = "relu";
                if (arch_ == tt::ARCH::WORMHOLE_B0) {
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                } else {
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
            }
            SUBCASE("exponential") {
                test_config.sfpu_op = "exponential";
                if (arch_ == tt::ARCH::WORMHOLE_B0) {
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                } else {
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
            }
            SUBCASE("reciprocal") {
                test_config.sfpu_op = "reciprocal";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
            SUBCASE("gelu") {
                test_config.sfpu_op = "gelu";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
            SUBCASE("sqrt") {
                test_config.sfpu_op = "sqrt";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
            SUBCASE("sigmoid") {
                test_config.sfpu_op = "sigmoid";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
            SUBCASE("log") {
                test_config.sfpu_op = "log";
                if (arch_ == tt::ARCH::WORMHOLE_B0) {
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                } else {
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
            }
            SUBCASE("tanh") {
                test_config.sfpu_op = "tanh";
                REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
            }
        }
    }

    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "Multi core") {
        CoreRange core_range = {{0, 0}};
        CoreRangeSet core_range_set({core_range});
        unit_tests::sfpu_compute::SfpuConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .output_dram_channel = 0,
            .output_dram_byte_address = 0,
            .input_dram_channel = 0,
            .input_dram_byte_address = 16 * 32 * 32,
            .l1_input_byte_address = UNRESERVED_BASE,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_byte_address = UNRESERVED_BASE + 16 * 32 * 32,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .cores = core_range_set};

        SUBCASE("Eltwise unary sfpu on two contiguous cores") {
            auto arch = this->arch_;

            if (arch != tt::ARCH::GRAYSKULL) {
                WARN_AND_EXIT("Eltwise unary sfpu on two contiguous cores only tested on grayskull")
            }

            CoreRange core_range = {.start = {0, 0}, .end = {1, 0}};

            CoreRangeSet core_set({core_range});
            test_config.cores = core_set;

            SUBCASE("SingleTile") {
                test_config.num_tiles = 1;

                SUBCASE("relu") {
                    test_config.sfpu_op = "relu";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("exponential") {
                    test_config.sfpu_op = "exponential";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("reciprocal") {
                    test_config.sfpu_op = "reciprocal";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("gelu") {
                    test_config.sfpu_op = "gelu";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("sqrt") {
                    test_config.sfpu_op = "sqrt";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("sigmoid") {
                    test_config.sfpu_op = "sigmoid";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("log") {
                    test_config.sfpu_op = "log";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("tanh") {
                    test_config.sfpu_op = "tanh";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
            }

            SUBCASE("MultiTile") {
                // TODO(agrebenisan): Figure out cause of stack smashing

                // WARN([]() {
                //     log_assert(false, "SFPU on two contiguous cores results in stack smashing");
                //     return false;
                // });
                // return;

                test_config.num_tiles = 4;

                SUBCASE("relu") {
                    test_config.sfpu_op = "relu";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("exponential") {
                    test_config.sfpu_op = "exponential";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("reciprocal") {
                    test_config.sfpu_op = "reciprocal";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("gelu") {
                    test_config.sfpu_op = "gelu";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("sqrt") {
                    test_config.sfpu_op = "sqrt";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("sigmoid") {
                    test_config.sfpu_op = "sigmoid";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("log") {
                    test_config.sfpu_op = "log";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("tanh") {
                    test_config.sfpu_op = "tanh";
                    REQUIRE(run_sfpu_all_same_buffer(device_, test_config));
                }
            }
        }
        SUBCASE("Eltwise unary sfpu on all worker cores") {
            // TODO(agrebenisan): Figure out cause of stack smashing
            WARN_AND_EXIT("SFPU on all worker cores results in stack smashing")

            auto arch = this->arch_;

            if (arch != tt::ARCH::GRAYSKULL) {
                WARN_AND_EXIT("Eltwise unary sfpu on all worker cores only tested on grayskull")
            }

            int chip_id = 0;
            const auto& sdesc = this->device_->cluster()->get_soc_desc(chip_id);
            CoreRange core_range = {
                .start = {0, 0}, .end = {sdesc.worker_grid_size.x - 1, sdesc.worker_grid_size.y - 2}};

            CoreRangeSet core_set({core_range});
            test_config.cores = core_set;

            SUBCASE("SingleTile") {
                test_config.num_tiles = 1;
                SUBCASE("relu") {
                    test_config.sfpu_op = "relu";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("exponential") {
                    test_config.sfpu_op = "exponential";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("reciprocal") {
                    test_config.sfpu_op = "reciprocal";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("gelu") {
                    test_config.sfpu_op = "gelu";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("sqrt") {
                    test_config.sfpu_op = "sqrt";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("sigmoid") {
                    test_config.sfpu_op = "sigmoid";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("log") {
                    test_config.sfpu_op = "log";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("tanh") {
                    test_config.sfpu_op = "tanh";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
            }
            SUBCASE("MultiTile") {
                test_config.num_tiles = 4;
                SUBCASE("relu") {
                    test_config.sfpu_op = "relu";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("exponential") {
                    test_config.sfpu_op = "exponential";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("reciprocal") {
                    test_config.sfpu_op = "reciprocal";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("gelu") {
                    test_config.sfpu_op = "gelu";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("sqrt") {
                    test_config.sfpu_op = "sqrt";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("sigmoid") {
                    test_config.sfpu_op = "sigmoid";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("log") {
                    test_config.sfpu_op = "log";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
                SUBCASE("tanh") {
                    test_config.sfpu_op = "tanh";
                    WARN(run_sfpu_all_same_buffer(device_, test_config));
                }
            }
        }
    }
}
