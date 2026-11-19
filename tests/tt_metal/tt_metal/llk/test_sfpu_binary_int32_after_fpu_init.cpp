// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include "hostdevcommon/kernel_structs.h"
#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

/*
 * Int32 binary SFPU ops (gcd, lcm) run in a kernel that also initialises an FPU binary op after the
 * operand copies, so the FPU binary init (ADDR_MOD_3 = Dest increment) rather than a datacopy init
 * is the last ADDR_MOD programmer before the SFPU op. The SFPU op must address Dest through its own
 * no-increment slot, so the result must still match the elementwise golden exactly.
 */
namespace unit_tests::compute::sfpu::binary_int32_after_fpu_init {

struct OpSpec {
    std::string init;
    std::string call;
    int32_t min_value;
    int32_t max_value;
    bool allow_zero;
    int32_t (*golden)(int32_t, int32_t);
};

const std::map<std::string, OpSpec>& op_specs() {
    static const std::map<std::string, OpSpec> specs = {
        {"gcd",
         {.init = "gcd_tile_init();",
          .call = "gcd_tile(0, 1, 2);",
          .min_value = -(1 << 20),
          .max_value = (1 << 20),
          .allow_zero = true,
          .golden = [](int32_t a, int32_t b) { return static_cast<int32_t>(std::gcd(a, b)); }}},
        // lcm_tile is documented for |value| <= 2^15 - 1.
        {"lcm",
         {.init = "lcm_tile_init();",
          .call = "lcm_tile(0, 1, 2);",
          .min_value = -32767,
          .max_value = 32767,
          .allow_zero = false,
          .golden = [](int32_t a, int32_t b) { return static_cast<int32_t>(std::lcm(a, b)); }}},
    };
    return specs;
}

std::vector<uint32_t> random_int32_words(size_t numel, const OpSpec& spec, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(spec.min_value, spec.max_value);
    std::vector<uint32_t> words(numel);
    for (auto& w : words) {
        int32_t v = dist(rng);
        while (!spec.allow_zero && v == 0) {
            v = dist(rng);
        }
        w = static_cast<uint32_t>(v);
    }
    return words;
}

bool run(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const std::string& op_name, uint32_t num_tiles) {
    const OpSpec& spec = op_specs().at(op_name);

    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    constexpr CoreCoord core = {0, 0};
    const tt::DataFormat data_format = tt::DataFormat::Int32;
    const uint32_t single_tile_size = tt::tile_size(data_format);
    const uint32_t dram_buffer_size = single_tile_size * num_tiles;
    constexpr uint32_t cb_num_pages = 2;  // double-buffered CBs

    distributed::ReplicatedBufferConfig global_config{.size = dram_buffer_size};
    distributed::DeviceLocalBufferConfig dram_config{.page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto src0_dram_buffer = distributed::MeshBuffer::create(global_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(global_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(global_config, dram_config, mesh_device.get());

    for (auto cb_index : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_16}) {
        CircularBufferConfig cb_config =
            CircularBufferConfig(cb_num_pages * single_tile_size, {{cb_index, data_format}})
                .set_page_size(cb_index, single_tile_size);
        CreateCircularBuffer(program_, core, cb_config);
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    KernelHandle writer_kernel_id = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Int32 operands go to DST through unpack-to-dest with a 32-bit DST.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
    const std::map<std::string, std::string> defines = {
        {"SFPU_OP_INIT_0", spec.init},
        {"SFPU_OP_CHAIN_0", spec.call},
    };
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/sfpu_binary_int32_after_fpu_init.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = {num_tiles},
            .defines = defines});

    const size_t numel = dram_buffer_size / sizeof(uint32_t);
    const std::vector<uint32_t> src0 = random_int32_words(numel, spec, /*seed=*/0x5f3759df);
    const std::vector<uint32_t> src1 = random_int32_words(numel, spec, /*seed=*/0x2545f491);
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1, /*blocking=*/true);

    SetRuntimeArgs(
        program_, reader_kernel_id, core, {src0_dram_buffer->address(), 0, src1_dram_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(program_, writer_kernel_id, core, {dst_dram_buffer->address(), 0, num_tiles});

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    std::vector<uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, dst_dram_buffer, /*blocking=*/true);
    EXPECT_EQ(result.size(), numel);

    constexpr size_t max_logged_mismatches = 8;
    size_t mismatches = 0;
    for (size_t i = 0; i < numel && i < result.size(); ++i) {
        const int32_t a = static_cast<int32_t>(src0[i]);
        const int32_t b = static_cast<int32_t>(src1[i]);
        const int32_t expected = spec.golden(a, b);
        const int32_t actual = static_cast<int32_t>(result[i]);
        if (actual != expected) {
            if (mismatches < max_logged_mismatches) {
                log_error(
                    tt::LogTest,
                    "{}: element {} (a={}, b={}): expected {}, got {}",
                    op_name,
                    i,
                    a,
                    b,
                    expected,
                    actual);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        log_error(tt::LogTest, "{}: {} of {} elements mismatch", op_name, mismatches, numel);
    }
    return mismatches == 0;
}

}  // namespace unit_tests::compute::sfpu::binary_int32_after_fpu_init

class SfpuBinaryInt32AfterFpuInitFixture : public LLKMeshDeviceFixture,
                                           public testing::WithParamInterface<std::string> {};

TEST_P(SfpuBinaryInt32AfterFpuInitFixture, TensixSfpuBinaryInt32AfterFpuInit) {
    if (this->arch_ == ARCH::QUASAR) {
        GTEST_SKIP() << "Int32 copy_tile + FPU binary init path not validated on Quasar";
    }
    const std::string& op_name = GetParam();
    EXPECT_TRUE(
        unit_tests::compute::sfpu::binary_int32_after_fpu_init::run(this->devices_.at(0), op_name, 1 /*num_tiles*/));
}

INSTANTIATE_TEST_SUITE_P(
    SfpuBinaryInt32AfterFpuInit,
    SfpuBinaryInt32AfterFpuInitFixture,
    ::testing::Values("gcd", "lcm"),
    [](const testing::TestParamInfo<std::string>& info) { return info.param; });

}  // namespace tt::tt_metal
