// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/hw/inc/internal/dataflow_buffer_interface.h"
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

// Test parameter struct for parameterized DFB tests
struct DFBTestParams {
    std::string name;
    uint32_t num_producers;
    uint32_t num_consumers;
    ::experimental::AccessPattern pap;
    ::experimental::AccessPattern cap;
    bool enable_implicit_sync;
};

// Helper to generate test name
std::string DFBTestName(const testing::TestParamInfo<DFBTestParams>& info) {
    std::string sync_suffix = info.param.enable_implicit_sync ? "_ImplicitSync" : "_ExplicitSync";
    return info.param.name + sync_suffix;
}

class DataflowBufferTest : public MeshDeviceFixture, public testing::WithParamInterface<DFBTestParams> {};

void run_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, experimental::dfb::DataflowBufferConfig& dfb_config) {
    Program program = CreateProgram();

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    uint32_t buffer_size = dfb_config.entry_size * dfb_config.num_entries;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = buffer_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    log_info(tt::LogTest, "In Buffer: [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
    log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

    CoreCoord logical_core = CoreCoord(0, 0);

    uint32_t num_entries_per_producer = dfb_config.num_entries / dfb_config.num_producers;
    std::vector<uint32_t> producer_cta = {(uint32_t)in_buffer->address(), num_entries_per_producer};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);
    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_processors_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});

    uint32_t num_entries_per_consumer = dfb_config.cap == ::experimental::AccessPattern::STRIDED
                                            ? dfb_config.num_entries / dfb_config.num_consumers
                                            : dfb_config.num_entries;
    std::vector<uint32_t> consumer_cta = {(uint32_t)out_buffer->address(), num_entries_per_consumer};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);
    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_processors_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});

    auto producer_quasar = std::dynamic_pointer_cast<experimental::quasar::QuasarDataMovementKernel>(
        program.impl().get_kernel(producer_kernel));
    auto consumer_quasar = std::dynamic_pointer_cast<experimental::quasar::QuasarDataMovementKernel>(
        program.impl().get_kernel(consumer_kernel));
    TT_FATAL(producer_quasar && consumer_quasar, "DFB test kernels must be QuasarDataMovementKernel");
    const auto& producer_dms = producer_quasar->get_dm_processors();
    const auto& consumer_dms = consumer_quasar->get_dm_processors();

    dfb_config.producer_risc_mask = 0;
    for (DataMovementProcessor dm : producer_dms) {
        dfb_config.producer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm));
    }
    dfb_config.consumer_risc_mask = 0;
    for (DataMovementProcessor dm : consumer_dms) {
        dfb_config.consumer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm));
    }

    log_info(
        tt::LogTest,
        "Producer risc mask: 0x{:x}. Consumer risc mask: 0x{:x}",
        dfb_config.producer_risc_mask,
        dfb_config.consumer_risc_mask);

    /*auto logical_dfb_id = */ experimental::dfb::CreateDataflowBuffer(program, logical_core, dfb_config);

    SetRuntimeArgs(program, producer_kernel, logical_core, {(uint32_t)dfb_config.producer_risc_mask});
    SetRuntimeArgs(program, consumer_kernel, logical_core, {(uint32_t)dfb_config.consumer_risc_mask});

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, buffer_size / sizeof(uint32_t));
    distributed::WriteShard(mesh_device->mesh_command_queue(), in_buffer, input, zero_coord, true);

    // Execute using slow dispatch (DFBs not yet supported in MeshWorkload path)
    IDevice* device = mesh_device->get_devices()[0];
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    std::vector<uint32_t> output;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffer, zero_coord, true);

    if (input != output) {
        log_info(tt::LogTest, "Printing input");
        for (auto i : input) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
        log_info(tt::LogTest, "Printing output");
        for (auto i : output) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    EXPECT_EQ(input, output);
}

TEST_P(DataflowBufferTest, RunDFB) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }

    const auto& params = GetParam();
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = params.num_producers,
        .pap = params.pap,
        .num_consumers = params.num_consumers,
        .cap = params.cap,
        .enable_implicit_sync = params.enable_implicit_sync};

    run_dfb_program(this->devices_.at(0), config);
}

// Generate test parameters for all configs with both implicit sync modes
std::vector<DFBTestParams> GenerateDFBTestParams() {
    std::vector<DFBTestParams> params;

    // Base configurations (name, num_producers, num_consumers, pap, cap)
    std::vector<
        std::tuple<std::string, uint32_t, uint32_t, ::experimental::AccessPattern, ::experimental::AccessPattern>>
        base_configs = {
            {"1Sx1S", 1, 1, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::STRIDED},
            {"1Sx4S", 1, 4, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::STRIDED},
            {"4Sx1S", 4, 1, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::STRIDED},
            {"4Sx4S", 4, 4, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::STRIDED},
            {"2Sx4S", 2, 4, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::STRIDED},
            {"4Sx2S", 4, 2, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::STRIDED},
            {"1Sx4B", 1, 4, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::BLOCKED},
            {"4Sx1B", 4, 1, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::BLOCKED},
            {"4Sx4B", 4, 4, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::BLOCKED},
            {"4Sx2B", 4, 2, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::BLOCKED},
            {"2Sx4B", 2, 4, ::experimental::AccessPattern::STRIDED, ::experimental::AccessPattern::BLOCKED},
        };

    // Generate params for both implicit sync modes
    for (bool enable_implicit_sync : {false, true}) {
        for (const auto& [name, num_producers, num_consumers, pap, cap] : base_configs) {
            params.push_back(DFBTestParams{
                .name = name,
                .num_producers = num_producers,
                .num_consumers = num_consumers,
                .pap = pap,
                .cap = cap,
                .enable_implicit_sync = enable_implicit_sync});
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(TensixTest1xDFB, DataflowBufferTest, testing::ValuesIn(GenerateDFBTestParams()), DFBTestName);

}  // end namespace tt::tt_metal
