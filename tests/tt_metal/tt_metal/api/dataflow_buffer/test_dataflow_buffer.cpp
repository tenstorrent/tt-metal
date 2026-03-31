// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <memory>
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
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

enum class DFBPorCType : uint8_t { DM, TENSIX };

class DFBImplicitSyncParamFixture : public MeshDeviceFixture, public ::testing::WithParamInterface<bool> {};

static std::string ImplicitSyncParamName(const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "ImplicitSyncTrue" : "ImplicitSyncFalse";
}

void execute_program_and_verify(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    Program& program,
    const std::shared_ptr<distributed::MeshBuffer>& in_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& out_buffer,
    distributed::MeshCoordinate& zero_coord,
    std::vector<uint32_t>& input,
    bool verify_output = true) {
    distributed::WriteShard(mesh_device->mesh_command_queue(), in_buffer, input, zero_coord, true);

    // TODO #38042: Need to wait for data to be written, the barrier needs to be uplifted for Quasar
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::vector<uint32_t> rdback_dram;
    distributed::ReadShard(mesh_device->mesh_command_queue(), rdback_dram, in_buffer, zero_coord, true);

    tt_driver_atomics::mfence();

    EXPECT_EQ(rdback_dram, input);

    // Execute using slow dispatch (DFBs not yet supported in MeshWorkload path)
    IDevice* device = mesh_device->get_devices()[0];
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    std::vector<uint32_t> output;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffer, zero_coord, true);

    if (verify_output) {
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
        }
        EXPECT_EQ(input, output);
    }
}

// Runs a single DFB program on one or more cores and verifies output == input.
//
// When core_range_set contains N > 1 cores the global DRAM buffers are sized
// N x entries_per_core x entry_size and each core receives a unique
// chunk_offset (= core_idx * entries_per_core) so it accesses a disjoint
// slice of the buffer.  Multi-core use requires DM producer and consumer.
void run_single_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::dfb::DataflowBufferConfig& dfb_config,
    DFBPorCType producer_type,
    DFBPorCType consumer_type,
    const CoreRangeSet& core_range_set = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)))) {

    TT_FATAL(
        !(producer_type == DFBPorCType::TENSIX && consumer_type == DFBPorCType::TENSIX),
        "Both producer and consumer cannot be Tensix. At least one must be a DM kernel for NOC transfers.");
    TT_FATAL(
        core_range_set.num_cores() == 1 ||
            (producer_type == DFBPorCType::DM && consumer_type == DFBPorCType::DM),
        "Multi-core DFB programs only support DM producer and consumer.");

    Program program = CreateProgram();
    auto zero_coord = distributed::MeshCoordinate(0, 0);

    const uint32_t num_cores = core_range_set.num_cores();
    const uint32_t entries_per_core = dfb_config.num_entries;
    const uint32_t entry_size = dfb_config.entry_size;
    // page_size = entry_size makes every entry independently addressable by
    // page_id.  For single-core this is equivalent to page_size = buffer_size
    // because chunk_offset = 0 and page_ids 0..num_entries-1 stay in range.
    const uint32_t total_buffer_size = num_cores * entries_per_core * entry_size;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    log_info(tt::LogTest, "In Buffer:  [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
    log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

    uint32_t num_entries_per_producer = entries_per_core / dfb_config.num_producers;
    const bool is_blocked = (dfb_config.cap == dfb::AccessPattern::BLOCKED);
    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(),
        num_entries_per_producer,
        (uint32_t)dfb_config.enable_implicit_sync,
        (uint32_t)is_blocked};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

    KernelHandle producer_kernel;
    if (producer_type == DFBPorCType::DM) {
        producer_kernel = experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
            core_range_set,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});
    } else {
        producer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer.cpp",
            core_range_set,
            experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});
    }

    // is_blocked is already defined above
    uint32_t num_entries_per_consumer = is_blocked ? entries_per_core : entries_per_core / dfb_config.num_consumers;
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        num_entries_per_consumer,
        (uint32_t)is_blocked,
        (uint32_t)dfb_config.enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);

    KernelHandle consumer_kernel;
    if (consumer_type == DFBPorCType::DM) {
        consumer_kernel = experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
            core_range_set,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});
    } else {
        consumer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp",
            core_range_set,
            experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});
    }

    auto logical_dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, logical_dfb_id, producer_kernel, consumer_kernel);

    auto dfb = program.impl().get_dataflow_buffer(logical_dfb_id);
    const uint32_t producer_mask = dfb->config.producer_risc_mask;
    const uint32_t consumer_mask = dfb->config.consumer_risc_mask;

    // Build a per-core chunk-offset map (used for both runtime args and L1 pre-fill/verify).
    std::map<CoreCoord, uint32_t> core_to_chunk_offset;
    uint32_t core_idx = 0;
    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
            for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                core_to_chunk_offset[CoreCoord(x, y)] = core_idx++ * entries_per_core;
            }
        }
    }

    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
            for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                const CoreCoord core(x, y);
                const uint32_t chunk_offset = core_to_chunk_offset.at(core);
                SetRuntimeArgs(program, producer_kernel, core, {producer_mask, chunk_offset});
                SetRuntimeArgs(
                    program, consumer_kernel, core,
                    {consumer_mask, (uint32_t)logical_dfb_id, chunk_offset});
            }
        }
    }

    // Generate input once; shared by in_buffer write, L1 pre-fill, and verification.
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_buffer_size / sizeof(uint32_t));

    IDevice* device = mesh_device->get_devices()[0];
    const uint32_t words_per_core = entries_per_core * entry_size / sizeof(uint32_t);

    // For Tensix → DM: pre-fill each core's DFB L1 with its input chunk so the
    // Tensix producer kernel can read from L1 while DM consumer drains to DRAM.
    if (producer_type == DFBPorCType::TENSIX) {
        for (const auto& group : dfb->groups) {
            for (const auto& [core, alloc_addr] : group.l1_by_core) {
                const uint32_t co = core_to_chunk_offset.at(core);
                std::vector<uint32_t> slice(
                    input.begin() + co * entry_size / sizeof(uint32_t),
                    input.begin() + co * entry_size / sizeof(uint32_t) + words_per_core);
                detail::WriteToDeviceL1(device, core, alloc_addr, slice);
            }
        }
    }

    // Launch program; verify out_buffer only for DM → DM paths (Tensix consumer
    // does not write to DRAM, so out_buffer verification is skipped there).
    execute_program_and_verify(
        mesh_device, program, in_buffer, out_buffer, zero_coord,
        input,
        /*verify_output=*/(consumer_type == DFBPorCType::DM));

    // For DM → Tensix: verify each core's DFB L1 against the expected input chunk.
    if (consumer_type == DFBPorCType::TENSIX) {
        for (const auto& group : dfb->groups) {
            for (const auto& [core, alloc_addr] : group.l1_by_core) {
                const uint32_t co = core_to_chunk_offset.at(core);
                std::vector<uint32_t> l1_data;
                detail::ReadFromDeviceL1(device, core, alloc_addr, dfb->total_size(), l1_data);
                std::vector<uint32_t> expected(
                    input.begin() + co * entry_size / sizeof(uint32_t),
                    input.begin() + co * entry_size / sizeof(uint32_t) + words_per_core);
                if (expected != l1_data) {
                    std::cout << "expected: ";
                    for (const auto& e : expected) {
                        std::cout << e << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "l1_data: ";
                    for (const auto& l : l1_data) {
                        std::cout << l << " ";
                    }
                    std::cout << std::endl;
                }
                EXPECT_EQ(expected, l1_data)
                    << "DFB L1 mismatch on core (" << core.x << "," << core.y << ")";
            }
        }
    }
}

void run_in_dfb_out_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::dfb::DataflowBufferConfig& dm2tensix_config,
    experimental::dfb::DataflowBufferConfig& tensix2dm_config) {
    TT_FATAL(
        dm2tensix_config.num_entries == tensix2dm_config.num_entries,
        "Num entries must be the same for in and out DFBs");
    TT_FATAL(
        dm2tensix_config.entry_size == tensix2dm_config.entry_size, "Entry size must be the same for in and out DFBs");

    Program program = CreateProgram();

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    uint32_t buffer_size = dm2tensix_config.entry_size * dm2tensix_config.num_entries;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = buffer_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    log_info(tt::LogTest, "In Buffer: [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
    log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

    CoreCoord logical_core = CoreCoord(0, 0);

    uint32_t num_entries_per_producer = dm2tensix_config.num_entries / dm2tensix_config.num_producers;
    std::vector<uint32_t> producer_cta = {(uint32_t)in_buffer->address(), num_entries_per_producer};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = dm2tensix_config.num_producers, .compile_args = producer_cta});

    uint32_t num_entries_per_unpacker = dm2tensix_config.num_entries / dm2tensix_config.num_consumers;
    uint32_t num_entries_per_packer = tensix2dm_config.num_entries / tensix2dm_config.num_producers;
    TT_FATAL(
        num_entries_per_unpacker == num_entries_per_packer, "Num entries per unpacker and packer must be the same");
    std::vector<uint32_t> compute_cta = {num_entries_per_unpacker};
    auto compute_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6.cpp",
        logical_core,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 1, .compile_args = compute_cta});

    uint32_t num_entries_per_consumer = tensix2dm_config.num_entries / tensix2dm_config.num_consumers;
    std::vector<uint32_t> consumer_cta = {(uint32_t)out_buffer->address(), num_entries_per_consumer, (uint32_t)tensix2dm_config.cap == dfb::AccessPattern::BLOCKED};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);
    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = tensix2dm_config.num_consumers, .compile_args = consumer_cta});

    auto input_dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, dm2tensix_config);
    auto output_dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, tensix2dm_config);

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, input_dfb_id, producer_kernel, compute_kernel);
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, output_dfb_id, compute_kernel, consumer_kernel);

    auto input_dfb = program.impl().get_dataflow_buffer(input_dfb_id);
    auto output_dfb = program.impl().get_dataflow_buffer(output_dfb_id);

    SetRuntimeArgs(program, producer_kernel, logical_core, {(uint32_t)input_dfb->config.producer_risc_mask, 0u});
    SetRuntimeArgs(
        program,
        compute_kernel,
        logical_core,
        {(uint32_t)input_dfb_id, (uint32_t)output_dfb_id});
    SetRuntimeArgs(program, consumer_kernel, logical_core, {(uint32_t)output_dfb->config.consumer_risc_mask, (uint32_t)output_dfb_id, 0u});

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, buffer_size / sizeof(uint32_t));
    execute_program_and_verify(mesh_device, program, in_buffer, out_buffer, zero_coord, input);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

// TEST_F(MeshDeviceFixture, DMTensixDMTest2xDFB1Sx1S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//             .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

// TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 2,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 2,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

// TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 4,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 4,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB1Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB1Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB1Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB4Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB4Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB4Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB4Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB4Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB4Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

// Blocked

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB1Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB1Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB1Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB4Sx1B) { // mismatching
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB4Sx1B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB4Sx1B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB4Sx4B) { // mismatching
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB4Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB4Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB4Sx2B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB4Sx2B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB4Sx2B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB2Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB2Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB2Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_2Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_1Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

INSTANTIATE_TEST_SUITE_P(
    ImplicitSync,
    DFBImplicitSyncParamFixture,
    ::testing::Bool(),
    ImplicitSyncParamName);

}  // end namespace tt::tt_metal
