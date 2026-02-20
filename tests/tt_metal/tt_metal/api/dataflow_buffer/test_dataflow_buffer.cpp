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
#include "tt_metal/hw/inc/internal/dataflow_buffer_interface.h"
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

enum class DFBPorCType : uint8_t { DM, TENSIX };

void execute_program_and_verify(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    Program& program,
    const std::shared_ptr<distributed::MeshBuffer>& in_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& out_buffer,
    distributed::MeshCoordinate& zero_coord,
    uint32_t buffer_size,
    bool verify_output = true) {
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, buffer_size / sizeof(uint32_t));
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

    std::vector<uint32_t> intermediate(buffer_size / sizeof(uint32_t));
    detail::ReadFromDeviceL1(device, CoreCoord(0, 0), 729088, buffer_size, intermediate);

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
            std::cout << std::endl;
            if (intermediate != input) {
                log_info(tt::LogTest, "Printing intermediate");
                for (auto i : intermediate) {
                    std::cout << i << " ";
                }
                std::cout << std::endl;
            }
        }
        EXPECT_EQ(input, output);
    }
}

void run_single_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::dfb::DataflowBufferConfig& dfb_config,
    DFBPorCType producer_type,
    DFBPorCType consumer_type) {
    TT_FATAL(
        !(producer_type == DFBPorCType::TENSIX && consumer_type == DFBPorCType::TENSIX),
        "Both producer and consumer cannot be Tensix. At least one must be a DM kernel for NOC transfers.");

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

    dfb_config.producer_risc_mask = 0;
    dfb_config.consumer_risc_mask = 0;

    uint32_t num_entries_per_producer = dfb_config.num_entries / dfb_config.num_producers;
    std::vector<uint32_t> producer_cta = {(uint32_t)in_buffer->address(), num_entries_per_producer};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

    KernelHandle producer_kernel;
    if (producer_type == DFBPorCType::DM) {
        producer_kernel = experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
            logical_core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_processors_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});
        auto producer_quasar = std::dynamic_pointer_cast<experimental::quasar::QuasarDataMovementKernel>(
            program.impl().get_kernel(producer_kernel));
        const auto& producer_dms = producer_quasar->get_dm_processors();
        for (DataMovementProcessor dm : producer_dms) {
            dfb_config.producer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm));
        }
    } else {
        producer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer.cpp",
            logical_core,
            ComputeConfig{.compile_args = producer_cta});
        dfb_config.producer_risc_mask = 0x100;
    }

    uint32_t num_entries_per_consumer = dfb_config.cap == ::experimental::AccessPattern::STRIDED
                                            ? dfb_config.num_entries / dfb_config.num_consumers
                                            : dfb_config.num_entries;
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        num_entries_per_consumer,
        (uint32_t)dfb_config.cap == ::experimental::AccessPattern::BLOCKED};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);

    KernelHandle consumer_kernel;
    if (consumer_type == DFBPorCType::DM) {
        consumer_kernel = experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
            logical_core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_processors_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});
        auto consumer_quasar = std::dynamic_pointer_cast<experimental::quasar::QuasarDataMovementKernel>(
            program.impl().get_kernel(consumer_kernel));
        const auto& consumer_dms = consumer_quasar->get_dm_processors();
        for (DataMovementProcessor dm : consumer_dms) {
            dfb_config.consumer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm));
        }
    } else {
        consumer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp",
            logical_core,
            ComputeConfig{.compile_args = consumer_cta});
        dfb_config.consumer_risc_mask = 0x100;
    }

    log_info(
        tt::LogTest,
        "Producer risc mask: 0x{:x}. Consumer risc mask: 0x{:x}",
        dfb_config.producer_risc_mask,
        dfb_config.consumer_risc_mask);

    auto logical_dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, dfb_config);

    SetRuntimeArgs(program, producer_kernel, logical_core, {(uint32_t)dfb_config.producer_risc_mask});
    SetRuntimeArgs(program, consumer_kernel, logical_core, {(uint32_t)dfb_config.consumer_risc_mask, (uint32_t)logical_dfb_id});

    execute_program_and_verify(
        mesh_device,
        program,
        in_buffer,
        out_buffer,
        zero_coord,
        buffer_size,
        (producer_type == DFBPorCType::DM and consumer_type == DFBPorCType::DM));
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

    dm2tensix_config.producer_risc_mask = 0;
    dm2tensix_config.consumer_risc_mask = 0;

    tensix2dm_config.producer_risc_mask = 0;
    tensix2dm_config.consumer_risc_mask = 0;

    uint32_t num_entries_per_producer = dm2tensix_config.num_entries / dm2tensix_config.num_producers;
    std::vector<uint32_t> producer_cta = {(uint32_t)in_buffer->address(), num_entries_per_producer};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_processors_per_cluster = dm2tensix_config.num_producers, .compile_args = producer_cta});
    auto producer_quasar = std::dynamic_pointer_cast<experimental::quasar::QuasarDataMovementKernel>(
        program.impl().get_kernel(producer_kernel));
    const auto& producer_dms = producer_quasar->get_dm_processors();
    for (DataMovementProcessor dm : producer_dms) {
        dm2tensix_config.producer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm));
    }

    uint32_t num_entries_per_unpacker = dm2tensix_config.num_entries / dm2tensix_config.num_consumers;
    uint32_t num_entries_per_packer = tensix2dm_config.num_entries / tensix2dm_config.num_producers;
    TT_FATAL(
        num_entries_per_unpacker == num_entries_per_packer, "Num entries per unpacker and packer must be the same");
    std::vector<uint32_t> compute_cta = {num_entries_per_unpacker};
    auto compute_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6.cpp",
        logical_core,
        ComputeConfig{.compile_args = compute_cta});
    dm2tensix_config.consumer_risc_mask = 0x100;
    tensix2dm_config.producer_risc_mask = 0x100;

    uint32_t num_entries_per_consumer = tensix2dm_config.num_entries / tensix2dm_config.num_consumers;
    std::vector<uint32_t> consumer_cta = {(uint32_t)out_buffer->address(), num_entries_per_consumer, (uint32_t)tensix2dm_config.cap == ::experimental::AccessPattern::BLOCKED};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);
    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_processors_per_cluster = tensix2dm_config.num_consumers, .compile_args = consumer_cta});
    auto consumer_quasar = std::dynamic_pointer_cast<experimental::quasar::QuasarDataMovementKernel>(
        program.impl().get_kernel(consumer_kernel));
    const auto& consumer_dms = consumer_quasar->get_dm_processors();
    for (DataMovementProcessor dm : consumer_dms) {
        tensix2dm_config.consumer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm));
    }

    auto input_dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, dm2tensix_config);
    auto output_dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, tensix2dm_config);

    SetRuntimeArgs(program, producer_kernel, logical_core, {(uint32_t)dm2tensix_config.producer_risc_mask});
    SetRuntimeArgs(
        program,
        compute_kernel,
        logical_core,
        {(uint32_t)input_dfb_id, (uint32_t)output_dfb_id});
    SetRuntimeArgs(program, consumer_kernel, logical_core, {(uint32_t)tensix2dm_config.consumer_risc_mask, (uint32_t)output_dfb_id});

    execute_program_and_verify(mesh_device, program, in_buffer, out_buffer, zero_coord, buffer_size);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTensixTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX);
}

TEST_F(MeshDeviceFixture, TensixDMTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTensixDMTest2xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig dm2tensix_config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    experimental::dfb::DataflowBufferConfig tensix2dm_config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
}

TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig dm2tensix_config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    experimental::dfb::DataflowBufferConfig tensix2dm_config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
}

TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig dm2tensix_config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    experimental::dfb::DataflowBufferConfig tensix2dm_config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB1Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 8,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB1Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx1B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx2B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB2Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
}

}  // end namespace tt::tt_metal
