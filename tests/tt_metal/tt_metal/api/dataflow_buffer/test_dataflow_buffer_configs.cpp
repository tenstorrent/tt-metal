// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
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
#include <tt-metalium/tensor_accessor_args.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/hw/inc/internal/dataflow_buffer_interface.h"
#include "tt_metal/experimental/dataflow_buffer/dataflow_buffer.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

void run_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const experimental::dfb::DataflowBufferConfig& dfb_config) {
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
    /*auto logical_dfb_id = */ experimental::dfb::CreateDataflowBuffer(program, logical_core, dfb_config);

    std::vector<uint32_t> producer_cta = {(uint32_t)in_buffer->address()};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);
    /*auto producer_kernel = */ CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = producer_cta});

    std::vector<uint32_t> consumer_cta = {(uint32_t)out_buffer->address()};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);
    /*auto consumer_kernel = */ CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0, .compile_args = consumer_cta});

    // SetRuntimeArgs(program, producer_kernel, logical_core, {(uint32_t)in_buffer->address()});
    // SetRuntimeArgs(program, consumer_kernel, logical_core, {(uint32_t)out_buffer->address()});

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, buffer_size / sizeof(uint32_t));
    distributed::WriteShard(mesh_device->mesh_command_queue(), in_buffer, input, zero_coord, true);

    // Execute using slow dispatch (DFBs not yet supported in MeshWorkload path)
    IDevice* device = mesh_device->get_devices()[0];
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    std::vector<uint32_t> output;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffer, zero_coord, true);

    EXPECT_EQ(input, output);
}

// move this into test_dataflow_buffer_e2e.cpp
TEST_F(MeshDeviceFixture, TensixTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    run_dfb_program(this->devices_.at(0), config);
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB1Sx4SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x1E,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    const auto& dfbs_on_core = program.impl().dataflow_buffers_on_core(logical_core);

    for (const auto& dfb : dfbs_on_core) {
        EXPECT_EQ(dfb->risc_configs.size(), 5);
        std::vector<::experimental::PackedTileCounter> producer_tcs;
        std::vector<::experimental::PackedTileCounter> consumer_tcs;
        for (const auto& risc_config : dfb->risc_configs) {
            if (risc_config.is_producer) {
                EXPECT_EQ(risc_config.config.num_tcs_to_rr, 4);
                EXPECT_EQ(risc_config.config.packed_tile_counter.size(), 4);
                for (const auto& tc : risc_config.config.packed_tile_counter) {
                    producer_tcs.push_back(tc);
                }
            } else {
                EXPECT_EQ(risc_config.config.num_tcs_to_rr, 1);
                EXPECT_EQ(risc_config.config.packed_tile_counter.size(), 1);
                for (const auto& tc : risc_config.config.packed_tile_counter) {
                    consumer_tcs.push_back(tc);
                }
            }
        }
        for (auto ptc : producer_tcs) {
            log_info(
                tt::LogTest,
                "Producer TC: (0x{:x}, 0x{:x})",
                (uint32_t)::experimental::get_tensix_id(ptc),
                (uint32_t)::experimental::get_counter_id(ptc));
        }
        for (auto ctc : consumer_tcs) {
            log_info(
                tt::LogTest,
                "Consumer TC: (0x{:x}, 0x{:x})",
                (uint32_t)::experimental::get_tensix_id(ctc),
                (uint32_t)::experimental::get_counter_id(ctc));
        }
    }
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB4Sx1SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1E,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x1,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    const auto& dfbs_on_core = program.impl().dataflow_buffers_on_core(logical_core);

    for (const auto& dfb : dfbs_on_core) {
        EXPECT_EQ(dfb->risc_configs.size(), 5);
        std::vector<::experimental::PackedTileCounter> producer_tcs;
        std::vector<::experimental::PackedTileCounter> consumer_tcs;
        for (const auto& risc_config : dfb->risc_configs) {
            if (risc_config.is_producer) {
                // EXPECT_EQ(risc_config.config.num_tcs_to_rr, 1);
                // EXPECT_EQ(risc_config.config.packed_tile_counter.size(), 1);
                for (const auto& tc : risc_config.config.packed_tile_counter) {
                    producer_tcs.push_back(tc);
                }
            } else {
                // EXPECT_EQ(risc_config.config.num_tcs_to_rr, 4);
                // EXPECT_EQ(risc_config.config.packed_tile_counter.size(), 4);
                for (const auto& tc : risc_config.config.packed_tile_counter) {
                    consumer_tcs.push_back(tc);
                }
            }
        }
        // for (auto ptc : producer_tcs) {
        //     log_info(tt::LogTest, "Producer TC: (0x{:x}, 0x{:x})", (uint32_t)::experimental::get_tensix_id(ptc),
        //     (uint32_t)::experimental::get_counter_id(ptc));
        // }
        // for (auto ctc : consumer_tcs) {
        //     log_info(tt::LogTest, "Consumer TC: (0x{:x}, 0x{:x})", (uint32_t)::experimental::get_tensix_id(ctc),
        //     (uint32_t)::experimental::get_counter_id(ctc));
        // }
    }
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB4Sx4SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0xF0,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    const auto& dfbs_on_core = program.impl().dataflow_buffers_on_core(logical_core);

    for (const auto& dfb : dfbs_on_core) {
        // EXPECT_EQ(dfb->risc_configs.size(), 5);
        std::vector<::experimental::PackedTileCounter> producer_tcs;
        std::vector<::experimental::PackedTileCounter> consumer_tcs;
        for (const auto& risc_config : dfb->risc_configs) {
            if (risc_config.is_producer) {
                // EXPECT_EQ(risc_config.config.num_tcs_to_rr, 1);
                // EXPECT_EQ(risc_config.config.packed_tile_counter.size(), 1);
                for (const auto& tc : risc_config.config.packed_tile_counter) {
                    producer_tcs.push_back(tc);
                }
            } else {
                // EXPECT_EQ(risc_config.config.num_tcs_to_rr, 4);
                // EXPECT_EQ(risc_config.config.packed_tile_counter.size(), 4);
                for (const auto& tc : risc_config.config.packed_tile_counter) {
                    consumer_tcs.push_back(tc);
                }
            }
        }
        // for (auto ptc : producer_tcs) {
        //     log_info(tt::LogTest, "Producer TC: (0x{:x}, 0x{:x})", (uint32_t)::experimental::get_tensix_id(ptc),
        //     (uint32_t)::experimental::get_counter_id(ptc));
        // }
        // for (auto ctc : consumer_tcs) {
        //     log_info(tt::LogTest, "Consumer TC: (0x{:x}, 0x{:x})", (uint32_t)::experimental::get_tensix_id(ctc),
        //     (uint32_t)::experimental::get_counter_id(ctc));
        // }
    }
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB2Sx4SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x3,
        .num_producers = 2,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x3C,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB4Sx2SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x30,
        .num_consumers = 2,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB1Sx1BConfig) {  // update this to not use the remapper
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB1Sx4BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x1E,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB4Sx1BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB4Sx4BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0xF0,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB4Sx2BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x30,
        .num_consumers = 2,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
}

TEST_F(
    MeshDeviceFixture, TensixTest1xDFB2Sx4BConfig) {  // get rid of num producers/consumers and use risc mask for counts
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x3,
        .num_producers = 2,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x3C,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
}

}  // end namespace tt::tt_metal
