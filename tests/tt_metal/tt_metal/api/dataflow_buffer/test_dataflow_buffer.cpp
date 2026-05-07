// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
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

    if (mesh_device->get_devices()[0]->arch() == ARCH::QUASAR) {
        // TODO #38042: Need to wait for data to be written, the barrier needs to be uplifted for Quasar
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::vector<uint32_t> rdback_dram;
        distributed::ReadShard(mesh_device->mesh_command_queue(), rdback_dram, in_buffer, zero_coord, true);

        tt_driver_atomics::mfence();

        EXPECT_EQ(rdback_dram, input);
    }

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
    const CoreRangeSet& core_range_set = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))),
    std::optional<uint32_t> num_entries_in_buffer = std::nullopt) {

    TT_FATAL(
        !(producer_type == DFBPorCType::TENSIX && consumer_type == DFBPorCType::TENSIX),
        "Both producer and consumer cannot be Tensix. At least one must be a DM kernel for NOC transfers.");
    TT_FATAL(
        core_range_set.num_cores() == 1 ||
            (producer_type == DFBPorCType::DM && consumer_type == DFBPorCType::DM),
        "Multi-core DFB programs only support DM producer and consumer.");

    const auto arch = mesh_device->get_devices()[0]->arch();
    const bool is_quasar = (arch == ARCH::QUASAR);

    if (!is_quasar) {
        // WH/BH DM: one BRISC (RISCV_0) as producer and one NCRISC (RISCV_1) as consumer.
        // Configs with num_producers > 1 or num_consumers > 1 require multi-threaded DM
        // which is not available on WH/BH.
        if (dfb_config.num_producers > 1 || dfb_config.num_consumers > 1) {
            GTEST_SKIP() << "WH/BH DFB supports only 1 DM producer (BRISC) and 1 DM consumer (NCRISC)";
        }
        // read_in / write_out are Quasar-only; the device-side kernel would fail to compile
        // if enable_implicit_sync=true is propagated as a compile-time arg.
        dfb_config.enable_implicit_sync = false;
    }

    Program program = CreateProgram();
    auto zero_coord = distributed::MeshCoordinate(0, 0);

    const uint32_t num_cores = core_range_set.num_cores();
    const uint32_t entries_per_core = num_entries_in_buffer.has_value() ? num_entries_in_buffer.value() : dfb_config.num_entries;
    const uint32_t entry_size = dfb_config.entry_size;
    // page_size = entry_size makes every entry independently addressable by page_id.
    const uint32_t total_buffer_size = num_cores * entries_per_core * entry_size;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    log_info(tt::LogTest, "In Buffer:  [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
    log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

    uint32_t num_entries_per_producer = entries_per_core / dfb_config.num_producers;
    const bool is_all = (dfb_config.cap == dfb::AccessPattern::ALL);
    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(),
        num_entries_per_producer,
        (uint32_t)dfb_config.enable_implicit_sync,
        (uint32_t)is_all};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

    KernelHandle producer_kernel;
    if (producer_type == DFBPorCType::DM) {
        const std::string dm_producer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp";
        if (is_quasar) {
            producer_kernel = experimental::quasar::CreateKernel(
                program,
                dm_producer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarDataMovementConfig{
                    .num_threads_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});
        } else {
            producer_kernel = CreateKernel(
                program,
                dm_producer_kernel_path,
                core_range_set,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .compile_args = producer_cta});
        }
    } else {
        const std::string t6_producer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer.cpp";
        if (is_quasar) {
            producer_kernel = CreateKernel(
                program,
                t6_producer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarComputeConfig{
                    .num_threads_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});
        } else {
            producer_kernel = CreateKernel(
                program, t6_producer_kernel_path, core_range_set, ComputeConfig{.compile_args = producer_cta});
        }
    }

    uint32_t num_entries_per_consumer = is_all ? entries_per_core : entries_per_core / dfb_config.num_consumers;
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        num_entries_per_consumer,
        (uint32_t)is_all,
        (uint32_t)dfb_config.enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);

    KernelHandle consumer_kernel;
    if (consumer_type == DFBPorCType::DM) {
        const std::string dm_consumer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp";
        if (is_quasar) {
            consumer_kernel = experimental::quasar::CreateKernel(
                program,
                dm_consumer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarDataMovementConfig{
                    .num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});
        } else {
            consumer_kernel = CreateKernel(
                program,
                dm_consumer_kernel_path,
                core_range_set,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .compile_args = consumer_cta});
        }
    } else {
        const std::string t6_consumer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp";
        if (is_quasar) {
            consumer_kernel = CreateKernel(
                program,
                t6_consumer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarComputeConfig{
                    .num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});
        } else {
            consumer_kernel = CreateKernel(
                program, t6_consumer_kernel_path, core_range_set, ComputeConfig{.compile_args = consumer_cta});
        }
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
    //
    // l1_by_core addresses are not populated until allocate_dataflow_buffers() runs
    // during program compilation. Since this is a single-DFB test it is always placed at the L1 base allocator address.
    if (producer_type == DFBPorCType::TENSIX) {
        const uint32_t dfb_l1_addr =
            static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
        for (const CoreRange& cr : core_range_set.ranges()) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                    const CoreCoord core(x, y);
                    const uint32_t co = core_to_chunk_offset.at(core);
                    std::vector<uint32_t> slice(
                        input.begin() + co * entry_size / sizeof(uint32_t),
                        input.begin() + co * entry_size / sizeof(uint32_t) + words_per_core);
                    detail::WriteToDeviceL1(device, core, dfb_l1_addr, slice);
                }
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
    const bool in_is_all = (dm2tensix_config.cap == dfb::AccessPattern::ALL);
    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(),
        num_entries_per_producer,
        0 /*implicit_sync=false*/,
        (uint32_t)in_is_all};
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

    const bool out_is_all = (tensix2dm_config.cap == dfb::AccessPattern::ALL);
    uint32_t num_entries_per_consumer = out_is_all ? tensix2dm_config.num_entries : tensix2dm_config.num_entries / tensix2dm_config.num_consumers;
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        num_entries_per_consumer,
        (uint32_t)out_is_all,
        0 /*implicit_sync=false*/};
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
    if (devices_.at(0)->arch() != ARCH::QUASAR and GetParam()) {
        GTEST_SKIP();
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    uint32_t num_entries_in_buffer = 18;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set, num_entries_in_buffer);
}

TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR and GetParam()) {
        GTEST_SKIP();
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
    if (devices_.at(0)->arch() != ARCH::QUASAR and GetParam()) {
        GTEST_SKIP();
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

    uint32_t num_entries_in_buffer = 29;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set, num_entries_in_buffer);
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

    uint32_t num_entries_in_buffer = 21;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set, num_entries_in_buffer);
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

// AccessPattern::ALL

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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
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
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

INSTANTIATE_TEST_SUITE_P(
    ImplicitSync,
    DFBImplicitSyncParamFixture,
    ::testing::Bool(),
    ImplicitSyncParamName);

// Runs an intra-tensix DFB program on one core.
static void run_intra_tensix_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_threads) {
    IDevice* device = mesh_device->get_devices()[0];

    experimental::dfb::DataflowBufferConfig dfb_config{
        .entry_size = entry_size,
        .num_entries = num_entries,
        .num_producers = num_threads,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = num_threads,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_scope = experimental::dfb::TensixScope::INTRA};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);

    TT_FATAL(
        num_entries % num_threads == 0,
        "num_entries ({}) must be divisible by num_threads ({}) for intra-tensix block partitioning",
        num_entries, num_threads);
    const uint32_t entries_per_neo = num_entries / num_threads;

    const std::string intra_kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_intra.cpp";
    std::vector<uint32_t> cta = {entries_per_neo, words_per_entry};

    KernelHandle compute_kernel = CreateKernel(
        program,
        intra_kernel_path,
        core_range_set,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = num_threads,
            .compile_args = cta});

    auto logical_dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
    // Bind the same kernel as both producer and consumer: packer TRISC2 and unpacker TRISC0
    // on each Neo share the Tensix-only TC allocated for that Neo.
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, logical_dfb_id, compute_kernel, compute_kernel);

    const uint32_t total_size = num_entries * entry_size;
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, total_size / sizeof(uint32_t));

    const uint32_t dfb_l1_addr =
        static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));

    detail::WriteToDeviceL1(device, logical_core, dfb_l1_addr, input);

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Packer increments each word by 1, then unpacker increments it by 1 → +2 per word.
    // This holds for every Neo's ring independently, so the entire L1 region is input + 2.
    std::vector<uint32_t> expected(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        expected[i] = input[i] + 2;
    }

    std::vector<uint32_t> l1_data;
    detail::ReadFromDeviceL1(device, logical_core, dfb_l1_addr, total_size, l1_data);
    EXPECT_EQ(expected, l1_data) << "Intra-tensix DFB L1 mismatch";
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping intra-tensix DFB test for WH/BH until DFB is backported";
    }
    run_intra_tensix_dfb_program(this->devices_.at(0), /*entry_size=*/1024, /*num_entries=*/16, /*num_threads=*/1);
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping intra-tensix DFB test for WH/BH until DFB is backported";
    }
    run_intra_tensix_dfb_program(this->devices_.at(0), /*entry_size=*/1024, /*num_entries=*/16, /*num_threads=*/4);
}

TEST_F(MeshDeviceFixture, TensixIntraAndRemapperTest_4Neo_DM1Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping combined intra-tensix + remapper DFB test for WH/BH until DFB is backported";
    }

    IDevice* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core(0, 0);
    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    constexpr uint32_t entry_size  = 1024;
    constexpr uint32_t num_entries = 16;
    constexpr uint32_t num_neos    = 4;
    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);
    const uint32_t entries_per_neo = num_entries / num_neos;  // = 4

    // dfb(0): DM->Tensix, 1Sx4B with remapper, implicit sync enabled.
    experimental::dfb::DataflowBufferConfig remapper_dfb_config{
        .entry_size           = entry_size,
        .num_entries          = num_entries,
        .num_producers        = 1,
        .pap                  = dfb::AccessPattern::STRIDED,
        .num_consumers        = num_neos,
        .cap                  = dfb::AccessPattern::ALL,
        .enable_implicit_sync = true};

    // dfb(1): intra-tensix, 4 packer->unpacker pairs, hidden TCs.
    experimental::dfb::DataflowBufferConfig intra_dfb_config{
        .entry_size           = entry_size,
        .num_entries          = num_entries,
        .num_producers        = num_neos,
        .pap                  = dfb::AccessPattern::STRIDED,
        .num_consumers        = num_neos,
        .cap                  = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_scope         = experimental::dfb::TensixScope::INTRA};

    const uint32_t buf_size = num_entries * entry_size;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    distributed::DeviceLocalBufferConfig local_cfg{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    auto in_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buf_size}, local_cfg, this->devices_.at(0).get());

    Program program = CreateProgram();

    // DM producer (1 thread, implicit sync enabled): reads DRAM in_buffer -> dfb(0) via NOC.
    std::vector<uint32_t> dm_cta = {
        (uint32_t)in_buffer->address(),
        num_entries,   // num_entries_per_producer: 1 producer owns all entries
        1u,            // implicit_sync = true
        0u};           // consume_all = false (producer is always STRIDED)
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(dm_cta);

    auto dm_producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = dm_cta});

    // Create DFBs: remapper first (-> logical ID 0), intra-tensix second (-> logical ID 1).
    auto remapper_dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, remapper_dfb_config);
    auto intra_dfb_id    = experimental::dfb::CreateDataflowBuffer(program, core_range_set, intra_dfb_config);

    // Combined compute kernel (4 Neo clusters):
    //   CTA[0] = num_entries      - ALL consumer loop count (each UNPACK sees all 16 entries)
    //   CTA[1] = entries_per_neo  - intra-tensix loop count per Neo (= 4)
    //   CTA[2] = words_per_entry  - words per dfb(1) entry for in-place increment
    std::vector<uint32_t> compute_cta = {num_entries, entries_per_neo, words_per_entry};

    auto compute_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_intra_and_consume_all.cpp",
        core_range_set,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = num_neos, .compile_args = compute_cta});

    // dfb(0): DM producer -> compute ALL consumer (remapper fans out 1 TC post to 4 UNPACK TCs).
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, remapper_dfb_id, dm_producer_kernel, compute_kernel);
    // dfb(1): compute kernel is both producer (PACK TRISC) and consumer (UNPACK TRISC).
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, intra_dfb_id, compute_kernel, compute_kernel);

    // Runtime args for DM producer (dfb_producer.cpp: [0]=producer_mask, [1]=chunk_offset).
    auto remapper_dfb_impl = program.impl().get_dataflow_buffer(remapper_dfb_id);
    const uint32_t dm_producer_mask = remapper_dfb_impl->config.producer_risc_mask;
    SetRuntimeArgs(program, dm_producer_kernel, logical_core, {dm_producer_mask, 0u /*chunk_offset*/});

    // L1 layout follows DFB creation order:
    //   [l1_base + 0              ] -> dfb(0) remapper ring  (num_entries * entry_size bytes)
    //   [l1_base + remapper_size  ] -> dfb(1) intra ring     (num_entries * entry_size bytes)
    const uint32_t l1_base           = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t remapper_ring_size = num_entries * entry_size;
    const uint32_t intra_l1_addr     = l1_base + remapper_ring_size;

    // Pre-fill dfb(1)'s intra-tensix ring; kernel adds +2 per word.
    auto input_intra = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, num_entries * words_per_entry);
    detail::WriteToDeviceL1(device, logical_core, intra_l1_addr, input_intra);

    // Fill DRAM in_buffer; DM NOC-reads this into dfb(0)'s ring.
    auto input_remapper = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, num_entries * words_per_entry);
    distributed::WriteShard(this->devices_.at(0)->mesh_command_queue(), in_buffer, input_remapper, zero_coord, true);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Verify dfb(1): packer +1, unpacker +1 -> net +2 per word.
    {
        std::vector<uint32_t> expected(input_intra.size());
        for (size_t i = 0; i < input_intra.size(); i++) {
            expected[i] = input_intra[i] + 2;
        }
        std::vector<uint32_t> l1_data;
        detail::ReadFromDeviceL1(device, logical_core, intra_l1_addr, num_entries * entry_size, l1_data);
        EXPECT_EQ(expected, l1_data) << "Intra-tensix DFB L1 mismatch";
    }

    // Verify dfb(0): DM NOC-wrote input_remapper into L1; Tensix consumed but did not overwrite.
    {
        std::vector<uint32_t> l1_data;
        detail::ReadFromDeviceL1(device, logical_core, l1_base, num_entries * entry_size, l1_data);
        EXPECT_EQ(input_remapper, l1_data) << "DM->Tensix strided x all DFB L1 mismatch";
    }
}

}  // end namespace tt::tt_metal
