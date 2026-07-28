// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end smoke test for Noc::async_write_zeros that exercises both overloads in one
// dispatch using a shared DFB:
//
//   - Host stamps an L1 status-flag word with the sentinel 0xBAADF00D.
//   - Host stamps a DRAM MeshTensor with 0xFFFFFFFF; verifies the stamp.
//   - Dispatch the program (one producer + one consumer sharing dfb::scratch).
//       Producer (zero_memory_api_l1_producer.cpp):
//         tests overload (1) on the DFB — CPU-stamps 0xFF, verifies stamp,
//         calls noc.async_write_zeros(dfb, ...), verifies the result is zero, reports
//         pass/fail via the L1 flag word, push_back's the (now zero-filled)
//         DFB entry.
//       Consumer (zero_memory_api_dram_consumer.cpp):
//         wait_front on the DFB (gets the L1-zeroed entry), loops
//         noc.async_write_zeros(accessor, page_size, {.page_id = p}, dfb) over
//         the DRAM tensor's pages, dram_barrier, pop_front.
//   - Host reads the L1 flag word; expects kStatusOk (kernel-level verify pass).
//   - Host reads the DRAM tensor; expects all zeros (overload-2 result).
//
// The L1 zero is dual-purpose: it's the test target for overload (1) AND the
// scratch fill for overload (2).

#include "device_fixture.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

#include "gtest/gtest.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

const experimental::DFBSpecName SCRATCH_DFB{"scratch"};
const experimental::TensorParamName OUT_TENSOR{"out"};

const experimental::KernelSpecName L1_PRODUCER{"l1_producer"};
const experimental::KernelSpecName DRAM_CONSUMER{"dram_consumer"};
const experimental::KernelSpecName L1_BATCHED_PRODUCER{"l1_batched_producer"};

constexpr uint32_t kStatusOk = 0xCAFEBABEu;

// Flat 1D UINT32 page layout: one DRAM page per logical row, page_size_bytes each.
// num_pages rows, page_size_bytes / 4 words per row.
TensorSpec make_flat_dram_tensor_spec(uint32_t page_size_bytes, uint32_t num_pages) {
    const uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::UINT32, page_config, memory_config);
    return TensorSpec(Shape{num_pages, page_size_words}, tensor_layout);
}

experimental::DataMovementHardwareConfig make_dm_config(tt::ARCH arch, DataMovementProcessor processor, NOC noc) {
    if (arch == tt::ARCH::QUASAR) {
        return experimental::DataMovementGen2Config{
            .disable_dfb_implicit_sync_for_all = true,
        };
    }
    return experimental::DataMovementGen1Config{.processor = processor, .noc = noc};
}

}  // namespace

namespace tt::tt_metal {

TEST_F(MeshDeviceSingleCardFixture, ZeroMemoryApi) {
    auto& mesh_device = *devices_[0];
    IDevice* dev = mesh_device.get_devices()[0];

    constexpr uint32_t scratch_bytes = 8 * 1024;
    constexpr uint32_t num_pages = 4;
    constexpr uint32_t page_size_bytes = 4 * 1024;
    constexpr uint32_t total_words = num_pages * (page_size_bytes / sizeof(uint32_t));
    constexpr uint32_t flag_addr = 100 * 1024;  // fixed L1 scratch addr for the status word
    const experimental::NodeCoord node{0, 0};

    // ----- Host stamps -----
    // L1 status flag: sentinel that the kernel demotes to kStatusOk on success.
    std::vector<uint32_t> flag_init{0xBAADF00Du};
    tt_metal::detail::WriteToDeviceL1(dev, node, flag_addr, flag_init);

    // DRAM tensor: 0xFF everywhere, so a no-op kernel can't pass the post-zero check.
    auto tensor = MeshTensor::allocate_on_device(
        mesh_device, make_flat_dram_tensor_spec(page_size_bytes, num_pages), TensorTopology{});
    std::vector<uint32_t> stamped(total_words, 0xFFFFFFFFu);
    detail::WriteToBuffer(*tensor.mesh_buffer().get_reference_buffer(), stamped);

    // Pre-write verify: confirm the DRAM stamp landed so the post-zero check is meaningful.
    std::vector<uint32_t> stamp_check;
    detail::ReadFromBuffer(*tensor.mesh_buffer().get_reference_buffer(), stamp_check);
    ASSERT_EQ(stamp_check.size(), total_words);
    for (uint32_t i = 0; i < total_words; ++i) {
        ASSERT_EQ(stamp_check[i], 0xFFFFFFFFu) << "Pre-write 0xFF stamp did not land at DRAM word " << i;
    }

    // ----- Program spec -----
    experimental::DataflowBufferSpec scratch_spec{
        .unique_id = SCRATCH_DFB,
        .entry_size = scratch_bytes,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Producer: tests overload (1) on the DFB, then push_backs the now-zero entry.
    experimental::KernelSpec producer_spec{
        .unique_id = L1_PRODUCER,
        .source =
            std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_l1_producer.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::PRODUCER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .runtime_arg_schema = {.runtime_arg_names = {"total_bytes", "flag_addr"}},
        .hw_config = make_dm_config(mesh_device.arch(), DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };

    // Consumer: wait_fronts on the L1-zeroed DFB entry, uses it as DRAM scratch for overload (2).
    experimental::KernelSpec consumer_spec{
        .unique_id = DRAM_CONSUMER,
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_consumer.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = {{"ZERO_DRAM", "1"}}},
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::CONSUMER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "out"}},
        .runtime_arg_schema = {.runtime_arg_names = {"page_start", "page_end", "page_size"}},
        .hw_config = make_dm_config(mesh_device.arch(), DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
    };

    experimental::ProgramSpec spec{
        .name = "zero_memory_api_end_to_end",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {scratch_spec},
        .tensor_parameters = {{.unique_id = OUT_TENSOR, .spec = tensor.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {L1_PRODUCER, DRAM_CONSUMER}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = L1_PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"total_bytes", scratch_bytes}, {"flag_addr", flag_addr}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DRAM_CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"page_start", 0u}, {"page_end", num_pages}, {"page_size", page_size_bytes}}),
        },
    };
    params.tensor_args = {{OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{tensor}}};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    // ----- Host verifies -----
    // L1: kernel reports its in-kernel verify result via the flag word.
    std::vector<uint32_t> flag_out;
    tt_metal::detail::ReadFromDeviceL1(dev, node, flag_addr, sizeof(uint32_t), flag_out);
    ASSERT_EQ(flag_out.size(), 1u);
    EXPECT_EQ(flag_out[0], kStatusOk) << "L1 zero test status word was 0x" << std::hex << flag_out[0] << " (expected 0x"
                                      << kStatusOk << ")";

    // DRAM: every word should be zero after overload (2).
    std::vector<uint32_t> result;
    detail::ReadFromBuffer(*tensor.mesh_buffer().get_reference_buffer(), result);
    ASSERT_EQ(result.size(), total_words);
    for (uint32_t i = 0; i < total_words; ++i) {
        EXPECT_EQ(result[i], 0u) << "DRAM word " << i << " not zeroed; got 0x" << std::hex << result[i];
        if (result[i] != 0u) {
            return;  // First mismatch is enough; bail to avoid spamming the log.
        }
    }
}

// Batched L1 zeroing: a kernel issues several noc.async_write_zeros() calls into disjoint
// chunks of one DFB entry and then barriers once.
TEST_F(MeshDeviceSingleCardFixture, ZeroMemoryApiBatchedL1) {
    auto& mesh_device = *devices_[0];
    IDevice* dev = mesh_device.get_devices()[0];

    constexpr uint32_t scratch_bytes = 32 * 1024;
    constexpr uint32_t num_chunks = 4;  // 4 disjoint 8 KB L1 zeros, then a single barrier
    constexpr uint32_t num_pages = 4;
    constexpr uint32_t page_size_bytes = 4 * 1024;
    constexpr uint32_t total_words = num_pages * (page_size_bytes / sizeof(uint32_t));
    constexpr uint32_t flag_addr = 100 * 1024;
    const experimental::NodeCoord node{0, 0};

    // L1 status flag: sentinel that the batched producer demotes to kStatusOk on success.
    std::vector<uint32_t> flag_init{0xBAADF00Du};
    tt_metal::detail::WriteToDeviceL1(dev, node, flag_addr, flag_init);

    auto tensor = MeshTensor::allocate_on_device(
        mesh_device, make_flat_dram_tensor_spec(page_size_bytes, num_pages), TensorTopology{});
    std::vector<uint32_t> stamped(total_words, 0xFFFFFFFFu);
    detail::WriteToBuffer(*tensor.mesh_buffer().get_reference_buffer(), stamped);

    experimental::DataflowBufferSpec scratch_spec{
        .unique_id = SCRATCH_DFB,
        .entry_size = scratch_bytes,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::KernelSpec producer_spec{
        .unique_id = L1_BATCHED_PRODUCER,
        .source =
            std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_l1_producer.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = {{"ZERO_NUM_CHUNKS", std::to_string(num_chunks)}}},
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::PRODUCER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .runtime_arg_schema = {.runtime_arg_names = {"total_bytes", "flag_addr"}},
        .hw_config = make_dm_config(mesh_device.arch(), DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };

    experimental::KernelSpec consumer_spec{
        .unique_id = DRAM_CONSUMER,
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_consumer.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = {{"ZERO_DRAM", "1"}}},
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::CONSUMER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "out"}},
        .runtime_arg_schema = {.runtime_arg_names = {"page_start", "page_end", "page_size"}},
        .hw_config = make_dm_config(mesh_device.arch(), DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
    };

    experimental::ProgramSpec spec{
        .name = "zero_memory_api_batched_l1",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {scratch_spec},
        .tensor_parameters = {{.unique_id = OUT_TENSOR, .spec = tensor.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {L1_BATCHED_PRODUCER, DRAM_CONSUMER}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = L1_BATCHED_PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"total_bytes", scratch_bytes}, {"flag_addr", flag_addr}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DRAM_CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"page_start", 0u}, {"page_end", num_pages}, {"page_size", page_size_bytes}}),
        },
    };
    params.tensor_args = {{OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{tensor}}};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    // The batched producer's in-kernel verify is the primary signal: kStatusOk only if
    // every byte across all chunks is zero after the single barrier.
    std::vector<uint32_t> flag_out;
    tt_metal::detail::ReadFromDeviceL1(dev, node, flag_addr, sizeof(uint32_t), flag_out);
    ASSERT_EQ(flag_out.size(), 1u);
    EXPECT_EQ(flag_out[0], kStatusOk) << "Batched L1 zero status word was 0x" << std::hex << flag_out[0]
                                      << " (expected 0x" << kStatusOk << "); a non-OK value means a batched zero was "
                                      << "lost (stale bytes remained after the single barrier).";
}

}  // namespace tt::tt_metal
