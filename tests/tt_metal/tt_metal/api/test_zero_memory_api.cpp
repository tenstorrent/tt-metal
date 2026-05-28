// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end smoke tests for Noc::write_zeros — the device-side zero-memory API
// added by tt-metal #11214. Two overloads, one TEST_F each:
//
//   1. L1 destination via DataflowBuffer (overload 1).
//      Kernel stamps the DFB region with 0xFF, verifies, calls noc.write_zeros,
//      verifies all-zero, then reports pass/fail via a flag L1 address that the
//      host reads back.
//
//   2. DRAM destination via TensorAccessor — caller-pre-zeroed scratch (overload 2).
//      Host stamps a DRAM MeshTensor with 0xFFFFFFFF, verifies the stamp landed,
//      then dispatches a kernel pair:
//        - producer kernel: zeros a DFB scratch via overload (1), push_back.
//        - writer kernel:   wait_front on the scratch, then loops
//                           noc.write_zeros(accessor, ..., scratch).
//      Host reads back and asserts all-zero. Tests that overload (2) reuses a
//      pre-zeroed L1 scratch correctly.

#include "device_fixture.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

#include "gtest/gtest.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr const char* SCRATCH_DFB = "scratch";
constexpr const char* OUT_TENSOR = "out";

constexpr const char* L1_PRODUCER = "l1_producer";
constexpr const char* L1_CONSUMER = "l1_consumer";
constexpr const char* SCRATCH_ZEROER = "scratch_zeroer";
constexpr const char* DRAM_WRITER = "dram_writer";

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

experimental::metal2_host_api::DataMovementConfiguration make_dm_config(DataMovementProcessor processor, NOC noc) {
    return experimental::metal2_host_api::DataMovementConfiguration{
        .gen1_data_movement_config =
            experimental::metal2_host_api::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = processor, .noc = noc},
        .gen2_data_movement_config =
            experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}};
}

}  // namespace

namespace tt::tt_metal {

// ============================================================================
// Overload (1): L1 destination via DataflowBuffer
// ============================================================================
TEST_F(MeshDeviceSingleCardFixture, ZeroMemoryApiL1ViaOverload1) {
    auto& mesh_device = *devices_[0];
    IDevice* dev = mesh_device.get_devices()[0];

    constexpr uint32_t total_bytes = 4 * 1024;  // 4 KB — exercises the chunked path
    constexpr uint32_t flag_addr = 100 * 1024;  // fixed L1 scratch addr for the status word
    const experimental::metal2_host_api::NodeCoord node{0, 0};

    // Host-side: initialize the flag to a sentinel so a no-op kernel can be detected.
    std::vector<uint32_t> flag_init{0xBAADF00Du};
    tt_metal::detail::WriteToDeviceL1(dev, node, flag_addr, flag_init);

    experimental::metal2_host_api::DataflowBufferSpec scratch_spec{
        .unique_id = SCRATCH_DFB,
        .entry_size = total_bytes,
        .num_entries = 1,
        // Float16_b is the only data format both WH/BH and Quasar accept here; the DFB is
        // just used as a raw L1 scratch region so the choice doesn't affect correctness.
        .data_format_metadata = tt::DataFormat::Float16_b,
        .disable_implicit_sync = true,
    };

    experimental::metal2_host_api::KernelSpec producer_spec{
        .unique_id = L1_PRODUCER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_l1_producer.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .local_accessor_name = "scratch",
              .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
              .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED}},
        .runtime_arguments_schema = {.named_runtime_args = {"total_bytes", "flag_addr"}},
        .config_spec = make_dm_config(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };

    experimental::metal2_host_api::KernelSpec consumer_spec{
        .unique_id = L1_CONSUMER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_l1_consumer.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .local_accessor_name = "scratch",
              .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
              .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED}},
        .config_spec = make_dm_config(DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "zero_memory_api_l1",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {scratch_spec},
        .work_units = {{.unique_id = "main", .kernels = {L1_PRODUCER, L1_CONSUMER}, .target_nodes = node}},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = L1_PRODUCER,
            .named_runtime_args = {{.node = node, .args = {{"total_bytes", total_bytes}, {"flag_addr", flag_addr}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{.kernel_spec_name = L1_CONSUMER},
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    std::vector<uint32_t> flag_out;
    tt_metal::detail::ReadFromDeviceL1(dev, node, flag_addr, sizeof(uint32_t), flag_out);
    ASSERT_EQ(flag_out.size(), 1u);
    EXPECT_EQ(flag_out[0], kStatusOk) << "L1 zero test status word was 0x" << std::hex << flag_out[0] << " (expected 0x"
                                      << kStatusOk << ")";
}

// ============================================================================
// Overload (2): DRAM destination via TensorAccessor — caller-pre-zeroed scratch
// ============================================================================
TEST_F(MeshDeviceSingleCardFixture, ZeroMemoryApiDramViaOverload2) {
    auto& mesh_device = *devices_[0];

    constexpr uint32_t num_pages = 4;
    constexpr uint32_t page_size_bytes = 4 * 1024;
    // Scratch must contain at least min(page_size_bytes, NOC_MAX_BURST_SIZE) pre-zeroed bytes.
    // 8 KB ≥ 4 KB on all archs (WH/BH/Quasar NOC_MAX_BURST_SIZE are 8 KB / 16 KB / 64 KB).
    constexpr uint32_t scratch_size_bytes = 8 * 1024;
    constexpr uint32_t total_words = num_pages * (page_size_bytes / sizeof(uint32_t));
    const experimental::metal2_host_api::NodeCoord node{0, 0};

    auto tensor = MeshTensor::allocate_on_device(
        mesh_device, make_flat_dram_tensor_spec(page_size_bytes, num_pages), TensorTopology{});

    std::vector<uint32_t> stamped(total_words, 0xFFFFFFFFu);
    detail::WriteToBuffer(*tensor.mesh_buffer().get_reference_buffer(), stamped);

    std::vector<uint32_t> stamp_check;
    detail::ReadFromBuffer(*tensor.mesh_buffer().get_reference_buffer(), stamp_check);
    ASSERT_EQ(stamp_check.size(), total_words);
    for (uint32_t i = 0; i < total_words; ++i) {
        ASSERT_EQ(stamp_check[i], 0xFFFFFFFFu) << "Pre-write 0xFF stamp did not land at word " << i;
    }

    experimental::metal2_host_api::DataflowBufferSpec scratch_spec{
        .unique_id = SCRATCH_DFB,
        .entry_size = scratch_size_bytes,
        .num_entries = 1,
        // Float16_b is the only data format both WH/BH and Quasar accept here; the DFB is
        // just used as a raw L1 scratch region so the choice doesn't affect correctness.
        .data_format_metadata = tt::DataFormat::Float16_b,
        .disable_implicit_sync = true,
    };

    experimental::metal2_host_api::KernelSpec zeroer_spec{
        .unique_id = SCRATCH_ZEROER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_scratch_zeroer.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .local_accessor_name = "scratch",
              .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
              .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED}},
        .runtime_arguments_schema = {.named_runtime_args = {"scratch_size_bytes"}},
        .config_spec = make_dm_config(DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
    };

    experimental::metal2_host_api::KernelSpec writer_spec{
        .unique_id = DRAM_WRITER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_dram_with_scratch_writer.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .local_accessor_name = "scratch",
              .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
              .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED}},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "out"}},
        .runtime_arguments_schema = {.named_runtime_args = {"page_start", "page_end", "page_size"}},
        .config_spec = make_dm_config(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "zero_memory_api_dram_with_scratch",
        .kernels = {zeroer_spec, writer_spec},
        .dataflow_buffers = {scratch_spec},
        .tensor_parameters = {{.unique_id = OUT_TENSOR, .spec = tensor.tensor_spec()}},
        .work_units = {{.unique_id = "main", .kernels = {SCRATCH_ZEROER, DRAM_WRITER}, .target_nodes = node}},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = SCRATCH_ZEROER,
            .named_runtime_args = {{.node = node, .args = {{"scratch_size_bytes", scratch_size_bytes}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = DRAM_WRITER,
            .named_runtime_args =
                {{.node = node, .args = {{"page_start", 0u}, {"page_end", num_pages}, {"page_size", page_size_bytes}}}},
        },
    };
    params.tensor_args = {{.tensor_parameter_name = OUT_TENSOR, .tensor = tensor}};
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    std::vector<uint32_t> result;
    detail::ReadFromBuffer(*tensor.mesh_buffer().get_reference_buffer(), result);
    ASSERT_EQ(result.size(), total_words);
    for (uint32_t i = 0; i < total_words; ++i) {
        EXPECT_EQ(result[i], 0u) << "DRAM word " << i << " not zeroed; got 0x" << std::hex << result[i];
        if (result[i] != 0u) {
            return;
        }
    }
}

}  // namespace tt::tt_metal
