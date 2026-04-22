// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-hardware tests for Metal 2.0 Host API: ProgramSpec on WH/BH.
//
// These tests require a Wormhole B0 or Blackhole device and slow dispatch mode.
// They prove the ProgramSpec → MakeProgramFromSpec → compile → dispatch → verify pipeline
// end-to-end on real hardware, with particular focus on DFB local accessor names.
//
// Requires: TT_METAL_SLOW_DISPATCH_MODE=1
//
// TODO: Switch to using fast dispatch once the MeshWorkload code paths are added.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "test_helpers.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

using test_helpers::BindDFBToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorker;

// ============================================================================
// Test Fixture
// ============================================================================

class ProgramSpecHWTest : public tt::tt_metal::MeshDeviceFixture {
protected:
    void SetUp() override {
        MeshDeviceFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        // These tests target Gen1 (WH/BH) only
        if (devices_.at(0)->arch() != tt::ARCH::WORMHOLE_B0 && devices_.at(0)->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

// ============================================================================
// DFB Local Accessor Name Loopback Test
// ============================================================================
//
// Proves that DFB local accessor names work end-to-end on real WH/BH hardware:
//   1. kernel_bindings_generated.h is emitted correctly (dfb::buf resolves at compile time)
//   2. The DFBAccessor mechanism works (DFB ID maps to the correct underlying CB)
//   3. Data flows correctly through the DFB from producer to consumer
//
// Pipeline:
//   Host writes random data → DRAM input buffer (single page = one bank)
//   Producer DM kernel (BRISC) reads DRAM → DFB (using dfb::buf)
//   Consumer DM kernel (NCRISC) reads DFB → DRAM (using dfb::buf)
//   Host reads DRAM output buffer and verifies match

TEST_F(ProgramSpecHWTest, DFBAccessorNameLoopback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    // Test parameters
    constexpr uint32_t entry_size = 1024;  // bytes per DFB entry
    constexpr uint32_t num_entries = 4;    // DFB depth (double-buffer + margin)
    constexpr uint32_t num_transfers = 8;  // total entries to move through the DFB
    constexpr uint32_t total_bytes = entry_size * num_transfers;

    // Use a single core for simplicity
    const NodeCoord node{0, 0};

    // -------------------------------------------------------
    // Create DRAM buffers (single-page so all data is on one bank)
    // -------------------------------------------------------
    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    // -------------------------------------------------------
    // Build ProgramSpec
    // -------------------------------------------------------
    ProgramSpec spec;
    spec.program_id = "dfb_accessor_loopback";

    // Producer: BRISC reads from DRAM → DFB
    auto producer = MakeMinimalGen1DMKernel("producer", node, DataMovementProcessor::RISCV_0);
    producer.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_producer.cpp"};
    producer.runtime_arguments_schema.num_runtime_varargs = 3;

    // Consumer: NCRISC reads DFB → DRAM
    auto consumer = MakeMinimalGen1DMKernel("consumer", node, DataMovementProcessor::RISCV_1);
    consumer.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp"};
    consumer.runtime_arguments_schema.num_runtime_varargs = 3;

    // DFB: both kernels bind it, with different local accessor names
    auto dfb = MakeMinimalDFB("loopback_dfb", node, entry_size, num_entries);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    BindDFBToKernel(producer, "loopback_dfb", "my_local_dfb_name", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "loopback_dfb", "a_dfb_named_bob", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers =
        std::vector<WorkerSpec>{MakeMinimalWorker("worker_0", node, {"producer", "consumer"}, {"loopback_dfb"})};

    // -------------------------------------------------------
    // Create Program
    // -------------------------------------------------------
    Program program = MakeProgramFromSpec(spec);

    // -------------------------------------------------------
    // Set runtime args
    // -------------------------------------------------------
    ProgramRunParams params;
    params.kernel_run_params = {
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "producer",
            .runtime_varargs =
                {{node,
                  {
                      input_buffer->address(),
                      0u,  // bank_id (single-page buffer → bank 0)
                      num_transfers,
                  }}},
        },
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "consumer",
            .runtime_varargs =
                {{node,
                  {
                      output_buffer->address(),
                      0u,  // bank_id
                      num_transfers,
                  }}},
        },
    };
    SetProgramRunParameters(program, params);

    // -------------------------------------------------------
    // Fill input buffer with known data
    // -------------------------------------------------------
    std::vector<uint32_t> input_data(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<uint32_t>(i);
    }
    detail::WriteToBuffer(input_buffer, input_data);

    // -------------------------------------------------------
    // Dispatch
    // -------------------------------------------------------
    detail::LaunchProgram(device, program);

    // -------------------------------------------------------
    // Verify
    // -------------------------------------------------------
    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);

    ASSERT_EQ(output_data.size(), input_data.size());
    EXPECT_EQ(output_data, input_data);
}

// ============================================================================
// Named RTA / CRTA / CTA Loopback Test
// ============================================================================
//
// End-to-end on real WH/BH hardware:
//   1. kernel_args_generated.h is emitted (args::src_addr, args::dst_addr, args::num_entries,
//      args::bank_id, args::entry_size resolve at kernel compile time)
//   2. Named RTAs + CRTAs + CTAs all route correctly through the dispatch buffer
//   3. get_vararg(0) correctly indexes past the named RTA section
//   4. Data flows end-to-end: DRAM → producer (named args) → DFB → consumer (named args) → DRAM
//
// Same shape as DFBAccessorNameLoopback but all runtime args are sourced via the new
// named-arg accessors, plus one vararg on the producer to verify the offset mechanism.

TEST_F(ProgramSpecHWTest, NamedArgsLoopback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries_in_dfb = 4;
    constexpr uint32_t num_transfers = 8;
    constexpr uint32_t total_bytes = entry_size * num_transfers;
    constexpr uint32_t producer_vararg_sentinel = 0xC0FFEEu;

    const NodeCoord node{0, 0};

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.program_id = "named_args_loopback";

    // Producer: BRISC reads DRAM → DFB, using named RTA (src_addr) + named CRTA (num_entries)
    // + named CTAs (bank_id, entry_size) + one vararg RTA (a sentinel for offset verification).
    auto producer = MakeMinimalGen1DMKernel("producer", node, DataMovementProcessor::RISCV_0);
    producer.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/dataflow/named_args_loopback_producer.cpp"};
    producer.runtime_arguments_schema.named_runtime_args = {"src_addr"};
    producer.runtime_arguments_schema.named_common_runtime_args = {"num_entries"};
    producer.runtime_arguments_schema.num_runtime_varargs = 1;  // 1 vararg
    producer.compile_time_arg_bindings = {{"bank_id", 0}, {"entry_size", entry_size}};

    // Consumer: NCRISC reads DFB → DRAM, using named RTA (dst_addr) + named CRTA (num_entries)
    // + named CTAs (bank_id, entry_size). No varargs.
    auto consumer = MakeMinimalGen1DMKernel("consumer", node, DataMovementProcessor::RISCV_1);
    consumer.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/dataflow/named_args_loopback_consumer.cpp"};
    consumer.runtime_arguments_schema.named_runtime_args = {"dst_addr"};
    consumer.runtime_arguments_schema.named_common_runtime_args = {"num_entries"};
    consumer.compile_time_arg_bindings = {{"bank_id", 0}, {"entry_size", entry_size}};

    auto dfb = MakeMinimalDFB("loopback_dfb", node, entry_size, num_entries_in_dfb);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    BindDFBToKernel(producer, "loopback_dfb", "loopback_dfb", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "loopback_dfb", "loopback_dfb", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers =
        std::vector<WorkerSpec>{MakeMinimalWorker("worker_0", node, {"producer", "consumer"}, {"loopback_dfb"})};

    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params = {
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "producer",
            .named_runtime_args = {{.node = node, .args = {{"src_addr", input_buffer->address()}}}},
            .named_common_runtime_args = {{"num_entries", num_transfers}},
            .runtime_varargs = {{node, {producer_vararg_sentinel}}},
        },
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "consumer",
            .named_runtime_args = {{.node = node, .args = {{"dst_addr", output_buffer->address()}}}},
            .named_common_runtime_args = {{"num_entries", num_transfers}},
        },
    };
    SetProgramRunParameters(program, params);

    std::vector<uint32_t> input_data(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<uint32_t>(i);
    }
    detail::WriteToBuffer(input_buffer, input_data);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);

    ASSERT_EQ(output_data.size(), input_data.size());
    EXPECT_EQ(output_data, input_data);
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
