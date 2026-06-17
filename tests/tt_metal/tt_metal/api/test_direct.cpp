// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <cstdint>
#include <sys/types.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;
using namespace tt::tt_metal;

namespace unit_tests::dram::direct {
/// @brief Does Dram --> Reader --> L1 on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_only(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const size_t& byte_size,
    const size_t& l1_byte_address,
    const CoreCoord& reader_core) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_byte_address = input_dram_buffer->address();
    // TODO (abhullar): Use L1 buffer after bug with L1 banking and writing to < 1 MB is fixed.
    //                  Try this after KM uplifts TLB setup
    // auto l1_buffer =
    //     CreateBuffer(device, byte_size, l1_byte_address, byte_size, tt_metal::BufferType::L1);

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_dram_to_l1.cpp",
        reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::detail::WriteToBuffer(input_dram_buffer, inputs);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        reader_core,
        {
            static_cast<uint32_t>(dram_byte_address),
            0,
            static_cast<uint32_t>(l1_byte_address),
            static_cast<uint32_t>(byte_size),
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> dest_core_data;
    // tt_metal::detail::ReadFromBuffer(l1_buffer, dest_core_data);
    tt_metal::detail::ReadFromDeviceL1(device, reader_core, l1_byte_address, byte_size, dest_core_data);
    pass &= (dest_core_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << reader_core.str() << std::endl;
    }
    return pass;
}

/// @brief Does L1 --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool writer_only(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const size_t& byte_size,
    const size_t& l1_byte_address,
    const CoreCoord& writer_core) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt_metal::BufferType::DRAM};

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_byte_address = output_dram_buffer->address();
    // TODO (abhullar): Use L1 buffer after bug with L1 banking and writing to < 1 MB is fixed.
    //                  Try this after KM uplifts TLB setup
    // auto l1_buffer =
    //     CreateBuffer(device, byte_size, l1_byte_address, byte_size, tt_metal::BufferType::L1);

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_l1_to_dram.cpp",
        writer_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::detail::WriteToDeviceL1(device, writer_core, l1_byte_address, inputs);
    // tt_metal::detail::WriteToBuffer(l1_buffer, inputs);

    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        writer_core,
        {
            static_cast<uint32_t>(dram_byte_address),
            0,
            static_cast<uint32_t>(l1_byte_address),
            static_cast<uint32_t>(byte_size),
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= (dest_buffer_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << writer_core.str() << std::endl;
    }
    return pass;
}

struct ReaderWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_data_format = tt::DataFormat::Invalid;
    experimental::NodeCoord node;
};
/// @brief Does Dram --> Reader --> DFB --> Writer --> Dram on a single core.
/// Uses the Metal 2.0 host API on every arch; on WH/BH the runtime selects the
/// Gen1 DM configs, on Quasar the Gen2 configs.
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_writer(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ReaderWriterConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto output_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    std::vector<uint32_t> inputs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::detail::WriteToBuffer(input_dram_buffer, inputs);

    // DRAM buffer is configured with page_size = byte_size (whole buffer),
    // so aligned_page_size() returns the whole-buffer stride, not per-tile.
    // Derive the per-tile DRAM stride directly from byte_size / num_tiles.
    const uint32_t per_tile_stride = static_cast<uint32_t>(byte_size / test_config.num_tiles);

    const bool is_quasar = device->arch() == ARCH::QUASAR;
    // On Quasar we can split work across two DM threads when num_tiles > 1;
    // WH/BH gen1 has one DM thread per processor, so the kernel only runs on
    // a single thread there.
    if (is_quasar && test_config.num_tiles > 1) {
        TT_FATAL(test_config.num_tiles % 2 == 0, "Number of tiles must be divisible by 2");
    }
    const uint32_t num_threads = (is_quasar && test_config.num_tiles > 1) ? 2u : 1u;
    const uint32_t num_tiles_per_thread = test_config.num_tiles / num_threads;

    const experimental::DFBSpecName L1_DFB{"l1_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};

    experimental::DataflowBufferSpec l1_dfb_spec{
        .unique_id = L1_DFB,
        .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
        .num_entries = static_cast<uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_data_format,
    };

    // Both gen1 and gen2 configs are populated; the runtime picks the one
    // matching the active arch.
    experimental::DataMovementHardwareConfig reader_dm_cfg{
        .gen1_config =
            experimental::DataMovementHardwareConfig::Gen1Config{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default},
        .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{},
    };
    experimental::DataMovementHardwareConfig writer_dm_cfg{
        .gen1_config =
            experimental::DataMovementHardwareConfig::Gen1Config{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
        .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{},
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = num_threads,
        .dfb_bindings = {experimental::ProducerOf(L1_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = reader_dm_cfg,
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = num_threads,
        .dfb_bindings = {experimental::ConsumerOf(L1_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = writer_dm_cfg,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = test_config.node,
    };

    experimental::ProgramSpec spec{
        .name = "reader_writer",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {l1_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{test_config.node,
                  {{"src_addr", input_dram_byte_address},
                   {"src_bank_id", 0u},
                   {"num_tiles", num_tiles_per_thread},
                   {"dram_page_stride", per_tile_stride}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{test_config.node,
                  {{"dst_addr", output_dram_byte_address},
                   {"dst_bank_id", 0u},
                   {"num_tiles", num_tiles_per_thread},
                   {"dram_page_stride", per_tile_stride}}}},
        },
    };
    experimental::SetProgramRunArgs(program, params);

    tt_metal::detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    return inputs == dest_buffer_data;
}
struct ReaderDatacopyWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    experimental::NodeCoord node;
};

// Shared host-side state for reader_datacopy_writer: DRAM buffers, sizing,
// generated input data (already written to input DRAM), and the per-tile DRAM
// stride used when wiring reader/writer runtime args.
struct ReaderDatacopyWriterContext {
    std::shared_ptr<tt::tt_metal::Buffer> input_dram_buffer;
    std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer;
    uint32_t input_dram_byte_address = 0;
    uint32_t output_dram_byte_address = 0;
    size_t byte_size = 0;
    uint32_t per_tile_stride = 0;
    std::vector<uint32_t> inputs;
};

static ReaderDatacopyWriterContext setup_reader_datacopy_writer_context(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ReaderDatacopyWriterConfig& test_config) {
    ReaderDatacopyWriterContext ctx;
    ctx.byte_size = test_config.num_tiles * test_config.tile_byte_size;

    auto* device = mesh_device->get_devices()[0];
    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = ctx.byte_size,
        .page_size = ctx.byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    ctx.input_dram_buffer = tt_metal::CreateBuffer(dram_config);
    ctx.input_dram_byte_address = ctx.input_dram_buffer->address();
    ctx.output_dram_buffer = tt_metal::CreateBuffer(dram_config);
    ctx.output_dram_byte_address = ctx.output_dram_buffer->address();

    log_info(tt::LogTest, "Input DRAM byte address: {}", ctx.input_dram_byte_address);
    log_info(tt::LogTest, "Output DRAM byte address: {}", ctx.output_dram_byte_address);

    ctx.inputs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, ctx.byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::detail::WriteToBuffer(ctx.input_dram_buffer, ctx.inputs);

    // DRAM buffer uses page_size = byte_size (whole-buffer), so derive the
    // per-tile DRAM stride directly from byte_size / num_tiles.
    ctx.per_tile_stride = static_cast<uint32_t>(ctx.byte_size / test_config.num_tiles);

    return ctx;
}

static bool verify_reader_datacopy_writer_output(const ReaderDatacopyWriterContext& ctx) {
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(ctx.output_dram_buffer, dest_buffer_data);
    return ctx.inputs == dest_buffer_data;
}

/// @brief Does Dram --> Reader --> CB --> Datacopy --> CB --> Writer --> Dram on a single core.
/// Uses the Metal 2.0 host API on every arch; on WH/BH the runtime selects the
/// Gen1 DM configs, on Quasar the Gen2 configs.
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_datacopy_writer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ReaderDatacopyWriterConfig& test_config) {
    auto ctx = setup_reader_datacopy_writer_context(mesh_device, test_config);
    auto* device = mesh_device->get_devices()[0];

    const bool is_quasar = device->arch() == ARCH::QUASAR;
    // On Quasar we can split work across two DM threads when num_tiles > 1;
    // WH/BH gen1 has one DM thread per processor, so the kernel only runs on
    // a single thread there.
    if (is_quasar && test_config.num_tiles > 1) {
        TT_FATAL(test_config.num_tiles % 2 == 0, "Number of tiles must be divisible by 2");
    }
    const uint32_t num_threads = (is_quasar && test_config.num_tiles > 1) ? 2u : 1u;
    const uint32_t per_core_tile_cnt = test_config.num_tiles / num_threads;
    const uint32_t num_tiles_per_thread = test_config.num_tiles / num_threads;

    const experimental::DFBSpecName INPUT_DFB{"input_dfb"};
    const experimental::DFBSpecName OUTPUT_DFB{"output_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    // Implicit sync is enabled by default for both DFBs (no DM kernel opts out
    // via Gen2Config::disable_implicit_sync_for). The program-level
    // reservation flag set below is independent of per-DFB sync mode.
    experimental::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
        .num_entries = static_cast<uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_input_data_format,
    };
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
        .num_entries = static_cast<uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_output_data_format,
    };

    // Both gen1 and gen2 configs are populated; the runtime picks the one
    // matching the active arch.
    experimental::DataMovementHardwareConfig reader_dm_cfg{
        .gen1_config =
            experimental::DataMovementHardwareConfig::Gen1Config{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default},
        .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{},
    };
    experimental::DataMovementHardwareConfig writer_dm_cfg{
        .gen1_config =
            experimental::DataMovementHardwareConfig::Gen1Config{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
        .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{},
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = num_threads,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = reader_dm_cfg,
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = num_threads,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = writer_dm_cfg,
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_2_0.cpp",
        .num_threads = num_threads,
        .dfb_bindings =
            {{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"per_core_tile_cnt", per_core_tile_cnt}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = test_config.node,
    };

    experimental::ProgramSpec spec{
        .name = "reader_datacopy_writer",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    log_info(tt::LogTest, "Num tiles per thread: {}", num_tiles_per_thread);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{test_config.node,
                  {{"src_addr", ctx.input_dram_byte_address},
                   {"src_bank_id", 0u},
                   {"num_tiles", num_tiles_per_thread},
                   {"dram_page_stride", ctx.per_tile_stride}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{test_config.node,
                  {{"dst_addr", ctx.output_dram_byte_address},
                   {"dst_bank_id", 0u},
                   {"num_tiles", num_tiles_per_thread},
                   {"dram_page_stride", ctx.per_tile_stride}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, params);

    tt_metal::detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    return verify_reader_datacopy_writer_output(ctx);
}
}  // namespace unit_tests::dram::direct

namespace tt::tt_metal {

TEST_F(MeshDeviceFixture, TensixSingleCoreDirectDramReaderOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        uint32_t l1_unreserved_base = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::L1);
        ASSERT_TRUE(
            unit_tests::dram::direct::reader_only(devices_.at(id), 1 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
        ASSERT_TRUE(
            unit_tests::dram::direct::reader_only(devices_.at(id), 2 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
        ASSERT_TRUE(
            unit_tests::dram::direct::reader_only(devices_.at(id), 16 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
    }
}
TEST_F(MeshDeviceFixture, TensixSingleCoreDirectDramWriterOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        uint32_t l1_unreserved_base = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::L1);
        ASSERT_TRUE(
            unit_tests::dram::direct::writer_only(devices_.at(id), 1 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
        ASSERT_TRUE(
            unit_tests::dram::direct::writer_only(devices_.at(id), 2 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
        ASSERT_TRUE(
            unit_tests::dram::direct::writer_only(devices_.at(id), 16 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
    }
}
TEST_F(MeshDeviceFixture, TensixSingleCoreDirectDramReaderWriter) {
    unit_tests::dram::direct::ReaderWriterConfig test_config = {
        .num_tiles = 1,
        .tile_byte_size = 2 * 32 * 32,
        .l1_data_format = tt::DataFormat::Float16_b,
        .node = experimental::NodeCoord(0, 0)};
    for (unsigned int id = 0; id < num_devices_; id++) {
        test_config.num_tiles = 1;
        ASSERT_TRUE(unit_tests::dram::direct::reader_writer(devices_.at(id), test_config));
        test_config.num_tiles = 4;
        ASSERT_TRUE(unit_tests::dram::direct::reader_writer(devices_.at(id), test_config));
        test_config.num_tiles = 8;
        ASSERT_TRUE(unit_tests::dram::direct::reader_writer(devices_.at(id), test_config));
    }
}
TEST_F(MeshDeviceFixture, TensixSingleCoreDirectDramReaderDatacopyWriter) {
    unit_tests::dram::direct::ReaderDatacopyWriterConfig test_config = {
        .num_tiles = 1,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .node = experimental::NodeCoord(0, 0)};
    for (unsigned int id = 0; id < num_devices_; id++) {
        if (devices_.at(id)->arch() != ARCH::QUASAR) { // TODO (#38092): Remove when we can run back to back tests on Quasar
            test_config.num_tiles = 1;
            ASSERT_TRUE(unit_tests::dram::direct::reader_datacopy_writer(devices_.at(id), test_config));
            test_config.num_tiles = 4;
            ASSERT_TRUE(unit_tests::dram::direct::reader_datacopy_writer(devices_.at(id), test_config));
        }
        test_config.num_tiles = 8;
        ASSERT_TRUE(unit_tests::dram::direct::reader_datacopy_writer(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
