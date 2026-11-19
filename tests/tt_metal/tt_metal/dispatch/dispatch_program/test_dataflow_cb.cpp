// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/program/program_impl.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"

using namespace tt;
using namespace tt::tt_metal;

// Tests dataflow through CBs at indices 0, 8, 16, ... and topmost CB
// Validates data integrity: DRAM -> Reader -> CB -> Writer -> DRAM (src == dst)
TEST_F(UnitMeshAnyDispatchFixture, DataflowCb) {
    if (this->device().arch() == ARCH::QUASAR) {
        GTEST_SKIP() << "legacy circular buffers are not usable on Quasar: legacy kernels are rejected and Metal 2.0 "
                        "programs take DFBs";
    }
    Program program = CreateProgram(this->device());

    CoreCoord core = {0, 0};

    // Tile configuration
    constexpr uint32_t single_tile_size = 2 * 1024;
    constexpr uint32_t cb_capacity_tiles = 8;
    constexpr uint32_t tiles_to_transfer_per_cb = 200;
    constexpr uint32_t reader_ublock_size = 2;
    constexpr uint32_t writer_ublock_size = 4;

    static_assert(
        tiles_to_transfer_per_cb % writer_ublock_size == 0,
        "tiles_to_transfer_per_cb must be divisible by writer_ublock_size");

    // CB index configuration
    constexpr uint32_t start_cb = 0;
    constexpr uint32_t stride = 8;
    const uint32_t strided_cb_count = max_dfbs_ / stride;  // CBs at 0, 8, 16, ...
    const uint32_t topmost_cb = max_dfbs_ - 1;
    const uint32_t total_cbs = strided_cb_count + 1;  // Strided + topmost

    const uint32_t num_tiles = tiles_to_transfer_per_cb * total_cbs;
    const uint32_t dram_buffer_size = single_tile_size * num_tiles;

    auto& cq = this->device().mesh_command_queue();
    distributed::DeviceLocalBufferConfig dram_config{.page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig global_config{.size = dram_buffer_size};

    auto src_dram_buffer = distributed::MeshBuffer::create(global_config, dram_config, &this->device());
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = distributed::MeshBuffer::create(global_config, dram_config, &this->device());
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    auto create_cb = [&](uint32_t cb_index) {
        CircularBufferConfig cb_config =
            CircularBufferConfig(cb_capacity_tiles * single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_config);
    };

    // Create CBs at 0, 8, 16, ...
    for (uint32_t cb_idx = 0; cb_idx < max_dfbs_; cb_idx += stride) {
        create_cb(cb_idx);
    }
    // Also test topmost
    create_cb(topmost_cb);

    std::vector<uint32_t> reader_cb_compile_args = {
        start_cb, strided_cb_count, stride, topmost_cb, reader_ublock_size, /*use_dfbs=*/false};
    std::vector<uint32_t> writer_cb_compile_args = {
        start_cb, strided_cb_count, stride, topmost_cb, writer_ublock_size, /*use_dfbs=*/false};

    auto reader_cb_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_cb_test.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_cb_compile_args});

    auto writer_cb_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_cb_test.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_cb_compile_args});

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, /*blocking=*/true);

    SetRuntimeArgs(program, reader_cb_kernel, core, {dram_buffer_src_addr, 0, tiles_to_transfer_per_cb});
    SetRuntimeArgs(program, writer_cb_kernel, core, {dram_buffer_dst_addr, 0, tiles_to_transfer_per_cb});

    LaunchProgram(this->device(), std::move(program));

    std::vector<uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    // Validation
    EXPECT_EQ(src_vec, result_vec);
}

TEST_F(UnitMeshAnyDispatchFixture, DataflowDfb) {
    const NodeCoord node = {0, 0};

    constexpr uint32_t single_tile_size = 2 * 1024;
    constexpr uint32_t dfb_capacity_tiles = 8;
    constexpr uint32_t tiles_to_transfer_per_dfb = 200;
    constexpr uint32_t reader_ublock_size = 2;
    constexpr uint32_t writer_ublock_size = 4;

    static_assert(
        tiles_to_transfer_per_dfb % writer_ublock_size == 0,
        "tiles_to_transfer_per_dfb must be divisible by writer_ublock_size");

    const uint32_t total_dfbs = MetalContext::instance().hal().has_tile_counter_registers()
                                    ? std::min<uint32_t>(max_dfbs_, ::dfb::NUM_TENSIX_TILE_COUNTERS_FOR_DM)
                                    : max_dfbs_;
    const uint32_t dram_buffer_size = single_tile_size * tiles_to_transfer_per_dfb * total_dfbs;

    distributed::MeshCommandQueue& cq = this->device().mesh_command_queue();
    const distributed::DeviceLocalBufferConfig dram_config{
        .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    const distributed::ReplicatedBufferConfig global_config{.size = dram_buffer_size};
    std::shared_ptr<distributed::MeshBuffer> src_dram_buffer =
        distributed::MeshBuffer::create(global_config, dram_config, &this->device());
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer =
        distributed::MeshBuffer::create(global_config, dram_config, &this->device());

    auto explicit_sync_config = [](DataMovementProcessor processor, NOC noc) {
        return DataMovementHardwareConfig{
            .config_1xx = DataMovementHardwareConfig::DataMovement1XXConfig{.processor = processor, .noc = noc},
            .config_2xx = DataMovementHardwareConfig::DataMovement2XXConfig{.disable_dfb_implicit_sync_for_all = true}};
    };
    const KernelSpecName reader_name{"reader"};
    const KernelSpecName writer_name{"writer"};
    KernelSpec reader{
        .unique_id = reader_name,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dfb_test_2_0.cpp",
        .compile_time_args =
            {{"num_dfbs", total_dfbs},
             {"num_tiles_per_dfb", tiles_to_transfer_per_dfb},
             {"ublock_size_tiles", reader_ublock_size}},
        .runtime_arg_schema = {.common_runtime_arg_names = {"dram_address"}},
        .hw_config = explicit_sync_config(DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
    };
    KernelSpec writer{
        .unique_id = writer_name,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_dfb_test_2_0.cpp",
        .compile_time_args =
            {{"num_dfbs", total_dfbs},
             {"num_tiles_per_dfb", tiles_to_transfer_per_dfb},
             {"ublock_size_tiles", writer_ublock_size}},
        .runtime_arg_schema = {.common_runtime_arg_names = {"dram_address"}},
        .hw_config = explicit_sync_config(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };

    std::vector<DataflowBufferSpec> dfb_specs;
    for (uint32_t dfb_index = 0; dfb_index < total_dfbs; dfb_index++) {
        const std::string dfb_name = fmt::format("dfb_{}", dfb_index);
        dfb_specs.push_back(DataflowBufferSpec{
            .unique_id = DFBSpecName{dfb_name},
            .entry_size = single_tile_size,
            .num_entries = dfb_capacity_tiles,
            .data_format_metadata = tt::DataFormat::Float16_b,
        });
        reader.dfb_bindings.push_back(ProducerOf(DFBSpecName{dfb_name}, dfb_name));
        writer.dfb_bindings.push_back(StridedConsumerOf(DFBSpecName{dfb_name}, dfb_name));
    }

    const ProgramSpec spec{
        .name = "dataflow_dfb",
        .kernels = {reader, writer},
        .dataflow_buffers = dfb_specs,
        .work_units = {WorkUnitSpec{.name = "work_unit", .kernels = {reader_name, writer_name}, .target_nodes = node}},
    };
    Program program = Program(this->device(), spec);

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = reader_name, .common_runtime_arg_values = {{"dram_address", src_dram_buffer->address()}}},
        ProgramRunArgs::KernelRunArgs{
            .kernel = writer_name, .common_runtime_arg_values = {{"dram_address", dst_dram_buffer->address()}}},
    };
    program.set_program_run_args(run_args);

    const std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, /*blocking=*/true);

    LaunchProgram(this->device(), std::move(program));

    std::vector<uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    EXPECT_EQ(src_vec, result_vec);
}
