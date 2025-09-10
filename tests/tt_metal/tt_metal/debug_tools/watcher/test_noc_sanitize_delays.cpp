// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "umd/device/tt_core_coordinates.h"
#include <tt-metalium/utils.hpp>

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher NOC sanitization.
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal;

// Incrementally populate a vector with bfloat16 values starting from a given float value
// and incrementing by 1.0f for each element.
void inc_populate(std::vector<std::uint32_t>& vec, float start_from) {
    float val = start_from;
    for (std::uint32_t i = 0; i < vec.size(); i++) {
        bfloat16 num_1_bfloat16 = bfloat16(val);
        val = val + 1.0f;
        bfloat16 num_2_bfloat16 = bfloat16(val);
        val = val + 1.0f;
        vec.at(i) = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }
}

void RunDelayTestOnCore(
    MeshWatcherDelayFixture* fixture, std::shared_ptr<distributed::MeshDevice> mesh_device, CoreCoord& core) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    auto& program_ = workload.get_programs().at(device_range);
    auto device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();

    const uint32_t SINGLE_TILE_SIZE = 2 * 1024;
    const uint32_t NUM_TILES = 4;
    const uint32_t DRAM_BUFFER_SIZE =
        SINGLE_TILE_SIZE * NUM_TILES;  // NUM_TILES of FP16_B, hard-coded in the reader/writer kernels
    const uint32_t PAGE_SIZE = DRAM_BUFFER_SIZE;

    distributed::DeviceLocalBufferConfig dram_config{.page_size = PAGE_SIZE, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = DRAM_BUFFER_SIZE};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * SINGLE_TILE_SIZE, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, SINGLE_TILE_SIZE);
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(num_input_tiles * SINGLE_TILE_SIZE, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, SINGLE_TILE_SIZE);
    tt_metal::CreateCircularBuffer(program_, core, cb_src1_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * SINGLE_TILE_SIZE, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, SINGLE_TILE_SIZE);
    tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

    auto binary_reader_kernel = tt_metal::CreateKernel(
        program_,
        "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program_,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {};

    std::map<std::string, std::string> binary_defines = {
        {"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}};
    auto eltwise_binary_kernel = tt_metal::CreateKernel(
        program_,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

    SetRuntimeArgs(program_, eltwise_binary_kernel, core, {NUM_TILES, 1});

    float constant = 0.0f;
    float start_from = 0.0f;
    std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(DRAM_BUFFER_SIZE, constant);
    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(DRAM_BUFFER_SIZE, 0.0f);
    inc_populate(src1_vec, start_from);
    std::vector<uint32_t> expected_vec = create_constant_vector_of_bfloat16(DRAM_BUFFER_SIZE, 0.0f);
    inc_populate(expected_vec, start_from + constant);

    distributed::WriteShard(cq, src0_dram_buffer, src0_vec, zero_coord);
    distributed::WriteShard(cq, src1_dram_buffer, src1_vec, zero_coord);

    vector<uint32_t> reader_args = {
        dram_buffer_src0_addr, (std::uint32_t)0, NUM_TILES, dram_buffer_src1_addr, (std::uint32_t)0, NUM_TILES, 0};

    vector<uint32_t> writer_args = {dram_buffer_dst_addr, (std::uint32_t)0, NUM_TILES};

    SetRuntimeArgs(program_, unary_writer_kernel, core, writer_args);
    SetRuntimeArgs(program_, binary_reader_kernel, core, reader_args);

    fixture->RunProgram(mesh_device, workload);
    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, zero_coord);

    // Print the feedback generated by debug_delay functionality
    CoreCoord virtual_core = device->virtual_core_from_logical_core({0, 0}, CoreType::WORKER);
    auto dev_msgs_factory = tt::tt_metal::MetalContext::instance().hal().get_dev_msgs_factory(
        tt::tt_metal::HalProgrammableCoreType::TENSIX);
    auto msg = dev_msgs_factory.create<dev_msgs::debug_insert_delays_msg_t>();
    uint64_t read_addr =
        device->get_dev_addr(virtual_core, HalL1MemAddrType::WATCHER) +
        dev_msgs_factory.offset_of<dev_msgs::watcher_msg_t>(dev_msgs::watcher_msg_t::Field::debug_insert_delays);
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        msg.data(), msg.size(), {device->id(), virtual_core}, read_addr);

    log_info(
        tt::LogTest,
        "Read back debug_insert_delays: {:x},{:x},{:x},{:x}",
        msg.view().read_delay_processor_mask(),
        msg.view().write_delay_processor_mask(),
        msg.view().atomic_delay_processor_mask(),
        msg.view().feedback());
    EXPECT_TRUE(msg.view().feedback() == 0x3);
}

TEST_F(MeshWatcherDelayFixture, TensixTestWatcherSanitizeInsertDelays) {
    if (this->slow_dispatch_)
        GTEST_SKIP();

    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, std::shared_ptr<distributed::MeshDevice> mesh_device) {
            CoreCoord core{0, 0};
            RunDelayTestOnCore(dynamic_cast<MeshWatcherDelayFixture*>(fixture), mesh_device, core);
        },
        this->devices_[0]);
}
