// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt/rtoptions.hpp"
#include "watcher_fixture.hpp"
#include "test_utils.hpp"
#include "llrt/llrt.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "hostdevcommon/common_runtime_address_map.h"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher NOC sanitization.
//////////////////////////////////////////////////////////////////////////////////////////
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

void RunDelayTestOnCore(WatcherDelayFixture* fixture, Device* device, CoreCoord &core) {
    auto program = tt_metal::CreateScopedProgram();

        const uint32_t SINGLE_TILE_SIZE = 2 * 1024;
        const uint32_t NUM_TILES = 4;
        const uint32_t DRAM_BUFFER_SIZE = SINGLE_TILE_SIZE * NUM_TILES;  // NUM_TILES of FP16_B, hard-coded in the reader/writer kernels
        const uint32_t PAGE_SIZE = DRAM_BUFFER_SIZE;

        tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = DRAM_BUFFER_SIZE,
                    .page_size = PAGE_SIZE,
                    .buffer_type = tt_metal::BufferType::DRAM
                    };

        auto src0_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
        auto src1_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
        auto dst_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * SINGLE_TILE_SIZE, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, SINGLE_TILE_SIZE);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * SINGLE_TILE_SIZE, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, SINGLE_TILE_SIZE);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = 16;  // output operands start at index 16
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * SINGLE_TILE_SIZE, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, SINGLE_TILE_SIZE);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        auto binary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = { };

        std::map<string, string> binary_defines = {
            { "ELTWISE_OP", "add_tiles" },
            { "ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD" }
        };
        auto eltwise_binary_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

        SetRuntimeArgs(
            program,
            eltwise_binary_kernel,
            core,
            {NUM_TILES, 1}
        );

        float constant = 0.0f;
        float start_from = 0.0f;
        std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(DRAM_BUFFER_SIZE, constant);
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(DRAM_BUFFER_SIZE, 0.0f);
        inc_populate(src1_vec, start_from);
        std::vector<uint32_t> expected_vec = create_constant_vector_of_bfloat16(DRAM_BUFFER_SIZE, 0.0f);
        inc_populate(expected_vec, start_from + constant);

        CommandQueue& cq = device->command_queue();

        EnqueueWriteBuffer(cq, std::ref(src0_dram_buffer), src0_vec, false);
        EnqueueWriteBuffer(cq, std::ref(src1_dram_buffer), src1_vec, false);

        vector<uint32_t> reader_args = {
            dram_buffer_src0_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            NUM_TILES,
            dram_buffer_src1_addr,
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            NUM_TILES,
            0};

        vector<uint32_t> writer_args = {
            dram_buffer_dst_addr, (std::uint32_t)dram_dst_noc_xy.x, (std::uint32_t)dram_dst_noc_xy.y, NUM_TILES
        };

        SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
        SetRuntimeArgs(program, binary_reader_kernel, core, reader_args);

        EnqueueProgram(cq, program, false);
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

        // Print the feedback generated by debug_delay functionality
        std::vector<uint32_t> read_vec;

        CoreCoord worker_core = fixture->delayed_cores[CoreType::WORKER][0]; // Just check that the first delayed core has the feedback set
        CoreCoord phys_core = device->physical_core_from_logical_core({0,0}, CoreType::WORKER);
        read_vec = tt::llrt::read_hex_vec_from_core (
            device->id(),
            phys_core,
            device->get_dev_addr(phys_core, HalL1MemAddrType::WATCHER) + offsetof(watcher_msg_t, debug_insert_delays),
            sizeof(debug_insert_delays_msg_t));

        log_info(tt::LogTest, "Read back debug_insert_delays: 0x{:x}", read_vec[0]);
        EXPECT_TRUE((read_vec[0] >> 24) == 0x3);
}

TEST_F(WatcherDelayFixture, TestWatcherSanitizeInsertDelays) {
    if (this->slow_dispatch_)
        GTEST_SKIP();

    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){
            CoreCoord core{0, 0};
            RunDelayTestOnCore(dynamic_cast<WatcherDelayFixture*>(fixture), device, core);
        },
        this->devices_[0]
    );
}
