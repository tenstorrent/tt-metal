// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for debug checkpoint and dump utilities using DEVICE_PRINT backend.
// Tests: single-core checkpoint, checkpoint in loop + dump_dest, global (cross-core)
// checkpoint, debug_dump_cb, debug_dump_l1, debug_dump_cb_typed.

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace dp_ckpt {
constexpr uint32_t INPUT_CB = 0;
constexpr uint32_t OUTPUT_CB = 16;
constexpr size_t NUM_TILES = 1;
constexpr tt::DataFormat FMT = tt::DataFormat::Float16_b;
constexpr uint32_t NUM_CORES = 2;
}  // namespace dp_ckpt

// Fixture: DevicePrintFixture with checkpoint enabled
class DevicePrintCheckpointTest : public DevicePrintFixture {
protected:
    void ExtraSetUp() override { tt::tt_metal::MetalContext::instance().rtoptions().set_checkpoint_enabled(true); }
    void ExtraTearDown() override { tt::tt_metal::MetalContext::instance().rtoptions().set_checkpoint_enabled(false); }
};

// Helper: create standard DRAM buffers, CBs, and run program on single core
struct SingleCoreSetup {
    std::shared_ptr<distributed::MeshBuffer> input_dram, output_dram;
    distributed::MeshWorkload workload;
    Program* program;
    distributed::MeshCoordinate zero{0, 0};
    CoreCoord core{0, 0};

    SingleCoreSetup(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        using namespace dp_ckpt;
        size_t tile_size = tt::tile_size(FMT);
        size_t buf_sz = NUM_TILES * tile_size;

        auto device_range = distributed::MeshCoordinateRange(zero, zero);
        Program prog = CreateProgram();
        workload.add_program(device_range, std::move(prog));
        program = &workload.get_programs().at(device_range);

        distributed::DeviceLocalBufferConfig lc = {.page_size = buf_sz, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig bc = {.size = buf_sz};
        input_dram = distributed::MeshBuffer::create(bc, lc, mesh_device.get());
        output_dram = distributed::MeshBuffer::create(bc, lc, mesh_device.get());

        CircularBufferConfig in_cfg =
            CircularBufferConfig(buf_sz, {{INPUT_CB, FMT}}).set_page_size(INPUT_CB, tile_size);
        CreateCircularBuffer(*program, core, in_cfg);
        CircularBufferConfig out_cfg =
            CircularBufferConfig(buf_sz, {{OUTPUT_CB, FMT}}).set_page_size(OUTPUT_CB, tile_size);
        CreateCircularBuffer(*program, core, out_cfg);
    }
};

// ======================= Single-core checkpoint =======================

static void run_basic_checkpoint(
    DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    using namespace dp_ckpt;
    SingleCoreSetup s(mesh_device);
    auto& cq = mesh_device->mesh_command_queue();

    auto reader = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_checkpoint.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {INPUT_CB, 0}});
    auto writer = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_checkpoint.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {OUTPUT_CB, 0}});
    CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_checkpoint.cpp",
        s.core,
        ComputeConfig{.compile_args = {static_cast<uint32_t>(NUM_TILES)}});

    SetRuntimeArgs(
        *s.program,
        reader,
        s.core,
        {static_cast<uint32_t>(s.input_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});
    SetRuntimeArgs(
        *s.program,
        writer,
        s.core,
        {static_cast<uint32_t>(s.output_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});

    auto input =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, NUM_TILES * 1024);
    distributed::WriteShard(cq, s.input_dram, input, s.zero);
    fixture->RunProgram(mesh_device, s.workload);

    std::vector<uint32_t> output;
    distributed::ReadShard(cq, output, s.output_dram, s.zero);
    EXPECT_EQ(input, output);
    EXPECT_TRUE(FileContainsAllStrings(fixture->dprint_file_name, {"=== CKPT 1 CBs ===", "CB0 sz=*"}));
}

TEST_F(DevicePrintCheckpointTest, BasicCheckpoint) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) { run_basic_checkpoint(f, d); },
        this->devices_[0]);
}

// ======================= Checkpoint loop + dump_dest =======================

static void run_checkpoint_loop(
    DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    using namespace dp_ckpt;
    SingleCoreSetup s(mesh_device);
    auto& cq = mesh_device->mesh_command_queue();

    auto reader = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_checkpoint_loop.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {INPUT_CB, 0}});
    auto writer = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_checkpoint_loop.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {OUTPUT_CB, 0}});
    CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_checkpoint_loop.cpp",
        s.core,
        ComputeConfig{.compile_args = {static_cast<uint32_t>(NUM_TILES)}});

    SetRuntimeArgs(
        *s.program,
        reader,
        s.core,
        {static_cast<uint32_t>(s.input_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});
    SetRuntimeArgs(
        *s.program,
        writer,
        s.core,
        {static_cast<uint32_t>(s.output_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});

    auto input =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, NUM_TILES * 1024);
    distributed::WriteShard(cq, s.input_dram, input, s.zero);
    fixture->RunProgram(mesh_device, s.workload);

    std::vector<uint32_t> output;
    distributed::ReadShard(cq, output, s.output_dram, s.zero);
    EXPECT_EQ(input, output);
    EXPECT_TRUE(FileContainsAllStrings(fixture->dprint_file_name, {"=== CKPT 1 CBs ===", "=== CKPT 2 dest regs ==="}));
}

TEST_F(DevicePrintCheckpointTest, CheckpointLoopAndDumpDest) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) { run_checkpoint_loop(f, d); },
        this->devices_[0]);
}

// ======================= Global (cross-core) checkpoint =======================

static void run_global_checkpoint(
    DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    using namespace dp_ckpt;
    auto* device = mesh_device->get_devices()[0];
    size_t tile_size = tt::tile_size(FMT);
    size_t buf_sz = NUM_TILES * tile_size;

    CoreCoord core0 = {0, 0}, core1 = {0, 1};
    CoreRange cores(core0, core1);
    CoreCoord barrier_coord = device->worker_core_from_logical_core(core0);

    distributed::MeshWorkload workload;
    auto zero = distributed::MeshCoordinate(0, 0);
    auto dr = distributed::MeshCoordinateRange(zero, zero);
    Program prog = CreateProgram();
    workload.add_program(dr, std::move(prog));
    auto& program = workload.get_programs().at(dr);
    auto& cq = mesh_device->mesh_command_queue();

    uint32_t sem_id = CreateSemaphore(program, cores, 0);

    distributed::DeviceLocalBufferConfig lc = {.page_size = buf_sz, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig bc = {.size = buf_sz};
    auto in0 = distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    auto out0 = distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    auto in1 = distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    auto out1 = distributed::MeshBuffer::create(bc, lc, mesh_device.get());

    for (auto c : {core0, core1}) {
        CreateCircularBuffer(
            program, c, CircularBufferConfig(buf_sz, {{INPUT_CB, FMT}}).set_page_size(INPUT_CB, tile_size));
        CreateCircularBuffer(
            program, c, CircularBufferConfig(buf_sz, {{OUTPUT_CB, FMT}}).set_page_size(OUTPUT_CB, tile_size));
    }

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_global_checkpoint.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {INPUT_CB, 0}});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_global_checkpoint.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {OUTPUT_CB, 0}});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_global_checkpoint.cpp",
        cores,
        ComputeConfig{
            .compile_args = {static_cast<uint32_t>(NUM_TILES), sem_id, barrier_coord.x, barrier_coord.y, NUM_CORES}});

    for (uint32_t i = 0; i < NUM_CORES; i++) {
        CoreCoord c = (i == 0) ? core0 : core1;
        auto in = (i == 0) ? in0 : in1;
        auto out = (i == 0) ? out0 : out1;
        SetRuntimeArgs(
            program,
            reader,
            c,
            {static_cast<uint32_t>(in->address()),
             0u,
             static_cast<uint32_t>(NUM_TILES),
             sem_id,
             barrier_coord.x,
             barrier_coord.y,
             NUM_CORES});
        SetRuntimeArgs(
            program,
            writer,
            c,
            {static_cast<uint32_t>(out->address()),
             0u,
             static_cast<uint32_t>(NUM_TILES),
             sem_id,
             barrier_coord.x,
             barrier_coord.y,
             NUM_CORES});
    }

    auto data0 =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, NUM_TILES * 1024);
    auto data1 =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, NUM_TILES * 1024);
    distributed::WriteShard(cq, in0, data0, zero);
    distributed::WriteShard(cq, in1, data1, zero);
    fixture->RunProgram(mesh_device, workload);

    std::vector<uint32_t> o0, o1;
    distributed::ReadShard(cq, o0, out0, zero);
    distributed::ReadShard(cq, o1, out1, zero);
    EXPECT_EQ(data0, o0);
    EXPECT_EQ(data1, o1);
    EXPECT_TRUE(FileContainsAllStrings(fixture->dprint_file_name, {"=== CKPT 1 CBs ===", "CB0 sz=*"}));
}

TEST_F(DevicePrintCheckpointTest, GlobalCheckpoint) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) { run_global_checkpoint(f, d); },
        this->devices_[0]);
}

// ======================= Standalone dump: debug_dump_cb =======================

static void run_dump_cb(DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    using namespace dp_ckpt;
    SingleCoreSetup s(mesh_device);
    auto& cq = mesh_device->mesh_command_queue();

    auto reader = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {INPUT_CB, 0}});
    auto writer = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {OUTPUT_CB, 0}});
    CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_dump_cb.cpp",
        s.core,
        ComputeConfig{.compile_args = {static_cast<uint32_t>(NUM_TILES)}});

    SetRuntimeArgs(
        *s.program,
        reader,
        s.core,
        {static_cast<uint32_t>(s.input_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});
    SetRuntimeArgs(
        *s.program,
        writer,
        s.core,
        {static_cast<uint32_t>(s.output_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});

    auto input =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, NUM_TILES * 1024);
    distributed::WriteShard(cq, s.input_dram, input, s.zero);
    fixture->RunProgram(mesh_device, s.workload);

    std::vector<uint32_t> output;
    distributed::ReadShard(cq, output, s.output_dram, s.zero);
    EXPECT_EQ(input, output);
    EXPECT_TRUE(FileContainsAllStrings(fixture->dprint_file_name, {"CB0 sz=*", "[0]*"}));
}

TEST_F(DevicePrintCheckpointTest, DumpCB) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) { run_dump_cb(f, d); },
        this->devices_[0]);
}

// ======================= Standalone dump: debug_dump_l1 =======================

static void run_dump_l1(DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    using namespace dp_ckpt;
    SingleCoreSetup s(mesh_device);
    auto& cq = mesh_device->mesh_command_queue();

    auto reader = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {INPUT_CB, 0}});
    auto writer = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {OUTPUT_CB, 0}});
    CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_dump_all.cpp",
        s.core,
        ComputeConfig{.compile_args = {static_cast<uint32_t>(NUM_TILES)}});

    SetRuntimeArgs(
        *s.program,
        reader,
        s.core,
        {static_cast<uint32_t>(s.input_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});
    SetRuntimeArgs(
        *s.program,
        writer,
        s.core,
        {static_cast<uint32_t>(s.output_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});

    auto input =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, NUM_TILES * 1024);
    distributed::WriteShard(cq, s.input_dram, input, s.zero);
    fixture->RunProgram(mesh_device, s.workload);

    std::vector<uint32_t> output;
    distributed::ReadShard(cq, output, s.output_dram, s.zero);
    EXPECT_EQ(input, output);
    EXPECT_TRUE(FileContainsAllStrings(fixture->dprint_file_name, {"CB0 sz=*", "L1[*"}));
}

TEST_F(DevicePrintCheckpointTest, DumpL1) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) { run_dump_l1(f, d); },
        this->devices_[0]);
}

// ======================= Standalone dump: debug_dump_cb_typed =======================

static void run_dump_typed(DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    using namespace dp_ckpt;
    SingleCoreSetup s(mesh_device);
    auto& cq = mesh_device->mesh_command_queue();

    auto reader = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {INPUT_CB, 0}});
    auto writer = CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {OUTPUT_CB, 0}});
    CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_dump_typed.cpp",
        s.core,
        ComputeConfig{.compile_args = {static_cast<uint32_t>(NUM_TILES)}});

    SetRuntimeArgs(
        *s.program,
        reader,
        s.core,
        {static_cast<uint32_t>(s.input_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});
    SetRuntimeArgs(
        *s.program,
        writer,
        s.core,
        {static_cast<uint32_t>(s.output_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});

    auto input =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, NUM_TILES * 1024);
    distributed::WriteShard(cq, s.input_dram, input, s.zero);
    fixture->RunProgram(mesh_device, s.workload);

    std::vector<uint32_t> output;
    distributed::ReadShard(cq, output, s.output_dram, s.zero);
    EXPECT_EQ(input, output);
    EXPECT_TRUE(FileContainsAllStrings(fixture->dprint_file_name, {"CB0 tile 0 (typed):*"}));
}

TEST_F(DevicePrintCheckpointTest, DumpCBTyped) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) { run_dump_typed(f, d); },
        this->devices_[0]);
}
