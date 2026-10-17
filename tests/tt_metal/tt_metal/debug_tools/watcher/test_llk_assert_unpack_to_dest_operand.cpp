// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <umd/device/types/arch.hpp>
#include "debug_tools_fixture.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include "hostdevcommon/kernel_structs.h"

//////////////////////////////////////////////////////////////////////////////////////////
// A CB marked UnpackToDestMode::UnpackToDestFp32 is unpacked straight into Dest and cannot
// feed the FPU, whose operands come from SrcA/SrcB. With LLK asserts enabled the binary-op
// unpack init rejects such an operand and the watcher reports the tripped assert.
//////////////////////////////////////////////////////////////////////////////////////////

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Watcher fixture with LLK asserts switched on before the devices open. The setting is part
// of the kernel compile hash, so the context re-initialises and the kernels are rebuilt with
// the asserts compiled in.
class MeshWatcherLlkAssertFixture : public MeshWatcherFixture {
protected:
    bool llk_asserts_previous_{};

    void SetUp() override {
        auto& rtoptions = MetalContext::instance().rtoptions();
        llk_asserts_previous_ = rtoptions.get_llk_asserts();
        rtoptions.set_llk_asserts(true);
        MeshWatcherFixture::SetUp();
    }

    void TearDown() override {
        MeshWatcherFixture::TearDown();
        MetalContext::instance().rtoptions().set_llk_asserts(llk_asserts_previous_);
    }
};

// add_tiles(c_0, c_1) with c_0 marked UnpackToDestFp32. The unpack init must trip the LLK
// assert; a kernel that runs to completion instead has added a SrcA operand the unpacker
// cannot deliver, and the output is reported for the record.
void RunAddTilesWithUnpackToDestOperand(
    MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr uint32_t kNumTiles = 1;
    constexpr uint32_t kDatums = 32 * 32;
    constexpr uint32_t kTileBytes = kDatums * sizeof(float);
    const CoreCoord core = {0, 0};

    distributed::MeshWorkload workload;
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, CreateProgram());
    auto& program = workload.get_programs().at(device_range);
    auto& cq = mesh_device->mesh_command_queue();

    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{.page_size = bytes, .buffer_type = BufferType::DRAM},
            mesh_device.get());
    };
    auto src0 = make_dram(kNumTiles * kTileBytes);
    auto src1 = make_dram(kNumTiles * kTileBytes);
    auto dst = make_dram(kNumTiles * kTileBytes);

    auto make_cb = [&](CBIndex index) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(kNumTiles * kTileBytes, {{index, tt::DataFormat::Float32}})
                .set_page_size(index, kTileBytes));
    };
    make_cb(CBIndex::c_0);
    make_cb(CBIndex::c_1);
    make_cb(CBIndex::c_16);

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // c_0 goes to Dest, c_1 to SrcB: the FPU add reads c_0 through SrcA, which the mode forbids.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    auto compute = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .defines = {{"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}}});

    std::vector<float> a(kNumTiles * kDatums);
    std::vector<float> b(kNumTiles * kDatums);
    for (uint32_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 0.5f;
    }
    auto as_words = [](const std::vector<float>& v) {
        std::vector<uint32_t> words(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            words[i] = std::bit_cast<uint32_t>(v[i]);
        }
        return words;
    };
    std::vector<uint32_t> a_words = as_words(a);
    std::vector<uint32_t> b_words = as_words(b);
    distributed::WriteShard(cq, src0, a_words, zero_coord);
    distributed::WriteShard(cq, src1, b_words, zero_coord);

    SetRuntimeArgs(program, reader, core, {src0->address(), 0u, src1->address(), 0u, kNumTiles});
    SetRuntimeArgs(program, writer, core, {dst->address(), 0u, kNumTiles});
    SetRuntimeArgs(program, compute, core, {kNumTiles, 1u, 0u});

    fixture->RunProgramExpectingWatcherError(mesh_device, workload);

    // Give the watcher one more poll to record the assert.
    std::string exception;
    constexpr auto timeout = std::chrono::milliseconds(5000);
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        exception = MetalContext::instance().watcher_server()->exception_message();
        if (!exception.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (exception.empty()) {
        std::vector<uint32_t> result;
        distributed::ReadShard(cq, result, dst, zero_coord);
        uint32_t mismatches = 0;
        for (size_t i = 0; i < result.size(); ++i) {
            mismatches += std::bit_cast<float>(result[i]) != a[i] + b[i];
        }
        FAIL() << "add_tiles ran to completion with an UnpackToDestFp32 operand instead of tripping the LLK assert; "
               << mismatches << " of " << result.size() << " output datums differ from a + b; first outputs "
               << std::bit_cast<float>(result[0]) << ", " << std::bit_cast<float>(result[1]) << ", "
               << std::bit_cast<float>(result[2]) << ", " << std::bit_cast<float>(result[3]) << " (expected "
               << a[0] + b[0] << ", " << a[1] + b[1] << ", " << a[2] + b[2] << ", " << a[3] + b[3] << ")";
    }
    log_info(tt::LogTest, "Watcher reported: {}", exception);
    EXPECT_NE(exception.find("tripped an assert"), std::string::npos) << exception;
}

}  // namespace

TEST_F(MeshWatcherLlkAssertFixture, TensixLlkAssertRejectsUnpackToDestFpuOperand) {
    if (this->arch_ == ARCH::QUASAR) {
        GTEST_SKIP() << "The SrcA/SrcB operand guard is a Wormhole/Blackhole LLK API check";
    }
    for (const auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(RunAddTilesWithUnpackToDestOperand, mesh_device);
    }
}
