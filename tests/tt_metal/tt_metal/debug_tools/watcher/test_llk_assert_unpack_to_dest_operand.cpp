// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
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
// feed the FPU, whose operands come from SrcA/SrcB. With LLK asserts enabled the SrcA/SrcB
// unpack inits (binary, matmul) reject such an operand and the watcher reports the tripped
// assert on TRISC0. Each case first runs the full op with no CB marked, which must finish with
// the watcher quiet and the exact expected output, so a trip in the marked run can only come
// from the operand guard. The marked run executes only the unpack init (unpack_ab_init_only.cpp):
// the trip halts the unpack thread before any data moves, and a full op would leave the math
// and pack threads stalled on it, which the next program on the core inherits.
//////////////////////////////////////////////////////////////////////////////////////////

using namespace tt;
using namespace tt::tt_metal;

namespace {

enum class FpuOp { Add, Matmul };

struct UnpackToDestOperandCase {
    const char* name;
    FpuOp op;
    CBIndex marked_cb;  // the operand marked UnpackToDestFp32 in the violating run
};

constexpr uint32_t kNumTiles = 1;
constexpr uint32_t kDatums = 32 * 32;
constexpr uint32_t kTileBytes = kDatums * sizeof(float);
constexpr uint32_t kBankId = 0;
constexpr uint32_t kPerCoreBlockSize = 1;
constexpr uint32_t kAccToDst = 0;
constexpr const char* kAddKernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary.cpp";
constexpr const char* kMatmulKernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp";
constexpr const char* kInitOnlyKernel = "tests/tt_metal/tt_metal/test_kernels/compute/unpack_ab_init_only.cpp";

const char* kernel_path(FpuOp op) { return op == FpuOp::Add ? kAddKernel : kMatmulKernel; }

// Inputs chosen so the expected output is exact in TF32 x TF32 -> FP32 arithmetic.
void make_inputs(FpuOp op, std::vector<float>& a, std::vector<float>& b, std::vector<float>& expected) {
    a.assign(kNumTiles * kDatums, 1.0f);
    b.assign(kNumTiles * kDatums, 0.5f);
    expected.resize(kNumTiles * kDatums);
    for (uint32_t i = 0; i < a.size(); ++i) {
        if (op == FpuOp::Add) {
            a[i] = static_cast<float>(i);
            expected[i] = a[i] + b[i];
        } else {
            // Every output datum of a 32x32 by 32x32 product of ones and halves is 32 * 0.5.
            expected[i] = 32 * 0.5f;
        }
    }
}

std::vector<uint32_t> as_words(const std::vector<float>& v) {
    std::vector<uint32_t> words(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        words[i] = std::bit_cast<uint32_t>(v[i]);
    }
    return words;
}

struct BuiltProgram {
    distributed::MeshWorkload workload;
    std::shared_ptr<distributed::MeshBuffer> src0;
    std::shared_ptr<distributed::MeshBuffer> src1;
    std::shared_ptr<distributed::MeshBuffer> dst;
};

// Float32 CBs c_0, c_1 -> c_16 on core (0,0), fp32 Dest. Unmarked: the full one-tile op with its reader
// and writer. Marked: only the op's unpack init, with `marked_cb` marked UnpackToDestFp32 for the SrcA/SrcB
// unpack init to reject; no data is read or written.
BuiltProgram build_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    FpuOp op,
    std::optional<CBIndex> marked_cb,
    const std::vector<float>& a,
    const std::vector<float>& b) {
    const CoreCoord core = {0, 0};
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    BuiltProgram built;
    built.workload.add_program(device_range, CreateProgram());
    auto& program = built.workload.get_programs().at(device_range);
    auto& cq = mesh_device->mesh_command_queue();

    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{.page_size = bytes, .buffer_type = BufferType::DRAM},
            mesh_device.get());
    };
    built.src0 = make_dram(kNumTiles * kTileBytes);
    built.src1 = make_dram(kNumTiles * kTileBytes);
    built.dst = make_dram(kNumTiles * kTileBytes);

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

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (marked_cb.has_value()) {
        unpack_to_dest_mode[*marked_cb] = UnpackToDestMode::UnpackToDestFp32;
        ComputeConfig init_only{.fp32_dest_acc_en = true, .unpack_to_dest_mode = unpack_to_dest_mode};
        if (op == FpuOp::Matmul) {
            init_only.defines = {{"MATMUL_INIT", "1"}};
        }
        CreateKernel(program, kInitOnlyKernel, core, init_only);
        return built;
    }

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

    ComputeConfig compute_config{.fp32_dest_acc_en = true, .unpack_to_dest_mode = unpack_to_dest_mode};
    if (op == FpuOp::Add) {
        compute_config.defines = {{"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}};
    } else {
        // matmul.cpp: block_tile_dim, dst_tile_rows, dst_tile_cols, block_cnt, in0/in1/out block tile counts.
        compute_config.compile_args = {1, 1, 1, 1, kNumTiles, kNumTiles, kNumTiles};
    }
    auto compute = CreateKernel(program, kernel_path(op), core, compute_config);

    std::vector<uint32_t> a_words = as_words(a);  // WriteShard takes the host vector by non-const reference
    std::vector<uint32_t> b_words = as_words(b);
    distributed::WriteShard(cq, built.src0, a_words, zero_coord);
    distributed::WriteShard(cq, built.src1, b_words, zero_coord);

    SetRuntimeArgs(program, reader, core, {built.src0->address(), kBankId, built.src1->address(), kBankId, kNumTiles});
    SetRuntimeArgs(program, writer, core, {built.dst->address(), kBankId, kNumTiles});
    if (op == FpuOp::Add) {
        SetRuntimeArgs(program, compute, core, {kNumTiles, kPerCoreBlockSize, kAccToDst});
    }
    return built;
}

uint32_t count_mismatches(const std::vector<uint32_t>& result, const std::vector<float>& expected) {
    uint32_t mismatches = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        mismatches += std::bit_cast<float>(result[i]) != expected[i];
    }
    return mismatches;
}

void RunUnpackToDestOperandCase(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const UnpackToDestOperandCase& test_case) {
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> expected;
    make_inputs(test_case.op, a, b, expected);

    // Control: no CB marked. The kernel runs with the same asserts compiled in and must finish with the
    // watcher quiet and the exact result, so nothing but the operand guard can trip in the marked run.
    {
        auto control = build_program(mesh_device, test_case.op, std::nullopt, a, b);
        fixture->RunProgram(mesh_device, control.workload, /*wait_for_dump=*/true);
        const std::string exception = MetalContext::instance().watcher_server()->exception_message();
        ASSERT_TRUE(exception.empty()) << "Control run (no CB marked) tripped the watcher: " << exception;
        std::vector<uint32_t> result;
        distributed::ReadShard(cq, result, control.dst, zero_coord);
        ASSERT_EQ(result.size(), expected.size());
        ASSERT_EQ(count_mismatches(result, expected), 0u)
            << "Control run (no CB marked) produced wrong data; first outputs " << std::bit_cast<float>(result[0])
            << ", " << std::bit_cast<float>(result[1]) << " (expected " << expected[0] << ", " << expected[1] << ")";
    }

    // Violation: the marked operand must be rejected by the SrcA/SrcB unpack init on TRISC0. An init that
    // returns instead lets the op compute on a source register the unpacker cannot deliver.
    auto violation = build_program(mesh_device, test_case.op, test_case.marked_cb, a, b);
    fixture->RunProgramExpectingWatcherError(mesh_device, violation.workload);

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

    ASSERT_FALSE(exception.empty()) << kernel_path(test_case.op) << " unpack init returned with CB "
                                    << static_cast<uint32_t>(test_case.marked_cb)
                                    << " marked UnpackToDestFp32 instead of tripping the LLK assert";
    log_info(tt::LogTest, "Watcher reported: {}", exception);
    EXPECT_NE(exception.find("tripped an assert"), std::string::npos) << exception;
    EXPECT_NE(exception.find("TRISC0"), std::string::npos)
        << "The assert did not trip on the unpack thread: " << exception;
    EXPECT_NE(exception.find(kInitOnlyKernel), std::string::npos)
        << "The assert did not trip in the compute kernel under test: " << exception;
}

// Watcher fixture with LLK asserts switched on before the devices open. The setting is part
// of the kernel compile hash, so the context re-initialises and the kernels are rebuilt with
// the asserts compiled in.
class MeshWatcherLlkAssertFixture : public MeshWatcherFixture,
                                    public ::testing::WithParamInterface<UnpackToDestOperandCase> {
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

}  // namespace

TEST_P(MeshWatcherLlkAssertFixture, TensixLlkAssertRejectsUnpackToDestFpuOperand) {
    if (this->arch_ == ARCH::QUASAR) {
        GTEST_SKIP() << "The SrcA/SrcB operand guard is a Wormhole/Blackhole LLK API check";
    }
    const auto& test_case = GetParam();
    // A deliberate assert stops the shared watcher server; run once per fixture, on one device.
    this->RunTestOnDevice(
        [&test_case](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunUnpackToDestOperandCase(fixture, mesh_device, test_case);
        },
        this->devices_[0]);
}

INSTANTIATE_TEST_SUITE_P(
    UnpackToDestOperand,
    MeshWatcherLlkAssertFixture,
    ::testing::Values(
        UnpackToDestOperandCase{"AddIn0Marked", FpuOp::Add, CBIndex::c_0},
        UnpackToDestOperandCase{"AddIn1Marked", FpuOp::Add, CBIndex::c_1},
        UnpackToDestOperandCase{"MatmulIn0Marked", FpuOp::Matmul, CBIndex::c_0},
        UnpackToDestOperandCase{"MatmulIn1Marked", FpuOp::Matmul, CBIndex::c_1}),
    [](const ::testing::TestParamInfo<UnpackToDestOperandCase>& info) { return std::string(info.param.name); });
