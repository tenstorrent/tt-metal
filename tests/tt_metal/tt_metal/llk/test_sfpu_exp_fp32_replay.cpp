// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include <impl/context/metal_context.hpp>

#include "llk_device_fixture.hpp"

// A/B test of the fp32-accurate exp_tile on Blackhole: the recorded replay form (default build) against the sfpi
// form (DISABLE_SFPLOADMACRO), bit for bit, over fp32 inputs that cover every exponent with both signs (zeros,
// subnormals, infinities and NaNs included), the Juffa breakpoints with their ulp neighbours and the rounding ties
// of round(a / ln2), unscaled and with a runtime input scale (scale_en). The sfpi form is also held against the host:
// within 2e-6 relative on [-87, 88], +Inf above the overflow threshold, +0 below the underflow threshold and for
// -Inf, NaN for NaN.
//
// One core runs reader_unary -> eltwise_sfpu (exp_tile_init(); exp_tile(0);) -> writer_unary with Float32 CBs; the
// input takes the unpack-to-dest path so every input bit pattern reaches DST unchanged.
//
// TensixSfpuExpFp32MatmulAfterExp runs matmul -> exp -> matmul under one matmul_init: the replayed exp must leave
// the FPU's replay slots, where matmul keeps its recorded MVMUL sequence, intact.

namespace tt::tt_metal {

namespace unit_tests::compute::sfpu::exp_fp32_replay {

constexpr uint32_t kTileHW = 32 * 32;
constexpr uint32_t kTileBytes = kTileHW * sizeof(float);
constexpr uint32_t kNumTiles = 64;
constexpr uint32_t kNumValues = kNumTiles * kTileHW;
constexpr uint32_t kCbTiles = 2;
constexpr uint32_t kDramBankId = 0;  // single-bank DRAM buffers, as reader_unary and writer_unary address them
constexpr size_t kMaxReportedMismatches = 8;
constexpr uint32_t kPerCoreBlockDim = 1;  // eltwise_sfpu compile arg 1: tiles per block
// Runtime input scales of the scale_en cases, as fp16b: 1.5 and -0.5.
constexpr uint16_t kScalePos = 0x3FC0;
constexpr uint16_t kScaleNeg = 0xBF00;

constexpr float kLn2 = std::numbers::ln2_v<float>;
constexpr float kExpOverflow = 88.72283905f;     // ln(FLT_MAX)
constexpr float kExpUnderflow = -87.33654475f;   // ln(FLT_MIN): the kernel flushes every result below it to +0
constexpr float kExpSubnormal = -103.97207708f;  // ln(smallest subnormal)

float nudge(float f, int ulps) { return std::bit_cast<float>(std::bit_cast<int32_t>(f) + ulps); }

// Deterministic fp32 stimulus: special values, breakpoints, multiples and half-multiples of ln2, every exponent
// and sign with random mantissas, then uniform values over the finite range of the function.
std::vector<uint32_t> make_stimulus(uint32_t seed) {
    std::vector<uint32_t> v;
    v.reserve(kNumValues);
    auto push_bits = [&](uint32_t b) { v.push_back(b); };
    auto push_f = [&](float f) { v.push_back(std::bit_cast<uint32_t>(f)); };

    for (uint32_t b :
         {0x00000000u,
          0x80000000u,
          0x7F800000u,
          0xFF800000u,
          0x7FC00000u,
          0xFFC00000u,
          0x7F800001u,
          0xFF800001u,
          0x7FBFFFFFu,
          0xFFBFFFFFu,
          0x00000001u,
          0x80000001u,
          0x007FFFFFu,
          0x807FFFFFu,
          0x00800000u,
          0x80800000u,
          0x7F7FFFFFu,
          0xFF7FFFFFu}) {
        push_bits(b);
    }
    for (float f :
         {kExpOverflow,
          88.0f,
          89.0f,
          kExpUnderflow,
          -87.3f,
          -87.4f,
          kExpSubnormal,
          -103.9f,
          -104.0f,
          0.5f * kLn2,
          -0.5f * kLn2,
          kLn2,
          -kLn2,
          1.0f,
          -1.0f,
          1e-20f,
          -1e-20f}) {
        for (int ulps = -2; ulps <= 2; ++ulps) {
            push_f(nudge(f, ulps));
        }
    }
    for (int k = -150; k <= 130; ++k) {
        for (int ulps = -1; ulps <= 1; ++ulps) {
            push_f(nudge(static_cast<float>(k) * kLn2, ulps));
            push_f(nudge((static_cast<float>(k) + 0.5f) * kLn2, ulps));
        }
    }
    std::mt19937 rng(seed);
    constexpr uint32_t kPerExponentAndSign = 78;
    for (uint32_t e = 0; e < 256; ++e) {
        for (uint32_t s = 0; s < 2; ++s) {
            for (uint32_t n = 0; n < kPerExponentAndSign; ++n) {
                push_bits((s << 31) | (e << 23) | (rng() & 0x7FFFFFu));
            }
        }
    }
    std::uniform_real_distribution<float> dense(-110.0f, 95.0f);
    while (v.size() < kNumValues) {
        push_f(dense(rng));
    }
    v.resize(kNumValues);
    return v;
}

// exp_tile over the input on one core, unscaled or with scale_en and the given fp16b scale; plain_path builds the
// kernel with DISABLE_SFPLOADMACRO.
std::vector<uint32_t> run_exp(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::vector<uint32_t>& input,
    bool plain_path,
    std::optional<uint16_t> scale = std::nullopt) {
    auto& cq = mesh_device->mesh_command_queue();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    const CoreCoord core = {0, 0};

    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{
                .page_size = bytes, .buffer_type = BufferType::DRAM, .bottom_up = false},
            mesh_device.get());
    };
    auto in_buffer = make_dram(kNumTiles * kTileBytes);
    auto out_buffer = make_dram(kNumTiles * kTileBytes);

    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(kCbTiles * kTileBytes, {{CBIndex::c_0, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_0, kTileBytes));
    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(kCbTiles * kTileBytes, {{CBIndex::c_16, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_16, kTileBytes));

    auto reader_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    const std::string chain =
        scale ? fmt::format("exp_tile_init(); exp_tile<false, true>(0, VectorMode::RC, 0x{:04x});", *scale)
              : std::string("exp_tile_init(); exp_tile(0);");
    std::map<std::string, std::string> defines = {{"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", chain}};
    if (plain_path) {
        defines["DISABLE_SFPLOADMACRO"] = "1";
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = {kNumTiles, kPerCoreBlockDim},
            .defines = defines});

    std::vector<uint32_t> in_host = input;
    distributed::WriteShard(cq, in_buffer, in_host, zero_coord);
    SetRuntimeArgs(
        program_, reader_kernel, core, {static_cast<uint32_t>(in_buffer->address()), kDramBankId, kNumTiles});
    SetRuntimeArgs(
        program_, writer_kernel, core, {static_cast<uint32_t>(out_buffer->address()), kDramBankId, kNumTiles});
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> output;
    distributed::ReadShard(cq, output, out_buffer, zero_coord);
    return output;
}

// The sfpi form against the host, per the kernel's documented contract; with a scale the kernel's input is the
// fp32 product of the value and the fp16b scale.
size_t count_host_mismatches(
    const std::vector<uint32_t>& input,
    const std::vector<uint32_t>& output,
    std::optional<uint16_t> scale = std::nullopt) {
    const float s = scale ? std::bit_cast<float>(static_cast<uint32_t>(*scale) << 16) : 1.0f;
    size_t mismatches = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        const float x = std::bit_cast<float>(input[i]) * s;
        const float y = std::bit_cast<float>(output[i]);
        bool ok = true;
        if (std::isnan(x)) {
            ok = std::isnan(y);
        } else if (x > 88.8f) {
            ok = output[i] == 0x7F800000u;
        } else if (x < -87.4f) {  // below ln(FLT_MIN), with a margin for rounding at the threshold
            ok = output[i] == 0u;
        } else if (x >= -87.0f && x <= 88.0f) {
            const double ref = std::exp(static_cast<double>(x));
            ok = std::fabs(static_cast<double>(y) - ref) <= 2e-6 * ref;
        }
        if (!ok) {
            if (mismatches < kMaxReportedMismatches) {
                log_error(tt::LogTest, "exp host mismatch at {}: x={:.9g} device={:.9g}", i, x, y);
            }
            ++mismatches;
        }
    }
    return mismatches;
}

}  // namespace unit_tests::compute::sfpu::exp_fp32_replay

namespace unit_tests::compute::sfpu::exp_fp32_replay {

// Replay against sfpi, bit for bit, and sfpi against the host.
void check_replay_against_sfpi(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, std::optional<uint16_t> scale) {
    const auto input = make_stimulus(/*seed=*/20261001);
    const auto plain = run_exp(mesh_device, input, /*plain_path=*/true, scale);
    const auto replay = run_exp(mesh_device, input, /*plain_path=*/false, scale);
    ASSERT_EQ(plain.size(), input.size());
    ASSERT_EQ(replay.size(), input.size());

    size_t ab_mismatches = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        if (plain[i] != replay[i]) {
            if (ab_mismatches < kMaxReportedMismatches) {
                log_error(
                    tt::LogTest,
                    "exp A/B mismatch at {}: x={:.9g} (0x{:08x}) sfpi=0x{:08x} replay=0x{:08x}",
                    i,
                    std::bit_cast<float>(input[i]),
                    input[i],
                    plain[i],
                    replay[i]);
            }
            ++ab_mismatches;
        }
    }
    log_info(
        tt::LogTest,
        "exp fp32 A/B over {} values, scale {}: {} mismatches",
        input.size(),
        scale ? fmt::format("0x{:04x}", *scale) : std::string("none"),
        ab_mismatches);
    EXPECT_EQ(ab_mismatches, 0u);
    EXPECT_EQ(count_host_mismatches(input, plain, scale), 0u);
}

// matmul_exp_matmul.cpp on one core; returns its two output tiles.
std::vector<uint32_t> run_matmul_exp_matmul(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::vector<uint32_t>& a,
    const std::vector<uint32_t>& b,
    bool exp_between) {
    auto& cq = mesh_device->mesh_command_queue();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    const CoreCoord core = {0, 0};

    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{
                .page_size = bytes, .buffer_type = BufferType::DRAM, .bottom_up = false},
            mesh_device.get());
    };
    auto a_buffer = make_dram(kTileBytes);
    auto b_buffer = make_dram(kTileBytes);
    auto out_buffer = make_dram(2 * kTileBytes);

    for (auto [cb, tiles] : {std::pair{CBIndex::c_0, 1u}, std::pair{CBIndex::c_1, 1u}, std::pair{CBIndex::c_16, 2u}}) {
        CreateCircularBuffer(
            program_,
            core,
            CircularBufferConfig(tiles * kTileBytes, {{cb, tt::DataFormat::Float32}}).set_page_size(cb, kTileBytes));
    }
    auto reader_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::map<std::string, std::string> defines;
    if (exp_between) {
        defines["EXP_BETWEEN"] = "1";
    }
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul_exp_matmul.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .defines = defines});

    std::vector<uint32_t> a_host = a;
    std::vector<uint32_t> b_host = b;
    distributed::WriteShard(cq, a_buffer, a_host, zero_coord);
    distributed::WriteShard(cq, b_buffer, b_host, zero_coord);
    SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {static_cast<uint32_t>(a_buffer->address()),
         kDramBankId,
         static_cast<uint32_t>(b_buffer->address()),
         kDramBankId,
         1u});
    SetRuntimeArgs(program_, writer_kernel, core, {static_cast<uint32_t>(out_buffer->address()), kDramBankId, 2u});
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> output;
    distributed::ReadShard(cq, output, out_buffer, zero_coord);
    return output;
}

}  // namespace unit_tests::compute::sfpu::exp_fp32_replay

TEST_F(LLKBlackholeSingleCardFixture, TensixSfpuExpFp32ReplayBitExact) {
    using namespace unit_tests::compute::sfpu::exp_fp32_replay;
    if (MetalContext::instance().rtoptions().get_disable_sfploadmacro()) {
        GTEST_SKIP() << "TT_METAL_DISABLE_SFPLOADMACRO builds both forms as sfpi; the A/B comparison would be vacuous";
    }
    check_replay_against_sfpi(this->devices_.at(0), std::nullopt);
}

TEST_F(LLKBlackholeSingleCardFixture, TensixSfpuExpFp32ReplayBitExactScaled) {
    using namespace unit_tests::compute::sfpu::exp_fp32_replay;
    if (MetalContext::instance().rtoptions().get_disable_sfploadmacro()) {
        GTEST_SKIP() << "TT_METAL_DISABLE_SFPLOADMACRO builds both forms as sfpi; the A/B comparison would be vacuous";
    }
    check_replay_against_sfpi(this->devices_.at(0), kScalePos);
    check_replay_against_sfpi(this->devices_.at(0), kScaleNeg);
}

TEST_F(LLKBlackholeSingleCardFixture, TensixSfpuExpFp32MatmulAfterExp) {
    using namespace unit_tests::compute::sfpu::exp_fp32_replay;
    // Entries k / 8 with k in [-2, 2]: every product and every 32-term sum is exact in fp32, at any fidelity.
    std::mt19937 rng(20261021);
    std::uniform_int_distribution<int> k(-2, 2);
    std::vector<uint32_t> a(kTileHW);
    std::vector<uint32_t> b(kTileHW);
    for (auto* m : {&a, &b}) {
        for (auto& w : *m) {
            w = std::bit_cast<uint32_t>(static_cast<float>(k(rng)) / 8.0f);
        }
    }
    const auto reference = run_matmul_exp_matmul(this->devices_.at(0), a, b, /*exp_between=*/false);
    const auto with_exp = run_matmul_exp_matmul(this->devices_.at(0), a, b, /*exp_between=*/true);
    ASSERT_EQ(reference.size(), 2 * kTileHW);
    ASSERT_EQ(with_exp.size(), 2 * kTileHW);

    size_t nonzero = 0;
    size_t product_mismatches = 0;
    size_t exp_mismatches = 0;
    for (size_t i = 0; i < kTileHW; ++i) {
        const float p = std::bit_cast<float>(reference[i]);
        nonzero += p != 0.0f;
        // The first product of the reference run, unchanged by the second one and by an exp in between.
        if (reference[kTileHW + i] != reference[i] || with_exp[kTileHW + i] != reference[i]) {
            if (product_mismatches < kMaxReportedMismatches) {
                log_error(
                    tt::LogTest,
                    "product after exp, element {}: expected 0x{:08x}, got 0x{:08x}",
                    i,
                    reference[i],
                    with_exp[kTileHW + i]);
            }
            ++product_mismatches;
        }
        const double e = std::exp(static_cast<double>(p));
        if (std::fabs(static_cast<double>(std::bit_cast<float>(with_exp[i])) - e) > 2e-6 * e) {
            ++exp_mismatches;
        }
    }
    log_info(
        tt::LogTest,
        "matmul -> exp -> matmul: {} product mismatches, {} exp mismatches ({} nonzero products)",
        product_mismatches,
        exp_mismatches,
        nonzero);
    EXPECT_GT(nonzero, kTileHW / 2);
    EXPECT_EQ(product_mismatches, 0u);
    EXPECT_EQ(exp_mismatches, 0u);
}

}  // namespace tt::tt_metal
