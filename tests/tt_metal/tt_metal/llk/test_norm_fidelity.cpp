// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// An explicit fidelity F run under the opposite program fidelity must reproduce the program-fidelity API run
// at F bit for bit. The inputs set every bf16 significand bit, so LoFi, which multiplies only the top 5 of
// SrcA's 8 and the top 7 of SrcB's, gives a different result from HiFi4 and a swapped or dropped fidelity
// cannot pass.

#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "llk_device_fixture.hpp"

namespace tt::tt_metal::unit_tests::compute::norm_fidelity {
namespace {

constexpr std::uint32_t tile_elements = 1024;
constexpr auto LoFi = MathFidelity::LoFi;
constexpr auto HiFi2 = MathFidelity::HiFi2;
constexpr auto HiFi3 = MathFidelity::HiFi3;
constexpr auto HiFi4 = MathFidelity::HiFi4;

MathFidelity opposite(MathFidelity fidelity) { return fidelity == LoFi ? HiFi4 : LoFi; }

// Values 1 + k/2^bits with random k: bits=7 fills a bf16's explicit significand, bits=4 fits LoFi's SrcA.
std::vector<bfloat16> significand_values(std::uint32_t count, int bits, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> k(0, (1 << bits) - 1);
    std::vector<bfloat16> values(count);
    for (auto& value : values) {
        value = bfloat16(1.0f + std::ldexp(static_cast<float>(k(rng)), -bits));
    }
    return values;
}

struct KernelRun {
    const char* kernel;
    std::vector<std::uint32_t> compile_args;
    MathFidelity program_fidelity;
    bool fp32_dest;
    std::uint32_t num_tiles;
};

std::vector<float> run(
    distributed::MeshDevice& mesh,
    const KernelRun& config,
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b) {
    const CoreCoord core{0, 0};
    constexpr std::uint32_t input_tile_bytes = tile_elements * sizeof(bfloat16);
    const auto output_format = config.fp32_dest ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t output_tile_bytes = tile_elements * (config.fp32_dest ? sizeof(float) : sizeof(bfloat16));
    Program program = CreateProgram();
    auto& cq = mesh.mesh_command_queue();
    auto make_buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    for (const auto index : {tt::CBIndex::c_0, tt::CBIndex::c_1}) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(config.num_tiles * input_tile_bytes, {{index, tt::DataFormat::Float16_b}})
                .set_page_size(index, input_tile_bytes));
    }
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(output_tile_bytes, {{tt::CBIndex::c_16, output_format}})
            .set_page_size(tt::CBIndex::c_16, output_tile_bytes));

    auto input_a = make_buffer(config.num_tiles * input_tile_bytes);
    auto input_b = make_buffer(config.num_tiles * input_tile_bytes);
    auto output = make_buffer(output_tile_bytes);
    distributed::EnqueueWriteMeshBuffer(cq, input_a, pack_bfloat16_vec_into_uint32_vec(a), true);
    distributed::EnqueueWriteMeshBuffer(cq, input_b, pack_bfloat16_vec_into_uint32_vec(b), true);

    const auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {input_a->address(), 0, input_b->address(), 0, config.num_tiles});
    const auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {output->address(), 0, 1});
    CreateKernel(
        program,
        config.kernel,
        core,
        ComputeConfig{
            .math_fidelity = config.program_fidelity,
            .fp32_dest_acc_en = config.fp32_dest,
            .math_approx_mode = false,
            .compile_args = config.compile_args});

    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> packed;
    distributed::EnqueueReadMeshBuffer(cq, packed, output, true);
    std::vector<float> result;
    result.reserve(tile_elements);
    for (const auto word : packed) {
        if (config.fp32_dest) {
            result.push_back(std::bit_cast<float>(word));
        } else {
            result.push_back(static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(word))));
            result.push_back(static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(word >> 16))));
        }
    }
    return result;
}

constexpr const char* mul_reduce_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/norm_fidelity_mul_reduce.cpp";

struct MulReduceShape {
    std::uint32_t num_tiles;
    bool accumulate_in_one_tile;
    bool fp32_dest;
};

// explicit_fidelity=false runs the program-fidelity API, which ignores mul and reduce.
float mul_reduce(
    distributed::MeshDevice& mesh,
    const MulReduceShape& shape,
    bool explicit_fidelity,
    MathFidelity program_fidelity,
    MathFidelity mul,
    MathFidelity reduce,
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b) {
    const std::vector<std::uint32_t> args = {
        shape.num_tiles,
        static_cast<std::uint32_t>(explicit_fidelity),
        static_cast<std::uint32_t>(mul),
        static_cast<std::uint32_t>(reduce),
        static_cast<std::uint32_t>(shape.accumulate_in_one_tile),
        0,
        std::bit_cast<std::uint32_t>(1.0f),
        0};
    return run(mesh, {mul_reduce_kernel, args, program_fidelity, shape.fp32_dest, shape.num_tiles}, a, b).at(0);
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, MulReduceScalarTileFidelityMatchesProgramFidelity) {
    // Per-tile products take one DEST slot each, 4 in fp32 half-sync; one-tile accumulation runs past that.
    const MulReduceShape shapes[] = {{4, false, false}, {4, false, true}, {16, true, false}, {16, true, true}};
    for (const auto& shape : shapes) {
        SCOPED_TRACE(
            ::testing::Message() << "num_tiles=" << shape.num_tiles << ", one_tile=" << shape.accumulate_in_one_tile
                                 << ", fp32_dest=" << shape.fp32_dest);
        auto& mesh = *devices_.at(0);
        const auto fine = significand_values(shape.num_tiles * tile_elements, 7, 2026);
        // B = 1.0 is exact at any fidelity, so each phase's fidelity acts on A and the partial sums alone.
        const std::vector<bfloat16> ones(fine.size(), bfloat16(1.0f));

        for (const auto fidelity : {LoFi, HiFi2, HiFi4}) {
            SCOPED_TRACE(::testing::Message() << "fidelity=" << static_cast<int>(fidelity));
            const float program = mul_reduce(mesh, shape, false, fidelity, LoFi, LoFi, fine, ones);
            const float explicit_ = mul_reduce(mesh, shape, true, opposite(fidelity), fidelity, fidelity, fine, ones);
            EXPECT_EQ(explicit_, program);
        }
        EXPECT_NE(
            mul_reduce(mesh, shape, false, LoFi, LoFi, LoFi, fine, ones),
            mul_reduce(mesh, shape, false, HiFi4, LoFi, LoFi, fine, ones));

        // Each phase's fidelity reaches that phase. A multiply of 5-significand-bit values is exact at LoFi,
        // so LoFi there changes nothing; their sums need more bits, so LoFi on the reduce always shows.
        const auto coarse = significand_values(shape.num_tiles * tile_elements, 4, 2027);
        const float coarse_hifi4 = mul_reduce(mesh, shape, true, HiFi4, HiFi4, HiFi4, coarse, ones);
        EXPECT_EQ(mul_reduce(mesh, shape, true, HiFi4, LoFi, HiFi4, coarse, ones), coarse_hifi4);
        EXPECT_NE(mul_reduce(mesh, shape, true, HiFi4, HiFi4, LoFi, coarse, ones), coarse_hifi4);
        EXPECT_NE(
            mul_reduce(mesh, shape, true, HiFi4, LoFi, HiFi4, fine, ones),
            mul_reduce(mesh, shape, true, HiFi4, HiFi4, HiFi4, fine, ones));
    }
}

// The fast path of an RMSNorm: sum(x^2) reduced unscaled, then rsqrt(sum * (1/N) + eps) with 1/N applied at
// fp32 by add_rsqrt_tile's input_scale. N = 7168 makes 1/N one bf16 does not hold exactly. The bound leaves
// room for the reduce's bf16 accumulation; a dropped or doubled input_scale is off by a factor of sqrt(N).
TEST_F(LLKBlackholeSingleCardFixture, AddRsqrtTileInputScaleAfterOneTileReduce) {
    constexpr std::uint32_t num_tiles = 7;
    constexpr float epsilon = 1e-6f;
    const auto x = significand_values(num_tiles * tile_elements, 7, 7);
    const float input_scale = 1.0f / static_cast<float>(x.size());
    double sum_of_squares = 0.0;
    for (const auto value : x) {
        sum_of_squares += static_cast<double>(value) * static_cast<double>(value);
    }
    const double golden = 1.0 / std::sqrt(sum_of_squares * input_scale + epsilon);
    const std::vector<std::uint32_t> args = {
        num_tiles,
        1,
        static_cast<std::uint32_t>(HiFi2),
        static_cast<std::uint32_t>(HiFi2),
        1,
        1,
        std::bit_cast<std::uint32_t>(input_scale),
        std::bit_cast<std::uint32_t>(epsilon)};
    const auto result = run(*devices_.at(0), {mul_reduce_kernel, args, LoFi, false, num_tiles}, x, x);
    EXPECT_NEAR(result.at(0), golden, 0.02 * golden);
}

TEST_F(LLKBlackholeSingleCardFixture, MulTilesFidelityMatchesProgramFidelity) {
    auto& mesh = *devices_.at(0);
    const auto a = significand_values(tile_elements, 7, 11);
    // The broadcast reads only b[0]; 1.9921875 sets every significand bit, so LoFi's 7-bit SrcB drops one too.
    auto b = significand_values(tile_elements, 7, 13);
    b[0] = bfloat16(1.9921875f);
    auto mul = [&](bool explicit_fidelity, MathFidelity program_fidelity, MathFidelity fidelity, bool reuse_dest) {
        const std::vector<std::uint32_t> args = {
            static_cast<std::uint32_t>(explicit_fidelity),
            static_cast<std::uint32_t>(fidelity),
            static_cast<std::uint32_t>(reuse_dest)};
        return run(
            mesh,
            {"tests/tt_metal/tt_metal/test_kernels/compute/norm_fidelity_mul.cpp", args, program_fidelity, false, 1},
            a,
            b);
    };
    for (const bool reuse_dest : {false, true}) {
        SCOPED_TRACE(reuse_dest ? "mul_reuse_dest_tiles" : "mul_tiles_bcast_scalar");
        for (const auto fidelity : {LoFi, HiFi2, HiFi3, HiFi4}) {
            SCOPED_TRACE(::testing::Message() << "fidelity=" << static_cast<int>(fidelity));
            EXPECT_EQ(mul(true, opposite(fidelity), fidelity, reuse_dest), mul(false, fidelity, LoFi, reuse_dest));
        }
        EXPECT_NE(mul(false, LoFi, LoFi, reuse_dest), mul(false, HiFi4, LoFi, reuse_dest));
    }
}

}  // namespace tt::tt_metal::unit_tests::compute::norm_fidelity
