// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// srca_forced_exp.h: a matmul of bf16 in0 with BFP2_b mantissas read at exponent 127 (-1, 0, +1), then, after
// srca_forced_exp_off and a SrcA reconfig, with a bf16 tile, in one kernel (test_kernels/compute/srca_forced_exp.cpp).

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace {

std::uint32_t face_index(std::uint32_t r, std::uint32_t c) {
    return (r / 16) * 512 + (c / 16) * 256 + (r % 16) * 16 + c % 16;
}

float bf16_value(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint16_t random_bf16(std::mt19937& rng) {
    return static_cast<std::uint16_t>((rng() & 0x8000u) | ((124 + rng() % 7) << 7) | (rng() & 0x7Fu));
}

// |got - in0 x w| within the FPU's summation error, for every output element.
void check_matmul(
    const std::vector<std::uint32_t>& out,
    std::uint32_t tile,
    const std::vector<float>& a,
    const std::vector<float>& w,
    const char* what) {
    for (std::uint32_t m = 0; m < 32; m++) {
        for (std::uint32_t n = 0; n < 32; n++) {
            double sum = 0.0, bound = 0.0;
            for (std::uint32_t k = 0; k < 32; k++) {
                const double p = static_cast<double>(a[m * 32 + k]) * w[k * 32 + n];
                sum += p;
                bound += std::fabs(p);
            }
            float got;
            std::memcpy(&got, &out[tile * 1024 + face_index(m, n)], 4);
            ASSERT_LE(std::fabs(got - sum), bound * std::ldexp(1.0, -9))
                << what << ": row " << m << " column " << n << ": " << got << " vs " << sum;
        }
    }
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, SrcaForcedExpBfp2ThenBf16) {
    std::mt19937 rng(7);
    std::vector<float> a(1024), w1(1024), w2(1024);
    std::vector<std::uint16_t> in0(1024), bf16(1024);
    std::vector<std::uint8_t> mant(256, 0);
    for (std::uint32_t r = 0; r < 32; r++) {
        for (std::uint32_t c = 0; c < 32; c++) {
            in0[face_index(r, c)] = random_bf16(rng);
            a[r * 32 + c] = bf16_value(in0[face_index(r, c)]);
            bf16[face_index(r, c)] = random_bf16(rng);
            w2[r * 32 + c] = bf16_value(bf16[face_index(r, c)]);
            // BFP2 code: sign in bit 1, magnitude in bit 0 (0 -> +0, 1 -> +1, 3 -> -1; 2 would read as -inf).
            const std::uint32_t code = std::array<std::uint32_t, 3>{0, 1, 3}[rng() % 3];
            w1[r * 32 + c] = std::array<float, 4>{0.0f, 1.0f, 0.0f, -1.0f}[code];
            const std::uint32_t e = face_index(r, c);
            mant[e / 4] |= static_cast<std::uint8_t>(code << (2 * (e % 4)));
        }
    }
    std::vector<std::uint32_t> input(512 + 64 + 512);
    std::memcpy(input.data(), in0.data(), 2048);
    std::memcpy(input.data() + 512, mant.data(), 256);
    std::memcpy(input.data() + 576, bf16.data(), 2048);

    distributed::MeshDevice& mesh = *devices_.at(0);
    const CoreCoord core{0, 0};
    Program program = CreateProgram(mesh);
    auto buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    auto in_buffer = buffer(input.size() * 4);
    auto out_buffer = buffer(2 * 4096);
    auto cb = [&](tt::CBIndex index, tt::DataFormat format, std::uint32_t page, std::uint32_t pages) {
        CreateCircularBuffer(
            program, core, CircularBufferConfig(page * pages, {{index, format}}).set_page_size(index, page));
    };
    cb(tt::CBIndex::c_0, tt::DataFormat::Float16_b, 2048, 1);
    cb(tt::CBIndex::c_1, tt::DataFormat::Bfp4_b, 256, 1);  // as the GGUF TQ2_0 matmul's weight tiles
    cb(tt::CBIndex::c_2, tt::DataFormat::Float16_b, 2048, 1);
    cb(tt::CBIndex::c_16, tt::DataFormat::Float32, 4096, 2);
    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/srca_forced_exp_reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/srca_forced_exp.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
    SetRuntimeArgs(program, reader, core, {in_buffer->address(), 2048, 256, 2048});
    SetRuntimeArgs(program, writer, core, {out_buffer->address(), 0, 2});
    auto& cq = mesh.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, in_buffer, input, true);
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> out;
    distributed::EnqueueReadMeshBuffer(cq, out, out_buffer, true);
    ASSERT_EQ(out.size(), 2048u);
    check_matmul(out, 0, a, w1, "forced exponent");
    check_matmul(out, 1, a, w2, "after srca_forced_exp_off");
}

}  // namespace tt::tt_metal
