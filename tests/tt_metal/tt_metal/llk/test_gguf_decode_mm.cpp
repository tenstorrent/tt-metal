// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute API of the Blackhole GGUF decoders and the DEST-fed matmul (api/compute/experimental/gguf_decode.h,
// dest_srca_mm.h), through test_kernels/compute/gguf_decode_mm.cpp. The decoders' arithmetic is tested bit for bit in
// tt-llk; these tests check the API around it on Q8_0 and Q4_0 pages whose codes are one value per page and whose
// scales are one value per column, so the expected weights need no page layout beyond the scale row's column order.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace {

constexpr std::uint32_t kQ8_0 = 1;
constexpr std::uint32_t kQ4_0 = 2;

struct Case {
    std::uint32_t fmt = kQ8_0;
    std::uint32_t k_tiles = 1;
    std::uint32_t in0_rows = 32;
    bool mm = true;
    bool overlap = false;
    bool split = true;
};

float bits_to_float(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

std::uint32_t float_to_bits(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

// Index of element (r, c) of a 32-column tile stored as 16x16 faces.
std::uint32_t face_index(std::uint32_t r, std::uint32_t c) {
    return (r / 16) * 512 + (c / 16) * 256 + (r % 16) * 16 + c % 16;
}

struct Inputs {
    std::vector<std::uint32_t> pages;  // k_tiles pages
    std::vector<std::uint32_t> in0;    // k_tiles bf16 tiles, two values per word
    std::vector<float> weights;        // [k_tiles * 32][32]
    std::vector<float> activations;    // [32][k_tiles * 32]
};

// Page bytes and code bytes of the two formats (ckernel_gguf_page.h page_layout): Q8_0 2 KiB with 1 KiB of codes,
// Q4_0 1 KiB with 512 bytes; the scale row (fp16 d of column 16 p + 2 c + b at half b of word 2 c + p) follows them.
std::uint32_t page_bytes(std::uint32_t fmt) { return fmt == kQ8_0 ? 2048 : 1024; }
std::uint32_t code_bytes(std::uint32_t fmt) { return fmt == kQ8_0 ? 1024 : 512; }

Inputs make_inputs(const Case& c, std::mt19937& rng) {
    Inputs in;
    const std::uint32_t words = page_bytes(c.fmt) / 4;
    const std::uint32_t k = c.k_tiles * 32;
    in.pages.assign(c.k_tiles * words, 0);
    in.weights.assign(k * 32, 0.0f);
    std::vector<std::uint32_t> d(32);
    std::vector<float> d_value(32);
    for (std::uint32_t n = 0; n < 32; n++) {
        // Normal fp16, 2^-6 .. 2^5, random sign and mantissa.
        const std::uint32_t exp = 9 + rng() % 12;
        d[n] = (rng() & 0x8000u) | (exp << 10) | (rng() & 0x3FFu);
        d_value[n] = std::ldexp((d[n] & 0x8000u) ? -1.0f : 1.0f, static_cast<int>(exp) - 25) * (1024 + (d[n] & 0x3FFu));
    }
    for (std::uint32_t kt = 0; kt < c.k_tiles; kt++) {
        std::uint32_t* page = in.pages.data() + kt * words;
        std::uint32_t code_byte;
        float code_value;
        if (c.fmt == kQ8_0) {
            code_byte = 1 + rng() % 255;  // no zero codes: the decoder's sign of a zero is not ggml's
            code_value = static_cast<float>(static_cast<std::int8_t>(code_byte));
        } else {
            const std::uint32_t q = (9 + rng() % 15) & 0xFu;  // not 8 (zero)
            code_byte = q | q << 4;
            code_value = static_cast<float>(static_cast<int>(q) - 8);
        }
        for (std::uint32_t w = 0; w < code_bytes(c.fmt) / 4; w++) {
            page[w] = code_byte * 0x01010101u;
        }
        for (std::uint32_t w = 0; w < 16; w++) {
            const std::uint32_t col = 16 * (w % 2) + 2 * (w / 2);
            page[code_bytes(c.fmt) / 4 + w] = d[col] | d[col + 1] << 16;
        }
        for (std::uint32_t r = 0; r < 32; r++) {
            for (std::uint32_t n = 0; n < 32; n++) {
                in.weights[(kt * 32 + r) * 32 + n] = code_value * d_value[n];
            }
        }
    }
    // in0: [16, 32] tiles for in0_rows 8 / 16, [32, 32] for 32; bf16 values 2^-3 .. 2^3.
    const std::uint32_t tile_rows = c.in0_rows == 32 ? 32 : 16;
    in.activations.assign(32 * k, 0.0f);
    std::vector<std::uint16_t> in0_bf16(c.k_tiles * tile_rows * 32, 0);
    for (std::uint32_t kt = 0; kt < c.k_tiles; kt++) {
        for (std::uint32_t r = 0; r < tile_rows; r++) {
            for (std::uint32_t col = 0; col < 32; col++) {
                const std::uint32_t exp = 124 + rng() % 7;
                const std::uint16_t bits = static_cast<std::uint16_t>((rng() & 0x8000u) | (exp << 7) | (rng() & 0x7Fu));
                in0_bf16[kt * tile_rows * 32 + face_index(r, col)] = bits;
                in.activations[r * k + kt * 32 + col] = bits_to_float(static_cast<std::uint32_t>(bits) << 16);
            }
        }
    }
    in.in0.resize(in0_bf16.size() / 2);
    for (std::size_t i = 0; i < in.in0.size(); i++) {
        in.in0[i] = in0_bf16[2 * i] | static_cast<std::uint32_t>(in0_bf16[2 * i + 1]) << 16;
    }
    return in;
}

// The output tiles (Float32 bits) of one run of gguf_decode_mm.cpp.
std::vector<std::uint32_t> run(distributed::MeshDevice& mesh, const Case& c, const Inputs& in) {
    const CoreCoord core{0, 0};
    Program program = CreateProgram(mesh);
    const std::uint32_t pbytes = page_bytes(c.fmt);
    const Tile page_tile(pbytes == 1024 ? std::array<uint32_t, 2>{16, 16} : std::array<uint32_t, 2>{16, 32});
    const Tile in0_tile(c.in0_rows == 32 ? std::array<uint32_t, 2>{32, 32} : std::array<uint32_t, 2>{16, 32});
    const Tile out_tile;
    const std::uint32_t in0_bytes = in0_tile.get_tile_size(tt::DataFormat::Float16_b);
    const std::uint32_t out_tiles = c.mm ? 1 : c.k_tiles;
    auto buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    auto page_buffer = buffer(c.k_tiles * pbytes);
    auto in0_buffer = buffer(c.k_tiles * in0_bytes);
    auto out_buffer = buffer(out_tiles * 4096);
    auto make_cb = [&](tt::CBIndex index, tt::DataFormat format, const Tile& tile, std::uint32_t depth) {
        const auto bytes = tile.get_tile_size(format);
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(depth * bytes, {{index, format}})
                .set_page_size(index, bytes)
                .set_tile_dims(index, tile));
    };
    make_cb(tt::CBIndex::c_0, tt::DataFormat::UInt32, page_tile, c.k_tiles);
    make_cb(tt::CBIndex::c_1, tt::DataFormat::Float16_b, in0_tile, c.k_tiles);
    make_cb(tt::CBIndex::c_16, tt::DataFormat::Float32, out_tile, 2);
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
    auto compute = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/gguf_decode_mm.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .defines = {
                {"GGUF_FMT", std::to_string(c.fmt)},
                {"GGUF_IN0_ROWS", std::to_string(c.in0_rows)},
                {"GGUF_MM", std::to_string(c.mm)},
                {"GGUF_OVERLAP", std::to_string(c.overlap)},
                {"GGUF_SPLIT", std::to_string(c.split)}}});
    auto& cq = mesh.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, page_buffer, in.pages, true);
    distributed::EnqueueWriteMeshBuffer(cq, in0_buffer, in.in0, true);
    SetRuntimeArgs(program, reader, core, {page_buffer->address(), 0, in0_buffer->address(), 0, c.k_tiles});
    SetRuntimeArgs(program, writer, core, {out_buffer->address(), 0, out_tiles});
    SetRuntimeArgs(program, compute, core, {c.k_tiles});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, out_buffer, true);
    return result;
}

std::string name(const Case& c) {
    return std::string(c.fmt == kQ8_0 ? "Q8_0" : "Q4_0") + " k_tiles=" + std::to_string(c.k_tiles) +
           " in0_rows=" + std::to_string(c.in0_rows) + " overlap=" + std::to_string(c.overlap) +
           " split=" + std::to_string(c.split);
}

}  // namespace

// Each decoded tile equals code * d, exactly.
TEST_F(LLKBlackholeSingleCardFixture, GgufDecodeTileExact) {
    std::mt19937 rng(1);
    for (const auto fmt : {kQ8_0, kQ4_0}) {
        const Case c{.fmt = fmt, .k_tiles = 3, .mm = false};
        SCOPED_TRACE(name(c));
        const Inputs in = make_inputs(c, rng);
        const auto result = run(*devices_.at(0), c, in);
        ASSERT_EQ(result.size(), c.k_tiles * 1024u);
        for (std::uint32_t kt = 0; kt < c.k_tiles; kt++) {
            for (std::uint32_t r = 0; r < 32; r++) {
                for (std::uint32_t n = 0; n < 32; n++) {
                    ASSERT_EQ(result[kt * 1024 + face_index(r, n)], float_to_bits(in.weights[(kt * 32 + r) * 32 + n]))
                        << "k tile " << kt << " row " << r << " column " << n;
                }
            }
        }
    }
}

// dest_srca_mm against a host matmul of the same weights truncated to TF32 (MOVD2A), within the FPU's summation
// error; the pack-thread decode (overlap) and the one-step page copy give the same bits as the reference run.
TEST_F(LLKBlackholeSingleCardFixture, GgufDestSrcaMatmul) {
    std::mt19937 rng(2);
    for (const auto fmt : {kQ8_0, kQ4_0}) {
        for (const std::uint32_t k_tiles : {1u, 2u, 8u}) {
            for (const std::uint32_t in0_rows : {8u, 16u, 32u}) {
                const Case c{.fmt = fmt, .k_tiles = k_tiles, .in0_rows = in0_rows};
                SCOPED_TRACE(name(c));
                const Inputs in = make_inputs(c, rng);
                const auto result = run(*devices_.at(0), c, in);
                ASSERT_EQ(result.size(), 1024u);
                const std::uint32_t k = k_tiles * 32;
                for (std::uint32_t r = 0; r < in0_rows; r++) {
                    for (std::uint32_t n = 0; n < 32; n++) {
                        double sum = 0.0, bound = 0.0;
                        for (std::uint32_t i = 0; i < k; i++) {
                            const float w = bits_to_float(float_to_bits(in.weights[i * 32 + n]) & 0xFFFFE000u);
                            const double p = static_cast<double>(in.activations[r * k + i]) * w;
                            sum += p;
                            bound += std::fabs(p);
                        }
                        const double got = bits_to_float(result[face_index(r, n)]);
                        // The FPU sums each lane of 8 products in fixed point at 10 fractional bits of its largest
                        // exponent, then adds lanes and DEST in fp32.
                        ASSERT_LE(std::fabs(got - sum), bound * std::ldexp(1.0, -9))
                            << "row " << r << " column " << n << ": " << got << " vs " << sum;
                    }
                }
                Case overlap = c;
                overlap.overlap = true;
                EXPECT_EQ(run(*devices_.at(0), overlap, in), result) << "overlap on vs off";
                if (in0_rows == 32) {
                    Case whole = c;
                    whole.split = false;
                    EXPECT_EQ(run(*devices_.at(0), whole, in), result) << "page_to_dest vs split copy";
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
