// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute API of the Blackhole GGUF decoders and dest_srca_mm (test_kernels/compute/gguf_decode_mm.cpp) on pages with
// one code per page and one scale per column; the decoders' arithmetic is tested bit for bit in tt-llk.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "llk_device_fixture.hpp"
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/experimental/ckernel_gguf_page.h"

namespace tt::tt_metal {
namespace {

constexpr std::uint32_t kQ8_0 = ckernel::gguf::Q8_0;
constexpr std::uint32_t kQ4_0 = ckernel::gguf::Q4_0;
constexpr std::uint32_t kMXFP4 = ckernel::gguf::MXFP4;
constexpr std::uint32_t kQ4_K = ckernel::gguf::Q4_K;

struct Case {
    std::uint32_t fmt = kQ8_0;
    std::uint32_t k_tiles = 1;
    std::uint32_t in0_rows = 32;
    bool mm = true;
    bool overlap = false;
    bool split = true;
    bool full_sync = true;
    bool group_copy = false;
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

ckernel::gguf::PageLayout layout(std::uint32_t fmt) {
    return ckernel::gguf::page_layout(static_cast<ckernel::gguf::Format>(fmt));
}

// c_0 page bytes: MXFP4's records as two pages of 576 bytes (the second takes the first's scale row), else the page.
std::uint32_t page_bytes(std::uint32_t fmt) {
    return fmt == kMXFP4 ? layout(fmt).scale_offset + 64 : layout(fmt).rows * 64;
}

// Scale-row word w holds columns lane_col(w) and lane_col(w) + 1.
std::uint32_t lane_col(std::uint32_t w) { return 16 * (w % 2) + 2 * (w / 2); }

// fp16 bits of a normal value with exponent field exp and a random mantissa and sign; and its value.
std::uint32_t random_fp16(std::mt19937& rng, std::uint32_t exp, float& value) {
    const std::uint32_t h = (rng() & 0x8000u) | (exp << 10) | (rng() & 0x3FFu);
    value = std::ldexp((h & 0x8000u) ? -1.0f : 1.0f, static_cast<int>(exp) - 25) * (1024 + (h & 0x3FFu));
    return h;
}

// MXFP4 pages: E2M1 code q (sign bit 3) as the page's block-float code, and its value.
std::uint32_t fp4_page_code(std::uint32_t q) {
    return (q & 8) | std::array<std::uint32_t, 8>{0, 1, 4, 5, 6, 7, 2, 3}[q & 7];
}
float fp4_value(std::uint32_t q) {
    return ((q & 8) ? -1.0f : 1.0f) * std::array<float, 8>{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f}[q & 7];
}

Inputs make_inputs(const Case& c, std::mt19937& rng) {
    Inputs in;
    const auto L = layout(c.fmt);
    const std::uint32_t words = page_bytes(c.fmt) / 4, code_words = L.code_bytes / 4, scale_word = L.scale_offset / 4;
    const std::uint32_t k = c.k_tiles * 32;
    in.pages.assign(c.k_tiles * words, 0);
    in.weights.assign(k * 32, 0.0f);
    // Per column: Q8_0 / Q4_0 d; Q4_K d and dmin; MXFP4 the E8M0 scales of the group's two K tiles.
    std::vector<std::uint32_t> d(32), dmin(32), e(64);
    std::vector<float> d_value(32), dmin_value(32);
    std::uint32_t sc = 0, m = 0;
    for (std::uint32_t kt = 0; kt < c.k_tiles; kt++) {
        const std::uint32_t t = kt % L.group_tiles;
        if (t == 0) {
            for (std::uint32_t n = 0; n < 32; n++) {
                d[n] = random_fp16(rng, c.fmt == kQ4_K ? 13 + rng() % 3 : 9 + rng() % 12, d_value[n]);
                dmin[n] = random_fp16(rng, 13 + rng() % 3, dmin_value[n]);
                e[n] = 120 + rng() % 15;
                e[32 + n] = 120 + rng() % 15;
            }
            sc = 1 + rng() % 15;
            m = rng() % 16;
        }
        std::uint32_t* page = in.pages.data() + kt * words;
        std::uint32_t code_byte = 0;
        std::vector<float> value(32);
        if (c.fmt == kQ8_0) {
            code_byte = 1 + rng() % 255;  // no zero codes: the decoder's sign of a zero is not ggml's
            for (std::uint32_t n = 0; n < 32; n++) {
                value[n] = static_cast<float>(static_cast<std::int8_t>(code_byte)) * d_value[n];
            }
        } else if (c.fmt == kQ4_0) {
            const std::uint32_t q = (9 + rng() % 15) & 0xFu;  // not 8 (zero)
            code_byte = q | q << 4;
            for (std::uint32_t n = 0; n < 32; n++) {
                value[n] = static_cast<float>(static_cast<int>(q) - 8) * d_value[n];
            }
        } else if (c.fmt == kQ4_K) {
            const std::uint32_t q = 1 + rng() % 15;
            code_byte = q | q << 4;
            for (std::uint32_t n = 0; n < 32; n++) {
                value[n] = static_cast<float>(
                    static_cast<double>(d_value[n]) * sc * q - static_cast<double>(dmin_value[n]) * m);
            }
        } else {  // MXFP4: no zero magnitudes
            const std::uint32_t q = (rng() & 8) | (1 + rng() % 7);
            const std::uint32_t pc = fp4_page_code(q);
            code_byte = pc | pc << 4;
            for (std::uint32_t n = 0; n < 32; n++) {
                value[n] = std::ldexp(fp4_value(q), static_cast<int>(e[32 * t + n]) - 127);
            }
        }
        for (std::uint32_t w = 0; w < code_words; w++) {
            page[w] = code_byte * 0x01010101u;
        }
        for (std::uint32_t w = 0; w < 16; w++) {
            const std::uint32_t col = lane_col(w);
            if (c.fmt == kMXFP4) {
                if (t == 0) {  // the record's scale row, after the first page's codes
                    page[scale_word + w] = e[col] | e[col + 1] << 8 | e[32 + col] << 16 | e[32 + col + 1] << 24;
                }
            } else if (c.fmt == kQ4_K) {  // 12 scale bytes per column (rows 0..2, next column 3..5), d, dmin
                const std::array<std::uint32_t, 3> fields = {sc, m, sc | m << 4};
                for (std::uint32_t r = 0; r < 6; r++) {
                    page[scale_word + 16 * r + w] = fields[r % 3] * 0x01010101u;
                }
                page[scale_word + 16 * 6 + w] = d[col] | d[col + 1] << 16;
                page[scale_word + 16 * 7 + w] = dmin[col] | dmin[col + 1] << 16;
            } else {
                page[scale_word + w] = d[col] | d[col + 1] << 16;
            }
        }
        for (std::uint32_t r = 0; r < 32; r++) {
            for (std::uint32_t n = 0; n < 32; n++) {
                in.weights[(kt * 32 + r) * 32 + n] = value[n];
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
    const Tile page_tile(layout(c.fmt).rows == 16 ? std::array<uint32_t, 2>{16, 16} : std::array<uint32_t, 2>{16, 32});
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
    auto make_cb =
        [&](tt::CBIndex index, tt::DataFormat format, const Tile& tile, std::uint32_t depth, std::uint32_t page = 0) {
            const auto bytes = page ? page : tile.get_tile_size(format);
            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(depth * bytes, {{index, format}})
                    .set_page_size(index, bytes)
                    .set_tile_dims(index, tile));
        };
    make_cb(tt::CBIndex::c_0, tt::DataFormat::UInt32, page_tile, c.k_tiles, pbytes);
    make_cb(tt::CBIndex::c_1, tt::DataFormat::Float16_b, in0_tile, c.k_tiles);
    make_cb(tt::CBIndex::c_16, tt::DataFormat::Float32, out_tile, 2);
    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/gguf_pages_reader.cpp",
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
            .dst_full_sync_en = c.full_sync,
            .defines = {
                {"GGUF_FMT", std::to_string(c.fmt)},
                {"GGUF_IN0_ROWS", std::to_string(c.in0_rows)},
                {"GGUF_MM", std::to_string(c.mm)},
                {"GGUF_OVERLAP", std::to_string(c.overlap)},
                {"GGUF_SPLIT", std::to_string(c.split)},
                {"GGUF_GROUP_COPY", std::to_string(c.group_copy)}}});
    auto& cq = mesh.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, page_buffer, in.pages, true);
    distributed::EnqueueWriteMeshBuffer(cq, in0_buffer, in.in0, true);
    SetRuntimeArgs(
        program, reader, core, {page_buffer->address(), c.k_tiles * pbytes, in0_buffer->address(), c.k_tiles});
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
    std::string fmt = "Q4_K";
    if (c.fmt == kQ8_0) {
        fmt = "Q8_0";
    } else if (c.fmt == kQ4_0) {
        fmt = "Q4_0";
    } else if (c.fmt == kMXFP4) {
        fmt = "MXFP4";
    }
    return fmt + " k_tiles=" + std::to_string(c.k_tiles) + " in0_rows=" + std::to_string(c.in0_rows) +
           " overlap=" + std::to_string(c.overlap) + " split=" + std::to_string(c.split) +
           " full_sync=" + std::to_string(c.full_sync) + " group_copy=" + std::to_string(c.group_copy);
}

// Each decoded tile equals its weights exactly (a zero of either sign).
void check_decoded(const Case& c, const Inputs& in, const std::vector<std::uint32_t>& result) {
    ASSERT_EQ(result.size(), c.k_tiles * 1024u);
    for (std::uint32_t kt = 0; kt < c.k_tiles; kt++) {
        for (std::uint32_t r = 0; r < 32; r++) {
            for (std::uint32_t n = 0; n < 32; n++) {
                const std::uint32_t got = result[kt * 1024 + face_index(r, n)];
                const float want = in.weights[(kt * 32 + r) * 32 + n];
                if (want == 0.0f) {
                    ASSERT_EQ(got & 0x7FFFFFFFu, 0u) << "k tile " << kt << " row " << r << " column " << n;
                } else {
                    ASSERT_EQ(got, float_to_bits(want)) << "k tile " << kt << " row " << r << " column " << n;
                }
            }
        }
    }
}

}  // namespace

// Each decoded tile equals its weights exactly: plain pages, codes pages (MXFP4's paired pages and words_shift) and
// the group step (Q4_K).
TEST_F(LLKBlackholeSingleCardFixture, GgufDecodeTileExact) {
    std::mt19937 rng(1);
    for (const auto [fmt, k_tiles] :
         std::vector<std::pair<std::uint32_t, std::uint32_t>>{{kQ8_0, 3}, {kQ4_0, 3}, {kMXFP4, 4}, {kQ4_K, 16}}) {
        const Case c{.fmt = fmt, .k_tiles = k_tiles, .mm = false};
        SCOPED_TRACE(name(c));
        const Inputs in = make_inputs(c, rng);
        check_decoded(c, in, run(*devices_.at(0), c, in));
    }
}

// The same in a half-sync DEST: every other K tile decodes in the upper half.
TEST_F(LLKBlackholeSingleCardFixture, GgufDecodeTileHalfSync) {
    std::mt19937 rng(3);
    for (const auto fmt : {kQ4_0, kMXFP4}) {
        const Case c{.fmt = fmt, .k_tiles = 4, .mm = false, .full_sync = false};
        SCOPED_TRACE(name(c));
        const Inputs in = make_inputs(c, rng);
        check_decoded(c, in, run(*devices_.at(0), c, in));
    }
}

// dest_srca_mm against a host matmul of the same weights truncated to TF32 (MOVD2A), within the FPU's summation
// error; the pack-thread decode (overlap) and the one-step page copy give the same bits as the reference run.
TEST_F(LLKBlackholeSingleCardFixture, GgufDestSrcaMatmul) {
    std::mt19937 rng(2);
    for (const auto fmt : {kQ8_0, kQ4_0, kMXFP4, kQ4_K}) {
        for (const std::uint32_t k_tiles : {fmt == kQ4_K ? 8u : 1u, fmt == kMXFP4 ? 4u : 2u, 16u}) {
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
                if (fmt == kQ4_K) {
                    Case copy = c;
                    copy.group_copy = true;
                    EXPECT_EQ(run(*devices_.at(0), copy, in), result) << "group_copy vs group to both sets";
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
