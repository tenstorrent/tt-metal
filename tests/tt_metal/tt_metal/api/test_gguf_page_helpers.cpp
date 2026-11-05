// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The GGUF page helpers (tt_metal/hw/inc/experimental/gguf_page.h, gguf_lookup_rvv.h) on random pages, bit for bit
// against a host reference written from the page layout's description:
//   - prepare_page on a data movement core, for every format, both scale conversions;
//   - lookup_grid and lookup_grid_rvv on the pack thread (ComputeConfig::enable_trisc2_rvv), for the codebook formats.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "tt_metal/hw/inc/experimental/gguf_level_tables.h"
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/experimental/ckernel_gguf_page.h"

namespace tt::tt_metal {
namespace {

using ckernel::gguf::Format;
using ckernel::gguf::PageLayout;

constexpr std::uint32_t kPages = 4;

template <std::size_t... I>
PageLayout layout_of(std::uint32_t fmt, std::index_sequence<I...>) {
    PageLayout l{};
    ((fmt == I + 1 ? (l = ckernel::gguf::page_layout<static_cast<Format>(I + 1)>(), 0) : 0), ...);
    return l;
}

PageLayout layout_of(std::uint32_t fmt) { return layout_of(fmt, std::make_index_sequence<25>{}); }

const std::uint16_t* table_of(std::uint32_t fmt) {
    namespace lv = ::experimental::gguf::levels;
    switch (static_cast<Format>(fmt)) {
        case Format::IQ2_XXS: return lv::iq2_xxs;
        case Format::IQ2_XS: return lv::iq2_xs;
        case Format::IQ2_S: return lv::iq2_s;
        case Format::IQ3_XXS: return lv::iq3_xxs;
        case Format::IQ3_S: return lv::iq3_s;
        case Format::IQ1_S:
        case Format::IQ1_M: return lv::iq1_s;
        default: return nullptr;
    }
}

// fp16 to fp32 through float arithmetic; inf and NaN keep their payload bits.
std::uint32_t ref_fp16_to_fp32(std::uint32_t h) {
    const std::uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31, mant = h & 0x3FF;
    if (exp == 31) {
        return sign << 31 | 0x7F800000u | mant << 13;
    }
    float v = exp == 0 ? std::ldexp(static_cast<float>(mant), -24)
                       : std::ldexp(static_cast<float>(1024 + mant), static_cast<int>(exp) - 25);
    v = sign ? -v : v;
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return bits;
}

// Grid index of byte k of index word m (with its high bits), and the scale nibbles IQ3_S ORs into the code.
std::uint32_t ref_index(std::uint32_t fmt, const std::uint32_t* idx, std::uint32_t m, std::uint32_t k) {
    const std::uint32_t low = (idx[m] >> (8 * k)) & 0xFF;
    switch (static_cast<Format>(fmt)) {
        case Format::IQ1_S:
        case Format::IQ1_M:
        case Format::IQ2_S: {
            const std::uint32_t w = fmt == Format::IQ2_S ? 2 : 3;
            const std::uint32_t even[4] = {8, 8 + w, 24, 24 + w}, odd[4] = {0, w, 16, 16 + w};
            const std::uint32_t shift = m % 2 ? odd[k] : even[k];
            return low | ((idx[32 + m / 2] >> shift) & ((1u << w) - 1)) << 8;
        }
        case Format::IQ2_XS: {
            const std::uint32_t sign_word = 16 * (m / 8 / 2) + 2 * (m % 8) + (m / 8) % 2;
            return low | (((idx - 32)[sign_word] >> (8 * k + 7)) & 1) << 8;  // sign words precede the indices
        }
        case Format::IQ3_S: return low | ((idx[64 + m / 8] >> (8 * k + m % 8)) & 1) << 8;
        default: return low;
    }
}

// prepare_page on the host: the scale fields' fp32 copies and the codebook formats' level codes.
std::vector<std::uint32_t> ref_prepare(std::uint32_t fmt, const std::uint32_t* page, bool scales, bool levels) {
    const PageLayout L = layout_of(fmt);
    std::vector<std::uint32_t> out(page, page + L.rows * 16);
    if (scales) {
        const std::uint32_t base = L.scale_offset / 4, r0 = (L.scale_rows + 3) / 4 * 4;
        for (std::uint32_t f = 0; f < L.fp16_rows; f++) {
            for (std::uint32_t w = 0; w < 16; w++) {
                const std::uint32_t x = page[base + (L.scale_rows - L.fp16_rows + f) * 16 + w];
                for (std::uint32_t b = 0; b < 2; b++) {
                    for (std::uint32_t k = 0; k < 4; k++) {
                        out[base + (r0 + 8 * f + 4 * b + k) * 16 + w] = ref_fp16_to_fp32((x >> (16 * b)) & 0xFFFF);
                    }
                }
            }
        }
    }
    if (levels && table_of(fmt) != nullptr) {
        const bool iq1 = fmt == Format::IQ1_S || fmt == Format::IQ1_M;
        const std::uint32_t* idx = page + (L.code_offset + (iq1 ? 0 : 128)) / 4;
        const std::uint32_t words = fmt == Format::IQ3_S || fmt == Format::IQ3_XXS ? 64 : 32;
        auto* codes = reinterpret_cast<std::uint16_t*>(out.data());
        const auto* scale_bytes = reinterpret_cast<const std::uint8_t*>(idx + 72);
        for (std::uint32_t m = 0; m < words; m++) {
            for (std::uint32_t k = 0; k < 4; k++) {
                std::uint16_t code = table_of(fmt)[ref_index(fmt, idx, m, k)];
                if (fmt == Format::IQ3_S && (m / 8) % 4 == 0 && k < 2) {
                    const std::uint32_t s = scale_bytes[8 * (m / 32) + m % 8];
                    code |= ((k == 0 ? s : s >> 4) & 15) << 12;
                }
                codes[4 * m + k] = code;
            }
        }
    }
    return out;
}

// Random pages; per fp16 scale row, either all values normal or with zero, subnormal, inf and NaN halves.
std::vector<std::uint32_t> random_pages(std::uint32_t fmt, std::mt19937& rng) {
    const PageLayout L = layout_of(fmt);
    std::vector<std::uint32_t> pages(kPages * L.rows * 16);
    for (auto& w : pages) {
        w = rng();
    }
    for (std::uint32_t p = 0; p < kPages; p++) {
        for (std::uint32_t f = 0; f < L.fp16_rows; f++) {
            std::uint32_t* row =
                pages.data() + p * L.rows * 16 + L.scale_offset / 4 + (L.scale_rows - L.fp16_rows + f) * 16;
            auto* halves = reinterpret_cast<std::uint16_t*>(row);
            for (std::uint32_t i = 0; i < 32; i++) {
                const std::uint32_t e = (halves[i] >> 10) & 31;
                if ((p + f) % 2 == 0 && (e == 0 || e == 31)) {
                    halves[i] ^= 0x2000;  // exponent 8 or 23
                }
            }
            if ((p + f) % 2 == 1) {
                const std::uint16_t special[] = {0x0000, 0x8000, 0x0001, 0x83FF, 0x7C00, 0xFC00, 0x7E01, 0x0400};
                for (std::uint32_t i = 0; i < 8; i++) {
                    halves[(5 * i + p) % 32] = special[i];
                }
            }
        }
    }
    return pages;
}

class GgufPageHelpersFixture : public UnitMeshAnyDispatchFixture {
protected:
    void SetUp() override {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP();
        }
        UnitMeshAnyDispatchFixture::SetUp();
    }

    std::shared_ptr<distributed::MeshBuffer> dram(std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            devices_.at(0).get());
    }

    // The pages through gguf_prepare_page.cpp (rvv: and gguf_lookup_rvv.cpp); the output buffers' contents.
    std::vector<std::vector<std::uint32_t>> run(
        std::uint32_t fmt, const std::vector<std::uint32_t>& pages, bool branch_free, bool rvv) {
        distributed::MeshDevice& mesh = *devices_.at(0);
        const std::uint32_t bytes = pages.size() * 4, page_bytes = bytes / kPages;
        const CoreCoord core{0, 0};
        Program program = CreateProgram(mesh);
        auto src = dram(bytes), dst = dram(bytes), dst2 = dram(bytes);
        for (const std::uint32_t cb : {0u, 16u, 17u}) {
            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(bytes, {{cb, tt::DataFormat::UInt32}}).set_page_size(cb, page_bytes));
        }
        const std::map<std::string, std::string> defines = {
            {"GGUF_FMT", std::to_string(fmt)},
            {"GGUF_BRANCH_FREE", std::to_string(branch_free)},
            {"GGUF_LOOKUP_RVV", std::to_string(rvv)}};
        auto dm = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/gguf_prepare_page.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});
        SetRuntimeArgs(program, dm, core, {src->address(), dst->address(), dst2->address(), kPages});
        if (rvv) {
            auto compute = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/compute/gguf_lookup_rvv.cpp",
                core,
                ComputeConfig{.enable_trisc2_rvv = true, .defines = defines});
            SetRuntimeArgs(program, compute, core, {kPages});
        }
        auto& cq = mesh.mesh_command_queue();
        distributed::EnqueueWriteMeshBuffer(cq, src, pages, true);
        distributed::MeshWorkload workload;
        const distributed::MeshCoordinate zero(0, 0);
        workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::vector<std::vector<std::uint32_t>> out(rvv ? 2 : 1);
        distributed::EnqueueReadMeshBuffer(cq, out[0], dst, true);
        if (rvv) {
            distributed::EnqueueReadMeshBuffer(cq, out[1], dst2, true);
        }
        return out;
    }
};

}  // namespace

TEST_F(GgufPageHelpersFixture, PreparePageAllFormats) {
    std::mt19937 rng(3);
    for (std::uint32_t fmt = 1; fmt <= 25; fmt++) {
        const PageLayout L = layout_of(fmt);
        const auto pages = random_pages(fmt, rng);
        for (const bool branch_free : {false, true}) {
            SCOPED_TRACE("format " + std::to_string(fmt) + " branch_free " + std::to_string(branch_free));
            const auto out = run(fmt, pages, branch_free, false)[0];
            ASSERT_EQ(out.size(), pages.size());
            for (std::uint32_t p = 0; p < kPages; p++) {
                const auto ref = ref_prepare(fmt, pages.data() + p * L.rows * 16, true, true);
                for (std::uint32_t w = 0; w < L.rows * 16; w++) {
                    ASSERT_EQ(out[p * L.rows * 16 + w], ref[w]) << "page " << p << " word " << w;
                }
            }
        }
    }
}

TEST_F(GgufPageHelpersFixture, LookupGridRvvMatchesScalar) {
    std::mt19937 rng(4);
    for (std::uint32_t fmt = 1; fmt <= 25; fmt++) {
        const PageLayout L = layout_of(fmt);
        if (table_of(fmt) == nullptr) {
            continue;
        }
        SCOPED_TRACE("format " + std::to_string(fmt));
        const auto pages = random_pages(fmt, rng);
        const auto out = run(fmt, pages, true, true);
        for (std::uint32_t p = 0; p < kPages; p++) {
            const auto ref = ref_prepare(fmt, pages.data() + p * L.rows * 16, false, true);
            for (std::uint32_t w = 0; w < L.code_offset / 4; w++) {
                ASSERT_EQ(out[0][p * L.rows * 16 + w], ref[w]) << "scalar, page " << p << " word " << w;
                ASSERT_EQ(out[1][p * L.rows * 16 + w], ref[w]) << "vector, page " << p << " word " << w;
            }
        }
    }
}

}  // namespace tt::tt_metal
