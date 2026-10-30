// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Spike E of the GDN decode v2 plan: the causal-conv state update as one kernel over a conv-state pool
// [slots, W-1, D] (bf16, row-major, DRAM), x [T, D], weight [D, W], bias [D], with a slot index (negative = skip),
// FIR + SiLU in fp32, the window shifted by T. Checks: y within one bf16 ulp of the host reference, the slot holds the
// last W-1 inputs, untouched slots and a skipped row stay byte-identical, and one call of T = K + 1 tokens equals
// K + 1 chained T = 1 calls bit for bit. The kernel is test_kernels/dataflow/gdn_conv_update.cpp (software fp32 on a
// data-movement RISC: a design probe, not a performance one).

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace unit_tests::compute::gdn_conv {

constexpr uint32_t D = 256, W = 4, SLOTS = 4, HIST = W - 1;
constexpr uint32_t kDramBankId = 0;

inline float bf2f(uint16_t b) { return std::bit_cast<float>(uint32_t(b) << 16); }
inline uint16_t f2bf(float f) {
    const uint32_t u = std::bit_cast<uint32_t>(f);
    return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

struct Case {
    std::vector<uint16_t> state;  // [SLOTS][HIST][D]
    std::vector<uint16_t> x;      // [T][D]
    std::vector<uint16_t> w;      // [D][W]
    std::vector<uint16_t> b;      // [D]
    uint32_t T;
};

Case make_case(uint32_t T, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Case c;
    c.T = T;
    c.state.resize(SLOTS * HIST * D);
    c.x.resize(T * D);
    c.w.resize(D * W);
    c.b.resize(D);
    for (auto& v : c.state) {
        v = f2bf(nd(gen));
    }
    for (auto& v : c.x) {
        v = f2bf(nd(gen));
    }
    for (auto& v : c.w) {
        v = f2bf(0.5f * nd(gen));
    }
    for (auto& v : c.b) {
        v = f2bf(0.1f * nd(gen));
    }
    return c;
}

// host reference with the kernel's arithmetic order (fp32 accumulate, std::exp)
void reference(const Case& c, uint32_t slot, std::vector<uint16_t>& y, std::vector<uint16_t>& new_state) {
    std::vector<uint16_t> win(c.state.begin() + slot * HIST * D, c.state.begin() + (slot + 1) * HIST * D);
    win.insert(win.end(), c.x.begin(), c.x.end());
    y.assign(c.T * D, 0);
    for (uint32_t t = 0; t < c.T; t++) {
        for (uint32_t d = 0; d < D; d++) {
            float acc = bf2f(c.b[d]);
            for (uint32_t i = 0; i < W; i++) {
                acc += bf2f(c.w[d * W + i]) * bf2f(win[(t + i) * D + d]);
            }
            y[t * D + d] = f2bf(acc / (1.0f + std::exp(-acc)));
        }
    }
    new_state.assign(win.begin() + c.T * D, win.begin() + (c.T + HIST) * D);
}

struct Result {
    std::vector<uint16_t> state;  // whole pool after the call
    std::vector<uint16_t> y;
};

Result run(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const Case& c, int32_t slot) {
    auto& cq = mesh_device->mesh_command_queue();
    const auto zero = distributed::MeshCoordinate(0, 0);
    const auto range = distributed::MeshCoordinateRange(zero, zero);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(range, std::move(program));
    auto& program_ = workload.get_programs().at(range);
    const CoreCoord core = {0, 0};
    auto make_buf = [&](uint32_t bytes, BufferType type) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{.page_size = bytes, .buffer_type = type, .bottom_up = false},
            mesh_device.get());
    };
    auto as_u32 = [](const std::vector<uint16_t>& v) {
        std::vector<uint32_t> out((v.size() + 1) / 2, 0);
        for (size_t i = 0; i < v.size(); i++) {
            out[i / 2] |= uint32_t(v[i]) << (16 * (i % 2));
        }
        return out;
    };
    auto state_buf = make_buf(SLOTS * HIST * D * 2, BufferType::DRAM);
    auto x_buf = make_buf(c.T * D * 2, BufferType::DRAM);
    auto w_buf = make_buf(D * W * 2, BufferType::DRAM);
    auto b_buf = make_buf(D * 2, BufferType::DRAM);
    auto y_buf = make_buf(c.T * D * 2, BufferType::DRAM);
    auto idx_buf = make_buf(32, BufferType::DRAM);
    const uint32_t scratch_bytes = (HIST + c.T) * D * 2 + D * W * 2 + D * 2 + c.T * D * 2 + 64;
    auto scratch = make_buf(scratch_bytes, BufferType::L1);
    std::vector<uint32_t> st = as_u32(c.state), xx = as_u32(c.x), ww = as_u32(c.w), bb = as_u32(c.b);
    std::vector<uint32_t> ym(c.T * D / 2, 0xBEEFBEEFu);  // marker: a skipped row must leave it
    std::vector<uint32_t> idx(8, 0);
    idx[0] = static_cast<uint32_t>(slot);
    distributed::WriteShard(cq, state_buf, st, zero);
    distributed::WriteShard(cq, x_buf, xx, zero);
    distributed::WriteShard(cq, w_buf, ww, zero);
    distributed::WriteShard(cq, b_buf, bb, zero);
    distributed::WriteShard(cq, y_buf, ym, zero);
    distributed::WriteShard(cq, idx_buf, idx, zero);
    auto kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/gdn_conv_update.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {D, W}});
    SetRuntimeArgs(
        program_,
        kernel,
        core,
        {uint32_t(state_buf->address()),
         uint32_t(x_buf->address()),
         uint32_t(w_buf->address()),
         uint32_t(b_buf->address()),
         uint32_t(y_buf->address()),
         uint32_t(idx_buf->address()),
         uint32_t(scratch->address()),
         c.T});
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<uint32_t> st_out, y_out;
    distributed::ReadShard(cq, st_out, state_buf, zero);
    distributed::ReadShard(cq, y_out, y_buf, zero);
    auto to_u16 = [](const std::vector<uint32_t>& v, size_t n) {
        std::vector<uint16_t> out(n);
        for (size_t i = 0; i < n; i++) {
            out[i] = uint16_t(v[i / 2] >> (16 * (i % 2)));
        }
        return out;
    };
    return {to_u16(st_out, SLOTS * HIST * D), to_u16(y_out, c.T * D)};
}

uint32_t max_ulp(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    uint32_t m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        m = std::max<uint32_t>(m, uint32_t(std::abs(int(a[i]) - int(b[i]))));
    }
    return m;
}

TEST_F(LLKBlackholeSingleCardFixture, GdnConvUpdateOneCall) {
    const Case c = make_case(/*T=*/4, 11);
    const int32_t slot = 2;
    auto res = run(this->devices_.at(0), c, slot);
    std::vector<uint16_t> y_ref, s_ref;
    reference(c, slot, y_ref, s_ref);
    const uint32_t ulp = max_ulp(res.y, y_ref);
    EXPECT_LE(ulp, 1u) << "y differs from the fp32 host reference by more than one bf16 ulp";
    std::vector<uint16_t> slot_out(res.state.begin() + slot * HIST * D, res.state.begin() + (slot + 1) * HIST * D);
    EXPECT_EQ(slot_out, s_ref) << "the slot does not hold the last W-1 inputs";
    bool others_untouched = true;
    for (uint32_t s = 0; s < SLOTS; s++) {
        if (s == uint32_t(slot)) {
            continue;
        }
        for (uint32_t i = 0; i < HIST * D; i++) {
            others_untouched &= res.state[s * HIST * D + i] == c.state[s * HIST * D + i];
        }
    }
    EXPECT_TRUE(others_untouched);
    uint32_t exact = 0;
    for (size_t i = 0; i < y_ref.size(); i++) {
        exact += res.y[i] == y_ref[i];
    }
    log_info(
        tt::LogTest,
        "conv update T 4, D {}, W {}: y max bf16 ulp vs host {} ({} of {} words exact; the rest is the device's exp "
        "polynomial), slot shifted by T {}, other slots untouched {}",
        D,
        W,
        ulp,
        exact,
        y_ref.size(),
        slot_out == s_ref,
        others_untouched);
}

TEST_F(LLKBlackholeSingleCardFixture, GdnConvUpdateSkippedRow) {
    const Case c = make_case(/*T=*/2, 12);
    auto res = run(this->devices_.at(0), c, -1);
    EXPECT_EQ(res.state, c.state) << "a negative slot index must leave the pool untouched";
    bool y_marker = true;
    for (uint16_t v : res.y) {
        y_marker &= v == 0xBEEFu;
    }
    EXPECT_TRUE(y_marker) << "a skipped row must not write y";
}

TEST_F(LLKBlackholeSingleCardFixture, GdnConvUpdateChainedEqualsOneCall) {
    const uint32_t K = 3;
    const Case full = make_case(K + 1, 13);
    const int32_t slot = 1;
    auto once = run(this->devices_.at(0), full, slot);
    // K + 1 calls of T = 1, each on the pool the previous call left
    Case step = full;
    step.T = 1;
    std::vector<uint16_t> y_chain;
    for (uint32_t t = 0; t <= K; t++) {
        step.x.assign(full.x.begin() + t * D, full.x.begin() + (t + 1) * D);
        auto r = run(this->devices_.at(0), step, slot);
        y_chain.insert(y_chain.end(), r.y.begin(), r.y.end());
        step.state = r.state;
    }
    EXPECT_EQ(once.y, y_chain) << "one T = K + 1 call and K + 1 chained T = 1 calls differ in y";
    EXPECT_EQ(once.state, step.state) << "... or in the final conv state";
    log_info(
        tt::LogTest,
        "conv update: one call of T {} vs {} chained T 1 calls: y bit-equal {}, state bit-equal {}",
        K + 1,
        K + 1,
        once.y == y_chain,
        once.state == step.state);
}

}  // namespace unit_tests::compute::gdn_conv
}  // namespace tt::tt_metal
