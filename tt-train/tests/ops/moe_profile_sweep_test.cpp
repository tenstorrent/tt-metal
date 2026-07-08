// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Opt-in profile sweeps for moe_group + moe_ungroup. Mirrors the Python
// suite under tests/python/test_moe_*.py::TestMoe*Profile but runs the same
// shape grid as native gtest cases so the whole thing can be driven from
// the C++ test binary under Tracy.
//
// Enable by setting TTML_RUN_PROFILE_TESTS=1. Each test:
//   1. Emits a `TT_SIGNPOST: moe_<op>_start_<routing>` tracy message
//   2. Warmups (2) and timed iters (10) of the device op
//   3. Emits the matching `_end_` signpost
//
// tools/profiling/parse_moe_profile.py picks the signpost-bracketed rows
// out of the tracy CSV.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <tools/profiler/op_profiler.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <ttnn/operations/core/core.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/moe_group/moe_group.hpp"
#include "metal/ops/moe_ungroup/moe_ungroup.hpp"
#include "moe_test_utils.hpp"

namespace {

bool profile_tests_enabled() {
    const char* env = std::getenv("TTML_RUN_PROFILE_TESTS");
    if (env == nullptr)
        return false;
    const std::string v(env);
    return v == "1" || v == "true" || v == "True";
}

void signpost(const std::string& name) {
    // Format the Tracy parser recognizes — matches the Python `signpost`
    // helper in tools/tracy/__init__.py.
    tt::tt_metal::op_profiler::tracy_message(std::string("`TT_SIGNPOST: ") + name + "`");
}

class MoeProfileTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!profile_tests_enabled()) {
            GTEST_SKIP() << "Profile sweep is opt-in: set TTML_RUN_PROFILE_TESTS=1 to enable";
        }
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }
    static void TearDownTestSuite() {
        if (!profile_tests_enabled())
            return;
        ttml::autograd::ctx().close_device();
    }
};

// "fully_skewed" routing: every token's top-K is entirely local (peak T_active).
// Local to profile sweep — not shared via moe_test_utils since correctness tests
// don't exercise this routing pattern.
xt::xarray<uint32_t> make_metadata_all_local(
    uint32_t D, uint32_t B, uint32_t S, uint32_t K, const std::vector<uint16_t>& leids) {
    xt::xarray<uint32_t> out = xt::zeros<uint32_t>({D, B, S, K});
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 0; s < S; ++s) {
                for (uint32_t ki = 0; ki < K; ++ki) {
                    out(d, b, s, ki) = leids[ki % leids.size()];
                }
            }
        }
    }
    return out;
}

struct ShapeCfg {
    uint32_t D, B, S, H, E, K, E_local;
};

constexpr std::array<ShapeCfg, 22> kShapeSweep = {{
    // Tiny / smoke
    {2, 1, 128, 512, 4, 2, 2},
    {2, 1, 128, 1024, 8, 2, 2},
    {4, 1, 256, 2048, 16, 4, 4},
    // H sweep (D=8, S=2048, E=32, K=4, E_local=4)
    {8, 1, 2048, 1024, 32, 4, 4},
    {8, 1, 2048, 2048, 32, 4, 4},
    {8, 1, 2048, 4096, 32, 4, 4},
    {8, 1, 2048, 7168, 32, 4, 4},
    {8, 1, 2048, 8192, 32, 4, 4},
    // S sweep (D=8, H=4096, E=96, K=8, E_local=12)
    {8, 1, 512, 4096, 96, 8, 12},
    {8, 1, 1024, 4096, 96, 8, 12},
    {8, 1, 2048, 4096, 96, 8, 12},
    {8, 1, 4096, 4096, 96, 8, 12},  // roofline Config B
    {8, 1, 8192, 4096, 96, 8, 12},
    // Routing sparsity at fixed D=8, S=4096, H=4096
    {8, 1, 4096, 4096, 16, 2, 2},
    {8, 1, 4096, 4096, 32, 8, 2},
    {8, 1, 4096, 4096, 64, 8, 4},
    {8, 1, 4096, 4096, 96, 8, 12},
    {8, 1, 4096, 4096, 128, 8, 16},
    // Large-H (DeepSeek-like)
    {8, 1, 1024, 7168, 64, 8, 2},
    {8, 1, 2048, 7168, 64, 8, 2},
    {8, 1, 4096, 7168, 64, 8, 2},
    {8, 1, 4096, 8192, 64, 8, 4},
}};

constexpr uint32_t kWarmup = 2;
constexpr uint32_t kTimedIters = 10;

void run_group_iters(const ttml::test_utils::moe::MoeDeviceInputs& in, uint32_t e_local, uint32_t k) {
    auto& dev = ttml::autograd::ctx().get_device();
    for (uint32_t i = 0; i < kWarmup; ++i) {
        auto _ = ttml::metal::moe_group(in.dispatched_bf16, in.metadata_u16, in.scores_bf16, in.leids_u16, e_local, k);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    for (uint32_t i = 0; i < kTimedIters; ++i) {
        auto _ = ttml::metal::moe_group(in.dispatched_bf16, in.metadata_u16, in.scores_bf16, in.leids_u16, e_local, k);
        tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
        // Tracy `-r` flushes the device profiler buffer at process exit; no
        // per-iter ReadDeviceProfilerResults call here (the IDevice* overload
        // doesn't accept MeshDevice).
    }
}

void run_ungroup_iters(
    const ttnn::Tensor& expert_out,
    const ttnn::Tensor& plan,
    const ttnn::Tensor& offsets,
    const ttnn::Tensor& grouped_scores,
    uint32_t e_local,
    uint32_t D,
    uint32_t B,
    uint32_t S) {
    auto& dev = ttml::autograd::ctx().get_device();
    for (uint32_t i = 0; i < kWarmup; ++i) {
        auto _ = ttml::metal::moe_ungroup(expert_out, plan, offsets, grouped_scores, e_local, D, B, S);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    for (uint32_t i = 0; i < kTimedIters; ++i) {
        auto _ = ttml::metal::moe_ungroup(expert_out, plan, offsets, grouped_scores, e_local, D, B, S);
        tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
        // Tracy `-r` flushes the device profiler buffer at process exit; no
        // per-iter ReadDeviceProfilerResults call here (the IDevice* overload
        // doesn't accept MeshDevice).
    }
}

// Build host inputs + leids for a shape config. Profile sweeps skip bf16
// roundtripping (no reference comparison happens); the device-side BFLOAT16
// conversion in to_device_inputs is the only cast we need.
std::pair<ttml::test_utils::moe::MoeHostInputs, std::vector<uint16_t>> sweep_inputs(const ShapeCfg& c, bool all_local) {
    std::vector<uint16_t> leids;
    leids.reserve(c.E_local);
    for (uint32_t i = 0; i < c.E_local; ++i) leids.push_back(static_cast<uint16_t>(i));
    auto host = ttml::test_utils::moe::make_moe_host_inputs({
        .D = c.D,
        .B = c.B,
        .S = c.S,
        .H = c.H,
        .E = c.E,
        .K = c.K,
    });
    if (all_local) {
        host.metadata = make_metadata_all_local(c.D, c.B, c.S, c.K, leids);
    }
    return {std::move(host), std::move(leids)};
}

void group_sweep_one(const ShapeCfg& c, bool all_local) {
    auto& dev = ttml::autograd::ctx().get_device();
    auto [host, leids] = sweep_inputs(c, all_local);
    auto in = ttml::test_utils::moe::to_device_inputs(host, leids, &dev);

    const std::string routing = all_local ? "fully_skewed" : "balanced";
    signpost("moe_group_start_" + routing);
    run_group_iters(in, c.E_local, c.K);
    signpost("moe_group_end_" + routing);
}

void ungroup_sweep_one(const ShapeCfg& c, bool all_local) {
    auto& dev = ttml::autograd::ctx().get_device();
    auto [host, leids] = sweep_inputs(c, all_local);
    auto in = ttml::test_utils::moe::to_device_inputs(host, leids, &dev);

    // Build (expert_out, plan, offsets, grouped_scores) from moe_group; we reuse
    // `grouped` as the FFN's `expert_out` (identity FFN — fine for benchmarking
    // ungroup's DRAM bandwidth).
    auto [grouped, grouped_scores, k_slot, counts, offsets, plan] =
        ttml::metal::moe_group(in.dispatched_bf16, in.metadata_u16, in.scores_bf16, in.leids_u16, c.E_local, c.K);

    const std::string routing = all_local ? "fully_skewed" : "balanced";
    signpost("moe_ungroup_start_" + routing);
    run_ungroup_iters(grouped, plan, offsets, grouped_scores, c.E_local, c.D, c.B, c.S);
    signpost("moe_ungroup_end_" + routing);
}

}  // namespace

class MoeGroupProfileSweep : public MoeProfileTest, public ::testing::WithParamInterface<ShapeCfg> {};
class MoeUngroupProfileSweep : public MoeProfileTest, public ::testing::WithParamInterface<ShapeCfg> {};

TEST_P(MoeGroupProfileSweep, DISABLED_Balanced) {
    group_sweep_one(GetParam(), /*all_local=*/false);
}

TEST_P(MoeUngroupProfileSweep, DISABLED_Balanced) {
    ungroup_sweep_one(GetParam(), /*all_local=*/false);
}

INSTANTIATE_TEST_SUITE_P(Shapes, MoeGroupProfileSweep, ::testing::ValuesIn(kShapeSweep));
INSTANTIATE_TEST_SUITE_P(Shapes, MoeUngroupProfileSweep, ::testing::ValuesIn(kShapeSweep));

// Worst-case routing: every token's top-K is entirely local experts. Uses the
// same shape as the balanced roofline test — the signpost label differs so the
// parser splits the two patterns in the summary table.
TEST_F(MoeProfileTest, DISABLED_GroupAllLocalRouting) {
    group_sweep_one(ShapeCfg{8, 1, 4096, 4096, 96, 8, 12}, /*all_local=*/true);
}

TEST_F(MoeProfileTest, DISABLED_UngroupAllLocalRouting) {
    ungroup_sweep_one(ShapeCfg{8, 1, 4096, 4096, 96, 8, 12}, /*all_local=*/true);
}
