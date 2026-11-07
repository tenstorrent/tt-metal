// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/moe_group/moe_group.hpp"
#include "moe_test_utils.hpp"
#include "test_utils/program_cache.hpp"

namespace {

constexpr uint32_t kTileH = 32U;
constexpr uint32_t kSentinel = 0xFFFFFFFFU;
constexpr uint16_t kKSlotSentinel = 0xFFFFU;

// Float-comparison tolerance for bf16-derived outputs. Single source of truth so
// rtol/atol stay consistent across all checks in this test file.
constexpr float kRtol = 1e-3F;
constexpr float kAtol = 1e-3F;

class MoeGroupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// Cursor alignment in element-count units — mirrors moe_group_device_operation.cpp.
uint32_t cursor_align_elems() {
    return tt::tt_metal::hal::get_l1_alignment() / sizeof(uint16_t);
}

// Mirrors moe_group_device_operation.cpp's t_cap formula.
uint32_t compute_t_cap(uint32_t e_local, uint32_t k, uint32_t d, uint32_t b, uint32_t s) {
    const uint32_t num_total_cores = ttml::test_utils::num_worker_cores();
    const uint32_t cur_align = cursor_align_elems();
    const uint32_t unaligned =
        std::min(e_local, k) * d * b * s + e_local * (kTileH + (cur_align - 1U) * num_total_cores);
    return tt::round_up(unaligned, kTileH);
}

// num_workers as picked by split_work_to_cores(grid, total_tiles).
uint32_t compute_num_workers(uint32_t e_local, uint32_t k, uint32_t d, uint32_t b, uint32_t s) {
    const uint32_t grid = ttml::test_utils::num_worker_cores();
    const uint32_t t_cap = compute_t_cap(e_local, k, d, b, s);
    return std::min(t_cap / kTileH, grid);
}

struct GroupReference {
    std::vector<float> grouped;         // [T_cap * H], row-major
    std::vector<float> grouped_scores;  // [T_cap]
    std::vector<uint32_t> k_slot;       // [T_cap]
    std::vector<uint32_t> counts;       // [e_local]
    std::vector<uint32_t> offsets;      // [e_local + 1]
    std::vector<uint32_t> plan;         // [T_cap]
    uint32_t t_cap{};
    uint32_t h{};
};

// CPU reference matching moe_group: identical scan layout, per-core padding,
// 32-row tail rounding. `num_total_cores` is the host-side scan core count
// (= num_workers from split_work_to_cores).
GroupReference moe_group_reference(
    const xt::xarray<float>& dispatched,   // [D, B, S, H] (already bf16-roundtripped if needed)
    const xt::xarray<uint32_t>& metadata,  // [D, B, S, K]
    const xt::xarray<float>& scores,       // [D, B, S, K] (bf16-roundtripped)
    const std::vector<uint16_t>& local_expert_ids,
    uint32_t k,
    uint32_t num_total_cores,
    uint32_t t_cap_override) {
    const uint32_t D = static_cast<uint32_t>(dispatched.shape(0));
    const uint32_t B = static_cast<uint32_t>(dispatched.shape(1));
    const uint32_t S = static_cast<uint32_t>(dispatched.shape(2));
    const uint32_t H = static_cast<uint32_t>(dispatched.shape(3));
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());
    const uint32_t total_rows = D * B * S;
    const uint32_t T_cap = t_cap_override > 0 ? t_cap_override : compute_t_cap(E_local, k, D, B, S);

    // Hit mask + k_slot per (expert, row).
    std::vector<uint8_t> hits(static_cast<size_t>(E_local) * total_rows, 0);
    std::vector<uint32_t> k_slots(static_cast<size_t>(E_local) * total_rows, kKSlotSentinel);
    std::vector<uint32_t> counts(E_local, 0);
    for (uint32_t e = 0; e < E_local; ++e) {
        const uint32_t leid = local_expert_ids[e];
        for (uint32_t t = 0; t < total_rows; ++t) {
            for (uint32_t ki = 0; ki < k; ++ki) {
                const uint32_t md = metadata.flat(t * k + ki);
                if (md == leid) {
                    hits[e * total_rows + t] = 1;
                    k_slots[e * total_rows + t] = ki;
                    ++counts[e];
                    break;
                }
            }
        }
    }

    const uint32_t slice_size = (total_rows + num_total_cores - 1U) / num_total_cores;
    const uint32_t cur_align = cursor_align_elems();

    // local_counts[c, e]
    std::vector<uint32_t> local_counts(num_total_cores * E_local, 0);
    for (uint32_t c = 0; c < num_total_cores; ++c) {
        const uint32_t s_start = c * slice_size;
        if (s_start >= total_rows) {
            continue;
        }
        const uint32_t s_end = std::min(s_start + slice_size, total_rows);
        for (uint32_t e = 0; e < E_local; ++e) {
            uint32_t cc = 0;
            for (uint32_t t = s_start; t < s_end; ++t) cc += hits[e * total_rows + t];
            local_counts[c * E_local + e] = cc;
        }
    }

    std::vector<uint32_t> offsets(E_local + 1, 0);
    for (uint32_t e = 0; e < E_local; ++e) {
        uint32_t running = offsets[e];
        for (uint32_t c = 0; c < num_total_cores; ++c) {
            running += tt::round_up(local_counts[c * E_local + e], cur_align);
        }
        offsets[e + 1] = tt::round_up(running, kTileH);
    }

    std::vector<uint32_t> plan(T_cap, kSentinel);
    std::vector<float> grouped_scores(T_cap, 0.0F);
    std::vector<uint32_t> k_slot_out(T_cap, kKSlotSentinel);
    std::vector<float> grouped(static_cast<size_t>(T_cap) * H, 0.0F);

    for (uint32_t e = 0; e < E_local; ++e) {
        uint32_t running = offsets[e];
        for (uint32_t c = 0; c < num_total_cores; ++c) {
            const uint32_t s_start = c * slice_size;
            if (s_start >= total_rows)
                continue;
            const uint32_t s_end = std::min(s_start + slice_size, total_rows);
            uint32_t n = 0;
            for (uint32_t t = s_start; t < s_end; ++t) {
                if (!hits[e * total_rows + t])
                    continue;
                plan[running + n] = t;
                k_slot_out[running + n] = k_slots[e * total_rows + t];
                const uint32_t ks = k_slots[e * total_rows + t];
                grouped_scores[running + n] = scores.flat(t * k + ks);
                for (uint32_t hh = 0; hh < H; ++hh) {
                    grouped[(running + n) * H + hh] = dispatched.flat(t * H + hh);
                }
                ++n;
            }
            running += tt::round_up(n, cur_align);
        }
    }

    return GroupReference{
        std::move(grouped),
        std::move(grouped_scores),
        std::move(k_slot_out),
        std::move(counts),
        std::move(offsets),
        std::move(plan),
        T_cap,
        H};
}

// Build host inputs (dispatched + scores already bf16-roundtripped on the test
// device, so the host reference compares apples-to-apples against device output).
// Row-index broadcast, the default, keeps moe_group reorder checks (EXPECT_FLOAT_EQ on grouped rows)
// trivially debuggable: out row N must equal the constant float(N).
ttml::test_utils::moe::MoeHostInputs make_inputs(
    uint32_t D,
    uint32_t B,
    uint32_t S,
    uint32_t H,
    uint32_t E,
    uint32_t K,
    ttml::test_utils::moe::DispatchedPattern dispatched_pattern =
        ttml::test_utils::moe::DispatchedPattern::RowIndexBroadcast,
    uint32_t seed = 42U) {  // MoeHostInputConfig's default seed
    return ttml::test_utils::moe::make_moe_host_inputs({
        .D = D,
        .B = B,
        .S = S,
        .H = H,
        .E = E,
        .K = K,
        .dispatched_pattern = dispatched_pattern,
        .roundtrip_device = &ttml::autograd::ctx().get_device(),
        .seed = seed,
    });
}

// Host-side copies of the device op's outputs, as plain arrays.
struct DeviceOutputs {
    std::vector<float> grouped;         // [T_cap * H]
    std::vector<float> grouped_scores;  // [T_cap]
    std::vector<uint32_t> k_slot;       // [T_cap]
    std::vector<uint32_t> counts;       // [E_local]
    std::vector<uint32_t> offsets;      // [E_local + 1]
    std::vector<uint32_t> plan;         // [T_cap]
    uint32_t T_cap{};
    uint32_t H{};
};

DeviceOutputs read_outputs(const ttml::metal::MoeGroupResult& result) {
    const auto& [grouped, grouped_scores, k_slot, counts, offsets, plan] = result;

    DeviceOutputs out;
    auto grouped_rm = ttnn::to_layout(grouped, ttnn::ROW_MAJOR_LAYOUT);
    auto grouped_xt = ttml::core::to_xtensor(grouped_rm);
    out.T_cap = static_cast<uint32_t>(grouped_xt.shape(2));
    out.H = static_cast<uint32_t>(grouped_xt.shape(3));
    out.grouped.assign(grouped_xt.begin(), grouped_xt.end());

    auto gs_xt = ttml::core::to_xtensor(grouped_scores);
    out.grouped_scores.assign(gs_xt.begin(), gs_xt.end());

    auto ks_xt = ttml::core::to_xtensor<uint16_t>(k_slot);
    out.k_slot.assign(ks_xt.begin(), ks_xt.end());

    auto cnts_xt = ttml::core::to_xtensor<uint32_t>(counts);
    out.counts.assign(cnts_xt.begin(), cnts_xt.end());

    auto offs_xt = ttml::core::to_xtensor<uint32_t>(offsets);
    out.offsets.assign(offs_xt.begin(), offs_xt.end());

    auto plan_xt = ttml::core::to_xtensor<uint32_t>(plan);
    out.plan.assign(plan_xt.begin(), plan_xt.end());

    return out;
}

// Run the device op + return host-side outputs as plain arrays.
DeviceOutputs run_op(
    const ttml::test_utils::moe::MoeHostInputs& host, const std::vector<uint16_t>& local_expert_ids, uint32_t k) {
    auto& dev = ttml::autograd::ctx().get_device();
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());

    auto dev_in = ttml::test_utils::moe::to_device_inputs(host, local_expert_ids, &dev);

    return read_outputs(ttml::metal::moe_group(
        dev_in.dispatched_bf16, dev_in.metadata_u16, dev_in.scores_bf16, dev_in.leids_u16, E_local, k));
}

void expect_outputs_match_reference(
    const DeviceOutputs& out,
    const ttml::test_utils::moe::MoeHostInputs& host,
    const std::vector<uint16_t>& local_expert_ids,
    uint32_t k) {
    const uint32_t D = static_cast<uint32_t>(host.dispatched.shape(0));
    const uint32_t B = static_cast<uint32_t>(host.dispatched.shape(1));
    const uint32_t S = static_cast<uint32_t>(host.dispatched.shape(2));
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());

    // host.dispatched is already bf16-roundtripped by make_inputs(), so it matches what the
    // device sees and can be fed straight into the reference.
    const auto& disp_rt = host.dispatched;

    const uint32_t num_workers = compute_num_workers(E_local, k, D, B, S);
    const uint32_t t_cap = compute_t_cap(E_local, k, D, B, S);
    auto ref = moe_group_reference(disp_rt, host.metadata, host.scores, local_expert_ids, k, num_workers, t_cap);

    ASSERT_EQ(out.T_cap, ref.t_cap);
    ASSERT_EQ(out.counts.size(), ref.counts.size());
    for (size_t i = 0; i < ref.counts.size(); ++i) {
        EXPECT_EQ(out.counts[i], ref.counts[i]) << "counts[" << i << "]";
    }
    ASSERT_EQ(out.offsets.size(), ref.offsets.size());
    for (size_t i = 0; i < ref.offsets.size(); ++i) {
        EXPECT_EQ(out.offsets[i], ref.offsets[i]) << "offsets[" << i << "]";
    }
    ASSERT_EQ(out.plan.size(), ref.plan.size());
    for (size_t i = 0; i < ref.plan.size(); ++i) {
        EXPECT_EQ(out.plan[i], ref.plan[i]) << "plan[" << i << "]";
    }
    ASSERT_EQ(out.k_slot.size(), ref.k_slot.size());
    for (size_t i = 0; i < ref.k_slot.size(); ++i) {
        EXPECT_EQ(out.k_slot[i], ref.k_slot[i]) << "k_slot[" << i << "]";
    }
    ASSERT_EQ(out.grouped_scores.size(), ref.grouped_scores.size());
    {
        auto out_gs = xt::adapt(out.grouped_scores, std::vector<size_t>{out.grouped_scores.size()});
        auto ref_gs = xt::adapt(ref.grouped_scores, std::vector<size_t>{ref.grouped_scores.size()});
        EXPECT_TRUE(xt::allclose(out_gs, ref_gs, kRtol, kAtol))
            << "grouped_scores allclose failed (rtol=" << kRtol << " atol=" << kAtol << ")";
    }
    // grouped: only ACTIVE rows (plan[i] != SENTINEL) are guaranteed equal to
    // dispatched[plan[i]]. Pad rows are not part of the op contract — the
    // ungroup op skips them via plan[i] == SENTINEL and weights them by 0
    // grouped_scores, so any value there is harmless.
    ASSERT_EQ(out.grouped.size(), ref.grouped.size());
    const uint32_t H = out.H;
    // moe_group only reorders dispatched rows into the grouped layout — no math is
    // performed on the values — so each active grouped row must be bit-identical
    // to the bf16-roundtripped dispatched source. EXPECT_FLOAT_EQ is intentional.
    for (uint32_t i = 0; i < ref.t_cap; ++i) {
        const uint32_t src = ref.plan[i];
        if (src == kSentinel)
            continue;
        for (uint32_t hh = 0; hh < H; ++hh) {
            const float got = out.grouped[i * H + hh];
            const float exp = disp_rt.flat(src * H + hh);
            EXPECT_FLOAT_EQ(got, exp) << "grouped[" << i << ", " << hh << "] (plan=" << src << ")";
        }
    }
}

void check_against_reference(
    const ttml::test_utils::moe::MoeHostInputs& host, const std::vector<uint16_t>& local_expert_ids, uint32_t k) {
    expect_outputs_match_reference(run_op(host, local_expert_ids, k), host, local_expert_ids, k);
}

}  // namespace

TEST_F(MoeGroupTest, BasicShape) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, LargerH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 256;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, NonTileAlignedH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 80;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, ExpertZeroActive) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    // local expert 5 is never present in metadata (E=4 → ids in [0,3]).
    const std::vector<uint16_t> leids = {0, 5};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, AllTokensActiveForAllExperts) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 2, K = 2;  // K == E → every token hits both
    const std::vector<uint16_t> leids = {0, 1};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, LargeELocal) {
    constexpr uint32_t D = 2, B = 1, S = 64, H = 64;
    constexpr uint32_t E = 64, K = 4;
    std::vector<uint16_t> leids;
    leids.reserve(32);
    for (uint16_t i = 0; i < 32; ++i) leids.push_back(i);
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

// ============================================================================
// Program-cache hits
// ============================================================================
// moe_group keys its programs on its attributes (e_local, k, D, B, S, H, and T_cap, which also depends on the grid) and
// on the specs of its four inputs, so a launch on new tensors of the same shapes reuses the program. Each test clears
// the cache so that its first launch builds the program, then launches again on new inputs while every tensor of the
// first launch stays alive: the hit must read the new inputs, write the new outputs and leave the first launch's
// outputs as they were. The two launches differ in every input: hidden rows and scores (seeds), routing (launch A
// draws K = 2 of E = 4 experts per token, so about half the tokens reach each local expert; launch B has E = 2, so
// every token reaches both) and the local expert ids. So every output differs too, counts and offsets included.
// ============================================================================

namespace {

using ttml::test_utils::expect_new_buffer;

constexpr uint32_t kCacheTestELocal = 2U;
constexpr uint32_t kCacheTestK = 2U;

// moe_group splits the T_cap / 32 tile-rows of `grouped` over the cores, and the D * B * S tokens over the same cores
// in contiguous scan slices of ceil(tokens / cores). Returns the smallest S, a multiple of 32, for which there are more
// than twice as many tile-rows as cores, and not a multiple of the core count, so that split_work_to_cores hands out
// two core groups (two combined, writer and tilize kernels) on any grid; and for which the last core's scan slice is
// not empty, so that every core reads metadata and scores and scatters plan, grouped_scores and k_slot entries.
uint32_t two_core_group_s(uint32_t D, uint32_t B) {
    const uint32_t cores = ttml::test_utils::num_worker_cores();
    for (uint32_t s = kTileH;; s += kTileH) {
        const uint32_t tile_rows = compute_t_cap(kCacheTestELocal, kCacheTestK, D, B, s) / kTileH;
        const uint32_t tokens = D * B * s;
        const uint32_t scan_slice = (tokens + cores - 1U) / cores;  // the scan runs on min(tile_rows, cores) cores
        if (tile_rows > 2U * cores && tile_rows % cores != 0U && (cores - 1U) * scan_slice < tokens) {
            return s;
        }
    }
}

ttml::metal::MoeGroupResult launch_moe_group(const ttml::test_utils::moe::MoeDeviceInputs& in) {
    return ttml::metal::moe_group(
        in.dispatched_bf16, in.metadata_u16, in.scores_bf16, in.leids_u16, kCacheTestELocal, kCacheTestK);
}

template <typename T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Bit for bit: the rows of `grouped` past offsets[e_local] are never written, and whatever they hold must stay.
void expect_same_outputs(const DeviceOutputs& actual, const DeviceOutputs& expected) {
    EXPECT_TRUE(same_bits(actual.grouped, expected.grouped)) << "grouped";
    EXPECT_TRUE(same_bits(actual.grouped_scores, expected.grouped_scores)) << "grouped_scores";
    EXPECT_EQ(actual.k_slot, expected.k_slot) << "k_slot";
    EXPECT_EQ(actual.counts, expected.counts) << "counts";
    EXPECT_EQ(actual.offsets, expected.offsets) << "offsets";
    EXPECT_EQ(actual.plan, expected.plan) << "plan";
}

// Launch A builds the program. Launch B, on new inputs of the same shapes while every tensor of launch A stays alive,
// must reuse it.
void expect_cache_hit_with_fresh_tensors(uint32_t D, uint32_t B, uint32_t S, uint32_t H) {
    using ttml::test_utils::moe::DispatchedPattern;
    auto& device = ttml::test_utils::device_with_empty_program_cache();

    const std::vector<uint16_t> leids_a = {0, 1};
    const auto host_a = make_inputs(D, B, S, H, /*E=*/4U, kCacheTestK, DispatchedPattern::UniformRandom, 1001U);
    const auto in_a = ttml::test_utils::moe::to_device_inputs(host_a, leids_a, &device);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto out_a = launch_moe_group(in_a);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "launch A must build a program";
    const auto values_a = read_outputs(out_a);
    {
        SCOPED_TRACE("launch A");
        expect_outputs_match_reference(values_a, host_a, leids_a, kCacheTestK);
    }

    const std::vector<uint16_t> leids_b = {1, 0};
    const auto host_b = make_inputs(D, B, S, H, /*E=*/2U, kCacheTestK, DispatchedPattern::UniformRandom, 2002U);
    const auto in_b = ttml::test_utils::moe::to_device_inputs(host_b, leids_b, &device);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto out_b = launch_moe_group(in_b);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    expect_new_buffer(in_a.dispatched_bf16, in_b.dispatched_bf16, "dispatched");
    expect_new_buffer(in_a.metadata_u16, in_b.metadata_u16, "metadata");
    expect_new_buffer(in_a.scores_bf16, in_b.scores_bf16, "scores");
    expect_new_buffer(in_a.leids_u16, in_b.leids_u16, "local_expert_ids");
    expect_new_buffer(std::get<0>(out_a), std::get<0>(out_b), "grouped");
    expect_new_buffer(std::get<1>(out_a), std::get<1>(out_b), "grouped_scores");
    expect_new_buffer(std::get<2>(out_a), std::get<2>(out_b), "k_slot");
    expect_new_buffer(std::get<3>(out_a), std::get<3>(out_b), "counts");
    expect_new_buffer(std::get<4>(out_a), std::get<4>(out_b), "offsets");
    expect_new_buffer(std::get<5>(out_a), std::get<5>(out_b), "plan");
    {
        SCOPED_TRACE("launch B");
        expect_outputs_match_reference(read_outputs(out_b), host_b, leids_b, kCacheTestK);
    }

    SCOPED_TRACE("launch A's outputs after launch B");
    expect_same_outputs(read_outputs(out_a), values_a);
}

}  // namespace

TEST_F(MoeGroupTest, CacheHitFreshTensorsTwoCoreGroups) {
    constexpr uint32_t D = 1, B = 1, H = 64;
    expect_cache_hit_with_fresh_tensors(D, B, two_core_group_s(D, B), H);
}

TEST_F(MoeGroupTest, CacheHitFreshTensorsPartialGrid) {
    // BasicShape's tokens with NonTileAlignedH's H: fewer T_cap tile-rows than cores, so a single core group on part
    // of the grid.
    expect_cache_hit_with_fresh_tensors(/*D=*/2, /*B=*/1, /*S=*/32, /*H=*/80);
}
