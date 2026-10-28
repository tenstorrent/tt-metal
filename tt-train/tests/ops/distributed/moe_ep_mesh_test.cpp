// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Expert-parallel (sparse EP) training runs moe_group and moe_ungroup on every chip of the EP axis: the tokens, their
// routing and their scores are replicated across the axis and the local expert ids are sharded, so each chip groups,
// and later ungroups, only the tokens routed to its own experts. ops/moe_group_op_test.cpp and
// ops/moe_ungroup_op_test.cpp run on a 1x1 mesh. This test runs moe_group, an identity FFN and moe_ungroup on a 1x2
// mesh (an N300, or two Blackhole chips as on bh_quietbox_2), then again on fresh tensors with other expert ids while
// every tensor of the first launch stays alive: the second launch of each op must reuse the cached workload and read
// and write the new buffers on both devices. Smaller systems skip it.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "../moe_test_utils.hpp"
#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/moe_group/moe_group.hpp"
#include "metal/ops/moe_ungroup/moe_ungroup.hpp"
#include "test_utils/mesh_utils.hpp"
#include "test_utils/program_cache.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace {

using ttml::test_utils::expect_new_buffer;

constexpr uint32_t kNumDevices = 2U;
constexpr uint32_t kD = 2U;
constexpr uint32_t kB = 1U;
constexpr uint32_t kH = 64U;
constexpr uint32_t kE = 4U;                     // experts in the model
constexpr uint32_t kK = 2U;                     // experts per token
constexpr uint32_t kELocal = kE / kNumDevices;  // experts per device
constexpr uint32_t kPlanSentinel = 0xFFFFFFFFU;
// moe_ungroup_op_test.cpp's tolerances.
constexpr float kRtol = 5e-2F;
constexpr float kAtol = 1e-2F;

// D = 2 and S = 32 * cores, as in MoeUngroupTest.CacheHitFreshTensorsManyCores: on each device moe_group scans 64
// tokens on every core and splits its tile-rows into two core groups, and moe_ungroup zeroes 64 rows on every core.
uint32_t many_cores_s() {
    return 32U * ttml::test_utils::num_worker_cores();
}

// One launch's inputs, and the values every device reads from them: dispatched, metadata and scores are replicated,
// and each device holds kELocal of the expert ids.
struct EpInputs {
    ttnn::Tensor dispatched;  // [D, B, S, H] ROW_MAJOR bf16
    ttnn::Tensor metadata;    // [D, B, S, K] ROW_MAJOR uint16: K distinct expert ids per token
    ttnn::Tensor scores;      // [D, B, S, K] ROW_MAJOR bf16
    ttnn::Tensor leids;       // [E_local]    ROW_MAJOR uint16 on each device
    xt::xarray<float> dispatched_values;
    xt::xarray<uint16_t> metadata_values;
    xt::xarray<float> scores_values;
    std::vector<uint16_t> leids_values;  // device 0's ids, then device 1's
};

EpInputs make_inputs(uint32_t S, const std::vector<uint16_t>& leids, uint32_t seed) {
    using ttml::core::from_xtensor;
    auto& device = ttml::autograd::ctx().get_device();
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(device, 0);

    const auto host = ttml::test_utils::moe::make_moe_host_inputs({
        .D = kD,
        .B = kB,
        .S = S,
        .H = kH,
        .E = kE,
        .K = kK,
        .seed = seed,
    });
    const xt::xarray<uint16_t> metadata = xt::cast<uint16_t>(host.metadata);
    const xt::xarray<uint16_t> leids_xt = xt::adapt(leids, std::vector<std::size_t>{leids.size()});

    EpInputs in{
        from_xtensor<float, ttnn::DataType::BFLOAT16>(
            host.dispatched, &device, ttnn::Layout::ROW_MAJOR, replicate.get()),
        from_xtensor<uint16_t, ttnn::DataType::UINT16>(metadata, &device, ttnn::Layout::ROW_MAJOR, replicate.get()),
        from_xtensor<float, ttnn::DataType::BFLOAT16>(host.scores, &device, ttnn::Layout::ROW_MAJOR, replicate.get()),
        from_xtensor<uint16_t, ttnn::DataType::UINT16>(leids_xt, &device, ttnn::Layout::ROW_MAJOR, shard.get()),
        {},
        metadata,
        {},
        leids};
    in.dispatched_values = ttml::core::to_xtensor<float>(in.dispatched, ttml::core::IdentityComposer{}).at(0);
    in.scores_values = ttml::core::to_xtensor<float>(in.scores, ttml::core::IdentityComposer{}).at(0);
    return in;
}

// moe_group's outputs feed moe_ungroup directly: the identity FFN's output is `grouped` itself.
ttnn::Tensor ungroup(const ttml::metal::MoeGroupResult& group, uint32_t S) {
    const auto& [grouped, grouped_scores, k_slot, counts, offsets, plan] = group;
    return ttml::metal::moe_ungroup(grouped, plan, offsets, grouped_scores, kELocal, kD, kB, S);
}

// One device's outputs of moe_group and moe_ungroup.
struct DeviceOutputs {
    xt::xarray<float> grouped;          // [1, 1, T_cap, H]
    std::vector<uint32_t> counts;       // [E_local]
    std::vector<uint32_t> offsets;      // [E_local + 1]
    std::vector<uint32_t> plan;         // [T_cap]
    std::vector<uint16_t> k_slot;       // [T_cap]
    std::vector<float> grouped_scores;  // [T_cap]
    xt::xarray<float> ungrouped;        // [D, B, S, H]
};

template <typename T>
std::vector<std::vector<T>> per_device(const ttnn::Tensor& tensor) {
    std::vector<std::vector<T>> values;
    for (const auto& shard : ttml::core::to_xtensor<T>(tensor, ttml::core::IdentityComposer{})) {
        values.emplace_back(shard.begin(), shard.end());
    }
    return values;
}

std::vector<DeviceOutputs> read_outputs(const ttml::metal::MoeGroupResult& group, const ttnn::Tensor& ungrouped) {
    const auto& [grouped, grouped_scores, k_slot, counts, offsets, plan] = group;
    auto grouped_values = ttml::core::to_xtensor<float>(grouped, ttml::core::IdentityComposer{});
    const auto counts_values = per_device<uint32_t>(counts);
    const auto offsets_values = per_device<uint32_t>(offsets);
    const auto plan_values = per_device<uint32_t>(plan);
    const auto k_slot_values = per_device<uint16_t>(k_slot);
    const auto grouped_scores_values = per_device<float>(grouped_scores);
    auto ungrouped_values = ttml::core::to_xtensor<float>(ungrouped, ttml::core::IdentityComposer{});
    std::vector<DeviceOutputs> outputs;
    outputs.reserve(ungrouped_values.size());
    for (std::size_t d = 0; d < ungrouped_values.size(); ++d) {
        outputs.push_back(
            {std::move(grouped_values.at(d)),
             counts_values.at(d),
             offsets_values.at(d),
             plan_values.at(d),
             k_slot_values.at(d),
             grouped_scores_values.at(d),
             std::move(ungrouped_values[d])});
    }
    return outputs;
}

// Checks one device's outputs against the tokens routed to its expert ids. moe_group lists the tokens that picked the
// device's expert e in plan[offsets[e], offsets[e + 1]), padded with sentinel entries, each with the slot of its
// routing that picked the expert (k_slot), its score there (grouped_scores) and its hidden row (the row of grouped);
// every plan entry past the last expert is a sentinel. moe_ungroup then gives every token the sum of its scores for the
// device's experts times its hidden row, and zero when the token picked none of them.
void expect_matches_reference(const DeviceOutputs& out, const EpInputs& in, uint32_t device_index) {
    const std::size_t tokens = in.metadata_values.size() / kK;
    const auto leids_begin = in.leids_values.begin() + device_index * kELocal;
    const std::vector<uint16_t> leids(leids_begin, leids_begin + kELocal);
    ASSERT_EQ(out.counts.size(), kELocal);
    ASSERT_EQ(out.offsets.size(), kELocal + 1U);
    ASSERT_EQ(out.k_slot.size(), out.plan.size());
    ASSERT_EQ(out.grouped_scores.size(), out.plan.size());
    ASSERT_EQ(out.grouped.size(), out.plan.size() * kH);
    EXPECT_EQ(out.offsets[0], 0U);

    for (uint32_t e = 0; e < kELocal; ++e) {
        std::vector<uint32_t> routed;  // the tokens that picked leids[e], in token order
        for (uint32_t t = 0; t < tokens; ++t) {
            const uint16_t* routing = in.metadata_values.data() + t * kK;
            if (std::find(routing, routing + kK, leids[e]) != routing + kK) {
                routed.push_back(t);
            }
        }
        EXPECT_EQ(out.counts[e], routed.size()) << "counts[" << e << "]";
        ASSERT_LE(out.offsets[e], out.offsets[e + 1U]);
        ASSERT_LE(out.offsets[e + 1U], out.plan.size());

        std::vector<uint32_t> listed;
        uint32_t wrong_entries = 0U;
        uint32_t wrong_rows = 0U;
        for (uint32_t i = out.offsets[e]; i < out.offsets[e + 1U]; ++i) {
            const uint32_t t = out.plan[i];
            if (t == kPlanSentinel) {
                continue;
            }
            const uint32_t slot = t * kK + out.k_slot[i];
            const bool consistent = t < tokens && out.k_slot[i] < kK && in.metadata_values.flat(slot) == leids[e] &&
                                    out.grouped_scores[i] == in.scores_values.flat(slot);
            wrong_entries += consistent ? 0U : 1U;
            const float* row = out.grouped.data() + static_cast<std::size_t>(i) * kH;
            const bool hidden_row =
                t < tokens && std::equal(row, row + kH, in.dispatched_values.data() + static_cast<std::size_t>(t) * kH);
            wrong_rows += hidden_row ? 0U : 1U;
            listed.push_back(t);
        }
        EXPECT_EQ(wrong_entries, 0U) << "plan entries of expert " << e << " with a wrong k_slot or grouped_scores";
        EXPECT_EQ(wrong_rows, 0U) << "rows of grouped for expert " << e << " that are not their token's hidden row";
        std::sort(listed.begin(), listed.end());
        EXPECT_EQ(listed, routed) << "the tokens plan lists for expert " << e;
    }
    const auto tail = out.plan.begin() + out.offsets[kELocal];
    EXPECT_EQ(std::count(tail, out.plan.end(), kPlanSentinel), out.plan.end() - tail)
        << "plan entries past offsets[E_local] that are not the sentinel";

    xt::xarray<float> expected = xt::zeros<float>(in.dispatched_values.shape());
    for (uint32_t t = 0; t < tokens; ++t) {
        float weight = 0.F;
        for (uint32_t k = 0; k < kK; ++k) {
            if (std::find(leids.begin(), leids.end(), in.metadata_values.flat(t * kK + k)) != leids.end()) {
                weight += in.scores_values.flat(t * kK + k);
            }
        }
        for (uint32_t h = 0; h < kH; ++h) {
            expected.flat(t * kH + h) = weight * in.dispatched_values.flat(t * kH + h);
        }
    }
    EXPECT_TRUE(xt::allclose(out.ungrouped, expected, kRtol, kAtol)) << "ungrouped";
}

// Bit for bit: a later launch must not write into an earlier launch's buffers. The pad rows of grouped are never
// written, so any bits they hold must stay too.
void expect_unchanged(const DeviceOutputs& actual, const DeviceOutputs& expected) {
    EXPECT_EQ(ttml::test_utils::count_bit_mismatches(expected.grouped, actual.grouped), 0U) << "grouped";
    EXPECT_EQ(actual.counts, expected.counts) << "counts";
    EXPECT_EQ(actual.offsets, expected.offsets) << "offsets";
    EXPECT_EQ(actual.plan, expected.plan) << "plan";
    EXPECT_EQ(actual.k_slot, expected.k_slot) << "k_slot";
    EXPECT_EQ(actual.grouped_scores, expected.grouped_scores) << "grouped_scores";
    EXPECT_TRUE(xt::all(xt::equal(actual.ungrouped, expected.ungrouped))) << "ungrouped";
}

}  // namespace

class MoeEpMeshTest : public ttml::test_utils::Mesh1x2SuiteFixture {};

TEST_F(MoeEpMeshTest, GroupUngroupCacheHitFreshTensors) {
    auto& device = ttml::autograd::ctx().get_device();
    const uint32_t S = many_cores_s();

    // Launch A: device 0 holds experts {0, 1} and device 1 holds {2, 3}. Launch B reverses the ids, so every device
    // groups other tokens than in launch A.
    const auto a = make_inputs(S, {0, 1, 2, 3}, 1001U);
    const auto entries_before_a = device.num_program_cache_entries();
    const auto group_a = ttml::metal::moe_group(a.dispatched, a.metadata, a.scores, a.leids, kELocal, kK);
    const auto entries_after_group_a = device.num_program_cache_entries();
    ASSERT_GT(entries_after_group_a, entries_before_a) << "launch A's moe_group must build a program";
    const auto ungrouped_a = ungroup(group_a, S);
    ASSERT_GT(device.num_program_cache_entries(), entries_after_group_a)
        << "launch A's moe_ungroup must build a program";
    const auto outputs_a = read_outputs(group_a, ungrouped_a);
    ASSERT_EQ(outputs_a.size(), kNumDevices);
    for (uint32_t d = 0; d < kNumDevices; ++d) {
        SCOPED_TRACE("launch A, device " + std::to_string(d));
        expect_matches_reference(outputs_a[d], a, d);
    }

    const auto b = make_inputs(S, {3, 2, 1, 0}, 2002U);
    const auto entries_before_b = device.num_program_cache_entries();
    const auto group_b = ttml::metal::moe_group(b.dispatched, b.metadata, b.scores, b.leids, kELocal, kK);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_b) << "launch B's moe_group must reuse launch A's";
    const auto ungrouped_b = ungroup(group_b, S);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_b) << "launch B's moe_ungroup must reuse launch A's";
    expect_new_buffer(a.dispatched, b.dispatched, "dispatched");
    expect_new_buffer(a.metadata, b.metadata, "metadata");
    expect_new_buffer(a.scores, b.scores, "scores");
    expect_new_buffer(a.leids, b.leids, "local_expert_ids");
    expect_new_buffer(std::get<0>(group_a), std::get<0>(group_b), "grouped");
    expect_new_buffer(std::get<1>(group_a), std::get<1>(group_b), "grouped_scores");
    expect_new_buffer(std::get<2>(group_a), std::get<2>(group_b), "k_slot");
    expect_new_buffer(std::get<3>(group_a), std::get<3>(group_b), "counts");
    expect_new_buffer(std::get<4>(group_a), std::get<4>(group_b), "offsets");
    expect_new_buffer(std::get<5>(group_a), std::get<5>(group_b), "plan");
    expect_new_buffer(ungrouped_a, ungrouped_b, "ungrouped");
    const auto outputs_b = read_outputs(group_b, ungrouped_b);
    ASSERT_EQ(outputs_b.size(), kNumDevices);
    for (uint32_t d = 0; d < kNumDevices; ++d) {
        SCOPED_TRACE("launch B, device " + std::to_string(d));
        expect_matches_reference(outputs_b[d], b, d);
    }

    const auto outputs_a_after_b = read_outputs(group_a, ungrouped_a);
    ASSERT_EQ(outputs_a_after_b.size(), kNumDevices);
    for (uint32_t d = 0; d < kNumDevices; ++d) {
        SCOPED_TRACE("launch A's outputs after launch B, device " + std::to_string(d));
        expect_unchanged(outputs_a_after_b[d], outputs_a[d]);
    }
}
