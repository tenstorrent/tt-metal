// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gumbel_sample builds one program per mesh coordinate: each device's compute kernels seed their RNG with
// the device's index along the seeded mesh axes (seed_axes), so data-parallel devices draw distinct noise
// and replicas draw identical noise. ops/gumbel_sample_op_test.cpp runs on a 1x1 mesh, where neither the
// per-device streams nor a cache hit on a multi-program workload exists. These tests run on a 1x2 mesh and
// launch the op several times on one program key: the hits must re-apply the new seed, token count and
// buffer addresses on EVERY device's program and keep each device's own stream.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/gumbel_sample/gumbel_sample.hpp"
#include "test_utils/mesh_utils.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace {

using ttml::test_utils::blocks_for_two_core_groups;

constexpr uint32_t kNumDevices = 2U;
constexpr uint32_t kBatch = 2U;
constexpr uint32_t kVocab = 77U;  // Wt = 3 with a ragged last tile
constexpr uint32_t kWt = 3U;
// Logits in [-1, 1] at temperature 1: the noise decides most rows, so two devices on different streams
// disagree on most of the rows each one samples, and so do two seeds on one device.
constexpr float kTemperature = 1.0F;
constexpr uint32_t kFirstSeed = 1234U;
constexpr uint32_t kSecondSeed = 5678U;

// Tokens of a [kBatch, 1, T, kVocab] every-row input whose split has two core groups. Each entry's last tile
// row keeps 27 of its 32 rows.
uint32_t two_core_group_tokens() {
    return blocks_for_two_core_groups(kBatch * kWt) * 32U - 5U;
}

// Per-entry positions walking the token range with stride 13, coprime with every token count used here.
std::vector<uint32_t> walk_positions(uint32_t batch, uint32_t tokens, uint32_t offset) {
    std::vector<uint32_t> positions(batch);
    for (uint32_t b = 0; b < batch; ++b) {
        positions[b] = (b * 13U + offset) % tokens;
    }
    return positions;
}

// Every device's sampled ids, in mesh order.
std::vector<std::vector<uint32_t>> ids_per_device(const ttnn::Tensor& sampled) {
    std::vector<std::vector<uint32_t>> ids;
    for (const auto& shard : ttml::core::to_xtensor<uint32_t>(sampled, ttml::core::IdentityComposer{})) {
        ids.emplace_back(shard.begin(), shard.end());
    }
    return ids;
}

}  // namespace

class GumbelSampleMeshTest : public ttml::test_utils::Mesh1x2SuiteFixture {
protected:
    // Three launches on REPLICATED logits and one program key, with every tensor kept alive: the miss with
    // kFirstSeed; a hit on a fresh tensor holding the SAME logits with kSecondSeed, so every change in a device's
    // ids is the seed's; and a hit on a fresh tensor holding OTHER logits. `devices_differ` says whether seed_axes
    // gives the two devices distinct RNG streams.
    static void run_seed_change_on_cache_hit(const std::vector<uint32_t>& seed_axes, bool devices_differ) {
        auto& device = ttml::autograd::ctx().get_device();
        ASSERT_EQ(device.num_devices(), kNumDevices);
        const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(device);
        const std::array<uint32_t, 4> shape = {kBatch, 1U, two_core_group_tokens(), kVocab};
        const auto values = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, 42U);
        const auto other_values = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, 43U);
        auto upload = [&](const xt::xarray<float>& logits) {
            return ttml::core::from_xtensor(logits, &device, ttnn::Layout::TILE, replicate.get());
        };
        auto sample = [&](const ttnn::Tensor& logits, uint32_t seed) {
            return ttml::metal::gumbel_sample(logits, kTemperature, seed, seed_axes);
        };

        auto first_logits = upload(values);
        const auto entries_before_first = device.num_program_cache_entries();
        auto first_output = sample(first_logits, kFirstSeed);
        ASSERT_GT(device.num_program_cache_entries(), entries_before_first)
            << "the first launch did not populate the program cache";
        const auto first_ids = ids_per_device(first_output);

        auto second_logits = upload(values);
        const auto entries_before_second = device.num_program_cache_entries();
        auto second_output = sample(second_logits, kSecondSeed);
        EXPECT_EQ(device.num_program_cache_entries(), entries_before_second) << "a new seed must reuse the workload";
        const auto second_ids = ids_per_device(second_output);

        auto third_logits = upload(other_values);
        const auto entries_before_third = device.num_program_cache_entries();
        auto third_output = sample(third_logits, kSecondSeed);
        EXPECT_EQ(device.num_program_cache_entries(), entries_before_third) << "new logits must reuse the workload";
        const auto third_ids = ids_per_device(third_output);

        EXPECT_NE(first_logits.buffer()->address(), second_logits.buffer()->address());
        EXPECT_NE(second_logits.buffer()->address(), third_logits.buffer()->address());
        EXPECT_NE(first_output.buffer()->address(), second_output.buffer()->address());
        EXPECT_NE(second_output.buffer()->address(), third_output.buffer()->address());

        ASSERT_EQ(first_ids.size(), kNumDevices);
        ASSERT_EQ(second_ids.size(), kNumDevices);
        ASSERT_EQ(third_ids.size(), kNumDevices);
        for (uint32_t d = 0; d < kNumDevices; ++d) {
            EXPECT_NE(second_ids[d], first_ids[d]) << "device " << d << " kept the first seed on the cache hit";
        }
        for (const auto* ids : {&first_ids, &second_ids, &third_ids}) {
            if (devices_differ) {
                EXPECT_NE((*ids)[0], (*ids)[1]) << "the two devices drew the same noise";
            } else {
                EXPECT_EQ((*ids)[0], (*ids)[1]) << "the two replicas drew different noise";
            }
        }

        // No hit may write into an earlier launch's output, on any device.
        EXPECT_EQ(ids_per_device(first_output), first_ids) << "launch 1's output changed during a later launch";
        EXPECT_EQ(ids_per_device(second_output), second_ids) << "launch 2's output changed during launch 3";

        // Each hit, rebuilt from an empty program cache, must reproduce its ids bit for bit on every device.
        device.clear_program_cache();
        EXPECT_EQ(ids_per_device(sample(second_logits, kSecondSeed)), second_ids) << "launch 2 differs from a miss";
        device.clear_program_cache();
        EXPECT_EQ(ids_per_device(sample(third_logits, kSecondSeed)), third_ids) << "launch 3 differs from a miss";
    }
};

TEST_F(GumbelSampleMeshTest, TestSamplingFsdpSeedChangeOnCacheHit) {
    // seed_axes {1}: the two devices of the 1x2 mesh are data-parallel (FSDP / DP) ranks and must draw
    // distinct noise, on the miss and on every hit.
    run_seed_change_on_cache_hit({1U}, /* devices_differ */ true);
}

TEST_F(GumbelSampleMeshTest, TestSamplingTpSeedChangeOnCacheHit) {
    // seed_axes {}: the two devices are tensor-parallel replicas and must draw identical noise, before and
    // after the seed changes on a cache hit.
    run_seed_change_on_cache_hit({}, /* devices_differ */ false);
}

TEST_F(GumbelSampleMeshTest, TestSamplingFsdpPositionModeTokenCountChangeOnCacheHit) {
    // Position mode with the logits and positions sharded over the batch, as GRPO prefill runs it, and seed_axes
    // {1}. The token count is not hashed in position mode, so the 134-token launch reuses the 70-token workload.
    // Device 1's RNG stream ids (device index * B * Wt + start tile) must stay the build's: a hit that re-derives
    // them from the token count changes device 1's noise, which device 0 (index 0) and the 1x1 tests cannot show.
    auto& device = ttml::autograd::ctx().get_device();
    ASSERT_EQ(device.num_devices(), kNumDevices);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(device, 0);
    const uint32_t local_batch = blocks_for_two_core_groups(kWt);  // position mode splits B * Wt tiles per device
    const uint32_t batch = kNumDevices * local_batch;
    const std::vector<uint32_t> seed_axes = {1U};

    struct Launch {
        ttnn::Tensor logits;
        ttnn::Tensor positions;
        uint32_t seed{};
        ttnn::Tensor output;
        std::vector<std::vector<uint32_t>> ids;  // per device, read right after the launch
    };
    auto make_launch = [&](uint32_t tokens, uint32_t data_seed, uint32_t position_offset, uint32_t seed) {
        Launch launch;
        launch.logits = ttml::core::from_xtensor(
            ttml::test_utils::make_uniform_xarray<float>(
                std::array<uint32_t, 4>{batch, 1U, tokens, kVocab}, -1.0F, 1.0F, data_seed),
            &device,
            ttnn::Layout::TILE,
            shard.get());
        launch.positions = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            walk_positions(batch, tokens, position_offset),
            ttnn::Shape({batch, 1U, 1U, 1U}),
            &device,
            ttnn::Layout::ROW_MAJOR,
            shard.get());
        launch.seed = seed;
        return launch;
    };
    auto sample = [&](const Launch& launch) {
        return ttml::metal::gumbel_sample(
            launch.logits, kTemperature, launch.seed, seed_axes, /* logits_mask */ std::nullopt, launch.positions);
    };

    auto first = make_launch(70U, 42U, 5U, kFirstSeed);
    const auto entries_before_first = device.num_program_cache_entries();
    first.output = sample(first);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_first)
        << "the first launch did not populate the program cache";
    first.ids = ids_per_device(first.output);

    // Positions up to 133, past the first launch's 70 tokens.
    auto second = make_launch(134U, 43U, 9U, kSecondSeed);
    const auto entries_before_second = device.num_program_cache_entries();
    second.output = sample(second);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_second)
        << "the 134-token launch must reuse the 70-token workload";
    second.ids = ids_per_device(second.output);

    EXPECT_NE(first.logits.buffer()->address(), second.logits.buffer()->address());
    EXPECT_NE(first.positions.buffer()->address(), second.positions.buffer()->address());
    EXPECT_NE(first.output.buffer()->address(), second.output.buffer()->address());
    ASSERT_EQ(first.ids.size(), kNumDevices);
    ASSERT_EQ(second.ids.size(), kNumDevices);
    for (uint32_t d = 0; d < kNumDevices; ++d) {
        ASSERT_EQ(second.ids[d].size(), local_batch) << "device " << d << ": one id per local batch entry";
        EXPECT_TRUE(std::all_of(second.ids[d].begin(), second.ids[d].end(), [](uint32_t id) { return id < kVocab; }))
            << "device " << d << " sampled past the vocabulary";
    }
    EXPECT_EQ(ids_per_device(first.output), first.ids) << "the first launch's output changed during the hit";

    device.clear_program_cache();
    const auto rebuilt = ids_per_device(sample(second));
    ASSERT_EQ(rebuilt.size(), kNumDevices);
    for (uint32_t d = 0; d < kNumDevices; ++d) {
        EXPECT_EQ(second.ids[d], rebuilt[d]) << "device " << d << ": the hit differs from the same launch as a miss";
    }
}
