// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// select_target_logit and subtract_at_target build one program per mesh coordinate, because the vocabulary window
// a device handles depends on its TP rank, and on a program-cache hit patch the buffer addresses of every device's
// program. Their op tests open a 1x1 mesh, so neither the per-device windows nor the mesh-workload cache-hit path is
// exercised there. This runs both ops on the N300's 1x2 mesh twice with fresh tensors: the second launch must reuse
// the cached workload and still give each device its own window and the new addresses.

#include <gtest/gtest.h>

#include <random>
#include <umd/device/cluster.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace {

bool board_is_n300() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0) == tt::BoardType::N300;
}

constexpr uint32_t kN = 2U;
constexpr uint32_t kS = 91U;
constexpr uint32_t kLocalV = 157U;    // vocabulary shard held by each device
constexpr uint32_t kNumDevices = 2U;  // 1x2 mesh, TP along axis 1

xt::xarray<float> make_random_4d(uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-5.F, 5.F);
    xt::xarray<float> t = xt::empty<float>({kN, 1U, kS, kLocalV});
    for (auto& v : t) {
        v = dist(gen);
    }
    return t;
}

// Targets over the whole vocabulary, so each device sees hits inside and outside its window.
xt::xarray<uint32_t> make_random_targets(uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0U, kNumDevices * kLocalV - 1U);
    xt::xarray<uint32_t> t = xt::zeros<uint32_t>({kN, kS});
    for (auto& v : t) {
        v = dist(gen);
    }
    return t;
}

xt::xarray<float> select_target_logit_reference(
    const xt::xarray<float>& logit, const xt::xarray<uint32_t>& target, uint32_t first_v, uint32_t last_v) {
    xt::xarray<float> result = xt::zeros<float>({kN, 1U, kS, 1U});
    for (uint32_t n = 0; n < kN; ++n) {
        for (uint32_t s = 0; s < kS; ++s) {
            const uint32_t c = target(n, s);
            if (c >= first_v && c < last_v) {
                result(n, 0U, s, 0U) = logit(n, 0U, s, c - first_v);
            }
        }
    }
    return result;
}

xt::xarray<float> subtract_at_target_reference(
    const xt::xarray<float>& input, const xt::xarray<uint32_t>& target, uint32_t first_v, uint32_t last_v) {
    xt::xarray<float> result = input;
    for (uint32_t n = 0; n < kN; ++n) {
        for (uint32_t s = 0; s < kS; ++s) {
            const uint32_t c = target(n, s);
            if (c >= first_v && c < last_v) {
                result(n, 0U, s, c - first_v) -= 1.0F;
            }
        }
    }
    return result;
}

}  // namespace

class N300MeshWorkloadCacheHitTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!board_is_n300()) {
            GTEST_SKIP() << "Needs the N300's 1x2 mesh";
        }
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, kNumDevices));
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }

    // Launches `op` twice with fresh replicated inputs of the same shape and checks every device's result against
    // `reference` for that device's window. The first launch must populate the program cache, the second must not add
    // to it. Earlier launches' tensors stay alive so a later one cannot land at an earlier one's address.
    template <typename Op, typename Reference>
    void run_twice(Op&& op, Reference&& reference) {
        using namespace ttml;
        auto* device = &autograd::ctx().get_device();
        ASSERT_EQ(device->num_devices(), kNumDevices);
        auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);

        std::vector<ttnn::Tensor> alive;
        const auto entries_start = device->num_program_cache_entries();
        auto entries_after_first = entries_start;
        for (uint32_t launch = 0; launch < 2U; ++launch) {
            const auto input_t = make_random_4d(100U + launch);
            const auto target_t = make_random_targets(200U + launch);
            auto input_dev = core::from_xtensor(input_t, device, ttnn::Layout::TILE, mapper.get());
            auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
                target_t, device, ttnn::Layout::ROW_MAJOR, mapper.get());

            auto result = op(input_dev, target_dev);
            alive.push_back(input_dev);
            alive.push_back(target_dev);
            alive.push_back(result);

            if (launch == 0U) {
                entries_after_first = device->num_program_cache_entries();
                ASSERT_GT(entries_after_first, entries_start) << "first launch did not populate the program cache";
            } else {
                EXPECT_EQ(device->num_program_cache_entries(), entries_after_first)
                    << "second launch did not reuse the cached mesh workload";
            }

            const auto per_device = core::to_xtensor(result, core::IdentityComposer{});
            ASSERT_EQ(per_device.size(), kNumDevices);
            for (uint32_t d = 0; d < kNumDevices; ++d) {
                const auto expected = reference(input_t, target_t, d * kLocalV, (d + 1U) * kLocalV);
                ASSERT_EQ(per_device[d].shape(), expected.shape());
                EXPECT_TRUE(xt::allclose(per_device[d], expected, /*rtol=*/3e-2F, /*atol=*/1e-2F))
                    << "launch " << launch << ", device " << d;
            }
        }
    }
};

TEST_F(N300MeshWorkloadCacheHitTest, SelectTargetLogitPerDeviceWindowSurvivesCacheHit) {
    run_twice(
        [](const ttnn::Tensor& logit, const ttnn::Tensor& target) {
            return ttml::metal::select_target_logit(logit, target, kLocalV, /*cluster_axis=*/1U);
        },
        select_target_logit_reference);
}

TEST_F(N300MeshWorkloadCacheHitTest, SubtractAtTargetPerDeviceWindowSurvivesCacheHit) {
    run_twice(
        [](const ttnn::Tensor& input, const ttnn::Tensor& target) {
            return ttml::metal::subtract_at_target(input, target, kLocalV, /*cluster_axis=*/1U);
        },
        subtract_at_target_reference);
}
