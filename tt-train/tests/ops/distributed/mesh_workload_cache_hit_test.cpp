// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// select_target_logit and subtract_at_target derive the vocabulary window each device handles from its mesh
// coordinate, and on a program-cache hit re-apply every device's buffer addresses, window and per-call scalars. Their
// op tests open a 1x1 mesh, where the only device has TP rank 0. This runs both ops on a 1x2 mesh (an N300, or two
// Blackhole chips as on bh_quietbox_2) on fresh tensors: every launch after the first must reuse the cached workload
// and still give each device its own window, the new addresses and the new scalars. Smaller systems skip it.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/subtract_at_target/device/subtract_at_target_device_operation.hpp"
#include "test_utils/mesh_utils.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace {

constexpr uint32_t kN = 2U;
constexpr uint32_t kS = 91U;
constexpr uint32_t kLocalV = 157U;    // vocabulary shard held by each device
constexpr uint32_t kNumDevices = 2U;  // 1x2 mesh, TP along axis 1

// What a launch passes besides its tensors. The program cache key leaves all three out, so a hit must re-apply them.
struct CallScalars {
    uint32_t first_v = 0U;
    uint32_t local_V = kLocalV;   // at most kLocalV, so a window never reaches past the logits a device holds
    float subtract_value = 1.0F;  // subtract_at_target only

    bool operator==(const CallScalars&) const = default;
};

// A narrower window at another offset, and MEAN's subtract_value over four tokens where the default is NONE's 1.
constexpr CallScalars kChangedScalars{.first_v = 37U, .local_V = 100U, .subtract_value = 0.25F};

// The TP rank is the device's coordinate along cluster_axis, or its linear index when there is none. On a 1x2 mesh
// axis 1 and the linear index give each device its own window, while axis 0 gives both devices rank 0.
constexpr std::array<std::optional<uint32_t>, 3> kClusterAxes{1U, std::nullopt, 0U};

std::string describe(std::optional<uint32_t> cluster_axis) {
    return cluster_axis.has_value() ? "cluster_axis = " + std::to_string(*cluster_axis) : "no cluster_axis";
}

// [first_v, last_v) of the device at mesh coordinate (0, d).
std::pair<uint32_t, uint32_t> device_window(const CallScalars& s, std::optional<uint32_t> cluster_axis, uint32_t d) {
    const uint32_t tp_rank = cluster_axis == 0U ? 0U : d;
    const uint32_t first_v = s.first_v + tp_rank * s.local_V;
    return {first_v, first_v + s.local_V};
}

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
    const xt::xarray<float>& input,
    const xt::xarray<uint32_t>& target,
    uint32_t first_v,
    uint32_t last_v,
    float subtract_value) {
    xt::xarray<float> result = input;
    for (uint32_t n = 0; n < kN; ++n) {
        for (uint32_t s = 0; s < kS; ++s) {
            const uint32_t c = target(n, s);
            if (c >= first_v && c < last_v) {
                result(n, 0U, s, c - first_v) -= subtract_value;
            }
        }
    }
    return result;
}

// The ops as the tests launch them, each with its reference for one device's window.
struct SelectTargetLogit {
    static ttnn::Tensor launch(
        const ttnn::Tensor& logit,
        const ttnn::Tensor& target,
        std::optional<uint32_t> cluster_axis,
        const CallScalars& s) {
        return ttml::metal::select_target_logit(logit, target, s.local_V, cluster_axis, s.first_v);
    }

    static xt::xarray<float> reference(
        const xt::xarray<float>& logit,
        const xt::xarray<uint32_t>& target,
        uint32_t first_v,
        uint32_t last_v,
        const CallScalars& /*s*/) {
        return select_target_logit_reference(logit, target, first_v, last_v);
    }
};

struct SubtractAtTarget {
    static ttnn::Tensor launch(
        const ttnn::Tensor& input,
        const ttnn::Tensor& target,
        std::optional<uint32_t> cluster_axis,
        const CallScalars& s) {
        return ttml::metal::subtract_at_target(input, target, s.local_V, cluster_axis, s.first_v, s.subtract_value);
    }

    static xt::xarray<float> reference(
        const xt::xarray<float>& input,
        const xt::xarray<uint32_t>& target,
        uint32_t first_v,
        uint32_t last_v,
        const CallScalars& s) {
        return subtract_at_target_reference(input, target, first_v, last_v, s.subtract_value);
    }
};

// The vocab-parallel cross-entropy backward's call: the prim in place (preallocated_output = input). Returns the input,
// so the checks read the caller's tensor.
struct SubtractAtTargetInPlace : SubtractAtTarget {
    static ttnn::Tensor launch(
        const ttnn::Tensor& input,
        const ttnn::Tensor& target,
        std::optional<uint32_t> cluster_axis,
        const CallScalars& s) {
        const auto result = ttnn::prim::ttml_subtract_at_target(
            input, target, s.local_V, cluster_axis, s.first_v, input, s.subtract_value);
        EXPECT_EQ(result.buffer()->address(), input.buffer()->address()) << "the launch did not run in place";
        return input;
    }
};

}  // namespace

class Mesh1x2WorkloadCacheHitTest : public ttml::test_utils::Mesh1x2SuiteFixture {
public:
    static void SetUpTestSuite() {
        Mesh1x2SuiteFixture::SetUpTestSuite();
        if (ttml::test_utils::system_supports_mesh(mesh_shape())) {
            ttml::autograd::ctx().set_seed(42);
        }
    }

protected:
    // Launches Op once per entry of `calls`, each on fresh replicated inputs of the same shape, and checks every
    // device's result against the reference for that device's window and the call's scalars. The first launch must
    // populate the program cache, every later one must reuse it. Earlier launches' tensors stay alive, so a later one
    // cannot land at an earlier one's addresses, and their results are re-read at the end: a hit must not write into
    // them.
    template <typename Op>
    void run_launches(std::optional<uint32_t> cluster_axis, const std::vector<CallScalars>& calls) {
        using namespace ttml;
        auto* device = &autograd::ctx().get_device();
        ASSERT_EQ(device->num_devices(), kNumDevices);
        // The mesh is shared across the suite; start from an empty program cache so the first launch is a cache miss
        // even under --gtest_repeat.
        device->clear_program_cache();
        auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);

        struct Launch {
            ttnn::Tensor input;
            ttnn::Tensor target;
            ttnn::Tensor result;
            std::vector<xt::xarray<float>> result_values;  // per device, read right after the launch
        };
        std::vector<Launch> launches;
        for (uint32_t launch = 0; launch < calls.size(); ++launch) {
            const auto& scalars = calls[launch];
            const auto input_t = make_random_4d(100U + launch);
            const auto target_t = make_random_targets(200U + launch);
            auto input_dev = core::from_xtensor(input_t, device, ttnn::Layout::TILE, mapper.get());
            auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
                target_t, device, ttnn::Layout::ROW_MAJOR, mapper.get());

            const auto entries_before = device->num_program_cache_entries();
            auto result = Op::launch(input_dev, target_dev, cluster_axis, scalars);
            if (launch == 0U) {
                ASSERT_GT(device->num_program_cache_entries(), entries_before)
                    << "first launch did not populate the program cache";
            } else {
                EXPECT_EQ(device->num_program_cache_entries(), entries_before)
                    << "launch " << launch << " did not reuse the cached mesh workload";
                EXPECT_NE(input_dev.buffer()->address(), launches.front().input.buffer()->address());
                EXPECT_NE(target_dev.buffer()->address(), launches.front().target.buffer()->address());
                EXPECT_NE(result.buffer()->address(), launches.front().result.buffer()->address());
            }

            auto per_device = core::to_xtensor(result, core::IdentityComposer{});
            ASSERT_EQ(per_device.size(), kNumDevices);
            for (uint32_t d = 0; d < kNumDevices; ++d) {
                const auto [first_v, last_v] = device_window(scalars, cluster_axis, d);
                const auto expected = Op::reference(input_t, target_t, first_v, last_v, scalars);
                ASSERT_EQ(per_device[d].shape(), expected.shape());
                EXPECT_TRUE(xt::allclose(per_device[d], expected, /*rtol=*/3e-2F, /*atol=*/1e-2F))
                    << "launch " << launch << ", device " << d;
                if (scalars != calls.front()) {
                    // Else a hit that kept the first launch's scalars would pass.
                    const auto [stale_first_v, stale_last_v] = device_window(calls.front(), cluster_axis, d);
                    ASSERT_FALSE(xt::allclose(
                        Op::reference(input_t, target_t, stale_first_v, stale_last_v, calls.front()),
                        expected,
                        /*rtol=*/3e-2F,
                        /*atol=*/1e-2F))
                        << "launch " << launch << ", device " << d << ": the scalars do not change the result";
                }
            }
            launches.push_back({std::move(input_dev), std::move(target_dev), std::move(result), std::move(per_device)});
        }

        for (uint32_t launch = 0; launch + 1U < launches.size(); ++launch) {
            const auto per_device = core::to_xtensor(launches[launch].result, core::IdentityComposer{});
            for (uint32_t d = 0; d < kNumDevices; ++d) {
                EXPECT_TRUE(per_device[d] == launches[launch].result_values[d])
                    << "a later launch wrote into the result of launch " << launch << " on device " << d;
            }
        }
    }
};

TEST_F(Mesh1x2WorkloadCacheHitTest, SelectTargetLogitPerDeviceWindowSurvivesCacheHit) {
    run_launches<SelectTargetLogit>(/*cluster_axis=*/1U, {CallScalars{}, CallScalars{}});
}

TEST_F(Mesh1x2WorkloadCacheHitTest, SubtractAtTargetPerDeviceWindowSurvivesCacheHit) {
    run_launches<SubtractAtTarget>(/*cluster_axis=*/1U, {CallScalars{}, CallScalars{}});
}

// first_v and local_V are not part of the program cache key: a hit must re-derive every device's window from them.
TEST_F(Mesh1x2WorkloadCacheHitTest, SelectTargetLogitChangedWindowSurvivesCacheHit) {
    for (const auto cluster_axis : kClusterAxes) {
        SCOPED_TRACE(describe(cluster_axis));
        run_launches<SelectTargetLogit>(cluster_axis, {CallScalars{}, kChangedScalars});
    }
}

// first_v, local_V and subtract_value are not part of the program cache key: a hit must re-apply all three on every
// device.
TEST_F(Mesh1x2WorkloadCacheHitTest, SubtractAtTargetChangedScalarsSurviveCacheHit) {
    for (const auto cluster_axis : kClusterAxes) {
        SCOPED_TRACE(describe(cluster_axis));
        run_launches<SubtractAtTarget>(cluster_axis, {CallScalars{}, kChangedScalars});
    }
}

// In place, as the vocab-parallel cross-entropy backward calls it: the output is the input on every launch.
TEST_F(Mesh1x2WorkloadCacheHitTest, SubtractAtTargetInPlaceSurvivesCacheHit) {
    for (const auto cluster_axis : kClusterAxes) {
        SCOPED_TRACE(describe(cluster_axis));
        run_launches<SubtractAtTargetInPlace>(cluster_axis, {CallScalars{}, kChangedScalars});
    }
}
