// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Device tests for RingLaneSelection::ZigZag in the ring SDPA ops.
 *
 * With zig-zag lane selection one ring_sdpa_fw / ring_sdpa_bw call carries two sets of
 * tensors ("lanes"), and the ring program factory binds each device's program to the lane
 * the schedule assigns it (lane 0 when src < d, lane 1 when src > d, nothing on the
 * diagonal step). These tests replicate distinct random inputs per lane to every device,
 * run every ring step in both ring directions, and check on each device that
 *   - the assigned lane's outputs match an unmasked SDPA reference of that lane's inputs,
 *   - the other lane's outputs are left untouched (pre-filled sentinels survive).
 * Every step runs twice: a program-cache miss (create_mesh_workload) and then a hit with
 * fresh inputs and outputs at new buffer addresses (override_runtime_arguments), so both
 * paths that bind lanes are exercised.
 * They use the whole system mesh with the ring along its longest axis, so they run on a
 * LoudBox (1x8, TT_MESH_GRAPH_DESC_PATH=bh_lb_1x8_line_graph_descriptor) as well as a
 * Galaxy (8x4) and skip on single-device systems.
 */

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <core/xtensor_utils.hpp>
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ops/ring_sdpa_bw/ring_sdpa_bw.hpp"
#include "metal/ops/ring_sdpa_fw/ring_sdpa_fw.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using ttml::metal::AttentionMaskType;
using ttml::metal::ops::get_ring_step_work;
using ttml::metal::ops::RingLaneSelection;
using ttml::ttnn_fixed::distributed::RingShiftDirection;

constexpr float kSentinel = 7.0F;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
// B, Q heads, KV heads (GQA), lane sequence length C (4 tiles), head dim.
constexpr size_t kB = 1, kH = 4, kG = 2, kC = 128, kD = 64;
// Q is drawn from U(-kQScale, kQScale) so the scaled scores have std ~1: a near-uniform softmax would
// make the gradients too insensitive to the lane's K and u = rowsum(dO * O) to catch a wrong binding.
constexpr float kQScale = 3.0F;
// Relative L2 error accepted against the FP32 host reference (bf16 inputs and outputs).
constexpr double kForwardRelTol = 2e-2;
constexpr double kBackwardRelTol = 5e-2;

tt::tt_metal::distributed::MeshShape system_mesh_shape() {
    return tt::tt_metal::distributed::SystemMesh::instance().shape();
}

// Ring along the longest mesh axis (LoudBox 1x8 -> axis 1, Galaxy 8x4 -> axis 0).
uint32_t ring_axis_of(const tt::tt_metal::distributed::MeshShape& shape) {
    uint32_t axis = 0;
    for (uint32_t i = 1; i < shape.dims(); ++i) {
        if (shape[i] > shape[axis]) {
            axis = i;
        }
    }
    return axis;
}

class RingSDPALaneTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (system_mesh_shape().mesh_size() > 1U) {
            ttml::ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(system_mesh_shape().mesh_size()));
            ttml::autograd::ctx().open_device(system_mesh_shape());
        }
    }
    static void TearDownTestSuite() {
        if (system_mesh_shape().mesh_size() > 1U) {
            ttml::autograd::ctx().close_device();
        }
    }
    void SetUp() override {
        if (system_mesh_shape().mesh_size() <= 1U) {
            GTEST_SKIP() << "Ring SDPA lane tests need a multi-device mesh";
        }
        ttml::autograd::ctx().get_device().clear_program_cache();
    }
};

xt::xarray<float> to_bf16(const xt::xarray<float>& x) {
    xt::xarray<float> rounded = x;
    for (auto& value : rounded) {
        value = static_cast<float>(bfloat16(value));
    }
    return rounded;
}

xt::xarray<float> uniform(const std::array<size_t, 4>& shape, float bound, uint32_t seed) {
    return to_bf16(ttml::test_utils::make_uniform_xarray<float>(shape, -bound, bound, seed));
}

// Host reference: unmasked GQA SDPA on (B, H, S, D) queries and (B, G, S, D) keys/values.
struct ForwardReference {
    xt::xarray<float> output;         // (B, H, S, D)
    xt::xarray<float> lse;            // (B, H, S): log sum_j exp(scale * q.k_j)
    xt::xarray<float> probabilities;  // (B, H, S, S)
};

ForwardReference reference_forward(const xt::xarray<float>& q, const xt::xarray<float>& k, const xt::xarray<float>& v) {
    const size_t B = q.shape()[0], H = q.shape()[1], S = q.shape()[2], D = q.shape()[3];
    const size_t group = H / k.shape()[1];
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    ForwardReference ref{xt::zeros<float>({B, H, S, D}), xt::zeros<float>({B, H, S}), xt::zeros<float>({B, H, S, S})};
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t g = h / group;
            for (size_t i = 0; i < S; ++i) {
                std::vector<double> scores(S);
                double max_score = -std::numeric_limits<double>::infinity();
                for (size_t j = 0; j < S; ++j) {
                    double dot = 0.0;
                    for (size_t d = 0; d < D; ++d) {
                        dot += static_cast<double>(q(b, h, i, d)) * k(b, g, j, d);
                    }
                    scores[j] = dot * scale;
                    max_score = std::max(max_score, scores[j]);
                }
                double sum = 0.0;
                for (size_t j = 0; j < S; ++j) {
                    sum += std::exp(scores[j] - max_score);
                }
                ref.lse(b, h, i) = static_cast<float>(max_score + std::log(sum));
                for (size_t j = 0; j < S; ++j) {
                    const double p = std::exp(scores[j] - max_score) / sum;
                    ref.probabilities(b, h, i, j) = static_cast<float>(p);
                    for (size_t d = 0; d < D; ++d) {
                        ref.output(b, h, i, d) += static_cast<float>(p * v(b, g, j, d));
                    }
                }
            }
        }
    }
    return ref;
}

struct Gradients {
    xt::xarray<float> dq;
    xt::xarray<float> dk;
    xt::xarray<float> dv;
};

// SDPA backward with the softmax correction u = rowsum(dO * O) taken from the supplied O,
// exactly as the device kernels do with the attention output they are given.
Gradients reference_backward(
    const xt::xarray<float>& q,
    const xt::xarray<float>& k,
    const xt::xarray<float>& v,
    const xt::xarray<float>& o,
    const xt::xarray<float>& probabilities,
    const xt::xarray<float>& d_o) {
    const size_t B = q.shape()[0], H = q.shape()[1], S = q.shape()[2], D = q.shape()[3];
    const size_t group = H / k.shape()[1];
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    Gradients grads{xt::zeros<float>(q.shape()), xt::zeros<float>(k.shape()), xt::zeros<float>(v.shape())};
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t g = h / group;
            for (size_t i = 0; i < S; ++i) {
                double u = 0.0;
                for (size_t d = 0; d < D; ++d) {
                    u += static_cast<double>(d_o(b, h, i, d)) * o(b, h, i, d);
                }
                for (size_t j = 0; j < S; ++j) {
                    const double p = probabilities(b, h, i, j);
                    double dp = 0.0;
                    for (size_t d = 0; d < D; ++d) {
                        dp += static_cast<double>(d_o(b, h, i, d)) * v(b, g, j, d);
                        grads.dv(b, g, j, d) += static_cast<float>(p * d_o(b, h, i, d));
                    }
                    const double ds = p * (dp - u) * scale;
                    for (size_t d = 0; d < D; ++d) {
                        grads.dq(b, h, i, d) += static_cast<float>(ds * k(b, g, j, d));
                        grads.dk(b, g, j, d) += static_cast<float>(ds * q(b, h, i, d));
                    }
                }
            }
        }
    }
    return grads;
}

double rel_l2(const xt::xarray<float>& actual, const xt::xarray<float>& expected) {
    double diff = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double e = expected.flat(i);
        const double a = actual.flat(i);
        diff += (a - e) * (a - e);
        norm += e * e;
    }
    return std::sqrt(diff / std::max(norm, 1e-30));
}

// Host data of one lane: bf16-rounded inputs and FP32 references.
struct Lane {
    xt::xarray<float> q, k, v;
    xt::xarray<float> o;    // attention output fed to backward (independent of the forward, see make_lanes)
    xt::xarray<float> d_o;  // upstream gradient
    ForwardReference fw;
    Gradients bw;
};

// Two lanes with independent random data. For backward each lane gets an independent random
// attention output O, so u = rowsum(dO * O) is O(1) and dominates dS: a device that read the
// other lane's O (or a stale u_scaler) produces gradients far outside the tolerance, which the
// test asserts explicitly via the cross-lane reference.
std::array<Lane, 2> make_lanes(uint32_t seed) {
    std::array<Lane, 2> lanes;
    for (uint32_t i = 0; i < 2; ++i) {
        auto& lane = lanes[i];
        const uint32_t s = seed + 100U * i;
        lane.q = uniform({kB, kH, kC, kD}, kQScale, s);
        lane.k = uniform({kB, kG, kC, kD}, 1.0F, s + 1);
        lane.v = uniform({kB, kG, kC, kD}, 1.0F, s + 2);
        lane.o = uniform({kB, kH, kC, kD}, 1.0F, s + 3);
        lane.d_o = uniform({kB, kH, kC, kD}, 1.0F, s + 4);
        lane.fw = reference_forward(lane.q, lane.k, lane.v);
        lane.bw = reference_backward(lane.q, lane.k, lane.v, lane.o, lane.fw.probabilities, lane.d_o);
    }
    return lanes;
}

// LSE in the (B, H, S, 32) FP32 intermediates layout the SDPA kernels use (column 0).
xt::xarray<float> lse_to_intermediates(const xt::xarray<float>& lse) {
    const size_t B = lse.shape()[0], H = lse.shape()[1], S = lse.shape()[2];
    xt::xarray<float> out = xt::zeros<float>({B, H, S, size_t{32}});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
                out(b, h, s, 0) = lse(b, h, s);
            }
        }
    }
    return out;
}

xt::xarray<float> lse_column(const xt::xarray<float>& intermediates) {
    return xt::view(intermediates, xt::all(), xt::all(), xt::all(), 0);
}

ttnn::Tensor to_device_bf16(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        x, &ttml::autograd::ctx().get_device(), ttnn::Layout::TILE);
}

ttnn::Tensor to_device_fp32(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        x, &ttml::autograd::ctx().get_device(), ttnn::Layout::TILE);
}

ttnn::Tensor filled(const ttnn::Shape& shape, float value, ttnn::DataType dtype) {
    return ttnn::full(shape, value, dtype, ttnn::Layout::TILE, std::ref(ttml::autograd::ctx().get_device()));
}

std::vector<xt::xarray<float>> per_device(const ttnn::Tensor& tensor) {
    return ttml::core::to_xtensor<float>(tensor, ttml::core::IdentityComposer{});
}

// Ring position of the device at a row-major mesh index (the order get_device_tensors returns).
uint32_t ring_id_of(uint32_t device_index, const tt::tt_metal::distributed::MeshShape& shape, uint32_t ring_axis) {
    uint32_t stride = 1;
    for (uint32_t i = shape.dims(); i-- > ring_axis + 1;) {
        stride *= shape[i];
    }
    return (device_index / stride) % shape[ring_axis];
}

bool all_equal(const xt::xarray<float>& x, float value) {
    return std::all_of(x.begin(), x.end(), [value](float v) { return v == value; });
}

struct RingSetup {
    tt::tt_metal::distributed::MeshShape mesh_shape;
    uint32_t ring_axis;
    uint32_t ring_size;
    uint32_t num_devices;
};

RingSetup ring_setup() {
    const auto mesh_shape = ttml::autograd::ctx().get_device().shape();
    const uint32_t ring_axis = ring_axis_of(mesh_shape);
    return RingSetup{mesh_shape, ring_axis, mesh_shape[ring_axis], static_cast<uint32_t>(mesh_shape.mesh_size())};
}

// Lane the schedule assigns to a device; every device runs a lane at steps 1..N-1.
uint32_t active_lane(uint32_t ring_id, uint32_t step, const RingSetup& ring, RingShiftDirection direction) {
    const auto work = get_ring_step_work(
        ring_id, step, ring.ring_size, AttentionMaskType::Causal, direction, RingLaneSelection::ZigZag);
    EXPECT_TRUE(work.execute);
    EXPECT_EQ(work.mask_type, AttentionMaskType::None);
    return work.lane;
}

// Runs `launch` for every (direction, step) twice: a program-cache miss, then a hit with fresh data at
// new buffer addresses (tensors of the first pass stay alive, so the allocator cannot reuse them).
template <typename Launch>
void for_each_direction_step_and_pass(const RingSetup& ring, Launch&& launch) {
    auto& device = ttml::autograd::ctx().get_device();
    for (auto direction : {RingShiftDirection::Backward, RingShiftDirection::Forward}) {
        for (uint32_t step = 1; step < ring.ring_size; ++step) {
            std::vector<ttnn::Tensor> keep_alive;
            for (uint32_t pass = 0; pass < 2; ++pass) {
                SCOPED_TRACE(fmt::format(
                    "direction {} step {} pass {}",
                    direction == RingShiftDirection::Backward ? "Backward" : "Forward",
                    step,
                    pass));
                const auto entries_before = device.num_program_cache_entries();
                launch(
                    direction,
                    step,
                    /*seed=*/1000U * step + 10U * pass + (direction == RingShiftDirection::Forward ? 5U : 0U),
                    keep_alive);
                if (pass == 0) {
                    EXPECT_GT(device.num_program_cache_entries(), entries_before) << "first launch must build programs";
                } else {
                    EXPECT_EQ(device.num_program_cache_entries(), entries_before)
                        << "second launch must hit the program cache (override_runtime_arguments path)";
                }
            }
        }
    }
}

}  // namespace

TEST_F(RingSDPALaneTest, ZigZagForwardBindsAssignedLanePerDevice) {
    const auto ring = ring_setup();
    const ttnn::Shape out_shape{kB, kH, kC, kD};
    const ttnn::Shape inter_shape{kB, kH, kC, 32};

    for_each_direction_step_and_pass(
        ring, [&](RingShiftDirection direction, uint32_t step, uint32_t seed, std::vector<ttnn::Tensor>& keep_alive) {
            const auto lanes = make_lanes(seed);
            std::array<ttnn::Tensor, 2> q, k, v, out, inter;
            for (uint32_t lane = 0; lane < 2; ++lane) {
                q[lane] = to_device_bf16(lanes[lane].q);
                k[lane] = to_device_bf16(lanes[lane].k);
                v[lane] = to_device_bf16(lanes[lane].v);
                out[lane] = filled(out_shape, kSentinel, ttnn::DataType::BFLOAT16);
                inter[lane] = filled(inter_shape, kNegInf, ttnn::DataType::FLOAT32);
                keep_alive.insert(keep_alive.end(), {q[lane], k[lane], v[lane], out[lane], inter[lane]});
            }

            ttml::metal::ring_sdpa_fw(
                q[0],
                k[0],
                v[0],
                ring.ring_size,
                ring.ring_axis,
                step,
                AttentionMaskType::Causal,
                direction,
                out[0],
                inter[0],
                RingLaneSelection::ZigZag,
                ttml::metal::ops::ring_sdpa_fw::RingSDPAFwLane{
                    .query = q[1], .key = k[1], .value = v[1], .output = out[1], .intermediates = inter[1]});

            const std::array<std::vector<xt::xarray<float>>, 2> out_host{per_device(out[0]), per_device(out[1])};
            const std::array<std::vector<xt::xarray<float>>, 2> inter_host{per_device(inter[0]), per_device(inter[1])};

            for (uint32_t dev = 0; dev < ring.num_devices; ++dev) {
                const uint32_t ring_id = ring_id_of(dev, ring.mesh_shape, ring.ring_axis);
                const uint32_t active = active_lane(ring_id, step, ring, direction);
                const uint32_t idle = 1U - active;
                SCOPED_TRACE(fmt::format("device {} (ring id {}) active lane {}", dev, ring_id, active));

                EXPECT_LT(rel_l2(out_host[active][dev], lanes[active].fw.output), kForwardRelTol);
                EXPECT_LT(rel_l2(lse_column(inter_host[active][dev]), lanes[active].fw.lse), kForwardRelTol);
                // The other lane's reference must be far off, so a swapped binding cannot pass.
                EXPECT_GT(rel_l2(out_host[active][dev], lanes[idle].fw.output), 10 * kForwardRelTol);

                EXPECT_TRUE(all_equal(out_host[idle][dev], kSentinel)) << "wrote into the idle lane's output";
                EXPECT_TRUE(all_equal(lse_column(inter_host[idle][dev]), kNegInf))
                    << "wrote into the idle lane's intermediates";
            }
        });
}

TEST_F(RingSDPALaneTest, ZigZagBackwardBindsAssignedLanePerDevice) {
    const auto ring = ring_setup();

    for_each_direction_step_and_pass(
        ring, [&](RingShiftDirection direction, uint32_t step, uint32_t seed, std::vector<ttnn::Tensor>& keep_alive) {
            const auto lanes = make_lanes(seed);
            // Sanity check of the test's sensitivity: gradients computed with the other lane's O
            // (i.e. a wrong u = rowsum(dO * O)) must be far outside the tolerance.
            for (uint32_t lane = 0; lane < 2; ++lane) {
                const auto& l = lanes[lane];
                const auto wrong_u = reference_backward(l.q, l.k, l.v, lanes[1U - lane].o, l.fw.probabilities, l.d_o);
                ASSERT_GT(rel_l2(wrong_u.dq, l.bw.dq), 4 * kBackwardRelTol);
                ASSERT_GT(rel_l2(wrong_u.dk, l.bw.dk), 4 * kBackwardRelTol);
            }

            std::array<ttnn::Tensor, 2> q, k, v, o, d_o, inter, dq, dk, dv;
            for (uint32_t lane = 0; lane < 2; ++lane) {
                q[lane] = to_device_bf16(lanes[lane].q);
                k[lane] = to_device_bf16(lanes[lane].k);
                v[lane] = to_device_bf16(lanes[lane].v);
                o[lane] = to_device_bf16(lanes[lane].o);
                d_o[lane] = to_device_bf16(lanes[lane].d_o);
                inter[lane] = to_device_fp32(lse_to_intermediates(lanes[lane].fw.lse));
                dq[lane] = filled(q[lane].logical_shape(), kSentinel, ttnn::DataType::BFLOAT16);
                dk[lane] = filled(k[lane].logical_shape(), kSentinel, ttnn::DataType::BFLOAT16);
                dv[lane] = filled(v[lane].logical_shape(), kSentinel, ttnn::DataType::BFLOAT16);
                keep_alive.insert(
                    keep_alive.end(),
                    {q[lane], k[lane], v[lane], o[lane], d_o[lane], inter[lane], dq[lane], dk[lane], dv[lane]});
            }

            ttml::metal::ring_sdpa_bw(
                d_o[0],
                o[0],
                q[0],
                k[0],
                v[0],
                inter[0],
                ring.ring_size,
                ring.ring_axis,
                step,
                AttentionMaskType::Causal,
                direction,
                dq[0],
                dk[0],
                dv[0],
                RingLaneSelection::ZigZag,
                ttml::metal::RingSDPABwLane{
                    .grad_output = d_o[1],
                    .attn_output = o[1],
                    .query = q[1],
                    .key = k[1],
                    .value = v[1],
                    .intermediates = inter[1],
                    .grad_query = dq[1],
                    .grad_key = dk[1],
                    .grad_value = dv[1]});

            const std::array<std::vector<xt::xarray<float>>, 2> dq_host{per_device(dq[0]), per_device(dq[1])};
            const std::array<std::vector<xt::xarray<float>>, 2> dk_host{per_device(dk[0]), per_device(dk[1])};
            const std::array<std::vector<xt::xarray<float>>, 2> dv_host{per_device(dv[0]), per_device(dv[1])};

            for (uint32_t dev = 0; dev < ring.num_devices; ++dev) {
                const uint32_t ring_id = ring_id_of(dev, ring.mesh_shape, ring.ring_axis);
                const uint32_t active = active_lane(ring_id, step, ring, direction);
                const uint32_t idle = 1U - active;
                SCOPED_TRACE(fmt::format("device {} (ring id {}) active lane {}", dev, ring_id, active));
                const auto& ref = lanes[active].bw;

                EXPECT_LT(rel_l2(dq_host[active][dev], ref.dq), kBackwardRelTol);
                EXPECT_LT(rel_l2(dk_host[active][dev], ref.dk), kBackwardRelTol);
                EXPECT_LT(rel_l2(dv_host[active][dev], ref.dv), kBackwardRelTol);

                EXPECT_TRUE(all_equal(dq_host[idle][dev], kSentinel)) << "wrote into the idle lane's dQ";
                EXPECT_TRUE(all_equal(dk_host[idle][dev], kSentinel)) << "wrote into the idle lane's dK";
                EXPECT_TRUE(all_equal(dv_host[idle][dev], kSentinel)) << "wrote into the idle lane's dV";
            }
        });
}

// Lane selection is validated before any program is built, with a specific reason.
TEST_F(RingSDPALaneTest, ZigZagValidationRejectsInconsistentLaneArguments) {
    const auto ring = ring_setup();
    const auto lanes = make_lanes(4242U);
    const auto q = to_device_bf16(lanes[0].q);
    const auto k = to_device_bf16(lanes[0].k);
    const auto v = to_device_bf16(lanes[0].v);
    const auto o = to_device_bf16(lanes[0].o);
    const auto d_o = to_device_bf16(lanes[0].d_o);
    const auto inter = to_device_fp32(lse_to_intermediates(lanes[0].fw.lse));
    const ttnn::Shape out_shape{kB, kH, kC, kD};
    const ttnn::Shape inter_shape{kB, kH, kC, 32};
    const ttml::metal::ops::ring_sdpa_fw::RingSDPAFwLane fw_lane{
        .query = q,
        .key = k,
        .value = v,
        .output = filled(out_shape, 0.0F, ttnn::DataType::BFLOAT16),
        .intermediates = filled(inter_shape, 0.0F, ttnn::DataType::FLOAT32)};
    const ttml::metal::RingSDPABwLane bw_lane{
        .grad_output = d_o,
        .attn_output = o,
        .query = q,
        .key = k,
        .value = v,
        .intermediates = inter,
        .grad_query = filled(q.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16),
        .grad_key = filled(k.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16),
        .grad_value = filled(v.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16)};

    // Second lane with a different sequence length.
    const auto short_lanes = [] {
        auto q_short = uniform({kB, kH, kC / 2, kD}, kQScale, 7U);
        auto kv_short = uniform({kB, kG, kC / 2, kD}, 1.0F, 8U);
        return std::pair{to_device_bf16(q_short), to_device_bf16(kv_short)};
    }();
    const ttml::metal::ops::ring_sdpa_fw::RingSDPAFwLane fw_short_lane{
        .query = short_lanes.first,
        .key = short_lanes.second,
        .value = short_lanes.second,
        .output = filled(short_lanes.first.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16),
        .intermediates = filled(ttnn::Shape{kB, kH, kC / 2, 32}, 0.0F, ttnn::DataType::FLOAT32)};

    using FwLaneOpt = std::optional<ttml::metal::ops::ring_sdpa_fw::RingSDPAFwLane>;
    auto fw = [&](uint32_t step, RingLaneSelection selection, const FwLaneOpt& lane1) {
        return [&, step, selection, lane1] {
            ttml::metal::ring_sdpa_fw(
                q,
                k,
                v,
                ring.ring_size,
                ring.ring_axis,
                step,
                AttentionMaskType::Causal,
                RingShiftDirection::Backward,
                std::nullopt,
                std::nullopt,
                selection,
                lane1);
        };
    };
    auto bw = [&](uint32_t step, RingLaneSelection selection, const std::optional<ttml::metal::RingSDPABwLane>& lane1) {
        return [&, step, selection, lane1] {
            ttml::metal::ring_sdpa_bw(
                d_o,
                o,
                q,
                k,
                v,
                inter,
                ring.ring_size,
                ring.ring_axis,
                step,
                AttentionMaskType::Causal,
                RingShiftDirection::Backward,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                selection,
                lane1);
        };
    };

    EXPECT_THAT(
        fw(1U, RingLaneSelection::ZigZag, FwLaneOpt{}),
        ThrowsMessage<std::runtime_error>(HasSubstr("requires the second-lane tensors")));
    EXPECT_THAT(
        fw(0U, RingLaneSelection::ZigZag, FwLaneOpt{fw_lane}),
        ThrowsMessage<std::runtime_error>(HasSubstr("not defined for step 0")));
    EXPECT_THAT(
        fw(1U, RingLaneSelection::None, FwLaneOpt{fw_lane}),
        ThrowsMessage<std::runtime_error>(HasSubstr("lane selection is None")));
    EXPECT_THAT(
        fw(1U, RingLaneSelection::ZigZag, FwLaneOpt{fw_short_lane}),
        ThrowsMessage<std::runtime_error>(HasSubstr("Second-lane tensor 'query'")));

    EXPECT_THAT(
        bw(1U, RingLaneSelection::ZigZag, std::nullopt),
        ThrowsMessage<std::runtime_error>(HasSubstr("requires the second-lane tensors")));
    EXPECT_THAT(
        bw(0U, RingLaneSelection::ZigZag, bw_lane),
        ThrowsMessage<std::runtime_error>(HasSubstr("not defined for step 0")));
    EXPECT_THAT(
        bw(1U, RingLaneSelection::None, bw_lane),
        ThrowsMessage<std::runtime_error>(HasSubstr("lane selection is None")));
}
