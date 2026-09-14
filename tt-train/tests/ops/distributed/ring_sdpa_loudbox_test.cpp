// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Ring attention on an 8-board Blackhole host ("loudbox"), as opposed to the
 * 32-chip Galaxy that `ring_sdpa_test.cpp` requires.
 *
 * That file is left alone; this one is an independent fixture over the same
 * `ops::distributed::ring_attention_sdpa`, so the two can coexist and each
 * skips on hardware it does not fit.
 *
 * What the hardware is. Eight p150b boards, cabled as a 2 x 4 mesh:
 *
 *     chip 0 -- chip 3 -- chip 4 -- chip 7
 *       |         |         |         |
 *     chip 1 -- chip 2 -- chip 5 -- chip 6
 *
 * (derived from the UMD cluster descriptor's ethernet connections; the corner
 * chips have two links and the interior ones three). That is exactly the
 * topology `tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto`
 * declares, so the fixture points `TT_MESH_GRAPH_DESC_PATH` at it. Without
 * that, `enable_fabric(8)` defaults to the T3000 descriptor, which is a
 * Wormhole part.
 *
 * Why the ring is 4 and not 8. The context-parallel ring runs along one mesh
 * axis, and the mesh is 2 x 4, so the longest axis gives `cp_size = 4` with
 * `seq_len / 4` rows per device. The physical row is a line, so the ring's
 * wraparound hop (device 3 back to device 0) is routed over three fabric
 * hops rather than one; that is a cost, not a correctness problem, and it is
 * paid identically by anything else measured on this fixture. A ring of 8
 * would need a 1 x 8 mesh laid along the Hamiltonian cycle
 * 0-3-4-7-6-5-2-1-0, which the stock descriptor does not describe.
 *
 * The remaining axis (extent 2) is given to DDP, which is what makes the
 * parallelism context hand CP the axis of extent 4: for a 2-D mesh it assigns
 * axes in the order DDP, CP, TP. Tensors are therefore replicated across the
 * two rows and sharded along the four columns, and both rows compute the same
 * thing. That is deliberate: it keeps every board busy and it means a
 * disagreement between the rows would show up as a mismatch against the
 * reference.
 *
 * Correctness is against a dense host reference in float32, the same
 * convention as the Galaxy file: scale 1/sqrt(head_dim), causal mask, softmax
 * in the last dimension. The reference is written out again here rather than
 * shared, so that this file depends on nothing that file might change.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <core/xtensor_utils.hpp>
#include <cstdlib>
#include <optional>
#include <string>
#include <tt-metalium/distributed_context.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/distributed/ring_attention_sdpa.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/create_socket.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace {

using ttml::ttnn_fixed::distributed::RingShiftDirection;

constexpr uint32_t kExpectedChips = 8U;
constexpr uint32_t kMeshRows = 2U;
constexpr uint32_t kMeshCols = 4U;

//: Eight Blackhole chips, i.e. the machine this fixture is written for.
bool is_eight_chip_blackhole() {
    try {
        auto cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
        const auto all_chips = cluster_desc->get_all_chips();
        if (all_chips.size() != kExpectedChips) {
            return false;
        }
        for (const auto chip : all_chips) {
            if (cluster_desc->get_arch(chip) != tt::ARCH::BLACKHOLE) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

// The 2 x 4 Blackhole descriptor, found under whichever root this build uses.
// Returns nullopt if it cannot be located, in which case the fixture skips
// rather than letting enable_fabric() fall back to the Wormhole T3000 file.
std::optional<std::string> p150_x8_descriptor_path() {
    const char* const roots[] = {std::getenv("TT_METAL_RUNTIME_ROOT"), std::getenv("TT_METAL_HOME")};
    for (const char* root : roots) {
        if (root == nullptr) {
            continue;
        }
        std::string path =
            std::string(root) + "/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto";
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return std::nullopt;
}

bool loudbox_available() {
    return is_eight_chip_blackhole() && p150_x8_descriptor_path().has_value();
}

// ---------------------------------------------------------------- reference

struct RefForward {
    xt::xarray<float> output;
    xt::xarray<float> weights;  // softmax(P), kept for the backward
    float scale;
};

//: Dense causal attention forward in float32. Inputs are (B, H, S, D).
RefForward reference_forward(
    const xt::xarray<float>& query, const xt::xarray<float>& key, const xt::xarray<float>& value) {
    const auto shape = query.shape();
    const size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];
    const float scale = 1.0F / std::sqrt(static_cast<float>(D));

    xt::xarray<float> weights = xt::zeros<float>({B, H, S, S});
    xt::xarray<float> output = xt::zeros<float>({B, H, S, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S; ++i) {
                // Causal: only keys 0..i contribute, so the row's softmax runs
                // over that prefix and the rest stay zero.
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j <= i; ++j) {
                    float dot = 0.0F;
                    for (size_t d = 0; d < D; ++d) {
                        dot += query(b, h, i, d) * key(b, h, j, d);
                    }
                    weights(b, h, i, j) = dot * scale;
                    max_val = std::max(max_val, weights(b, h, i, j));
                }
                float sum_exp = 0.0F;
                for (size_t j = 0; j <= i; ++j) {
                    weights(b, h, i, j) = std::exp(weights(b, h, i, j) - max_val);
                    sum_exp += weights(b, h, i, j);
                }
                for (size_t j = 0; j <= i; ++j) {
                    weights(b, h, i, j) /= sum_exp;
                }
                for (size_t d = 0; d < D; ++d) {
                    float acc = 0.0F;
                    for (size_t j = 0; j <= i; ++j) {
                        acc += weights(b, h, i, j) * value(b, h, j, d);
                    }
                    output(b, h, i, d) = acc;
                }
            }
        }
    }
    return {output, weights, scale};
}

struct RefGrads {
    xt::xarray<float> dQ;
    xt::xarray<float> dK;
    xt::xarray<float> dV;
};

//: Dense causal attention backward in float32, from the saved weights.
RefGrads reference_backward(
    const xt::xarray<float>& query,
    const xt::xarray<float>& key,
    const xt::xarray<float>& value,
    const xt::xarray<float>& weights,
    const xt::xarray<float>& grad_output,
    const float scale) {
    const auto shape = query.shape();
    const size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];

    xt::xarray<float> dQ = xt::zeros<float>({B, H, S, D});
    xt::xarray<float> dK = xt::zeros<float>({B, H, S, D});
    xt::xarray<float> dV = xt::zeros<float>({B, H, S, D});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S; ++i) {
                // dV_j += P_ij dO_i, and dP_ij = dO_i . V_j
                std::vector<float> dP(i + 1, 0.0F);
                for (size_t j = 0; j <= i; ++j) {
                    float acc = 0.0F;
                    for (size_t d = 0; d < D; ++d) {
                        dV(b, h, j, d) += weights(b, h, i, j) * grad_output(b, h, i, d);
                        acc += grad_output(b, h, i, d) * value(b, h, j, d);
                    }
                    dP[j] = acc;
                }
                // dS = P o (dP - rowsum(P o dP)), the softmax Jacobian.
                float row = 0.0F;
                for (size_t j = 0; j <= i; ++j) {
                    row += weights(b, h, i, j) * dP[j];
                }
                for (size_t j = 0; j <= i; ++j) {
                    const float dS = weights(b, h, i, j) * (dP[j] - row) * scale;
                    for (size_t d = 0; d < D; ++d) {
                        dQ(b, h, i, d) += dS * key(b, h, j, d);
                        dK(b, h, j, d) += dS * query(b, h, i, d);
                    }
                }
            }
        }
    }
    return {dQ, dK, dV};
}

}  // namespace

class LoudboxRingSDPATest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        if (!loudbox_available()) {
            return;
        }
        // Before enable_fabric(), which otherwise picks the T3000 descriptor
        // for a device count of 8. Do not overwrite a path the caller set.
        setenv("TT_MESH_GRAPH_DESC_PATH", p150_x8_descriptor_path()->c_str(), /* overwrite */ 0);

        // Any suite that touched the device before this one reached it through
        // AutoContext::get_device(), which opens a default 1x1 mesh lazily and
        // never closes it. SetFabricConfig then refuses -- "not allowed while
        // devices are still open" -- and every test here skips with the suite
        // marked failed. Releasing it first is safe whether or not one is open,
        // and close_device() also drops the fabric config on the way out.
        ttml::autograd::ctx().close_device();

        // Both the accessor and the initializer throw -- one when it is not
        // yet initialized, the other when it already is -- so there is no
        // query to branch on; an earlier suite in the same binary may have
        // done it.
        try {
            ttml::autograd::ctx().initialize_distributed_context(0, nullptr);
        } catch (const std::exception&) {
            // Already initialized. Nothing to do.
        }
        ttml::ttnn_fixed::distributed::enable_fabric(kExpectedChips);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(kMeshRows, kMeshCols));
        ttml::autograd::ctx().set_seed(42);
        ttml::autograd::ctx().initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);

        // DDP on axis 0 (extent 2), CP on axis 1 (extent 4): the context
        // assigns axes in the order DDP, CP, TP, and a 2-D mesh needs exactly
        // two parallelisms enabled. This is what gives the ring four devices.
        // It is built from the open device, so it comes after open_device();
        // there is no API to replace one, hence the guard.
        if (!ttml::autograd::ctx().is_parallelism_context_initialized()) {
            ttml::autograd::ctx().initialize_parallelism_context(
                {.enable_ddp = true, .enable_tp = false, .enable_cp = true});
        }
    }

    static void TearDownTestSuite() {
        if (loudbox_available()) {
            // Leaves no device open, so a later suite's lazy 1x1 open works.
            ttml::autograd::ctx().close_device();
        }
    }

    void SetUp() override {
        if (!loudbox_available()) {
            GTEST_SKIP() << "Needs eight Blackhole boards and the p150_x8 mesh graph descriptor";
        }
    }
};

// ------------------------------------------------------------------ fabric

// The cheapest thing that exercises fabric and the socket manager: no kernels,
// no attention, just data moving around the ring. If the boards are not
// actually reachable over fabric, this fails in seconds instead of hanging
// inside a ring-attention step.
TEST_F(LoudboxRingSDPATest, RingShiftAroundTheCpAxis) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    ASSERT_TRUE(pctx.is_cp_enabled());
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    ASSERT_EQ(cp_axis, 1U);
    ASSERT_EQ(cp_size, kMeshCols);

    auto& rng = autograd::ctx().get_generator();
    const xt::xarray<float> full = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{1UL, 1UL, 128UL, 64UL}, 0.0F, 2.0F, rng());

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto tt_tensor =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(full, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = autograd::create_tensor(tt_tensor);

    const auto before = core::to_xtensor<float>(tensor->get_value(), core::IdentityComposer{});
    const auto shifted = ops::distributed::ring_shift(tensor, cp_axis, RingShiftDirection::Backward);
    const auto after = core::to_xtensor<float>(shifted->get_value(), core::IdentityComposer{});

    // Backward shift: device i receives what device (i + 1) mod cp_size held.
    for (uint32_t row = 0; row < kMeshRows; ++row) {
        for (uint32_t col = 0; col < kMeshCols; ++col) {
            const size_t dst = row * kMeshCols + col;
            const size_t src = row * kMeshCols + ((col + 1U) % kMeshCols);
            EXPECT_TRUE(xt::allclose(before[src], after[dst], 1e-3F, 1e-5F))
                << "device " << dst << " should hold what device " << src << " had";
        }
    }
}

// ---------------------------------------------------------- ring attention

namespace {

//: Gather a CP-sharded (B, H, S, D) tensor back into one full array.
xt::xarray<float> gather_cp(
    const std::vector<xt::xarray<float>>& per_device,
    const size_t batch,
    const size_t num_heads,
    const size_t seq_len,
    const size_t head_dim,
    const size_t seq_per_device) {
    xt::xarray<float> out = xt::zeros<float>({batch, num_heads, seq_len, head_dim});
    for (size_t dev = 0; dev < per_device.size(); ++dev) {
        // CP is axis 1, so the column index within the row is the CP index;
        // the two rows are replicas and write the same values.
        const size_t cp_idx = dev % kMeshCols;
        const size_t seq_start = cp_idx * seq_per_device;
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t s = 0; s < seq_per_device; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        out(b, h, seq_start + s, d) = per_device[dev](b, h, s, d);
                    }
                }
            }
        }
    }
    return out;
}

//: Attention with each device restricted to its own shard, i.e. what a ring
//: that never moved any data would produce. Used as a negative control: the
//: real result has to be far from this, or the test is not testing the ring.
xt::xarray<float> block_diagonal_reference(
    const xt::xarray<float>& query,
    const xt::xarray<float>& key,
    const xt::xarray<float>& value,
    const size_t seq_per_device) {
    const auto shape = query.shape();
    const size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];
    xt::xarray<float> out = xt::zeros<float>({B, H, S, D});
    for (size_t start = 0; start < S; start += seq_per_device) {
        const xt::xarray<float> q =
            xt::view(query, xt::all(), xt::all(), xt::range(start, start + seq_per_device), xt::all());
        const xt::xarray<float> k =
            xt::view(key, xt::all(), xt::all(), xt::range(start, start + seq_per_device), xt::all());
        const xt::xarray<float> v =
            xt::view(value, xt::all(), xt::all(), xt::range(start, start + seq_per_device), xt::all());
        const auto chunk = reference_forward(q, k, v);
        xt::view(out, xt::all(), xt::all(), xt::range(start, start + seq_per_device), xt::all()) = chunk.output;
    }
    return out;
}

void run_ring_attention(
    const size_t batch,
    const size_t num_heads,
    const size_t seq_len,
    const size_t head_dim,
    const bool test_backward) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    // Guard against the silent degeneracy: with CP off, ring_attention_sdpa
    // falls back to plain single-device attention and every assertion below
    // would still pass while testing nothing distributed.
    ASSERT_TRUE(pctx.is_cp_enabled());
    ASSERT_GT(cp_size, 1U) << "a ring of one device would make this test vacuous";
    ASSERT_EQ(seq_len % cp_size, 0U) << "sequence must divide across the ring";
    const size_t seq_per_device = seq_len / cp_size;
    ASSERT_EQ(seq_per_device % 32U, 0U) << "each device's shard must be a whole number of tiles";

    auto& rng = autograd::ctx().get_generator();
    const std::array<std::size_t, 4> qkv_shape{batch, num_heads, seq_len, head_dim};
    const xt::xarray<float> query_xt = ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());
    const xt::xarray<float> key_xt = ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());
    const xt::xarray<float> value_xt = ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());

    const auto ref = reference_forward(query_xt, key_xt, value_xt);

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    auto query_tensor = autograd::create_tensor(
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(query_xt, device, ttnn::Layout::TILE, mapper.get()),
        /* requires_grad */ true);
    auto key_tensor = autograd::create_tensor(
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(key_xt, device, ttnn::Layout::TILE, mapper.get()),
        /* requires_grad */ true);
    auto value_tensor = autograd::create_tensor(
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(value_xt, device, ttnn::Layout::TILE, mapper.get()),
        /* requires_grad */ true);

    // Causal masking is generated on device from the mask type; the op rejects
    // an explicit mask tensor in CP mode.
    auto output_tensor = ops::distributed::ring_attention_sdpa(
        query_tensor, key_tensor, value_tensor, /*mask=*/std::nullopt, ttml::metal::AttentionMaskType::Causal);

    const auto per_device_output = core::to_xtensor<float>(output_tensor->get_value(), core::IdentityComposer{});
    // The two mesh rows are replicas of one another. Checking them against
    // each other catches a divergence that gathering would otherwise hide,
    // since the second row's values overwrite the first's.
    for (uint32_t col = 0; col < kMeshCols; ++col) {
        EXPECT_TRUE(xt::allclose(per_device_output[col], per_device_output[kMeshCols + col], 1e-5F, 1e-6F))
            << "the two replica rows disagree at CP index " << col;
    }

    const auto gathered_output =
        gather_cp(per_device_output, batch, num_heads, seq_len, head_dim, seq_per_device);
    const float fw_rtol = 1e-2F;
    const float fw_atol = 5e-1F;
    EXPECT_TRUE(xt::allclose(ref.output, gathered_output, fw_rtol, fw_atol))
        << "ring attention output does not match the dense reference";

    // Negative control. A ring that never moved a chunk would compute
    // attention block-diagonally, which is a plausible-looking wrong answer.
    // If the tolerance above cannot separate the two, it is too loose to be
    // evidence of anything.
    const auto block_diagonal = block_diagonal_reference(query_xt, key_xt, value_xt, seq_per_device);
    EXPECT_FALSE(xt::allclose(ref.output, block_diagonal, fw_rtol, fw_atol))
        << "the forward tolerance cannot tell full attention from block-diagonal attention, "
           "so passing it says nothing about the ring";

    if (!test_backward) {
        return;
    }

    const xt::xarray<float> grad_output_xt =
        ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());
    const auto ref_grads =
        reference_backward(query_xt, key_xt, value_xt, ref.weights, grad_output_xt, ref.scale);

    output_tensor->set_grad(
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_output_xt, device, ttnn::Layout::TILE, mapper.get()));
    output_tensor->backward();

    const auto gathered_dQ = gather_cp(
        core::to_xtensor<float>(query_tensor->get_grad(), core::IdentityComposer{}),
        batch, num_heads, seq_len, head_dim, seq_per_device);
    const auto gathered_dK = gather_cp(
        core::to_xtensor<float>(key_tensor->get_grad(), core::IdentityComposer{}),
        batch, num_heads, seq_len, head_dim, seq_per_device);
    const auto gathered_dV = gather_cp(
        core::to_xtensor<float>(value_tensor->get_grad(), core::IdentityComposer{}),
        batch, num_heads, seq_len, head_dim, seq_per_device);

    // The same grading the Galaxy suite uses, and for the same reasons:
    // uniform(0, 2) inputs make rowsum(dO o O) large against the (dP - u)
    // cancellation, so bf16 rounding of the saved output and of each ring
    // step lands mostly in dK and dQ. dK is graded against its own largest
    // true value because its accumulation length is the query sequence, so
    // its magnitude and matmul noise grow with S while dQ and dV stay O(1).
    const float rtol = 3e-2F;
    const float atol = 5e-2F;
    const float dk_atol = std::max(atol, 2e-2F * xt::amax(xt::abs(ref_grads.dK))());
    const auto report = [](const xt::xarray<float>& expected, const xt::xarray<float>& got) {
        return "max_abs_diff=" + std::to_string(xt::amax(xt::abs(got - expected))()) +
               " ref_amax=" + std::to_string(xt::amax(xt::abs(expected))());
    };
    EXPECT_TRUE(xt::allclose(ref_grads.dQ, gathered_dQ, rtol, atol)) << "dQ: " << report(ref_grads.dQ, gathered_dQ);
    EXPECT_TRUE(xt::allclose(ref_grads.dK, gathered_dK, rtol, dk_atol)) << "dK: " << report(ref_grads.dK, gathered_dK);
    EXPECT_TRUE(xt::allclose(ref_grads.dV, gathered_dV, rtol, atol)) << "dV: " << report(ref_grads.dV, gathered_dV);
}

}  // namespace

TEST_F(LoudboxRingSDPATest, CausalForward) {
    run_ring_attention(/*batch=*/1, /*num_heads=*/4, /*seq_len=*/128, /*head_dim=*/64, /*test_backward=*/false);
}

TEST_F(LoudboxRingSDPATest, CausalBackward) {
    run_ring_attention(/*batch=*/1, /*num_heads=*/4, /*seq_len=*/128, /*head_dim=*/64, /*test_backward=*/true);
}

TEST_F(LoudboxRingSDPATest, LargerSequenceCausalBackward) {
    run_ring_attention(/*batch=*/1, /*num_heads=*/4, /*seq_len=*/512, /*head_dim=*/64, /*test_backward=*/true);
}

TEST_F(LoudboxRingSDPATest, LargerBatchCausalBackward) {
    run_ring_attention(/*batch=*/2, /*num_heads=*/8, /*seq_len=*/256, /*head_dim=*/64, /*test_backward=*/true);
}
