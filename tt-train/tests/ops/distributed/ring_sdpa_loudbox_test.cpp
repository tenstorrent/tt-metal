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
 * Two ways to use it, chosen by TTML_LOUDBOX_RING8, because the ring runs
 * along one mesh axis and the axis lengths differ.
 *
 * Default, `cp_size = 4`. The stock 2 x 4 descriptor, CP on the axis of
 * extent 4 and DDP on the axis of extent 2 -- the parallelism context assigns
 * axes in the order DDP, CP, TP, and a 2-D mesh needs exactly two enabled.
 * Tensors are replicated across the two rows, so both rows compute the same
 * thing and a disagreement between them is checked for directly. The physical
 * row is a line, so the ring's wraparound (device 3 back to device 0) is
 * routed over three fabric hops.
 *
 * `TTML_LOUDBOX_RING8=1`, `cp_size = 8`. The cabling contains the Hamiltonian
 * cycle 0-3-4-7-6-5-2-1-0, so every edge a ring of eight needs is a physical
 * link; p150_x8_ring_mesh_graph_descriptor.textproto declares the mesh as
 * 1 x 8 with a RING dim type, and fabric comes up as FABRIC_2D_TORUS_X, so
 * the wraparound here is a real link rather than three hops. The cost is that
 * a mesh with an axis of extent 1 is a line topology, where exactly one
 * parallelism may be enabled: no DDP, hence no replica to cross-check, and
 * one ring rather than two.
 *
 * Which to measure on. The 8-ring is the more meaningful shape -- a longer
 * ring, a real wraparound, one ring using the whole machine. The 4-ring runs
 * two independent rings side by side that share fabric and host dispatch,
 * which is fair between two implementations measured the same way but is not
 * a clean absolute number. Both are legal for any comparison provided both
 * sides use the same one.
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
#include <functional>
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

// Two ways to lay the eight boards out, selected by TTML_LOUDBOX_RING8.
//
// Default, the 2 x 4 mesh the stock descriptor declares: CP gets the axis of
// extent 4 and DDP the axis of extent 2, so the ring is four chips and the
// two rows are replicas.
//
// With TTML_LOUDBOX_RING8=1, a 1 x 8 ring. The cabling contains the
// Hamiltonian cycle 0-3-4-7-6-5-2-1-0, so every edge a ring of eight needs is
// a physical link, and p150_x8_ring_mesh_graph_descriptor.textproto declares
// it that way. The ring is then eight chips, which is the more interesting
// shape for context parallelism -- at the cost that a mesh with an axis of
// extent 1 is a line topology, where the parallelism context allows exactly
// one parallelism, so there is no DDP replica to check against.
struct Topology {
    uint32_t rows;
    uint32_t cols;
    const char* descriptor;
    ttml::autograd::DistributedConfig parallelism;
    bool has_replicas;  // more than one row, so each CP index appears twice
};

bool want_ring_of_eight() {
    const char* v = std::getenv("TTML_LOUDBOX_RING8");
    return v != nullptr && v[0] == '1';
}

Topology topology() {
    if (want_ring_of_eight()) {
        return {
            1U,
            8U,
            "p150_x8_ring_mesh_graph_descriptor.textproto",
            {.enable_ddp = false, .enable_tp = false, .enable_cp = true},
            false};
    }
    return {
        2U,
        4U,
        "p150_x8_mesh_graph_descriptor.textproto",
        {.enable_ddp = true, .enable_tp = false, .enable_cp = true},
        true};
}

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

// This topology's Blackhole descriptor, under whichever root this build uses.
// Returns nullopt if it cannot be located, in which case the fixture skips
// rather than letting enable_fabric() fall back to the Wormhole T3000 file.
std::optional<std::string> descriptor_path() {
    const char* const roots[] = {std::getenv("TT_METAL_RUNTIME_ROOT"), std::getenv("TT_METAL_HOME")};
    for (const char* root : roots) {
        if (root == nullptr) {
            continue;
        }
        std::string path =
            std::string(root) + "/tt_metal/fabric/mesh_graph_descriptors/" + topology().descriptor;
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return std::nullopt;
}

bool loudbox_available() {
    return is_eight_chip_blackhole() && descriptor_path().has_value();
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
        setenv("TT_MESH_GRAPH_DESC_PATH", descriptor_path()->c_str(), /* overwrite */ 0);

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
        ttml::autograd::ctx().open_device(
            tt::tt_metal::distributed::MeshShape(topology().rows, topology().cols));
        ttml::autograd::ctx().set_seed(42);
        ttml::autograd::ctx().initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);

        // DDP on axis 0 (extent 2), CP on axis 1 (extent 4): the context
        // assigns axes in the order DDP, CP, TP, and a 2-D mesh needs exactly
        // two parallelisms enabled. This is what gives the ring four devices.
        // It is built from the open device, so it comes after open_device();
        // there is no API to replace one, hence the guard.
        if (!ttml::autograd::ctx().is_parallelism_context_initialized()) {
            ttml::autograd::ctx().initialize_parallelism_context(topology().parallelism);
        }
    }

    static void TearDownTestSuite() {
        if (loudbox_available()) {
            // Leaves no device open, so a later suite's lazy 1x1 open works.
            ttml::autograd::ctx().close_device();
            // And no context saying four devices are sharing a sequence. Ops
            // consult it: build_rope_params shards its frequencies when CP is
            // enabled and requires the sequence length to divide cp_size, so
            // leaving CP on fails RoPETest -- whose sequence length is 5 --
            // several suites later, with nothing pointing back to here.
            ttml::autograd::ctx().reset_parallelism_context();
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
    ASSERT_EQ(cp_size, topology().cols);

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
    const uint32_t cols = topology().cols;
    for (uint32_t row = 0; row < topology().rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const size_t dst = row * cols + col;
            const size_t src = row * cols + ((col + 1U) % cols);
            EXPECT_TRUE(xt::allclose(before[src], after[dst], 1e-3F, 1e-5F))
                << "device " << dst << " should hold what device " << src << " had";
        }
    }
}

// ---------------------------------------------------------- ring attention

namespace {

//: Fix the random draw for one case, so it does not depend on which tests ran
//: before it.
//
// The fixture seeds once and every test draws from the same generator, so a
// filtered run and a full run give a case different data -- and a tolerance
// that holds for one draw can fail on another. Seeding per case from its own
// shape makes a failure reproducible with a --gtest_filter, which is the only
// way to chase one.
void seed_for_case(const std::array<size_t, 5>& shape) {
    size_t h = 1469598103934665603ULL;
    for (const size_t v : shape) {
        h = (h ^ v) * 1099511628211ULL;
    }
    ttml::autograd::ctx().set_seed(static_cast<uint32_t>(h & 0x7FFFFFFFULL));
}

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
        const size_t cp_idx = dev % topology().cols;
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
    // gather_cp() reads the CP index out of the device index as `dev % cols`,
    // which is only the column when CP owns axis 1.
    ASSERT_EQ(cp_axis, 1U) << "the gather assumes CP is the column axis";
    ASSERT_EQ(seq_len % cp_size, 0U) << "sequence must divide across the ring";
    const size_t seq_per_device = seq_len / cp_size;
    ASSERT_EQ(seq_per_device % 32U, 0U) << "each device's shard must be a whole number of tiles";

    seed_for_case({batch, num_heads, seq_len, head_dim, test_backward ? 1U : 0U});
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
    if (topology().has_replicas) {
        for (uint32_t col = 0; col < topology().cols; ++col) {
            EXPECT_TRUE(
                xt::allclose(per_device_output[col], per_device_output[topology().cols + col], 1e-5F, 1e-6F))
                << "the two replica rows disagree at CP index " << col;
        }
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

// Shapes are given as rows *per device* rather than as a sequence length, so
// the same case is legal whether the ring is four chips or eight: the shard
// must be a whole number of 32-row tiles, which a fixed sequence length stops
// being as soon as the ring grows.
size_t seq_for(const size_t rows_per_device) {
    return rows_per_device * ttml::autograd::ctx().get_parallelism_context().get_cp_size();
}

TEST_F(LoudboxRingSDPATest, CausalForward) {
    run_ring_attention(1, 4, seq_for(32), 64, /*test_backward=*/false);
}

TEST_F(LoudboxRingSDPATest, CausalBackward) {
    run_ring_attention(1, 4, seq_for(32), 64, /*test_backward=*/true);
}

TEST_F(LoudboxRingSDPATest, LargerSequenceCausalBackward) {
    run_ring_attention(1, 4, seq_for(128), 64, /*test_backward=*/true);
}

TEST_F(LoudboxRingSDPATest, LargerBatchCausalBackward) {
    run_ring_attention(2, 8, seq_for(64), 64, /*test_backward=*/true);
}

// ------------------------------------------- the two backward implementations

namespace {

// Both per-step backwards, on identical inputs, with their gradients compared
// against each other and against the dense host reference.
//
// This is the comparison the cyclic schedule exists to make. The two paths
// share the forward, the ring loop, the shift, the skip pattern and the
// global statistics; they differ only in what an executing chip runs per
// step. Agreement to rounding is the claim, not bitwise equality: one path
// recomputes the score stage twice in bf16 and accumulates its steps through
// bf16 buffers, the other fuses into one kernel and stays in FP32, so they
// round differently by construction.
void compare_backward_implementations(
    const size_t batch,
    const size_t num_heads,
    const size_t seq_len,
    const size_t head_dim,
    const uint32_t rows_per_block_tiles) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    const size_t seq_per_device = seq_len / cp_size;
    ASSERT_EQ(seq_len % cp_size, 0U);
    // The cyclic schedule needs C = n / (2 * Bt * 32) whole cores per chunk.
    ASSERT_EQ(seq_per_device % (2U * rows_per_block_tiles * 32U), 0U)
        << "chunk of " << seq_per_device << " rows does not divide into cyclic blocks";

    seed_for_case({batch, num_heads, seq_len, head_dim, rows_per_block_tiles});
    auto& rng = autograd::ctx().get_generator();
    const std::array<std::size_t, 4> qkv_shape{batch, num_heads, seq_len, head_dim};
    const xt::xarray<float> query_xt = ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());
    const xt::xarray<float> key_xt = ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());
    const xt::xarray<float> value_xt = ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());
    const xt::xarray<float> grad_output_xt =
        ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());

    const auto ref = reference_forward(query_xt, key_xt, value_xt);
    const auto ref_grads = reference_backward(query_xt, key_xt, value_xt, ref.weights, grad_output_xt, ref.scale);

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto to_device = [&](const xt::xarray<float>& x) {
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(x, device, ttnn::Layout::TILE, mapper.get());
    };

    struct Result {
        xt::xarray<float> dQ, dK, dV;
    };
    const auto run = [&](ops::distributed::RingBackwardKind kind) {
        auto q = autograd::create_tensor(to_device(query_xt), /* requires_grad */ true);
        auto k = autograd::create_tensor(to_device(key_xt), /* requires_grad */ true);
        auto v = autograd::create_tensor(to_device(value_xt), /* requires_grad */ true);
        auto out = ops::distributed::ring_attention_sdpa(
            q, k, v, /*mask=*/std::nullopt, ttml::metal::AttentionMaskType::Causal, kind, rows_per_block_tiles);
        out->set_grad(to_device(grad_output_xt));
        out->backward();
        const auto gather = [&](const ttnn::Tensor& t) {
            return gather_cp(
                core::to_xtensor<float>(t, core::IdentityComposer{}),
                batch, num_heads, seq_len, head_dim, seq_per_device);
        };
        return Result{gather(q->get_grad()), gather(k->get_grad()), gather(v->get_grad())};
    };

    const auto two_pass = run(ops::distributed::RingBackwardKind::TwoPass);
    const auto cyclic = run(ops::distributed::RingBackwardKind::Cyclic);

    // The in-place variant runs the same kernels with the step's contribution
    // added into the running accumulator on device instead of on the host. It
    // is not bitwise equal to Cyclic, for two reasons that are both rounding:
    // the sum runs in a different order (incoming accumulator first, then the
    // step's terms, where Cyclic sums the terms from zero and adds once), and
    // a seeded column gradient passes through the Src registers on its way
    // into the accumulator, which carry about ten mantissa bits, once per
    // column per step. What it must not be is contribution-sized: a first
    // visit that overwrote instead of adding would drop a whole step's worth,
    // which is orders of magnitude above either effect. So it is graded by
    // RMS against the reference, allowed a small factor over Cyclic for the
    // truncations, and the max difference to Cyclic is printed.
    const auto in_place = run(ops::distributed::RingBackwardKind::CyclicInPlace);

    // Run the cyclic path again on the same inputs. Its relay has credits,
    // readiness tags and endpoint counters, and a race in any of them would
    // show as a result that moves between runs. Bitwise equality says any
    // difference from the two-pass path is arithmetic, not a protocol bug --
    // which is the distinction a tolerance can never make.
    const auto cyclic_again = run(ops::distributed::RingBackwardKind::Cyclic);
    EXPECT_TRUE(cyclic.dQ == cyclic_again.dQ) << "the cyclic backward is not deterministic in dQ";
    EXPECT_TRUE(cyclic.dK == cyclic_again.dK) << "the cyclic backward is not deterministic in dK";
    EXPECT_TRUE(cyclic.dV == cyclic_again.dV) << "the cyclic backward is not deterministic in dV";

    // How these are graded, and why not by a max-error tolerance.
    //
    // The inputs are uniform(0, 2), so with d = 64 the scores are large and
    // positive and the softmax is nearly one-hot. bf16 carries eight mantissa
    // bits, so a 0.4% error on a score of ~8 moves the exponent by ~0.03 and
    // the attention weight by several percent; where the backward's
    // (dP - D) cancellation then bites, individual gradient elements land
    // percent-level off. Both implementations do this, in different places,
    // because they round differently.
    //
    // A max over ~10^5 elements is therefore a statistic about the unluckiest
    // element, and it moves by a factor of two on nothing. It is kept here,
    // graded against each tensor's own largest value, only as a coarse net;
    // the claims worth making are the two below it.
    const auto amax = [](const xt::xarray<float>& x) { return xt::amax(xt::abs(x))(); };
    const auto rms_error = [](const xt::xarray<float>& expected, const xt::xarray<float>& got) {
        return std::sqrt(xt::mean(xt::square(got - expected))());
    };
    const char* names[] = {"dQ", "dK", "dV"};
    const xt::xarray<float>* refs[] = {&ref_grads.dQ, &ref_grads.dK, &ref_grads.dV};
    const xt::xarray<float>* tps[] = {&two_pass.dQ, &two_pass.dK, &two_pass.dV};
    const xt::xarray<float>* cys[] = {&cyclic.dQ, &cyclic.dK, &cyclic.dV};

    for (uint32_t k = 0; k < 3U; ++k) {
        const float scale = amax(*refs[k]);
        const float two_pass_max = amax(*refs[k] - *tps[k]);
        const float cyclic_max = amax(*refs[k] - *cys[k]);
        EXPECT_LE(two_pass_max, 0.25F * scale) << names[k] << ": two-pass is far from the reference";
        EXPECT_LE(cyclic_max, 0.25F * scale) << names[k] << ": cyclic is far from the reference";

        // The real accuracy claim. Root-mean-square says whether a whole
        // tensor is worse rather than one element of it, and it is graded
        // two ways: against the reference in absolute terms, and against the
        // other implementation. Measured, the cyclic path is consistently the
        // more accurate of the two -- it fuses into one kernel where the
        // two-pass path recomputes the score stage in bf16, and it keeps its
        // per-step gradients in FP32 where the other rounds them to bf16
        // before the host accumulates.
        const float two_pass_rms = rms_error(*refs[k], *tps[k]);
        const float cyclic_rms = rms_error(*refs[k], *cys[k]);
        EXPECT_LE(cyclic_rms, 0.05F * scale)
            << names[k] << ": cyclic rms " << cyclic_rms << " against reference scale " << scale;
        EXPECT_LE(cyclic_rms, 1.5F * two_pass_rms)
            << names[k] << ": the cyclic backward is less accurate than the two-pass one over the whole "
            << "tensor, rms " << cyclic_rms << " against " << two_pass_rms;

        const xt::xarray<float>* ips[] = {&in_place.dQ, &in_place.dK, &in_place.dV};
        const float in_place_rms = rms_error(*refs[k], *ips[k]);
        const float in_place_vs_cyclic = amax(*cys[k] - *ips[k]);
        EXPECT_LE(in_place_rms, 1.25F * cyclic_rms)
            << names[k] << ": in-place accumulation is less accurate than accumulating on the host, rms "
            << in_place_rms << " against " << cyclic_rms;
        EXPECT_LE(in_place_vs_cyclic, 0.05F * scale)
            << names[k] << ": in-place differs from Cyclic by " << in_place_vs_cyclic << " on scale " << scale
            << ", which is a dropped or doubled contribution, not rounding";
        std::cout << "    in-place " << names[k] << ": rms vs ref " << in_place_rms << " (cyclic " << cyclic_rms
                  << "), max |in-place - cyclic| " << in_place_vs_cyclic << " on scale " << scale << "\n";
    }

    // Printed so that a cross-check failure can be read: if each path's own
    // error against the reference is unchanged and only their difference
    // grows, they are rounding apart, not drifting from the answer.
    const auto err = [](const xt::xarray<float>& a, const xt::xarray<float>& b) {
        return xt::amax(xt::abs(a - b))();
    };
    // RMS as well as max: a max over half a million elements is one unlucky
    // element, and says nothing about whether a whole tensor is worse.
    const auto rms = [](const xt::xarray<float>& a, const xt::xarray<float>& b) {
        return std::sqrt(xt::mean(xt::square(a - b))());
    };
    std::cout << "  N=" << seq_len << " heads=" << num_heads << " Bt=" << rows_per_block_tiles << "\n"
              << "    vs reference  two-pass dQ " << err(ref_grads.dQ, two_pass.dQ) << " dK "
              << err(ref_grads.dK, two_pass.dK) << " dV " << err(ref_grads.dV, two_pass.dV) << "\n"
              << "    vs reference  cyclic   dQ " << err(ref_grads.dQ, cyclic.dQ) << " dK "
              << err(ref_grads.dK, cyclic.dK) << " dV " << err(ref_grads.dV, cyclic.dV) << "\n"
              << "    against each other     dQ " << err(two_pass.dQ, cyclic.dQ) << " dK "
              << err(two_pass.dK, cyclic.dK) << " dV " << err(two_pass.dV, cyclic.dV) << "\n"
              << "    rms vs reference  two-pass dQ " << rms(ref_grads.dQ, two_pass.dQ) << " dK "
              << rms(ref_grads.dK, two_pass.dK) << " dV " << rms(ref_grads.dV, two_pass.dV) << "\n"
              << "    rms vs reference  cyclic   dQ " << rms(ref_grads.dQ, cyclic.dQ) << " dK "
              << rms(ref_grads.dK, cyclic.dK) << " dV " << rms(ref_grads.dV, cyclic.dV) << "\n";

    // Where the worst element is. Outliers scattered through the tensor are
    // rounding; outliers clustered on a block boundary, a chunk, or one row
    // of a tile are a kernel bug wearing rounding's clothes.
    const auto worst_at = [&](const xt::xarray<float>& expected, const xt::xarray<float>& got) {
        size_t wb = 0, wh = 0, ws = 0, wd = 0;
        float worst = -1.0F;
        uint32_t over_half = 0;
        const float threshold = 0.5F * xt::amax(xt::abs(got - expected))();
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t sq = 0; sq < seq_len; ++sq) {
                    for (size_t dd = 0; dd < head_dim; ++dd) {
                        const float e = std::abs(got(b, h, sq, dd) - expected(b, h, sq, dd));
                        if (e > worst) {
                            worst = e;
                            wb = b; wh = h; ws = sq; wd = dd;
                        }
                        if (e >= threshold) {
                            ++over_half;
                        }
                    }
                }
            }
        }
        return "worst " + std::to_string(worst) + " at (b" + std::to_string(wb) + ",h" + std::to_string(wh) +
               ",s" + std::to_string(ws) + ",d" + std::to_string(wd) + ") chunk " +
               std::to_string(ws / seq_per_device) + " row-in-chunk " + std::to_string(ws % seq_per_device) +
               "; " + std::to_string(over_half) + " elements above half of it";
    };
    std::cout << "    two-pass dQ " << worst_at(ref_grads.dQ, two_pass.dQ) << "\n"
              << "    cyclic   dQ " << worst_at(ref_grads.dQ, cyclic.dQ) << "\n";
}

}  // namespace

TEST_F(LoudboxRingSDPATest, BothBackwardsAgree) {
    compare_backward_implementations(1, 4, seq_for(64), 64, /* Bt */ 1);
}

// Same problem as the tall-block case below, one tile per block instead of
// two: it separates the block height from the problem size, since the two
// differ only in Bt and hence in the core count the schedule derives.
TEST_F(LoudboxRingSDPATest, BothBackwardsAgreeAtTheSameSizeWithShortBlocks) {
    compare_backward_implementations(1, 4, seq_for(128), 64, /* Bt */ 1);
}

TEST_F(LoudboxRingSDPATest, BothBackwardsAgreeWithTallBlocks) {
    compare_backward_implementations(1, 4, seq_for(128), 64, /* Bt */ 2);
}

TEST_F(LoudboxRingSDPATest, BothBackwardsAgreeWithWiderHead) {
    compare_backward_implementations(1, 2, seq_for(64), 128, /* Bt */ 1);
}

// ------------------------------------------------------------------ timing

namespace {

//: Median wall clock of one full backward through the ring, host side.
//
// This is the baseline any replacement of the per-step op is measured
// against. It times the whole backward -- every ring step, the host-side
// accumulation and the shifts -- because that is what a caller pays; a
// per-step kernel number would flatter whichever implementation moves its
// cost into the driver.
double time_ring_backward(
    const size_t batch,
    const size_t num_heads,
    const size_t seq_len,
    const size_t head_dim,
    ttml::ops::distributed::RingBackwardKind kind = ttml::ops::distributed::RingBackwardKind::TwoPass,
    uint32_t rows_per_block_tiles = 1U) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const uint32_t cp_axis = autograd::ctx().get_parallelism_context().get_cp_axis().value();
    auto& rng = autograd::ctx().get_generator();

    const std::array<std::size_t, 4> qkv_shape{batch, num_heads, seq_len, head_dim};
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto to_device = [&](const xt::xarray<float>& x) {
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(x, device, ttnn::Layout::TILE, mapper.get());
    };

    const auto sample = [&]() {
        auto query = autograd::create_tensor(
            to_device(ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng())), true);
        auto key = autograd::create_tensor(
            to_device(ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng())), true);
        auto value = autograd::create_tensor(
            to_device(ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng())), true);
        auto out = ops::distributed::ring_attention_sdpa(
            query, key, value, std::nullopt, ttml::metal::AttentionMaskType::Causal, kind, rows_per_block_tiles);
        out->set_grad(to_device(ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng())));
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});

        const auto start = std::chrono::steady_clock::now();
        out->backward();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };

    sample();  // warm the program cache and the kernel build
    std::vector<double> samples;
    for (uint32_t k = 0; k < 5; ++k) {
        samples.push_back(sample());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

// The two implementations timed against each other, whole backward, median of
// five. The whole backward and not the kernel, because that is what a caller
// pays: a per-step kernel number would flatter whichever implementation moves
// its cost into the driver, and these two move different amounts there.
TEST_F(LoudboxRingSDPATest, DISABLED_CompareTheTwoBackwards) {
    const uint32_t cp_size = ttml::autograd::ctx().get_parallelism_context().get_cp_size();
    std::cout << "ring backward on " << cp_size << " chips, median of five\n";
    for (const auto& cfg : std::vector<std::array<size_t, 5>>{
             // batch, heads, rows per chip, head dim, Bt
             {1, 4, 128, 64, 1},
             {1, 4, 256, 64, 1},
             {1, 4, 256, 64, 2},
             {1, 8, 256, 64, 2},
             {1, 4, 512, 64, 2},
             {1, 4, 512, 64, 4},
             {1, 4, 256, 128, 2},
         }) {
        const size_t rows_per_chip = cfg[2];
        const auto Bt = static_cast<uint32_t>(cfg[4]);
        const size_t seq_len = rows_per_chip * cp_size;
        if (rows_per_chip % (2U * Bt * 32U) != 0U) {
            continue;  // the cyclic schedule needs whole cores per chunk
        }
        using Kind = ttml::ops::distributed::RingBackwardKind;
        const double two_pass = time_ring_backward(cfg[0], cfg[1], seq_len, cfg[3], Kind::TwoPass, Bt);
        const double cyclic = time_ring_backward(cfg[0], cfg[1], seq_len, cfg[3], Kind::Cyclic, Bt);
        const double in_place = time_ring_backward(cfg[0], cfg[1], seq_len, cfg[3], Kind::CyclicInPlace, Bt);
        std::cout << "  batch=" << cfg[0] << " heads=" << cfg[1] << " N=" << seq_len << " d=" << cfg[3]
                  << " Bt=" << Bt << " (" << rows_per_chip << " rows/chip, C=" << rows_per_chip / (2U * Bt * 32U)
                  << "): two-pass " << two_pass * 1e3 << " ms, cyclic " << cyclic * 1e3 << " ms ("
                  << two_pass / cyclic << "x), cyclic in-place " << in_place * 1e3 << " ms (" << two_pass / in_place
                  << "x)\n";
    }
}

// Where a ring step's time goes. The whole-backward numbers above barely move
// with the shape or with the kernel, which says the kernel is not what is
// being paid for. This times each component of one step in isolation: the
// per-step op of each implementation, one ring shift, one zeroing copy, one
// accumulate. Multiplied out over the steps and the tensors, they should
// account for the whole; whatever they do not is dispatch between them.
TEST_F(LoudboxRingSDPATest, DISABLED_BreakDownOneStep) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    auto& rng = autograd::ctx().get_generator();

    const auto median = [&](const std::function<void()>& f) {
        f();
        std::vector<double> samples;
        for (uint32_t k = 0; k < 7; ++k) {
            tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
            const auto start = std::chrono::steady_clock::now();
            f();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
            samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2] * 1e6;  // us
    };

    for (const auto& cfg : std::vector<std::array<size_t, 4>>{{1, 4, 128, 64}, {1, 4, 512, 64}}) {
        const size_t batch = cfg[0], heads = cfg[1], rows_per_chip = cfg[2], head_dim = cfg[3];
        const size_t seq_len = rows_per_chip * cp_size;
        const std::array<std::size_t, 4> shape{batch, heads, seq_len, head_dim};
        const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
        const auto bf16 = [&](const xt::xarray<float>& x) {
            return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(x, device, ttnn::Layout::TILE, mapper.get());
        };
        const auto q = bf16(ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, rng()));
        const auto k = bf16(ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, rng()));
        const auto v = bf16(ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, rng()));
        const auto dO = bf16(ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, rng()));
        const auto O = bf16(ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, rng()));
        const ttnn::Tensor lse = ttnn::full(
            ttnn::Shape{batch, heads, rows_per_chip, 32U}, 4.0F, ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
            std::ref(*device));
        const ttnn::Tensor D = ttnn::full(
            ttnn::Shape{batch, heads, rows_per_chip, 32U}, 0.5F, ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
            std::ref(*device));
        ttnn::Tensor acc = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
        ttnn::Tensor step_bf16 = ttnn::zeros_like(q);
        ttnn::Tensor step_fp32 = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
        const ttnn::Tensor zero_bf16 = ttnn::zeros_like(q);
        const ttnn::Tensor zero_fp32 = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);

        std::cout << "one ring step, " << cp_size << " chips, " << rows_per_chip << " rows/chip, heads=" << heads
                  << " d=" << head_dim << " (us, median of 7)\n";

        const double shift = median([&]() {
            (void)ttnn_fixed::distributed::ring_shift(k, cp_axis, RingShiftDirection::Forward);
        });
        std::cout << "  ring_shift of one K-sized bf16 tensor       " << shift << "\n";
        const double shift32 = median([&]() {
            (void)ttnn_fixed::distributed::ring_shift(acc, cp_axis, RingShiftDirection::Forward);
        });
        std::cout << "  ring_shift of one accumulator (fp32)        " << shift32 << "\n";
        const double zero = median([&]() { ttnn::copy(zero_bf16, step_bf16); });
        std::cout << "  zero one step buffer (copy)                 " << zero << "\n";
        const double add = median([&]() {
            acc = ttnn::add(acc, ttnn::typecast(step_bf16, ttnn::DataType::FLOAT32));
        });
        std::cout << "  typecast + add into accumulator             " << add << "\n";
        const double add32 = median([&]() { acc = ttnn::add(acc, step_fp32); });
        std::cout << "  add fp32 into accumulator                   " << add32 << "\n";
        // Step 0 is the diagonal (causal) step on every chip; no chip skips it.
        const double two_pass = median([&]() {
            (void)ttml::metal::ring_sdpa_bw(
                dO, O, q, k, v, lse, cp_size, cp_axis, /* step */ 0, ttml::metal::AttentionMaskType::Causal,
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward, step_bf16, step_bf16, step_bf16);
        });
        std::cout << "  ring_sdpa_bw, one step (Q pass + KV pass)   " << two_pass << "\n";
        for (const uint32_t Bt : {1u, 2u}) {
            if (rows_per_chip % (2u * Bt * 32u) != 0u) {
                continue;
            }
            const double cyclic = median([&]() {
                ttnn::copy(zero_fp32, step_fp32);
                (void)ttml::metal::ring_cyclic_sdpa_bw(
                    q, k, v, dO, lse, D, cp_size, cp_axis, /* step */ 0, ttml::metal::AttentionMaskType::Causal,
                    ttml::metal::RingCyclicDirection::Backward, Bt, false, /* accumulate */ false, step_fp32, step_fp32, step_fp32);
            });
            std::cout << "  ring_cyclic_sdpa_bw, one step, Bt=" << Bt << " (+zero) " << cyclic << "\n";
        }
        // The per-step host-side total each driver pays besides its op:
        // 3 zeroes, 3 accumulates, 4 shifts (2 tensors + 2 accumulators).
        std::cout << "  => per-step glue: 3 zero + 3 add + 4 shift ~ "
                  << 3 * zero + 3 * add + 2 * shift + 2 * shift32 << " us, x" << cp_size << " steps ~ "
                  << (3 * zero + 3 * add + 2 * shift + 2 * shift32) * cp_size / 1e3 << " ms\n";
    }
}

// Disabled by default: it is a measurement, not an assertion, and it costs a
// few seconds per shape. Run with --gtest_also_run_disabled_tests.
TEST_F(LoudboxRingSDPATest, DISABLED_TimeTheBaselineBackward) {
    const uint32_t cp_size = ttml::autograd::ctx().get_parallelism_context().get_cp_size();
    std::cout << "ring backward, " << cp_size << " chips, ttml::metal::sdpa_bw per step\n";
    for (const auto [batch, heads, seq_len, head_dim] :
         std::vector<std::array<size_t, 4>>{
             {1, 4, 128 * cp_size, 64},
             {1, 4, 256 * cp_size, 64},
             {1, 8, 256 * cp_size, 64},
             {1, 4, 512 * cp_size, 64},
             {1, 4, 256 * cp_size, 128}}) {
        const double seconds = time_ring_backward(batch, heads, seq_len, head_dim);
        std::cout << "  batch=" << batch << " heads=" << heads << " N=" << seq_len << " d=" << head_dim
                  << " (" << seq_len / cp_size << " rows per chip): " << seconds * 1e3 << " ms\n";
    }
}
