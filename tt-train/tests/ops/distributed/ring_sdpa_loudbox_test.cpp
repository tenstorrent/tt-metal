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
#include <sstream>
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
using ttml::ttnn_fixed::distributed::RingShiftTransport;
using ttml::metal::ops::RingLayout;

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

//: Dense causal attention forward in float32. Query is (B, H, S, D); key and
//: value are (B, G, S, D) with G dividing H (grouped-query attention: query
//: head h uses key head h / (H / G)), G = H being plain multi-head attention.
RefForward reference_forward(
    const xt::xarray<float>& query, const xt::xarray<float>& key, const xt::xarray<float>& value) {
    const auto shape = query.shape();
    const size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];
    const size_t G = key.shape()[1];
    const size_t heads_per_group = H / G;
    const float scale = 1.0F / std::sqrt(static_cast<float>(D));

    xt::xarray<float> weights = xt::zeros<float>({B, H, S, S});
    xt::xarray<float> output = xt::zeros<float>({B, H, S, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t g = h / heads_per_group;
            for (size_t i = 0; i < S; ++i) {
                // Causal: only keys 0..i contribute, so the row's softmax runs
                // over that prefix and the rest stay zero.
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j <= i; ++j) {
                    float dot = 0.0F;
                    for (size_t d = 0; d < D; ++d) {
                        dot += query(b, h, i, d) * key(b, g, j, d);
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
                        acc += weights(b, h, i, j) * value(b, g, j, d);
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

//: Dense causal attention backward in float32, from the saved weights. dK and
//: dV take the key's shape: under grouped-query attention the query heads of
//: one key head add into it.
RefGrads reference_backward(
    const xt::xarray<float>& query,
    const xt::xarray<float>& key,
    const xt::xarray<float>& value,
    const xt::xarray<float>& weights,
    const xt::xarray<float>& grad_output,
    const float scale) {
    const auto shape = query.shape();
    const size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];
    const size_t G = key.shape()[1];
    const size_t heads_per_group = H / G;

    xt::xarray<float> dQ = xt::zeros<float>({B, H, S, D});
    xt::xarray<float> dK = xt::zeros<float>({B, G, S, D});
    xt::xarray<float> dV = xt::zeros<float>({B, G, S, D});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t g = h / heads_per_group;
            for (size_t i = 0; i < S; ++i) {
                // dV_j += P_ij dO_i, and dP_ij = dO_i . V_j
                std::vector<float> dP(i + 1, 0.0F);
                for (size_t j = 0; j <= i; ++j) {
                    float acc = 0.0F;
                    for (size_t d = 0; d < D; ++d) {
                        dV(b, g, j, d) += weights(b, h, i, j) * grad_output(b, h, i, d);
                        acc += grad_output(b, h, i, d) * value(b, g, j, d);
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
                        dQ(b, h, i, d) += dS * key(b, g, j, d);
                        dK(b, g, j, d) += dS * query(b, h, i, d);
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
        // The fabric packet payload. tt-metal's default of 4352 bytes carries
        // one 4 KB FP32 tile per packet, which keeps the socket ops from
        // packing pages by bank and halves the shift's bandwidth for the
        // accumulators (TimeTheShift: 4 MB in 616 us at 4352, 254 us at
        // 8704). 8704 is what tt-metal's own socket benchmark uses; Blackhole
        // allows up to 15232, which measured the same. Both transports and
        // both backwards run under the same value, so it moves nothing
        // between them. TTML_LOUDBOX_FABRIC_PAYLOAD overrides it.
        std::optional<size_t> payload = 8704U;
        if (const char* env = std::getenv("TTML_LOUDBOX_FABRIC_PAYLOAD"); env != nullptr && *env != '\0') {
            payload = static_cast<size_t>(std::strtoul(env, nullptr, 10));
        }
        ttml::ttnn_fixed::distributed::enable_fabric(kExpectedChips, payload);
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

    // The direct transport is a different socket, different kernels and a
    // different number of cores; it has to land the same bytes. Both
    // directions, both dtypes the ring moves, and every link count from one
    // up to what the fabric offers, bitwise against the Fifo result.
    const ttnn::Tensor fp32 = ttnn::typecast(tt_tensor, ttnn::DataType::FLOAT32);
    for (const auto direction : {RingShiftDirection::Backward, RingShiftDirection::Forward}) {
        for (const ttnn::Tensor& source : {tt_tensor, fp32}) {
            const auto fifo = core::to_xtensor<float>(
                ttnn_fixed::distributed::ring_shift(source, cp_axis, direction, RingShiftTransport::Fifo),
                core::IdentityComposer{});
            for (const uint32_t connections : {1U, 0U}) {
                const auto direct = core::to_xtensor<float>(
                    ttnn_fixed::distributed::ring_shift(
                        source, cp_axis, direction, RingShiftTransport::Direct, connections),
                    core::IdentityComposer{});
                ASSERT_EQ(fifo.size(), direct.size());
                for (size_t dev = 0; dev < fifo.size(); ++dev) {
                    EXPECT_TRUE(fifo[dev] == direct[dev])
                        << "direct transport (" << (connections == 0U ? "all links" : "one link") << ") differs from fifo on device "
                        << dev;
                }
            }
        }
    }
}

// One tensor through the fused shift, both directions, checked against the
// neighbour's data. The smallest thing that can fail.
TEST_F(LoudboxRingSDPATest, RingShiftFusedOneTensor) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    auto& rng = autograd::ctx().get_generator();
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const xt::xarray<float> full = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{1UL, 2UL, 256UL * cp_size, 64UL}, -2.0F, 2.0F, rng());
    const auto tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(full, device, ttnn::Layout::TILE, mapper.get());
    const auto before = core::to_xtensor<float>(tensor, core::IdentityComposer{});
    const uint32_t cols = topology().cols;
    for (const auto direction : {RingShiftDirection::Forward, RingShiftDirection::Backward}) {
        const auto shifted = ttnn_fixed::distributed::ring_shift_many({tensor}, cp_axis, direction, RingShiftTransport::Direct);
        const auto after = core::to_xtensor<float>(shifted[0], core::IdentityComposer{});
        for (uint32_t row = 0; row < topology().rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                const size_t dst = row * cols + col;
                const size_t src = direction == RingShiftDirection::Forward ? row * cols + ((col + cols - 1U) % cols)
                                                                            : row * cols + ((col + 1U) % cols);
                EXPECT_TRUE(xt::all(xt::equal(before[src], after[dst])))
                    << "device " << dst << " should hold what device " << src << " had ("
                    << (direction == RingShiftDirection::Forward ? "forward" : "backward") << ")";
            }
        }
    }
}

// The fused shift: one launch that moves several tensors, every chip a
// sender and a receiver at once. Bitwise against the two-phase direct
// transport (itself checked against Fifo above), both directions, bf16 and
// FP32, tensors of different sizes in one call, and a lone tensor.
TEST_F(LoudboxRingSDPATest, RingShiftFusedMatchesTwoPhase) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    ASSERT_TRUE(pctx.is_cp_enabled());
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    auto& rng = autograd::ctx().get_generator();
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto make = [&](uint32_t heads, uint32_t rows_per_chip, uint32_t dim, ttnn::DataType dtype) {
        const xt::xarray<float> full = ttml::test_utils::make_uniform_xarray<float>(
            std::array<std::size_t, 4>{1UL, heads, static_cast<std::size_t>(rows_per_chip) * cp_size, dim}, -2.0F,
            2.0F, rng());
        const auto bf16 = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(full, device, ttnn::Layout::TILE, mapper.get());
        return dtype == ttnn::DataType::BFLOAT16 ? bf16 : ttnn::typecast(bf16, dtype);
    };
    const std::vector<ttnn::Tensor> tensors{
        make(2U, 256U, 64U, ttnn::DataType::BFLOAT16),
        make(2U, 256U, 64U, ttnn::DataType::FLOAT32),
        make(1U, 64U, 128U, ttnn::DataType::BFLOAT16),
        make(3U, 96U, 32U, ttnn::DataType::FLOAT32),
    };
    for (const auto direction : {RingShiftDirection::Backward, RingShiftDirection::Forward}) {
        const auto fused = ttnn_fixed::distributed::ring_shift_many(tensors, cp_axis, direction, RingShiftTransport::Direct);
        ASSERT_EQ(fused.size(), tensors.size());
        for (size_t t = 0; t < tensors.size(); ++t) {
            const auto expected = core::to_xtensor<float>(
                ttnn_fixed::distributed::ring_shift(tensors[t], cp_axis, direction, RingShiftTransport::DirectTwoPhase),
                core::IdentityComposer{});
            const auto got = core::to_xtensor<float>(fused[t], core::IdentityComposer{});
            ASSERT_EQ(expected.size(), got.size());
            for (size_t dev = 0; dev < expected.size(); ++dev) {
                EXPECT_TRUE(xt::all(xt::equal(expected[dev], got[dev])))
                    << "fused shift differs from the two-phase one: tensor " << t << ", device " << dev << ", "
                    << (direction == RingShiftDirection::Forward ? "forward" : "backward");
            }
        }
        // A single tensor goes through the same op.
        const auto lone = core::to_xtensor<float>(
            ttnn_fixed::distributed::ring_shift(tensors[1], cp_axis, direction, RingShiftTransport::Direct),
            core::IdentityComposer{});
        const auto lone_expected = core::to_xtensor<float>(
            ttnn_fixed::distributed::ring_shift(tensors[1], cp_axis, direction, RingShiftTransport::DirectTwoPhase),
            core::IdentityComposer{});
        for (size_t dev = 0; dev < lone.size(); ++dev) {
            EXPECT_TRUE(xt::all(xt::equal(lone_expected[dev], lone[dev]))) << "lone fused shift differs on device " << dev;
        }
    }
    // Twice more with the same tensors: the program cache path and the
    // socket's state carried from one launch to the next.
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto again = ttnn_fixed::distributed::ring_shift_many(
            tensors, cp_axis, RingShiftDirection::Forward, RingShiftTransport::Direct);
        const auto expected = core::to_xtensor<float>(
            ttnn_fixed::distributed::ring_shift(
                tensors[0], cp_axis, RingShiftDirection::Forward, RingShiftTransport::DirectTwoPhase),
            core::IdentityComposer{});
        const auto got = core::to_xtensor<float>(again[0], core::IdentityComposer{});
        for (size_t dev = 0; dev < got.size(); ++dev) {
            EXPECT_TRUE(xt::all(xt::equal(expected[dev], got[dev]))) << "repeat " << repeat << " device " << dev;
        }
    }
}

// How fast a ring shift moves bytes, per transport, at the sizes the ring
// attention backward shifts: K-sized bf16 and an FP32 accumulator of the same
// shape. The breakdown of a step found the Fifo shift at 2 MB in 1.2 ms and
// 4 MB in 2.2 ms, under 2.5 GB/s; this is the number to move, and the
// direct transport is the candidate.
TEST_F(LoudboxRingSDPATest, DISABLED_TimeTheShift) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    auto& rng = autograd::ctx().get_generator();

    const auto median_us = [&](const std::function<void()>& f) {
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
        return samples[samples.size() / 2] * 1e6;
    };

    std::cout << "ring_shift on " << cp_size << " chips, heads=4 d=64 (us, median of 7; GB/s per chip)\n";
    for (const size_t rows_per_chip : {128UL, 512UL, 2048UL, 4096UL, 8192UL}) {
        const std::array<std::size_t, 4> shape{1UL, 4UL, rows_per_chip * cp_size, 64UL};
        const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
        const auto k = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, rng()), device, ttnn::Layout::TILE,
            mapper.get());
        const ttnn::Tensor acc = ttnn::zeros_like(k, ttnn::DataType::FLOAT32);
        const double bf16_bytes = static_cast<double>(4 * rows_per_chip * 64 * 2);
        std::ostringstream line;
        line << "  rows/chip=" << rows_per_chip << " (bf16 " << bf16_bytes / 1e6 << " MB, fp32 " << 2 * bf16_bytes / 1e6
             << " MB):";
        const auto report = [&](const char* name, const ttnn::Tensor& t, const double bytes,
                                const RingShiftTransport transport, const uint32_t connections) {
            const double us = median_us([&]() {
                (void)ttnn_fixed::distributed::ring_shift(
                    t, cp_axis, RingShiftDirection::Forward, transport, connections);
            });
            line << " " << name << " " << us << " (" << bytes / us / 1e3 << " GB/s)";
        };
        report("| fifo bf16", k, bf16_bytes, RingShiftTransport::Fifo, 1U);
        report("two-phase bf16", k, bf16_bytes, RingShiftTransport::DirectTwoPhase, 0U);
        report("fused bf16", k, bf16_bytes, RingShiftTransport::Direct, 0U);
        report("| fifo fp32", acc, 2 * bf16_bytes, RingShiftTransport::Fifo, 1U);
        report("two-phase fp32", acc, 2 * bf16_bytes, RingShiftTransport::DirectTwoPhase, 0U);
        report("fused fp32", acc, 2 * bf16_bytes, RingShiftTransport::Direct, 0U);
        std::cout << line.str() << "\n";
    }

    // The backward step's set at the model shape: K and V in bf16, dK and dV
    // in FP32, 10 key heads of 5632 rows a chip -- four two-phase shifts
    // against one fused launch.
    {
        const std::array<std::size_t, 4> shape{1UL, 10UL, 5632UL * cp_size, 64UL};
        const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
        const auto k = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, rng()), device, ttnn::Layout::TILE,
            mapper.get());
        const auto v = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, rng()), device, ttnn::Layout::TILE,
            mapper.get());
        const ttnn::Tensor dk = ttnn::zeros_like(k, ttnn::DataType::FLOAT32);
        const ttnn::Tensor dv = ttnn::zeros_like(v, ttnn::DataType::FLOAT32);
        const std::vector<ttnn::Tensor> set{k, v, dk, dv};
        const double bytes = 2.0 * (10.0 * 5632.0 * 64.0 * 2.0) + 2.0 * (10.0 * 5632.0 * 64.0 * 4.0);
        const double two_phase = median_us([&]() {
            for (const auto& t : set) {
                (void)ttnn_fixed::distributed::ring_shift(
                    t, cp_axis, RingShiftDirection::Forward, RingShiftTransport::DirectTwoPhase, 0U);
            }
        });
        const double fused = median_us([&]() {
            (void)ttnn_fixed::distributed::ring_shift_many(
                set, cp_axis, RingShiftDirection::Forward, RingShiftTransport::Direct, 0U);
        });
        std::cout << "  backward set (K, V bf16 + dK, dV fp32, 10 heads x 5632 rows, " << bytes / 1e6
                  << " MB a chip): two-phase " << two_phase << " us (" << bytes / two_phase / 1e3
                  << " GB/s), fused " << fused << " us (" << bytes / fused / 1e3 << " GB/s), "
                  << (fused / two_phase - 1.0) * 100.0 << "%\n";
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

size_t cp_size_of() {
    return ttml::autograd::ctx().get_parallelism_context().get_cp_size();
}

//: The zigzag dealing of a sequence to d chips, as a permutation of rows.
//
// The sequence is 2d chunks and chip r holds chunks r and 2d - 1 - r back to
// back. Reordering the full tensor so that its r-th contiguous d-th is
// exactly that lets the ordinary contiguous shard mapper place it; the
// inverse puts gathered results back in sequence order.
xt::xarray<float> zigzag_order(const xt::xarray<float>& x, const size_t d) {
    const size_t N = x.shape()[2];
    const size_t n = N / (2 * d);
    xt::xarray<float> out = xt::zeros<float>(x.shape());
    for (size_t r = 0; r < d; ++r) {
        for (const auto [slot, chunk] : {std::pair{0UL, r}, std::pair{1UL, 2 * d - 1 - r}}) {
            const size_t dst = (2 * r + slot) * n;
            xt::view(out, xt::all(), xt::all(), xt::range(dst, dst + n), xt::all()) =
                xt::view(x, xt::all(), xt::all(), xt::range(chunk * n, chunk * n + n), xt::all());
        }
    }
    return out;
}

xt::xarray<float> zigzag_unorder(const xt::xarray<float>& y, const size_t d) {
    const size_t N = y.shape()[2];
    const size_t n = N / (2 * d);
    xt::xarray<float> out = xt::zeros<float>(y.shape());
    for (size_t r = 0; r < d; ++r) {
        for (const auto [slot, chunk] : {std::pair{0UL, r}, std::pair{1UL, 2 * d - 1 - r}}) {
            const size_t src = (2 * r + slot) * n;
            xt::view(out, xt::all(), xt::all(), xt::range(chunk * n, chunk * n + n), xt::all()) =
                xt::view(y, xt::all(), xt::all(), xt::range(src, src + n), xt::all());
        }
    }
    return out;
}

void run_ring_attention(
    const size_t batch,
    const size_t num_heads,
    const size_t seq_len,
    const size_t head_dim,
    const bool test_backward,
    const RingShiftTransport transport = RingShiftTransport::Fifo,
    const RingLayout layout = RingLayout::Contiguous,
    const ttml::ops::distributed::RingBackwardKind kind = ttml::ops::distributed::RingBackwardKind::TwoPass,
    const uint32_t rows_per_block_tiles = 1U,
    // Key/value heads; fewer than num_heads is grouped-query attention. 0
    // means num_heads.
    const size_t num_kv_heads_or_zero = 0,
    const ttml::ops::distributed::RingForwardKind forward = ttml::ops::distributed::RingForwardKind::TwoPass,
    // Inputs are uniform(0, input_scale). The default 2 makes the scores large and the softmax nearly
    // one-hot, where the backward's (dP - D) cancellation amplifies any forward rounding; 0.5 is a
    // regime like a trained model's, softer rows.
    const float input_scale = 2.0F) {
    using namespace ttml;
    const size_t num_kv_heads = num_kv_heads_or_zero == 0 ? num_heads : num_kv_heads_or_zero;
    ASSERT_EQ(num_heads % num_kv_heads, 0U) << "query heads must be a multiple of key/value heads";
    const bool zigzag = layout == RingLayout::Zigzag;
    // How the host lays the sequence out for the chips, and how it reads the
    // chips' results back into sequence order.
    const auto to_chips = [&](const xt::xarray<float>& x) { return zigzag ? zigzag_order(x, cp_size_of()) : x; };
    const auto from_chips = [&](const xt::xarray<float>& x) { return zigzag ? zigzag_unorder(x, cp_size_of()) : x; };

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

    seed_for_case({batch, num_heads + 1000 * num_kv_heads, seq_len, head_dim, test_backward ? 1U : 0U});
    auto& rng = autograd::ctx().get_generator();
    const std::array<std::size_t, 4> qkv_shape{batch, num_heads, seq_len, head_dim};
    const std::array<std::size_t, 4> kv_shape{batch, num_kv_heads, seq_len, head_dim};
    const xt::xarray<float> query_xt =
        ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, input_scale, rng());
    const xt::xarray<float> key_xt = ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, input_scale, rng());
    const xt::xarray<float> value_xt =
        ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, input_scale, rng());

    const auto ref = reference_forward(query_xt, key_xt, value_xt);

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    auto query_tensor = autograd::create_tensor(
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            to_chips(query_xt), device, ttnn::Layout::TILE, mapper.get()),
        /* requires_grad */ true);
    auto key_tensor = autograd::create_tensor(
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(to_chips(key_xt), device, ttnn::Layout::TILE, mapper.get()),
        /* requires_grad */ true);
    auto value_tensor = autograd::create_tensor(
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            to_chips(value_xt), device, ttnn::Layout::TILE, mapper.get()),
        /* requires_grad */ true);

    // Causal masking is generated on device from the mask type; the op rejects
    // an explicit mask tensor in CP mode.
    auto output_tensor = ops::distributed::ring_attention_sdpa(
        query_tensor, key_tensor, value_tensor, /*mask=*/std::nullopt, ttml::metal::AttentionMaskType::Causal,
        kind, rows_per_block_tiles, transport, layout, forward);

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
        from_chips(gather_cp(per_device_output, batch, num_heads, seq_len, head_dim, seq_per_device));
    const float fw_rtol = 1e-2F;
    const float fw_atol = 5e-1F;
    EXPECT_TRUE(xt::allclose(ref.output, gathered_output, fw_rtol, fw_atol))
        << "ring attention output does not match the dense reference";

    // Negative control. A ring that never moved a chunk would compute
    // attention block-diagonally, which is a plausible-looking wrong answer.
    // If the tolerance above cannot separate the two, it is too loose to be
    // evidence of anything.
    // Only where the softmax is sharp: with soft rows (a small input_scale) full and block-diagonal
    // attention both average most of V and land within any tolerance of each other, so the control
    // says nothing there; the sharp-regime tests are what guard the ring's data movement.
    if (input_scale >= 2.0F) {
        const auto block_diagonal = block_diagonal_reference(query_xt, key_xt, value_xt, seq_per_device);
        EXPECT_FALSE(xt::allclose(ref.output, block_diagonal, fw_rtol, fw_atol))
            << "the forward tolerance cannot tell full attention from block-diagonal attention, "
               "so passing it says nothing about the ring";
    }

    if (!test_backward) {
        return;
    }

    const xt::xarray<float> grad_output_xt =
        ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());
    const auto ref_grads =
        reference_backward(query_xt, key_xt, value_xt, ref.weights, grad_output_xt, ref.scale);

    output_tensor->set_grad(core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        to_chips(grad_output_xt), device, ttnn::Layout::TILE, mapper.get()));
    output_tensor->backward();

    const auto gathered_dQ = from_chips(gather_cp(
        core::to_xtensor<float>(query_tensor->get_grad(), core::IdentityComposer{}),
        batch, num_heads, seq_len, head_dim, seq_per_device));
    const auto gathered_dK = from_chips(gather_cp(
        core::to_xtensor<float>(key_tensor->get_grad(), core::IdentityComposer{}),
        batch, num_kv_heads, seq_len, head_dim, seq_per_device));
    const auto gathered_dV = from_chips(gather_cp(
        core::to_xtensor<float>(value_tensor->get_grad(), core::IdentityComposer{}),
        batch, num_kv_heads, seq_len, head_dim, seq_per_device));

    // The same grading the Galaxy suite uses, and for the same reasons:
    // uniform(0, 2) inputs make rowsum(dO o O) large against the (dP - u)
    // cancellation, so bf16 rounding of the saved output and of each ring
    // step lands mostly in dK and dQ. dK is graded against its own largest
    // true value because its accumulation length is the query sequence, so
    // its magnitude and matmul noise grow with S while dQ and dV stay O(1).
    const float rtol = 3e-2F;
    const float atol = 5e-2F;
    const float dk_atol = std::max(atol, 2e-2F * xt::amax(xt::abs(ref_grads.dK))());
    const auto report = [&](const xt::xarray<float>& expected, const xt::xarray<float>& got) {
        const xt::xarray<float> diff = xt::abs(got - expected);
        const auto flat = static_cast<size_t>(std::distance(diff.begin(), std::max_element(diff.begin(), diff.end())));
        const size_t row = (flat / expected.shape()[3]) % expected.shape()[2];
        const auto rel_rms = [](const xt::xarray<float>& e, const xt::xarray<float>& g) {
            return std::sqrt(xt::mean(xt::square(g - e))()) / (std::sqrt(xt::mean(xt::square(e))()) + 1e-12F);
        };
        std::string per_chip;
        for (size_t c = 0; c < cp_size; ++c) {
            const size_t r0 = c * seq_per_device;
            const size_t r1 = (c + 1) * seq_per_device;
            per_chip += " chip" + std::to_string(c) + "=" +
                        std::to_string(rel_rms(
                            xt::xarray<float>(xt::view(expected, xt::all(), xt::all(), xt::range(r0, r1), xt::all())),
                            xt::xarray<float>(xt::view(got, xt::all(), xt::all(), xt::range(r0, r1), xt::all()))));
        }
        return "max_abs_diff=" + std::to_string(xt::amax(diff)()) + " at row " + std::to_string(row) +
               " ref_amax=" + std::to_string(xt::amax(xt::abs(expected))()) +
               " rel_rms=" + std::to_string(rel_rms(expected, got)) + " per chip (sequence order):" + per_chip;
    };
    if (forward == ttml::ops::distributed::RingForwardKind::Ttnn) {
        // ttnn's forward accumulates its output in bf16 (~3% RMS off where sdpa_fw is ~0.5%) and its
        // lse is within 2.6e-2. The backward's dS = P (dP - D) cancels for rows with few keys -- the
        // first rows of every causal chunk, whose true dQ is nearly zero -- and there the 3% on
        // D = rowsum(dO . O) becomes most of dQ. So this grades what the kernel delivers today: the
        // forward output tightly, dV and dK by RMS at 5% of scale, dQ by RMS at 30% of scale, all
        // printed, until ttnn's intermediates can be Float32 (they broke the kernel when tried).
        const auto grade = [&](const xt::xarray<float>& want, const xt::xarray<float>& got, const char* name,
                               float rms_bound) {
            const float scale = xt::amax(xt::abs(want))();
            const float rms = std::sqrt(xt::mean(xt::square(got - want))());
            std::cout << "  " << name << " (ttnn forward): rms " << rms << " = " << 100.0F * rms / scale
                      << "% of scale " << scale << "\n";
            EXPECT_LE(rms, rms_bound * scale) << name << ": " << report(want, got);
        };
        EXPECT_TRUE(xt::allclose(ref.output, gathered_output, 1e-2F, 2e-2F))
            << "forward output (ttnn): " << report(ref.output, gathered_output);
        grade(ref_grads.dV, gathered_dV, "dV", 0.05F);
        grade(ref_grads.dK, gathered_dK, "dK", 0.05F);
        grade(ref_grads.dQ, gathered_dQ, "dQ", 0.30F);
    } else if (kind == ttml::ops::distributed::RingBackwardKind::TwoPass) {
        EXPECT_TRUE(xt::allclose(ref_grads.dQ, gathered_dQ, rtol, atol)) << "dQ: " << report(ref_grads.dQ, gathered_dQ);
        EXPECT_TRUE(xt::allclose(ref_grads.dK, gathered_dK, rtol, dk_atol))
            << "dK: " << report(ref_grads.dK, gathered_dK);
        EXPECT_TRUE(xt::allclose(ref_grads.dV, gathered_dV, rtol, atol)) << "dV: " << report(ref_grads.dV, gathered_dV);
    } else {
        // The cyclic kinds are graded as compare_backward_implementations
        // grades them: by root-mean-square against the reference, with the
        // max only as a coarse net. In this uniform(0, 2) regime the softmax
        // is nearly one-hot and dQ is small for late rows, so a max over 10^5
        // elements is a statistic about the unluckiest one; measured, the
        // cyclic dQ max lands at 5-7% of scale where two-pass lands at 3-5%,
        // with both at an RMS under 2% of scale.
        const auto grade = [&](const xt::xarray<float>& want, const xt::xarray<float>& got, const char* name) {
            const float scale = xt::amax(xt::abs(want))();
            const float rms = std::sqrt(xt::mean(xt::square(got - want))());
            // Where the large errors are, by (batch, head, tile row): a whole
            // wrong tile hides inside the RMS but not here.
            {
                const auto sh = want.shape();
                std::ostringstream where;
                size_t tiles_over = 0;
                for (size_t b = 0; b < sh[0]; ++b) {
                    for (size_t h = 0; h < sh[1]; ++h) {
                        for (size_t t0 = 0; t0 < sh[2]; t0 += 32) {
                            const xt::xarray<float> d = xt::abs(
                                xt::view(got, b, h, xt::range(t0, t0 + 32), xt::all()) -
                                xt::view(want, b, h, xt::range(t0, t0 + 32), xt::all()));
                            const float m = xt::amax(d)();
                            if (m > 0.1F * scale) {
                                ++tiles_over;
                                if (tiles_over <= 12) {
                                    where << " (b" << b << ",h" << h << ",rows " << t0 << "-" << t0 + 31 << ": max "
                                          << m << ", " << xt::sum(xt::cast<int>(d > 0.1F * scale))() << " elems)";
                                }
                            }
                        }
                    }
                }
                std::cout << "  " << name << " tiles with an element over 10% of scale: " << tiles_over << where.str()
                          << "\n";
            }
            std::cout << "  " << name << ": rms " << rms << " = " << 100.0F * rms / scale << "% of scale " << scale
                      << "\n";
            EXPECT_LE(rms, 0.02F * scale) << name << ": rms " << rms << " on scale " << scale << "; " << report(want, got);
            EXPECT_LE(xt::amax(xt::abs(got - want))(), 0.25F * scale) << name << ": " << report(want, got);
        };
        grade(ref_grads.dQ, gathered_dQ, "dQ");
        grade(ref_grads.dK, gathered_dK, "dK");
        grade(ref_grads.dV, gathered_dV, "dV");
    }
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

// The same reference check with every shift in the forward and the backward
// on the direct transport: six shifts a step, both dtypes, both directions.
TEST_F(LoudboxRingSDPATest, CausalBackwardWithDirectShifts) {
    run_ring_attention(1, 4, seq_for(128), 64, /*test_backward=*/true, RingShiftTransport::Direct);
}

// The zigzag layout: two chunks per chip, both live pairs a step, no chip
// skipping. The host deals the sequence out in zigzag order and reads the
// results back into sequence order, so the reference is the same dense one.
TEST_F(LoudboxRingSDPATest, ZigzagCausalForward) {
    run_ring_attention(1, 4, seq_for(128), 64, /*test_backward=*/false, RingShiftTransport::Direct, RingLayout::Zigzag);
}

TEST_F(LoudboxRingSDPATest, ZigzagCausalBackwardTwoPass) {
    run_ring_attention(1, 4, seq_for(128), 64, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Zigzag);
}

TEST_F(LoudboxRingSDPATest, ZigzagCausalBackwardCyclicInPlace) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    run_ring_attention(
        1, 4, seq_for(128), 64, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 1U);
}

TEST_F(LoudboxRingSDPATest, ZigzagCausalBackwardCyclicWithTallBlocks) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    run_ring_attention(
        2, 3, seq_for(256), 64, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 2U);
}

// Grouped-query attention through the ring: four query heads on two key
// heads, both backwards, on the zigzag layout the training run will use.
TEST_F(LoudboxRingSDPATest, ZigzagCausalBackwardTwoPassGroupedHeads) {
    run_ring_attention(
        1, 4, seq_for(128), 64, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Zigzag,
        ttml::ops::distributed::RingBackwardKind::TwoPass, /*Bt*/ 1U, /*kv heads*/ 2);
}

TEST_F(LoudboxRingSDPATest, ZigzagCausalBackwardCyclicInPlaceGroupedHeads) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    run_ring_attention(
        1, 4, seq_for(128), 64, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 1U, /*kv heads*/ 2);
}

// Eight query heads on two key heads with tall blocks: the ratio the Llama
// configs use, and the sequencing of four heads in turn on one core group.
TEST_F(LoudboxRingSDPATest, ZigzagCausalBackwardCyclicGroupedHeadsFourPerKeyHead) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    run_ring_attention(
        1, 8, seq_for(256), 64, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 2U, /*kv heads*/ 2);
}

// The ttnn forward (ttnn's chunk-blocked kernel with the lse output) through
// the ring, feeding both backwards: the forward's lse is what the backward
// recomputes its probabilities from, so this grades the whole chain against
// the dense reference. In a softer regime than the other tests (inputs
// uniform(0, 0.5)): ttnn's kernel accumulates its output in bf16 and its
// output is ~3% RMS off where sdpa_fw is ~0.5%, and in the near-one-hot
// regime of the default inputs that 3% enters D = rowsum(dO . O) and the
// backward's (dP - D) cancellation turns it into a dQ that is wrong by more
// than its own size. In training (character Llama, 45k tokens a step) the
// loss curve with this forward matched the two-pass forward's to bf16
// noise over 40 steps; the sharp regime is where the accuracy gap shows.
TEST_F(LoudboxRingSDPATest, TtnnForwardContiguousTwoPassBackward) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    using Fwd = ttml::ops::distributed::RingForwardKind;
    run_ring_attention(
        1, 4, seq_for(128), 64, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Contiguous,
        Kind::TwoPass, /*Bt*/ 1U, /*kv heads*/ 0, Fwd::Ttnn, /*input_scale*/ 0.5F);
}
TEST_F(LoudboxRingSDPATest, TtnnForwardZigzagCyclicBackward) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    using Fwd = ttml::ops::distributed::RingForwardKind;
    run_ring_attention(
        1, 4, seq_for(256), 64, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 2U, /*kv heads*/ 0, Fwd::Ttnn, /*input_scale*/ 0.5F);
}
TEST_F(LoudboxRingSDPATest, TtnnForwardZigzagCyclicBackwardGroupedHeads) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    using Fwd = ttml::ops::distributed::RingForwardKind;
    run_ring_attention(
        1, 8, seq_for(256), 64, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 2U, /*kv heads*/ 2, Fwd::Ttnn, /*input_scale*/ 0.5F);
}
// The cyclic forward: Float32 statistics, so it is graded like sdpa_fw (the
// default branch), at the default sharp input regime.
TEST_F(LoudboxRingSDPATest, CyclicForwardContiguousTwoPassBackward) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    using Fwd = ttml::ops::distributed::RingForwardKind;
    run_ring_attention(
        1, 4, seq_for(512), 64, /*test_backward*/ true, RingShiftTransport::Direct, RingLayout::Contiguous,
        Kind::TwoPass, /*Bt*/ 1U, /*kv heads*/ 0, Fwd::Cyclic);
}
TEST_F(LoudboxRingSDPATest, CyclicForwardZigzagCyclicBackward) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    using Fwd = ttml::ops::distributed::RingForwardKind;
    run_ring_attention(
        1, 4, seq_for(1024), 64, /*test_backward*/ true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 2U, /*kv heads*/ 0, Fwd::Cyclic);
}
TEST_F(LoudboxRingSDPATest, CyclicForwardZigzagCyclicBackwardGroupedHeads) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    using Fwd = ttml::ops::distributed::RingForwardKind;
    run_ring_attention(
        1, 4, seq_for(1024), 64, /*test_backward*/ true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 2U, /*kv heads*/ 2, Fwd::Cyclic);
}
TEST_F(LoudboxRingSDPATest, CyclicForwardZigzagWiderHead) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    using Fwd = ttml::ops::distributed::RingForwardKind;
    run_ring_attention(
        1, 4, seq_for(1024), 128, /*test_backward*/ true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 2U, /*kv heads*/ 2, Fwd::Cyclic);
}

TEST_F(LoudboxRingSDPATest, TtnnForwardZigzagWiderHead) {
    using Kind = ttml::ops::distributed::RingBackwardKind;
    using Fwd = ttml::ops::distributed::RingForwardKind;
    run_ring_attention(
        1, 4, seq_for(256), 128, /*test_backward=*/true, RingShiftTransport::Direct, RingLayout::Zigzag,
        Kind::CyclicInPlace, /*Bt*/ 2U, /*kv heads*/ 2, Fwd::Ttnn, /*input_scale*/ 0.5F);
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
    const uint32_t rows_per_block_tiles,
    const size_t num_kv_heads_or_zero = 0) {
    using namespace ttml;
    const size_t num_kv_heads = num_kv_heads_or_zero == 0 ? num_heads : num_kv_heads_or_zero;
    ASSERT_EQ(num_heads % num_kv_heads, 0U) << "query heads must be a multiple of key/value heads";

    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    const size_t seq_per_device = seq_len / cp_size;
    ASSERT_EQ(seq_len % cp_size, 0U);
    // The cyclic schedule needs C = n / (2 * Bt * 32) whole cores per chunk.
    ASSERT_EQ(seq_per_device % (2U * rows_per_block_tiles * 32U), 0U)
        << "chunk of " << seq_per_device << " rows does not divide into cyclic blocks";

    seed_for_case({batch, num_heads + 1000 * num_kv_heads, seq_len, head_dim, rows_per_block_tiles});
    auto& rng = autograd::ctx().get_generator();
    const std::array<std::size_t, 4> qkv_shape{batch, num_heads, seq_len, head_dim};
    const std::array<std::size_t, 4> kv_shape{batch, num_kv_heads, seq_len, head_dim};
    const xt::xarray<float> query_xt = ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng());
    const xt::xarray<float> key_xt = ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, 2.0F, rng());
    const xt::xarray<float> value_xt = ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, 2.0F, rng());
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
        const auto gather = [&](const ttnn::Tensor& t, size_t heads) {
            return gather_cp(
                core::to_xtensor<float>(t, core::IdentityComposer{}), batch, heads, seq_len, head_dim, seq_per_device);
        };
        return Result{
            gather(q->get_grad(), num_heads), gather(k->get_grad(), num_kv_heads), gather(v->get_grad(), num_kv_heads)};
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
        std::cout << "    max |error| vs reference  " << names[k] << ": two-pass " << two_pass_max << " cyclic "
                  << cyclic_max << " on scale " << scale << "\n";
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

// Tall blocks on more than one core per slice. The tall-block case above
// is C = 1, where the relay never forwards; these put the block height and
// the relay together, which is what the ring runs at any real size.
TEST_F(LoudboxRingSDPATest, BothBackwardsAgreeWithTallBlocksOnTwoCores) {
    compare_backward_implementations(1, 4, seq_for(256), 64, /* Bt */ 2);
}
TEST_F(LoudboxRingSDPATest, BothBackwardsAgreeWithTallBlocksOnFourCores) {
    compare_backward_implementations(1, 4, seq_for(512), 64, /* Bt */ 2);
}
TEST_F(LoudboxRingSDPATest, BothBackwardsAgreeWithTallestBlocks) {
    compare_backward_implementations(1, 4, seq_for(512), 64, /* Bt */ 4);
}
// Batch 2, three heads, tall blocks on two cores: the shape the zigzag tests
// use, on the contiguous layout, both implementations against the reference.
TEST_F(LoudboxRingSDPATest, BothBackwardsAgreeAtBatchTwo) {
    compare_backward_implementations(2, 3, seq_for(256), 64, /* Bt */ 2);
}

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

// Grouped-query attention on the contiguous layout: both backwards sum the
// query heads of a key head into its dK and dV, and must agree with the
// reference and each other.
TEST_F(LoudboxRingSDPATest, BothBackwardsAgreeWithGroupedHeads) {
    compare_backward_implementations(1, 4, seq_for(128), 64, /* Bt */ 2, /* kv heads */ 2);
}
TEST_F(LoudboxRingSDPATest, BothBackwardsAgreeWithGroupedHeadsAtBatchTwo) {
    compare_backward_implementations(2, 6, seq_for(128), 64, /* Bt */ 1, /* kv heads */ 3);
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
    uint32_t rows_per_block_tiles = 1U,
    RingShiftTransport transport = RingShiftTransport::Fifo,
    uint32_t samples_to_take = 5U,
    RingLayout layout = RingLayout::Contiguous,
    size_t num_kv_heads_or_zero = 0,
    ttml::ops::distributed::RingForwardKind forward = ttml::ops::distributed::RingForwardKind::TwoPass) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const uint32_t cp_axis = autograd::ctx().get_parallelism_context().get_cp_axis().value();
    auto& rng = autograd::ctx().get_generator();

    const size_t num_kv_heads = num_kv_heads_or_zero == 0 ? num_heads : num_kv_heads_or_zero;
    const std::array<std::size_t, 4> qkv_shape{batch, num_heads, seq_len, head_dim};
    const std::array<std::size_t, 4> kv_shape{batch, num_kv_heads, seq_len, head_dim};
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto to_device = [&](const xt::xarray<float>& x) {
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(x, device, ttnn::Layout::TILE, mapper.get());
    };

    const auto sample = [&]() {
        auto query = autograd::create_tensor(
            to_device(ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng())), true);
        auto key = autograd::create_tensor(
            to_device(ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, 2.0F, rng())), true);
        auto value = autograd::create_tensor(
            to_device(ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, 2.0F, rng())), true);
        auto out = ops::distributed::ring_attention_sdpa(
            query, key, value, std::nullopt, ttml::metal::AttentionMaskType::Causal, kind, rows_per_block_tiles,
            transport, layout, forward);
        out->set_grad(to_device(ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng())));
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});

        const auto start = std::chrono::steady_clock::now();
        out->backward();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };

    sample();  // warm the program cache and the kernel build
    std::vector<double> samples;
    for (uint32_t k = 0; k < samples_to_take; ++k) {
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
             // Larger chunks: where the per-step kernel is a visible share of
             // the whole backward, and the ring shift moves real data.
             {1, 4, 1024, 64, 2},
             {1, 4, 2048, 64, 2},
             {1, 4, 2048, 64, 4},
             {1, 4, 4096, 64, 4},
             // Larger still. 8192 rows is C = 32 at Bt = 4 (three groups of
             // 32 cores, slices looped) or C = 64 at Bt = 2 (one group of 64);
             // Bt = 1 would be C = 128, which has no rectangle and is
             // refused, so the sequence cap (review item 1) is what bounds
             // this table, not memory. 16384 rows is C = 64 at Bt = 4 only.
             {1, 4, 8192, 64, 4},
             {1, 4, 8192, 64, 2},
             {1, 8, 8192, 64, 4},
             {1, 8, 8192, 64, 2},
             {1, 1, 8192, 64, 2},
             {1, 1, 16384, 64, 4},
             {1, 2, 16384, 64, 4},
         }) {
        const size_t rows_per_chip = cfg[2];
        // TTML_LOUDBOX_MIN_ROWS skips the smaller cases, for adding rows to
        // the table without rerunning the ones already in it.
        if (const char* env = std::getenv("TTML_LOUDBOX_MIN_ROWS"); env != nullptr && *env != '\0') {
            if (rows_per_chip < std::strtoul(env, nullptr, 10)) {
                continue;
            }
        }
        const auto Bt = static_cast<uint32_t>(cfg[4]);
        const size_t seq_len = rows_per_chip * cp_size;
        if (rows_per_chip % (2U * Bt * 32U) != 0U) {
            continue;  // the cyclic schedule needs whole cores per chunk
        }
        using Kind = ttml::ops::distributed::RingBackwardKind;
        for (const auto transport : {RingShiftTransport::Fifo, RingShiftTransport::Direct}) {
            const double two_pass =
                time_ring_backward(cfg[0], cfg[1], seq_len, cfg[3], Kind::TwoPass, Bt, transport);
            const double cyclic = time_ring_backward(cfg[0], cfg[1], seq_len, cfg[3], Kind::Cyclic, Bt, transport);
            const double in_place =
                time_ring_backward(cfg[0], cfg[1], seq_len, cfg[3], Kind::CyclicInPlace, Bt, transport);
            std::cout << "  batch=" << cfg[0] << " heads=" << cfg[1] << " N=" << seq_len << " d=" << cfg[3]
                      << " Bt=" << Bt << " (" << rows_per_chip << " rows/chip, C=" << rows_per_chip / (2U * Bt * 32U)
                      << ") " << (transport == RingShiftTransport::Fifo ? "fifo  " : "direct")
                      << " shifts: two-pass " << two_pass * 1e3 << " ms, cyclic " << cyclic * 1e3 << " ms ("
                      << two_pass / cyclic << "x), cyclic in-place " << in_place * 1e3 << " ms ("
                      << two_pass / in_place << "x)\n";
        }
    }
}

// The comparison at sizes where the cyclic side uses every core. A group is
// a C-core rectangle whose parity snake is all single hops, and 110 = 11 x 10
// is tiled by such rectangles only for C in {1, 2, 5, 10, 11, 22, 55, 110};
// with heads a multiple of the group count, every core is busy. Two-pass
// always uses the whole grid, so these rows compare the kernels at equal
// occupancy, and the pairs at 2816 rows x 10 heads (Bt = 2: C = 22, five
// groups; Bt = 4: C = 11, ten groups) separate block height from fusion at
// full chip. Direct shifts only.
TEST_F(LoudboxRingSDPATest, DISABLED_CompareOnTheWholeGrid) {
    const uint32_t cp_size = ttml::autograd::ctx().get_parallelism_context().get_cp_size();
    std::cout << "ring backward on " << cp_size << " chips, whole grid on both sides, median of five\n";
    for (const auto& cfg : std::vector<std::array<size_t, 5>>{
             // batch, heads, rows per chip, head dim, Bt
             {1, 11, 2560, 64, 4},   // C = 10, 11 groups
             {1, 10, 2816, 64, 4},   // C = 11, 10 groups
             {1, 10, 2816, 64, 2},   // C = 22, 5 groups x 2 slices
             {1, 5, 2816, 64, 2},    // C = 22, 5 groups
             {1, 5, 5632, 64, 4},    // C = 22, 5 groups
             {1, 2, 14080, 64, 4},   // C = 55, 2 groups
             {1, 1, 14080, 64, 2},   // C = 110, 1 group
             {1, 1, 28160, 64, 4},   // C = 110, 1 group: the schedule's cap
         }) {
        const size_t rows_per_chip = cfg[2];
        const auto Bt = static_cast<uint32_t>(cfg[4]);
        const size_t seq_len = rows_per_chip * cp_size;
        ASSERT_EQ(rows_per_chip % (2U * Bt * 32U), 0U);
        using Kind = ttml::ops::distributed::RingBackwardKind;
        const auto transport = RingShiftTransport::Direct;
        const double two_pass = time_ring_backward(cfg[0], cfg[1], seq_len, cfg[3], Kind::TwoPass, Bt, transport);
        const double cyclic = time_ring_backward(cfg[0], cfg[1], seq_len, cfg[3], Kind::Cyclic, Bt, transport);
        const double in_place =
            time_ring_backward(cfg[0], cfg[1], seq_len, cfg[3], Kind::CyclicInPlace, Bt, transport);
        std::cout << "  heads=" << cfg[1] << " rows/chip=" << rows_per_chip << " d=" << cfg[3] << " Bt=" << Bt
                  << " (C=" << rows_per_chip / (2U * Bt * 32U) << "): two-pass " << two_pass * 1e3 << " ms, cyclic "
                  << cyclic * 1e3 << " ms (" << two_pass / cyclic << "x), cyclic in-place " << in_place * 1e3
                  << " ms (" << two_pass / in_place << "x)\n";
    }
}

// The layouts against each other: contiguous, where the last chip does
// 2d - 1 times the first's work and sets the pace, and zigzag, where every
// chip does the same two half-size blocks a step. Both kinds on both
// layouts, direct shifts, median of five, ms. Rows per chip is the whole
// local sequence; under zigzag a chunk is half of it, so the cyclic C is
// rows / (4 Bt 32) there.
TEST_F(LoudboxRingSDPATest, DISABLED_CompareLayouts) {
    const uint32_t cp_size = ttml::autograd::ctx().get_parallelism_context().get_cp_size();
    std::cout << "ring backward on " << cp_size << " chips, contiguous against zigzag, direct shifts, median of five\n";
    using Kind = ttml::ops::distributed::RingBackwardKind;
    for (const auto& cfg : std::vector<std::array<size_t, 4>>{
             // heads, rows per chip, head dim, Bt for the cyclic kind
             {4, 1024, 64, 1},
             {4, 2048, 64, 2},
             {4, 4096, 64, 2},
             {4, 4096, 64, 4},
             {4, 8192, 64, 4},
             {10, 5632, 64, 4},  // zigzag chunk 2816: C = 11, ten groups, the whole grid
             {1, 16384, 64, 4},
         }) {
        const size_t heads = cfg[0], rows = cfg[1], d = cfg[2];
        const auto Bt = static_cast<uint32_t>(cfg[3]);
        const size_t seq_len = rows * cp_size;
        if (rows % (4U * Bt * 32U) != 0U) {
            continue;
        }
        const auto run = [&](Kind kind, RingLayout layout) {
            return time_ring_backward(1, heads, seq_len, d, kind, Bt, RingShiftTransport::Direct, 5U, layout) * 1e3;
        };
        const double tp_c = run(Kind::TwoPass, RingLayout::Contiguous);
        const double cy_c = run(Kind::CyclicInPlace, RingLayout::Contiguous);
        const double tp_z = run(Kind::TwoPass, RingLayout::Zigzag);
        const double cy_z = run(Kind::CyclicInPlace, RingLayout::Zigzag);
        const auto pct = [](double a, double b) { return (b / a - 1.0) * 100.0; };
        std::cout << "  heads=" << heads << " rows/chip=" << rows << " d=" << d << " Bt=" << Bt
                  << ": two-pass contiguous " << tp_c << " | two-pass zigzag " << tp_z << " (" << pct(tp_c, tp_z)
                  << "%) | cyclic in-place contiguous " << cy_c << " (" << pct(tp_c, cy_c) << "%) | cyclic in-place zigzag "
                  << cy_z << " (" << pct(tp_c, cy_z) << "% vs two-pass contiguous, " << pct(cy_c, cy_z)
                  << "% vs cyclic contiguous, " << pct(tp_z, cy_z) << "% vs two-pass zigzag)\n";
    }
}

// The two backwards on grouped-query shapes like the Llama configs' -- 32
// query heads on 8 or 4 key heads, 6 on 3 -- zigzag layout, direct shifts,
// median of five. Rows per chip as in CompareLayouts; TTML_LOUDBOX_MIN_ROWS
// skips the small cases, TTML_LOUDBOX_GQA_SHAPES="heads:kv:rows:d:Bt,..."
// replaces the table.
TEST_F(LoudboxRingSDPATest, DISABLED_CompareGroupedHeads) {
    const uint32_t cp_size = ttml::autograd::ctx().get_parallelism_context().get_cp_size();
    std::cout << "ring backward on " << cp_size
              << " chips, grouped-query heads, zigzag, direct shifts, median of five (ms)\n";
    using Kind = ttml::ops::distributed::RingBackwardKind;
    std::vector<std::array<size_t, 5>> table = {
        // query heads, key heads, rows per chip, head dim, Bt for the cyclic
        // kind on the contiguous layout (zigzag sweeps every height)
        {6, 3, 4096, 64, 4},
        {32, 4, 2048, 64, 2},
        {32, 4, 4096, 64, 4},
        {32, 8, 2048, 128, 2},
        {32, 8, 4096, 128, 4},
    };
    if (const char* spec = std::getenv("TTML_LOUDBOX_GQA_SHAPES")) {
        table.clear();
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::array<size_t, 5> cfg{};
            std::stringstream is(item);
            std::string field;
            for (size_t k = 0; k < 5 && std::getline(is, field, ':'); ++k) {
                cfg[k] = std::stoul(field);
            }
            table.push_back(cfg);
        }
    }
    const size_t min_rows = std::getenv("TTML_LOUDBOX_MIN_ROWS") ? std::stoul(std::getenv("TTML_LOUDBOX_MIN_ROWS")) : 0;
    for (const auto& cfg : table) {
        const size_t heads = cfg[0], kv_heads = cfg[1], rows = cfg[2], d = cfg[3];
        const auto Bt = static_cast<uint32_t>(cfg[4]);
        const size_t seq_len = rows * cp_size;
        if (rows < min_rows || rows % (4U * Bt * 32U) != 0U) {
            continue;
        }
        const auto run = [&](Kind kind, RingLayout layout, uint32_t bt) {
            return time_ring_backward(1, heads, seq_len, d, kind, bt, RingShiftTransport::Direct, 5U, layout, kv_heads) *
                   1e3;
        };
        // The two-pass backward on both layouts, since which is faster
        // depends on the chunk size; the cyclic one on zigzag at every block
        // height that divides the chunk (the planner's choice is the best of
        // them), and on the contiguous layout at the table's height. Cores
        // busy on the cyclic side: C = chunk / (2 Bt 32) per group, and the
        // group count is capped to the key heads (batch 1 here).
        const double tp_c = run(Kind::TwoPass, RingLayout::Contiguous, 1U);
        const double tp_z = run(Kind::TwoPass, RingLayout::Zigzag, 1U);
        const double tp_best = std::min(tp_c, tp_z);
        std::ostringstream line;
        line << "  heads=" << heads << " kv_heads=" << kv_heads << " rows/chip=" << rows << " d=" << d
             << ": two-pass contiguous " << tp_c << " zigzag " << tp_z;
        double cy_best = std::numeric_limits<double>::infinity();
        uint32_t cy_best_bt = 0;
        for (const uint32_t bt : {1U, 2U, 4U}) {
            if ((rows / 2) % (2U * bt * 32U) != 0U) {
                continue;
            }
            const double cy = run(Kind::CyclicInPlace, RingLayout::Zigzag, bt);
            // As the planner counts them: groups = min(key heads, how many
            // rectangles of C fit), lowered to a divisor of the key heads.
            const size_t C = (rows / 2) / (2U * bt * 32U);
            size_t groups = std::min<size_t>(kv_heads, 110 / C);
            while (kv_heads % groups != 0) {
                --groups;
            }
            const size_t cores = groups * C;
            line << " | cyclic zigzag Bt=" << bt << " " << cy << " (" << (cy / tp_best - 1.0) * 100.0 << "%, ~"
                 << cores << " cores)";
            if (cy < cy_best) {
                cy_best = cy;
                cy_best_bt = bt;
            }
        }
        const double cy_c = run(Kind::CyclicInPlace, RingLayout::Contiguous, Bt);
        line << " | cyclic contiguous Bt=" << Bt << " " << cy_c << " (" << (cy_c / tp_best - 1.0) * 100.0
             << "%) | best cyclic zigzag Bt=" << cy_best_bt << " " << cy_best << " vs best two-pass " << tp_best
             << ": " << (cy_best / tp_best - 1.0) * 100.0 << "%";
        std::cout << line.str() << "\n";
    }
}

// A training step pays the forward and the backward, and the forward is the
// same two-pass ring for both kinds; this times both, so the end-to-end gain
// of a step can be read next to the backward's. Also printed: the backward's
// useful FLOP rate over the whole ring and its share of the ring's LoFi peak
// (594 TFLOP/s per chip), with the conventions of the single-chip table --
// five matmuls over the causal triangle, 5 S^2 d per head.
namespace {
struct StepTimes {
    double forward_ms{}, backward_ms{};
};
StepTimes time_ring_step(
    size_t batch, size_t num_heads, size_t num_kv_heads, size_t seq_len, size_t head_dim,
    ttml::ops::distributed::RingBackwardKind kind, uint32_t Bt, RingLayout layout, uint32_t samples_to_take,
    ttml::ops::distributed::RingForwardKind forward = ttml::ops::distributed::RingForwardKind::TwoPass,
    RingShiftTransport transport = RingShiftTransport::Direct) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const uint32_t cp_axis = autograd::ctx().get_parallelism_context().get_cp_axis().value();
    auto& rng = autograd::ctx().get_generator();
    const std::array<std::size_t, 4> qkv_shape{batch, num_heads, seq_len, head_dim};
    const std::array<std::size_t, 4> kv_shape{batch, num_kv_heads, seq_len, head_dim};
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto to_device = [&](const xt::xarray<float>& x) {
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(x, device, ttnn::Layout::TILE, mapper.get());
    };
    const auto sample = [&]() {
        auto query = autograd::create_tensor(
            to_device(ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng())), true);
        auto key = autograd::create_tensor(
            to_device(ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, 2.0F, rng())), true);
        auto value = autograd::create_tensor(
            to_device(ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, 2.0F, rng())), true);
        const auto grad = to_device(ttml::test_utils::make_uniform_xarray<float>(qkv_shape, 0.0F, 2.0F, rng()));
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
        const auto t0 = std::chrono::steady_clock::now();
        auto out = ops::distributed::ring_attention_sdpa(
            query, key, value, std::nullopt, ttml::metal::AttentionMaskType::Causal, kind, Bt,
            transport, layout, forward);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
        const auto t1 = std::chrono::steady_clock::now();
        out->set_grad(grad);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
        const auto t2 = std::chrono::steady_clock::now();
        out->backward();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
        const auto t3 = std::chrono::steady_clock::now();
        return StepTimes{
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(t3 - t2).count()};
    };
    sample();  // warm the program cache and the kernel build
    std::vector<double> fwd, bwd;
    for (uint32_t k = 0; k < samples_to_take; ++k) {
        const auto t = sample();
        fwd.push_back(t.forward_ms);
        bwd.push_back(t.backward_ms);
    }
    std::sort(fwd.begin(), fwd.end());
    std::sort(bwd.begin(), bwd.end());
    return {fwd[fwd.size() / 2], bwd[bwd.size() / 2]};
}
}  // namespace

// The ring step, two-pass kernels against the cyclic ones, on the best ring
// either can use: the direct (fused) shifts for both, and the two-pass side on
// whichever layout is faster for it (both are timed). The cyclic side is the
// cyclic forward and the in-place backward on zigzag at block height 4 (the
// planner's choice for these shapes): what differs is the kernels and what a
// single fused kernel allows (in-place Float32 accumulation, tall blocks).
// Median of seven after a warm-up. TTML_LOUDBOX_STEP_SHAPES="heads:kv:rows:d,..."
// replaces the table.
TEST_F(LoudboxRingSDPATest, DISABLED_CompareWithTtTrain) {
    const uint32_t cp_size = ttml::autograd::ctx().get_parallelism_context().get_cp_size();
    std::cout << "ring step on " << cp_size << " chips, two-pass against cyclic, direct shifts, median of seven (ms)\n";
    using Kind = ttml::ops::distributed::RingBackwardKind;
    using Fwd = ttml::ops::distributed::RingForwardKind;
    std::vector<std::array<size_t, 4>> table = {
        {4, 4, 4096, 64},
        {20, 10, 5632, 64},
        {32, 8, 5632, 128},
    };
    if (const char* spec = std::getenv("TTML_LOUDBOX_STEP_SHAPES")) {
        table.clear();
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::array<size_t, 4> cfg{};
            std::stringstream is(item);
            std::string field;
            for (size_t k = 0; k < 4 && std::getline(is, field, ':'); ++k) {
                cfg[k] = std::stoul(field);
            }
            table.push_back(cfg);
        }
    }
    for (const auto& cfg : table) {
        const size_t heads = cfg[0], kv_heads = cfg[1], rows = cfg[2], d = cfg[3];
        const size_t seq_len = rows * cp_size;
        const auto contig = time_ring_step(
            1, heads, kv_heads, seq_len, d, Kind::TwoPass, 1U, RingLayout::Contiguous, 7U, Fwd::TwoPass,
            RingShiftTransport::Direct);
        const auto zigzag = time_ring_step(
            1, heads, kv_heads, seq_len, d, Kind::TwoPass, 1U, RingLayout::Zigzag, 7U, Fwd::TwoPass,
            RingShiftTransport::Direct);
        const auto cyc = time_ring_step(
            1, heads, kv_heads, seq_len, d, Kind::CyclicInPlace, 4U, RingLayout::Zigzag, 7U, Fwd::Cyclic,
            RingShiftTransport::Direct);
        const auto step = [](const StepTimes& t) { return t.forward_ms + t.backward_ms; };
        const bool zz = step(zigzag) <= step(contig);
        const auto& base = zz ? zigzag : contig;
        std::cout << "  heads=" << heads << " kv_heads=" << kv_heads << " rows/chip=" << rows << " d=" << d << ":\n"
                  << "    two-pass contiguous forward " << contig.forward_ms << " backward " << contig.backward_ms
                  << " step " << step(contig) << "\n"
                  << "    two-pass zigzag     forward " << zigzag.forward_ms << " backward " << zigzag.backward_ms
                  << " step " << step(zigzag) << "\n"
                  << "    cyclic zigzag       forward " << cyc.forward_ms << " backward " << cyc.backward_ms
                  << " step " << step(cyc) << "\n"
                  << "    speed-up vs two-pass " << (zz ? "zigzag" : "contiguous") << ": forward "
                  << base.forward_ms / cyc.forward_ms << "x backward " << base.backward_ms / cyc.backward_ms
                  << "x step " << step(base) / step(cyc) << "x\n";
    }
}

// Forward and backward of a ring step for both backward kinds, zigzag,
// direct shifts, median of five. TTML_LOUDBOX_STEP_SHAPES="heads:kv:rows:d:Bt,..."
// replaces the table.
TEST_F(LoudboxRingSDPATest, DISABLED_CompareStepTimes) {
    const uint32_t cp_size = ttml::autograd::ctx().get_parallelism_context().get_cp_size();
    std::cout << "ring forward + backward on " << cp_size << " chips, zigzag, direct shifts, median of five (ms)\n";
    using Kind = ttml::ops::distributed::RingBackwardKind;
    std::vector<std::array<size_t, 5>> table = {
        {4, 4, 4096, 64, 4},
        {20, 10, 5632, 64, 4},
        {32, 8, 5632, 128, 4},
    };
    if (const char* spec = std::getenv("TTML_LOUDBOX_STEP_SHAPES")) {
        table.clear();
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::array<size_t, 5> cfg{};
            std::stringstream is(item);
            std::string field;
            for (size_t k = 0; k < 5 && std::getline(is, field, ':'); ++k) {
                cfg[k] = std::stoul(field);
            }
            table.push_back(cfg);
        }
    }
    for (const auto& cfg : table) {
        const size_t heads = cfg[0], kv_heads = cfg[1], rows = cfg[2], d = cfg[3];
        const auto Bt = static_cast<uint32_t>(cfg[4]);
        const size_t seq_len = rows * cp_size;
        using Fwd = ttml::ops::distributed::RingForwardKind;
        const auto tp = time_ring_step(1, heads, kv_heads, seq_len, d, Kind::TwoPass, 1U, RingLayout::Zigzag, 5U);
        const auto cy = time_ring_step(1, heads, kv_heads, seq_len, d, Kind::CyclicInPlace, Bt, RingLayout::Zigzag, 5U);
        const auto cy_ttnn = time_ring_step(
            1, heads, kv_heads, seq_len, d, Kind::CyclicInPlace, Bt, RingLayout::Zigzag, 5U, Fwd::Ttnn);
        const auto cy_cyc = time_ring_step(
            1, heads, kv_heads, seq_len, d, Kind::CyclicInPlace, Bt, RingLayout::Zigzag, 5U, Fwd::Cyclic);
        // Useful FLOPs of the causal backward over the whole sequence: five
        // matmuls of 2 S^2 d, halved by the triangle, per head. The forward
        // is two such matmuls.
        const double S = static_cast<double>(seq_len);
        const double bwd_flop = 5.0 * S * S * static_cast<double>(d) * static_cast<double>(heads);
        const double peak = 594.0 * cp_size;  // TFLOP/s, LoFi, 110 cores per chip
        const auto tflops = [&](double flop, double ms) { return flop / (ms * 1e-3) / 1e12; };
        const auto pct = [](double a, double b) { return (b / a - 1.0) * 100.0; };
        std::cout << "  heads=" << heads << " kv_heads=" << kv_heads << " rows/chip=" << rows << " d=" << d
                  << " Bt=" << Bt << ":\n"
                  << "    two-pass  forward " << tp.forward_ms << " backward " << tp.backward_ms << " step "
                  << tp.forward_ms + tp.backward_ms << " | backward " << tflops(bwd_flop, tp.backward_ms)
                  << " TFLOP/s, " << 100.0 * tflops(bwd_flop, tp.backward_ms) / peak << "% of the ring's LoFi peak\n"
                  << "    cyclic    forward " << cy.forward_ms << " backward " << cy.backward_ms << " step "
                  << cy.forward_ms + cy.backward_ms << " | backward " << tflops(bwd_flop, cy.backward_ms)
                  << " TFLOP/s, " << 100.0 * tflops(bwd_flop, cy.backward_ms) / peak << "% of the ring's LoFi peak\n"
                  << "    change: backward " << pct(tp.backward_ms, cy.backward_ms) << "%, step "
                  << pct(tp.forward_ms + tp.backward_ms, cy.forward_ms + cy.backward_ms) << "%\n"
                  << "    cyclic + ttnn forward: forward " << cy_ttnn.forward_ms << " backward " << cy_ttnn.backward_ms
                  << " step " << cy_ttnn.forward_ms + cy_ttnn.backward_ms << " | step "
                  << pct(tp.forward_ms + tp.backward_ms, cy_ttnn.forward_ms + cy_ttnn.backward_ms)
                  << "% vs two-pass, forward " << pct(tp.forward_ms, cy_ttnn.forward_ms) << "%\n"
                  << "    cyclic + cyclic forward: forward " << cy_cyc.forward_ms << " backward " << cy_cyc.backward_ms
                  << " step " << cy_cyc.forward_ms + cy_cyc.backward_ms << " | step "
                  << pct(tp.forward_ms + tp.backward_ms, cy_cyc.forward_ms + cy_cyc.backward_ms)
                  << "% vs two-pass, forward " << pct(tp.forward_ms, cy_cyc.forward_ms) << "% vs two-pass, "
                  << pct(cy_ttnn.forward_ms, cy_cyc.forward_ms) << "% vs ttnn\n";
    }
}

// One whole backward per implementation and transport, with the phase
// profile in ring_attention_sdpa switched on (TTML_RING_PROFILE), so the
// whole-backward total above can be split into kernel, accumulate, shift
// and setup instead of inferred from components timed on their own. The
// profile synchronises after each phase, so its total runs a little over the
// unprofiled one. Rows per chip from TTML_LOUDBOX_PROFILE_ROWS (default 4096),
// heads=4, d=64, Bt=4.
TEST_F(LoudboxRingSDPATest, DISABLED_ProfileOneBackward) {
    setenv("TTML_RING_PROFILE", "1", /* overwrite */ 1);
    const uint32_t cp_size = ttml::autograd::ctx().get_parallelism_context().get_cp_size();
    size_t rows_per_chip = 4096;
    if (const char* env = std::getenv("TTML_LOUDBOX_PROFILE_ROWS"); env != nullptr && *env != '\0') {
        rows_per_chip = std::strtoul(env, nullptr, 10);
    }
    using Kind = ttml::ops::distributed::RingBackwardKind;
    // TTML_LOUDBOX_PROFILE_LAYOUT=zigzag profiles the zigzag layout instead,
    // and TTML_LOUDBOX_PROFILE_BT the block height (default 4).
    RingLayout layout = RingLayout::Contiguous;
    uint32_t Bt = 4U;
    if (const char* env = std::getenv("TTML_LOUDBOX_PROFILE_BT"); env != nullptr && *env != '\0') {
        Bt = static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
    }
    if (const char* env = std::getenv("TTML_LOUDBOX_PROFILE_LAYOUT"); env != nullptr && std::string(env) == "zigzag") {
        layout = RingLayout::Zigzag;
    }
    // TTML_LOUDBOX_PROFILE_FORWARD=ttnn profiles with the ttnn step forward; the heads from
    // TTML_LOUDBOX_PROFILE_HEADS (default 4); TTML_LOUDBOX_PROFILE_DIRECT_ONLY=1 skips the FIFO rows.
    using Fwd = ttml::ops::distributed::RingForwardKind;
    Fwd forward = Fwd::TwoPass;
    if (const char* env = std::getenv("TTML_LOUDBOX_PROFILE_FORWARD"); env != nullptr && std::string(env) == "ttnn") {
        forward = Fwd::Ttnn;
    } else if (env != nullptr && std::string(env) == "cyclic") {
        forward = Fwd::Cyclic;
    }
    size_t heads = 4;
    if (const char* env = std::getenv("TTML_LOUDBOX_PROFILE_HEADS"); env != nullptr && *env != '\0') {
        heads = std::strtoul(env, nullptr, 10);
    }
    const bool direct_only = std::getenv("TTML_LOUDBOX_PROFILE_DIRECT_ONLY") != nullptr;
    for (const auto transport : {RingShiftTransport::Fifo, RingShiftTransport::Direct}) {
        if (direct_only && transport == RingShiftTransport::Fifo) {
            continue;
        }
        for (const auto kind : {Kind::TwoPass, Kind::Cyclic, Kind::CyclicInPlace}) {
            std::cout << "== " << (layout == RingLayout::Zigzag ? "zigzag, " : "") << rows_per_chip << " rows/chip, Bt=" << Bt << ", "
                      << (kind == Kind::TwoPass ? "two-pass" : kind == Kind::Cyclic ? "cyclic" : "cyclic in-place")
                      << ", " << (transport == RingShiftTransport::Fifo ? "fifo" : "direct")
                      << " shifts (second profile is the timed one)\n";
            const double seconds = time_ring_backward(
                1, heads, rows_per_chip * cp_size, 64, kind, Bt, transport, /* samples */ 1U, layout, 0, forward);
            std::cout << "   unprofiled-style total (with profile syncs): " << seconds * 1e3 << " ms\n";
        }
    }
    unsetenv("TTML_RING_PROFILE");
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

    for (const auto& cfg :
         std::vector<std::array<size_t, 4>>{{1, 4, 128, 64}, {1, 4, 512, 64}, {1, 4, 2048, 64}, {1, 4, 4096, 64}}) {
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
        const double direct_shift = median([&]() {
            (void)ttnn_fixed::distributed::ring_shift(
                k, cp_axis, RingShiftDirection::Forward, RingShiftTransport::Direct);
        });
        std::cout << "  ... the same, direct transport (bf16)       " << direct_shift << "\n";
        const double direct_shift32 = median([&]() {
            (void)ttnn_fixed::distributed::ring_shift(
                acc, cp_axis, RingShiftDirection::Forward, RingShiftTransport::Direct);
        });
        std::cout << "  ... the same, direct transport (fp32)       " << direct_shift32 << "\n";
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
        for (const uint32_t Bt : {1u, 2u, 4u}) {
            if (rows_per_chip % (2u * Bt * 32u) != 0u) {
                continue;
            }
            const uint32_t C_here = static_cast<uint32_t>(rows_per_chip) / (2u * Bt * 32u);
            if (C_here > 64u) {
                continue;  // no rectangle of that area in the grid
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
                  << (3 * zero + 3 * add + 2 * shift + 2 * shift32) * cp_size / 1e3 << " ms; with direct shifts ~ "
                  << 3 * zero + 3 * add + 2 * direct_shift + 2 * direct_shift32 << " us/step\n";
    }
}

// The per-step op of each implementation alone, swept in size until the
// kernel's own time is visible above dispatch. The breakdown above found the
// step ops flat between 128 and 512 rows per chip, which means neither kernel
// was doing enough work to show; this finds where that stops being true, and
// what each costs per unit of work once it does. Step 0 is the diagonal, so
// every chip runs the causal schedule. A fixed dispatch floor plus a term
// linear in the work is what to expect; the floor is the number to beat.
TEST_F(LoudboxRingSDPATest, DISABLED_SweepTheStepOps) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    auto& rng = autograd::ctx().get_generator();

    const auto median_us = [&](const std::function<void()>& f) {
        f();
        std::vector<double> samples;
        for (uint32_t k = 0; k < 5; ++k) {
            tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
            const auto start = std::chrono::steady_clock::now();
            f();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt, {});
            samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2] * 1e6;
    };

    std::cout << "one diagonal step on " << cp_size << " chips (us, median of 5); work = heads * rows^2 * d\n";
    for (const auto& cfg : std::vector<std::array<size_t, 3>>{
             // heads, rows per chip, head dim
             {4, 128, 64},
             {4, 256, 64},
             {4, 512, 64},
             {4, 1024, 64},
             {4, 2048, 64},
             {8, 1024, 64},
             {16, 1024, 64},
             {4, 1024, 128},
             // Larger chunks, where the kernel is most of the step. At 4096
             // rows Bt = 1 wants 64 cores per slice (one 8x8 group, four
             // slices looped); at 8192 rows Bt = 1 wants 128 cores, which no
             // rectangle of the 11x10 grid has, and the op says so.
             {4, 4096, 64},
             {8, 4096, 64},
             {4, 8192, 64},
             {1, 8192, 64},
             {1, 16384, 64},
         }) {
        const size_t heads = cfg[0], rows = cfg[1], d = cfg[2];
        const std::array<std::size_t, 4> shape{1UL, heads, rows * cp_size, d};
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
            ttnn::Shape{1U, heads, rows, 32U}, 4.0F, ttnn::DataType::FLOAT32, ttnn::Layout::TILE, std::ref(*device));
        const ttnn::Tensor D = ttnn::full(
            ttnn::Shape{1U, heads, rows, 32U}, 0.5F, ttnn::DataType::FLOAT32, ttnn::Layout::TILE, std::ref(*device));
        ttnn::Tensor step_bf16 = ttnn::zeros_like(q);
        ttnn::Tensor acc_q = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
        ttnn::Tensor acc_k = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
        ttnn::Tensor acc_v = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);

        const double two_pass = median_us([&]() {
            (void)ttml::metal::ring_sdpa_bw(
                dO, O, q, k, v, lse, cp_size, cp_axis, 0, ttml::metal::AttentionMaskType::Causal,
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward, step_bf16, step_bf16, step_bf16);
        });
        std::ostringstream line;
        line << "  heads=" << heads << " rows/chip=" << rows << " d=" << d << ": two-pass " << two_pass;
        // Largest block height the chunk allows, up to 4, and C from it.
        for (const uint32_t Bt : {1u, 2u, 4u}) {
            if (rows % (2u * Bt * 32u) != 0u) {
                continue;
            }
            const uint32_t C = static_cast<uint32_t>(rows) / (2u * Bt * 32u);
            // The op deals slices round-robin to as many C-core groups as
            // fit and runs the rest in turn, so heads x C beyond the grid is
            // no longer a refusal; it shows up as time instead. Print the
            // groups so the loop's depth is visible next to the number. What
            // is still refused is a C with no rectangle of that area in the
            // grid -- 128 or 256 cores -- which the try/catch reports.
            const auto grid = device->compute_with_storage_grid_size();
            const size_t capacity = static_cast<size_t>(grid.x) * grid.y / C;
            const size_t groups = std::max<size_t>(1, std::min<size_t>(heads, capacity));
            double cyclic = 0.0;
            try {
                cyclic = median_us([&]() {
                    (void)ttml::metal::ring_cyclic_sdpa_bw(
                        q, k, v, dO, lse, D, cp_size, cp_axis, 0, ttml::metal::AttentionMaskType::Causal,
                        ttml::metal::RingCyclicDirection::Backward, Bt, false, /* accumulate */ true, acc_q, acc_k,
                        acc_v);
                });
            } catch (const std::exception&) {
                line << " | cyclic Bt=" << Bt << " (C=" << C << ") no rectangle of area " << C << " fits";
                continue;
            }
            line << " | cyclic Bt=" << Bt << " (C=" << C << ", " << groups << " groups x "
                 << (heads + groups - 1) / groups << " slices) " << cyclic;
        }
        std::cout << line.str() << "\n";
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
