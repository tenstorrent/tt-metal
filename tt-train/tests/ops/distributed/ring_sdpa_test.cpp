// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Test for ring_attention_sdpa distributed operation.
 *
 * Ring attention computes full-sequence attention when the sequence is
 * sharded across CP (context parallel) devices.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <core/xtensor_utils.hpp>
#include <functional>
#include <optional>
#include <tt-metalium/mesh/multihost/distributed_context.hpp>

#include "autograd/auto_context.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/ring_attention_sdpa.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ring_sdpa_test_utils.hpp"
#include "test_utils/mesh_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/create_socket.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

class GalaxyRingSDPATest : public ::testing::Test {
public:
    static inline const tt::tt_metal::distributed::MeshShape kMeshShape{8, 4};

    static void SetUpTestSuite() {
        if (ttml::test_utils::system_supports_mesh(kMeshShape)) {
            if (!ttml::autograd::ctx().is_distributed_context_initialized()) {
                ttml::autograd::ctx().initialize_distributed_context(0, nullptr);
            }
            ttml::ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(kMeshShape.mesh_size()));
            ttml::autograd::ctx().open_device(kMeshShape);
            ttml::autograd::ctx().set_seed(42);
            ttml::autograd::ctx().initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);

            // Configure parallelism context for CP (CP axis will be 0)
            ttml::autograd::ctx().initialize_parallelism_context(
                {.enable_ddp = false, .enable_tp = true, .cp = ttml::autograd::ContextParallelConfig{}});
        }
    }
    static void TearDownTestSuite() {
        if (ttml::test_utils::system_supports_mesh(kMeshShape)) {
            ttml::autograd::ctx().close_device();
        }
    }

    void SetUp() override {
        SKIP_UNLESS_MESH_SUPPORTED(kMeshShape);
    }
};

// CP2 ring attention on a 1x2 mesh.
class Mesh1x2RingSDPATest : public ttml::test_utils::Mesh1x2SuiteFixture {
protected:
    static void SetUpTestSuite() {
        Mesh1x2SuiteFixture::SetUpTestSuite();
        if (ttml::test_utils::system_supports_mesh(mesh_shape())) {
            if (!ttml::autograd::ctx().is_distributed_context_initialized()) {
                ttml::autograd::ctx().initialize_distributed_context(0, nullptr);
            }
            ttml::autograd::ctx().set_seed(42);
            ttml::autograd::ctx().initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);
            // A 1x2 mesh is a line, so CP takes its only non-trivial axis: cp_axis = 1, ring size 2.
            ttml::autograd::ctx().initialize_parallelism_context(
                {.enable_ddp = false, .enable_tp = false, .cp = ttml::autograd::ContextParallelConfig{}});
        }
    }
};

struct RingAttentionRun {
    xt::xarray<float> query, key, value;
    ttml::test_utils::ring_sdpa::SdpaForward ref;
    ttml::autograd::TensorPtr query_tensor, key_tensor, value_tensor, output;
};

static void ExpectRingAttentionBackward(
    RingAttentionRun& run,
    const std::function<ttnn::Tensor(const xt::xarray<float>&)>& to_device,
    const uint32_t cp_axis,
    const ttml::core::distributed::CpSharding sharding) {
    using ttml::test_utils::ring_sdpa::gather_cp_sequence;
    auto& rng = ttml::autograd::ctx().get_generator();
    const xt::xarray<float> grad_output_xt =
        ttml::test_utils::make_uniform_xarray<float>(run.query.shape(), 0.0F, 2.0F, rng());
    const auto ref_grads = ttml::test_utils::ring_sdpa::sdpa_backward(
        run.query, run.key, run.value, run.ref.output, run.ref.probabilities, grad_output_xt);

    run.output->set_grad(to_device(grad_output_xt));
    run.output->backward();
    const auto gathered_dQ = gather_cp_sequence(run.query_tensor->get_grad(), cp_axis, sharding);
    const auto gathered_dK = gather_cp_sequence(run.key_tensor->get_grad(), cp_axis, sharding);
    const auto gathered_dV = gather_cp_sequence(run.value_tensor->get_grad(), cp_axis, sharding);

    // One grade for all three gradients: rtol matches the single-device sdpa_bw
    // suite; atol has headroom for what that suite does not exercise — uniform(0,2)
    // inputs make u = rowsum(dO*O) large against the (dP - u) cancellation, so the
    // bf16 rounding of the saved O and of each ring step's kernel output lands
    // mostly in dK/dQ. Numerical simulation of the pipeline puts a correct
    // implementation at <= 0.7x of this grade across 32 seeds, while softmax-backward
    // math errors (wrong u, double scale) overshoot it 25-1000x, so detection
    // power is intact.
    const float bw_rtol = 3e-2F;
    const float bw_atol = 5e-2F;
    // dK is the only gradient whose accumulation length is the query sequence
    // (dV rows are softmax-convex combinations of dO; dQ contracts over head_dim),
    // so its magnitude and its TF32-matmul noise grow with S while dQ/dV stay O(1).
    // Grade it relative to its own largest true value: 2% of amax(|ref dK|), with
    // the fixed floor for tiny-magnitude cases.
    const float dk_atol = std::max(bw_atol, 2e-2F * xt::amax(xt::abs(ref_grads.dk))());
    const auto report = [](const xt::xarray<float>& ref, const xt::xarray<float>& got) {
        const float max_abs = xt::amax(xt::abs(got - ref))();
        const float max_rel = xt::amax(xt::abs(got - ref) / (xt::abs(ref) + 1e-6F))();
        const float ref_amax = xt::amax(xt::abs(ref))();
        return "max_abs_diff=" + std::to_string(max_abs) + " max_rel_diff=" + std::to_string(max_rel) +
               " ref_amax=" + std::to_string(ref_amax);
    };
    EXPECT_TRUE(xt::allclose(ref_grads.dq, gathered_dQ, bw_rtol, bw_atol))
        << "Ring attention dQ gradient does not match reference: " << report(ref_grads.dq, gathered_dQ);
    EXPECT_TRUE(xt::allclose(ref_grads.dk, gathered_dK, bw_rtol, dk_atol))
        << "Ring attention dK gradient does not match reference: " << report(ref_grads.dk, gathered_dK);
    EXPECT_TRUE(xt::allclose(ref_grads.dv, gathered_dV, bw_rtol, bw_atol))
        << "Ring attention dV gradient does not match reference: " << report(ref_grads.dv, gathered_dV);
}

// kv_heads = 0 means as many K/V heads as query heads.
static void TestRingAttention(
    const size_t batch,
    const size_t num_heads,
    const size_t seq_len,
    const size_t head_dim,
    const bool test_backward = false,
    const float rtol = 1e-2F,
    const float atol = 5e-1F,
    const ttml::core::distributed::CpSharding sharding = ttml::core::distributed::CpSharding::Contiguous,
    const std::optional<bool> reuse_forward_kv = std::nullopt,
    const size_t kv_heads = 0U) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    auto& rng = autograd::ctx().get_generator();

    const std::array<std::size_t, 4> q_shape{batch, num_heads, seq_len, head_dim};
    const std::array<std::size_t, 4> kv_shape{batch, kv_heads == 0U ? num_heads : kv_heads, seq_len, head_dim};
    RingAttentionRun run;
    run.query = ttml::test_utils::make_uniform_xarray<float>(q_shape, 0.0F, 2.0F, rng());
    run.key = ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, 2.0F, rng());
    run.value = ttml::test_utils::make_uniform_xarray<float>(kv_shape, 0.0F, 2.0F, rng());
    run.ref = ttml::test_utils::ring_sdpa::sdpa_forward(run.query, run.key, run.value, /*causal=*/true);

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const std::function<ttnn::Tensor(const xt::xarray<float>&)> to_device = [&](const xt::xarray<float>& x) {
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            ttml::test_utils::ring_sdpa::permute_cp_order(x, sharding, cp_size, /*to_device=*/true),
            device,
            ttnn::Layout::TILE,
            mapper.get());
    };
    run.query_tensor = autograd::create_tensor(to_device(run.query), /* requires_grad */ true);
    run.key_tensor = autograd::create_tensor(to_device(run.key), /* requires_grad */ true);
    run.value_tensor = autograd::create_tensor(to_device(run.value), /* requires_grad */ true);

    // No explicit mask tensor: the op rejects one in CP mode; causal masking comes from
    // mask_type and is generated on device.
    run.output = ops::distributed::ring_attention_sdpa(
        run.query_tensor,
        run.key_tensor,
        run.value_tensor,
        /*mask=*/std::nullopt,
        ttml::metal::AttentionMaskType::Causal,
        reuse_forward_kv,
        sharding);

    const auto gathered_output =
        ttml::test_utils::ring_sdpa::gather_cp_sequence(run.output->get_value(), cp_axis, sharding);
    EXPECT_TRUE(xt::allclose(run.ref.output, gathered_output, rtol, atol))
        << "Ring attention output does not match reference SDPA output";

    if (test_backward) {
        ExpectRingAttentionBackward(run, to_device, cp_axis, sharding);
    }
}

// Ring attention with causal mask (forward only)
TEST_F(GalaxyRingSDPATest, WithCausalMask) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/false);
}

// Ring attention with causal mask and backward pass
TEST_F(GalaxyRingSDPATest, WithCausalMaskBackward) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/true);
}

// Larger batch size test
TEST_F(GalaxyRingSDPATest, LargerBatch) {
    TestRingAttention(
        /*batch=*/4,
        /*num_heads=*/8,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/false);
}

// Test with larger sequence
TEST_F(GalaxyRingSDPATest, LargerSequence) {
    TestRingAttention(
        /*batch=*/2,
        /*num_heads=*/4,
        /*seq_len=*/256,  // 64 per device
        /*head_dim=*/64,
        /*test_backward=*/false);
}

// Full test: larger batch with causal mask and backward
TEST_F(GalaxyRingSDPATest, LargerBatchCausalMaskBackward) {
    TestRingAttention(
        /*batch=*/4,
        /*num_heads=*/8,
        /*seq_len=*/1024,
        /*head_dim=*/64,
        /*test_backward=*/true);
}

// Full test: larger batch with causal mask and backward
TEST_F(GalaxyRingSDPATest, CacheTestBackward) {
    TestRingAttention(
        /*batch=*/4,
        /*num_heads=*/8,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/true);
}

// Full test: larger sequence with causal mask and backward
TEST_F(GalaxyRingSDPATest, LargerSequenceCausalMaskBackward) {
    TestRingAttention(
        /*batch=*/2,
        /*num_heads=*/4,
        /*seq_len=*/256,
        /*head_dim=*/64,
        /*test_backward=*/true);
}

// CP2: step 0 is the diagonal block on both devices; at step 1 device 0 skips (its K/V come from later in the
// sequence) while device 1 attends to device 0's chunk without a mask.
TEST_F(Mesh1x2RingSDPATest, WithCausalMask) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/false);
}

TEST_F(Mesh1x2RingSDPATest, WithCausalMaskBackward) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/true);
}

// 256 rows per device, so each step's SDPA spans several tile rows, heads and batch entries.
TEST_F(Mesh1x2RingSDPATest, LargerBatchSequenceBackward) {
    TestRingAttention(
        /*batch=*/2,
        /*num_heads=*/8,
        /*seq_len=*/512,
        /*head_dim=*/64,
        /*test_backward=*/true);
}

// Backward with the forward's K/V handles must give the very same bits as re-shifting K/V:
// a ring shift only moves data, so both paths feed identical tensors to the kernels.
static void ExpectReuseMatchesReshiftBitExact(
    const size_t seq_len, const ttml::core::distributed::CpSharding sharding) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t cp_axis = autograd::ctx().get_parallelism_context().get_cp_axis().value();
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);

    auto& rng = autograd::ctx().get_generator();
    const std::array<std::size_t, 4> shape{2, 4, seq_len, 64};
    std::array<xt::xarray<float>, 4> inputs;  // Q, K, V, dO
    for (auto& input : inputs) {
        input = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, rng());
    }
    const auto to_device = [&](const xt::xarray<float>& data) {
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, device, ttnn::Layout::TILE, mapper.get());
    };

    // Per-device dQ, dK, dV
    const auto run = [&](const bool reuse_forward_kv) {
        auto query = autograd::create_tensor(to_device(inputs[0]), /* requires_grad */ true);
        auto key = autograd::create_tensor(to_device(inputs[1]), /* requires_grad */ true);
        auto value = autograd::create_tensor(to_device(inputs[2]), /* requires_grad */ true);
        auto output = ops::distributed::ring_attention_sdpa(
            query,
            key,
            value,
            /*mask=*/std::nullopt,
            ttml::metal::AttentionMaskType::Causal,
            reuse_forward_kv,
            sharding);
        output->set_grad(to_device(inputs[3]));
        output->backward();
        return std::array{
            core::to_xtensor<float>(query->get_grad(), core::IdentityComposer{}),
            core::to_xtensor<float>(key->get_grad(), core::IdentityComposer{}),
            core::to_xtensor<float>(value->get_grad(), core::IdentityComposer{})};
    };

    const auto reused = run(/*reuse_forward_kv=*/true);
    const auto reshifted = run(/*reuse_forward_kv=*/false);
    EXPECT_TRUE(reused[0] == reshifted[0]) << "dQ differs";
    EXPECT_TRUE(reused[1] == reshifted[1]) << "dK differs";
    EXPECT_TRUE(reused[2] == reshifted[2]) << "dV differs";
}

TEST_F(GalaxyRingSDPATest, ReuseMatchesReshiftBitExact) {
    ExpectReuseMatchesReshiftBitExact(/*seq_len=*/256, ttml::core::distributed::CpSharding::Contiguous);
}

// CP8 zig-zag fw+bw matches non-ring causal SDPA on inputs permuted into zig-zag order.
TEST_F(GalaxyRingSDPATest, ZigZagCausalMaskBackward) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/1024,
        /*head_dim=*/64,
        /*test_backward=*/true,
        /*rtol=*/1e-2F,
        /*atol=*/5e-1F,
        ttml::core::distributed::CpSharding::ZigZag);
}

// Same with batch 2, 8 heads and 256 rows per device.
TEST_F(GalaxyRingSDPATest, ZigZagLargerBatchSequenceBackward) {
    TestRingAttention(
        /*batch=*/2,
        /*num_heads=*/8,
        /*seq_len=*/2048,
        /*head_dim=*/64,
        /*test_backward=*/true,
        /*rtol=*/1e-2F,
        /*atol=*/5e-1F,
        ttml::core::distributed::CpSharding::ZigZag);
}

// Zig-zag backward gives bit-identical grads with kept forward K/V and with re-shifted K/V.
TEST_F(GalaxyRingSDPATest, ZigZagReuseMatchesReshiftBitExact) {
    ExpectReuseMatchesReshiftBitExact(/*seq_len=*/1024, ttml::core::distributed::CpSharding::ZigZag);
}

// CP2 zig-zag fw+bw matches non-ring causal SDPA on inputs permuted into zig-zag order.
TEST_F(Mesh1x2RingSDPATest, ZigZagBackward) {
    TestRingAttention(
        /*batch=*/2,
        /*num_heads=*/4,
        /*seq_len=*/512,
        /*head_dim=*/64,
        /*test_backward=*/true,
        /*rtol=*/1e-2F,
        /*atol=*/5e-1F,
        ttml::core::distributed::CpSharding::ZigZag);
}

// CP2 zig-zag with grouped-query attention: 8 query heads read 2 K/V heads.
TEST_F(Mesh1x2RingSDPATest, ZigZagGroupedQueryBackward) {
    TestRingAttention(
        /*batch=*/2,
        /*num_heads=*/8,
        /*seq_len=*/512,
        /*head_dim=*/64,
        /*test_backward=*/true,
        /*rtol=*/1e-2F,
        /*atol=*/5e-1F,
        ttml::core::distributed::CpSharding::ZigZag,
        /*reuse_forward_kv=*/std::nullopt,
        /*kv_heads=*/2U);
}

TEST_F(Mesh1x2RingSDPATest, ReuseMatchesReshiftBitExact) {
    ExpectReuseMatchesReshiftBitExact(/*seq_len=*/256, ttml::core::distributed::CpSharding::Contiguous);
}

TEST_F(Mesh1x2RingSDPATest, ZigZagReuseMatchesReshiftBitExact) {
    ExpectReuseMatchesReshiftBitExact(/*seq_len=*/256, ttml::core::distributed::CpSharding::ZigZag);
}

// Zig-zag rejects a non-causal mask and a local sequence that does not split into two tile-aligned halves.
TEST_F(Mesh1x2RingSDPATest, ZigZagRejectsUnsupportedCalls) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const uint32_t cp_axis = autograd::ctx().get_parallelism_context().get_cp_axis().value();
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto make_input = [&](size_t seq_len) {
        const std::array<size_t, 4> shape{1, 4, seq_len, 64};
        const auto data = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 2.0F, 7U);
        return autograd::create_tensor(
            core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, device, ttnn::Layout::TILE, mapper.get()));
    };
    const auto call = [](const autograd::TensorPtr& x, ttml::metal::AttentionMaskType mask_type) {
        ops::distributed::ring_attention_sdpa(
            x,
            x,
            x,
            /*mask=*/std::nullopt,
            mask_type,
            /*reuse_forward_kv=*/std::nullopt,
            core::distributed::CpSharding::ZigZag);
    };
    const auto x = make_input(256U);
    const auto x_short = make_input(192U);
    EXPECT_THAT(
        [&] { call(x, ttml::metal::AttentionMaskType::None); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Zig-zag CP sharding balances causal attention")));
    EXPECT_THAT(
        [&] { call(x_short, ttml::metal::AttentionMaskType::Causal); },
        ThrowsMessage<std::runtime_error>(HasSubstr("must be a multiple of 64")));
}

// Ring attention on the system mesh. From CP4 on (CP sizes are even), a zig-zag device switches chunks between ring
// steps and the backward re-shifts K/V more than once, which CP2 never does.
class SystemMeshRingSDPATest : public ttml::test_utils::SystemMeshSuiteFixture {
protected:
    static constexpr uint32_t kMinCpSize = 4U;

    static void SetUpTestSuite() {
        SystemMeshSuiteFixture::SetUpTestSuite();
        if (ttml::test_utils::system_mesh_size() > 1U) {
            if (!ttml::autograd::ctx().is_distributed_context_initialized()) {
                ttml::autograd::ctx().initialize_distributed_context(0, nullptr);
            }
            ttml::autograd::ctx().set_seed(42);
            ttml::autograd::ctx().initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);
            ttml::test_utils::ring_sdpa::initialize_cp_parallelism_context();
        }
    }

    void SetUp() override {
        SystemMeshSuiteFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        const uint32_t cp_size = ttml::autograd::ctx().get_parallelism_context().get_cp_size();
        if (cp_size < kMinCpSize) {
            GTEST_SKIP() << fmt::format(
                "Skipping test: needs a CP axis of at least {} devices, got {} "
                "(e.g. TT_MESH_GRAPH_DESC_PATH=tt-train/configs/mgd/bh_1_4_line_line.textproto)",
                kMinCpSize,
                cp_size);
        }
    }

    static size_t seq_len_with_local(size_t local_seq_len) {
        return local_seq_len * ttml::autograd::ctx().get_parallelism_context().get_cp_size();
    }
};

TEST_F(SystemMeshRingSDPATest, ZigZagBackward) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        seq_len_with_local(128U),
        /*head_dim=*/64,
        /*test_backward=*/true,
        /*rtol=*/1e-2F,
        /*atol=*/5e-1F,
        ttml::core::distributed::CpSharding::ZigZag);
}

TEST_F(SystemMeshRingSDPATest, ZigZagBackwardWithForwardKV) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        seq_len_with_local(128U),
        /*head_dim=*/64,
        /*test_backward=*/true,
        /*rtol=*/1e-2F,
        /*atol=*/5e-1F,
        ttml::core::distributed::CpSharding::ZigZag,
        /*reuse_forward_kv=*/true);
}

TEST_F(SystemMeshRingSDPATest, ContiguousBackward) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        seq_len_with_local(128U),
        /*head_dim=*/64,
        /*test_backward=*/true);
}

TEST_F(SystemMeshRingSDPATest, ZigZagReuseMatchesReshiftBitExact) {
    ExpectReuseMatchesReshiftBitExact(seq_len_with_local(128U), ttml::core::distributed::CpSharding::ZigZag);
}
