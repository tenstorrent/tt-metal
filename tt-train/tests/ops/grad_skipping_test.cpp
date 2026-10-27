// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <vector>
#include <xtensor/core/xmath.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/views/xview.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/moe_group/moe_group.hpp"
#include "moe_test_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/layernorm_op.hpp"
#include "ops/mla_kv_assemble_op.hpp"
#include "ops/moe_ffn_swiglu_op.hpp"
#include "ops/moe_group_op.hpp"
#include "ops/moe_ungroup_op.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/polynorm_op.hpp"
#include "ops/rmsnorm_op.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ops/swiglu_op.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

namespace {

using ttml::autograd::TensorPtr;

// One differentiable input of the op under test: its host value and the layout the op expects it in.
struct HostInput {
    xt::xarray<float> value;
    ttnn::Layout layout = ttnn::Layout::TILE;
};

// An op under test. Built inside the test, so it may create device tensors.
struct OpUnderTest {
    std::vector<HostInput> inputs;
    // Runs the op on device copies of `inputs` (same order) and returns the tensor backward starts from.
    std::function<TensorPtr(const std::vector<TensorPtr>&)> forward;
};

// Device-op name -> launch count.
using OpCounts = std::map<std::string, int>;

struct GradSkipCase {
    std::string name;
    std::function<OpUnderTest()> make;
    std::vector<size_t> frozen;  // indices into OpUnderTest::inputs
    // Device ops that tt-train itself launches for the frozen inputs' grads; the frozen run must launch fewer of each.
    std::vector<std::string> must_skip;
    bool blackhole_only = false;
};

// gtest prints a failing row's parameter through this; without it the row prints as raw bytes.
void PrintTo(const GradSkipCase& c, std::ostream* os) {
    *os << c.name;
}

// Device-op names as the graph capture records them.
constexpr const char* kBinary = "BinaryNgDeviceOperation";
constexpr const char* kUnary = "UnaryDeviceOperation";
constexpr const char* kMatmul = "MatmulDeviceOperation";
constexpr const char* kVariableMatmul = "VariableMatmulDeviceOperation";
constexpr const char* kMorehSum = "MorehSumOperation";
constexpr const char* kMorehMean = "MorehMeanOperation";
constexpr const char* kTranspose = "TransposeDeviceOperation";
constexpr const char* kReduce = "ReduceDeviceOperation";
constexpr const char* kSlice = "SliceDeviceOperation";
constexpr const char* kConcat = "ConcatDeviceOperation";
constexpr const char* kSwigluElemwiseBw = "SwigluElemwiseBwDeviceOperation";
constexpr const char* kUnaryBackward = "UnaryBackwardDeviceOperation";
constexpr const char* kNlpConcatHeads = "NLPConcatHeadsDeviceOperation";
constexpr const char* kMoeUngroup = "MoeUngroupDeviceOperation";
constexpr const char* kTernary = "TernaryDeviceOperation";
constexpr const char* kSdpaBwQ = "SDPABackwardQDeviceOperation";
constexpr const char* kSdpaBwKV = "SDPABackwardKVDeviceOperation";

xt::xarray<float> uniform(const std::vector<uint32_t>& shape, float lo, float hi, uint32_t seed) {
    return ttml::test_utils::make_uniform_xarray<float>(shape, lo, hi, seed);
}

// The grad of every input that received one, and the ops backward launched.
struct BackwardRun {
    std::vector<std::optional<xt::xarray<float>>> grads;
    std::vector<std::string> calls;
};

BackwardRun run_backward(const OpUnderTest& op, const std::vector<bool>& requires_grad) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<TensorPtr> inputs;
    for (size_t i = 0; i < op.inputs.size(); ++i) {
        inputs.push_back(ttml::autograd::create_tensor(
            ttml::core::from_xtensor(op.inputs[i].value, device, op.inputs[i].layout), requires_grad[i]));
    }
    const TensorPtr out = op.forward(inputs);
    const auto& out_value = out->get_value();
    const auto& out_shape = out_value.logical_shape();
    const auto out_grad = uniform(std::vector<uint32_t>(out_shape.cbegin(), out_shape.cend()), -1.F, 1.F, 99U);
    out->set_grad(ttml::core::from_xtensor(out_grad, device, out_value.layout()));

    BackwardRun run;
    {
        // The scoped capture ends even if backward throws, so a failing row cannot leave later tests capturing.
        ttnn::graph::ScopedGraphCapture capture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        out->backward();
        run.calls = ttnn::graph::extract_calltrace(capture.end_graph_capture());
    }
    for (const auto& input : inputs) {
        run.grads.push_back(
            input->is_grad_initialized() ? std::optional(ttml::core::to_xtensor(input->get_grad())) : std::nullopt);
    }
    ttml::autograd::ctx().reset_graph();
    return run;
}

// Counts the device ops in a backward calltrace. Host-side entries such as Tensor::reshape or buffer creation are
// not grad work.
OpCounts count_device_ops(const std::vector<std::string>& calls) {
    OpCounts counts;
    for (const auto& name : calls) {
        if (name.ends_with("Operation") && name.find("::") == std::string::npos) {
            ++counts[name];
        }
    }
    return counts;
}

using BinaryOp = std::function<TensorPtr(const TensorPtr&, const TensorPtr&)>;

const std::vector<uint32_t> kActivation{2U, 1U, 32U, 64U};
const std::vector<uint32_t> kFeatureRow{1U, 1U, 1U, 64U};

OpUnderTest binary(
    BinaryOp op, const std::vector<uint32_t>& a_shape, const std::vector<uint32_t>& b_shape, float b_lo = -1.F) {
    return OpUnderTest{
        .inputs = {{uniform(a_shape, -1.F, 1.F, 1U)}, {uniform(b_shape, b_lo, b_lo + 2.F, 2U)}},
        .forward = [op](const std::vector<TensorPtr>& in) { return op(in[0], in[1]); }};
}

using LayerNormOp = TensorPtr (*)(const TensorPtr&, const TensorPtr&, std::optional<TensorPtr>);

// Inputs: tensor, gamma, then beta when `with_beta`.
OpUnderTest layernorm(LayerNormOp layernorm_op, bool with_beta) {
    OpUnderTest op{
        .inputs = {{uniform(kActivation, -1.F, 1.F, 1U)}, {uniform(kFeatureRow, 0.5F, 1.5F, 2U)}},
        .forward = [layernorm_op](const std::vector<TensorPtr>& in) {
            const auto beta = in.size() > 2U ? std::optional(in[2]) : std::nullopt;
            return layernorm_op(in[0], in[1], beta);
        }};
    if (with_beta) {
        op.inputs.push_back({uniform(kFeatureRow, -1.F, 1.F, 3U)});
    }
    return op;
}

OpUnderTest rmsnorm_composite() {
    return OpUnderTest{
        .inputs = {{uniform(kActivation, -1.F, 1.F, 1U)}, {uniform(kFeatureRow, 0.5F, 1.5F, 2U)}},
        .forward = [](const std::vector<TensorPtr>& in) {
            return ttml::ops::rmsnorm_composite(in[0], in[1], /*epsilon=*/1e-6F);
        }};
}

OpUnderTest rmsnorm() {
    return OpUnderTest{
        .inputs = {{uniform(kActivation, -1.F, 1.F, 1U)}, {uniform(kFeatureRow, 0.5F, 1.5F, 2U)}},
        .forward = [](const std::vector<TensorPtr>& in) {
            return ttml::ops::rmsnorm(in[0], in[1], /*epsilon=*/1e-6F);
        }};
}

// Inputs: x [2, 1, 32, 64], weight [1, 1, 1, 3], bias [1, 1, 1, 1].
OpUnderTest polynorm3(bool composite) {
    return OpUnderTest{
        .inputs =
            {{uniform(kActivation, -1.F, 1.F, 1U)},
             {uniform({1U, 1U, 1U, 3U}, 0.5F, 1.5F, 2U)},
             {uniform({1U, 1U, 1U, 1U}, -1.F, 1.F, 3U)}},
        .forward = [composite](const std::vector<TensorPtr>& in) {
            return composite ? ttml::ops::polynorm3_composite(in[0], in[1], in[2])
                             : ttml::ops::polynorm3(in[0], in[1], in[2]);
        }};
}

// x: [2, 1, 32, D], w1 / w3: [1, 1, H, D], w2: [1, 1, D, H] with D = 64, H = 128.
OpUnderTest swiglu() {
    return OpUnderTest{
        .inputs =
            {{uniform(kActivation, -1.F, 1.F, 1U)},
             {uniform({1U, 1U, 128U, 64U}, -0.1F, 0.1F, 2U)},
             {uniform({1U, 1U, 64U, 128U}, -0.1F, 0.1F, 3U)},
             {uniform({1U, 1U, 128U, 64U}, -0.1F, 0.1F, 4U)}},
        .forward = [](const std::vector<TensorPtr>& in) { return ttml::ops::swiglu(in[0], in[1], in[2], in[3]); }};
}

// query, key, value: [1, 2, S, 32] with S = 64. With `causal_mask`, a [1, 1, S, S] lower-triangular mask of ones is
// passed too; it is not an input, so it never requires grad.
OpUnderTest attention(bool sigmoid, bool causal_mask = false) {
    constexpr uint32_t S = 64U;
    const std::vector<uint32_t> shape{1U, 2U, S, 32U};
    std::optional<xt::xarray<float>> mask;
    if (causal_mask) {
        mask = xt::xarray<float>(xt::tril(xt::ones<float>({1U, 1U, S, S})));
    }
    return OpUnderTest{
        .inputs = {{uniform(shape, -1.F, 1.F, 1U)}, {uniform(shape, -1.F, 1.F, 2U)}, {uniform(shape, -1.F, 1.F, 3U)}},
        .forward = [sigmoid, mask](const std::vector<TensorPtr>& in) {
            std::optional<TensorPtr> t_mask;
            if (mask.has_value()) {
                t_mask = ttml::autograd::create_tensor(
                    ttml::core::from_xtensor(*mask, &ttml::autograd::ctx().get_device()), /*requires_grad=*/false);
            }
            return sigmoid ? ttml::ops::scaled_sigmoid_dot_product_attention(in[0], in[1], in[2], t_mask)
                           : ttml::ops::scaled_dot_product_attention_composite(in[0], in[1], in[2], t_mask);
        }};
}

// Fused kernels (sdpa_fw / sdpa_bw) with the causal mask they use when no mask is passed. query, key, value: [1, 2,
// 64, 32].
OpUnderTest fused_attention() {
    const std::vector<uint32_t> shape{1U, 2U, 64U, 32U};
    return OpUnderTest{
        .inputs = {{uniform(shape, -1.F, 1.F, 1U)}, {uniform(shape, -1.F, 1.F, 2U)}, {uniform(shape, -1.F, 1.F, 3U)}},
        .forward = [](const std::vector<TensorPtr>& in) {
            return ttml::ops::scaled_dot_product_attention(in[0], in[1], in[2]);
        }};
}

// qs: [1, 1, 32, 64] split into 2 query heads, kvs: [1, 1, 32, 64] split into 1 key and 1 value head, all of size 32.
// The heads feed composite attention so backward has a single root.
OpUnderTest grouped_heads() {
    return OpUnderTest{
        .inputs = {{uniform({1U, 1U, 32U, 64U}, -1.F, 1.F, 1U)}, {uniform({1U, 1U, 32U, 64U}, -1.F, 1.F, 2U)}},
        .forward = [](const std::vector<TensorPtr>& in) {
            auto [q, k, v] = ttml::ops::grouped_heads_creation(in[0], in[1], /*num_heads=*/2U, /*num_groups=*/1U);
            return ttml::ops::scaled_dot_product_attention_composite(q, k, v);
        }};
}

// Inputs: kv_up [2, 1, 32, 2 * (32 + 64)], k_pe [2, 1, 32, 32]: 2 heads with qk_nope = qk_rope = 32 and v = 64. K and
// V then share their [2, 2, 32, 64] shape, so their sum is a single root that sends a random grad into each.
OpUnderTest mla_kv_assemble() {
    constexpr uint32_t n_heads = 2U, qk_nope = 32U, qk_rope = 32U, v_dim = 64U;
    return OpUnderTest{
        .inputs =
            {{uniform({2U, 1U, 32U, n_heads * (qk_nope + v_dim)}, -1.F, 1.F, 1U)},
             {uniform({2U, 1U, 32U, qk_rope}, -1.F, 1.F, 2U)}},
        .forward = [](const std::vector<TensorPtr>& in) {
            auto [k, v] = ttml::ops::mla_kv_assemble(in[0], in[1], n_heads, qk_nope, qk_rope, v_dim);
            return ttml::ops::add(k, v);
        }};
}

// Hidden size shared by the MoE builders below.
constexpr uint32_t kMoeH = 64U;

// Inputs: grouped [1, 1, T_cap, H], then w_gate[0..E), w_up[0..E), w_down[0..E) with E = 2, H = kMoeH, I = 128.
// Expert e owns rows offsets[e]..offsets[e] + counts[e]; the rest of its tile-padded slot stays zero.
constexpr uint32_t kMoeExperts = 2U;

OpUnderTest moe_ffn_swiglu() {
    constexpr uint32_t H = kMoeH, I = 128U;
    const std::vector<uint32_t> counts{32U, 40U};
    const std::vector<uint32_t> offsets{0U, 32U, 96U};

    xt::xarray<float> grouped = uniform({1U, 1U, offsets.back(), H}, -1.F, 1.F, 1U);
    for (uint32_t e = 0; e < kMoeExperts; ++e) {
        xt::view(grouped, 0, 0, xt::range(offsets[e] + counts[e], offsets[e + 1U]), xt::all()) = 0.F;
    }
    OpUnderTest op{.inputs = {{grouped}}};
    for (uint32_t e = 0; e < kMoeExperts; ++e) {
        op.inputs.push_back({uniform({1U, 1U, I, H}, -0.5F, 0.5F, 10U + e)});
    }
    for (uint32_t e = 0; e < kMoeExperts; ++e) {
        op.inputs.push_back({uniform({1U, 1U, I, H}, -0.5F, 0.5F, 20U + e)});
    }
    for (uint32_t e = 0; e < kMoeExperts; ++e) {
        op.inputs.push_back({uniform({1U, 1U, H, I}, -0.5F, 0.5F, 30U + e)});
    }
    op.forward = [offsets](const std::vector<TensorPtr>& in) {
        const auto t_offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            offsets,
            ttnn::Shape({static_cast<uint32_t>(offsets.size())}),
            &ttml::autograd::ctx().get_device(),
            ttnn::Layout::ROW_MAJOR);
        const auto experts = [&in](uint32_t first) {
            return std::vector<TensorPtr>(in.begin() + first, in.begin() + first + kMoeExperts);
        };
        return ttml::ops::moe_ffn_swiglu_fw(
            in[0], t_offsets, experts(1U), experts(1U + kMoeExperts), experts(1U + 2U * kMoeExperts));
    };
    return op;
}

// Routing for D = B = 1, S = 32 tokens picking K = 2 of E = 4 experts, with experts 0 and 1 local.
constexpr uint32_t kMoeD = 1U, kMoeB = 1U, kMoeS = 32U, kMoeE = 4U, kMoeK = 2U;
constexpr uint32_t kMoeLocal = 2U;

ttml::test_utils::moe::MoeHostInputs moe_routing_inputs() {
    return ttml::test_utils::moe::make_moe_host_inputs(
        {.D = kMoeD, .B = kMoeB, .S = kMoeS, .H = kMoeH, .E = kMoeE, .K = kMoeK});
}

ttml::test_utils::moe::MoeDeviceInputs moe_device_inputs(const ttml::test_utils::moe::MoeHostInputs& host) {
    return ttml::test_utils::moe::to_device_inputs(host, {0U, 1U}, &ttml::autograd::ctx().get_device());
}

// Inputs: dispatched [D, B, S, H], scores [D, B, S, K]. moe_group_op feeds moe_ungroup_op so backward has a single
// root; the frozen input's branch is the one inside moe_group_op.
OpUnderTest moe_group() {
    const auto host = moe_routing_inputs();
    return OpUnderTest{
        .inputs = {{host.dispatched, ttnn::Layout::ROW_MAJOR}, {host.scores, ttnn::Layout::ROW_MAJOR}},
        .forward = [host](const std::vector<TensorPtr>& in) {
            const auto dev = moe_device_inputs(host);
            const auto g = ttml::ops::moe_group_op(in[0], dev.metadata_u16, in[1], dev.leids_u16, kMoeLocal, kMoeK);
            return ttml::ops::moe_ungroup_op(
                g.grouped,
                g.grouped_scores,
                dev.metadata_u16,
                dev.leids_u16,
                g.plan,
                g.offsets,
                kMoeLocal,
                kMoeK,
                kMoeD,
                kMoeB,
                kMoeS);
        }};
}

// Inputs: expert_out [1, 1, T_cap, H] and grouped_scores [1, 1, 1, T_cap], taken from one metal::moe_group run so
// moe_ungroup_op sees them as leaves. Pad rows are never written, so non-finite garbage there is zeroed.
OpUnderTest moe_ungroup() {
    const auto host = moe_routing_inputs();
    const auto dev = moe_device_inputs(host);
    auto [grouped, grouped_scores, k_slot, counts, offsets, plan] =
        ttml::metal::moe_group(dev.dispatched_bf16, dev.metadata_u16, dev.scores_bf16, dev.leids_u16, kMoeLocal, kMoeK);
    const auto finite = [](const ttnn::Tensor& t) {
        const xt::xarray<float> v = ttml::core::to_xtensor(t);
        return xt::xarray<float>(xt::where(xt::isfinite(v), v, 0.F));
    };
    return OpUnderTest{
        .inputs = {{finite(grouped)}, {finite(grouped_scores), ttnn::Layout::ROW_MAJOR}},
        .forward = [dev, plan, offsets](const std::vector<TensorPtr>& in) {
            return ttml::ops::moe_ungroup_op(
                in[0], in[1], dev.metadata_u16, dev.leids_u16, plan, offsets, kMoeLocal, kMoeK, kMoeD, kMoeB, kMoeS);
        }};
}

TensorPtr add_op(const TensorPtr& a, const TensorPtr& b) {
    return ttml::ops::add(a, b);
}
TensorPtr sub_op(const TensorPtr& a, const TensorPtr& b) {
    return ttml::ops::sub(a, b);
}
TensorPtr mul_op(const TensorPtr& a, const TensorPtr& b) {
    return ttml::ops::mul(a, b);
}
TensorPtr div_op(const TensorPtr& a, const TensorPtr& b) {
    return ttml::ops::div(a, b);
}
TensorPtr min_op(const TensorPtr& a, const TensorPtr& b) {
    return ttml::ops::min(a, b);
}
TensorPtr max_op(const TensorPtr& a, const TensorPtr& b) {
    return ttml::ops::max(a, b);
}

}  // namespace

class GradSkippingTest : public ::testing::TestWithParam<GradSkipCase> {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

// add_grad drops a frozen input's grad, so the grads alone cannot show its work was skipped. Each row runs backward
// with every input trainable, then with `frozen` frozen. The second run must launch fewer of each of the row's
// `must_skip` device ops and no more of any op, and every trainable input must get the same grad bit for bit.
TEST_P(GradSkippingTest, FrozenInputsSkipTheirGradWork) {
    const GradSkipCase& c = GetParam();
    if (c.blackhole_only && ttml::autograd::ctx().get_device().arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << c.name << " is only supported on Blackhole.";
    }
    const OpUnderTest op = c.make();
    std::vector<bool> requires_grad(op.inputs.size(), true);
    const BackwardRun reference = run_backward(op, requires_grad);
    for (const size_t i : c.frozen) {
        requires_grad[i] = false;
    }
    const BackwardRun frozen = run_backward(op, requires_grad);

    OpCounts skipped = count_device_ops(reference.calls);
    for (const auto& [name, count] : count_device_ops(frozen.calls)) {
        skipped[name] -= count;
    }
    const auto frozen_ops = fmt::format("{}", fmt::join(frozen.calls, ", "));
    for (const auto& [name, count] : skipped) {
        EXPECT_GE(count, 0) << c.name << ": frozen backward launches " << name << " more often. Ops: " << frozen_ops;
    }
    for (const auto& name : c.must_skip) {
        EXPECT_GT(skipped[name], 0) << c.name << ": frozen backward did not skip any " << name
                                    << ". Ops: " << frozen_ops;
    }
    for (size_t i = 0; i < op.inputs.size(); ++i) {
        if (!requires_grad[i]) {
            continue;
        }
        ASSERT_TRUE(frozen.grads[i].has_value()) << c.name << ": trainable input " << i << " got no grad";
        ASSERT_TRUE(reference.grads[i].has_value()) << c.name;
        EXPECT_TRUE(xt::all(xt::equal(*frozen.grads[i], *reference.grads[i])))
            << c.name << ": trainable input " << i << " grad changed when other inputs were frozen";
    }
}

// Each row freezes inputs whose grad does device work: for add and sub's a that is only the broadcast reduction, so
// those rows freeze the broadcast operand.
INSTANTIATE_TEST_SUITE_P(
    GradSkipCases,
    GradSkippingTest,
    ::testing::Values(
        GradSkipCase{
            .name = "Add_FrozenBroadcastA",
            .make = [] { return binary(add_op, kFeatureRow, kActivation); },
            .frozen = {0},
            .must_skip = {kMorehSum}},
        GradSkipCase{
            .name = "Add_FrozenBroadcastB",
            .make = [] { return binary(add_op, kActivation, kFeatureRow); },
            .frozen = {1},
            .must_skip = {kMorehSum}},
        GradSkipCase{
            .name = "Sub_FrozenBroadcastA",
            .make = [] { return binary(sub_op, kFeatureRow, kActivation); },
            .frozen = {0},
            .must_skip = {kMorehSum}},
        // neg
        GradSkipCase{
            .name = "Sub_FrozenB",
            .make = [] { return binary(sub_op, kActivation, kActivation); },
            .frozen = {1},
            .must_skip = {kUnary}},
        GradSkipCase{
            .name = "Mul_FrozenA",
            .make = [] { return binary(mul_op, kActivation, kActivation); },
            .frozen = {0},
            .must_skip = {kBinary}},
        GradSkipCase{
            .name = "Mul_FrozenB",
            .make = [] { return binary(mul_op, kActivation, kActivation); },
            .frozen = {1},
            .must_skip = {kBinary}},
        // b * out_grad, then the broadcast reduction back to b's shape.
        GradSkipCase{
            .name = "Mul_FrozenBroadcastB",
            .make = [] { return binary(mul_op, kActivation, kFeatureRow); },
            .frozen = {1},
            .must_skip = {kBinary, kMorehSum}},
        // b in [1, 3) keeps 1 / b well away from overflow. a: grad * reciprocal(b). b: -grad * (a * 1/b) * 1/b.
        GradSkipCase{
            .name = "Div_FrozenA",
            .make = [] { return binary(div_op, kActivation, kActivation, 1.F); },
            .frozen = {0},
            .must_skip = {kUnary, kBinary}},
        GradSkipCase{
            .name = "Div_FrozenB",
            .make = [] { return binary(div_op, kActivation, kActivation, 1.F); },
            .frozen = {1},
            .must_skip = {kUnary, kBinary}},
        // The frozen side's comparison of b - a with zero and its where; the tie split is shared.
        GradSkipCase{
            .name = "Min_FrozenA",
            .make = [] { return binary(min_op, kActivation, kActivation); },
            .frozen = {0},
            .must_skip = {kUnary, kTernary}},
        GradSkipCase{
            .name = "Min_FrozenB",
            .make = [] { return binary(min_op, kActivation, kActivation); },
            .frozen = {1},
            .must_skip = {kUnary, kTernary}},
        GradSkipCase{
            .name = "Max_FrozenA",
            .make = [] { return binary(max_op, kActivation, kActivation); },
            .frozen = {0},
            .must_skip = {kUnary, kTernary}},
        GradSkipCase{
            .name = "Max_FrozenB",
            .make = [] { return binary(max_op, kActivation, kActivation); },
            .frozen = {1},
            .must_skip = {kUnary, kTernary}},
        // The dtensor chain; normalized_tensor is still recomputed for dgamma.
        GradSkipCase{
            .name = "CompositeLayerNorm_FrozenTensor",
            .make = [] { return layernorm(ttml::ops::composite_layernorm, /*with_beta=*/true); },
            .frozen = {0},
            .must_skip = {kBinary, kMorehMean}},
        GradSkipCase{
            .name = "CompositeLayerNorm_FrozenGamma",
            .make = [] { return layernorm(ttml::ops::composite_layernorm, /*with_beta=*/true); },
            .frozen = {1},
            .must_skip = {kBinary, kMorehSum}},
        // Without beta there is no dbeta branch at all; gamma's skip is unchanged.
        GradSkipCase{
            .name = "CompositeLayerNormNoBeta_FrozenGamma",
            .make = [] { return layernorm(ttml::ops::composite_layernorm, /*with_beta=*/false); },
            .frozen = {1},
            .must_skip = {kBinary, kMorehSum}},
        GradSkipCase{
            .name = "CompositeLayerNorm_FrozenBeta",
            .make = [] { return layernorm(ttml::ops::composite_layernorm, /*with_beta=*/true); },
            .frozen = {2},
            .must_skip = {kMorehSum}},
        // dx is skipped inside layernorm_bw, which is still launched for dgamma and dbeta, so there is no op to count;
        // the row checks that their grads are unchanged.
        GradSkipCase{
            .name = "LayerNorm_FrozenTensor",
            .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/true); },
            .frozen = {0}},
        // The kernel skips the frozen side's partials and the wrapper skips their reduction.
        GradSkipCase{
            .name = "LayerNorm_FrozenGamma",
            .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/true); },
            .frozen = {1},
            .must_skip = {kReduce}},
        GradSkipCase{
            .name = "LayerNormNoBeta_FrozenGamma",
            .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/false); },
            .frozen = {1},
            .must_skip = {kReduce}},
        GradSkipCase{
            .name = "LayerNorm_FrozenBeta",
            .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/true); },
            .frozen = {2},
            .must_skip = {kReduce}},
        GradSkipCase{
            .name = "RMSNormComposite_FrozenTensor",
            .make = rmsnorm_composite,
            .frozen = {0},
            .must_skip = {kBinary, kMorehSum}},
        GradSkipCase{
            .name = "RMSNormComposite_FrozenGamma", .make = rmsnorm_composite, .frozen = {1}, .must_skip = {kBinary}},
        // dx is skipped inside rmsnorm_bw, which is still launched for dgamma, so there is no op to count; the row
        // checks that gamma's grad is unchanged.
        GradSkipCase{.name = "RMSNorm_FrozenTensor", .make = rmsnorm, .frozen = {0}},
        // The kernel skips the dgamma partials and the wrapper skips their reduction.
        GradSkipCase{.name = "RMSNorm_FrozenGamma", .make = rmsnorm, .frozen = {1}, .must_skip = {kReduce}},
        // dx is skipped inside polynorm3_bw, which is still launched for dw and db, so there is no op to count; the
        // row checks that their grads are unchanged.
        GradSkipCase{
            .name = "PolyNorm3_FrozenInput", .make = [] { return polynorm3(/*composite=*/false); }, .frozen = {0}},
        // dw and db share the packed partials: the kernel skips them, and the wrapper their reduction, only when both
        // weight and bias are frozen; otherwise only the frozen side's extraction is skipped.
        GradSkipCase{
            .name = "PolyNorm3_FrozenWeight",
            .make = [] { return polynorm3(/*composite=*/false); },
            .frozen = {1},
            .must_skip = {kSlice, kConcat}},
        GradSkipCase{
            .name = "PolyNorm3_FrozenBias",
            .make = [] { return polynorm3(/*composite=*/false); },
            .frozen = {2},
            .must_skip = {kSlice}},
        GradSkipCase{
            .name = "PolyNorm3_FrozenWeightAndBias",
            .make = [] { return polynorm3(/*composite=*/false); },
            .frozen = {1, 2},
            .must_skip = {kReduce, kSlice, kConcat}},
        GradSkipCase{
            .name = "PolyNorm3Composite_FrozenInput",
            .make = [] { return polynorm3(/*composite=*/true); },
            .frozen = {0},
            .must_skip = {kBinary}},
        GradSkipCase{
            .name = "PolyNorm3Composite_FrozenWeight",
            .make = [] { return polynorm3(/*composite=*/true); },
            .frozen = {1},
            .must_skip = {kBinary, kReduce, kConcat}},
        GradSkipCase{
            .name = "PolyNorm3Composite_FrozenBias",
            .make = [] { return polynorm3(/*composite=*/true); },
            .frozen = {2},
            .must_skip = {kReduce}},
        // Only db is left, so the shared x^2, x^3 and their RMS normalizations are skipped too.
        GradSkipCase{
            .name = "PolyNorm3Composite_OnlyBias",
            .make = [] { return polynorm3(/*composite=*/true); },
            .frozen = {0, 1},
            .must_skip = {kUnary, kBinary, kReduce, kConcat}},
        // dx = dL_dlinear1 @ w1 + dL_dgate @ w3.
        GradSkipCase{.name = "SwiGLU_FrozenX", .make = swiglu, .frozen = {0}, .must_skip = {kMatmul}},
        GradSkipCase{.name = "SwiGLU_FrozenW1", .make = swiglu, .frozen = {1}, .must_skip = {kMatmul}},
        GradSkipCase{.name = "SwiGLU_FrozenW2", .make = swiglu, .frozen = {2}, .must_skip = {kMatmul}},
        GradSkipCase{.name = "SwiGLU_FrozenW3", .make = swiglu, .frozen = {3}, .must_skip = {kMatmul}},
        // The shared dL_dprod matmul and swiglu_elemwise_bw, then dx (2 matmuls + add), dW1 and dW3.
        GradSkipCase{
            .name = "SwiGLU_OnlyW2", .make = swiglu, .frozen = {0, 1, 3}, .must_skip = {kMatmul, kSwigluElemwiseBw}},
        GradSkipCase{
            .name = "SDPAComposite_FrozenQ",
            .make = [] { return attention(/*sigmoid=*/false); },
            .frozen = {0},
            .must_skip = {kMatmul}},
        GradSkipCase{
            .name = "SDPAComposite_FrozenK",
            .make = [] { return attention(/*sigmoid=*/false); },
            .frozen = {1},
            .must_skip = {kMatmul}},
        GradSkipCase{
            .name = "SDPAComposite_FrozenV",
            .make = [] { return attention(/*sigmoid=*/false); },
            .frozen = {2},
            .must_skip = {kMatmul}},
        // dL/d(attention weights), the softmax backward and its scale, dQ and dK.
        GradSkipCase{
            .name = "SDPAComposite_OnlyV",
            .make = [] { return attention(/*sigmoid=*/false); },
            .frozen = {0, 1},
            .must_skip = {kMatmul, "MorehSoftmaxBackwardOperation"}},
        GradSkipCase{
            .name = "SigmoidSDPA_FrozenQ",
            .make = [] { return attention(/*sigmoid=*/true); },
            .frozen = {0},
            .must_skip = {kMatmul}},
        GradSkipCase{
            .name = "SigmoidSDPA_FrozenK",
            .make = [] { return attention(/*sigmoid=*/true); },
            .frozen = {1},
            .must_skip = {kMatmul}},
        GradSkipCase{
            .name = "SigmoidSDPA_FrozenV",
            .make = [] { return attention(/*sigmoid=*/true); },
            .frozen = {2},
            .must_skip = {kMatmul}},
        // dL/d(attention weights), the sigmoid backward and its recomputed input, dQ and dK with their scales.
        GradSkipCase{
            .name = "SigmoidSDPA_OnlyV",
            .make = [] { return attention(/*sigmoid=*/true); },
            .frozen = {0, 1},
            .must_skip = {kMatmul, kUnaryBackward}},
        // As SigmoidSDPA_OnlyV, plus the where that masks the sigmoid backward (a BinaryNg op for a scalar `other`).
        GradSkipCase{
            .name = "SigmoidSDPAMasked_OnlyV",
            .make = [] { return attention(/*sigmoid=*/true, /*causal_mask=*/true); },
            .frozen = {0, 1},
            .must_skip = {kMatmul, kUnaryBackward}},
        // grad_Q is skipped inside the Q kernel, which still runs for u_scaler, so there is no op to count; the row
        // checks that grad_K and grad_V are unchanged.
        GradSkipCase{.name = "FusedSDPA_FrozenQ", .make = fused_attention, .frozen = {0}},
        GradSkipCase{.name = "FusedSDPA_FrozenKV", .make = fused_attention, .frozen = {1, 2}, .must_skip = {kSdpaBwKV}},
        // Without a gate, grad_V alone needs nothing from the Q kernel.
        GradSkipCase{.name = "FusedSDPA_OnlyV", .make = fused_attention, .frozen = {0, 1}, .must_skip = {kSdpaBwQ}},
        // The frozen grad is skipped inside mla_kv_assemble_bw, which is still launched for the other, so there is no
        // op to count; the rows check that the other grad is unchanged.
        GradSkipCase{.name = "MlaKvAssemble_FrozenKvUp", .make = mla_kv_assemble, .frozen = {0}},
        GradSkipCase{.name = "MlaKvAssemble_FrozenKPe", .make = mla_kv_assemble, .frozen = {1}},
        GradSkipCase{
            .name = "GroupedHeads_FrozenQs", .make = grouped_heads, .frozen = {0}, .must_skip = {kNlpConcatHeads}},
        GradSkipCase{
            .name = "GroupedHeads_FrozenKvs",
            .make = grouped_heads,
            .frozen = {1},
            .must_skip = {kNlpConcatHeads, "ConcatDeviceOperation"}},
        // Per expert: 2 dX matmuls; plus the two zeroed dX buffers and their sum.
        GradSkipCase{
            .name = "MoeFfnSwiglu_FrozenGrouped",
            .make = moe_ffn_swiglu,
            .frozen = {0},
            .must_skip = {kVariableMatmul},
            .blackhole_only = true},
        GradSkipCase{
            .name = "MoeFfnSwiglu_FrozenWGate",
            .make = moe_ffn_swiglu,
            .frozen = {1, 2},
            .must_skip = {kVariableMatmul},
            .blackhole_only = true},
        GradSkipCase{
            .name = "MoeFfnSwiglu_FrozenWUp",
            .make = moe_ffn_swiglu,
            .frozen = {3, 4},
            .must_skip = {kVariableMatmul},
            .blackhole_only = true},
        // Per expert: 1 dW_down matmul; plus the recomputed activated.
        GradSkipCase{
            .name = "MoeFfnSwiglu_FrozenWDown",
            .make = moe_ffn_swiglu,
            .frozen = {5, 6},
            .must_skip = {kVariableMatmul},
            .blackhole_only = true},
        GradSkipCase{
            .name = "MoeFfnSwiglu_FrozenExpert0",
            .make = moe_ffn_swiglu,
            .frozen = {1, 3, 5},
            .must_skip = {kVariableMatmul},
            .blackhole_only = true},
        // Per expert: d_activated, dW_gate, dW_up and 2 dX matmuls; plus swiglu_elemwise_bw and the dX buffers and sum.
        GradSkipCase{
            .name = "MoeFfnSwiglu_OnlyWDown",
            .make = moe_ffn_swiglu,
            .frozen = {0, 1, 2, 3, 4},
            .must_skip = {kVariableMatmul, kSwigluElemwiseBw},
            .blackhole_only = true},
        // The row-scatter moe_ungroup, and moe_ungroup_op's d(expert_out) since grouped no longer requires grad.
        GradSkipCase{
            .name = "MoeGroup_FrozenDispatched",
            .make = moe_group,
            .frozen = {0},
            .must_skip = {kMoeUngroup, kTranspose, kBinary}},
        // The one-hot scatter of d(grouped_scores) into d(scores) and its moe_ungroup, and moe_ungroup_op's
        // d(grouped_scores) since grouped_scores no longer requires grad.
        GradSkipCase{
            .name = "MoeGroup_FrozenScores",
            .make = moe_group,
            .frozen = {1},
            .must_skip = {kMoeUngroup, kTranspose, kBinary, kReduce}},
        GradSkipCase{
            .name = "MoeUngroup_FrozenExpertOut",
            .make = moe_ungroup,
            .frozen = {0},
            .must_skip = {kTranspose, kBinary}},
        GradSkipCase{
            .name = "MoeUngroup_FrozenGroupedScores",
            .make = moe_ungroup,
            .frozen = {1},
            .must_skip = {kTranspose, kBinary}}),
    [](const ::testing::TestParamInfo<GradSkipCase>& info) { return info.param.name; });
