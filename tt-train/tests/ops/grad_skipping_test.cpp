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
#include <tt_stl/reflection.hpp>
#include <vector>
#include <xtensor/core/xmath.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_q_device_operation.hpp"
#include "metal/ops/swiglu_elemwise_bw/device/swiglu_elemwise_bw_device_operation.hpp"
#include "ops/binary_ops.hpp"
#include "ops/layernorm_op.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/rmsnorm_op.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ops/swiglu_op.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/operations/eltwise/unary_backward/device/unary_backward_device_operation.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_mean/device/moreh_mean_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_sum/device/moreh_sum_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op_device_operation.hpp"

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
    // Device ops that tt-train itself launches for the frozen inputs' grads, with how many fewer times the frozen run
    // must launch each.
    OpCounts skips;
    // The frozen inputs' grads are skipped inside a fused kernel that still runs, so the frozen run must compile a
    // program of its own.
    bool changes_kernel_flags = false;
};

// gtest prints a failing row's parameter through this; without it the row prints as raw bytes.
void PrintTo(const GradSkipCase& c, std::ostream* os) {
    *os << c.name;
}

// Device-op names as the graph capture records them. They come from the op types, so renaming an op breaks the build
// rather than silently counting zero.
template <typename Op>
std::string op_name() {
    return std::string(ttsl::get_type_name<Op>());
}

const std::string kBinary = op_name<ttnn::operations::binary_ng::BinaryNgDeviceOperation>();
const std::string kUnary = op_name<ttnn::operations::unary::UnaryDeviceOperation>();
const std::string kMatmul = op_name<ttnn::prim::MatmulDeviceOperation>();
const std::string kMorehSum = op_name<ttnn::operations::moreh::moreh_sum::MorehSumOperation>();
const std::string kMorehMean = op_name<ttnn::operations::moreh::moreh_mean::MorehMeanOperation>();
const std::string kMorehSoftmaxBw =
    op_name<ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOperation>();
const std::string kReduce = op_name<ttnn::prim::ReduceDeviceOperation>();
const std::string kConcat = op_name<ttnn::prim::ConcatDeviceOperation>();
const std::string kSwigluElemwiseBw =
    op_name<ttml::metal::ops::swiglu_elemwise_bw::device::SwigluElemwiseBwDeviceOperation>();
const std::string kUnaryBackward = op_name<ttnn::operations::unary_backward::UnaryBackwardDeviceOperation>();
const std::string kNlpConcatHeads = op_name<ttnn::experimental::prim::NLPConcatHeadsDeviceOperation>();
const std::string kSdpaBwQ = op_name<ttml::metal::ops::sdpa_bw::device::SDPABackwardQDeviceOperation>();
const std::string kSdpaBwKV = op_name<ttml::metal::ops::sdpa_bw::device::SDPABackwardKVDeviceOperation>();

xt::xarray<float> uniform(const std::vector<uint32_t>& shape, float lo, float hi, uint32_t seed) {
    return ttml::test_utils::make_uniform_xarray<float>(shape, lo, hi, seed);
}

// The grad of every input that received one, and the ops backward launched.
struct BackwardRun {
    std::vector<std::optional<xt::xarray<float>>> grads;
    std::vector<std::string> calls;
};

// With freeze_before_backward, every input is frozen after forward recorded the op's node, so backward still runs it.
BackwardRun run_backward(
    const OpUnderTest& op, const std::vector<bool>& requires_grad, bool freeze_before_backward = false) {
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
    if (freeze_before_backward) {
        for (const auto& input : inputs) {
            input->set_requires_grad(false);
        }
    }

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
// with every input trainable, then with `frozen` frozen. The second run must launch each of the row's `skips` device
// ops exactly that many fewer times and no op more often, a fused kernel that takes the flags must compile a new
// program, and every trainable input must get the same grad bit for bit.
TEST_P(GradSkippingTest, FrozenInputsSkipTheirGradWork) {
    const GradSkipCase& c = GetParam();
    const OpUnderTest op = c.make();
    auto* device = &ttml::autograd::ctx().get_device();
    if (c.changes_kernel_flags) {
        // Another row may already have compiled the frozen run's program.
        device->enable_program_cache();
        device->clear_program_cache();
    }
    std::vector<bool> requires_grad(op.inputs.size(), true);
    const BackwardRun reference = run_backward(op, requires_grad);
    const auto entries_after_reference = device->num_program_cache_entries();
    for (const size_t i : c.frozen) {
        requires_grad[i] = false;
    }
    const BackwardRun frozen = run_backward(op, requires_grad);
    if (c.changes_kernel_flags) {
        EXPECT_GT(device->num_program_cache_entries(), entries_after_reference)
            << c.name
            << ": the frozen run reused the full backward's programs, so the skip flags did not reach the kernel";
    }

    OpCounts skipped = count_device_ops(reference.calls);
    for (const auto& [name, count] : count_device_ops(frozen.calls)) {
        skipped[name] -= count;
    }
    const auto frozen_ops = fmt::format("{}", fmt::join(frozen.calls, ", "));
    for (const auto& [name, count] : skipped) {
        EXPECT_GE(count, 0) << c.name << ": frozen backward launches " << name << " more often. Ops: " << frozen_ops;
    }
    for (const auto& [name, count] : c.skips) {
        EXPECT_EQ(skipped[name], count) << c.name << ": frozen backward skipped the wrong number of " << name
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
            .skips = {{kMorehSum, 2}}},
        GradSkipCase{
            .name = "Add_FrozenBroadcastB",
            .make = [] { return binary(add_op, kActivation, kFeatureRow); },
            .frozen = {1},
            .skips = {{kMorehSum, 2}}},
        GradSkipCase{
            .name = "Sub_FrozenBroadcastA",
            .make = [] { return binary(sub_op, kFeatureRow, kActivation); },
            .frozen = {0},
            .skips = {{kMorehSum, 2}}},
        // neg
        GradSkipCase{
            .name = "Sub_FrozenB",
            .make = [] { return binary(sub_op, kActivation, kActivation); },
            .frozen = {1},
            .skips = {{kUnary, 1}}},
        GradSkipCase{
            .name = "Mul_FrozenA",
            .make = [] { return binary(mul_op, kActivation, kActivation); },
            .frozen = {0},
            .skips = {{kBinary, 1}}},
        GradSkipCase{
            .name = "Mul_FrozenB",
            .make = [] { return binary(mul_op, kActivation, kActivation); },
            .frozen = {1},
            .skips = {{kBinary, 1}}},
        // b * out_grad, then the broadcast reduction back to b's shape.
        GradSkipCase{
            .name = "Mul_FrozenBroadcastB",
            .make = [] { return binary(mul_op, kActivation, kFeatureRow); },
            .frozen = {1},
            .skips = {{kBinary, 1}, {kMorehSum, 2}}},
        // b in [1, 3) keeps 1 / b well away from overflow. a: grad * reciprocal(b). b: -grad * (a * 1/b) * 1/b.
        GradSkipCase{
            .name = "Div_FrozenA",
            .make = [] { return binary(div_op, kActivation, kActivation, 1.F); },
            .frozen = {0},
            .skips = {{kUnary, 1}, {kBinary, 1}}},
        GradSkipCase{
            .name = "Div_FrozenB",
            .make = [] { return binary(div_op, kActivation, kActivation, 1.F); },
            .frozen = {1},
            .skips = {{kUnary, 2}, {kBinary, 3}}},
        // The dtensor chain; normalized_tensor is still recomputed for dgamma.
        GradSkipCase{
            .name = "CompositeLayerNorm_FrozenTensor",
            .make = [] { return layernorm(ttml::ops::composite_layernorm, /*with_beta=*/true); },
            .frozen = {0},
            .skips = {{kBinary, 6}, {kMorehMean, 2}}},
        GradSkipCase{
            .name = "CompositeLayerNorm_FrozenGamma",
            .make = [] { return layernorm(ttml::ops::composite_layernorm, /*with_beta=*/true); },
            .frozen = {1},
            .skips = {{kBinary, 1}, {kMorehSum, 3}}},
        // Without beta there is no dbeta branch at all; gamma's skip is unchanged.
        GradSkipCase{
            .name = "CompositeLayerNormNoBeta_FrozenGamma",
            .make = [] { return layernorm(ttml::ops::composite_layernorm, /*with_beta=*/false); },
            .frozen = {1},
            .skips = {{kBinary, 1}, {kMorehSum, 3}}},
        GradSkipCase{
            .name = "CompositeLayerNorm_FrozenBeta",
            .make = [] { return layernorm(ttml::ops::composite_layernorm, /*with_beta=*/true); },
            .frozen = {2},
            .skips = {{kMorehSum, 3}}},
        // dx is skipped inside layernorm_bw, which is still launched for dgamma, so there is no op to count; the row
        // checks that the other grads are unchanged.
        GradSkipCase{
            .name = "LayerNorm_FrozenTensor",
            .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/true); },
            .frozen = {0},
            .changes_kernel_flags = true},
        // The kernel skips the frozen gamma's partials and the wrapper skips their reduction.
        GradSkipCase{
            .name = "LayerNorm_FrozenGamma",
            .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/true); },
            .frozen = {1},
            .skips = {{kReduce, 1}},
            .changes_kernel_flags = true},
        GradSkipCase{
            .name = "LayerNormNoBeta_FrozenGamma",
            .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/false); },
            .frozen = {1},
            .skips = {{kReduce, 1}},
            .changes_kernel_flags = true},
        // The wrapper reduces dbeta from dL_dout, so a frozen beta skips that reduction and leaves the kernel as is.
        GradSkipCase{
            .name = "LayerNorm_FrozenBeta",
            .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/true); },
            .frozen = {2},
            .skips = {{kReduce, 1}}},
        GradSkipCase{
            .name = "RMSNormComposite_FrozenTensor",
            .make = rmsnorm_composite,
            .frozen = {0},
            .skips = {{kBinary, 7}, {kMorehSum, 1}}},
        GradSkipCase{
            .name = "RMSNormComposite_FrozenGamma", .make = rmsnorm_composite, .frozen = {1}, .skips = {{kBinary, 2}}},
        // dx is skipped inside rmsnorm_bw, which is still launched for dgamma, so there is no op to count; the row
        // checks that gamma's grad is unchanged.
        GradSkipCase{.name = "RMSNorm_FrozenTensor", .make = rmsnorm, .frozen = {0}, .changes_kernel_flags = true},
        // The kernel skips the dgamma partials and the wrapper skips their reduction.
        GradSkipCase{
            .name = "RMSNorm_FrozenGamma",
            .make = rmsnorm,
            .frozen = {1},
            .skips = {{kReduce, 1}},
            .changes_kernel_flags = true},
        // dx = dL_dlinear1 @ w1 + dL_dgate @ w3.
        GradSkipCase{.name = "SwiGLU_FrozenX", .make = swiglu, .frozen = {0}, .skips = {{kMatmul, 2}}},
        GradSkipCase{.name = "SwiGLU_FrozenW1", .make = swiglu, .frozen = {1}, .skips = {{kMatmul, 1}}},
        GradSkipCase{.name = "SwiGLU_FrozenW2", .make = swiglu, .frozen = {2}, .skips = {{kMatmul, 1}}},
        GradSkipCase{.name = "SwiGLU_FrozenW3", .make = swiglu, .frozen = {3}, .skips = {{kMatmul, 1}}},
        // The shared dL_dprod matmul and swiglu_elemwise_bw, then dx (2 matmuls + add), dW1 and dW3.
        GradSkipCase{
            .name = "SwiGLU_OnlyW2",
            .make = swiglu,
            .frozen = {0, 1, 3},
            .skips = {{kMatmul, 5}, {kSwigluElemwiseBw, 1}}},
        GradSkipCase{
            .name = "SDPAComposite_FrozenQ",
            .make = [] { return attention(/*sigmoid=*/false); },
            .frozen = {0},
            .skips = {{kMatmul, 1}}},
        GradSkipCase{
            .name = "SDPAComposite_FrozenK",
            .make = [] { return attention(/*sigmoid=*/false); },
            .frozen = {1},
            .skips = {{kMatmul, 1}}},
        GradSkipCase{
            .name = "SDPAComposite_FrozenV",
            .make = [] { return attention(/*sigmoid=*/false); },
            .frozen = {2},
            .skips = {{kMatmul, 1}}},
        // dL/d(attention weights), the softmax backward and its scale, dQ and dK.
        GradSkipCase{
            .name = "SDPAComposite_OnlyV",
            .make = [] { return attention(/*sigmoid=*/false); },
            .frozen = {0, 1},
            .skips = {{kMatmul, 3}, {kMorehSoftmaxBw, 1}}},
        GradSkipCase{
            .name = "SigmoidSDPA_FrozenQ",
            .make = [] { return attention(/*sigmoid=*/true); },
            .frozen = {0},
            .skips = {{kMatmul, 1}}},
        GradSkipCase{
            .name = "SigmoidSDPA_FrozenK",
            .make = [] { return attention(/*sigmoid=*/true); },
            .frozen = {1},
            .skips = {{kMatmul, 1}}},
        GradSkipCase{
            .name = "SigmoidSDPA_FrozenV",
            .make = [] { return attention(/*sigmoid=*/true); },
            .frozen = {2},
            .skips = {{kMatmul, 1}}},
        // dL/d(attention weights), the sigmoid backward and its recomputed input, dQ and dK with their scales.
        GradSkipCase{
            .name = "SigmoidSDPA_OnlyV",
            .make = [] { return attention(/*sigmoid=*/true); },
            .frozen = {0, 1},
            .skips = {{kMatmul, 3}, {kUnaryBackward, 1}}},
        // As SigmoidSDPA_OnlyV, plus the where that masks the sigmoid backward (a BinaryNg op for a scalar `other`).
        GradSkipCase{
            .name = "SigmoidSDPAMasked_OnlyV",
            .make = [] { return attention(/*sigmoid=*/true, /*causal_mask=*/true); },
            .frozen = {0, 1},
            .skips = {{kMatmul, 3}, {kUnaryBackward, 1}}},
        // grad_Q is skipped inside the Q kernel, which still runs for u_scaler, so there is no op to count; the row
        // checks that grad_K and grad_V are unchanged.
        GradSkipCase{.name = "FusedSDPA_FrozenQ", .make = fused_attention, .frozen = {0}, .changes_kernel_flags = true},
        GradSkipCase{
            .name = "FusedSDPA_FrozenKV",
            .make = fused_attention,
            .frozen = {1, 2},
            .skips = {{kSdpaBwKV, 1}},
            .changes_kernel_flags = true},
        // Without a gate, grad_V alone needs nothing from the Q kernel.
        GradSkipCase{
            .name = "FusedSDPA_OnlyV",
            .make = fused_attention,
            .frozen = {0, 1},
            .skips = {{kSdpaBwQ, 1}},
            .changes_kernel_flags = true},
        GradSkipCase{
            .name = "GroupedHeads_FrozenQs", .make = grouped_heads, .frozen = {0}, .skips = {{kNlpConcatHeads, 1}}},
        GradSkipCase{
            .name = "GroupedHeads_FrozenKvs",
            .make = grouped_heads,
            .frozen = {1},
            .skips = {{kNlpConcatHeads, 2}, {kConcat, 1}}}),
    [](const ::testing::TestParamInfo<GradSkipCase>& info) { return info.param.name; });

// Grad functions read requires_grad when backward runs, so an op whose inputs were all frozen after forward still has
// its node called. It must launch no grad work, rather than ask the fused kernel for no grads at all.
class FrozenAfterForwardTest : public GradSkippingTest {};

TEST_P(FrozenAfterForwardTest, BackwardLaunchesNoGradWork) {
    const GradSkipCase& c = GetParam();
    const OpUnderTest op = c.make();
    const BackwardRun run =
        run_backward(op, std::vector<bool>(op.inputs.size(), true), /*freeze_before_backward=*/true);
    EXPECT_TRUE(count_device_ops(run.calls).empty())
        << c.name << ": backward launched device ops: " << fmt::format("{}", fmt::join(run.calls, ", "));
    for (size_t i = 0; i < run.grads.size(); ++i) {
        EXPECT_FALSE(run.grads[i].has_value()) << c.name << ": frozen input " << i << " got a grad";
    }
}

INSTANTIATE_TEST_SUITE_P(
    FrozenAfterForwardCases,
    FrozenAfterForwardTest,
    ::testing::Values(
        GradSkipCase{.name = "RMSNorm", .make = rmsnorm},
        GradSkipCase{.name = "LayerNorm", .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/true); }},
        GradSkipCase{
            .name = "LayerNormNoBeta", .make = [] { return layernorm(ttml::ops::layernorm, /*with_beta=*/false); }}),
    [](const ::testing::TestParamInfo<GradSkipCase>& info) { return info.param.name; });
