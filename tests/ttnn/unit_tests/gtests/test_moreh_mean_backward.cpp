// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program-spec tests to read compute kernel's compile-time args.
//
// The factory stores one need_bcast_dim entry per dimension of input_grad's logical shape.
// Entry 0 is the W broadcast flag. Entry 1 is the H broadcast flag.
// A rank-1 input_grad has only entry 0.
// The factory used to read entry 1 anyway (#58406 item 3).
// That read returned whatever value was left in the vector's inline storage.
//
// Running the op does not show the bug.
// The compute kernel only broadcasts along H when ht_need_bcast is exactly 1.
// A rank-1 tensor also has a single logical row. Every broadcast mode gives the same values for that row.
// Reading the compile-time arg is the only way to catch the bad read.
//
// A leftover value of 0 would hide the bug. The rank-1 test avoids that in two steps.
// First it builds a rank-2 case that broadcasts along H. That call writes 1 into entry 1.
// Then it builds the rank-1 case right away, from the same call site.
// In practice the second call's vector reuses the same stack memory. Entry 1 still holds the 1 from the first call.
// Without the fix, the rank-1 case reads that 1 and the test fails.
// The two calls must stay back to back. Code between them could overwrite the leftover 1.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/device/moreh_mean_backward_device_operation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::moreh::moreh_mean_backward::test {
namespace {

using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::experimental::KernelSpec;
using Op = MorehMeanBackwardOperation;

struct Case {
    ttnn::Shape output_grad_shape;
    ttnn::Shape input_grad_shape;
};

Tensor make_tensor(tt::tt_metal::distributed::MeshDevice* device, const ttnn::Shape& shape) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

std::vector<KernelSpec::CompileTimeArgs> compute_compile_time_args(
    tt::tt_metal::distributed::MeshDevice* device, const std::vector<Case>& cases) {
    // Create every tensor before the first factory call. The factory calls can then run back to back.
    std::vector<Tensor> output_grads;
    std::vector<Tensor> input_grads;
    for (const auto& c : cases) {
        output_grads.push_back(make_tensor(device, c.output_grad_shape));
        input_grads.push_back(make_tensor(device, c.input_grad_shape));
    }
    const std::vector<std::optional<Tensor>> input_grad_args(input_grads.begin(), input_grads.end());
    const Op::operation_attributes_t attributes{
        .dims = {0},
        .keepdim = true,
        .input_grad_shape = std::nullopt,
        .memory_config = ttnn::DRAM_MEMORY_CONFIG,
        .compute_kernel_config =
            init_device_compute_kernel_config(device->arch(), std::nullopt, tt::tt_metal::MathFidelity::HiFi4),
    };

    // Keep every result alive until all calls finish.
    // Destroying a result between two calls would run code that can overwrite the leftover stack values.
    std::vector<std::optional<ttnn::device_operation::ProgramArtifacts>> artifacts(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        artifacts[i].emplace(Op::MorehMeanBackwardProgramFactory::create_program_artifacts(
            attributes, {output_grads[i], input_grad_args[i]}, input_grads[i]));
    }

    std::vector<KernelSpec::CompileTimeArgs> result;
    for (const auto& artifact : artifacts) {
        KernelSpec::CompileTimeArgs args;
        bool found = false;
        for (const auto& kernel : artifact->spec.kernels) {
            if (*kernel.unique_id == "compute_g1") {
                args = kernel.compile_time_args;
                found = true;
            }
        }
        if (!found) {
            ADD_FAILURE() << "ProgramSpec has no compute_g1 kernel";
        }
        result.push_back(args);
    }
    return result;
}

uint32_t arg_value(const KernelSpec::CompileTimeArgs& args, const std::string& name) {
    const auto it = args.find(name);
    if (it == args.end()) {
        ADD_FAILURE() << "compute kernel has no compile-time arg " << name;
        return 0;
    }
    return it->second;
}

class MorehMeanBackwardProgramSpecFixture : public TTNNFixtureWithSuiteDevice<MorehMeanBackwardProgramSpecFixture> {};

TEST_F(MorehMeanBackwardProgramSpecFixture, Rank1InputGradHasNoHeightBroadcast) {
    // Case 0: output_grad [1, 5] broadcasts along H to input_grad [7, 5]. Its entry 1 is 1.
    // Case 1: output_grad [1] broadcasts along W to input_grad [5]. A rank-1 tensor has no H dimension.
    const auto args =
        compute_compile_time_args(device_, {{ttnn::Shape{1, 5}, ttnn::Shape{7, 5}}, {ttnn::Shape{1}, ttnn::Shape{5}}});
    ASSERT_EQ(args.size(), 2u);

    EXPECT_EQ(arg_value(args[0], "wt_need_bcast"), 0u);
    EXPECT_EQ(arg_value(args[0], "ht_need_bcast"), 1u);

    EXPECT_EQ(arg_value(args[1], "wt_need_bcast"), 1u);
    EXPECT_EQ(arg_value(args[1], "ht_need_bcast"), 0u);
}

}  // namespace
}  // namespace ttnn::operations::moreh::moreh_mean_backward::test
