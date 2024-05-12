// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"

#include "tt_dnn/op_library/moreh_dot/moreh_dot_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         Util
////////////////////////////////////////////////////////////////////////////
namespace {
inline bool is_dot_forward(const Tensor& input, const Tensor& other, bool transpose_input, bool transpose_other) {
    // TODO: non-4d support for dot.
    if (input.get_legacy_shape().rank() != 4 || other.get_legacy_shape().rank() != 4) {
        return false;
    }

    if (transpose_input || transpose_other) {
        return false;
    }

    return is_1d_tensor(input) && is_1d_tensor(other) && is_same_shape(input, other);
}

inline void moreh_matmul_validate(
    const Tensor& input_tensor, const Tensor& other_tensor, bool transpose_input, bool transpose_other) {
    const auto& input_shape = input_tensor.get_legacy_shape().without_padding();
    const auto& other_shape = other_tensor.get_legacy_shape().without_padding();
    // check dim-1
    TT_ASSERT(
        (input_shape[1] == other_shape[1]) || input_shape[1] == 1 || other_shape[1] == 1,
        "The size of tensor a must match the size of tensor b at non-singleton dimension 1");

    // check dim-0
    TT_ASSERT(
        (input_shape[0] == other_shape[0]) || input_shape[0] == 1 || other_shape[0] == 1,
        "The size of tensor a must match the size of tensor b at non-singleton dimension 0");

    // only one tensor can be tranposed
    const auto& input_k = (transpose_input) ? (input_shape[2]) : (input_shape[3]);
    const auto& other_k = (transpose_other) ? (other_shape[3]) : (other_shape[2]);
    TT_ASSERT(input_k == other_k && "Dimension K must match for A and B in matmul op");
}

}  // namespace

operation::ProgramWithCallbacks MorehMatmul::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& other_tensor = input_tensors.at(1);
    const auto& output_tensor = output_tensors.at(0);
    const auto& bias_tensor = optional_input_tensors.at(0);
    return moreh_matmul_multi_core(
        input_tensor,
        other_tensor,
        output_tensor,
        bias_tensor,
        this->transpose_input,
        this->transpose_other);
}

inline Shape compute_output_shape(
    const Shape& input_shape, const Shape& other_shape, bool transpose_input, bool transpose_other) {
    const auto& input_shape_wo_padding = input_shape.without_padding();
    const auto& other_shape_wo_padding = other_shape.without_padding();

    auto h = (transpose_input) ? (input_shape[3]) : (input_shape[2]);
    auto w = (transpose_other) ? (other_shape[2]) : (other_shape[3]);
    auto h_wo_padding = (transpose_input) ? (input_shape_wo_padding[3]) : (input_shape_wo_padding[2]);
    auto w_wo_padding = (transpose_other) ? (other_shape_wo_padding[2]) : (other_shape_wo_padding[3]);

    Shape output_shape{std::max(input_shape[0], other_shape[0]), std::max(input_shape[1], other_shape[1]), h, w};
    auto padding = output_shape.padding();
    padding[2] = Padding::PadDimension{0, h - h_wo_padding};
    padding[3] = Padding::PadDimension{0, w - w_wo_padding};

    return {Shape(output_shape, padding)};
}

// Must be provided in the case where an optional output tensor was not provided
std::vector<Shape> MorehMatmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {compute_output_shape(
        input_tensors.at(0).get_legacy_shape(),
        input_tensors.at(1).get_legacy_shape(),
        this->transpose_input,
        this->transpose_other)};
}

std::vector<Tensor> MorehMatmul::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

void MorehMatmul::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    log_debug(LogOp, "{}:{}", __func__, __LINE__);
    // TODO
}

const operation::Hash MorehMatmul::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& other = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);

    operation::Hash hash = tt::stl::hash::hash_objects(
        0,
        typeid(*this).hash_code(),
        input,
        other,
        bias,
        this->transpose_input,
        this->transpose_other);
    return hash;
}

Tensor moreh_matmul_(
    const Tensor& input,
    const Tensor& other,
    bool transpose_input,
    bool transpose_other,
    const std::optional<Tensor> &output,
    const std::optional<Tensor> &bias,
    const MemoryConfig& output_mem_config) {
        log_debug(
            LogOp,
            "{}:{} run matmul {} {}",
            __func__,
            __LINE__,
            transpose_input,
            transpose_other);
        return operation::run(
            MorehMatmul{
                .output_mem_config = output_mem_config,
                .transpose_input = transpose_input,
                .transpose_other = transpose_other },
            { input, other},
            { bias},
            { output})
            .at(0);
}

Tensor moreh_matmul(
    const Tensor& input,
    const Tensor& other,
    bool transpose_input,
    bool transpose_other,
    const std::optional<const Tensor> output,
    const std::optional<const Tensor> bias,
    const MemoryConfig& output_mem_config) {

    // TODO(seunghwan100): Add the argument "output_tensor" to moreh_dot.
    if (is_dot_forward(input, other, transpose_input, transpose_other)) {
        TT_ASSERT(!bias.has_value());
        return moreh_dot(input, other, output_mem_config);
    }
    return moreh_matmul_(input, other, transpose_input, transpose_other, output, bias, output_mem_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
