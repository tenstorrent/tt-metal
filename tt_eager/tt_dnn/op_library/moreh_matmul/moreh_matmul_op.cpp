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
namespace tt_metal {

Tensor moreh_matmul(const Tensor& input_tensor, const Tensor& other_tensor, const MemoryConfig& mem_config) {
    return operations::primary::moreh_matmul(input_tensor, other_tensor, std::nullopt, false, false, mem_config);
}

}  // namespace tt_metal

namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         Util
////////////////////////////////////////////////////////////////////////////
inline bool is_dot_forward(const Tensor& input, const Tensor& other) {
    return is_1d_tensor(input) && is_1d_tensor(other) && is_same_shape(input, other);
}

void MorehMatmul::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& other_tensor = input_tensors.at(1);
    TT_ASSERT(
        (input_tensor.layout() == Layout::TILE && other_tensor.layout() == Layout::TILE),
        "Inputs to matmul must be tilized");

    TT_ASSERT(
        input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_ASSERT(
        input_tensor.storage_type() == StorageType::DEVICE and other_tensor.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_ASSERT(input_tensor.device() == other_tensor.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(
        input_tensor.buffer() != nullptr and other_tensor.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
}

std::vector<Shape> MorehMatmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // inplace
    return {};
}

std::vector<Tensor> MorehMatmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // inplace
    return {};
}

operation::ProgramWithCallbacks MorehMatmul::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& other_tensor = input_tensors.at(1);
    // inplace
    const auto& output_tensor = input_tensors.at(2);
    // TODO: add optimized matmul
    return moreh_matmul_multi_core(
        input_tensor,
        other_tensor,
        output_tensor,
        this->transpose_input,
        this->transpose_other,
        this->input_start_tile_id,
        this->other_start_tile_id,
        this->output_start_tile_id);
}

inline void moreh_matmul_validate(
    const Tensor& input_tensor, const Tensor& other_tensor, bool transpose_input, bool transpose_other) {
    const auto& input_shape = input_tensor.shape().without_padding();
    const auto& other_shape = other_tensor.shape().without_padding();
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

inline Tensor create_output_tensor(
    const Tensor& input_tensor, const Shape& output_shape, const MemoryConfig& mem_config) {
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE);
    return create_device_tensor(output_shape, input_tensor.dtype(), Layout::TILE, input_tensor.device(), mem_config);
}

Tensor moreh_matmul_(
    const Tensor& input_tensor,
    const Tensor& other_tensor,
    std::optional<std::reference_wrapper<const Tensor>> output_tensor,
    bool transpose_input,
    bool transpose_other,
    const MemoryConfig& mem_config) {
    moreh_matmul_validate(input_tensor, other_tensor, transpose_input, transpose_other);

    const auto& input_shape = input_tensor.shape();
    const auto& other_shape = other_tensor.shape();
    const auto& output_shape = compute_output_shape(input_shape, other_shape, transpose_input, transpose_other);
    auto output_tensor_select =
        (output_tensor) ? (output_tensor->get()) : (create_output_tensor(input_tensor, output_shape, mem_config));

    uint32_t input_other2MtKt = input_shape[1] * input_shape[2] * input_shape[3];
    uint32_t other_other2KtNt = other_shape[1] * other_shape[2] * other_shape[3];
    uint32_t output_other2MtNt = output_shape[1] * output_shape[2] * output_shape[3];
    for (uint32_t b1 = 0; b1 < output_shape[0]; ++b1) {
        uint32_t input_other1_index = (b1 >= input_shape[0]) ? (0) : (b1);
        uint32_t other_other1_index = (b1 >= other_shape[0]) ? (0) : (b1);

        uint32_t input_start_tile_id = input_other1_index * input_other2MtKt / TILE_HW;
        uint32_t other_start_tile_id = other_other1_index * other_other2KtNt / TILE_HW;
        uint32_t output_start_tile_id = b1 * output_other2MtNt / TILE_HW;

        operation::run(
            MorehMatmul{
                .transpose_input = transpose_input,
                .transpose_other = transpose_other,
                .input_start_tile_id = input_start_tile_id,
                .other_start_tile_id = other_start_tile_id,
                .output_start_tile_id = output_start_tile_id},
            {input_tensor, other_tensor, output_tensor_select});
    }

    return output_tensor_select;
}

Tensor moreh_matmul(
    const Tensor& input_tensor,
    const Tensor& other_tensor,
    std::optional<std::reference_wrapper<const Tensor>> output_tensor,
    bool transpose_input,
    bool transpose_other,
    const MemoryConfig& mem_config) {
    if (is_dot_forward(input_tensor, other_tensor) && (!transpose_input && !transpose_other)) {
        return moreh_dot(input_tensor, other_tensor, mem_config);
    }
    return moreh_matmul_(input_tensor, other_tensor, output_tensor, transpose_input, transpose_other, mem_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
