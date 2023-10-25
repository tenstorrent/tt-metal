// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"

#include "tt_dnn/op_library/moreh_dot/moreh_dot_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

Tensor moreh_matmul(const Tensor& input_tensor, const Tensor& other_tensor, const MemoryConfig& mem_config) {
    return tt::operations::primary::moreh_matmul(input_tensor, other_tensor, false, false, mem_config);
}

}  // namespace tt_metal

namespace operations {
namespace primary {

void MorehMatmul::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_ASSERT(
        (input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE),
        "Inputs to matmul must be tilized");

    TT_ASSERT(
        input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_ASSERT(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_ASSERT(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
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
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    // inplace
    const auto& output_tensor = input_tensors.at(2);
    // TODO: add optimized matmul
    return moreh_matmul_multi_core(
        input_tensor_a,
        input_tensor_b,
        output_tensor,
        this->transpose_a,
        this->transpose_b,
        this->a_start_tile_id,
        this->b_start_tile_id,
        this->output_start_tile_id);
}

tt::stl::reflection::Attributes MorehMatmul::attributes() const {
    return {
        {"transpose_a", this->transpose_a},
        {"transpose_b", this->transpose_b},
    };
}

inline bool is_1d_tensor(const Tensor& tensor) {
    const auto& shape = tensor.shape().without_padding();
    // because TT Tensor only support 4d shape, so if the first 3 dims are 1, assume it's 1d.
    return shape[0] == 1 && shape[1] == 1 && shape[2] == 1;
}

inline void moreh_matmul_validate(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, bool transpose_a, bool transpose_b) {
    const auto& a_shape = input_tensor_a.shape();
    const auto& b_shape = input_tensor_b.shape();
    // check dim-1
    TT_ASSERT(
        (a_shape[1] == b_shape[1]) || a_shape[1] == 1 || b_shape[1] == 1,
        "The size of tensor a must match the size of tensor b at non-singleton dimension 1");

    // check dim-0
    TT_ASSERT(
        (a_shape[0] == b_shape[0]) || a_shape[0] == 1 || b_shape[0] == 1,
        "The size of tensor a must match the size of tensor b at non-singleton dimension 0");

    // check matrix shape
    const auto& a_shape_wo_padding = input_tensor_a.shape().without_padding();
    const auto& b_shape_wo_padding = input_tensor_b.shape().without_padding();

    // only one tensor can be tranposed
    const auto& a_k = (transpose_a) ? (a_shape_wo_padding[2]) : (a_shape_wo_padding[3]);
    const auto& b_k = (transpose_b) ? (b_shape_wo_padding[3]) : (b_shape_wo_padding[2]);
    TT_ASSERT(a_k == b_k && "Dimension K must match for A and B in matmul op");
}

inline Shape compute_output_shape(const Shape& a_shape, const Shape& b_shape, bool transpose_a, bool transpose_b) {
    Shape output_shape{
        std::max(a_shape[0], b_shape[0]),
        std::max(a_shape[1], b_shape[1]),
        (transpose_a) ? (a_shape[3]) : (a_shape[2]),
        (transpose_b) ? (b_shape[2]) : (b_shape[3])};
    return output_shape;
}

inline Tensor create_output_tensor(
    const Tensor& input_tensor, const Shape& output_shape, const MemoryConfig& mem_config) {
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE);
    return create_device_tensor(output_shape, input_tensor.dtype(), Layout::TILE, input_tensor.device(), mem_config);
}

Tensor moreh_matmul_(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a,
    bool transpose_b,
    const MemoryConfig& mem_config) {
    moreh_matmul_validate(input_tensor_a, input_tensor_b, transpose_a, transpose_b);

    const auto& a_shape = input_tensor_a.shape();
    const auto& b_shape = input_tensor_b.shape();
    const auto& output_shape = compute_output_shape(a_shape, b_shape, transpose_a, transpose_b);
    auto output_tensor = create_output_tensor(input_tensor_a, output_shape, mem_config);

    uint32_t a_B2MtKt = a_shape[1] * a_shape[2] * a_shape[3];
    uint32_t b_B2KtNt = b_shape[1] * b_shape[2] * b_shape[3];
    uint32_t output_B2MtNt = output_shape[1] * output_shape[2] * output_shape[3];
    for (uint32_t b1 = 0; b1 < output_shape[0]; ++b1) {
        uint32_t a_b1_index = (b1 >= a_shape[0]) ? (0) : (b1);
        uint32_t b_b1_index = (b1 >= b_shape[0]) ? (0) : (b1);

        uint32_t a_start_tile_id = a_b1_index * a_B2MtKt / TILE_HW;
        uint32_t b_start_tile_id = b_b1_index * b_B2KtNt / TILE_HW;
        uint32_t output_start_tile_id = b1 * output_B2MtNt / TILE_HW;

        operation::run(
            MorehMatmul{
                .transpose_a = transpose_a,
                .transpose_b = transpose_b,
                .a_start_tile_id = a_start_tile_id,
                .b_start_tile_id = b_start_tile_id,
                .output_start_tile_id = output_start_tile_id},
            {input_tensor_a, input_tensor_b, output_tensor});
    }

    return output_tensor;
}

Tensor moreh_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a,
    bool transpose_b,
    const MemoryConfig& mem_config) {
    // 1d x 1d
    if (is_1d_tensor(input_tensor_a) && is_1d_tensor(input_tensor_b)) {
        TT_ASSERT(transpose_a == false);
        TT_ASSERT(transpose_b == false);
        return moreh_dot(input_tensor_a, input_tensor_b, mem_config);
    }

    return moreh_matmul_(input_tensor_a, input_tensor_b, transpose_a, transpose_b, mem_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
