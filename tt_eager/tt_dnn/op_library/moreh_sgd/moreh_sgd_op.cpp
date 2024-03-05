// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_sgd/moreh_sgd_op.hpp"

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehSGD::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 3, "Must have 1 input tensors");
    TT_ASSERT(
        optional_input_tensors.size() == 0 || optional_input_tensors.size() == 2, "Must have 0 or 2 optional tensors");

    auto& param_in = input_tensors.at(0);
    auto& grad = input_tensors.at(1);
    auto& param_out = input_tensors.at(2);

    TT_ASSERT(param_in.storage_type() == StorageType::DEVICE, "param_in to SGD need to be on device!");
    TT_ASSERT(grad.storage_type() == StorageType::DEVICE, "grad to SGD need to be on device!");
    TT_ASSERT(param_out.storage_type() == StorageType::DEVICE, "param_out to SGD need to be on device!");

    TT_ASSERT(param_in.buffer() != nullptr, "param_in to SGD need to be allocated in buffers on device!");
    TT_ASSERT(grad.buffer() != nullptr, "grad to SGD need to be allocated in buffers on device!");
    TT_ASSERT(param_out.buffer() != nullptr, "param_out to SGD need to be allocated in buffers on device!");

    TT_ASSERT((param_in.get_layout() == Layout::TILE), "param_in to SGD must be tilized");
    TT_ASSERT((grad.get_layout() == Layout::TILE), "grad to SGD must be tilized");
    TT_ASSERT((param_out.get_layout() == Layout::TILE), "param_out to SGD must be tilized");

    TT_ASSERT(param_in.get_dtype() == DataType::BFLOAT16 || param_in.get_dtype() == DataType::BFLOAT8_B);
    TT_ASSERT(grad.get_dtype() == DataType::BFLOAT16 || grad.get_dtype() == DataType::BFLOAT8_B);
    TT_ASSERT(param_out.get_dtype() == DataType::BFLOAT16 || param_out.get_dtype() == DataType::BFLOAT8_B);
}

std::vector<Shape> MorehSGD::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

std::vector<Tensor> MorehSGD::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

operation::ProgramWithCallbacks MorehSGD::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& param_in = input_tensors.at(0);
    auto& grad = input_tensors.at(1);
    auto& momentum_buffer_in = optional_input_tensors.at(0);
    auto& param_out = input_tensors.at(2);
    auto& momentum_buffer_out = optional_input_tensors.at(1);

    return {moreh_sgd_(
        param_in,
        grad,
        momentum_buffer_in,
        param_out,
        momentum_buffer_out,
        this->lr,
        this->momentum,
        this->dampening,
        this->weight_decay,
        this->nesterov,
        this->momentum_initialized,
        this->core_range)};
}

void moreh_sgd(
    const Tensor& param_in,
    const Tensor& grad,
    std::optional<const Tensor> momentum_buffer_in,
    const Tensor& param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized) {
    auto device = param_in.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    operation::run(
        MorehSGD{
            .lr = lr,
            .momentum = momentum,
            .dampening = dampening,
            .weight_decay = weight_decay,
            .nesterov = nesterov,
            .momentum_initialized = momentum_initialized,
            .core_range = all_cores},
        {param_in, grad, param_out},
        {momentum_buffer_in, momentum_buffer_out});
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
