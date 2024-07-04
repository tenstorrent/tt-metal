// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/pad/pad_op.hpp"

#include "tensor/host_buffer/functions.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

void PadOnHost::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() != StorageType::DEVICE);
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_FATAL(input_tensor.get_legacy_shape()[0] + this->input_tensor_start[0] <= this->output_tensor_shape[0], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_legacy_shape()[1] + this->input_tensor_start[1] <= this->output_tensor_shape[1], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_legacy_shape()[2] + this->input_tensor_start[2] <= this->output_tensor_shape[2], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_legacy_shape()[3] + this->input_tensor_start[3] <= this->output_tensor_shape[3], "Output size cannot fit input with offset");
}

std::vector<Shape> PadOnHost::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto input_shape = input_tensors.at(0).get_legacy_shape();
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape.rank(); index++) {
        auto front = this->input_tensor_start[index];
        auto back = this->output_tensor_shape[index] - (this->input_tensor_start[index] + input_shape[index]);
        dimensions_pads.push_back(Padding::PadDimension{.front=front, .back=back});
    }
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    return {Shape(this->output_tensor_shape, padding)};
}

std::vector<Tensor> PadOnHost::compute_output_tensors(const std::vector<Tensor>& input_tensors) const {
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.get_legacy_shape() == this->output_tensor_shape) {
        return {input_tensor};
    } else {
        return {input_tensor.pad(output_shape, this->input_tensor_start, this->pad_value)};
    }
}

Tensor pad_on_host(const Tensor &input_tensor, const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value) {
    return operation::run(PadOnHost{output_tensor_shape, input_tensor_start, pad_value}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
