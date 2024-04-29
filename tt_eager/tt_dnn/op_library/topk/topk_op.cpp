// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/topk/topk_op.hpp"
#include "tt_metal/host_api.hpp"

void TopK::validate(const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors.at(0).get_legacy_shape();
    TT_FATAL(k == 32, fmt::format("K must be equal to 32, pad with -infinity if necessary"));
    TT_FATAL(input_shape[-1] == 64, fmt::format("Input shape inner dim {} must be 64, pad with -infinity if necessary", input_shape[-1]));
    TT_FATAL((input_shape[0] * input_shape[1] * input_shape[2]) == 32, "Input height must be 32");
    TT_FATAL(this->output_mem_config.is_sharded() == false, "Sharded implementation not supported yet");
    TT_FATAL(input_tensors.at(0).get_layout() == Layout::TILE, "The input must be in tiled format");
}

std::vector<Shape> TopK::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    return {{input_shape[0], input_shape[1], input_shape[2], this->k}, {input_shape[0], input_shape[1], input_shape[2], this->k}};
}

std::vector<Tensor> TopK::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto shapes = compute_output_shapes(input_tensors);
    auto values_tensor = create_device_tensor(shapes[0], input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), this->output_mem_config);
    auto index_tensor = create_device_tensor(shapes[0], DataType::UINT16, Layout::TILE, input_tensor.device(), this->output_mem_config);
    return {values_tensor, index_tensor};
}

operation::ProgramWithCallbacks TopK::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return single_core_topk_interleaved(input_tensor, this->k, output_tensors.at(0), output_tensors.at(1));
}

tt::stl::reflection::Attributes TopK::attributes() const {
    return {
        {"k", this->k},
        {"output_mem_config", this->output_mem_config},
    };
}
