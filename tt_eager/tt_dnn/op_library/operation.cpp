// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/operation.hpp"

#include "ttnn/validation.hpp"

namespace tt {

namespace tt_metal {
namespace operation {

void validate_input_tensors(
    const std::string& operation_name,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const std::vector<ttnn::TensorSchema>& input_schemas) {
    tt::log_debug(tt::LogOp, "Validating input tensors of operation: {}", operation_name);
    if (input_tensors.size() + optional_input_tensors.size() != input_schemas.size()) {
        TT_THROW(
            "{}: Number of input tensors ({}) does not match the number of input schemas ({})",
            operation_name,
            input_tensors.size() + optional_input_tensors.size(),
            input_schemas.size());
    }
    auto schema_index = 0;
    for (auto tensor_index = 0; tensor_index < input_tensors.size(); tensor_index++) {
        const auto& input_tensor = input_tensors.at(tensor_index);
        ttnn::validate_input_tensor(operation_name, input_tensor, input_schemas[schema_index++]);
    }
    for (auto tensor_index = 0; tensor_index < optional_input_tensors.size(); tensor_index++) {
        const auto& input_tensor = optional_input_tensors.at(tensor_index);
        ttnn::validate_input_tensor(operation_name, input_tensor, input_schemas[schema_index++]);
    }
}

}  // namespace operation
}  // namespace tt_metal
}  // namespace tt
