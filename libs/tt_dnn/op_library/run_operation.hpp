#pragma once

#include "tt_dnn/op_library/operation.hpp"

#include <libs/tensor/tensor.hpp>

#include <optional>

namespace tt::tt_metal {

namespace operation {

std::vector<Tensor> run(
    const Operation& op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
);

std::vector<Tensor> generic_create_output_tensors(
    const Operation& op,
    const std::vector<Tensor>& input_tensors,
    Layout output_layout = tt::tt_metal::Layout::TILE,
    const MemoryConfig &output_mem_config = MemoryConfig{.interleaved = true}
);

Tensor run_without_autoformat(const Operation& op, const Tensor &input_tensor);
template<typename ConcreteOperation>
Tensor run_without_autoformat(ConcreteOperation&& concrete_op, const Tensor &input_tensor) {
    const Operation op = Operation(concrete_op);
    return run_without_autoformat(op, input_tensor);
}

Tensor run_with_autoformat(const Operation& op, const Tensor &input_tensor, float pad_value = 0, bool pad_c=false);
template<typename ConcreteOperation>
Tensor run_with_autoformat(ConcreteOperation&& concrete_op, const Tensor &input_tensor, float pad_value = 0, bool pad_c=false) {
    const Operation op = Operation(concrete_op);
    return run_with_autoformat(op, input_tensor, pad_value, pad_c);
}

Tensor run_with_autoformat(const Operation& op, const Tensor &input_tensor_a, const Tensor &input_tensor_b, float pad_value = 0);
template<typename ConcreteOperation>
Tensor run_with_autoformat(ConcreteOperation&& concrete_op, const Tensor &input_tensor_a, const Tensor &input_tensor_b, float pad_value = 0) {
    const Operation op = Operation(concrete_op);
    return run_with_autoformat(op, input_tensor_a, input_tensor_b, pad_value);
}

Hash hash_tensor(const Tensor& tensor);
Hash hash_memory_config(const MemoryConfig& memory_config);

}

}
