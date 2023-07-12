#pragma once

#include "tt_dnn/op_library/operation.hpp"

#include <libs/tensor/tensor.hpp>

#include <optional>

namespace tt::tt_metal {

namespace operation {


std::vector<Tensor> generic_create_output_tensors(
    const DeviceOperation& op,
    const std::vector<Tensor>& input_tensors,
    const Layout output_layout = tt::tt_metal::Layout::TILE,
    const MemoryConfig &output_mem_config = MemoryConfig{.interleaved = true}
);
template<typename ConcreteOperation>
std::vector<Tensor> generic_create_output_tensors(
    const ConcreteOperation& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const Layout output_layout = tt::tt_metal::Layout::TILE,
    const MemoryConfig &output_mem_config = MemoryConfig{.interleaved = true}
) {
    if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto op = DeviceOperation(concrete_op);
        return generic_create_output_tensors(op, input_tensors, output_layout, output_mem_config);
    } else {
        static_assert(detail::always_false<ConcreteOperation>, "Unsupported Operation");
    }
}

std::vector<Tensor> run(
    const HostOperation& op,
    const std::vector<Tensor>& input_tensors
);
std::vector<Tensor> run(
    const DeviceOperation& op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
) {
    if constexpr (detail::is_host_operation<ConcreteOperation>()) {
        TT_ASSERT(optional_input_tensors.empty());
        const auto op = HostOperation(concrete_op);
        return run(op, input_tensors);
    } else if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto op = DeviceOperation(concrete_op);
        return run(op, input_tensors, optional_input_tensors);
    } else {
        static_assert(detail::always_false<ConcreteOperation>, "Unsupported Operation");
    }
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run_without_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
) {
    const auto op = DeviceOperation(concrete_op);
    return run_without_autoformat(op, input_tensors, optional_input_tensors);
}

std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
) {
    const auto op = DeviceOperation(concrete_op);
    return run_with_autoformat(op, input_tensors, optional_input_tensors, pad_value, pad_c);
}


Hash hash_tensor(const Tensor& tensor);
Hash hash_memory_config(const MemoryConfig& memory_config);

}

}
