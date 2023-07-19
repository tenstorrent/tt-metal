#pragma once

#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/operation_history.hpp"

#include <libs/tensor/tensor.hpp>

#include <optional>

namespace tt::tt_metal {

namespace operation {


std::vector<Tensor> generic_create_output_tensors(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const Layout output_layout = tt::tt_metal::Layout::TILE,
    const MemoryConfig& output_mem_config = MemoryConfig{.interleaved = true}
);
template<typename ConcreteOperation>
std::vector<Tensor> generic_create_output_tensors(
    const ConcreteOperation& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const Layout output_layout = tt::tt_metal::Layout::TILE,
    const MemoryConfig& output_mem_config = MemoryConfig{.interleaved = true}
) {
    if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto operation = DeviceOperation(concrete_op);
        return generic_create_output_tensors(operation, input_tensors, output_layout, output_mem_config);
    } else {
        static_assert(detail::always_false<ConcreteOperation>, "Unsupported Operation");
    }
}

#ifdef DEBUG
namespace detail {

template<typename OperationType>
static void print_operation(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {
    tt::log_debug(tt::LogOp, "Operation Type: {}", operation.get_type_name());
    tt::log_debug(tt::LogOp, "Operation Attributes: {}", operation.attributes());
    tt::log_debug(tt::LogOp, "Input Tensors: {}", input_tensors);
    if (not optional_input_tensors.empty()) {
        tt::log_debug(tt::LogOp, "Optional Input Tensors: {}", optional_input_tensors);
    }
}

static auto create_tensor_record(const Tensor& tensor) {
    return std::visit(
        [&] (const auto& storage) -> operation_history::TensorRecord {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, HostStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.shape(), tensor.dtype(), tensor.layout(), std::nullopt
                };
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.shape(), tensor.dtype(), tensor.layout(), storage.memory_config
                };
            }
        },
        tensor.storage()
    );
}

template<typename OperationType>
static void append_operation_to_operation_history(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {

    std::vector<operation_history::TensorRecord> input_tensor_records;
    input_tensor_records.reserve(input_tensors.size() + optional_input_tensors.size());

    for (const auto& tensor : input_tensors) {
        input_tensor_records.push_back(create_tensor_record(tensor));
    }
    for (const auto& tensor : optional_input_tensors) {
        if (tensor.has_value()) {
            input_tensor_records.push_back(create_tensor_record(tensor.value()));
        }
    }
    operation_history::append(operation_history::OperationRecord{operation.get_type_name(), operation.attributes(), input_tensor_records});
}

}  // namespace detail

template<typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {
    detail::print_operation(operation, input_tensors, optional_input_tensors);
    if (operation_history::enabled()) {
        detail::append_operation_to_operation_history(operation, input_tensors, optional_input_tensors);
    }
}
#else
template<typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {}
#endif

std::vector<Tensor> run(
    const HostOperation& operation,
    const std::vector<Tensor>& input_tensors
);
std::vector<Tensor> run(
    const DeviceOperation& operation,
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
        const auto operation = HostOperation(concrete_op);
        return run(operation, input_tensors);
    } else if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto operation = DeviceOperation(concrete_op);
        return run(operation, input_tensors, optional_input_tensors);
    } else {
        static_assert(detail::always_false<ConcreteOperation>, "Unsupported Operation");
    }
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run_without_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
) {
    const auto operation = DeviceOperation(concrete_op);
    return run_without_autoformat(operation, input_tensors, optional_input_tensors);
}

std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
) {
    const auto operation = DeviceOperation(concrete_op);
    return run_with_autoformat(operation, input_tensors, optional_input_tensors, pad_value, pad_c);
}

}

}
