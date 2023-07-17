#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/operation_history.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_numpy/functions.hpp"

#include "tt_stl/reflection.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt::tt_metal::operation {

namespace detail {

static Device* get_device(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            return input_tensor.device();
        }
    }
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() == StorageType::DEVICE) {
            return optional_input_tensor.value().device();
        }
    }
    auto device = AutoFormat::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to operation are on device");
    return device;
}

void override_runtime_args(
    const OverrideRuntimeArgsCallback& override_runtime_args_callback,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<Tensor>& output_tensors
) {
    std::vector<Buffer*> input_buffers;
    for (auto& tensor : input_tensors) {
        input_buffers.push_back(tensor.buffer());
    }
    for (auto& tensor : optional_input_tensors) {
        auto buffer = tensor.has_value() ? tensor.value().buffer() : nullptr;
        input_buffers.push_back(buffer);
    }

    std::vector<Buffer*> output_buffers;
    for (auto& tensor : output_tensors) {
        output_buffers.push_back(tensor.buffer());
    }

    override_runtime_args_callback(input_buffers, output_buffers);
}

void setup_profiler(const HostOperation& operation, const std::vector<Tensor>& input_tensors) {
    auto profiler_info = operation.create_profiler_info(input_tensors);
    if (profiler_info.preferred_name.has_value()) {
        op_profiler::set_preferred_name(profiler_info.preferred_name.value());
    }
}

void setup_profiler(const DeviceOperation& operation, const std::vector<Tensor>& input_tensors) {
    auto profiler_info = operation.create_profiler_info(input_tensors);
    if (profiler_info.preferred_name.has_value()) {
        op_profiler::set_preferred_name(profiler_info.preferred_name.value());
    }
    if (profiler_info.parallelization_strategy.has_value()) {
        op_profiler::set_parallelization_strategy(profiler_info.parallelization_strategy.value());
    }
}

std::vector<Tensor> run_without_program_cache(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {

    auto profile_scope = op_profiler::ProfileScope(operation.get_type_name(), op_profiler::OpType::tt_dnn_device);
    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) { setup_profiler(operation, input_tensors); }

    operation.validate(input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors, optional_input_tensors);
    auto output_tensors = operation.create_output_tensors(input_tensors);

    auto program_with_callbacks = operation.create_program(input_tensors, optional_input_tensors, output_tensors);
    auto& program = program_with_callbacks.program;

    CompileProgram(device, program, do_profile);
    const char *TT_METAL_DEVICE_DISPATCH_MODE = std::getenv("TT_METAL_DEVICE_DISPATCH_MODE");
    if (TT_METAL_DEVICE_DISPATCH_MODE != nullptr) {
        EnqueueProgram(*HACK_CQ, program, false);

    } else {
        ConfigureDeviceWithProgram(device, program);
        WriteRuntimeArgsToDevice(device, program);
        LaunchKernels(device, program);
    }

    op_profiler::append_all_tensor_io_data(input_tensors, optional_input_tensors, output_tensors);
    op_profiler::dump_device_profiler_results(device, program);

    return output_tensors;
}

std::vector<Tensor> run_with_program_cache(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {

    auto profile_scope = op_profiler::ProfileScope(operation.get_type_name(), op_profiler::OpType::tt_dnn_device);
    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) { setup_profiler(operation, input_tensors); }

    operation.validate(input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors, optional_input_tensors);
    auto output_tensors = operation.create_output_tensors(input_tensors);

    auto& program_with_callbacks = program_cache::get_or_create(
        operation, input_tensors, optional_input_tensors, output_tensors, device, do_profile
    );

    override_runtime_args(
        program_with_callbacks.override_runtime_args_callback,
        input_tensors, optional_input_tensors, output_tensors
    );

    auto& program = program_with_callbacks.program;

    const char *TT_METAL_DEVICE_DISPATCH_MODE = std::getenv("TT_METAL_DEVICE_DISPATCH_MODE");
    if (TT_METAL_DEVICE_DISPATCH_MODE != nullptr) {
        EnqueueProgram(*HACK_CQ, program, false);

    } else {
        ConfigureDeviceWithProgram(device, program);
        WriteRuntimeArgsToDevice(device, program);
        LaunchKernels(device, program);
    }

    op_profiler::append_all_tensor_io_data(input_tensors, optional_input_tensors, output_tensors);
    op_profiler::dump_device_profiler_results(device, program);

    return output_tensors;
}

}

std::vector<Tensor> generic_create_output_tensors(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const Layout output_layout,
    const MemoryConfig& output_mem_config
) {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_shapes = operation.compute_output_shapes(input_tensors);

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE);

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(create_device_tensor(output_shape, input_tensor.dtype(), output_layout, input_tensor.device(), output_mem_config));
    }
    return output_tensors;
}

namespace detail {

#ifdef DEBUG

template<typename OperationType>
void print_operation(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {
    tt::log_debug(tt::LogOp, "Operation: {}", operation);
    tt::log_debug(tt::LogOp, "Input Tensors: {}", input_tensors);
    if (not optional_input_tensors.empty()) {
        tt::log_debug(tt::LogOp, "Optional Input Tensors: {}", optional_input_tensors);
    }
}

auto create_tensor_record(const Tensor& tensor) {
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
void append_operation_to_operation_history(
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
template<typename OperationType>
inline void run_common(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {
    print_operation(operation, input_tensors, optional_input_tensors);
    append_operation_to_operation_history(operation, input_tensors, optional_input_tensors);
}
#else
template<typename OperationType>
inline void run_common(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {}
#endif

}  // namespace detail

std::vector<Tensor> run(
    const HostOperation& operation,
    const std::vector<Tensor>& input_tensors
) {
    detail::run_common(operation, input_tensors);

    auto profile_scope = op_profiler::ProfileScope(operation.get_type_name(), op_profiler::OpType::tt_dnn_cpu);
    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) { detail::setup_profiler(operation, input_tensors); }

    operation.validate(input_tensors);
    auto output_tensors = operation.compute_output_tensors(input_tensors);

    op_profiler::append_all_tensor_io_data(input_tensors, {}, output_tensors);

    return output_tensors;
}

namespace detail {
void print_warning_if_operation_does_not_support_program_cache(const DeviceOperation& operation) {
#ifdef DEBUG
    if (program_cache::is_enabled()) {
        tt::log_warning(tt::LogOp, "Running {} operation without program cache", operation.get_type_name());
    }
#endif
}
}  // namespace detail

std::vector<Tensor> run(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors
) {
    detail::run_common(operation, input_tensors, optional_input_tensors);
    if (program_cache::is_enabled() and operation.supports_program_caching()) {
        return detail::run_with_program_cache(operation, input_tensors, optional_input_tensors);
    } else {
        detail::print_warning_if_operation_does_not_support_program_cache(operation);
        return detail::run_without_program_cache(operation, input_tensors, optional_input_tensors);
    }
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors
) {
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

    std::vector<Tensor> input_tensors_on_dev;
    input_tensors_on_dev.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() == StorageType::HOST) {
            input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(input_tensor, device));
        } else {
            input_tensors_on_dev.push_back(input_tensor);
        }
    }
    std::vector<std::optional<const Tensor>> optional_input_tensors_on_dev;
    optional_input_tensors_on_dev.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() == StorageType::HOST) {
            optional_input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run(operation, input_tensors_on_dev, optional_input_tensors_on_dev);
}


std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const float pad_value,
    const bool pad_c
) {
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

    auto output_shapes = operation.compute_output_shapes(input_tensors);

    std::vector<Tensor> formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape(), pad_c);
        auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
        if (pad_input) {
            formatted_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value));
        } else {
            formatted_input_tensors.push_back(input_tensor);
        }
    }

    std::vector<std::optional<const Tensor>> formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value()) {
            auto& input_tensor = optional_input_tensor.value();
            auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape(), pad_c);
            auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
            if (pad_input) {
                formatted_optional_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value));
            } else {
                formatted_optional_input_tensors.push_back(input_tensor);
            }
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensor);
        }
    }

    auto output_tensors = run(operation, formatted_input_tensors, formatted_optional_input_tensors);

    std::vector<Tensor> formatted_output_tensors;
    formatted_output_tensors.reserve(output_tensors.size());
    for (auto i = 0; i < output_tensors.size(); i++) {
        formatted_output_tensors.push_back(AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device));
    }
    return formatted_output_tensors;
}

}
