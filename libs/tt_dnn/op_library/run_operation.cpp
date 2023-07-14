#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_numpy/functions.hpp"

#include "tt_stl/reflection.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt::tt_metal::operation {

namespace detail {

static Device* get_device(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>> &optional_input_tensors = {}) {
    for (auto &input_tensor : input_tensors) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            return input_tensor.device();
        }
    }
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() && optional_input_tensor.value().storage_type() == StorageType::DEVICE) {
            return optional_input_tensor.value().device();
        }
    }
    auto device = AutoFormat::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    return device;
}

void override_runtime_args(
    const OverrideRuntimeArgsCallback& override_runtime_args_callback,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors,
    const std::vector<Tensor> &output_tensors
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

void setup_profiler(const HostOperation& op, const std::vector<Tensor> &input_tensors) {
    auto profiler_info = op.create_profiler_info(input_tensors);
    if (profiler_info.preferred_name.has_value()) {
        op_profiler::set_preferred_name(profiler_info.preferred_name.value());
    }
}

void setup_profiler(const DeviceOperation& op, const std::vector<Tensor> &input_tensors) {
    auto profiler_info = op.create_profiler_info(input_tensors);
    if (profiler_info.preferred_name.has_value()) {
        op_profiler::set_preferred_name(profiler_info.preferred_name.value());
    }
    if (profiler_info.parallelization_strategy.has_value()) {
        op_profiler::set_parallelization_strategy(profiler_info.parallelization_strategy.value());
    }
}

std::vector<Tensor> run_without_program_cache(
    const DeviceOperation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors) {

    auto profile_scope = op_profiler::ProfileScope(op.get_type_name());
    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) { setup_profiler(op, input_tensors); }

    op.validate(input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors, optional_input_tensors);
    auto output_tensors = op.create_output_tensors(input_tensors);

    auto program_with_callbacks = op.create_program(input_tensors, optional_input_tensors, output_tensors);
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
    const DeviceOperation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors) {

    auto profile_scope = op_profiler::ProfileScope(op.get_type_name());
    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) { setup_profiler(op, input_tensors); }

    op.validate(input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors, optional_input_tensors);
    auto output_tensors = op.create_output_tensors(input_tensors);

    auto& program_with_callbacks = program_cache::get_or_create(
        op, input_tensors, optional_input_tensors, output_tensors, device, do_profile
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
    const DeviceOperation& op,
    const std::vector<Tensor>& input_tensors,
    const Layout output_layout,
    const MemoryConfig &output_mem_config
) {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_shapes = op.compute_output_shapes(input_tensors);

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE);

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(create_device_tensor(output_shape, input_tensor.dtype(), output_layout, input_tensor.device(), output_mem_config));
    }
    return output_tensors;
}

template<typename Operation>
void log_running_operation(
    const Operation& op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {
#ifdef DEBUG
    tt::log_debug(tt::LogOp, "Operation: {}", op);
    tt::log_debug(tt::LogOp, "Input Tensors: {}", input_tensors);
    if (not optional_input_tensors.empty()) {
        tt::log_debug(tt::LogOp, "Optional Input Tensors: {}", optional_input_tensors);
    }
#endif
}

std::vector<Tensor> run(
    const HostOperation& op,
    const std::vector<Tensor> &input_tensors
) {
    log_running_operation(op, input_tensors);

    auto profile_scope = op_profiler::ProfileScope(op.get_type_name());
    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) { detail::setup_profiler(op, input_tensors); }

    op.validate(input_tensors);
    auto output_tensors = op.compute_output_tensors(input_tensors);

    op_profiler::append_all_tensor_io_data(input_tensors, {}, output_tensors);

    return output_tensors;
}

void log_operation_if_it_does_not_support_program_cache(const DeviceOperation& op) {
#ifdef DEBUG
    if (program_cache::is_enabled()) {
        tt::log_info(tt::LogOp, "Running {} op without program cache", op.get_type_name());
    }
#endif
}

std::vector<Tensor> run(
    const DeviceOperation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors
) {
    log_running_operation(op, input_tensors, optional_input_tensors);
    if (program_cache::is_enabled() and op.supports_program_caching()) {
        return detail::run_with_program_cache(op, input_tensors, optional_input_tensors);
    } else {
        log_operation_if_it_does_not_support_program_cache(op);
        return detail::run_without_program_cache(op, input_tensors, optional_input_tensors);
    }
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors
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
        if (optional_input_tensor.has_value() && optional_input_tensor.value().storage_type() == StorageType::HOST) {
            optional_input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run(op, input_tensors_on_dev, optional_input_tensors_on_dev);
}


std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors,
    const float pad_value,
    const bool pad_c
) {
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

    auto output_shapes = op.compute_output_shapes(input_tensors);

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

    auto output_tensors = run(op, formatted_input_tensors, formatted_optional_input_tensors);

    std::vector<Tensor> formatted_output_tensors;
    formatted_output_tensors.reserve(output_tensors.size());
    for (uint32_t i = 0; i < output_tensors.size(); i++) {
        formatted_output_tensors.push_back(AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device));
    }
    return formatted_output_tensors;
}

}
