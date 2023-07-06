#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <fmt/ranges.h>

namespace tt::tt_metal::operation {

namespace detail {

static Device* get_device(const std::vector<Tensor> &input_tensors) {
    for (auto &input_tensor : input_tensors) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            return input_tensor.device();
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

void setup_profiler(const Operation& op, const std::vector<Tensor> &input_tensors) {
    auto profiler_info = op.create_profiler_info(input_tensors);
    if (profiler_info.preferred_name.has_value()) {
        op_profiler::set_preferred_name(profiler_info.preferred_name.value());
    }
    if (profiler_info.parallelization_strategy.has_value()) {
        op_profiler::set_parallelization_strategy(profiler_info.parallelization_strategy.value());
    }
}

std::vector<Tensor> run_without_program_cache(
    const Operation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors) {

    auto profile_scope = op_profiler::ProfileScope(op.get_type_name());
    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) { setup_profiler(op, input_tensors); }

    op.validate(input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors);
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
    const Operation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors) {

    auto profile_scope = op_profiler::ProfileScope(op.get_type_name());
    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) { setup_profiler(op, input_tensors); }

    op.validate(input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors);
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

std::vector<Tensor> run(
    const Operation& op,
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors
) {
    if (program_cache::is_enabled() and op.supports_program_caching()) {
        return detail::run_with_program_cache(op, input_tensors, optional_input_tensors);
    } else {
        if (program_cache::is_enabled()) {
            tt::log_info(tt::LogOp, "Running {} op without program cache", op.get_type_name());
        }
        return detail::run_without_program_cache(op, input_tensors, optional_input_tensors);
    }
}

Tensor run_without_autoformat(const Operation& op, const Tensor &input_tensor) {
    Device* device;
    if (input_tensor.storage_type() == StorageType::HOST) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto input_tensor_on_dev = input_tensor;
    if (input_tensor_on_dev.storage_type() == StorageType::HOST) {
        input_tensor_on_dev = input_tensor_on_dev.to(device);
    }
    return run(op, {input_tensor_on_dev}).at(0);
}

Tensor run_with_autoformat(const Operation& op, const Tensor &input_tensor, float pad_value, bool pad_c) {
    Device* device;
    if (input_tensor.storage_type() == StorageType::HOST) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto output_shape = op.compute_output_shapes({input_tensor}).at(0);

    auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape(), pad_c);
    auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
    auto padded_input_tensor = input_tensor;
    if (pad_input) {
        padded_input_tensor = AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value);
    }
    auto output_tensor = run(op, {padded_input_tensor}).at(0);
    if (pad_input) {
        output_tensor = AutoFormat::format_output_tensor(output_tensor, output_shape, device);
    }
    return output_tensor;
}

Tensor run_with_autoformat(const Operation& op, const Tensor &input_tensor_a, const Tensor &input_tensor_b, float pad_value) {
    Device* device;
    if (input_tensor_a.storage_type() == StorageType::HOST && input_tensor_b.storage_type() == StorageType::HOST) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else if (input_tensor_a.storage_type() == StorageType::DEVICE){
        device = input_tensor_a.device();
    } else {
        device = input_tensor_b.device();
    }

    auto output_shape = op.compute_output_shapes({input_tensor_a, input_tensor_b}).at(0);

    auto padded_input_shape_a = AutoFormat::pad_to_tile_shape(input_tensor_a.shape());
    auto padded_input_shape_b = AutoFormat::pad_to_tile_shape(input_tensor_b.shape());

    auto pad_a = not AutoFormat::check_input_tensor_format(input_tensor_a, padded_input_shape_a);
    auto padded_input_tensor_a = input_tensor_a;
    if (pad_a) {
        padded_input_tensor_a = AutoFormat::format_input_tensor(input_tensor_a, device, padded_input_shape_a, pad_value);
    }

    auto pad_b = not AutoFormat::check_input_tensor_format(input_tensor_b, padded_input_shape_b);
    auto padded_input_tensor_b = input_tensor_b;
    if (pad_b) {
        padded_input_tensor_b = AutoFormat::format_input_tensor(input_tensor_b, device, padded_input_shape_b, pad_value);
    }
    auto output_tensor = run(op, {padded_input_tensor_a, padded_input_tensor_b}).at(0);
    if (pad_a or pad_b) {
        output_tensor = AutoFormat::format_output_tensor(output_tensor, output_shape, device);
    }
    return output_tensor;
}

Hash hash_tensor(const Tensor& tensor) {
    const auto shape = tensor.shape();
    return fmt::format(
        "{}_{}_{}_{}",
        shape,
        magic_enum::enum_name(tensor.dtype()),
        magic_enum::enum_name(tensor.layout()),
        tensor.storage_type() == StorageType::HOST ? "nullopt" : hash_memory_config(tensor.memory_config())
    );
}

Hash hash_memory_config(const MemoryConfig& memory_config) {
    return fmt::format(
        "{}_{}",
        memory_config.interleaved,
        magic_enum::enum_name(memory_config.buffer_type)
    );
}

}
