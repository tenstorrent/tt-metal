#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt::tt_metal::operation {

namespace detail {

static Device* get_device(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) {
    for (auto &input_tensor : input_tensors) {
        if (not input_tensor.get().on_host()) {
            return input_tensor.get().device();
        }
    }
    auto device = AutoFormat::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    return device;
}

void override_runtime_args(
    const OverrideRuntimeArgsCallback& override_runtime_args_callback,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors,
    const std::vector<Tensor> &output_tensors
) {
    std::vector<Buffer*> input_buffers;
    for (auto& tensor : input_tensors) {
        input_buffers.push_back(tensor.get().buffer());
    }
    for (auto& tensor : optional_input_tensors) {
        auto buffer = tensor.has_value() ? tensor.value().get().buffer() : nullptr;
        input_buffers.push_back(buffer);
    }

    std::vector<Buffer*> output_buffers;
    for (auto& tensor : output_tensors) {
        output_buffers.push_back(tensor.buffer());
    }

    override_runtime_args_callback(input_buffers, output_buffers);
}

std::vector<Tensor> run_without_program_cache(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors) {

    auto profile_run_without_program_cache = op_profiler::ProfileScope(op.get_type_name());

    op.validate(input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors);
    auto output_tensors = op.create_output_tensors(input_tensors);

    auto program_with_callbacks = op.create_program(input_tensors, optional_input_tensors, output_tensors);

    auto& program = program_with_callbacks.program;
    auto do_profile = op_profiler::get_profiler_flag();

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
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors) {

    auto profile_run_with_program_cache = op_profiler::ProfileScope(op.get_type_name());

    op.validate(input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors);
    auto output_tensors = op.create_output_tensors(input_tensors);

    auto do_profile = op_profiler::get_profiler_flag();

    auto& program_with_callbacks = program_cache::get_or_create(
        op, input_tensors, optional_input_tensors, output_tensors, device, do_profile
    );

    override_runtime_args(
        program_with_callbacks.override_runtime_args_callback,
        input_tensors, optional_input_tensors, output_tensors
    );

    auto& program = program_with_callbacks.program;
    for (auto& circular_buffer : program.circular_buffers()) {
        if (not circular_buffer->is_allocated()) {
            circular_buffer->reserve(device);
        }
    }

    const char *TT_METAL_DEVICE_DISPATCH_MODE = std::getenv("TT_METAL_DEVICE_DISPATCH_MODE");
    if (TT_METAL_DEVICE_DISPATCH_MODE != nullptr) {
        EnqueueProgram(*HACK_CQ, program, false);

    } else {
        ConfigureDeviceWithProgram(device, program);
        WriteRuntimeArgsToDevice(device, program);
        LaunchKernels(device, program);
    }

    for (auto& circular_buffer : program.circular_buffers()) {
        circular_buffer->deallocate();
    }

    op_profiler::append_all_tensor_io_data(input_tensors, optional_input_tensors, output_tensors);
    op_profiler::dump_device_profiler_results(device, program);

    return output_tensors;
}

}

std::vector<Tensor> run(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors
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

std::vector<Tensor> generic_create_output_tensors(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    Layout output_layout = Layout::TILE,
    const MemoryConfig &output_mem_config = MemoryConfig{.interleaved = true}
) {
    const auto& input_tensor = input_tensors.at(0).get();
    const auto& output_shapes = op.compute_output_shapes(input_tensors);

    // HACK to avoid copy constructors when using vectors
    // TODO: If we have default initializers for Tensor, we can do: std::vector<Tensor> output_tensor(num_tensors);
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(Tensor(output_shape, input_tensor.dtype(), output_layout, input_tensor.device(), output_mem_config));
    }
    return output_tensors;
}

Hash hash_tensor(const Tensor& tensor) {
    const auto shape = tensor.shape();
    return fmt::format(
        "{}_{}_{}_{}_{}_{}_{}",
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        magic_enum::enum_name(tensor.dtype()),
        magic_enum::enum_name(tensor.layout()),
        magic_enum::enum_name(tensor.buffer_type())
    );
}

Hash hash_memory_config(const MemoryConfig& memory_config) {
    return fmt::format(
        "{}_{}_{}",
        memory_config.interleaved,
        memory_config.bank_id,
        magic_enum::enum_name(memory_config.buffer_type)
    );
}

}
