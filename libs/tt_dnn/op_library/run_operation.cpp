#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/operation_cache.hpp"

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
    const std::vector<Tensor> &output_tensors
) {
    std::vector<Buffer*> input_buffers;
    for (auto& tensor : input_tensors) {
        input_buffers.push_back(tensor.get().buffer());
    }
    std::vector<Buffer*> output_buffers;
    for (auto& tensor : output_tensors) {
        output_buffers.push_back(tensor.buffer());
    }

    override_runtime_args_callback(input_buffers, output_buffers);
}

void validate(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors
) {
    if (optional_input_tensors.empty()) {
        return op.validate(input_tensors);
    }
    return op.validate(input_tensors, optional_input_tensors);
}

ProgramWithCallbacks create_program(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>>& input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors
)
{
    if (optional_input_tensors.empty()) {
        return op.create_program(input_tensors, output_tensors);
    }
    return op.create_program(input_tensors, optional_input_tensors, output_tensors);
}

std::vector<Tensor> run_without_program_cache(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors) {

    auto profile_run_wo_program_cache = op_profiler::ProfileScope(op.get_op_name());

    validate(op, input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors);
    auto output_tensors = op.create_output_tensors(input_tensors);

    auto program_with_callbacks = create_program(op, input_tensors, optional_input_tensors, output_tensors);

    auto& program = program_with_callbacks.program;
    auto do_profile = op_profiler::get_profiler_flag();

    CompileProgram(device, program, do_profile);
    ConfigureDeviceWithProgram(device, program);
    WriteRuntimeArgsToDevice(device, program);
    LaunchKernels(device, program);

    op_profiler::append_all_tensor_io_data(input_tensors, optional_input_tensors, output_tensors);
    op_profiler::dump_device_profiler_results(device, program);

    return output_tensors;
}


std::vector<Tensor> run_with_program_cache(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors) {

    validate(op, input_tensors, optional_input_tensors);

    auto device = detail::get_device(input_tensors);
    auto output_tensors = op.create_output_tensors(input_tensors);

    auto& program_with_callbacks = operation_cache::get_or_create(op, input_tensors, output_tensors, device);

    override_runtime_args(
        program_with_callbacks.override_runtime_args_callback,
        input_tensors, output_tensors);

    auto& program = program_with_callbacks.program;
    ConfigureDeviceWithProgram(device, program);
    WriteRuntimeArgsToDevice(device, program);
    LaunchKernels(device, program);

    return output_tensors;
}

}

std::vector<Tensor> run(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors
) {
    if (operation_cache::is_enabled() and op.supports_program_caching()) {
        TT_ASSERT (op_profiler::get_profiler_flag() == false , "Refer to ticket #1162 regarding profiling and program caching");
        return detail::run_with_program_cache(op, input_tensors, optional_input_tensors);
    } else {
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
        "{}_{}_{}_{}_{}_{}",
         shape[0],
         shape[1],
         shape[2],
         shape[3],
         magic_enum::enum_name(tensor.dtype()),
         magic_enum::enum_name(tensor.layout())
    );
}

}
