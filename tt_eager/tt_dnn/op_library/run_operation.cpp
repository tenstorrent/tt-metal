// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/run_operation.hpp"

#include <chrono>
#include <tt_eager/tensor/tensor.hpp>
#include <tt_eager/tensor/tensor_utils.hpp>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt::tt_metal::operation {

bool skip_profile = false;
std::map<chip_id_t, std::reference_wrapper<Program>> skipped_programs;

namespace detail {

inline bool any_tensor_on_multi_device(const std::vector<Tensor>& tensors) {
    return std::any_of(tensors.begin(), tensors.end(), [](const Tensor& tensor) { return tensor.storage_type() == StorageType::MULTI_DEVICE; });
}

static Device* get_device(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            TT_FATAL(input_tensor.buffer() != nullptr, "Operands need to be allocated in buffers on device");
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

void override_addresses(
    const OverrideAddressesCallback& override_addresses_callback,
    const Program &program,
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

    override_addresses_callback(program, input_buffers, output_buffers);
}

void setup_profiler(const HostOperation& operation, const std::vector<Tensor>& input_tensors) {
    auto profiler_info = operation.create_profiler_info(input_tensors);
    if (profiler_info.preferred_name.has_value()) {
        op_profiler::set_preferred_name(profiler_info.preferred_name.value());
    }
    op_profiler::append_meta_data(fmt::format("{}", operation.attributes()));
}

void setup_profiler(const DeviceOperation& operation, const std::vector<Tensor>& input_tensors, const Program& program) {
    auto profiler_info = operation.create_profiler_info(input_tensors);
    if (profiler_info.preferred_name.has_value()) {
        op_profiler::set_preferred_name(profiler_info.preferred_name.value());
    }
    if (profiler_info.parallelization_strategy.has_value()) {
        op_profiler::set_parallelization_strategy(profiler_info.parallelization_strategy.value());
    }

    op_profiler::append_kernel_info(program);
    op_profiler::append_meta_data(fmt::format("{}", operation.attributes()));
}

void setup_profiler(const DeviceOperation& operation, const std::vector<Tensor>& input_tensors, std::shared_ptr<const Program> program) {
    setup_profiler(operation, input_tensors, *program);
}

template <typename OperationType>
constexpr op_profiler::OpType get_profiler_operation_type() {
    if constexpr (std::is_same_v<OperationType, HostOperation>) {
        return op_profiler::OpType::tt_dnn_cpu;
    } else if constexpr (std::is_same_v<OperationType, DeviceOperation>) {
        return op_profiler::OpType::tt_dnn_device;
    } else if constexpr (std::is_same_v<OperationType, ExternalOperation>) {
        return op_profiler::OpType::python_fallback;
    } else {
        static_assert(tt::stl::concepts::always_false_v<OperationType>, "OperationType is not supported!");
    }
}

template <typename Function>
constexpr auto decorate_host_operation(const Function& function) {
    return [function]<typename Operation, typename... Args>(const Operation& operation, Args&&... args) {
        log_operation(operation, args...);
        auto output_tensors = function(operation, args...);
        return output_tensors;
    };
}

template <typename Function>
constexpr auto decorate_device_operation(const Function& function) {
    return [function]<typename Operation, typename... Tensors>(
               std::optional<std::reference_wrapper<CommandQueue>> queue,
               const Operation& operation,
               Tensors&&... tensors) {
        log_operation(operation, tensors...);
        auto output_tensors = function(queue, operation, tensors...);
        return output_tensors;
    };
}

std::vector<Tensor> run_host_operation(const HostOperation& operation, const std::vector<Tensor>& input_tensors) {
    ZoneScoped;
    ZoneText(operation.get_type_name().c_str(), operation.get_type_name().size());

    auto profile_scope = op_profiler::OpProfileScope(operation.get_type_name(), op_profiler::OpType::tt_dnn_cpu);
    auto do_profile = op_profiler::get_profiler_flag();

    operation.validate(input_tensors);
    auto output_tensors = operation.compute_output_tensors(input_tensors);

    if (do_profile) {
        detail::setup_profiler(operation, input_tensors);
        //op_profiler::set_perf_model(operation.create_op_performance_model(input_tensors, optional_input_tensors, output_tensors));
    }

    op_profiler::append_all_tensor_io_data(input_tensors, {}, output_tensors);

    return output_tensors;
}

inline const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;

std::vector<Tensor> run_device_operation(
    std::optional<std::reference_wrapper<CommandQueue>> queue,
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    ZoneScoped;
    ZoneText(operation.get_type_name().c_str(), operation.get_type_name().size());
    std::unique_ptr<op_profiler::OpProfileScope> profile_scope;
    if (!operation::skip_profile) {
        profile_scope = std::make_unique<op_profiler::OpProfileScope>(operation.get_type_name(), op_profiler::OpType::tt_dnn_device);
    }

    std::function<std::variant<std::shared_ptr<Program>, std::reference_wrapper<Program>>(
        const DeviceOperation&,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&,
        std::vector<Tensor>&)>
        get_or_create_program;
    auto& program_cache = input_tensors[0].device()->program_cache;
    if (program_cache.is_enabled()) {
        get_or_create_program = [&program_cache](const DeviceOperation& operation,
                                   const std::vector<Tensor>& input_tensors,
                                   const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                   std::vector<Tensor>& output_tensors) -> std::reference_wrapper<Program> {

            auto program_hash = operation.compute_program_hash(input_tensors, optional_input_tensors);
            auto&& [program_ptr, cache_hit] = program_cache.find(program_hash);
            if (not cache_hit) {
                program_ptr = std::make_shared<operation::ProgramWithCallbacks>(operation.create_program(input_tensors, optional_input_tensors, output_tensors));
                program_cache.insert(program_hash, program_ptr);
            }
            auto& program_with_callbacks = *(reinterpret_cast<operation::ProgramWithCallbacks*>(program_ptr.get()));
            TT_ASSERT(program_with_callbacks.supports_program_cache());

            if (cache_hit) {
                ZoneScopedN("Cache_hit_set_runtime_args");
                if (program_with_callbacks.override_addresses_callback.has_value()) {
                    auto override_addresses_callback = program_with_callbacks.override_addresses_callback.value();
                    override_addresses(
                        override_addresses_callback, program_with_callbacks.program, input_tensors, optional_input_tensors, output_tensors);
                }

                if (program_with_callbacks.override_runtime_arguments_callback.has_value()) {
                    auto override_runtime_arguments_callback =
                        program_with_callbacks.override_runtime_arguments_callback.value();
                    operation.override_runtime_arguments(
                        override_runtime_arguments_callback,
                        program_with_callbacks.program,
                        input_tensors,
                        optional_input_tensors,
                        output_tensors);
                }
            }
            return program_with_callbacks.program;
        };
    } else {
        get_or_create_program = [](const DeviceOperation& operation,
                                   const std::vector<Tensor>& input_tensors,
                                   const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                   std::vector<Tensor>& output_tensors) -> std::shared_ptr<Program> {
            auto program_with_callbacks =
                operation.create_program(input_tensors, optional_input_tensors, output_tensors);
            return std::make_shared<Program>(std::move(program_with_callbacks.program));
        };
    }
    operation.validate(input_tensors, optional_input_tensors, optional_output_tensors);
    auto output_tensors = operation.create_output_tensors(input_tensors, optional_output_tensors);
    auto program = get_or_create_program(operation, input_tensors, optional_input_tensors, output_tensors);

    // Enqueue or Launch Program
    std::visit(
        [&operation, &input_tensors, &optional_input_tensors, &output_tensors, queue](auto&& program) {
            auto device = detail::get_device(input_tensors, optional_input_tensors);
            using T = std::decay_t<decltype(program)>;
            if constexpr (std::is_same_v<T, std::reference_wrapper<Program>> || std::is_same_v<T, std::shared_ptr<Program>> ) {
                if (USE_FAST_DISPATCH) {
                    // Program will temporarily own the input buffers. This is required, since with Async command queues, the input
                    // tensor can preemptively be deallocted on device, unless program maintains explicit ownership.
                    // This invocation of the program will give up ownership once its enqueued.
                    for (const auto& input_tensor: input_tensors) {
                        if (input_tensor.storage_type() == StorageType::DEVICE) {
                            AssignGlobalBufferToProgram(input_tensor.device_buffer(), program);
                        }
                    }
                    for (auto& optional_input_tensor : optional_input_tensors) {
                        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() == StorageType::DEVICE) {
                            AssignGlobalBufferToProgram(optional_input_tensor.value().device_buffer(), program);
                        }
                    }
                    TT_ASSERT(queue.has_value(), "CommandQueue is required for fast dispatch mode");
                    CommandQueue& cq = queue.value().get();
                    EnqueueProgram(cq, program, false);
                    if (!operation::skip_profile) {
                        // Only need to dump device data when in dispatch mode
                        // LaunchKernel automatically dumps device data
                        op_profiler::dump_device_profiler_results(device, program);
                    } else {
                        if constexpr (std::is_same_v<T, std::shared_ptr<Program>>) {
                            operation::skipped_programs.emplace(device->id(), *program);
                        }
                        else if constexpr( std::is_same_v<T, std::reference_wrapper<Program>> ){
                            operation::skipped_programs.emplace(device->id(), program );
                        }
                    }
                } else {
                    ::detail::LaunchProgram(device, program);
                }
                if (!operation::skip_profile) {
                    auto do_profile = op_profiler::get_profiler_flag();
                    if (do_profile) {
                        detail::setup_profiler(operation, input_tensors, program);
                        op_profiler::set_perf_model(operation.create_op_performance_model(input_tensors, optional_input_tensors, output_tensors));
                    }
                }
            }
        },
        program);

    if (!operation::skip_profile) {
        op_profiler::append_all_tensor_io_data(input_tensors, optional_input_tensors, output_tensors);
    }
    return output_tensors;
}

std::vector<Tensor> run_multi_device_operation(
    std::optional<std::reference_wrapper<CommandQueue>> queue,
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors)
{
    // TODO: Assumes each input/output tensor is mapped to the same set of devices; relax this later
    std::vector<Device*> devices = get_devices(input_tensors[0]);

    std::map<Device*, std::vector<Tensor>> per_device_output_tensors;
    std::optional<std::size_t> num_output_tensors_per_device;
    for (Device *device : devices)
    {
        auto device_output_tensors = run_device_operation(
            device->command_queue(),
            operation,
            get_device_tensors(device, input_tensors),
            get_device_tensors(device, optional_input_tensors),
            get_device_tensors(device, optional_output_tensors));

        per_device_output_tensors[device] = device_output_tensors;

        if (not num_output_tensors_per_device.has_value()) {
            num_output_tensors_per_device = device_output_tensors.size();
        } else {
            TT_ASSERT(num_output_tensors_per_device == device_output_tensors.size(),
                "Output tensors per device should be same for all devices");
        }
    }

    std::vector<Tensor> multi_device_output_tensors;
    for (int i = 0; i < num_output_tensors_per_device; ++i)
    {
        std::vector<DeviceBuffer> buffers;
        std::vector<Shape> shapes;
        for (Device *device : devices) {
            buffers.push_back(per_device_output_tensors[device][i].device_buffer());
            shapes.push_back(per_device_output_tensors[device][i].get_legacy_shape());
        }

        multi_device_output_tensors.push_back(
            Tensor{
                MultiDeviceStorage{buffers, shapes},
                per_device_output_tensors[devices[0]][i].get_legacy_shape(),
                per_device_output_tensors[devices[0]][i].get_dtype(),
                per_device_output_tensors[devices[0]][i].get_layout()
            }
        );
    }
    return multi_device_output_tensors;
}

}  // namespace detail

std::vector<Tensor> run(const HostOperation& operation, const std::vector<Tensor>& input_tensors) {
    return detail::decorate_host_operation(detail::run_host_operation)(operation, input_tensors);
}

std::vector<Tensor> run(
    CommandQueue& queue,
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    if (detail::any_tensor_on_multi_device(input_tensors)) {
        return detail::decorate_device_operation(detail::run_multi_device_operation)(
            std::make_optional(std::ref(queue)), operation, input_tensors, optional_input_tensors, optional_output_tensors);
    }
    return detail::decorate_device_operation(detail::run_device_operation)(
        queue, operation, input_tensors, optional_input_tensors, optional_output_tensors);
}

std::vector<Tensor> run(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    if (detail::any_tensor_on_multi_device(input_tensors)) {
        return detail::decorate_device_operation(detail::run_multi_device_operation)(
            std::nullopt, operation, input_tensors, optional_input_tensors, optional_output_tensors);
    }
    auto device = detail::get_device(input_tensors, optional_input_tensors);
    return detail::decorate_device_operation(detail::run_device_operation)(
        detail::USE_FAST_DISPATCH ? std::make_optional(std::ref(device->command_queue())) : std::nullopt, operation, input_tensors, optional_input_tensors, optional_output_tensors);
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
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(input_tensor, device));
        } else {
            input_tensors_on_dev.push_back(input_tensor);
        }
    }
    std::vector<std::optional<const Tensor>> optional_input_tensors_on_dev;
    optional_input_tensors_on_dev.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() != StorageType::DEVICE) {
            optional_input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run(operation, input_tensors_on_dev, optional_input_tensors_on_dev, {});
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors
) {
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

    std::vector<Tensor> input_tensors_on_dev;
    input_tensors_on_dev.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(input_tensor, device));
        } else {
            input_tensors_on_dev.push_back(input_tensor);
        }
    }
    std::vector<std::optional<const Tensor>> optional_input_tensors_on_dev;
    optional_input_tensors_on_dev.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() != StorageType::DEVICE) {
            optional_input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run(operation, input_tensors_on_dev, optional_input_tensors_on_dev, optional_output_tensors);
}

// To be deprecated/removed in favor of new implementation where ops specifically request how to format inputs/outputss
std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const float pad_value,
    const bool pad_c
) {
    if (detail::any_tensor_on_multi_device(input_tensors)) {
        return run(operation, input_tensors, optional_input_tensors);
    }
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

    auto output_shapes = operation.compute_output_shapes(input_tensors);

    std::vector<Tensor> formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape(), pad_c);
        auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
        if (pad_input) {
            formatted_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
        } else {
            formatted_input_tensors.push_back(input_tensor);
        }
    }

    std::vector<std::optional<const Tensor>> formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value()) {
            auto& input_tensor = optional_input_tensor.value();
            auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape(), pad_c);
            auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
            if (pad_input) {
                formatted_optional_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
            } else {
                formatted_optional_input_tensors.push_back(input_tensor);
            }
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensor);
        }
    }

    auto output_tensors = run(operation, formatted_input_tensors, formatted_optional_input_tensors);

    TT_ASSERT(output_tensors.size() == output_shapes.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] = AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, Layout::TILE);
    }
    return output_tensors;
}

std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<FormatParams>>& optional_input_formatting
) {
    if (detail::any_tensor_on_multi_device(input_tensors)) {
        return run(operation, input_tensors, optional_input_tensors);
    }
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

    auto output_shapes = operation.compute_output_shapes(input_tensors);

    TT_ASSERT(input_tensors.size() == input_formatting.size());
    TT_ASSERT(optional_input_tensors.size() == optional_input_formatting.size());

    std::vector<Tensor> formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        formatted_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensors[i], device, input_formatting[i].pad_shape, input_formatting[i].pad_value, input_formatting[i].target_layout));
    }

    std::vector<std::optional<const Tensor>> formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (uint32_t i = 0; i < optional_input_tensors.size(); ++i) {
        if (optional_input_tensors[i].has_value()) {
            auto& input_tensor = optional_input_tensors[i].value();
            TT_ASSERT(optional_input_formatting[i].has_value());
            auto& input_formatting = optional_input_formatting[i].value();
            formatted_optional_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensor, device, input_formatting.pad_shape, input_formatting.pad_value, input_formatting.target_layout));
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensors[i]);
        }
    }

    auto output_tensors = run(operation, formatted_input_tensors, formatted_optional_input_tensors);

    TT_ASSERT(output_tensors.size() == output_shapes.size());
    TT_ASSERT(output_tensors.size() == output_layouts.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] = AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, output_layouts[i]);
    }

    return output_tensors;
}

}
