// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/run_operation.hpp"

#include <tt_eager/tensor/tensor.hpp>
#include <tt_eager/tensor/tensor_utils.hpp>

#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt::tt_metal {
    std::atomic<uint32_t> operation_id_atomic_count = 0;
}

namespace tt::tt_metal::operation {

namespace detail {

inline bool any_tensor_on_multi_device(const Tensors& tensors) {
    return std::any_of(tensors.begin(), tensors.end(), [](const Tensor& tensor) {
        return tensor.storage_type() == StorageType::MULTI_DEVICE;
    });
}

Device* get_device(const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors) {
    for (auto& input_tensor : input_tensors) {
        if (std::holds_alternative<DeviceStorage>(input_tensor.tensor_attributes->storage)) {
            return input_tensor.workers.at(0);
        }
    }
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and
            std::holds_alternative<DeviceStorage>(optional_input_tensor.value().tensor_attributes->storage)) {
            return optional_input_tensor.value().workers.at(0);
        }
    }
    auto device = AutoFormat::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to operation are on device");
    return device;
}

void validate_op_launch(Device* worker) {
    if (worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS) {
        TT_FATAL(
            not worker->in_main_thread(),
            "launch_op or launch_with_autoformat must be used when running in async mode.");
    }
}

template <class OutputTensors>
void override_addresses(
    const OverrideAddressesCallback& override_addresses_callback,
    const Program& program,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OutputTensors& output_tensors) {
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
        if constexpr (std::is_same_v<OptionalTensors, OutputTensors>) {
            auto buffer = tensor.has_value() ? tensor.value().buffer() : nullptr;
            output_buffers.push_back(buffer);
        } else {
            output_buffers.push_back(tensor.buffer());
        }
    }

    override_addresses_callback(program, input_buffers, output_buffers);
}

template void override_addresses<Tensors>(
    const OverrideAddressesCallback& override_addresses_callback,
    const Program& program,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const Tensors& output_tensors);

template void override_addresses<OptionalTensors>(
    const OverrideAddressesCallback& override_addresses_callback,
    const Program& program,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& output_tensors);

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
               std::reference_wrapper<CommandQueue> queue,
               const Operation& operation,
               Tensors&&... tensors) {
        log_operation(operation, tensors...);
        auto output_tensors = function(queue, operation, tensors...);
        return output_tensors;
    };
}

template <typename OutputTensors>
OutputTensors run_host_operation(const HostOperation<OutputTensors>& operation, const Tensors& input_tensors) {
    ZoneScopedN("TT_DNN_HOST_OP");
    uint32_t op_id = assign_operation_id();

    operation.validate(input_tensors);
    auto output_tensors = operation.compute_output_tensors(input_tensors);

    TracyOpTTNNHost(op_id, operation, input_tensors, output_tensors);

    return output_tensors;
}

template Tensors run_host_operation(const HostOperation<Tensors>& operation, const Tensors& input_tensors);
template OptionalTensors run_host_operation(
    const HostOperation<OptionalTensors>& operation, const Tensors& input_tensors);

inline const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;

template <typename OutputTensors>
OutputTensors run_device_operation(
    std::reference_wrapper<CommandQueue> queue,
    const DeviceOperation<OutputTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors) {
    ZoneScopedN("TT_DNN_DEVICE_OP");
    uint32_t op_id = assign_operation_id();

    std::function<std::variant<std::shared_ptr<Program>, std::reference_wrapper<Program>>(
        const DeviceOperation<OutputTensors>&,
        const Tensors&,
        const OptionalConstTensors&,
        OutputTensors&,
        const OptionalTensors&)>
        get_or_create_program;

    auto& program_cache = input_tensors[0].device()->program_cache;

    tt::stl::hash::hash_t program_hash = 0;
    if (program_cache.is_enabled()) {
        get_or_create_program = [&program_cache, &program_hash](
                                    const DeviceOperation<OutputTensors>& operation,
                                    const Tensors& input_tensors,
                                    const OptionalConstTensors& optional_input_tensors,
                                    OutputTensors& output_tensors,
                                    const OptionalTensors& optional_output_tensors) -> std::reference_wrapper<Program> {
            program_hash = operation.compute_program_hash(input_tensors, optional_input_tensors);
            auto cache_hit = program_cache.contains(program_hash);

            log_debug(tt::LogOp, "Program Hash: {} ({})", program_hash, cache_hit ? "HIT" : "MISS");

            if (not cache_hit or operation.uses_custom_program_hash()) {
                operation.validate(input_tensors, optional_input_tensors, optional_output_tensors);
            }

            if (not cache_hit) {
                program_cache.insert(
                    program_hash, operation.create_program(input_tensors, optional_input_tensors, output_tensors));
            }
            auto& program_with_callbacks = program_cache.get<operation::CacheableProgram<OutputTensors>>(program_hash);
            TT_ASSERT(program_with_callbacks.supports_program_cache());

            if (cache_hit) {
                ZoneScopedN("Cache_hit_set_runtime_args");
                if (program_with_callbacks.override_addresses_callback.has_value()) {
                    auto override_addresses_callback = program_with_callbacks.override_addresses_callback.value();
                    // Deprecated
                    override_addresses(
                        override_addresses_callback,
                        program_with_callbacks.program,
                        input_tensors,
                        optional_input_tensors,
                        output_tensors);
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
        get_or_create_program = [](const DeviceOperation<OutputTensors>& operation,
                                   const Tensors& input_tensors,
                                   const OptionalConstTensors& optional_input_tensors,
                                   OutputTensors& output_tensors,
                                   const OptionalTensors& optional_output_tensors) -> std::shared_ptr<Program> {
            operation.validate(input_tensors, optional_input_tensors, optional_output_tensors);
            auto program_with_callbacks =
                operation.create_program(input_tensors, optional_input_tensors, output_tensors);
            return std::make_shared<Program>(std::move(program_with_callbacks.program));
        };
    }

    auto output_tensors = operation.create_output_tensors(input_tensors, optional_output_tensors);
    auto program = get_or_create_program(
        operation, input_tensors, optional_input_tensors, output_tensors, optional_output_tensors);
    uint32_t device_id = detail::get_device(input_tensors, optional_input_tensors)->id();

    if (std::holds_alternative<std::reference_wrapper<Program>>(program))
    {
        std::get<std::reference_wrapper<Program>>(program).get().set_global_id(op_id);
    }
    else
    {
        std::get<std::shared_ptr<Program>>(program)->set_global_id(op_id);
    }

    // Enqueue or Launch Program
    std::visit(
        [&operation, &input_tensors, &optional_input_tensors, &output_tensors, queue](auto&& program) {
            auto device = detail::get_device(input_tensors, optional_input_tensors);
            using T = std::decay_t<decltype(program)>;
            if constexpr (
                std::is_same_v<T, std::reference_wrapper<Program>> || std::is_same_v<T, std::shared_ptr<Program>>) {
                if (USE_FAST_DISPATCH) {
                    // Program will temporarily own the input buffers. This is required, since with Async command
                    // queues, the input tensor can preemptively be deallocted on device, unless program maintains
                    // explicit ownership. This invocation of the program will give up ownership once its enqueued.
                    for (const auto& input_tensor : input_tensors) {
                        if (input_tensor.storage_type() == StorageType::DEVICE) {
                            AssignGlobalBufferToProgram(input_tensor.device_buffer(), program);
                        }
                    }
                    for (auto& optional_input_tensor : optional_input_tensors) {
                        if (optional_input_tensor.has_value() and
                            optional_input_tensor.value().storage_type() == StorageType::DEVICE) {
                            AssignGlobalBufferToProgram(optional_input_tensor.value().device_buffer(), program);
                        }
                    }
                    EnqueueProgram(queue, program, false);
                } else {
                    ::detail::LaunchProgram(device, program);
                }
            }
        },
        program);

    TracyOpTTNNDevice(
        op_id,
        program_hash,
        program_cache.is_enabled(),
        device_id,
        operation,
        program,
        input_tensors,
        optional_input_tensors,
        output_tensors);

    return output_tensors;
}

template Tensors run_device_operation(
    std::reference_wrapper<CommandQueue> queue,
    const DeviceOperation<Tensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors);

template OptionalTensors run_device_operation(
    std::reference_wrapper<CommandQueue> queue,
    const DeviceOperation<OptionalTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors);

}  // namespace detail

template <class OutputTensors>
OutputTensors run(const HostOperation<OutputTensors>& operation, const Tensors& input_tensors) {
    return detail::decorate_host_operation(detail::run_host_operation<OutputTensors>)(operation, input_tensors);
}
template Tensors run(const HostOperation<Tensors>& operation, const Tensors& input_tensors);
template OptionalTensors run(const HostOperation<OptionalTensors>& operation, const Tensors& input_tensors);

template <class OutputTensors>
OutputTensors run(
    const DeviceOperation<OutputTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id) {
    auto device = detail::get_device(input_tensors, optional_input_tensors);
#ifdef DEBUG
    operation.validate(input_tensors, optional_input_tensors, optional_output_tensors);
    detail::validate_op_launch(device);
#endif
    return detail::decorate_device_operation(detail::run_device_operation<OutputTensors>)(
        std::ref(device->command_queue(cq_id)),
        operation,
        input_tensors,
        optional_input_tensors,
        optional_output_tensors);
}

template Tensors run(
    const DeviceOperation<Tensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id);

template OptionalTensors run(
    const DeviceOperation<OptionalTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id);

template <class OutputTensors>
OutputTensors run_without_autoformat(
    const DeviceOperation<OutputTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id) {
    ZoneScoped;
    Device* device = detail::get_device(input_tensors, optional_input_tensors);
    detail::validate_op_launch(device);
    Tensors input_tensors_on_dev;
    input_tensors_on_dev.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(input_tensor, device));
        } else {
            input_tensors_on_dev.push_back(input_tensor);
        }
    }
    OptionalConstTensors optional_input_tensors_on_dev;
    optional_input_tensors_on_dev.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() != StorageType::DEVICE) {
            optional_input_tensors_on_dev.push_back(
                AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run<OutputTensors>(operation, input_tensors_on_dev, optional_input_tensors_on_dev, optional_output_tensors, cq_id);
}

template Tensors run_without_autoformat<Tensors>(
    const DeviceOperation<Tensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id);

template OptionalTensors run_without_autoformat<OptionalTensors>(
    const DeviceOperation<OptionalTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id);

// To be deprecated/removed in favor of new implementation where ops specifically request how to format inputs/outputss
Tensors run_with_autoformat(
    const DeviceOperation<Tensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    const float pad_value,
    const bool pad_c,
    uint8_t cq_id) {
    ZoneScoped;
    Device* device = detail::get_device(input_tensors, optional_input_tensors);
    detail::validate_op_launch(device);
    auto output_shapes = operation.compute_output_shapes(input_tensors);

    Tensors formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape(), pad_c);
        auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
        if (pad_input) {
            formatted_input_tensors.push_back(
                AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
        } else {
            formatted_input_tensors.push_back(input_tensor);
        }
    }

    OptionalConstTensors formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value()) {
            auto& input_tensor = optional_input_tensor.value();
            auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape(), pad_c);
            auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
            if (pad_input) {
                formatted_optional_input_tensors.push_back(
                    AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
            } else {
                formatted_optional_input_tensors.push_back(input_tensor);
            }
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensor);
        }
    }

    auto output_tensors = run<Tensors>(operation, formatted_input_tensors, formatted_optional_input_tensors, optional_output_tensors, cq_id);

    TT_ASSERT(output_tensors.size() == output_shapes.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] = AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, Layout::TILE);
    }
    return output_tensors;
}

Tensors run_with_autoformat(
    const DeviceOperation<Tensors>& operation,
    const Tensors& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const OptionalConstTensors& optional_input_tensors,
    const std::vector<std::optional<FormatParams>>& optional_input_formatting,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id) {
    ZoneScoped;
    Device* device = detail::get_device(input_tensors, optional_input_tensors);
    detail::validate_op_launch(device);
    auto output_shapes = operation.compute_output_shapes(input_tensors);

    TT_ASSERT(input_tensors.size() == input_formatting.size());
    TT_ASSERT(optional_input_tensors.size() == optional_input_formatting.size());

    Tensors formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        formatted_input_tensors.push_back(AutoFormat::format_input_tensor(
            input_tensors[i],
            device,
            input_formatting[i].pad_shape,
            input_formatting[i].pad_value,
            input_formatting[i].target_layout));
    }

    OptionalConstTensors formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (uint32_t i = 0; i < optional_input_tensors.size(); ++i) {
        if (optional_input_tensors[i].has_value()) {
            auto& input_tensor = optional_input_tensors[i].value();
            TT_ASSERT(optional_input_formatting[i].has_value());
            auto& input_formatting = optional_input_formatting[i].value();
            formatted_optional_input_tensors.push_back(AutoFormat::format_input_tensor(
                input_tensor,
                device,
                input_formatting.pad_shape,
                input_formatting.pad_value,
                input_formatting.target_layout));
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensors[i]);
        }
    }

    auto output_tensors = run<Tensors>(operation, formatted_input_tensors, formatted_optional_input_tensors, optional_output_tensors, cq_id);

    TT_ASSERT(output_tensors.size() == output_shapes.size());
    TT_ASSERT(output_tensors.size() == output_layouts.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] =
            AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, output_layouts[i]);
    }

    return output_tensors;
}

void launch_with_autoformat(
    std::function<Tensors(const Tensors&, const OptionalConstTensors&, const OptionalTensors&)>&& op_func,
    const Tensors input_tensors,
    Tensors& output_tensors,
    const OptionalConstTensors optional_input_tensors,
    const OptionalTensors optional_output_tensors) {
    // Mark each output tensor as having dynamic storage (can be on host or device, depending
    // on autoformat behaviour). Multi device tensors do not support dynamic storage.
    for (auto& output_tensor : output_tensors) {
        output_tensor.tensor_attributes->dynamic_storage = (output_tensor.workers.size() <= 1);
    }
    launch_op(std::move(op_func), input_tensors, output_tensors, optional_input_tensors, optional_output_tensors);
}

void launch_op(
    std::function<Tensors(const Tensors&, const OptionalConstTensors&, const OptionalTensors&)>&& op_func,
    const Tensors input_tensors,
    Tensors& output_tensors,
    const OptionalConstTensors optional_input_tensors,
    const OptionalTensors optional_output_tensors,
    bool enable_autoformat_device) {
    // Send host side op compile and run to the worker queue
    // Assert to ensure that worker threads are specified.
    ZoneScopedN("LaunchOp");
    auto& workers = output_tensors.at(0).workers;
    std::size_t workers_size = workers.size();
    if (not enable_autoformat_device and workers.empty() or not workers.at(0)->in_main_thread()) {
        // Run in main thread or immediately in worker thread
        output_tensors = op_func(input_tensors, optional_input_tensors, optional_output_tensors);
        return;
    }
    for (auto& output_tensor : output_tensors) {
        TT_FATAL(
            output_tensor.workers.size(),
            "Worker threads must be specified for outputs populated by launch_op. This API can only be used for "
            "creating output tensors on device.");
        TT_FATAL(
            output_tensor.workers == workers,
            "Worker threads must be consistent across all outputs populated by launch_op.");
    }
    validate_worker_modes(workers);
    // Record ref counts for all tensors before pushing to worker queue.
    std::vector<uint32_t> input_tensor_ref_count = std::vector<uint32_t>(input_tensors.size());
    std::vector<uint32_t> optional_input_tensor_ref_count = std::vector<uint32_t>(optional_input_tensors.size());
    std::vector<uint32_t> output_tensor_ref_count = std::vector<uint32_t>(output_tensors.size());
    std::vector<uint32_t> optional_output_tensor_ref_count = std::vector<uint32_t>(optional_output_tensors.size());
    ;

    std::vector<Tensor> async_safe_input_tensors = std::vector<Tensor>(input_tensors.size());
    std::vector<std::optional<const Tensor>> async_safe_optional_input_tensors = {};
    std::unordered_set<uint32_t> cross_worker_input_tensor_idx = {};
    std::unordered_set<uint32_t> cross_worker_optional_input_tensor_idx = {};
    // When running on a single device, input tensors can be using borrowed storage. If so, when running in async mode,
    // copy borrowed tensors to owned storage.
    for (int i = 0; i < input_tensors.size(); i++) {
        async_safe_input_tensors[i] = copy_borrowed_tensor_in_async_mode(workers.at(0), input_tensors.at(i));
        input_tensor_ref_count[i] = async_safe_input_tensors[i].tensor_attributes->record_main_thread_ref_count();
    }
    for (int i = 0; i < optional_input_tensors.size(); i++) {
        if (optional_input_tensors[i].has_value()) {
            async_safe_optional_input_tensors.push_back(
                copy_borrowed_tensor_in_async_mode(workers.at(0), optional_input_tensors[i].value()));
            optional_input_tensor_ref_count[i] =
                async_safe_optional_input_tensors[i].value().tensor_attributes->record_main_thread_ref_count();
        } else {
            async_safe_optional_input_tensors.push_back(std::nullopt);
            optional_input_tensor_ref_count[i] = 0;
        }
    }
    for (int i = 0; i < output_tensors.size(); i++) {
        output_tensor_ref_count[i] = output_tensors[i].tensor_attributes->record_main_thread_ref_count();
    }
    for (int i = 0; i < optional_output_tensors.size(); i++) {
        if (optional_output_tensors[i].has_value()) {
            optional_output_tensor_ref_count[i] =
                optional_output_tensors[i].value().tensor_attributes->record_main_thread_ref_count();
        } else {
            optional_output_tensor_ref_count[i] = 0;
        }
    }
    // Check if this op dispatch step relies on tensors from other workers.
    // If so, mark them in use by current worker. Tensors shared across workers
    // are only supported when each tensor is tied to a single device/worker
    // (example all-gather).
    if (workers_size == 1) {
        // Single worker per tensor and.
        for (int i = 0; i < async_safe_input_tensors.size(); i++) {
            if (async_safe_input_tensors.at(i).get_workers().size() and
                async_safe_input_tensors.at(i).get_workers().at(0) != workers.at(0)) {
                // This input has a worker assigned that doesn't match the worker of the output being created (its
                // shared).
                async_safe_input_tensors.at(i).tensor_attributes->num_sibling_workers_sharing_tensor++;
                cross_worker_input_tensor_idx.insert(i);
            }
        }
        for (int i = 0; i < async_safe_optional_input_tensors.size(); i++) {
            if (async_safe_optional_input_tensors.at(i).has_value() and
                async_safe_optional_input_tensors.at(i).value().get_workers().size() and
                async_safe_optional_input_tensors.at(i).value().get_workers().at(0) != workers.at(0)) {
                async_safe_optional_input_tensors.at(i).value().tensor_attributes->num_sibling_workers_sharing_tensor++;
                cross_worker_optional_input_tensor_idx.insert(i);
            }
        }
    }

    {
        ZoneScopedN("PushOpToWorkers");
        auto work_lambda = std::make_shared<std::function<void(Device*)>>(
            [workers_size,
             op_func,
             optional_output_tensors,
             async_safe_optional_input_tensors,
             inputs = async_safe_input_tensors,
             outputs = output_tensors,
             shared_input_idx = cross_worker_input_tensor_idx,
             shared_optional_input_idx = cross_worker_optional_input_tensor_idx](Device* target_device) mutable {
                std::vector<Tensor> input_shards = std::vector<Tensor>(inputs.size(), Tensor());
                std::vector<std::optional<const Tensor>> optional_input_shards = {};
                std::vector<std::optional<Tensor>> optional_output_shards = {};
                // Initialize all optional_outputs to std::nullopt
                optional_output_shards.resize(optional_output_tensors.size());

                {
                    ZoneScopedN("CreateShards");
                    for (int i = 0; i < input_shards.size(); i++) {
                        input_shards[i] = get_shard_for_device(inputs[i], target_device);
                    }

                    for (auto& input : async_safe_optional_input_tensors) {
                        if (input.has_value()) {
                            optional_input_shards.push_back(get_shard_for_device(input.value(), target_device));
                        } else {
                            optional_input_shards.push_back(std::nullopt);
                        }
                    }

                    for (std::size_t optional_output_idx = 0; optional_output_idx < optional_output_tensors.size();
                         optional_output_idx++) {
                        if (optional_output_tensors[optional_output_idx].has_value()) {
                            optional_output_shards[optional_output_idx] = get_shard_for_device(
                                optional_output_tensors[optional_output_idx].value(), target_device);
                        }
                    }
                }

                auto local_tensors = op_func(input_shards, optional_input_shards, optional_output_shards);

                {
                    ZoneScopedN("OpPostProcess");
                    // Release shared ownership of tensors belonging to other workers.
                    // If the workers for this tensor are stalled to deallocate
                    for (auto& shared_input : shared_input_idx) {
                        inputs.at(shared_input).tensor_attributes->num_sibling_workers_sharing_tensor--;
                    }

                    for (auto& shared_optional_input : shared_optional_input_idx) {
                        async_safe_optional_input_tensors.at(shared_optional_input)
                            .value()
                            .tensor_attributes->num_sibling_workers_sharing_tensor--;
                    }

                    for (int i = 0; i < local_tensors.size(); i++) {
                        if (std::holds_alternative<OwnedStorage>(local_tensors.at(i).tensor_attributes->storage)) {
                            TT_ASSERT(
                                outputs.at(i).tensor_attributes->dynamic_storage,
                                "launch_with_autoformat must be used if output tensor for op can be placed on host.");
                            // Make this a host side tensor - Set storage = Owned and clear workers
                            outputs.at(i).tensor_attributes->storage = OwnedStorage();
                            outputs.at(i).workers = {};
                        } else {
                            outputs.at(i).tensor_attributes->dynamic_storage = false;
                        }
                        insert_buffer_and_shape_for_device(target_device, local_tensors.at(i), outputs.at(i));
                        int num_workers_completed = (outputs.at(i).tensor_attributes->num_workers_completed)++;
                        if (not num_workers_completed) {
                            outputs.at(i).tensor_attributes->shape = local_tensors.at(i).tensor_attributes->shape;
                            outputs.at(i).tensor_attributes->dtype = local_tensors.at(i).tensor_attributes->dtype;
                            outputs.at(i).tensor_attributes->layout = local_tensors.at(i).tensor_attributes->layout;
                            outputs.at(i).tensor_attributes->metadata_populated = true;
                        }
                    }
                }
            });

        for (auto target_device : workers) {
            target_device->push_work(std::make_shared<std::function<void()>>(
                [target_device, work_lambda]() mutable { (*work_lambda)(target_device); }));
        }
    }

    // Update ref counts of all tensors after push was performed (done only in main thread).
    for (int i = 0; i < async_safe_input_tensors.size(); i++) {
        async_safe_input_tensors[i].tensor_attributes->update_main_thread_ref_count(
            workers.at(0), input_tensor_ref_count[i]);
    }
    for (int i = 0; i < async_safe_optional_input_tensors.size(); i++) {
        if (async_safe_optional_input_tensors[i].has_value()) {
            async_safe_optional_input_tensors[i].value().tensor_attributes->update_main_thread_ref_count(
                workers.at(0), optional_input_tensor_ref_count[i]);
        }
    }
    for (int i = 0; i < output_tensors.size(); i++) {
        output_tensors[i].tensor_attributes->update_main_thread_ref_count(workers.at(0), output_tensor_ref_count[i]);
    }
    for (int i = 0; i < optional_output_tensors.size(); i++) {
        if (optional_output_tensors[i].has_value()) {
            optional_output_tensors[i].value().tensor_attributes->update_main_thread_ref_count(
                workers.at(0), optional_output_tensor_ref_count[i]);
        }
    }
}

void validate_workers_and_storage(
    const std::vector<Tensor>& inputs,
    const std::vector<std::optional<const Tensor>>& optional_inputs,
    const std::vector<Device*>& workers) {
    bool single_device_storage = false;
    bool multi_device_storage = false;
    // Verify that storage types are consistent - cannot mix single and multi-device storage. For multi-device tensors,
    // ensure that workers are specified, since they cannot be inferred. This means that
    // launch_op/launch_with_autoformat cannot be called with MultiDeviceHostStorage.
    for (const auto& input : inputs) {
        if (std::holds_alternative<DeviceStorage>(input.tensor_attributes->storage) or
            std::holds_alternative<OwnedStorage>(input.tensor_attributes->storage)) {
            single_device_storage |= true;
        } else if (
            std::holds_alternative<MultiDeviceStorage>(input.tensor_attributes->storage) or
            std::holds_alternative<MultiDeviceHostStorage>(input.tensor_attributes->storage)) {
            multi_device_storage |= true;
        }
    }

    for (auto& input : optional_inputs) {
        if (input.has_value()) {
            if (std::holds_alternative<DeviceStorage>(input.value().tensor_attributes->storage) or
                std::holds_alternative<OwnedStorage>(input.value().tensor_attributes->storage)) {
                single_device_storage |= true;
            } else if (
                std::holds_alternative<MultiDeviceStorage>(input.value().tensor_attributes->storage) or
                std::holds_alternative<MultiDeviceHostStorage>(input.value().tensor_attributes->storage)) {
                multi_device_storage |= true;
            }
        }
    }

    TT_FATAL(
        not(single_device_storage and multi_device_storage),
        "Cannot mix single and multi-device tensors when calling launch op!");
    if (multi_device_storage) {
        TT_FATAL(
            workers.size(),
            "Workers must be specified when calling launch_op with with multi-device tensors. Workers cannot be "
            "inferred in this case.");
    }
}

std::vector<Device*> get_workers_for_op_output(
    const std::vector<Tensor>& inputs,
    const std::vector<std::optional<const Tensor>>& optional_inputs,
    bool enable_autoformat_device) {
    std::vector<Device*> workers_for_op = {};
    // Infer output workers from inputs. For multi-device tensors the number
    // of workers used for the op (and assigned to the ouput) is the minimum
    // number of workers across all inputs. Additionally, in this case, at least
    // 1 worker must be specified across all inputs, i.e. host inputs are not allowed.
    size_t min_workers_size = std::numeric_limits<uint32_t>::max();
    for (auto& input : inputs) {
        auto workers = input.get_workers();
        min_workers_size = std::min(min_workers_size, workers.size());
        if (workers.size() == min_workers_size) {
            workers_for_op = workers;
        }
    }

    if (not workers_for_op.size()) {
        for (auto& input : optional_inputs) {
            if (input.has_value()) {
                auto workers = input.value().get_workers();
                min_workers_size = std::min(min_workers_size, workers.size());
                if (workers.size() == min_workers_size) {
                    workers_for_op = workers;
                }
            }
        }
    }
    if (enable_autoformat_device) {
        validate_workers_and_storage(inputs, optional_inputs, workers_for_op);
        // Workers not specified - inputs are on host and not multi-device.
        // Use the default device from autoformat.
        if (not workers_for_op.size()) {
            TT_FATAL(
                AutoFormat::GetDefaultDevice(),
                "Default device must be specified using AutoFormat::SetDefaultDevice, if workers are not specified for "
                "inputs to op.");
            workers_for_op = {AutoFormat::GetDefaultDevice()};
        }
    }
    return workers_for_op;
}
}  // namespace tt::tt_metal::operation
