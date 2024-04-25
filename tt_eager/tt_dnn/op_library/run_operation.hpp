// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt_eager/tensor/tensor.hpp>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/operation_history.hpp"
#include "tt_stl/concepts.hpp"
#include "tensor/tensor_utils.hpp"

namespace tt::tt_metal {

namespace operation {

template <typename ConcreteOperation>
auto generic_create_output_tensors(
    const ConcreteOperation& operation,
    const Tensors& input_tensors,
    const std::optional<DataType> output_dtype,
    const Layout output_layout,
    const std::optional<MemoryConfig>& output_mem_config) -> ProgramOutputTensors<ConcreteOperation> {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_shapes = operation.compute_output_shapes(input_tensors);

    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    OutputTensors output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(create_device_tensor(
            output_shape,
            output_dtype.value_or(input_tensors.at(0).get_dtype()),
            output_layout,
            input_tensor.device(),
            output_mem_config.value_or(input_tensors.at(0).memory_config())));
    }
    return output_tensors;
}

namespace run_operation_state {
namespace detail {
struct RunOperationState {

    RunOperationState() {}

    void push_composite_parent_name(const char* parent_name) {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        this->composite_parent_names.push_back(parent_name);
    }

    void pop_composite_parent_name() {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        this->composite_parent_names.pop_back();
    }

    bool is_composite_operation() const {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        return not composite_parent_names.empty();
    }

    const auto& get_composite_parent_names() const {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        return this->composite_parent_names;
    }

  private:
    mutable std::mutex parent_name_mutex;
    std::vector<const char*> composite_parent_names{};
};

inline RunOperationState OPERATION_STATE{};

}  // namespace detail

inline void push_composite_parent_name(const char* parent_name) {
    detail::OPERATION_STATE.push_composite_parent_name(parent_name);
}

inline void pop_composite_parent_name() {
    detail::OPERATION_STATE.pop_composite_parent_name();
}

inline bool is_composite_operation() {
    return detail::OPERATION_STATE.is_composite_operation();
}

inline const auto& get_composite_parent_names() {
    return detail::OPERATION_STATE.get_composite_parent_names();
}

}  // namespace run_operation_state


namespace detail {
template<typename ReturnType, typename... Args>
struct CompositeOperation {

    const char* name;
    std::function<ReturnType(Args...)> function;

    constexpr ReturnType operator()(Args... args) const {
        run_operation_state::push_composite_parent_name(this->name);
        ReturnType output = this->function(args...);
        run_operation_state::pop_composite_parent_name();
        return output;
    }
};

}  // namespace detail

template<typename ReturnType, typename... Args>
constexpr auto decorate_as_composite(const char* name, std::function<ReturnType(Args...)>&& function) {
  return detail::CompositeOperation<ReturnType, Args...>{.name=name, .function=function};
}

template<typename FunctionType>
constexpr auto decorate_as_composite(const char* name, FunctionType function) {
  return decorate_as_composite(name, std::function(function));
}

#ifdef DEBUG
namespace detail {

template <typename OperationType>
std::string operation_type_to_string() {
    if constexpr (std::is_same_v<OperationType, HostOperation<Tensors>>) {
        return "host<Tensors>";
    } else if constexpr (std::is_same_v<OperationType, DeviceOperation<Tensors>>) {
        return "device<Tensors>";
    } else if constexpr (std::is_same_v<OperationType, HostOperation<OptionalTensors>>) {
        return "host<OptionalTensors>";
    } else if constexpr (std::is_same_v<OperationType, DeviceOperation<OptionalTensors>>) {
        return "device<OptionalTensors>";
    } else if constexpr (std::is_same_v<OperationType, ExternalOperation>) {
        return "external";
    } else {
        static_assert(tt::stl::concepts::always_false_v<OperationType>, "OperationType is not supported!");
    }
}

static operation_history::TensorRecord create_tensor_record(const Tensor& tensor) {
    return std::visit(
        [&](const auto& storage) -> operation_history::TensorRecord {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout(), std::nullopt
                };
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout(), tensor.memory_config()};
            }
            else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout()
                };
            }
            else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout()
                };
            }
            else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout()
                };
            } else {
                raise_unsupported_storage<T>();
            }
        },
        tensor.get_storage());
}

template <typename OperationType>
static void append_operation_to_operation_history(
    const std::size_t ttnn_operation_id,
    const OperationType& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors) {

    std::vector<operation_history::TensorRecord> input_tensor_records;
    input_tensor_records.reserve(input_tensors.size() + optional_input_tensors.size());

    for (const auto& tensor : input_tensors) {
        input_tensor_records.emplace_back(create_tensor_record(tensor));
    }
    for (const auto& tensor : optional_input_tensors) {
        if (tensor.has_value()) {
            input_tensor_records.emplace_back(create_tensor_record(tensor.value()));
        }
    }

    std::optional<bool> program_cache_hit = std::nullopt;
    std::optional<tt::stl::hash::hash_t> program_hash = std::nullopt;
    if constexpr (std::is_same_v<OperationType, DeviceOperation<typename OperationType::OutputTensors>>) {
        auto& program_cache = input_tensors[0].device()->program_cache;
        if (program_cache.is_enabled()) {
            program_hash = operation.compute_program_hash(input_tensors, optional_input_tensors);
            auto program_pointer = program_cache.find(program_hash.value());
            program_cache_hit = program_pointer.has_value();
        }
    }

    operation_history::append(operation_history::OperationRecord{
        ttnn_operation_id,
        boost::core::demangle(typeid(OperationType).name()),
        operation.get_type_name(),
        operation.attributes(),
        input_tensor_records,
        run_operation_state::get_composite_parent_names(),
        program_cache_hit,
        program_hash,
    });
}

}  // namespace detail

template<typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {}) {
    tt::log_debug(
        tt::LogOp,
        "Launching Operation: \"{}\" ({})",
        operation.get_type_name(),
        detail::operation_type_to_string<OperationType>());

    if (run_operation_state::is_composite_operation()) {
        tt::log_debug(tt::LogOp, "Composite Parents: {}", run_operation_state::get_composite_parent_names());
    }

    if (not operation.attributes().empty()) {
        tt::log_debug(tt::LogOp, "Attributes:");
        for (auto&& [name, value] : operation.attributes()) {
            tt::log_debug(tt::LogOp, "\t{} = {}", name, value);
        }
    }

    tt::log_debug(tt::LogOp, "Input Tensors:");
    for (auto index = 0; index < input_tensors.size(); index++) {
        const auto& tensor = input_tensors[index];
        tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
    }

    if (not optional_input_tensors.empty()) {
        tt::log_debug(tt::LogOp, "Optional Input Tensors:");
        for (auto index = 0; index < optional_input_tensors.size(); index++) {
            const auto& tensor = optional_input_tensors[index];
            tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
        }
    }

    tt::log_debug(tt::LogOp, "");

    if (operation_history::enabled()) {
        detail::append_operation_to_operation_history(
            ttnn::OPERATION_ID, operation, input_tensors, optional_input_tensors);
    }
}
#else

template <typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {}) {}
#endif

inline uint32_t assign_id()
{
    static std::atomic<uint32_t> atomic_count{0};
    return atomic_count.fetch_add(1);
}

template<class OutputTensors=Tensors>
OutputTensors run(
    const HostOperation<OutputTensors>& operation,
    const Tensors& input_tensors
);

template<class OutputTensors=Tensors>
OutputTensors run(
    CommandQueue& queue,
    const DeviceOperation<OutputTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {});

template<class OutputTensors=Tensors>
OutputTensors run(
    const DeviceOperation<OutputTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {});

template<typename ConcreteOperation>
inline auto run(
    ConcreteOperation&& concrete_op,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors={},
    const OptionalTensors& optional_output_tensors={}
) -> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    if constexpr (detail::is_host_operation<ConcreteOperation>()) {
        TT_ASSERT(optional_input_tensors.empty());
        const auto operation = HostOperation(concrete_op);
        return run<OutputTensors>(operation, input_tensors);
    } else if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto operation = DeviceOperation(concrete_op);
        return run<OutputTensors>(operation, input_tensors, optional_input_tensors, optional_output_tensors);
    } else {
        static_assert(tt::stl::concepts::always_false_v<ConcreteOperation>, "Unsupported Operation");
    }
}

template<class OutputTensors=Tensors>
OutputTensors run_without_autoformat(
    const DeviceOperation<OutputTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {}
);
template <typename ConcreteOperation>
inline auto run_without_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {})
    -> ProgramOutputTensors<ConcreteOperation>{
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    const auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_without_autoformat<OutputTensors>(operation, input_tensors, optional_input_tensors, optional_output_tensors);
}

Tensors run_with_autoformat(
    const DeviceOperation<Tensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
);

template<typename ConcreteOperation>
inline auto run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
)-> Tensors {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    const auto operation = DeviceOperation<Tensors>(concrete_op);
    return run_with_autoformat(operation, input_tensors, optional_input_tensors, pad_value, pad_c);
}

Tensors run_with_autoformat(
    const DeviceOperation<Tensors>& operation,
    const Tensors& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const OptionalConstTensors& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {}
);
template<typename ConcreteOperation>
inline auto run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {}
)-> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    const auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_with_autoformat(operation, input_tensors, input_formatting, output_layouts, optional_input_tensors, optional_input_formatting);
}

void launch_op(
    std::function<std::vector<Tensor>(const Tensors&, const OptionalConstTensors&)>&& op_func,
    const std::vector<Tensor> input_tensors,
    std::vector<Tensor>& output_tensors,
    const std::vector<std::optional<const Tensor>> optional_input_tensors = {}
);

void launch_with_autoformat(
    std::function<std::vector<Tensor>(const std::vector<Tensor>&, const std::vector<std::optional<const Tensor>>&)>&& op_func,
    const std::vector<Tensor> input_tensors,
    std::vector<Tensor>& output_tensors,
    const std::vector<std::optional<const Tensor>> optional_input_tensors = {}
);

std::vector<Device*> get_workers_for_op_output(const std::vector<Tensor>&& inputs, const std::vector<std::optional<const Tensor>>&& optional_inputs = {});

} //namespace operation

bool validate_worker_modes(const std::vector<Device*>& workers);

class TensorMonad {

    static void launch_async(
        std::function<TensorMonad(const TensorMonad&)>&& op_func,
        const TensorMonad& input_monad,
        TensorMonad& output_monad) {
        const OptionalTensors& input_tensors = input_monad.get_tensors();
        OptionalTensors& output_tensors = output_monad.get_tensors();

        // Send host side op compile and run to the worker queue
        // Assert to ensure that worker threads are specified.
        ZoneScopedN("LaunchTensorMonad");
        auto& workers = output_tensors.at(0).value().workers;
        bool run_on_device_threads = true;
        for (const auto& output_tensor : output_tensors) {
            if (output_tensor.has_value()) {
                if (output_tensor.value().workers.size() == 0 || output_tensor.value().workers != workers) {
                    // default to running on the calling thread
                    run_on_device_threads = false;
                    break;
                }
            }
        }
        if (run_on_device_threads) {
            tt::tt_metal::validate_worker_modes(workers);
        }
        // Record ref counts for all tensors before pushing to worker queue.
        std::vector<uint32_t> input_tensor_ref_count = {};
        input_tensor_ref_count.resize(input_tensors.size());
        std::vector<uint32_t> output_tensor_ref_count = {};
        output_tensor_ref_count.resize(output_tensors.size());

        OptionalTensors async_safe_input_tensors = {};
        async_safe_input_tensors.resize(input_tensors.size());
        // When running on a single device, input tensors can be using borrowed storage. If so, when running in async
        // mode, copy borrowed tensors to owned storage.
        for (int i = 0; i < input_tensors.size(); i++) {
            if (input_tensors[i].has_value()) {
                async_safe_input_tensors[i].emplace(tt::tt_metal::copy_borrowed_tensor_in_async_mode(workers.at(0), input_tensors[i].value()));
                input_tensor_ref_count[i] = async_safe_input_tensors[i].value().tensor_attributes->record_main_thread_ref_count();
            }
        }
        for (int i = 0; i < output_tensors.size(); i++) {
            if (output_tensors[i].has_value()) {
                output_tensor_ref_count[i] = output_tensors[i].value().tensor_attributes->record_main_thread_ref_count();
            }
        }
        {
            ZoneScopedN("PushTensorMonadToWorkersOrMainThread");

            if (run_on_device_threads) {
                for (auto target_device : workers) {
                    target_device->push_work([target_device,
                                              workers,
                                              op_func,
                                              inputs = async_safe_input_tensors,
                                              outputs = output_tensors]() mutable {
                        OptionalTensors input_shards = {};
                        input_shards.resize(inputs.size());

                        for (int i=0; i < inputs.size(); i++) {
                            if (inputs[i].has_value()) {
                                input_shards[i] = get_shard_for_device(inputs[i].value(), target_device);
                            }
                        }
                        auto local_monad = op_func(TensorMonad(input_shards));
                        auto local_tensors = local_monad.get_tensors();
                        for (int i = 0; i < local_tensors.size(); i++) {
                            if (local_tensors.at(i).has_value()) {
                                if (local_tensors.at(i).value().storage_type() == StorageType::OWNED) {
                                    TT_ASSERT(
                                        outputs.at(i).value().tensor_attributes->dynamic_storage,
                                        "launch_with_autoformat must be used if output tensor for op can be placed on "
                                        "host.");
                                    TT_ASSERT(
                                        std::holds_alternative<DeviceStorage>(
                                            outputs.at(i).value().tensor_attributes->storage),
                                        "All inputs and outputs to an op must be on device for multi-device tensors.");
                                    // Make this a host side tensor - Set storage = Owned and clear workers
                                    outputs.at(i).value().tensor_attributes->storage = OwnedStorage();
                                    outputs.at(i).value().workers = {};
                                } else {
                                    outputs.at(i).value().tensor_attributes->dynamic_storage = false;
                                }
                                tt::tt_metal::insert_buffer_and_shape_for_device(
                                    target_device, local_tensors.at(i).value(), outputs.at(i).value());
                                if (not target_device->id() or workers.size() == 1) {
                                    outputs.at(i).value().set_shape(local_tensors.at(i).value().get_shape());
                                    outputs.at(i).value().set_dtype(local_tensors.at(i).value().get_dtype());
                                    outputs.at(i).value().set_layout(local_tensors.at(i).value().get_layout());
                                }
                                if (workers.size() == 1) {
                                    outputs.at(i).value().set_populated();
                                } else {
                                    outputs.at(i).value().set_populated(target_device);
                                }
                            }
                        }
                    });
                }
            } else {
                // running on host thread
                auto local_monad = op_func(TensorMonad(async_safe_input_tensors));
                auto local_tensors = local_monad.get_tensors();
                for (int i = 0; i < local_tensors.size(); i++) {
                    if (local_tensors.at(i).has_value()) {
                        if (local_tensors.at(i).value().storage_type() == StorageType::OWNED) {
                            TT_ASSERT(
                                output_tensors.at(i).value().tensor_attributes->dynamic_storage,
                                "launch_with_autoformat must be used if output tensor for op can be placed on "
                                "host.");
                            TT_ASSERT(
                                std::holds_alternative<DeviceStorage>(
                                    output_tensors.at(i).value().tensor_attributes->storage),
                                "All inputs and output_tensors to an op must be on device for multi-device tensors.");
                            // Make this a host side tensor - Set storage = Owned and clear workers
                            output_tensors.at(i).value().tensor_attributes->storage = OwnedStorage();
                            output_tensors.at(i).value().workers = {};
                        } else {
                            output_tensors.at(i).value().tensor_attributes->dynamic_storage = false;
                        }
                        output_tensors.at(i).value().set_shape(local_tensors.at(i).value().get_shape());
                        output_tensors.at(i).value().set_dtype(local_tensors.at(i).value().get_dtype());
                        output_tensors.at(i).value().set_layout(local_tensors.at(i).value().get_layout());
                        output_tensors.at(i).value().set_populated();
                    }
                }
            }
        }

        // Update ref counts of all tensors after push was performed (done only in main thread).
        for (int i = 0; i < async_safe_input_tensors.size(); i++) {
            if (async_safe_input_tensors[i].has_value()) {
                async_safe_input_tensors[i].value().tensor_attributes->update_main_thread_ref_count(
                    workers.at(0), input_tensor_ref_count[i]);
            }
        }
        for (int i = 0; i < output_tensors.size(); i++) {
            if (output_tensors[i].has_value()) {
                output_tensors[i].value().tensor_attributes->update_main_thread_ref_count(
                    workers.at(0), output_tensor_ref_count[i]);
            }
        }
    }

    template <class ConcreteOperation>
    static operation::ProgramOutputTensors<ConcreteOperation> convert_before_run(const OptionalTensors& tensors) {
        if constexpr (std::is_same_v<OptionalTensors, operation::ProgramOutputTensors<ConcreteOperation>>) {
            return tensors;
        } else {
            operation::ProgramOutputTensors<ConcreteOperation> output_tensors;
            for (const auto& tensor : tensors) {
                if (tensor.has_value()) {
                    output_tensors.push_back(tensor.value());
                }
            }
            return output_tensors;
        }
    }

    template <class ConcreteOperation>
    static operation::ProgramOutputTensors<ConcreteOperation> convert_before_run(const Tensors& tensors) {
        if constexpr (std::is_same_v<Tensors, operation::ProgramOutputTensors<ConcreteOperation>>) {
            return tensors;
        } else {
            operation::ProgramOutputTensors<ConcreteOperation> output_tensors;
            for (const auto& tensor : tensors) {
                output_tensors.push_back(tensor);
            }
            return output_tensors;
        }
    }

    template <class ConcreteOperation>
    static OptionalTensors convert_after_run(const operation::ProgramOutputTensors<ConcreteOperation>& tensors) {
        if constexpr (std::is_same_v<OptionalTensors, operation::ProgramOutputTensors<ConcreteOperation>>) {
            return tensors;
        } else {
            OptionalTensors output_tensors;
            for (const auto& tensor : tensors) {
                output_tensors.push_back(tensor);
            }
            return output_tensors;
        }
    }

   OptionalTensors tensors;

   public:
    TensorMonad(const OptionalTensors tensors = {}) : tensors(std::move(tensors)) {}

    TensorMonad(const TensorMonad& other) = default;
    TensorMonad(TensorMonad&& other) noexcept : tensors(std::move(other.tensors)) {}

    TensorMonad& operator=(const TensorMonad& other) = delete;
    TensorMonad& operator=(TensorMonad&& other) = delete;

    template <class ConcreteOperation, class... Args>
    TensorMonad bind(Args&&... args) const {
        auto args_tuple = std::make_tuple(std::forward<Args>(args)...);

        auto output_tensors =
            ConcreteOperation::create_async_output_tensors(this->get_tensors(), std::forward<decltype(args)>(args)...);

        auto output_monad = TensorMonad(std::move(output_tensors));

        // TODO: conditionally turn this off for performance
        std::apply(
            [this](auto&&... packed_args) {
                return ConcreteOperation::validate_api_arguments(
                    get_tensors(), std::forward<decltype(packed_args)>(packed_args)...);
            },
            args_tuple);

        if constexpr (tt::tt_metal::operation::detail::is_device_operation<ConcreteOperation>()) {
            launch_async(
                [args_tuple = std::move(args_tuple)](const TensorMonad& input_monad) mutable -> TensorMonad {
                    return std::apply(
                        [&input_monad](auto&&... packed_args) {
                            return TensorMonad(convert_after_run<ConcreteOperation>(tt::tt_metal::operation::run(
                                ConcreteOperation::create_async_operation(
                                    input_monad.get_tensors(), std::forward<decltype(packed_args)>(packed_args)...),
                                convert_before_run<ConcreteOperation>(input_monad.get_tensors()))));
                        },
                        std::move(args_tuple));
                },
                *this,
                output_monad);

        } else {  // either we are running a host operation or we are chaining device operations
            launch_async(
                [args_tuple = std::move(args_tuple)](const TensorMonad& input_monad) mutable -> TensorMonad {
                    return std::apply(
                        [&input_monad](auto&&... packed_args) {
                            return ConcreteOperation::create_async_operation(
                                input_monad.get_tensors(),
                                std::forward<decltype(packed_args)>(packed_args)...)(input_monad);
                        },
                        std::move(args_tuple));
                },
                *this,
                output_monad);
        }
        return output_monad;
    }

    const OptionalTensors& get_tensors() const { return tensors; }
    OptionalTensors& get_tensors() { return tensors; }
};

} //namespace tt::tt_metal
