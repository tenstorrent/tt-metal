/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>
#include <boost/core/demangle.hpp>
#include <experimental/type_traits>
#include <tensor/tensor.hpp>

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program.hpp"
#include "tt_stl/concepts.hpp"
#include "tt_stl/reflection.hpp"

namespace tt {

namespace tt_metal {
namespace operation {

using Hash = tt::stl::hash::hash_t;

template <typename OperationType, typename... Types>
static Hash hash_operation(const Types&... objects) {
    auto operation_type_hash = typeid(OperationType).hash_code();
    return stl::hash::hash_objects(0, operation_type_hash, objects...);
}

using OverrideAddressesCallback =
    std::function<void(const Program&, const std::vector<Buffer*>&, const std::vector<Buffer*>&)>;

using OverrideRuntimeArgumentsCallback = std::function<void(
    const void* operation,
    Program&,
    const std::vector<Tensor>&,
    const std::vector<std::optional<const Tensor>>&,
    const std::vector<Tensor>&)>;

struct ProgramWithCallbacks {
    Program program{};
    std::optional<OverrideAddressesCallback> override_addresses_callback = std::nullopt;
    std::optional<OverrideRuntimeArgumentsCallback> override_runtime_arguments_callback = std::nullopt;

    bool supports_program_cache() const {
        return this->override_addresses_callback.has_value() or this->override_runtime_arguments_callback.has_value();
    }
};

struct ProfilerInfo {
    std::optional<std::string> preferred_name;
    std::optional<std::string> parallelization_strategy;
};

inline auto DEFAULT_OUTPUT_MEMORY_CONFIG =
    MemoryConfig{.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::DRAM};

static void set_default_operation_output_memory_config(const MemoryConfig& memory_config) {
    DEFAULT_OUTPUT_MEMORY_CONFIG = memory_config;
}

namespace detail {

// TODO: move 'NotImplemented' to a library file
class NotImplemented : public std::logic_error {
   public:
    NotImplemented(const std::string& message) : std::logic_error(message){};
};

template <class T, class... Args>
using has_get_type_name_t = decltype(std::declval<T>().get_type_name(std::declval<Args>()...));

template <class T>
constexpr bool implements_get_type_name() {
    return std::experimental::is_detected_v<has_get_type_name_t, T>;
}
template <class T, class... Args>
using has_validate_t = decltype(std::declval<T>().validate(std::declval<Args>()...));

template <class T>
constexpr bool implements_validate() {
    return std::experimental::is_detected_v<has_validate_t, T, const std::vector<Tensor>&>;
}

template <class T, class... Args>
using has_validate_with_optional_input_tensors_t = decltype(std::declval<T>().validate(std::declval<Args>()...));

template <class T>
constexpr bool implements_validate_with_optional_input_tensors() {
    return std::experimental::is_detected_v<
        has_validate_with_optional_input_tensors_t,
        T,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&>;
}

template <class T, class... Args>
using has_create_program_t = decltype(std::declval<T>().create_program(std::declval<Args>()...));

template <class T>
constexpr bool implements_create_program() {
    return std::experimental::is_detected_v<has_create_program_t, T, const std::vector<Tensor>&, std::vector<Tensor>&>;
}

template <class T, class... Args>
using has_create_program_with_optional_input_tensors_t =
    decltype(std::declval<T>().create_program(std::declval<Args>()...));

template <class T>
constexpr bool implements_create_program_with_optional_input_tensors() {
    return std::experimental::is_detected_v<
        has_create_program_with_optional_input_tensors_t,
        T,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&,
        std::vector<Tensor>&>;
}

template <class T, class... Args>
using has_compute_program_hash_t = decltype(std::declval<T>().compute_program_hash(std::declval<Args>()...));

template <class T>
constexpr bool implements_compute_program_hash() {
    return std::experimental::is_detected_v<has_compute_program_hash_t, T, const std::vector<Tensor>&>;
}

template <class T, class... Args>
using has_compute_program_hash_with_optional_input_tensors_t =
    decltype(std::declval<T>().compute_program_hash(std::declval<Args>()...));

template <class T>
constexpr bool implements_compute_program_hash_with_optional_input_tensors() {
    return std::experimental::is_detected_v<
        has_compute_program_hash_with_optional_input_tensors_t,
        T,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&>;
}

template <class T>
constexpr bool is_device_operation() {
    return implements_create_program<T>() or implements_create_program_with_optional_input_tensors<T>();
}

template <class T>
constexpr bool is_host_operation() {
    return not is_device_operation<T>();
}

template <class T, class... Args>
using has_get_parallelization_strategy_t =
    decltype(std::declval<T>().get_parallelization_strategy(std::declval<Args>()...));

template <class T>
constexpr bool implements_get_parallelization_strategy() {
    return std::experimental::is_detected_v<has_get_parallelization_strategy_t, T, const std::vector<Tensor>&>;
}

}  // namespace detail

struct HostOperation {

    const std::any type_erased_operation;

    // Methods
    const std::function<const std::string()> get_type_name;
    const std::function<void(const std::vector<Tensor>&)> validate;
    const std::function<const std::vector<Shape>(const std::vector<Tensor>&)> compute_output_shapes;
    const std::function<const std::vector<Tensor>(const std::vector<Tensor>&)> compute_output_tensors;
    const std::function<const ProfilerInfo(const std::vector<Tensor> &input_tensors)> create_profiler_info;
    const std::function<const tt::stl::reflection::Attributes()> attributes;

    template <typename T>
    explicit HostOperation(const T operation) :
        // Initialize attributes
        type_erased_operation(operation),

        // Initialize methods
        get_type_name{[]() -> const std::string { return boost::core::demangle(typeid(T).name()); }},
        validate{[this](const std::vector<Tensor>& input_tensors) {
            auto operation = std::any_cast<const T&>(this->type_erased_operation);
            operation.validate(input_tensors);
        }},
        compute_output_shapes{[this](const std::vector<Tensor>& input_tensors) {
            auto operation = std::any_cast<const T&>(this->type_erased_operation);
            return operation.compute_output_shapes(input_tensors);
        }},
        compute_output_tensors{[this](const std::vector<Tensor>& input_tensors) {
            auto operation = std::any_cast<const T&>(this->type_erased_operation);
            return operation.compute_output_tensors(input_tensors);
        }},
        create_profiler_info{[this](const std::vector<Tensor>& input_tensors) -> ProfilerInfo {
            std::optional<std::string> preferred_name = this->get_type_name();
            std::optional<std::string> parallelization_strategy = std::nullopt;
            return {.preferred_name = preferred_name, .parallelization_strategy = parallelization_strategy};
        }},
        attributes{[this] {
            auto operation = std::any_cast<const T&>(this->type_erased_operation);
            return operation.attributes();
        }} {}
};

struct DeviceOperation {

    const std::any type_erased_operation;

    // Methods
    const std::function<const std::string()> get_type_name;
    const std::function<void(const std::vector<Tensor>&, const std::vector<std::optional<const Tensor>>&)> validate;
    const std::function<const std::vector<Shape>(const std::vector<Tensor>&)> compute_output_shapes;
    const std::function<const std::vector<Tensor>(const std::vector<Tensor>&)> create_output_tensors;
    const std::function<ProgramWithCallbacks(const std::vector<Tensor>&, const std::vector<std::optional<const Tensor>>&, std::vector<Tensor>&)> create_program;
    const std::function<void(OverrideRuntimeArgumentsCallback&, Program&, const std::vector<Tensor>&, const std::vector<std::optional<const Tensor>>&, std::vector<Tensor>&)> override_runtime_arguments;
    const std::function<const Hash(const std::vector<Tensor>&, const std::vector<std::optional<const Tensor>>&)> compute_program_hash;
    const std::function<const ProfilerInfo(const std::vector<Tensor> &input_tensors)> create_profiler_info;
    const std::function<const tt::stl::reflection::Attributes()> attributes;

    template <typename T>
    explicit DeviceOperation(const T operation) :
        // Initialize attributes
        type_erased_operation(operation),

        // Initialize methods
        get_type_name{
            [this] () -> const std::string {
                auto operation = std::any_cast<const T&>(this->type_erased_operation);
                if constexpr (detail::implements_get_type_name<T>()) {
                    return operation.get_type_name();
                }
                else {
                    return boost::core::demangle(typeid(T).name());
                }
            }
        },
        validate{
            [this] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) {
                auto operation = std::any_cast<const T&>(this->type_erased_operation);
                if constexpr (detail::implements_validate<T>()) {
                    TT_ASSERT(optional_input_tensors.empty());
                    static_assert(detail::implements_create_program<T>());
                    return operation.validate(input_tensors);
                }
                else if constexpr (detail::implements_validate_with_optional_input_tensors<T>()) {
                    TT_ASSERT(not optional_input_tensors.empty());
                    static_assert(detail::implements_create_program_with_optional_input_tensors<T>());
                    return operation.validate(input_tensors, optional_input_tensors);
                }
                else {
                    static_assert(tt::stl::concepts::always_false_v<T>, "Operation doesn't implement validate");
                }
            }
        },
        compute_output_shapes{
            [this] (const std::vector<Tensor>& input_tensors) {
                auto operation = std::any_cast<const T&>(this->type_erased_operation);
                return operation.compute_output_shapes(input_tensors);
            }
        },
        create_output_tensors{
            [this] (const std::vector<Tensor>& input_tensors) {
                auto operation = std::any_cast<const T&>(this->type_erased_operation);
                return operation.create_output_tensors(input_tensors);
            }
        },
        create_program{
            [this] (
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                std::vector<Tensor>& output_tensors
            ) {
                auto operation = std::any_cast<const T&>(this->type_erased_operation);
                if constexpr (detail::implements_create_program<T>()) {
                    TT_ASSERT(optional_input_tensors.empty());
                    return operation.create_program(input_tensors, output_tensors);
                }
                else if constexpr (detail::implements_create_program_with_optional_input_tensors<T>()) {
                    TT_ASSERT(not optional_input_tensors.empty());
                    return operation.create_program(input_tensors, optional_input_tensors, output_tensors);
                }
                else {
                    static_assert(tt::stl::concepts::always_false_v<T>, "Operation doesn't implement create_program");
                }
            }
        },
        override_runtime_arguments{
            [this] (
                OverrideRuntimeArgumentsCallback& override_runtime_arguments_callback,
                Program& program,
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                std::vector<Tensor>& output_tensors
            ) {
                auto operation = std::any_cast<const T&>(this->type_erased_operation);
                override_runtime_arguments_callback(&operation, program, input_tensors, optional_input_tensors, output_tensors);
            }
        },
        compute_program_hash{
            [this] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) -> const Hash {
                auto operation = std::any_cast<const T&>(this->type_erased_operation);

                if constexpr (detail::implements_compute_program_hash<T>()) {
                    static_assert(detail::implements_create_program<T>());
                    TT_ASSERT(optional_input_tensors.empty());
                    return operation.compute_program_hash(input_tensors);
                }
                else if constexpr (detail::implements_compute_program_hash_with_optional_input_tensors<T>()) {
                    static_assert(detail::implements_create_program_with_optional_input_tensors<T>());
                    TT_ASSERT(not optional_input_tensors.empty());
                    return operation.compute_program_hash(input_tensors, optional_input_tensors);
                }
                else if constexpr (detail::implements_create_program<T>()) {
                    TT_ASSERT(optional_input_tensors.empty());
                    return hash_operation<T>(operation, input_tensors);
                }
                else if constexpr (detail::implements_create_program_with_optional_input_tensors<T>()) {
                    TT_ASSERT(not optional_input_tensors.empty());
                    return hash_operation<T>(operation, input_tensors, optional_input_tensors);
                }
                else {
                    static_assert(tt::stl::concepts::always_false_v<T>, "Operation doesn't implement create_program");
                }
            }
        },
        create_profiler_info{
            [this] (const std::vector<Tensor>& input_tensors) -> ProfilerInfo {
                auto operation = std::any_cast<const T&>(this->type_erased_operation);
                std::optional<std::string> preferred_name = this->get_type_name();

                std::optional<std::string> parallelization_strategy = std::nullopt;
                if constexpr (detail::implements_get_parallelization_strategy<T>()) {
                    parallelization_strategy = fmt::format("{}", operation.get_parallelization_strategy(input_tensors));
                }
                return {
                    .preferred_name = preferred_name,
                    .parallelization_strategy = parallelization_strategy
                };
            }
        },
        attributes{
            [this] {
                auto operation = std::any_cast<const T&>(this->type_erased_operation);
                return operation.attributes();
            }
        }
        {}

};

struct ExternalOperation {
    std::string function_name_;
    tt::stl::reflection::Attributes attributes_;

    std::string get_type_name() const { return fmt::format("{}", this->function_name_); }
    tt::stl::reflection::Attributes attributes() const { return this->attributes_; }
};

using Operation = std::variant<HostOperation, DeviceOperation, ExternalOperation>;

}  // namespace operation
}  // namespace tt_metal
}  // namespace tt
