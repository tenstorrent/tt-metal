// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <boost/core/demangle.hpp>
#include <experimental/type_traits>
#include <tensor/tensor.hpp>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/program.hpp" /* Need this for GetKernel, which is used in CQ Set/Update Runtime Args Functions */
#include "tt_metal/impl/program/program.hpp"
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

struct OpPerformanceModel {
    int ideal_compute_cycles = 1;
    int ideal_compute_ns = 1;
    int ideal_bandwidth_ns = 1;
    int ideal_ns = 1;
    std::vector<int> inputs_bytes = {};
    std::vector<int> outputs_bytes = {};

    OpPerformanceModel(std::vector<Tensor> input_tensors, std::vector<Tensor> output_tensors, int ideal_compute_cycles) {

        const auto& t = input_tensors.at(0);
        const auto arch = t.storage_type() == StorageType::DEVICE ? t.device()->arch() : ARCH::WORMHOLE_B0;

        this->ideal_compute_cycles = ideal_compute_cycles;

        float clock_rate_ghz = (arch == ARCH::WORMHOLE_B0) ? 1.0 : 1.2;
        this->ideal_compute_ns = std::ceil(ideal_compute_cycles / clock_rate_ghz);

        // GS L1 Bisection bandwidth
        // 655 B/cycle = sqrt(108) * 32 B/cycle * 2
        // 655 * 1.2Ghz = 786 GB/s
        // GS DRAM bandwidth
        // 96 GB/s = 12 GB/s * 8 channels

        // WH L1 Bisection bandwidth
        // 512 B/cycle = sqrt(64) * 32 B/cycle * 2
        // 512 * 1ghz clk
        // WH DRAM bandwidth
        // 258 GB/s = 21.5 GB/s * 6 channels * 2 banks

        float peak_dram_bw = (arch == ARCH::WORMHOLE_B0) ? 6 * 2 * 21.5 : 96.0;

        float noc_l1_bisection_bw = (arch == ARCH::WORMHOLE_B0) ? 512.0 : 786.0;

        auto tensor_ns = [peak_dram_bw, noc_l1_bisection_bw](const Tensor& t) {
            int size_bytes = t.volume() * t.element_size();
            if(t.memory_config().buffer_type == BufferType::DRAM) {
                return size_bytes / peak_dram_bw / 1024 / 1024 / 1024 * 1000 * 1000 * 1000;
            }
            else if(t.memory_config().buffer_type == BufferType::L1) {
                return 1.0f; // TODO: figure out better modelling scheme for L1->L1 Transfers
                //return size_bytes / noc_l1_bisection_bw / 1024 / 1024 / 1024 * 1000 * 1000 * 1000;
            }
            return 0.0f;
        };

        for(const auto & t: input_tensors) {
            this->inputs_bytes.push_back(t.volume() * t.element_size());
            if(tensor_ns(t) > this->ideal_bandwidth_ns) {
                this->ideal_bandwidth_ns = tensor_ns(t);
            }
        }

        for(const auto & t: output_tensors) {
            this->outputs_bytes.push_back(t.volume() * t.element_size());
            if(tensor_ns(t) > this->ideal_bandwidth_ns) {
                this->ideal_bandwidth_ns = tensor_ns(t);
            }
        }

        this->ideal_ns = std::max(this->ideal_compute_ns, this->ideal_bandwidth_ns);
    }
    OpPerformanceModel() = default;
    ~OpPerformanceModel() = default;

    int get_compute_ns() const {
        return this->ideal_compute_ns;
    }
    int get_ideal_ns() const {
        return this->ideal_ns;
    }
    int get_bandwidth_ns() const {
        return this->ideal_bandwidth_ns;
    }
    std::vector<float> get_input_bws() const {
        std::vector<float> input_bws(inputs_bytes.size());
        TT_ASSERT(this->ideal_ns > 0);
        std::transform(inputs_bytes.cbegin(), inputs_bytes.cend(), input_bws.begin(),
                   [this](float c) { return (float)c / this->ideal_ns; });
        return input_bws;
    }
    std::vector<float> get_output_bws() const {
        std::vector<float> output_bws(outputs_bytes.size());
        TT_ASSERT(this->ideal_ns > 0);
        std::transform(outputs_bytes.cbegin(), outputs_bytes.cend(), output_bws.begin(),
                   [this](float c) { return (float)c / this->ideal_ns; });
        return output_bws;
    }

    static int fidelity_multiplier(MathFidelity f) {
        if (MathFidelity::LoFi == f) {
            return 1;
        }
        else if (MathFidelity::HiFi2 == f) {
            return 2;
        }
        else if (MathFidelity::HiFi3 == f) {
            return 3;
        }
        else if (MathFidelity::HiFi4 == f) {
            return 4;
        }

        return 0;
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


template <class T>
constexpr bool implements_validate_with_optional_input_tensors() {
    return std::experimental::is_detected_v<
        has_validate_t,
        T,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&>;
}

template <class T, class... Args>
using has_validate_with_output_tensors_t = decltype(std::declval<T>().validate_with_output_tensors(std::declval<Args>()...));

template <class T>
constexpr bool implements_validate_with_output_tensors() {
    return std::experimental::is_detected_v<
        has_validate_with_output_tensors_t,
        T,
        const std::vector<Tensor>&, // input_tensors
        const std::vector<std::optional<Tensor>>&>; // optional output_tensors
}

template <class T>
constexpr bool implements_validate_with_output_tensors_and_optional_input_tensors() {
    return std::experimental::is_detected_v<
        has_validate_with_output_tensors_t,
        T,
        const std::vector<Tensor>&, // input_tensors
        const std::vector<std::optional<const Tensor>>&, // optional input_tensors
        const std::vector<std::optional<Tensor>>&>; // optional output_tensors
}


template <class T, class... Args>
using has_create_output_tensors_t = decltype(std::declval<T>().create_output_tensors(std::declval<Args>()...));

template <class T>
constexpr bool implements_create_output_tensors() {
    return std::experimental::is_detected_v<has_create_output_tensors_t, T, const std::vector<Tensor>&>;
}

template <class T>
constexpr bool implements_create_output_tensors_with_optional_output_tensors() {
    return std::experimental::is_detected_v<
        has_create_output_tensors_t,
        T,
        const std::vector<Tensor>&,
        const std::vector<std::optional<Tensor>>&>;
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
using has_create_op_performance_model_t =
    decltype(std::declval<T>().create_op_performance_model(std::declval<Args>()...));

template <class T>
constexpr bool implements_create_op_performance_model() {
    return std::experimental::is_detected_v<
        has_create_op_performance_model_t,
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

struct HostOperation final {
    using storage_t = std::array<std::byte, 512>;

    // Methods
    const std::function<const std::string()> get_type_name;
    const std::function<void(const std::vector<Tensor>&)> validate;
    const std::function<const std::vector<Shape>(const std::vector<Tensor>&)> compute_output_shapes;
    const std::function<const std::vector<Tensor>(const std::vector<Tensor>&)> compute_output_tensors;
    const std::function<const ProfilerInfo(const std::vector<Tensor> &input_tensors)> create_profiler_info;
    const std::function<const tt::stl::reflection::Attributes()> attributes;

    template <typename T>
    explicit HostOperation(T&& operation) :

        pointer{new(&type_erased_storage) std::decay_t<T>{std::forward<T>(operation)}},

        delete_storage{[](storage_t& self) {
            using Type = std::decay_t<T>;
            reinterpret_cast<Type*>(&self)->~Type();
        }},

        // Initialize methods
        get_type_name{[]() -> const std::string { return boost::core::demangle(typeid(T).name()); }},
        validate{[this](const std::vector<Tensor>& input_tensors) {
            const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&this->type_erased_storage);
            operation.validate(input_tensors);
        }},
        compute_output_shapes{[this](const std::vector<Tensor>& input_tensors) {
            const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&this->type_erased_storage);
            return operation.compute_output_shapes(input_tensors);
        }},
        compute_output_tensors{[this](const std::vector<Tensor>& input_tensors) {
            const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&this->type_erased_storage);
            return operation.compute_output_tensors(input_tensors);
        }},
        create_profiler_info{[this](const std::vector<Tensor>& input_tensors) -> ProfilerInfo {
            std::optional<std::string> preferred_name = this->get_type_name();
            std::optional<std::string> parallelization_strategy = std::nullopt;
            return {.preferred_name = preferred_name, .parallelization_strategy = parallelization_strategy};
        }},
        attributes{[this] {
            const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&this->type_erased_storage);
            return tt::stl::reflection::get_attributes(operation);
        }} {
        static_assert(sizeof(T) <= sizeof(storage_t));
    }

    HostOperation(const HostOperation&) = delete;
    HostOperation& operator=(const HostOperation&) = delete;

    HostOperation(HostOperation&&) = delete;
    HostOperation& operator=(HostOperation&&) = delete;

   private:
    alignas(32) void* pointer = nullptr;
    alignas(32) storage_t type_erased_storage;

    void (*delete_storage)(storage_t&) = nullptr;
};

struct DeviceOperation final {
    using storage_t = std::array<std::byte, 1152>;

    inline const std::string get_type_name() const { return this->get_type_name_impl_(this->type_erased_storage); }

    inline const void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
        return this->validate_impl_(this->type_erased_storage, input_tensors, optional_input_tensors, optional_output_tensors);
    }

    inline const std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
        return this->compute_output_shapes_impl_(this->type_erased_storage, input_tensors);
    }

    inline const std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
        return this->create_output_tensors_impl_(this->type_erased_storage, input_tensors, output_tensors);
    }

    inline ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const {
        return this->create_program_impl_(
            this->type_erased_storage, input_tensors, optional_input_tensors, output_tensors);
    }

    inline OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const {
        return this->create_op_performance_model_impl_(
            this->type_erased_storage, input_tensors, optional_input_tensors, output_tensors);
    }

    inline void override_runtime_arguments(
        OverrideRuntimeArgumentsCallback& override_runtime_arguments_callback,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const {
        return this->override_runtime_arguments_impl_(
            this->type_erased_storage,
            override_runtime_arguments_callback,
            program,
            input_tensors,
            optional_input_tensors,
            output_tensors);
    }

    inline const Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
        return this->compute_program_hash_impl_(this->type_erased_storage, input_tensors, optional_input_tensors);
    }

    inline const ProfilerInfo create_profiler_info(const std::vector<Tensor>& input_tensors) const {
        return this->create_profiler_info_impl_(this->type_erased_storage, input_tensors);
    }

    inline const tt::stl::reflection::Attributes attributes() const {
        return this->attributes_impl_(this->type_erased_storage);
    }

    template <typename T>
    explicit DeviceOperation(T&& operation) :

        pointer{new(&type_erased_storage) std::decay_t<T>{std::forward<T>(operation)}},

        delete_storage{[](storage_t& self) {
            using Type = std::decay_t<T>;
            reinterpret_cast<Type*>(&self)->~Type();
        }},

        // Initialize methods
        get_type_name_impl_{[](const storage_t& storage) -> const std::string {
            const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
            if constexpr (detail::implements_get_type_name<T>()) {
                return operation.get_type_name();
            } else {
                return boost::core::demangle(typeid(T).name());
            }
        }},
        validate_impl_{
            [](const storage_t& storage,
               const std::vector<Tensor>& input_tensors,
               const std::vector<std::optional<const Tensor>>& optional_input_tensors,
               const std::vector<std::optional<Tensor>>& optional_output_tensors) -> void {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr ((detail::implements_validate<T>() || detail::implements_validate_with_optional_input_tensors<T>()) &&
                        (detail::implements_validate_with_output_tensors<T>() || detail::implements_validate_with_output_tensors_and_optional_input_tensors<T>())){
                    static_assert(tt::stl::concepts::always_false_v<T>, "You cannot implement both validate and validate_with_output_tensors");
                }
                else if constexpr ((detail::implements_validate_with_output_tensors<T>() || detail::implements_validate_with_output_tensors_and_optional_input_tensors<T>())
                    && not detail::implements_create_output_tensors_with_optional_output_tensors<T>()){
                    static_assert(tt::stl::concepts::always_false_v<T>, "Operation doesn't implement create_output_tensors with ant optional output tensors argument when using validate_with_output_tensors");
                }
                else if constexpr(detail::implements_validate<T>() && !detail::implements_create_program<T>()){
                    static_assert(tt::stl::concepts::always_false_v<T>, "Operation doesn't implement both the validate and the correct create_program methods");
                }
                else if constexpr(detail::implements_validate_with_optional_input_tensors<T>() && !detail::implements_create_program_with_optional_input_tensors<T>()){
                    static_assert(tt::stl::concepts::always_false_v<T>, "Operation doesn't implement both the validate and the correct create_program methods with the optional input tensors");
                }

                if constexpr (detail::implements_validate<T>()) {
                    TT_FATAL(optional_input_tensors.empty());
                    operation.validate(input_tensors);
                }
                else if constexpr (detail::implements_validate_with_optional_input_tensors<T>()) {
                    TT_FATAL(not optional_input_tensors.empty());
                    operation.validate(input_tensors, optional_input_tensors);
                }
                else if constexpr (detail::implements_validate_with_output_tensors<T>()){
                    TT_FATAL(optional_input_tensors.empty());
                    TT_FATAL(not optional_output_tensors.empty());
                    operation.validate_with_output_tensors(input_tensors, optional_output_tensors);
                } else if constexpr (detail::implements_validate_with_output_tensors_and_optional_input_tensors<T>()){
                    TT_FATAL(not optional_input_tensors.empty());
                    TT_FATAL(not optional_output_tensors.empty());
                    operation.validate_with_output_tensors(input_tensors, optional_input_tensors, optional_output_tensors);
                }else{
                    static_assert(tt::stl::concepts::always_false_v<T>, "Operation must implement either validate or validate_with_output_tensors");
                }

            }},
        compute_output_shapes_impl_{
            [](const storage_t& storage, const std::vector<Tensor>& input_tensors) -> const std::vector<Shape> {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                return operation.compute_output_shapes(input_tensors);
            }},
        create_output_tensors_impl_{
            [](const storage_t& storage, const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) -> const std::vector<Tensor> {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr (detail::implements_create_output_tensors_with_optional_output_tensors<T>()) {
                    return operation.create_output_tensors(input_tensors, output_tensors);
                }
                else{
                    return operation.create_output_tensors(input_tensors);
                }
            }},
        create_program_impl_{
            [](const storage_t& storage,
               const std::vector<Tensor>& input_tensors,
               const std::vector<std::optional<const Tensor>>& optional_input_tensors,
               std::vector<Tensor>& output_tensors) -> ProgramWithCallbacks {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr (detail::implements_create_program<T>()) {
                    TT_ASSERT(optional_input_tensors.empty());
                    return operation.create_program(input_tensors, output_tensors);
                } else if constexpr (detail::implements_create_program_with_optional_input_tensors<T>()) {
                    TT_ASSERT(not optional_input_tensors.empty());
                    return operation.create_program(input_tensors, optional_input_tensors, output_tensors);
                } else {
                    static_assert(tt::stl::concepts::always_false_v<T>, "Operation doesn't implement create_program");
                }
            }},
        create_op_performance_model_impl_{
            [](const storage_t& storage,
               const std::vector<Tensor>& input_tensors,
               const std::vector<std::optional<const Tensor>>& optional_input_tensors,
               std::vector<Tensor>& output_tensors) -> OpPerformanceModel {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr (detail::implements_create_op_performance_model<T>()) {
                    return operation.create_op_performance_model(input_tensors, optional_input_tensors, output_tensors);
                } else {
                    return OpPerformanceModel(input_tensors, output_tensors, 1); // TODO: account for optional_input_tensors
                }
            }},
        override_runtime_arguments_impl_{
            [](const storage_t& storage,
               OverrideRuntimeArgumentsCallback& override_runtime_arguments_callback,
               Program& program,
               const std::vector<Tensor>& input_tensors,
               const std::vector<std::optional<const Tensor>>& optional_input_tensors,
               std::vector<Tensor>& output_tensors) -> void {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                override_runtime_arguments_callback(
                    &operation, program, input_tensors, optional_input_tensors, output_tensors);
            }},
        compute_program_hash_impl_{
            [](const storage_t& storage,
               const std::vector<Tensor>& input_tensors,
               const std::vector<std::optional<const Tensor>>& optional_input_tensors) -> const Hash {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);

                if constexpr (detail::implements_compute_program_hash<T>()) {
                    static_assert(detail::implements_create_program<T>());
                    TT_ASSERT(optional_input_tensors.empty());
                    return operation.compute_program_hash(input_tensors);
                } else if constexpr (detail::implements_compute_program_hash_with_optional_input_tensors<T>()) {
                    static_assert(detail::implements_create_program_with_optional_input_tensors<T>());
                    TT_ASSERT(not optional_input_tensors.empty());
                    return operation.compute_program_hash(input_tensors, optional_input_tensors);
                } else if constexpr (detail::implements_create_program<T>()) {
                    TT_ASSERT(optional_input_tensors.empty());
                    return hash_operation<T>(operation, input_tensors);
                } else if constexpr (detail::implements_create_program_with_optional_input_tensors<T>()) {
                    TT_ASSERT(not optional_input_tensors.empty());
                    return hash_operation<T>(operation, input_tensors, optional_input_tensors);
                } else {
                    static_assert(tt::stl::concepts::always_false_v<T>, "Operation doesn't implement create_program");
                }
            }},
        create_profiler_info_impl_{
            [](const storage_t& storage, const std::vector<Tensor>& input_tensors) -> const ProfilerInfo {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                std::optional<std::string> preferred_name = boost::core::demangle(typeid(T).name());

                std::optional<std::string> parallelization_strategy = std::nullopt;
                if constexpr (detail::implements_get_parallelization_strategy<T>()) {
                    parallelization_strategy = fmt::format("{}", operation.get_parallelization_strategy(input_tensors));
                }
                return {.preferred_name = preferred_name, .parallelization_strategy = parallelization_strategy};
            }},
        attributes_impl_{[](const storage_t& storage) -> const tt::stl::reflection::Attributes {
            const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
            return tt::stl::reflection::get_attributes(operation);
        }} {
        static_assert(sizeof(T) <= sizeof(storage_t));
    }

    DeviceOperation(const DeviceOperation&) = delete;
    DeviceOperation& operator=(const DeviceOperation&) = delete;

    DeviceOperation(DeviceOperation&&) = delete;
    DeviceOperation& operator=(DeviceOperation&&) = delete;

    ~DeviceOperation() {
        this->delete_storage(this->type_erased_storage);
        this->pointer = nullptr;
    }

   private:
    alignas(32) void* pointer = nullptr;
    alignas(32) storage_t type_erased_storage;

    void (*delete_storage)(storage_t&) = nullptr;

    const std::string (*get_type_name_impl_)(const storage_t& value);
    void (*validate_impl_)(
        const storage_t& value, const std::vector<Tensor>&, const std::vector<std::optional<const Tensor>>&, const std::vector<std::optional<Tensor>>&);
    void (*validate_with_output_tensors_impl_)(
        const storage_t& value, const std::vector<Tensor>&, const std::vector<std::optional<const Tensor>>&, const std::vector<std::optional<Tensor>>&);
    const std::vector<Shape> (*compute_output_shapes_impl_)(const storage_t& value, const std::vector<Tensor>&);
    const std::vector<Tensor> (*create_output_tensors_impl_)(const storage_t& value, const std::vector<Tensor>&, const std::vector<std::optional<Tensor>>&);
    ProgramWithCallbacks (*create_program_impl_)(
        const storage_t& value,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&,
        std::vector<Tensor>&);
    OpPerformanceModel (*create_op_performance_model_impl_)(
        const storage_t& value,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&,
        std::vector<Tensor>&);
    void (*override_runtime_arguments_impl_)(
        const storage_t& value,
        OverrideRuntimeArgumentsCallback&,
        Program&,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&,
        std::vector<Tensor>&);
    const Hash (*compute_program_hash_impl_)(
        const storage_t& value, const std::vector<Tensor>&, const std::vector<std::optional<const Tensor>>&);
    const ProfilerInfo (*create_profiler_info_impl_)(const storage_t& value, const std::vector<Tensor>& input_tensors);
    const tt::stl::reflection::Attributes (*attributes_impl_)(const storage_t& value);
};

struct ExternalOperation {
    const std::string function_name_;
    const tt::stl::reflection::Attributes attributes_;

    const std::string get_type_name() const { return this->function_name_; }
    const tt::stl::reflection::Attributes attributes() const { return this->attributes_; }
};

using Operation = std::variant<HostOperation, DeviceOperation, ExternalOperation>;

}  // namespace operation
}  // namespace tt_metal
}  // namespace tt
