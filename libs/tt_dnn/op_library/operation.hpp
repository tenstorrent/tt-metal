#pragma once

#include "tt_metal/host_api.hpp"

#include <tensor/tensor.hpp>
#include "tt_stl/reflection.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <boost/core/demangle.hpp>

#include <experimental/type_traits>

namespace tt::tt_metal {

namespace operation {

using Hash = std::string; // TODO(arakhmati): switch to an integral type?

using OverrideRuntimeArgsCallback = std::function<void(const std::vector<Buffer*>&, const std::vector<Buffer*>&)>;

struct ProgramWithCallbacks {
    Program program{};
    OverrideRuntimeArgsCallback override_runtime_args_callback = [](auto&& ... args) {};
};

struct ProfilerInfo {
    std::optional<std::string> preferred_name;
    std::optional<std::string> parallelization_strategy;
};


namespace detail {

template <class T>
constexpr std::false_type always_false{};

// TODO: move 'NotImplemented' to a library file
class NotImplemented : public std::logic_error
{
public:
    NotImplemented(const std::string& message) : std::logic_error(message) { };
};

template<class T, class... Args>
using has_validate_t = decltype(std::declval<T>().validate(std::declval<Args>()...));

template<class T>
constexpr bool implements_validate() {
    return std::experimental::is_detected<
        has_validate_t,
        T,
        const std::vector<Tensor>&
    >{};
}

template<class T, class... Args>
using has_validate_with_optional_input_tensors_t = decltype(std::declval<T>().validate(std::declval<Args>()...));

template<class T>
constexpr bool implements_validate_with_optional_input_tensors() {
    return std::experimental::is_detected<
        has_validate_with_optional_input_tensors_t,
        T,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&
    >{};
}

template<class T, class... Args>
using has_compute_program_hash_t = decltype(std::declval<T>().compute_program_hash(std::declval<Args>()...));

template<class T>
constexpr bool implements_compute_program_hash() {
    return std::experimental::is_detected<
        has_compute_program_hash_t,
        T,
        const std::vector<Tensor>&
    >{};
}

template<class T, class... Args>
using has_compute_program_hash_with_optional_input_tensors_t = decltype(std::declval<T>().compute_program_hash(std::declval<Args>()...));

template<class T>
constexpr bool implements_compute_program_hash_with_optional_input_tensors() {
    return std::experimental::is_detected<
        has_compute_program_hash_with_optional_input_tensors_t,
        T,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&
    >{};
}

template<class T, class... Args>
using has_create_program_t = decltype(std::declval<T>().create_program(std::declval<Args>()...));

template<class T>
constexpr bool implements_create_program() {
    return std::experimental::is_detected<
        has_create_program_t,
        T,
        const std::vector<Tensor>&,
        std::vector<Tensor>&
    >{};
}

template<class T, class... Args>
using has_create_program_with_optional_input_tensors_t = decltype(std::declval<T>().create_program(std::declval<Args>()...));

template<class T>
constexpr bool implements_create_program_with_optional_input_tensors() {
    return std::experimental::is_detected<
        has_create_program_with_optional_input_tensors_t,
        T,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&,
        std::vector<Tensor>&
    >{};
}

template<class T>
constexpr bool is_device_operation() {
    return implements_create_program<T>() or implements_create_program_with_optional_input_tensors<T>();
}

template<class T>
constexpr bool is_host_operation() {
    return not is_device_operation<T>();
}

template<class T, class... Args>
using has_get_parallelization_strategy_t = decltype(std::declval<T>().get_parallelization_strategy(std::declval<Args>()...));

template<class T>
constexpr bool implements_get_parallelization_strategy() {
    return std::experimental::is_detected<
        has_get_parallelization_strategy_t,
        T,
        const std::vector<Tensor>&
    >{};
}

}


class HostOperation {
    struct Interface {
        virtual ~Interface() {}

        virtual void validate(const std::vector<Tensor> &input_tensors) const = 0;

        virtual std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const = 0;

        virtual std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const = 0;

        virtual std::string get_type_name() const = 0 ;

        virtual ProfilerInfo create_profiler_info(const std::vector<Tensor> &input_tensors) const = 0;

        virtual std::string to_string() const = 0;

        virtual tt::stl::reflection::Attributes attributes() const = 0;
    };

    template< typename T >
    struct Implementation : Interface {

        explicit Implementation(const T& t) : object(t) {}

        void validate(const std::vector<Tensor> &input_tensors) const override {
            return this->object.validate(input_tensors);
        }

        std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const override {
            return this->object.compute_output_shapes(input_tensors);
        }

        std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const override {
            return this->object.compute_output_tensors(input_tensors);
        }

        std::string get_type_name() const {
            return boost::core::demangle(typeid(T).name());
        }

        ProfilerInfo create_profiler_info(const std::vector<Tensor> &input_tensors) const override {
            std::optional<std::string> preferred_name = this->get_type_name();
            std::optional<std::string> parallelization_strategy = std::nullopt;
            return {
                .preferred_name = preferred_name,
                .parallelization_strategy = parallelization_strategy
            };
        }

        std::string to_string() const override {
            return fmt::format("{}", this->object);
        }

        tt::stl::reflection::Attributes attributes() const {
            return this->object.attributes();
        }

      private:
        const T object;
    };

    std::unique_ptr<const Interface> implementation_;

  public:
    template <typename T>
    explicit HostOperation(T&& operation): implementation_(std::make_unique<Implementation<T>>(std::forward<T>(operation))) {}

    void validate(const std::vector<Tensor> &input_tensors) const {
        return this->implementation_->validate(input_tensors);
    }

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
        return this->implementation_->compute_output_shapes(input_tensors);
    }

    std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const {
        return this->implementation_->compute_output_tensors(input_tensors);
    }

    std::string get_type_name() const {
        return this->implementation_->get_type_name();
    }


    ProfilerInfo create_profiler_info(const std::vector<Tensor> &input_tensors) const {
        return this->implementation_->create_profiler_info(input_tensors);
    }

    std::string to_string() const {
        return this->implementation_->to_string();
    }

    tt::stl::reflection::Attributes attributes() const {
        return this->implementation_->attributes();
    }

};

static std::ostream& operator<<(std::ostream& os, const HostOperation& op) {
    os << op.to_string();
    return os;
}

class DeviceOperation {
    struct Interface {
        virtual ~Interface() {}

        virtual void validate(
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors
        ) const = 0;

        virtual std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const = 0;

        virtual std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const = 0;

        virtual ProgramWithCallbacks create_program(
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            std::vector<Tensor> &output_tensors
        ) const = 0;

        virtual Hash compute_program_hash(
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors
        ) const = 0;

        virtual bool supports_program_caching() const = 0;
        virtual std::string get_type_name() const = 0 ;

        virtual ProfilerInfo create_profiler_info(const std::vector<Tensor> &input_tensors) const = 0;

        virtual std::string to_string() const = 0;

        virtual tt::stl::reflection::Attributes attributes() const = 0;
    };

    template< typename T >
    struct Implementation : Interface {

        explicit Implementation(const T& t) : object(t) {}

        void validate(
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors
        ) const override {
            if constexpr (detail::implements_validate<T>()) {
                TT_ASSERT(optional_input_tensors.empty());
                static_assert(detail::implements_create_program<T>());
                return this->object.validate(input_tensors);
            }
            else if constexpr (detail::implements_validate_with_optional_input_tensors<T>()) {
                TT_ASSERT(not optional_input_tensors.empty());
                static_assert(detail::implements_create_program_with_optional_input_tensors<T>());
                return this->object.validate(input_tensors, optional_input_tensors);
            }
            else {
                static_assert(detail::always_false<T>, "Operation doesn't implement validate");
            }
        }

        std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const override {
            return this->object.compute_output_shapes(input_tensors);
        }

        std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const override {
            return this->object.create_output_tensors(input_tensors);
        }

        ProgramWithCallbacks create_program(
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            std::vector<Tensor> &output_tensors
        ) const override {
            if constexpr (detail::implements_create_program<T>()) {
                TT_ASSERT(optional_input_tensors.empty());
                return this->object.create_program(input_tensors, output_tensors);
            }
            else if constexpr (detail::implements_create_program_with_optional_input_tensors<T>()) {
                TT_ASSERT(not optional_input_tensors.empty());
                return this->object.create_program(input_tensors, optional_input_tensors, output_tensors);
            }
            else {
                static_assert(detail::always_false<T>, "Operation doesn't implement create_program");
                return {};
            }
        }

        Hash compute_program_hash(
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors
        ) const {
            if constexpr (detail::implements_compute_program_hash<T>()) {
                TT_ASSERT(optional_input_tensors.empty());
                static_assert(detail::implements_create_program<T>());
                return this->object.compute_program_hash(input_tensors);
            }
            else if constexpr (detail::implements_compute_program_hash_with_optional_input_tensors<T>()) {
                TT_ASSERT(not optional_input_tensors.empty());
                static_assert(detail::implements_create_program_with_optional_input_tensors<T>());
                return this->object.compute_program_hash(input_tensors, optional_input_tensors);
            }
            else {
                TT_THROW("Operation doesn't implement compute_program_hash");
            }
        }

        bool supports_program_caching() const override {
            constexpr auto result = detail::implements_compute_program_hash<T>() or detail::implements_compute_program_hash_with_optional_input_tensors<T>();
            return result;
        }

        std::string get_type_name() const {
            return boost::core::demangle(typeid(T).name());
        }

        ProfilerInfo create_profiler_info(const std::vector<Tensor> &input_tensors) const override {
            std::optional<std::string> preferred_name = this->get_type_name();

            std::optional<std::string> parallelization_strategy = std::nullopt;
            if constexpr (detail::implements_get_parallelization_strategy<T>()) {
                parallelization_strategy = magic_enum::enum_name(this->object.get_parallelization_strategy(input_tensors));
            }
            return {
                .preferred_name = preferred_name,
                .parallelization_strategy = parallelization_strategy
            };
        }

        std::string to_string() const override {
            return fmt::format("{}", this->object);
        }

        tt::stl::reflection::Attributes attributes() const {
            return this->object.attributes();
        }

      private:
        const T object;
    };

    std::unique_ptr<const Interface> implementation_;

  public:
    template <typename T>
    explicit DeviceOperation(T&& operation): implementation_(std::make_unique<Implementation<T>>(std::forward<T>(operation))) {}

    void validate(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors
    ) const {
        return this->implementation_->validate(input_tensors, optional_input_tensors);
    }

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
        return this->implementation_->compute_output_shapes(input_tensors);
    }

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const {
        return this->implementation_->create_output_tensors(input_tensors);
    }

    ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const {
        return this->implementation_->create_program(input_tensors, optional_input_tensors, output_tensors);
    }

    Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
    ) const {
        return this->implementation_->compute_program_hash(input_tensors, optional_input_tensors);
    }

    bool supports_program_caching() const {
        return this->implementation_->supports_program_caching();
    }

    std::string get_type_name() const {
        return this->implementation_->get_type_name();
    }


    ProfilerInfo create_profiler_info(const std::vector<Tensor> &input_tensors) const {
        return this->implementation_->create_profiler_info(input_tensors);
    }

    std::string to_string() const {
        return this->implementation_->to_string();
    }

    tt::stl::reflection::Attributes attributes() const {
        return this->implementation_->attributes();
    }

};

static std::ostream& operator<<(std::ostream& os, const DeviceOperation& op) {
    os << op.to_string();
    return os;
}

using Operation = std::variant<HostOperation, DeviceOperation>;

}
}
