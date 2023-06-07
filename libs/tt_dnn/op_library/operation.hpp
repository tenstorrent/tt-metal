#pragma once

#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/types.hpp"

#include <experimental/type_traits>

#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt::tt_metal {

namespace operation {

using Hash = std::string; // TODO(arakhmati): switch to an integral type?

using OverrideRuntimeArgsCallback = std::function<void(const std::vector<Buffer*>&, const std::vector<Buffer*>&)>;

struct ProgramWithCallbacks {
    Program program{};
    OverrideRuntimeArgsCallback override_runtime_args_callback = [](auto&& ... args) {};
};


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
        const std::vector<std::reference_wrapper<const Tensor>>
    >{};
}

template<class T, class... Args>
using has_validate_with_optional_input_tensors_t = decltype(std::declval<T>().validate(std::declval<Args>()...));

template<class T>
constexpr bool implements_validate_with_optional_input_tensors() {
    return std::experimental::is_detected<
        has_validate_with_optional_input_tensors_t,
        T,
        const std::vector<std::reference_wrapper<const Tensor>>,
        const std::vector<std::optional<std::reference_wrapper<const Tensor>>>
    >{};
}

template<class T, class... Args>
using hashable_t = decltype(std::declval<T>().compute_program_hash(std::declval<Args>()...));

template<class T>
constexpr bool implements_compute_program_hash() {
    return std::experimental::is_detected<hashable_t, T, const std::vector<std::reference_wrapper<const Tensor>>>{};
}

template<class T, class... Args>
using has_create_program_t = decltype(std::declval<T>().create_program(std::declval<Args>()...));

template<class T>
constexpr bool implements_create_program() {
    return std::experimental::is_detected<
        has_create_program_t,
        T,
        const std::vector<std::reference_wrapper<const Tensor>>,
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
        const std::vector<std::reference_wrapper<const Tensor>>,
        const std::vector<std::optional<std::reference_wrapper<const Tensor>>>,
        std::vector<Tensor>&
    >{};
}

class Operation {
    struct Interface {
        virtual ~Interface() {}

        virtual void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;

        virtual void validate(
            const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
            const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors
        ) const = 0;

        virtual std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;

        virtual std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;

        virtual ProgramWithCallbacks create_program(
            const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
            std::vector<Tensor> &output_tensors
        ) const = 0;

        virtual ProgramWithCallbacks create_program(
            const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
            const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors,
            std::vector<Tensor> &output_tensors
        ) const = 0;

        virtual operation::Hash compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;

        virtual bool supports_program_caching() const = 0;
        virtual std::string get_op_name() const = 0 ;
    };

    template< typename T >
    struct Implementation : Interface {

        Implementation(const T& t) : object(t) {}

        void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            if constexpr (implements_validate<T>()) {
                return this->object.validate(input_tensors);
            } else {
                static_assert(implements_validate_with_optional_input_tensors<T>());
                throw NotImplemented("this operation must take optional input tensors!");
            }
        }

        void validate(
            const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
            const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors
        ) const override {
            if constexpr (implements_validate_with_optional_input_tensors<T>()) {
                return this->object.validate(input_tensors, optional_input_tensors);
            } else {
                static_assert(implements_validate<T>());
                throw NotImplemented("this operation does not take optional input tensors!");
            }
        }

        std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            return this->object.compute_output_shapes(input_tensors);
        }

        std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            return this->object.create_output_tensors(input_tensors);
        }

        ProgramWithCallbacks create_program(
            const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
            std::vector<Tensor> &output_tensors
        ) const override {
            if constexpr (implements_create_program<T>()) {
                return this->object.create_program(input_tensors, output_tensors);
            } else {
                static_assert(implements_create_program_with_optional_input_tensors<T>());
                throw NotImplemented("this operation must take optional input tensors!");
            }
        }

        ProgramWithCallbacks create_program(
            const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
            const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors,
            std::vector<Tensor> &output_tensors
        ) const override {
            if constexpr (implements_create_program_with_optional_input_tensors<T>()) {
                return this->object.create_program(input_tensors, optional_input_tensors, output_tensors);
            } else {
                static_assert(implements_create_program<T>());
                throw NotImplemented("this operation does not take optional input tensors!");
            }
        }

        operation::Hash compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            if constexpr (implements_compute_program_hash<T>()) {
                return this->object.compute_program_hash(input_tensors);
            } else {
                throw NotImplemented("this operation does not implement compute_program_hash!");
            }
        }

        bool supports_program_caching() const override {
            return implements_compute_program_hash<T>();
        }
        
        std::string get_op_name() const {
            return typeid(T).name();
        }

      private:
        T object;
    };

    std::unique_ptr<const Interface> implementation_;

  public:
    template <typename T>
    Operation(T&& operation): implementation_(std::make_unique<Implementation<T>>(std::forward<T>(operation))) {}

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->validate(input_tensors);
    }

    void validate(
        const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
        const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors
    ) const {
        return this->implementation_->validate(input_tensors, optional_input_tensors);
    }

    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->compute_output_shapes(input_tensors);
    }

    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->create_output_tensors(input_tensors);
    }

    ProgramWithCallbacks create_program(
        const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
        std::vector<Tensor> &output_tensors
    ) const {
        return this->implementation_->create_program(input_tensors, output_tensors);
    }

    ProgramWithCallbacks create_program(
        const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
        const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const {
        return this->implementation_->create_program(input_tensors, optional_input_tensors, output_tensors);
    }

    operation::Hash compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->compute_program_hash(input_tensors);
    }

    bool supports_program_caching() const {
        return this->implementation_->supports_program_caching();
    }
    std::string get_op_name() const {
        return this->implementation_->get_op_name();
    }


};

}
}
