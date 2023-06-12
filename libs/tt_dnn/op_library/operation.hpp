#pragma once

#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_pad.hpp"
namespace tt::tt_metal {

using Shape = std::array<uint32_t, 4>;

namespace detail {

static Device* get_device(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) {
    for (auto &input_tensor : input_tensors) {
        if (not input_tensor.get().on_host()) {
            return input_tensor.get().device();
        }
    }
    auto device = AutoPad::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    return device;
}

}

class Operation {
    struct Interface {
        virtual ~Interface() {}

        virtual void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
        virtual std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
        virtual std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
        virtual Program create_program(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors, std::vector<Tensor> &output_tensors) const = 0;
    };

    template< typename T >
    struct Implementation : Interface {
        Implementation(const T& t) : object(t) {}

        void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            return this->object.validate(input_tensors);
        }

        std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            return this->object.compute_output_shapes(input_tensors);
        }

        std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            return this->object.create_output_tensors(input_tensors);
        }

        Program create_program(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors, std::vector<Tensor> &output_tensors) const override {
            return this->object.create_program(input_tensors, output_tensors);
        }

      private:
        T object;
    };

    std::shared_ptr<const Interface> implementation_;

  public:
    template <typename T>
    Operation(T&& operation): implementation_(std::make_shared<Implementation<T>>(std::forward<T>(operation))){}

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->validate(input_tensors);
    }

    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->compute_output_shapes(input_tensors);
    }

    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->create_output_tensors(input_tensors);
    }

    Program create_program(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors, std::vector<Tensor> &output_tensors) const {
        return this->implementation_->create_program(input_tensors, output_tensors);
    }
};

namespace detail {


static std::vector<Tensor> run(const Operation& op, const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) {

    op.validate(input_tensors);

    auto device = detail::get_device(input_tensors);
    auto output_shapes = op.compute_output_shapes(input_tensors);
    auto output_tensors = op.create_output_tensors(input_tensors);
    auto program = op.create_program(input_tensors, output_tensors);

    tt_metal::CompileProgram(device, program);
    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::LaunchKernels(device, program);

    return output_tensors;
}

static std::vector<Tensor> generic_create_output_tensors(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    Layout output_layout = tt::tt_metal::Layout::TILE,
    const MemoryConfig &output_mem_config = MemoryConfig{.interleaved = true}
) {
    const auto& input_tensor = input_tensors.at(0).get();
    const auto& output_shapes = op.compute_output_shapes(input_tensors);

    // HACK to avoid copy constructors when using vectors
    // TODO: If we have default initializers for Tensor, we can do: std::vector<Tensor> output_tensor(num_tensors);
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(tt_metal::Tensor(output_shape, input_tensor.dtype(), output_layout, input_tensor.device(), output_mem_config));
    }
    return output_tensors;
}


template<typename ConcreteOperation>
static Tensor run_without_autopad(ConcreteOperation&& concrete_op, const Tensor &input_tensor, float pad_value = 0) {
    const Operation op = Operation(concrete_op);

    Device* device;
    if (input_tensor.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto output_shape = op.compute_output_shapes({std::cref(input_tensor)}).at(0);
    if (not input_tensor.on_host()) {
        return std::move(run(op, {std::cref(input_tensor)}).at(0));
    } else {
        auto input_tensor_on_dev = input_tensor.to(device);
        return std::move(run(op, {std::cref(input_tensor_on_dev)}).at(0));
    }
}

template<typename ConcreteOperation>
static Tensor run_with_autopad(ConcreteOperation&& concrete_op, const Tensor &input_tensor, float pad_value = 0, bool pad_c=false) {
    const Operation op = Operation(concrete_op);

    Device* device;
    if (input_tensor.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto padded_input_shape = AutoPad::pad_to_tile_shape(input_tensor.shape(), pad_c);
    auto output_shape = op.compute_output_shapes({std::cref(input_tensor)}).at(0);
    if (AutoPad::check_input_tensor_format(input_tensor, padded_input_shape)) {
        return std::move(run(op, {std::cref(input_tensor)}).at(0));
    } else {
        const auto padded_tensor = AutoPad::format_input_tensor(input_tensor, device, padded_input_shape, pad_value);
        auto output_tensor = std::move(run(op, {std::cref(padded_tensor)}).at(0));
        AutoPad::format_output_tensor(input_tensor, output_tensor, output_shape, device);
        return output_tensor;
    }
}


template<typename ConcreteOperation>
static Tensor run_with_autopad(ConcreteOperation&& concrete_op, const Tensor &input_tensor_a, const Tensor &input_tensor_b, float pad_value = 0) {
    const Operation op = Operation(concrete_op);

    Device* device;
    if (input_tensor_a.on_host() && input_tensor_b.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else if (!input_tensor_a.on_host()){
        device = input_tensor_a.device();
    } else {
        device = input_tensor_b.device();
    }

    auto padded_input_shape_a = AutoPad::pad_to_tile_shape(input_tensor_a.shape());
    auto padded_input_shape_b = AutoPad::pad_to_tile_shape(input_tensor_b.shape());
    auto output_shape = op.compute_output_shapes({std::cref(input_tensor_a), std::cref(input_tensor_b)}).at(0);

    auto no_pad_a = AutoPad::check_input_tensor_format(input_tensor_a, padded_input_shape_a);
    auto no_pad_b = AutoPad::check_input_tensor_format(input_tensor_b, padded_input_shape_b);
    if (no_pad_a && no_pad_b) {
        return std::move(run(op, {std::cref(input_tensor_a), std::cref(input_tensor_b)}).at(0));
    } else if (no_pad_a) {
        const auto padded_input_tensor_b = AutoPad::format_input_tensor(input_tensor_b, device, padded_input_shape_b, pad_value);
        auto output_tensor = std::move(run(op, {std::cref(input_tensor_a), std::cref(padded_input_tensor_b)}).at(0));
        AutoPad::format_output_tensor(input_tensor_a, output_tensor, output_shape, device);
        return output_tensor;
    } else if (no_pad_b) {
        const auto padded_input_tensor_a = AutoPad::format_input_tensor(input_tensor_a, device, padded_input_shape_a, pad_value);
        auto output_tensor = std::move(run(op, {std::cref(padded_input_tensor_a), std::cref(input_tensor_b)}).at(0));
        AutoPad::format_output_tensor(input_tensor_a, output_tensor, output_shape, device);
        return output_tensor;
    } else {
        const auto padded_input_tensor_a = AutoPad::format_input_tensor(input_tensor_a, device, padded_input_shape_a, pad_value);
        const auto padded_input_tensor_b = AutoPad::format_input_tensor(input_tensor_b, device, padded_input_shape_b, pad_value);
        auto output_tensor = std::move(run(op, {std::cref(padded_input_tensor_a), std::cref(padded_input_tensor_b)}).at(0));
        AutoPad::format_output_tensor(input_tensor_a, output_tensor, output_shape, device);
        return output_tensor;
    }
}

}

}
