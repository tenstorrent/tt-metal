#pragma once

#include "infra.hpp"

enum class UnaryType {
    Relu,
};

struct UnaryDeviceOperation {
    struct operation_attributes_t {
        UnaryType unary_type;
        Dtype dtype;
    };

    struct tensor_args_t {
        Tensor input_tensor;
    };

    using tensor_return_value_t = Tensor;

    // select_program_factory
    // validate on program cache miss
    // validate on program cache hit
    // compute output shapes

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        return Tensor{
            tensor_args.input_tensor.shape, attributes.dtype, DeviceStorage{get_device(tensor_args.input_tensor)}};
    }

    static Program create_program(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        return Program{};
    }
};

// Primitive Operation
template <UnaryType type>
struct UnaryOperation {
    using device_operation_t = UnaryDeviceOperation;

    static auto map_args_to_device_operation(const Tensor& input_tensor, Dtype dtype = Dtype::bfloat16) {
        return std::make_tuple(
            device_operation_t::operation_attributes_t{.unary_type = type, .dtype = dtype},
            device_operation_t::tensor_args_t{.input_tensor = input_tensor});
    }
};

constexpr auto relu = register_operation<"relu", UnaryOperation<UnaryType::Relu>>();

enum class BinaryType { Add, Sub };

struct BinaryDeviceOperation {
    struct operation_attributes_t {
        BinaryType binary_type;
        Dtype dtype;
    };

    struct tensor_args_t {
        Tensor input_tensor_a;
        Tensor input_tensor_b;
    };

    using tensor_return_value_t = Tensor;

    // select_program_factory
    // validate on program cache miss
    // validate on program cache hit
    // compute output shapes

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        return Tensor{
            tensor_args.input_tensor_a.shape, attributes.dtype, DeviceStorage{get_device(tensor_args.input_tensor_a)}};
    }

    static Program create_program(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        return Program{};
    }
};

// Primitive Operation
template <BinaryType type>
struct BinaryOperation {
    using device_operation_t = BinaryDeviceOperation;

    static auto map_args_to_device_operation(
        Tensor input_tensor_a, Tensor input_tensor_b, Dtype dtype = Dtype::bfloat16) {
        return std::make_tuple(
            device_operation_t::operation_attributes_t{.binary_type = type, .dtype = dtype},
            device_operation_t::tensor_args_t{.input_tensor_a = input_tensor_a, .input_tensor_b = input_tensor_b});
    }
};

constexpr auto add = register_operation<"add", BinaryOperation<BinaryType::Add>>();

struct MatmulDeviceOperation {
    struct operation_attributes_t {};

    struct tensor_args_t {
        Tensor input_tensor_a;
        Tensor input_tensor_b;
    };

    // select_program_factory
    // validate on program cache miss
    // validate on program cache hit
    // compute output shapes

    using tensor_return_value_t = Tensor;

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        auto output_shape = tensor_args.input_tensor_a.shape;
        output_shape.dims.back() = tensor_args.input_tensor_b.shape.dims.back();
        return Tensor{output_shape, Dtype::bfloat16, DeviceStorage{get_device(tensor_args.input_tensor_a)}};
    }

    static Program create_program(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        return Program{};
    }
};

// Primitive Operation
struct MatmulOperation {
    using device_operation_t = MatmulDeviceOperation;

    static auto map_args_to_device_operation(
        Tensor input_tensor_a, Tensor input_tensor_b, Dtype dtype = Dtype::bfloat16) {
        return std::make_tuple(
            device_operation_t::operation_attributes_t{},
            device_operation_t::tensor_args_t{.input_tensor_a = input_tensor_a, .input_tensor_b = input_tensor_b});
    }
};

constexpr auto matmul = register_operation<"matmul", MatmulOperation>();

// Composite Operation
struct LinearOperation {
    static Tensor operator()(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const auto& bias) {
        auto matmul_output = matmul(input_tensor_a, input_tensor_b);
        return add(matmul_output, bias);
    }
};

constexpr auto linear = register_operation<"linear", LinearOperation>();

// Composite Operation
struct LinearWithReluOperation {
    static Tensor operator()(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const auto& bias) {
        auto output = linear(input_tensor_a, input_tensor_b, bias);
        return relu(output);
    }
};

constexpr auto linear_with_relu = register_operation<"linear_with_relu", LinearWithReluOperation>();

// Composite Operation
struct LinearWithIntermediateOperation {
    static std::tuple<Tensor, Tensor> operator()(
        const Tensor& input_tensor_a, const Tensor& input_tensor_b, const auto& bias) {
        auto output = matmul(input_tensor_a, input_tensor_b);
        auto output_plus_bias = add(output, bias);
        return {output, output_plus_bias};
    }
};

constexpr auto linear_with_intermediate =
    register_operation<"linear_with_intermediate", LinearWithIntermediateOperation>();

struct SplitDeviceOperation {
    struct operation_attributes_t {};

    struct tensor_args_t {
        Tensor input_tensor;
    };

    using tensor_return_value_t = std::vector<Tensor>;

    // select_program_factory
    // validate on program cache miss
    // validate on program cache hit
    // compute output shapes

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        auto output_shape = tensor_args.input_tensor.shape;
        output_shape.dims.back() = output_shape.dims.back() / 2;
        return {
            Tensor{output_shape, tensor_args.input_tensor.dtype, DeviceStorage{get_device(tensor_args.input_tensor)}},
            Tensor{output_shape, tensor_args.input_tensor.dtype, DeviceStorage{get_device(tensor_args.input_tensor)}}};
    }

    static Program create_program(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        return Program{};
    }
};

// Primitive Operation
struct SplitOperation {
    using device_operation_t = SplitDeviceOperation;

    static auto map_args_to_device_operation(const Tensor& input_tensor) {
        return std::make_tuple(
            device_operation_t::operation_attributes_t{},
            device_operation_t::tensor_args_t{.input_tensor = input_tensor});
    }
};

constexpr auto split = register_operation<"split", SplitOperation>();
