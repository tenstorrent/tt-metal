#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

struct Tilize : Operation {

    Tilize() {
    }

    Tilize(const Tilize&) = delete;
    Tilize& operator=(const Tilize&) = delete;
    ~Tilize() {}

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};

struct TilizeWithZeroPadding : Operation {

    TilizeWithZeroPadding() {
    }

    TilizeWithZeroPadding(const TilizeWithZeroPadding&) = delete;
    TilizeWithZeroPadding& operator=(const TilizeWithZeroPadding&) = delete;
    ~TilizeWithZeroPadding() {}

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};

struct TilizeWithValPadding : Operation {
    const std::array<uint32_t, 4> output_tensor_shape;
    const std::array<uint32_t, 4> input_tensor_start;
    float pad_value;
    TilizeWithValPadding(const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value)
        : output_tensor_shape(output_tensor_shape), input_tensor_start(input_tensor_start), pad_value(pad_value) {
    }

    TilizeWithValPadding(const TilizeWithValPadding&) = delete;
    TilizeWithValPadding& operator=(const TilizeWithValPadding&) = delete;
    ~TilizeWithValPadding() {}

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};

Tensor tilize (const Tensor &a);
Tensor tilize_with_zero_padding (const Tensor &a);
Tensor tilize_with_val_padding(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value);

}  // namespace tt_metal

}  // namespace tt
