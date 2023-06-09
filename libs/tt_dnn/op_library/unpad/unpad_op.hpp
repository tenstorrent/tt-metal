#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

Program unpad_rm(const Tensor &input_tensor_a, Tensor& output_tensor, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end);
Program unpad_tile(const Tensor &input_tensor_a, Tensor& output_tenso, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end);


struct Unpad : Operation {
    const std::array<uint32_t, 4> output_tensor_start;
    const std::array<uint32_t, 4> output_tensor_end;
    const std::array<uint32_t, 4> output_tensor_shape;

    Unpad(const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end)
        : output_tensor_start(output_tensor_start), output_tensor_end(output_tensor_end),
        output_tensor_shape{
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
        } {
    }

    Unpad(const Unpad&) = delete;
    Unpad& operator=(const Unpad&) = delete;
    ~Unpad() {}

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};

inline Tensor unpad(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    const std::array<uint32_t, 4> output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };
    if (input_tensor_a.shape() == output_tensor_shape) {
        log_warning("Perf warning: unpadding called on tensor with same shape as target shape.");
        return input_tensor_a;
    }
    return detail::run_without_autopad(Unpad(output_tensor_start, output_tensor_end), input_tensor_a);

}
}  // namespace tt_metal

}  // namespace tt
