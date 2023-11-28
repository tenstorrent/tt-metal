// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/backward/backward_ops.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/math.hpp"

namespace tt {

namespace tt_metal {

std::vector<Tensor> _addalpha_bw(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.push_back(grad);
    Tensor grad_b = mul_unary(grad, alpha, output_mem_config);
    grad_tensor.push_back(grad_b);

    return grad_tensor;
}
std::vector<Tensor> addalpha_bw(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addalpha_bw)(grad, input, other, alpha, output_mem_config);
}

std::vector<Tensor> add_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addalpha_bw)(grad, input, other, 1, output_mem_config);
}

std::vector<Tensor> _unary_mul_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul_unary(grad, scalar, output_mem_config);
    grad_tensor.push_back(result);
    return grad_tensor;
}
std::vector<Tensor> unary_mul_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_mul_bw)(grad, input, scalar, output_mem_config);
}


std::vector<Tensor> _mul_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, input_b, std::nullopt, output_mem_config);
    grad_tensor.push_back(grad_a);
    Tensor grad_b = mul(grad, input_a, std::nullopt, output_mem_config);
    grad_tensor.push_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> mul_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _mul_bw)(grad, input_a, input_b, output_mem_config);
}


std::vector<Tensor> _exp_bw(const Tensor& grad, const Tensor& exp_result, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul(grad, exp_result, std::nullopt, output_mem_config);
    grad_tensor.push_back(result);
    return grad_tensor;
}
std::vector<Tensor> exp_bw(const Tensor& grad, const Tensor& exp_result, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _exp_bw)(grad, exp_result, output_mem_config);
}


std::vector<Tensor> _addcmul_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.push_back(grad);
    Tensor grad_a = mul_unary(mul(grad, tensor2, std::nullopt, output_mem_config), value, output_mem_config);
    grad_tensor.push_back(grad_a);
    Tensor grad_b = mul_unary(mul(grad, tensor1, std::nullopt, output_mem_config), value, output_mem_config);
    grad_tensor.push_back(grad_b);

    return grad_tensor;
}
std::vector<Tensor> addcmul_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addcmul_bw)(grad, input, tensor1, tensor2, value, output_mem_config);
}


std::vector<Tensor> _unary_assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.push_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_assign_bw)(grad, input, output_mem_config);
}


}//namespace tt_metal

}//namespace tt
