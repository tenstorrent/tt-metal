// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/optimizer/optimizer_ops.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"

#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/concat/concat_op.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace tt {

namespace tt_metal {

// lamb_optimizer
// exp_avg = exp_avg * beta1 + (1 - beta1) * grad
// exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * (grad * grad)
// adam_step = exp_avg / (exp_avg_sq.sqrt() + eps)
// adam_step = adam_step + weight_decay * param
// weight_norm = param.norm(p=2).clamp(0, 10)
// adam_norm = adam_step.norm(p=2)
// trust_ratio = 1.0 if weight_norm == 0 or adam_norm == 0 else weight_norm / (adam_norm + eps)
// trust_ratio = where(greater(w_norm, 0), where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),1.0)
// param = param - step_size * trust_ratio * adam_step
std::vector<Tensor> _lamb_optimizer(const Tensor& data, const Tensor& grad, const Tensor& exp_avg, const Tensor& exp_avg_sq, float beta1, float beta2, float step_size, float eps, float weight_decay, const MemoryConfig& output_mem_config) {
    using namespace tt::tt_metal;
    const float beta1_out = 1.0f - beta1;
    const float beta2_out = 1.0f - beta2;

    std::vector<Tensor> output_tensor;
    Tensor exp_avg_out = ttnn::add(ttnn::multiply(exp_avg, beta1, std::nullopt, output_mem_config), ttnn::multiply(grad, beta1_out, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    Tensor exp_avg_sq_out = ttnn::add(ttnn::multiply(exp_avg_sq, beta2, std::nullopt, output_mem_config),  ttnn::multiply(ttnn::square(grad, output_mem_config), beta2_out, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    Tensor adam_step_mid = ttnn::multiply(exp_avg_out, ttnn::reciprocal(ttnn::add(ttnn::sqrt(exp_avg_sq_out, output_mem_config), eps, std::nullopt, output_mem_config), output_mem_config),  std::nullopt, output_mem_config);
    Tensor adam_step = ttnn::add(adam_step_mid, ttnn::multiply(data, weight_decay, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    auto rmsnorm = [&output_mem_config](Tensor data) -> Tensor {
        Tensor data_val = ttnn::square(data, output_mem_config);
        data_val = global_sum(data_val,output_mem_config);
        Tensor zeros = zeros_like(data, output_mem_config);
        data_val = ttnn::sqrt(ttnn::add(zeros, data_val,  std::nullopt, output_mem_config), output_mem_config);
        return data_val;
    };
    Tensor data_val = rmsnorm(data);
    Tensor weight_norm = clamp(data_val, 0.0f, 10.0f, output_mem_config);

    Tensor adam_norm = rmsnorm(adam_step);
    Tensor ones = ones_like(weight_norm, output_mem_config);

    Tensor trust_ratio_mid = ttnn::multiply(weight_norm, ttnn::reciprocal(ttnn::add(adam_norm, eps, std::nullopt, output_mem_config),output_mem_config), std::nullopt, output_mem_config);
    Tensor trust_ratio = where(ttnn::gtz(weight_norm, output_mem_config), where(ttnn::gtz(adam_norm, output_mem_config), trust_ratio_mid, ones, output_mem_config), ones);

    Tensor param = ttnn::subtract(
        data,
        ttnn::multiply(
            adam_step,
            ttnn::multiply(trust_ratio_mid, step_size, std::nullopt, output_mem_config),
            std::nullopt, output_mem_config),
        std::nullopt, output_mem_config);

    output_tensor.emplace_back(exp_avg_out);
    output_tensor.emplace_back(exp_avg_sq_out);
    output_tensor.emplace_back(param);

    return output_tensor;
}
std::vector<Tensor> lamb_optimizer(const Tensor& data, const Tensor& grad, const Tensor& exp_avg, const Tensor& exp_avg_sq, float beta1, float beta2, float step_size, float eps, float weight_decay, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _lamb_optimizer)(data, grad, exp_avg, exp_avg_sq, beta1, beta2, step_size, eps, weight_decay, output_mem_config);
}

}//namespace tt_metal

}//namespace tt
