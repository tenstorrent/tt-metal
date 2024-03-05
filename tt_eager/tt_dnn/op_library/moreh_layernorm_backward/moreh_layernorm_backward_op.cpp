// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp"

#include <functional>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace tt {

namespace operations {

namespace primary {

namespace {
inline void check_tensor(const Tensor& tensor, const std::string& op_name) {
    TT_ASSERT(tensor.get_layout() == Layout::TILE, fmt::format("{} only supports tiled layout.", op_name));
    TT_ASSERT(tensor.get_dtype() == DataType::BFLOAT16, fmt::format("{} only supports bfloat16.", op_name));
    TT_ASSERT(
        tensor.storage_type() == StorageType::DEVICE, fmt::format("Operands to {} need to be on device!", op_name));
    TT_ASSERT(
        tensor.buffer() != nullptr, fmt::format("Operands to {} need to be allocated in buffers on device!", op_name));
}
}  // namespace

// input_grad
void MorehLayerNormBackwardInputGrad::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(
        input_tensors.size() == 5 and optional_input_tensors.size() <= 1,
        "moreh_layernorm_backward_input_grad must have between 5 to 6 input tensors");

    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& mean = input_tensors.at(2);
    const auto& rstd = input_tensors.at(3);
    const auto& input_grad = input_tensors.at(4);

    const auto& gamma = optional_input_tensors.at(0);

    check_tensor(output_grad, "moreh_layernorm_backward_input_grad");
    check_tensor(input, "moreh_layernorm_backward_input_grad");
    check_tensor(mean, "moreh_layernorm_backward_input_grad");
    check_tensor(rstd, "moreh_layernorm_backward_input_grad");
    check_tensor(input_grad, "moreh_layernorm_backward_input_grad");

    TT_ASSERT(this->normalized_dims > 0);
    TT_ASSERT(this->normalized_dims <= output_grad.get_legacy_shape().rank());

    if (gamma.has_value()) {
        check_tensor(gamma.value(), "moreh_layernorm_backward_input_grad");
    }
}

std::vector<Shape> MorehLayerNormBackwardInputGrad::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    // Inplace
    return {};
}

std::vector<Tensor> MorehLayerNormBackwardInputGrad::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    // Inplace
    return {};
}

operation::ProgramWithCallbacks MorehLayerNormBackwardInputGrad::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& mean = input_tensors.at(2);
    const auto& rstd = input_tensors.at(3);
    const auto& input_grad = input_tensors.at(4);

    const auto& gamma = optional_input_tensors.at(0);

    return moreh_layernorm_backward_input_grad_impl(
        output_grad, input, mean, rstd, this->normalized_dims, input_grad, gamma);
}

// gamma_grad and beta_grad
void MorehLayerNormBackwardGammaBetaGrad::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(
        input_tensors.size() == 4 and optional_input_tensors.size() <= 2,
        "moreh_layernorm_backward_gamma_beta_grad must have between 4 to 6 input tensors");

    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& mean = input_tensors.at(2);
    const auto& rstd = input_tensors.at(3);

    const auto& gamma_grad = optional_input_tensors.at(0);
    const auto& beta_grad = optional_input_tensors.at(1);

    check_tensor(output_grad, "moreh_layernorm_backward_gamma_beta_grad");
    check_tensor(input, "moreh_layernorm_backward_gamma_beta_grad");
    check_tensor(mean, "moreh_layernorm_backward_gamma_beta_grad");
    check_tensor(rstd, "moreh_layernorm_backward_gamma_beta_grad");

    TT_ASSERT(this->normalized_dims > 0);
    TT_ASSERT(this->normalized_dims <= output_grad.get_legacy_shape().rank());

    if (gamma_grad.has_value()) {
        check_tensor(gamma_grad.value(), "moreh_layernorm_backward_gamma_beta_grad");
    }

    if (beta_grad.has_value()) {
        check_tensor(beta_grad.value(), "moreh_layernorm_backward_gamma_beta_grad");
    }
}

std::vector<Shape> MorehLayerNormBackwardGammaBetaGrad::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    // Inplace
    return {};
}

std::vector<Tensor> MorehLayerNormBackwardGammaBetaGrad::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    // Inplace
    return {};
}

operation::ProgramWithCallbacks MorehLayerNormBackwardGammaBetaGrad::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& mean = input_tensors.at(2);
    const auto& rstd = input_tensors.at(3);

    auto& gamma_grad = optional_input_tensors.at(0);
    auto& beta_grad = optional_input_tensors.at(1);

    return moreh_layernorm_backward_gamma_beta_grad_impl(
        output_grad, input, mean, rstd, this->normalized_dims, gamma_grad, beta_grad);
}

// input_grad
[[maybe_unused]] Tensor moreh_layernorm_backward_input_grad(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const tt_metal::Tensor& input_grad,
    const std::optional<std::reference_wrapper<const Tensor>> gamma,
    const MemoryConfig& output_mem_config) {
    // Inplace
    operation::run(
        MorehLayerNormBackwardInputGrad{
            .normalized_dims = normalized_dims, .output_mem_config = std::move(output_mem_config)},
        {output_grad, input, mean, rstd, input_grad},
        {gamma});

    return input_grad;
}

// gamma_grad and beta_grad
[[maybe_unused]] std::vector<std::variant<Tensor, char*>> moreh_layernorm_backward_gamma_beta_grad(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<std::reference_wrapper<const Tensor>> gamma_grad,
    const std::optional<std::reference_wrapper<const Tensor>> beta_grad,
    const MemoryConfig& output_mem_config) {
    std::vector<std::variant<Tensor, char*>> outputs{nullptr, nullptr};
    if (!gamma_grad.has_value() && !beta_grad.has_value()) {
        return outputs;
    }

    // Inplace
    operation::run(
        MorehLayerNormBackwardGammaBetaGrad{
            .normalized_dims = normalized_dims, .output_mem_config = std::move(output_mem_config)},
        {output_grad, input, mean, rstd},
        {gamma_grad, beta_grad});

    if (gamma_grad.has_value()) {
        outputs[0] = gamma_grad.value();
    }
    if (beta_grad.has_value()) {
        outputs[1] = beta_grad.value();
    }

    return outputs;
}

// input_grad and gamma_grad and beta_grad
[[maybe_unused]] std::vector<std::variant<Tensor, char*>> moreh_layernorm_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<std::reference_wrapper<const Tensor>> gamma,
    const std::optional<std::reference_wrapper<const Tensor>> input_grad,
    const std::optional<std::reference_wrapper<const Tensor>> gamma_grad,
    const std::optional<std::reference_wrapper<const Tensor>> beta_grad,
    const MemoryConfig& output_mem_config) {
    std::vector<std::variant<Tensor, char*>> outputs;
    outputs.reserve(3);

    // input_grad
    if (input_grad.has_value()) {
        outputs.push_back(moreh_layernorm_backward_input_grad(
            output_grad, input, mean, rstd, normalized_dims, input_grad->get(), gamma, output_mem_config));
    } else {
        outputs.push_back(nullptr);
    }

    // gamma_grad and beta_grad
    const auto& gamma_beta_grad = moreh_layernorm_backward_gamma_beta_grad(
        output_grad, input, mean, rstd, normalized_dims, gamma_grad, beta_grad, output_mem_config);
    outputs.push_back(gamma_beta_grad[0]);
    outputs.push_back(gamma_beta_grad[1]);

    return outputs;
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
