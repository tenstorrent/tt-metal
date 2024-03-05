// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_adam/moreh_adam_op.hpp"

#include <optional>
#include <utility>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

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

void MorehAdam::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(
        input_tensors.size() == 4 and optional_input_tensors.size() <= 1,
        "moreh_adam must have between 4 to 6 input tensors");

    const auto& param = input_tensors.at(0);
    const auto& grad = input_tensors.at(1);
    const auto& exp_avg = input_tensors.at(2);
    const auto& exp_avg_sq = input_tensors.at(3);

    const auto& max_exp_avg_sq = optional_input_tensors.at(0);

    check_tensor(param, "moreh_adam");
    check_tensor(grad, "moreh_adam");
    check_tensor(exp_avg, "moreh_adam");
    check_tensor(exp_avg_sq, "moreh_adam");

    if (max_exp_avg_sq.has_value()) {
        check_tensor(max_exp_avg_sq.value(), "moreh_adam");
    }
}

std::vector<Shape> MorehAdam::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    // inplace
    return {};
}

std::vector<Tensor> MorehAdam::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    // inplace
    return {};
}

tt::stl::reflection::Attributes MorehAdam::attributes() const {
    return {
        {"inplace", this->inplace},
        {"output_mem_config", this->output_mem_config},
    };
}

operation::ProgramWithCallbacks MorehAdam::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {

    const auto& param = input_tensors.at(0);
    const auto& grad = input_tensors.at(1);
    const auto& exp_avg = input_tensors.at(2);
    const auto& exp_avg_sq = input_tensors.at(3);
    const auto& max_exp_avg_sq = optional_input_tensors.at(0);

    return moreh_adam_(
        param, grad, exp_avg, exp_avg_sq,
        this->lr, this->beta1, this->beta2, this->eps, this->weight_decay, this->step, this->amsgrad,
        max_exp_avg_sq);
}

[[maybe_unused]] std::vector<std::variant<Tensor, char*>> moreh_adam(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& exp_avg,
    const Tensor& exp_avg_sq,
    float lr, float beta1, float beta2, float eps, float weight_decay, uint32_t step, bool amsgrad,
    const std::optional<std::reference_wrapper<const Tensor>> max_exp_avg_sq,
    const MemoryConfig& mem_config) {

    std::vector<std::variant<Tensor, char*>> outputs{nullptr, nullptr, nullptr, nullptr, nullptr};

    operation::run(
        MorehAdam{
            .inplace = true,
            .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .weight_decay = weight_decay, .step = step, .amsgrad = amsgrad,
            .output_mem_config = mem_config},
        {param, grad, exp_avg, exp_avg_sq},
        {max_exp_avg_sq});

    outputs[0] = param;
    outputs[1] = grad;
    outputs[2] = exp_avg;
    outputs[3] = exp_avg_sq;
    if (max_exp_avg_sq.has_value()) {
        outputs[4] = max_exp_avg_sq.value();
    }
    return outputs;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
