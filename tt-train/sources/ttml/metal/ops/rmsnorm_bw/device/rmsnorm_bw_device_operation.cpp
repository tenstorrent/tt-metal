// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_device_operation.hpp"

#include "metal/common/program_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "rmsnorm_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

namespace {

// dL_dout and dL_dgate follow the gate's head-merged layout in gated mode and the input's otherwise.
const ttnn::Shape& get_dL_dout_shape(const tensor_args_t& tensor_args) {
    return tensor_args.gate.has_value() ? tensor_args.gate->logical_shape() : tensor_args.input.logical_shape();
}

}  // namespace

void RMSNormBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_shape = [](const ttnn::Tensor& tensor, const std::string& name, const ttnn::Shape& expected) {
        TT_FATAL(tensor.logical_shape() == expected, "{} shape {} must be {}", name, tensor.logical_shape(), expected);
    };

    const auto& input_tensor = tensor_args.input;
    const auto& gamma_tensor = tensor_args.gamma;
    const auto& rms_tensor = tensor_args.rms;
    const auto& dL_dout_tensor = tensor_args.dL_dout;
    const auto& preallocated_da_tensor = tensor_args.preallocated_da;
    const auto& preallocated_dgamma_components_tensor = tensor_args.preallocated_dgamma_components;
    const auto& preallocated_dgate_tensor = tensor_args.preallocated_dgate;

    check_device_tensor(
        input_tensor,
        "RMSNormBackward",
        "Input",
        {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32}});
    check_device_tensor(gamma_tensor, "RMSNormBackward", "Gamma");
    check_device_tensor(rms_tensor, "RMSNormBackward", "RMS");
    check_device_tensor(dL_dout_tensor, "RMSNormBackward", "dL_dout");
    auto rms_shape = input_tensor.logical_shape();
    rms_shape[-1] = 1U;
    check_shape(rms_tensor, "RMS", rms_shape);
    if (tensor_args.gate.has_value()) {
        check_device_tensor(tensor_args.gate.value(), "RMSNormBackward", "Gate");
        get_num_heads_of_head_merged(input_tensor, tensor_args.gate.value(), "Gate");
    }
    check_shape(dL_dout_tensor, "dL_dout", get_dL_dout_shape(tensor_args));
    if (preallocated_da_tensor.has_value()) {
        check_device_tensor(preallocated_da_tensor.value(), "RMSNormBackward", "Preallocated dL_da");
        check_shape(preallocated_da_tensor.value(), "Preallocated dL_da", input_tensor.logical_shape());
    }
    if (preallocated_dgamma_components_tensor.has_value()) {
        TT_FATAL(args.compute_dgamma, "Preallocated dL_dgamma_components given but compute_dgamma is false");
        check_device_tensor(
            preallocated_dgamma_components_tensor.value(), "RMSNormBackward", "Preallocated dL_dgamma_components");
        check_shape(
            preallocated_dgamma_components_tensor.value(),
            "Preallocated dL_dgamma_components",
            input_tensor.logical_shape());
    }
    if (preallocated_dgate_tensor.has_value()) {
        TT_FATAL(tensor_args.gate.has_value(), "Preallocated dL_dgate given without a gate");
        check_device_tensor(preallocated_dgate_tensor.value(), "RMSNormBackward", "Preallocated dL_dgate");
        check_shape(preallocated_dgate_tensor.value(), "Preallocated dL_dgate", get_dL_dout_shape(tensor_args));
    }
}

spec_return_value_t RMSNormBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto make_spec = [](const ttnn::Shape& shape,
                              const tt::tt_metal::DataType dtype,
                              const tt::tt_metal::MemoryConfig& memory_config) {
        return tt::tt_metal::TensorSpec(
            shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, memory_config));
    };
    spec_return_value_t output_specs(3U);

    if (tensor_args.preallocated_da.has_value()) {
        output_specs[0] = tensor_args.preallocated_da->tensor_spec();
    } else {
        output_specs[0] = make_spec(
            tensor_args.input.logical_shape(), tt::tt_metal::DataType::BFLOAT16, tensor_args.input.memory_config());
    }

    if (args.compute_dgamma) {
        if (tensor_args.preallocated_dgamma_components.has_value()) {
            output_specs[1] = tensor_args.preallocated_dgamma_components->tensor_spec();
        } else {
            // Since we cannot compute dL_dgamma in the kernel, we need to return dL_dgamma_components, which will be
            // reduced outside the kernel. The shape of dL_dgamma_components is the same as the input shape.
            output_specs[1] = make_spec(
                tensor_args.input.logical_shape(), tensor_args.gamma.dtype(), tensor_args.gamma.memory_config());
        }
    }

    if (tensor_args.gate.has_value()) {
        if (tensor_args.preallocated_dgate.has_value()) {
            output_specs[2] = tensor_args.preallocated_dgate->tensor_spec();
        } else {
            output_specs[2] = make_spec(
                tensor_args.gate->logical_shape(), tt::tt_metal::DataType::BFLOAT16, tensor_args.gate->memory_config());
        }
    }

    return output_specs;
}

tensor_return_value_t RMSNormBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const spec_return_value_t output_specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.input.device();
    const auto create = [device](
                            const std::optional<tt::tt_metal::TensorSpec>& spec,
                            const std::optional<ttnn::Tensor>& preallocated) -> std::optional<ttnn::Tensor> {
        if (!spec.has_value()) {
            return std::nullopt;
        }
        if (preallocated.has_value()) {
            return preallocated;
        }
        return ttnn::create_device_tensor(spec.value(), device);
    };

    return {
        create(output_specs[0], tensor_args.preallocated_da),
        create(output_specs[1], tensor_args.preallocated_dgamma_components),
        create(output_specs[2], tensor_args.preallocated_dgate),
    };
}

ttsl::hash::hash_t RMSNormBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    // 0 is the ungated mode, which differs from gated mode with a single head.
    const uint32_t num_heads =
        tensor_args.gate.has_value() ? tensor_args.gate->logical_shape()[-1] / input_logical_shape[-1] : 0U;
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<RMSNormBackwardDeviceOperation>(
        args, input_tensor.dtype(), input_logical_shape, num_heads);

    return hash;
}

}  // namespace ttml::metal::ops::rmsnorm_bw::device

namespace ttnn::prim {

ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation::tensor_return_value_t ttml_rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_da,
    const std::optional<ttnn::Tensor>& preallocated_dgamma_components,
    bool compute_dgamma,
    const std::optional<ttnn::Tensor>& gate,
    const std::optional<ttnn::Tensor>& preallocated_dgate) {
    using OperationType = ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{.epsilon = epsilon, .compute_dgamma = compute_dgamma};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .rms = rms_tensor,
        .dL_dout = dL_dout_tensor,
        .preallocated_da = preallocated_da,
        .preallocated_dgamma_components = preallocated_dgamma_components,
        .gate = gate,
        .preallocated_dgate = preallocated_dgate,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
