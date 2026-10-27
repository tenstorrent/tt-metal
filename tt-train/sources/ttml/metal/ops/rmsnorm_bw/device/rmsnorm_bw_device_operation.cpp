// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_device_operation.hpp"

#include "metal/common/program_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "metal/ops/common/rmsnorm_common.hpp"
#include "rmsnorm_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

void RMSNormBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    constexpr std::string_view op = "RMSNormBackward";
    // The kernels address DRAM banks directly. Outputs the op allocates take the input's, gamma's or gate's memory
    // config.
    const DeviceTensorRequirements in_dram{.buffer_type = tt::tt_metal::BufferType::DRAM};
    const auto& input_tensor = tensor_args.input;
    const auto& gamma_tensor = tensor_args.gamma;
    const auto& rms_tensor = tensor_args.rms;
    const auto& dL_dout_tensor = tensor_args.dL_dout;
    const auto& preallocated_da_tensor = tensor_args.preallocated_da;
    const auto& preallocated_dgamma_components_tensor = tensor_args.preallocated_dgamma_components;
    const auto& preallocated_dgate_tensor = tensor_args.preallocated_dgate;

    check_device_tensor(
        input_tensor,
        op,
        "Input",
        {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32},
         .buffer_type = tt::tt_metal::BufferType::DRAM});
    check_device_tensor(gamma_tensor, op, "Gamma", in_dram);
    check_device_tensor(rms_tensor, op, "RMS", in_dram);
    check_device_tensor(dL_dout_tensor, op, "dL_dout", in_dram);
    rmsnorm::check_input_and_gamma_shapes(input_tensor, gamma_tensor, op);
    check_shape(rms_tensor, op, "RMS", rmsnorm::get_rms_shape(input_tensor));
    if (tensor_args.gate.has_value()) {
        check_device_tensor(tensor_args.gate.value(), op, "Gate", in_dram);
        check_head_merged(input_tensor, tensor_args.gate.value(), op, "Gate");
    }
    const auto& output_shape = rmsnorm::get_output_shape(input_tensor, tensor_args.gate);
    check_shape(dL_dout_tensor, op, "dL_dout", output_shape);
    if (preallocated_da_tensor.has_value()) {
        check_device_tensor(preallocated_da_tensor.value(), op, "Preallocated dL_da", in_dram);
        check_shape(preallocated_da_tensor.value(), op, "Preallocated dL_da", input_tensor.logical_shape());
    }
    if (preallocated_dgamma_components_tensor.has_value()) {
        TT_FATAL(args.compute_dgamma, "{}: Preallocated dL_dgamma_components given but compute_dgamma is false", op);
        check_device_tensor(
            preallocated_dgamma_components_tensor.value(), op, "Preallocated dL_dgamma_components", in_dram);
        check_shape(
            preallocated_dgamma_components_tensor.value(),
            op,
            "Preallocated dL_dgamma_components",
            input_tensor.logical_shape());
    }
    if (preallocated_dgate_tensor.has_value()) {
        TT_FATAL(tensor_args.gate.has_value(), "{}: Preallocated dL_dgate given without a gate", op);
        check_device_tensor(preallocated_dgate_tensor.value(), op, "Preallocated dL_dgate", in_dram);
        check_shape(preallocated_dgate_tensor.value(), op, "Preallocated dL_dgate", output_shape);
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
        tensor_args.gate.has_value() ? get_num_heads_of_head_merged(input_tensor, tensor_args.gate.value()) : 0U;
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
    const std::optional<ttnn::Tensor>& gate,
    bool compute_dgamma,
    const std::optional<ttnn::Tensor>& preallocated_da,
    const std::optional<ttnn::Tensor>& preallocated_dgamma_components,
    const std::optional<ttnn::Tensor>& preallocated_dgate) {
    using OperationType = ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{.compute_dgamma = compute_dgamma};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .rms = rms_tensor,
        .dL_dout = dL_dout_tensor,
        .gate = gate,
        .preallocated_da = preallocated_da,
        .preallocated_dgamma_components = preallocated_dgamma_components,
        .preallocated_dgate = preallocated_dgate,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
