// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "transformers/module.hpp"
#include "tt_dnn/op_library/groupnorm/groupnorm_op.hpp"
#include "tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_dnn/op_library/layernorm_distributed/layernorm_pre_allgather_op.hpp"
#include "tt_dnn/op_library/layernorm_distributed/layernorm_post_allgather_op.hpp"
#include "tt_dnn/op_library/moreh_adam/moreh_adam_op.hpp"
#include "tt_dnn/op_library/moreh_adamw/moreh_adamw_op.hpp"
#include "tt_dnn/op_library/moreh_arange/moreh_arange_op.hpp"
#include "tt_dnn/op_library/moreh_bmm/moreh_bmm_op.hpp"
#include "tt_dnn/op_library/moreh_bmm_backward/moreh_bmm_backward_op.hpp"
#include "tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_op.hpp"
#include "tt_dnn/op_library/moreh_cumsum/moreh_cumsum_op.hpp"
#include "tt_dnn/op_library/moreh_getitem/moreh_getitem_op.hpp"
#include "tt_dnn/op_library/moreh_groupnorm/moreh_groupnorm_op.hpp"
#include "tt_dnn/op_library/moreh_groupnorm_backward/moreh_groupnorm_backward_op.hpp"
#include "tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp"
#include "tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp"
#include "tt_dnn/op_library/moreh_linear/moreh_linear_op.hpp"
#include "tt_dnn/op_library/moreh_linear_backward/moreh_linear_backward_op.hpp"
#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_dnn/op_library/moreh_matmul_backward/moreh_matmul_backward_op.hpp"
#include "tt_dnn/op_library/moreh_mean/moreh_mean_op.hpp"
#include "tt_dnn/op_library/moreh_mean_backward/moreh_mean_backward_op.hpp"
#include "tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_op.hpp"
#include "tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward_op.hpp"
#include "tt_dnn/op_library/moreh_norm/moreh_norm_op.hpp"
#include "tt_dnn/op_library/moreh_norm_backward/moreh_norm_backward_op.hpp"
#include "tt_dnn/op_library/moreh_sgd/moreh_sgd_op.hpp"
#include "tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
#include "tt_dnn/op_library/moreh_softmax_backward/moreh_softmax_backward_op.hpp"
#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_op.hpp"
#include "tt_dnn/op_library/prod/prod_nc_op.hpp"
#include "tt_dnn/op_library/prod/prod_op_all.hpp"

namespace py = pybind11;

namespace tt {
namespace operations {
namespace primary {

void py_module(py::module& m_primary) {
    auto m_transformers = m_primary.def_submodule("transformers", "Primary transformers operations");
    transformers::py_module(m_transformers);

    py::class_<LayerNormDefaultProgramConfig>(m_primary, "LayerNormDefaultProgramConfig").def(py::init<>());

    py::class_<LayerNormShardedMultiCoreProgramConfig>(m_primary, "LayerNormShardedMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("subblock_w").noconvert(),
            py::arg("block_h").noconvert(),
            py::arg("block_w").noconvert(),
            py::arg("inplace").noconvert())
        .def(
            "__repr__", [](const LayerNormShardedMultiCoreProgramConfig& config) { return fmt::format("{}", config); });

    m_primary.def(
        "layernorm",
        tt::operations::primary::layernorm,
        py::arg("input").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = LayerNormDefaultProgramConfig{},
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
            Performs a layernorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
        )doc");

    m_primary.def(
        "add_layernorm",
        tt::operations::primary::add_layernorm,
        py::arg("a").noconvert(),
        py::arg("b").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = LayerNormDefaultProgramConfig{},
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
            Performs a layernorm(a+b)*gamma + beta operation.
        )doc");

    m_primary.def(
        "rmsnorm",
        tt::operations::primary::rmsnorm,
        py::arg("input").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = LayerNormDefaultProgramConfig{},
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
            Performs a rmsnorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
        )doc");

    m_primary.def(
        "add_rmsnorm",
        tt::operations::primary::add_rmsnorm,
        py::arg("a").noconvert(),
        py::arg("b").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = LayerNormDefaultProgramConfig{},
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
            Performs a rmsnorm(a+b)*gamma + beta operation.
        )doc");

    m_primary.def(
        "layernorm_pre_allgather",
        tt::operations::primary::layernorm_pre_allgather,
        py::arg("input").noconvert(),
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        py::arg("output_dtype").noconvert() = DataType::BFLOAT16,
        R"doc(
            Performs the first part of a distributed layernorm operation collecting local statistics E(x) and E(xˆ2).
        )doc");

    m_primary.def(
        "rmsnorm_pre_allgather",
        tt::operations::primary::rmsnorm_pre_allgather,
        py::arg("input").noconvert(),
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        py::arg("output_dtype").noconvert() = DataType::BFLOAT16,
        R"doc(
            Performs the first part of a distributed rms norm operation collecting local statistics E(x) and E(xˆ2).
        )doc");

    m_primary.def(
        "layernorm_post_allgather",
        tt::operations::primary::layernorm_post_allgather,
        py::arg("input").noconvert(),
        py::arg("stats").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
            Performs the second part of a distributed layernorm operation normalizing the input based on the gathered statistics input.
        )doc");

    m_primary.def(
        "rmsnorm_post_allgather",
        tt::operations::primary::rmsnorm_post_allgather,
        py::arg("input").noconvert(),
        py::arg("stats").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
            Performs the second part of a distributed rms norm operation normalizing the input based on the gathered statistics input.
        )doc");

    // prod along all dimensions
    m_primary.def(
        "prod_all",
        &prod_all,
        py::arg("input").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Computes prod along along dimensions of the tensor.");

    // moreh_adam
    m_primary.def(
        "moreh_adam",
        &moreh_adam,
        py::arg("param_in").noconvert(),
        py::arg("grad").noconvert(),
        py::arg("exp_avg_in").noconvert(),
        py::arg("exp_avg_sq_in").noconvert(),
        py::arg("lr").noconvert(),
        py::arg("beta1").noconvert(),
        py::arg("beta2").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("weight_decay").noconvert(),
        py::arg("step").noconvert(),
        py::arg("amsgrad").noconvert(),
        py::arg("max_exp_avg_sq_in").noconvert() = std::nullopt,
        py::arg("param_out").noconvert() = std::nullopt,
        py::arg("exp_avg_out").noconvert() = std::nullopt,
        py::arg("exp_avg_sq_out").noconvert() = std::nullopt,
        py::arg("max_exp_avg_sq_out").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_adam operation.
        )doc");

    // moreh_adamw
    m_primary.def(
        "moreh_adamw",
        &moreh_adamw,
        py::arg("param_in").noconvert(),
        py::arg("grad").noconvert(),
        py::arg("exp_avg_in").noconvert(),
        py::arg("exp_avg_sq_in").noconvert(),
        py::arg("lr").noconvert() = 0.001f,
        py::arg("beta1").noconvert() = 0.9f,
        py::arg("beta2").noconvert() = 0.999f,
        py::arg("eps").noconvert() = 1e-8f,
        py::arg("weight_decay").noconvert() = 0.01f,
        py::arg("step").noconvert(),
        py::arg("amsgrad").noconvert() = false,
        py::arg("max_exp_avg_sq_in").noconvert() = std::nullopt,
        py::arg("param_out").noconvert() = std::nullopt,
        py::arg("exp_avg_out").noconvert() = std::nullopt,
        py::arg("exp_avg_sq_out").noconvert() = std::nullopt,
        py::arg("max_exp_avg_sq_out").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_adamw operation.
        )doc");

    // moreh_clip_grad_norm
    m_primary.def(
        "moreh_clip_grad_norm_",
        &moreh_clip_grad_norm,
        py::arg("inputs").noconvert(),
        py::arg("max_norm").noconvert(),
        py::arg("norm_type").noconvert() = 2.0f,
        py::arg("error_if_nonfinite").noconvert() = false,
        py::kw_only(),
        py::arg("total_norm").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        "Performs a moreh_clip_grad_norm operation.
    )doc");

    m_primary.def(
        "moreh_bmm",
        &moreh_bmm,
        py::arg("input").noconvert(),
        py::arg("mat2").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        "Performs a moreh_bmm operation.
    )doc");
    m_primary.def(
        "moreh_bmm_backward",
        &moreh_bmm_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input").noconvert(),
        py::arg("mat2").noconvert(),
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("mat2_grad").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        "Performs a moreh_bmm_backward operation.
    )doc");

    m_primary.def(
        "moreh_linear",
        &moreh_linear,
        py::arg("input").noconvert(),
        py::arg("weight").noconvert(),
        py::kw_only(),
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("output").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_linear operation.
    )doc");

    m_primary.def(
        "moreh_linear_backward",
        &moreh_linear_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input").noconvert(),
        py::arg("weight").noconvert(),
        py::kw_only(),
        py::arg("are_required_outputs").noconvert() = std::vector<bool>{true, true, true},
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("weight_grad").noconvert() = std::nullopt,
        py::arg("bias_grad").noconvert() = std::nullopt,
        py::arg("input_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("weight_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("bias_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_linear_backward operation.
    )doc");

    // moreh_matmul
    m_primary.def(
        "moreh_matmul",
        &moreh_matmul,
        py::arg("input").noconvert(),
        py::arg("other").noconvert(),
        py::kw_only(),
        py::arg("transpose_input") = false,
        py::arg("transpose_other") = false,
        py::arg("output").noconvert() = std::nullopt,
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a moreh_matmul operation.");

    // moreh_matmul_backward
    m_primary.def(
        "moreh_matmul_backward",
        &moreh_matmul_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input_a").noconvert(),
        py::arg("input_b").noconvert(),
        py::arg("are_required_outputs").noconvert() = std::vector<bool>{true, true},
        py::arg("input_a_grad").noconvert() = std::nullopt,
        py::arg("input_b_grad").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_matmul_backward operation.
    )doc");

    // moreh_nll_loss
    m_primary.def(
        "moreh_nll_loss",
        &moreh_nll_loss,
        py::arg("input_tensor").noconvert(),
        py::arg("target_tensor").noconvert(),
        py::arg("weight_tensor").noconvert() = std::nullopt,
        py::arg("divisor_tensor").noconvert() = std::nullopt,
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("ignore_index").noconvert(),
        py::arg("reduction_mean").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a nll_loss operation. Returns an output tensor.");

    // moreh_nll_loss_backward
    m_primary.def(
        "moreh_nll_loss_backward",
        &moreh_nll_loss_backward,
        py::arg("target_tensor").noconvert(),
        py::arg("weight_tensor").noconvert() = std::nullopt,
        py::arg("divisor_tensor").noconvert() = std::nullopt,
        py::arg("output_grad_tensor").noconvert(),
        py::arg("input_grad_tensor").noconvert() = std::nullopt,
        py::arg("ignore_index").noconvert(),
        py::arg("reduction_mean").noconvert(),
        py::arg("input_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a nll_loss_backward operation. Returns an input_grad tensor.");

    // moreh_norm
    m_primary.def(
        "moreh_norm",
        &moreh_norm,
        py::arg("input").noconvert(),
        py::arg("p").noconvert() = 2.0f,
        py::arg("dim").noconvert() = std::nullopt,
        py::kw_only(),
        py::arg("output").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs a moreh_norm operation.");

    // moreh_norm_backward
    m_primary.def(
        "moreh_norm_backward",
        &moreh_norm_backward,
        py::arg("input").noconvert(),
        py::arg("output").noconvert(),
        py::arg("output_grad").noconvert(),
        py::arg("p").noconvert() = 2.0f,
        py::kw_only(),
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("input_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs a moreh_norm_backward operation.");

    m_primary.def(
        "moreh_layernorm",
        &moreh_layernorm,
        py::arg("input").noconvert(),
        py::arg("normalized_dims").noconvert(),
        py::arg("eps").noconvert() = 1e-5f,
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::kw_only(),
        py::arg("output").noconvert() = std::nullopt,
        py::arg("mean").noconvert() = std::nullopt,
        py::arg("rstd").noconvert() = std::nullopt,
        py::arg("memory_config").noconvert() = std::nullopt,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a moreh_layernorm operation.");
    m_primary.def(
        "moreh_layernorm_backward",
        &moreh_layernorm_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input").noconvert(),
        py::arg("mean").noconvert(),
        py::arg("rstd").noconvert(),
        py::arg("normalized_dims").noconvert(),
        py::kw_only(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("gamma_grad").noconvert() = std::nullopt,
        py::arg("beta_grad").noconvert() = std::nullopt,
        py::arg("memory_config").noconvert() = std::nullopt,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a moreh_layernorm_backward operation.");

    m_primary.def(
        "relu",
        &tt::operations::primary::relu,
        py::arg("input").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        Applies the rectified linear unit (ReLU) function to the elements of the input tensor ``input``.

        Input tensor must have TILE layout. Output tensor will have TILE layout.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor RELU is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_primary.def(
        "bcast",
        &tt::operations::primary::bcast,
        py::arg("input_a").noconvert(),
        py::arg("input_b").noconvert(),
        py::arg("math_op"),
        py::arg("dim"),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("in_place") = false,
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("queue_id").noconvert() = 0,
        R"doc(
        Perform a binary elementwise operation ``math_op`` between tensors ``input_a`` and ``input_b``, where values from tensor ``input_b`` are broadcast.

        Let tensor ``input_a`` have shape ``[W0, Z0, Y0, X0]`` and tensor ``input_b`` shape ``[W1, Z1, Y1, X1]``. ``dim`` determines the type of broadcast performed.

        For ``dim=BcastOpDim::W`` broadcast is performed on dimension ``X``. ``Y0`` and ``Y1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``dim=BcastOpDim::H`` broadcast is performed on dimension  ``Y``. ``X0`` and ``X1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

        For ``dim=BcastOpDim::HW`` broadcast is performed on dimensions ``X`` and ``Y``. Either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1) must hold for input shapes.

        Both input tensors must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        Input tensors must have TILE layout. Output tensors will have TILE layout.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input_a", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0], where Y0%32=0 and X0%32=0", "Yes"
            "input_b", "Input tensor to broadcast", "Tensor", "Tensor of shape [W1, Z1, Y1, X1], where Y1%32=0 and X1%32=0", "Yes"
            "math_op", "Aggregating math operation", " BcastOpMath", "ADD, SUB, MUL", "Yes"
            "dim", "Dimension on which to broadcast", "BcastOpDim", "W, H, HW", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            "in_place", "Whether to perform bcast in place, without allocating space for output tensor", "Bool", "Default is false", "No"
            "queue_id", "command queue id", "uint8_t", "Default is 0", "No"
    )doc");

    py::enum_<MorehSoftmaxOpParallelizationStrategy>(m_primary, "MorehSoftmaxOpParallelizationStrategy")
        .value("NONE", MorehSoftmaxOpParallelizationStrategy::NONE)
        .value("SMALL_W", MorehSoftmaxOpParallelizationStrategy::SMALL_W)
        .value("SMALL_H", MorehSoftmaxOpParallelizationStrategy::SMALL_H)
        .value("LARGE_W", MorehSoftmaxOpParallelizationStrategy::LARGE_W)
        .value("LARGE_H", MorehSoftmaxOpParallelizationStrategy::LARGE_H)
        .value("LARGE_C", MorehSoftmaxOpParallelizationStrategy::LARGE_C);
    py::enum_<MorehSoftmaxBackwardOpParallelizationStrategy>(m_primary, "MorehSoftmaxBackwardOpParallelizationStrategy")
        .value("NONE", MorehSoftmaxBackwardOpParallelizationStrategy::NONE)
        .value("SMALL_W", MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W)
        .value("SMALL_H", MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H)
        .value("LARGE_W", MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W)
        .value("LARGE_H", MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H)
        .value("LARGE_C", MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C);
    m_primary.def(
        "moreh_softmax",
        &moreh_softmax,
        py::arg("input_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a softmax operation. Returns an output tensor.");
    m_primary.def(
        "moreh_softmax_backward",
        &moreh_softmax_backward,
        py::arg("output_tensor").noconvert(),
        py::arg("output_grad_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("input_grad_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a softmax backward operation. Returns an input grad tensor.");
    m_primary.def(
        "moreh_softmin",
        &moreh_softmin,
        py::arg("input_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a softmin operation. Returns an output tensor.");
    m_primary.def(
        "moreh_softmin_backward",
        &moreh_softmin_backward,
        py::arg("output_tensor").noconvert(),
        py::arg("output_grad_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("input_grad_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a softmin backward operation. Returns an input grad tensor.");

    m_primary.def(
        "moreh_logsoftmax",
        &moreh_logsoftmax,
        py::arg("input_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a logsoftmax operation. Returns an output tensor.");

    m_primary.def(
        "moreh_logsoftmax_backward",
        &moreh_logsoftmax_backward,
        py::arg("output_tensor").noconvert(),
        py::arg("output_grad_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("input_grad_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a logsoftmax backward operation. Returns an input grad tensor.");

    m_primary.def(
        "moreh_sum",
        &moreh_sum,
        py::arg("input").noconvert(),
        py::kw_only(),
        py::arg("dim").noconvert() = std::nullopt,
        py::arg("keep_batch_dim").noconvert() = false,
        py::arg("output").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs sum operation. Returns an output tensor.");

    m_primary.def(
        "prod_nc",
        &prod_nc,
        py::arg("input").noconvert(),
        py::arg("output").noconvert(),
        py::kw_only(),
        py::arg("dims").noconvert() = std::vector<int64_t>(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs product operation. Returns an output tensor.");

    m_primary.def(
        "moreh_sum_backward",
        &moreh_sum_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input").noconvert(),
        py::kw_only(),
        py::arg("dim").noconvert() = std::nullopt,
        py::arg("keep_batch_dim").noconvert() = false,
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("input_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs sum backward operation. Returns an input_grad tensor.");

    m_primary.def(
        "moreh_cumsum",
        &moreh_cumsum,
        py::arg("input").noconvert(),
        py::arg("output").noconvert(),
        py::kw_only(),
        py::arg("dim").noconvert(),
        "Performs cumsum operation. Returns an output tensor.");
    m_primary.def(
        "moreh_cumsum_backward",
        &moreh_cumsum_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input_grad").noconvert(),
        py::kw_only(),
        py::arg("dim").noconvert(),
        "Performs cumsum backward operation. Returns an input_grad tensor.");

    m_primary.def(
        "moreh_arange",
        &moreh_arange,
        py::arg("start"),
        py::arg("end"),
        py::arg("step"),
        py::arg("any").noconvert(),
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("untilize_out").noconvert() = false,
        py::arg("output_dtype").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs an arange operation. Returns an output tensor.");

    m_primary.def(
        "moreh_sgd",
        &moreh_sgd,
        py::arg("param_in").noconvert(),
        py::arg("grad").noconvert(),
        py::arg("momentum_buffer_in").noconvert() = std::nullopt,
        py::arg("param_out").noconvert() = std::nullopt,
        py::arg("momentum_buffer_out").noconvert() = std::nullopt,
        py::arg("lr").noconvert(),
        py::arg("momentum").noconvert(),
        py::arg("dampening").noconvert(),
        py::arg("weight_decay").noconvert(),
        py::arg("nesterov").noconvert(),
        py::arg("momentum_initialized").noconvert(),
        py::arg("param_out_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("momentum_buffer_out_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a SGD operation.");

    py::class_<GroupNormShardedMultiCoreProgramConfig>(m_primary, "GroupNormShardedMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, MathFidelity, DataType, DataType, bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4,
            py::arg("im_data_format").noconvert() = DataType::BFLOAT16,
            py::arg("out_data_format").noconvert() = DataType::BFLOAT16,
            py::arg("inplace").noconvert() = false);

    m_primary.def(
        "groupnorm",
        &groupnorm,
        py::arg("input").noconvert(),
        py::arg("num_groups").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("input_mask").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = GroupNormShardedMultiCoreProgramConfig{},
        R"doc(
            Performs a groupnorm operation, returna a output tensor the same shape as input.
        )doc");

    // moreh_groupnorm
    m_primary.def(
        "moreh_groupnorm",
        &moreh_groupnorm,
        py::arg("input").noconvert(),
        py::arg("num_groups").noconvert(),
        py::arg("eps").noconvert() = 1e-5f,
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::kw_only(),
        py::arg("are_required_outputs").noconvert() = std::vector<bool>{true, false, false},
        py::arg("output").noconvert() = std::nullopt,
        py::arg("mean").noconvert() = std::nullopt,
        py::arg("rstd").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("mean_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("rstd_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        Performs a moreh_groupnorm operation.
    )doc");

    // moreh_groupnorm_backward
    m_primary.def(
        "moreh_groupnorm_backward",
        &moreh_groupnorm_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input").noconvert(),
        py::arg("mean").noconvert(),
        py::arg("rstd").noconvert(),
        py::arg("num_groups").noconvert(),
        py::kw_only(),
        py::arg("are_required_outputs").noconvert() = std::vector<bool>{true, true, true},
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("gamma_grad").noconvert() = std::nullopt,
        py::arg("beta_grad").noconvert() = std::nullopt,
        py::arg("input_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("gamma_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("beta_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        Performs a moreh_groupnorm_backward operation.
    )doc");

    m_primary.def(
        "moreh_mean",
        &moreh_mean,
        py::arg("input").noconvert(),
        py::arg("output").noconvert(),
        py::kw_only(),
        py::arg("dims").noconvert() = std::vector<int64_t>(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs mean operation. Returns an output tensor.");
    m_primary.def(
        "moreh_mean_backward",
        &moreh_mean_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input_grad").noconvert(),
        "Performs mean backward operation. Returns an input_grad tensor.");

    m_primary.def(
        "moreh_getitem",
        &moreh_getitem,
        py::arg("input_tensor").noconvert(),
        py::arg("index_tensors").noconvert(),
        py::arg("index_dims").noconvert(),
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs a getitem operation. Returns an output tensor.");
}

}  // namespace
   // primary
}  // namespace
   // operations
}  // namespace
   // tt
