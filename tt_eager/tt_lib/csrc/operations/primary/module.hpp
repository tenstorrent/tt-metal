// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "transformers/module.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp"
#include "tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp"
#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_dnn/op_library/moreh_matmul_backward/moreh_matmul_backward_op.hpp"
#include "tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
#include "tt_dnn/op_library/moreh_softmax_backward/moreh_softmax_backward_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"

namespace py = pybind11;

namespace tt {
namespace operations {
namespace primary {

void py_module(py::module& m_primary) {
    auto m_transformers = m_primary.def_submodule("transformers", "Primary transformers operations");
    transformers::py_module(m_transformers);

    py::class_<MatmulProgramConfig>(m_primary, "MatmulProgramConfig");

    py::class_<MatmulDefaultProgramConfig>(m_primary, "MatmulDefaultProgramConfig").def(py::init<>());

    py::class_<MatmulMultiCoreReuseProgramConfig>(m_primary, "MatmulMultiCoreReuseProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert());
    py::class_<MatmulMultiCoreReuseMultiCastProgramConfig>(m_primary, "MatmulMultiCoreReuseMultiCastProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, bool, std::optional<UnaryWithParam>>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("transpose_mcast").noconvert(),
            py::arg("fused_activation")
        )
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCastProgramConfig::fused_activation);

    py::class_<MatmulMultiCoreReuseMultiCast1DProgramConfig>(m_primary, "MatmulMultiCoreReuseMultiCast1DProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, bool, std::optional<UnaryWithParam>, bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("fuse_batch").noconvert(),
            py::arg("fused_activation"),
            py::arg("mcast_in0").noconvert())
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fused_activation);

    m_primary.def(
        "get_mcast_1d_config",
        &bmm_op_utils::get_mcast_1d_config,
        py::arg("input_tensor_a").noconvert(),
        py::arg("input_tensor_b").noconvert(),
        py::arg("fuse_batch").noconvert() = false,
        py::arg("fused_activation") = std::nullopt,
        py::arg("mcast_in0").noconvert() = true,
        py::arg("out_sharded").noconvert() = false);

    // TODO(arakhmati):
    // delete
    // redundant
    // matmul
    // overrides
    // by
    // figuring
    // out
    // how
    // to
    // pass
    // in
    // MatmulProgramConfig
    // (which
    // is
    // a
    // std::variant)
    m_primary.def(
        "matmul",
        [](const Tensor& input_tensor_a,
           const Tensor& input_tensor_b,
           const MatmulDefaultProgramConfig& program_config,
           const MemoryConfig& out_mem_config,
           std::optional<DataType> output_dtype,
           const MathFidelity math_fidelity,
           const bool fp32_dest_acc_en,
           const bool math_approx_mode,
           const bool packer_l1_acc) {
            return matmul(input_tensor_a, input_tensor_b, program_config, out_mem_config, output_dtype, math_fidelity, fp32_dest_acc_en, math_approx_mode, packer_l1_acc);
        },
        py::arg("input_tensor_a").noconvert(),
        py::arg("input_tensor_b").noconvert(),
        py::kw_only(),
        py::arg("program_config").noconvert() = MatmulDefaultProgramConfig(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("output_dtype").noconvert() = std::nullopt,
        py::arg("math_fidelity").noconvert() = MathFidelity::LoFi,
        py::arg("fp32_dest_acc_en").noconvert() = false,
        py::arg("math_approx_mode").noconvert() = true,
        py::arg("packer_l1_acc").noconvert() = false,
        R"doc(
            Perform a matrix multiplication ``input_tensor_a x input_tensor_b``.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_tensor_a",    "First tensor to multiply",                               "Tensor",                                     "Tensor of shape [B_a, C_a, M, K]",                               "Yes"
                "input_tensor_b",    "Second tensor to multiply",                              "Tensor",                                     "Tensor of shape [B_b, C_b, K, N]",                               "Yes"
                "program_config",    "",                                                       "MatmulDefaultProgramConfig",          "",                                                               "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig",                               "Default is interleaved in DRAM",                                 "No"
                "output_dtype",      "Output Data Type",                                       "DataType",                                   "By default it will be set to the data type of `input_tensor_a`", "No"
        )doc");

    m_primary.def(
        "matmul",
        [](const Tensor& input_tensor_a,
           const Tensor& input_tensor_b,
           const MatmulMultiCoreReuseProgramConfig& program_config,
           const MemoryConfig& out_mem_config,
           std::optional<DataType> output_dtype,
           const MathFidelity math_fidelity,
           const bool fp32_dest_acc_en,
           const bool math_approx_mode,
           const bool packer_l1_acc) {
            return matmul(input_tensor_a, input_tensor_b, program_config, out_mem_config, output_dtype, math_fidelity, fp32_dest_acc_en, math_approx_mode, packer_l1_acc);
        },
        py::arg("input_tensor_a").noconvert(),
        py::arg("input_tensor_b").noconvert(),
        py::kw_only(),
        py::arg("program_config").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("output_dtype").noconvert() = std::nullopt,
        py::arg("math_fidelity").noconvert() = MathFidelity::LoFi,
        py::arg("fp32_dest_acc_en").noconvert() = false,
        py::arg("math_approx_mode").noconvert() = true,
        py::arg("packer_l1_acc").noconvert() = false,
        R"doc(
            Perform a matrix multiplication ``input_tensor_a x input_tensor_b``.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_tensor_a",    "First tensor to multiply",                               "Tensor",                                     "Tensor of shape [B_a, C_a, M, K]",                               "Yes"
                "input_tensor_b",    "Second tensor to multiply",                              "Tensor",                                     "Tensor of shape [B_b, C_b, K, N]",                               "Yes"
                "program_config",    "",                                                       "MatmulMultiCoreReuseProgramConfig",          "",                                                               "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig",                               "Default is interleaved in DRAM",                                 "No"
                "output_dtype",      "Output Data Type",                                       "DataType",                                   "By default it will be set to the data type of `input_tensor_a`", "No"
        )doc");

    m_primary.def(
        "matmul",
        [](const Tensor& input_tensor_a,
           const Tensor& input_tensor_b,
           std::optional<const Tensor> bias,
           const MatmulDefaultProgramConfig& program_config,
           const MemoryConfig& out_mem_config,
           std::optional<DataType> output_dtype,
           const MathFidelity math_fidelity,
           const bool fp32_dest_acc_en,
           const bool math_approx_mode,
           const bool packer_l1_acc) {
            return matmul(
                input_tensor_a, input_tensor_b, bias, program_config, out_mem_config, output_dtype, math_fidelity, fp32_dest_acc_en, math_approx_mode, packer_l1_acc);
        },
        py::arg("input_tensor_a").noconvert(),
        py::arg("input_tensor_b").noconvert(),
        py::kw_only(),
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("program_config").noconvert() = MatmulDefaultProgramConfig(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("output_dtype").noconvert() = std::nullopt,
        py::arg("math_fidelity").noconvert() = MathFidelity::LoFi,
        py::arg("fp32_dest_acc_en").noconvert() = false,
        py::arg("math_approx_mode").noconvert() = true,
        py::arg("packer_l1_acc").noconvert() = false,
        R"doc(
            Perform a matrix multiplication ``input_tensor_a x input_tensor_b``.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_tensor_a",    "First tensor to multiply",                               "Tensor",                                     "Tensor of shape [B_a, C_a, M, K]",                               "Yes"
                "input_tensor_b",    "Second tensor to multiply",                              "Tensor",                                     "Tensor of shape [B_b, C_b, K, N]",                               "Yes"
                "bias",              "Bias to add",                                            "Tensor",                                     "Tensor of shape [1, 1, 1, N]",                                   "Yes"
                "program_config",    "",                                                       "MatmulDefaultProgramConfig", "",                                                               "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig",                               "Default is interleaved in DRAM",                                 "No"
                "output_dtype",      "Output Data Type",                                       "DataType",                                   "By default it will be set to the data type of `input_tensor_a`", "No"
        )doc");

    m_primary.def(
        "matmul",
        [](const Tensor& input_tensor_a,
           const Tensor& input_tensor_b,
           std::optional<const Tensor> bias,
           const MatmulMultiCoreReuseMultiCastProgramConfig& program_config,
           const MemoryConfig& out_mem_config,
           std::optional<DataType> output_dtype,
           const MathFidelity math_fidelity,
           const bool fp32_dest_acc_en,
           const bool math_approx_mode,
           const bool packer_l1_acc) {
            return matmul(
                input_tensor_a, input_tensor_b, bias, program_config, out_mem_config, output_dtype, math_fidelity, fp32_dest_acc_en, math_approx_mode, packer_l1_acc);
        },
        py::arg("input_tensor_a").noconvert(),
        py::arg("input_tensor_b").noconvert(),
        py::kw_only(),
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("program_config").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("output_dtype").noconvert() = std::nullopt,
        py::arg("math_fidelity").noconvert() = MathFidelity::LoFi,
        py::arg("fp32_dest_acc_en").noconvert() = false,
        py::arg("math_approx_mode").noconvert() = true,
        py::arg("packer_l1_acc").noconvert() = false,
        R"doc(
            Perform a matrix multiplication ``input_tensor_a x input_tensor_b``.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_tensor_a",    "First tensor to multiply",                               "Tensor",                                     "Tensor of shape [B_a, C_a, M, K]",                               "Yes"
                "input_tensor_b",    "Second tensor to multiply",                              "Tensor",                                     "Tensor of shape [B_b, C_b, K, N]",                               "Yes"
                "bias",              "Bias to add",                                            "Tensor",                                     "Tensor of shape [1, 1, 1, N]",                                   "Yes"
                "program_config",    "",                                                       "MatmulMultiCoreReuseMultiCastProgramConfig", "",                                                               "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig",                               "Default is interleaved in DRAM",                                 "No"
                "output_dtype",      "Output Data Type",                                       "DataType",                                   "By default it will be set to the data type of `input_tensor_a`", "No"
        )doc");

    m_primary.def(
        "matmul",
        [](const Tensor& input_tensor_a,
           const Tensor& input_tensor_b,
           std::optional<const Tensor> bias,
           const MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
           const MemoryConfig& out_mem_config,
           std::optional<DataType> output_dtype,
           const MathFidelity math_fidelity,
           const bool fp32_dest_acc_en,
           const bool math_approx_mode,
           const bool packer_l1_acc) {
            return matmul(
                input_tensor_a, input_tensor_b, bias, program_config, out_mem_config, output_dtype, math_fidelity, fp32_dest_acc_en, math_approx_mode, packer_l1_acc);
        },
        py::arg("input_tensor_a").noconvert(),
        py::arg("input_tensor_b").noconvert(),
        py::kw_only(),
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("program_config").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("output_dtype").noconvert() = std::nullopt,
        py::arg("math_fidelity").noconvert() = MathFidelity::LoFi,
        py::arg("fp32_dest_acc_en").noconvert() = false,
        py::arg("math_approx_mode").noconvert() = true,
        py::arg("packer_l1_acc").noconvert() = false,
        R"doc(
            Perform a matrix multiplication ``input_tensor_a x input_tensor_b``.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_tensor_a",    "First tensor to multiply",                               "Tensor",                                     "Tensor of shape [B_a, C_a, M, K]",                               "Yes"
                "input_tensor_b",    "Second tensor to multiply",                              "Tensor",                                     "Tensor of shape [B_b, C_b, K, N]",                               "Yes"
                "bias",              "Bias to add",                                            "Tensor",                                     "Tensor of shape [1, 1, 1, N]",                                   "Yes"
                "program_config",    "",                                                       "MatmulMultiCoreReuseMultiCast1DProgramConfig", "",                                                             "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig",                               "Default is interleaved in DRAM",                                 "No"
                "output_dtype",      "Output Data Type",                                       "DataType",                                   "By default it will be set to the data type of `input_tensor_a`", "No"
        )doc");

    m_primary.def(
        "matmul_1d",
        [](const Tensor& input_tensor_a,
           const Tensor& input_tensor_b,
           std::optional<const Tensor> bias,
           const std::optional<MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_config,
           const MemoryConfig& out_mem_config,
           std::optional<DataType> output_dtype,
           const MathFidelity math_fidelity,
           const bool fp32_dest_acc_en,
           const bool math_approx_mode,
           const bool packer_l1_acc) {
            return matmul_1d(
                input_tensor_a, input_tensor_b, bias, program_config, out_mem_config, output_dtype, math_fidelity, fp32_dest_acc_en, math_approx_mode, packer_l1_acc);
        },
        py::arg("input_tensor_a").noconvert(),
        py::arg("input_tensor_b").noconvert(),
        py::kw_only(),
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("program_config").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("output_dtype").noconvert() = std::nullopt,
        py::arg("math_fidelity").noconvert() = MathFidelity::LoFi,
        py::arg("fp32_dest_acc_en").noconvert() = false,
        py::arg("math_approx_mode").noconvert() = true,
        py::arg("packer_l1_acc").noconvert() = false,
        R"doc(
            Perform a matrix multiplication ``input_tensor_a x input_tensor_b``.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_tensor_a",    "First tensor to multiply",                               "Tensor",                                     "Tensor of shape [B_a, C_a, M, K]",                               "Yes"
                "input_tensor_b",    "Second tensor to multiply",                              "Tensor",                                     "Tensor of shape [B_b, C_b, K, N]",                               "Yes"
                "bias",              "Bias to add",                                            "Tensor",                                     "Tensor of shape [1, 1, 1, N]",                                   "Yes"
                "program_config",    "",                                                       "MatmulMultiCoreReuseMultiCast1DProgramConfig", "Config will be automatically determined if not passed",        "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig",                               "Default is interleaved in DRAM",                                 "No"
                "output_dtype",      "Output Data Type",                                       "DataType",                                   "By default it will be set to the data type of `input_tensor_a`", "No"
        )doc");

    py::class_<LayerNormDefaultProgramConfig>(m_primary, "LayerNormDefaultProgramConfig")
        .def(py::init<>());

    py::class_<LayerNormInterleavedMultiCoreProgramConfig>(m_primary, "LayerNormInterleavedMultiCoreProgramConfig")
        .def(
            py::init<MathFidelity, DataType, DataType>(),
            py::kw_only(),
            py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4,
            py::arg("im_data_format").noconvert(),
            py::arg("out_data_format").noconvert()
        );

    py::class_<LayerNormShardedMultiCoreProgramConfig>(m_primary, "LayerNormShardedMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, MathFidelity, DataType, DataType, bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("subblock_w").noconvert(),
            py::arg("block_h").noconvert(),
            py::arg("block_w").noconvert(),
            py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4,
            py::arg("im_data_format").noconvert(),
            py::arg("out_data_format").noconvert(),
            py::arg("inplace").noconvert()
        );

    m_primary.def(
        "layernorm",
        &layernorm,
        py::arg("input").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = LayerNormDefaultProgramConfig{},
        R"doc(
            Performs a layernorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
        )doc");

    m_primary.def(
        "add_layernorm",
        &add_layernorm,
        py::arg("a").noconvert(),
        py::arg("b").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = LayerNormDefaultProgramConfig{},
        R"doc(
            Performs a layernorm(a+b)*gamma + beta operation.
        )doc");

    // moreh_matmul
    m_primary.def(
        "moreh_matmul",
        &moreh_matmul,
        py::arg("input_a").noconvert(),
        py::arg("input_b").noconvert(),
        py::kw_only(),
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("transpose_input_a").noconvert() = false,
        py::arg("transpose_input_b").noconvert() = false,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs a moreh_matmul operation.");

    // moreh_matmul_backward
    m_primary.def(
        "moreh_matmul_backward",
        &moreh_matmul_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input_a").noconvert(),
        py::arg("input_b").noconvert(),
        py::arg("input_a_grad").noconvert() = std::nullopt,
        py::arg("input_b_grad").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        "Performs a moreh_matmul_backward operation.
    )doc");

    m_primary.def(
        "moreh_layernorm",
        &moreh_layernorm,
        py::arg("input").noconvert(),
        py::arg("normalized_dims").noconvert(),
        py::arg("eps").noconvert() = 1e-5f,
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::kw_only(),
        py::arg("mean").noconvert() = std::nullopt,
        py::arg("rstd").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
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
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs a moreh_layernorm_backward operation.");
    // softmax
    m_primary.def(
        "softmax_in_place",
        &softmax_in_place,
        py::arg("input_tensor").noconvert(),
        py::arg("program_config").noconvert() = transformers::SoftmaxDefaultProgramConfig{},
        "Performs a softmax operation on the last tensor dimension. Returns a reference to the input tensor modified "
        "in place.");

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
        py::arg("input_tensors").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("strategy").noconvert() = MorehSoftmaxOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs a softmax operation. Returns a output tensor.");
    m_primary.def(
        "moreh_softmax_backward",
        &moreh_softmax_backward,
        py::arg("output_tensor").noconvert(),
        py::arg("output_grad_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("strategy").noconvert() = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs a softmax backward operation. Returns a input grad tensor.");
    m_primary.def(
        "moreh_softmin",
        &moreh_softmin,
        py::arg("input_tensors").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("strategy").noconvert() = MorehSoftmaxOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs a softmax operation. Returns a output tensor.");
    m_primary.def(
        "moreh_softmin_backward",
        &moreh_softmin_backward,
        py::arg("output_tensor").noconvert(),
        py::arg("output_grad_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("strategy").noconvert() = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        "Performs a softmin backward operation. Returns a input grad tensor.");
}

}  // namespace
   // primary
}  // namespace
   // operations
}  // namespace
   // tt
