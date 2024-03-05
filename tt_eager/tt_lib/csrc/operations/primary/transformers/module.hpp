// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/transformer_tms/transformer_tms.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace tt {
namespace operations {
namespace primary {
namespace transformers {


void py_module(py::module& m_transformers) {
    m_transformers.def("split_query_key_value_and_split_heads", &split_query_key_value_and_split_heads,
        py::arg().noconvert(),
        py::arg("compute_with_storage_grid_size"),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("num_heads").noconvert() = 16,
        R"doc(
        Splits [9, 1, 384, 3072] fused qkv matrix into 3 heads with shapes [9, 16, 384, 64], [9, 16, 64, 384], and [9, 16, 384, 64].
    )doc");

    m_transformers.def("concatenate_heads", &concatenate_heads,
        py::arg().noconvert(), py::arg("compute_with_storage_grid_size").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Reshuffles [9, 16, 384, 64] tensor into tensor with shape [9, 1, 384, 1024].
    )doc");

    m_transformers.def("attn_matmul", &attn_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("compute_with_storage_grid_size").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        Performs a special pre-softmax matmul with [q_len, q_heads, batch, head_dim] and [batch, kv_heads, head_dim, kv_len]. q_len and kv_heads must be 1 and an intermediate value of [q_heads, batch, batch, kv_len] is produced (only on device cores). Batch dim from Z and Y is combined by taking the 1st, 2nd, ..., and 32nd row of Y from the batches in Z. Final output tensor is [1, q_heads, batch, kv_len]. In PyTorch, this is equivalent to: torch.matmul(A.transpose(0, 2), B).transpose(0, 2). Similar concept for post-softmax matmul.
    )doc");
    m_transformers.def("attn_matmul_from_cache", &attn_matmul_from_cache,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("num_tokens").noconvert(), py::arg("transpose_hw").noconvert(), py::arg("compute_with_storage_grid_size").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        Performs the same matmul as attn_matmul, but fuses additional functionality for reading in in1. For in1, read num_tokens (rounded up to 32) from full cache along in1.get_legacy_shape()[2] (num_tokens must be > 0 and <= max_cache_len). For example, 64 tokens will be read for 32 < token_idx <= 64. Additional option to apply transpose_hw to in1 for pre-attention matmul with transpose_hw=true. For post-attention matmul, transpose_hw should be false.
    )doc");
    m_transformers.def("group_attn_matmul", &group_attn_matmul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("compute_with_storage_grid_size").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        Performs a special pre-softmax matmul with [q_len, q_heads, batch, head_dim] and [batch, kv_heads, head_dim, kv_len]. q_len and q_heads must be divisible by kv_heads. If kv_heads is sharded, then batch must be 32; otherwise, batch can any multiple of 32. An intermediate value of [q_heads, batch, batch, kv_len] is produced (only on device cores). Batch dim from Z and Y is combined by taking the 1st, 2nd, ..., and 32nd row of Y from the batches in Z. Final output tensor is [1, q_heads, batch, kv_len]. In PyTorch, this is equivalent to:
            B = torch.repeat_interleave(B, q_heads // kv_heads, dim=1)
            torch.matmul(A.transpose(0, 2), B).transpose(0, 2). Similar concept for post-softmax matmul.
    )doc");

    py::class_<SoftmaxProgramConfig>(m_transformers, "SoftmaxProgramConfig").def(py::init<>());

    py::class_<SoftmaxDefaultProgramConfig>(m_transformers, "SoftmaxDefaultProgramConfig")
        .def(py::init<>());

    py::class_<SoftmaxInterleavedMultiCoreProgramConfig>(m_transformers, "SoftmaxInterleavedMultiCoreProgramConfig")
        .def(
            py::init<MathFidelity, DataType>(),
            py::kw_only(),
            py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4,
            py::arg("im_data_format").noconvert()
        );

    py::class_<SoftmaxShardedMultiCoreProgramConfig>(m_transformers, "SoftmaxShardedMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, MathFidelity, DataType>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("subblock_w").noconvert(),
            py::arg("block_h").noconvert(),
            py::arg("block_w").noconvert(),
            py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4,
            py::arg("im_data_format").noconvert())
        .def_readwrite("block_w", &SoftmaxShardedMultiCoreProgramConfig::block_w)
        .def("__repr__", [](const SoftmaxShardedMultiCoreProgramConfig& config) { return fmt::format("{}", config); });

    m_transformers.def(
        "scale_mask_softmax_in_place",
        &scale_mask_softmax_in_place,
        py::arg("input_tensor").noconvert(),
        py::arg("scale").noconvert() = std::nullopt,
        py::arg("mask").noconvert() = std::nullopt,
        py::arg("program_config").noconvert() = SoftmaxDefaultProgramConfig{},
        py::arg("is_causal_mask").noconvert() = false,
        "Performs a fused scale->attention_mask->softmax operation. Returns a reference to the input tensor modified in place."
        );
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
