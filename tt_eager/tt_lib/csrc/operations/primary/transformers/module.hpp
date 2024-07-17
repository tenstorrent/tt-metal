// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/transformer_tms/transformer_tms.hpp"
#include "tt_dnn/op_library/sdpa/sdpa_op.hpp"
#include "tt_dnn/op_library/paged_update_cache/paged_update_cache_op.hpp"

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
    m_transformers.def("ssm_eltwise_mul", &ssm_eltwise_mul,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4, R"doc(
        Performs a special eltwise multiply for SSM models. Given tensor A with shape [1, 1, 32, 32] and tensor B with shape [1, 1, 32, W] where W is some multiple of 32, perform the following PyTorch equivalent:
            A.repeat(1, 1, 1, W) * B.repeat_interleave(32, dim=-1)
    )doc");
    m_transformers.def("ssm_1d_sum_reduce", &ssm_1d_sum_reduce,
        py::arg().noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4, R"doc(
        Performs a custom reduction along dim 3 which is used in the SSM block of the Mamba architecture. Performs the following PyTorch equivalent (where latent_size = 32):
            x = torch.sum(x.reshape(1, 1, shape[2], shape[3] // latent_size, latent_size), dim=-1).reshape(1, 1, shape[2], shape[3] // latent_size)
    )doc");
    m_transformers.def(
        "ssm_prefix_scan",
        &ssm_prefix_scan,
        py::arg().noconvert(),
        py::arg().noconvert(),
        py::arg().noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("output_dtype").noconvert() = std::nullopt,
        py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4,
        R"doc(
        Performs a prefix scan to produce the SSM hidden states across an entire sequence. All input and output tensors are expected to be shape [1, 1, L, 2EN] where E = 2560 and N = 32. L can be any multiple of 32.)doc");


    py::class_<SDPADefaultProgramConfig>(m_transformers, "SDPADefaultProgramConfig")
        .def(py::init<>());

    py::class_<SDPAMultiCoreProgramConfig>(m_transformers, "SDPAMultiCoreProgramConfig")
        .def(py::init<CoreCoord, std::size_t, std::size_t>(), py::kw_only(), py::arg("compute_with_storage_grid_size"), py::arg("q_chunk_size").noconvert(), py::arg("k_chunk_size").noconvert())
        .def_readwrite("compute_with_storage_grid_size", &SDPAMultiCoreProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("q_chunk_size", &SDPAMultiCoreProgramConfig::q_chunk_size)
        .def_readwrite("k_chunk_size", &SDPAMultiCoreProgramConfig::k_chunk_size);

    m_transformers.def(
        "scaled_dot_product_attention",
        &scaled_dot_product_attention,
        py::arg("input_tensor_q").noconvert(),
        py::arg("input_tensor_k").noconvert(),
        py::arg("input_tensor_v").noconvert(),
        py::arg("causal_mask").noconvert(),
        py::arg("is_causal").noconvert() = true,
        py::arg("scale").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = SDPADefaultProgramConfig{},
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        py::arg("valid_seq_len").noconvert() = std::nullopt,
        "Causal scaled dot product attention. This API mimicks the PyTorch API of the same name."
        "The implementation is FlashAttention-2 and it currently only supports MQA with causal masking.\n"

        "Q:      [b x nqh x s x dh]"
        "K:      [b x 1   x s x dh]"
        "V:      [b x 1   x s x dh]"
        "mask:   [b x 1   x s x s ]"
        "output: [b x nqh x s x dh]"

        "Mask must be a causal mask with 0s in the lower triangle and -inf in the upper triangle."

        "Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the Q and K sequence lengths. The op parallelizes over `b`, `nqh`, and Q's `s` dimension."
        );

    m_transformers.def(
        "scaled_dot_product_attention_decode",
        &scaled_dot_product_attention_decode,
        py::arg("input_tensor_q").noconvert(),
        py::arg("input_tensor_k").noconvert(),
        py::arg("input_tensor_v").noconvert(),
        py::arg("mask").noconvert(),
        py::arg("scale").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = SDPADefaultProgramConfig{},
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        py::arg("valid_seq_len").noconvert() = std::nullopt,
        "A version of scaled dot product attention specifically for decode."
        "The implementation is Flash-Decode and it currently only supports MQA on decoding single token.\n"

        "Q:      [1 x b x pnh x dh]"
        "K:      [1 x b x   s x dh]"
        "V:      [1 x b x   s x dh]"
        "mask:   [1 x b x pnh x s ]"
        "output: [1 x b x pnh x dh]"

        "Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension."
        );

    m_transformers.def(
        "paged_update_cache",
        &paged_update_cache,
        py::arg("cache_tensor").noconvert(),
        py::arg("input_tensor").noconvert(),
        py::arg("update_idxs").noconvert(),
        py::arg("update_idxs_tensor").noconvert() = std::nullopt,
        py::arg("page_table").noconvert() = std::nullopt,
        py::arg("batch_offset") = 0,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        Paged update cache operation. This operation expects the following inputs: cache_tensor of shape [B, 1, kv_len, head_dim] and input_tensor of shape [1, B, 1[32], head_dim] where input_tensor is height sharded on B cores. update_idxs will specify for each batch element which token to update in the cache.
        )doc"
    );

    m_transformers.def(
        "paged_fill_cache",
        &paged_fill_cache,
        py::arg("cache_tensor").noconvert(),
        py::arg("input_tensor").noconvert(),
        py::arg("page_table").noconvert(),
        py::arg("batch_idx").noconvert(),
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        Paged fill cache operation. This operation expects the following inputs: cache_tensor of shape [B, 1, kv_len, head_dim] and input_tensor of shape [1, 1, seq_len, head_dim]. batch_idx specifies which index in the batch dimension to update with input_tensor.
        )doc"
    );

}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
