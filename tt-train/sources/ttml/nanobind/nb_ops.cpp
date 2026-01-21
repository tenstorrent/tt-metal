// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <span>

#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
#include "nb_export_enum.hpp"
#include "nb_fwd.hpp"
#include "ops/binary_ops.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/dropout_op.hpp"
#include "ops/embedding_op.hpp"
#include "ops/layernorm_op.hpp"
#include "ops/linear_op.hpp"
#include "ops/losses.hpp"
#include "ops/matmul_op.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/reshape_op.hpp"
#include "ops/rmsnorm_op.hpp"
#include "ops/rope_op.hpp"
#include "ops/sampling_op.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::nanobind::ops {
using namespace ttml::ops;

void py_module_types(nb::module_& m) {
    ttml::nanobind::util::export_enum<ReduceType>(m);

    m.def_submodule("binary");
    m.def_submodule("distributed");
    m.def_submodule("dropout");
    m.def_submodule("embedding");
    m.def_submodule("layernorm");
    m.def_submodule("linear");
    m.def_submodule("loss");

    {
        auto py_rope = m.def_submodule("rope");
        nb::class_<ttml::ops::RopeScalingParams>(py_rope, "RopeScalingParams");
        nb::class_<ttml::ops::RotaryEmbeddingParams>(py_rope, "RotaryEmbeddingParams");
    }

    m.def_submodule("matmul");
    m.def_submodule("multi_head_utils");
    m.def_submodule("attention");
    m.def_submodule("reshape");
    m.def_submodule("rmsnorm");
    m.def_submodule("sample");
    m.def_submodule("unary");
}

void py_module(nb::module_& m) {
    {
        auto py_binary = static_cast<nb::module_>(m.attr("binary"));
        py_binary.def(
            "add",
            static_cast<autograd::TensorPtr (*)(const autograd::TensorPtr&, const autograd::AutocastTensor&)>(
                &ops::operator+),
            nb::arg("a"),
            nb::arg("b"));
        py_binary.def(
            "add",
            static_cast<autograd::TensorPtr (*)(const autograd::TensorPtr&, const autograd::TensorPtr&)>(
                &ops::operator+),
            nb::arg("a"),
            nb::arg("b"));
        py_binary.def(
            "mul",
            static_cast<autograd::TensorPtr (*)(const autograd::TensorPtr&, const autograd::TensorPtr&)>(
                &ops::operator*),
            nb::arg("a"),
            nb::arg("b"));
        py_binary.def(
            "mul",
            static_cast<autograd::TensorPtr (*)(const autograd::TensorPtr&, float)>(ops::operator*),
            nb::arg("a"),
            nb::arg("b"));
        py_binary.def(
            "sub",
            static_cast<autograd::TensorPtr (*)(const autograd::TensorPtr&, const autograd::TensorPtr&)>(
                &ttml::ops::operator-),
            nb::arg("a"),
            nb::arg("b"));
        py_binary.def(
            "div",
            static_cast<autograd::TensorPtr (*)(const autograd::TensorPtr&, const autograd::TensorPtr&)>(
                &ttml::ops::operator/),
            nb::arg("a"),
            nb::arg("b"));
    }

    {
        auto py_distributed = static_cast<nb::module_>(m.attr("distributed"));
        py_distributed.def(
            "all_reduce",
            &ttml::ops::distributed::all_reduce,
            nb::arg("tensor"),
            nb::arg("noop_backward") = false);
        py_distributed.def(
            "reduce_scatter", &ttml::ops::distributed::reduce_scatter, nb::arg("tensor"), nb::arg("dim"));
        py_distributed.def("all_gather", &ttml::ops::distributed::all_gather, nb::arg("tensor"), nb::arg("dim"));
        py_distributed.def("broadcast", &ttml::ops::distributed::broadcast, nb::arg("tensor"));
    }

    {
        auto py_dropout = static_cast<nb::module_>(m.attr("dropout"));
        py_dropout.def(
            "dropout",
            &ttml::ops::dropout,
            nb::arg("tensor"),
            nb::arg("probability"),
            nb::arg("use_per_device_seed") = false);
    }

    {
        auto py_embedding = static_cast<nb::module_>(m.attr("embedding"));
        py_embedding.def("embedding", &ttml::ops::embedding_op, nb::arg("tensor"), nb::arg("weight"));
    }

    {
        auto py_layernorm = static_cast<nb::module_>(m.attr("layernorm"));
        py_layernorm.def("layernorm", &ttml::ops::layernorm, nb::arg("tensor"), nb::arg("gamma"), nb::arg("beta"));
        py_layernorm.def(
            "composite_layernorm",
            &ttml::ops::composite_layernorm,
            nb::arg("tensor"),
            nb::arg("gamma"),
            nb::arg("beta"));
    }

    {
        auto py_linear = static_cast<nb::module_>(m.attr("linear"));
        // Overload with optional bias (None support)
        py_linear.def(
            "linear",
            [](const autograd::TensorPtr& tensor,
               const autograd::TensorPtr& weight,
               std::optional<const autograd::TensorPtr> bias) -> autograd::TensorPtr {
                return ttml::ops::linear_op(tensor, weight, bias.value_or(nullptr));
            },
            nb::arg("tensor"),
            nb::arg("weight"),
            nb::arg("bias") = nb::none());
        py_linear.def(
            "ttnn_linear_backward",
            &ttml::ops::ttnn_linear_backward,
            nb::arg("tensor"),
            nb::arg("weight"),
            nb::arg("bias"),
            nb::arg("out"));
        py_linear.def(
            "moreh_linear_backward",
            &ttml::ops::moreh_linear_backward,
            nb::arg("tensor"),
            nb::arg("weight"),
            nb::arg("bias"),
            nb::arg("out"));
    }

    {
        auto py_loss = static_cast<nb::module_>(m.attr("loss"));
        py_loss.def(
            "cross_entropy_loss",
            &ttml::ops::cross_entropy_loss,
            nb::arg("prediction"),
            nb::arg("target"),
            nb::arg("reduce") = ReduceType::MEAN);
        py_loss.def(
            "mse_loss",
            &ttml::ops::mse_loss,
            nb::arg("prediction"),
            nb::arg("target"),
            nb::arg("reduce") = ReduceType::MEAN);
        py_loss.def(
            "nll_loss",
            &ttml::ops::nll_loss,
            nb::arg("prediction"),
            nb::arg("target"),
            nb::arg("reduce") = ReduceType::MEAN);
    }

    {
        auto py_matmul = static_cast<nb::module_>(m.attr("matmul"));
        py_matmul.def(
            "matmul_op",
            &ttml::ops::matmul_op,
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("transpose_a") = false,
            nb::arg("transpose_b") = false);
    }

    {
        auto py_multi_head_utils = static_cast<nb::module_>(m.attr("multi_head_utils"));
        py_multi_head_utils.def("heads_creation", &ttml::ops::heads_creation, nb::arg("qkv"), nb::arg("num_heads"));
        py_multi_head_utils.def("heads_fusion", &ttml::ops::heads_fusion, nb::arg("x"));
        py_multi_head_utils.def(
            "grouped_heads_creation",
            &ttml::ops::grouped_heads_creation,
            nb::arg("qs"),
            nb::arg("kvs"),
            nb::arg("num_heads"),
            nb::arg("num_groups"));
    }

    {
        auto py_attention = static_cast<nb::module_>(m.attr("attention"));
        // Overload 1: mask as ttml.autograd.Tensor (or None)
        py_attention.def(
            "scaled_dot_product_attention",
            [](const autograd::TensorPtr& query,
               const autograd::TensorPtr& key,
               const autograd::TensorPtr& value,
               const std::optional<autograd::TensorPtr>& mask) -> autograd::TensorPtr {
                return ttml::ops::scaled_dot_product_attention(query, key, value, mask);
            },
            nb::arg("query"),
            nb::arg("key"),
            nb::arg("value"),
            nb::arg("mask") = std::nullopt);
        // Overload 2: mask as ttnn.Tensor (or None) - wrap it in autograd::Tensor
        // ttnn.Tensor wraps tt::tt_metal::Tensor, so we accept that type
        py_attention.def(
            "scaled_dot_product_attention",
            [](const autograd::TensorPtr& query,
               const autograd::TensorPtr& key,
               const autograd::TensorPtr& value,
               const std::optional<tt::tt_metal::Tensor>& mask) -> autograd::TensorPtr {
                std::optional<autograd::TensorPtr> mask_ptr = std::nullopt;
                if (mask.has_value()) {
                    mask_ptr = autograd::create_tensor(mask.value(), false);
                }
                return ttml::ops::scaled_dot_product_attention(query, key, value, mask_ptr);
            },
            nb::arg("query"),
            nb::arg("key"),
            nb::arg("value"),
            nb::arg("mask") = std::nullopt);
    }

    {
        auto py_rope = static_cast<nb::module_>(m.attr("rope"));
        py_rope.def("rope", &ttml::ops::rope, nb::arg("input"), nb::arg("rope_params"), nb::arg("token_position") = 0);
        py_rope.def(
            "gen_freqs",
            &ttml::ops::gen_freqs,
            nb::arg("head_dim"),
            nb::arg("sequence_length"),
            nb::arg("theta"),
            nb::arg("rope_scaling_params"));
        py_rope.def(
            "build_rope_params",
            &ttml::ops::build_rope_params,
            nb::arg("sequence_length"),
            nb::arg("head_dim"),
            nb::arg("theta") = 10000.0F,
            nb::arg("rope_scaling_params") = RopeScalingParams{});
        py_rope.def(
            "validate_rope_input_and_params",
            &ttml::ops::validate_rope_input_and_params,
            nb::arg("input"),
            nb::arg("rope_params"));
        {
            auto py_rope_scaling_params =
                static_cast<nb::class_<ttml::ops::RopeScalingParams>>(py_rope.attr("RopeScalingParams"));
            py_rope_scaling_params.def(nb::init<>());
            py_rope_scaling_params.def_rw(
                "original_context_length", &ttml::ops::RopeScalingParams::original_context_length);
            py_rope_scaling_params.def_rw("scaling_factor", &ttml::ops::RopeScalingParams::scaling_factor);
            py_rope_scaling_params.def_rw("high_freq_factor", &ttml::ops::RopeScalingParams::high_freq_factor);
            py_rope_scaling_params.def_rw("low_freq_factor", &ttml::ops::RopeScalingParams::low_freq_factor);
        }
        {
            auto py_rotary_embedding_params =
                static_cast<nb::class_<ttml::ops::RotaryEmbeddingParams>>(py_rope.attr("RotaryEmbeddingParams"));
            py_rotary_embedding_params.def(nb::init<>());
            py_rotary_embedding_params.def_rw("cos_cache", &ttml::ops::RotaryEmbeddingParams::cos_cache);
            py_rotary_embedding_params.def_rw("sin_cache", &ttml::ops::RotaryEmbeddingParams::sin_cache);
            py_rotary_embedding_params.def_rw("neg_cos_cache", &ttml::ops::RotaryEmbeddingParams::neg_cos_cache);
            py_rotary_embedding_params.def_rw("neg_sin_cache", &ttml::ops::RotaryEmbeddingParams::neg_sin_cache);
            py_rotary_embedding_params.def_rw("trans_mat", &ttml::ops::RotaryEmbeddingParams::trans_mat);
            py_rotary_embedding_params.def_rw("sequence_length", &ttml::ops::RotaryEmbeddingParams::sequence_length);
            py_rotary_embedding_params.def_rw("head_dim", &ttml::ops::RotaryEmbeddingParams::head_dim);
            py_rotary_embedding_params.def_rw("theta", &ttml::ops::RotaryEmbeddingParams::theta);
            py_rotary_embedding_params.def_rw(
                "rope_scaling_params", &ttml::ops::RotaryEmbeddingParams::rope_scaling_params);
        }
    }

    {
        auto py_reshape = static_cast<nb::module_>(m.attr("reshape"));
        // Wrapper to convert Python list to std::span
        py_reshape.def(
            "reshape",
            [](const autograd::TensorPtr& tensor, const std::vector<int32_t>& shape) -> autograd::TensorPtr {
                // Convert int32_t vector to uint32_t span
                std::vector<uint32_t> shape_uint32(shape.begin(), shape.end());
                return ttml::ops::reshape(tensor, std::span<uint32_t>(shape_uint32));
            },
            nb::arg("tensor"),
            nb::arg("shape"));
    }

    {
        auto py_rmsnorm = static_cast<nb::module_>(m.attr("rmsnorm"));
        py_rmsnorm.def("rmsnorm", &ttml::ops::rmsnorm, nb::arg("tensor"), nb::arg("gamma"), nb::arg("epsilon"));
        py_rmsnorm.def(
            "rmsnorm_composite",
            &ttml::ops::rmsnorm_composite,
            nb::arg("tensor"),
            nb::arg("gamma"),
            nb::arg("epsilon"));
    }

    {
        auto py_sample = static_cast<nb::module_>(m.attr("sample"));
        py_sample.def(
            "sample_op",
            &ttml::ops::sample_op,
            nb::arg("logits"),
            nb::arg("temperature"),
            nb::arg("seed"),
            nb::arg("logits_padding_mask") = nb::none());
    }

    {
        auto py_unary = static_cast<nb::module_>(m.attr("unary"));
        py_unary.def("relu", &ttml::ops::relu, nb::arg("tensor"));
        py_unary.def("gelu", &ttml::ops::gelu, nb::arg("tensor"));
        py_unary.def("silu", &ttml::ops::silu, nb::arg("tensor"), nb::arg("use_composite_bw") = false);
        py_unary.def("mean", &ttml::ops::mean, nb::arg("tensor"));
        // py_unary.def("sum", &ttml::ops::sum,
        //              nb::arg("tensor"));
        py_unary.def("broadcast_batch", &ttml::ops::broadcast_batch, nb::arg("tensor"), nb::arg("new_batch_dim"));
        py_unary.def("log_softmax", &ttml::ops::log_softmax, nb::arg("tensor"), nb::arg("dim"));
        py_unary.def("log_softmax_moreh", &ttml::ops::log_softmax_moreh, nb::arg("tensor"), nb::arg("dim"));
    }
}

}  // namespace ttml::nanobind::ops
