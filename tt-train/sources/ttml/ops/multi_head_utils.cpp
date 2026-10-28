// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_utils.hpp"

#include <stdexcept>
#include <string_view>
#include <tt-metalium/constants.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::ops {

namespace {

// Checks that `width` splits into `num_heads` heads of a tile-aligned head_dim, and returns head_dim.
// `op_name` and `heads_name` (what `num_heads` counts) only label the error messages.
uint32_t validate_head_dim(uint32_t width, uint32_t num_heads, std::string_view op_name, std::string_view heads_name) {
    if (num_heads == 0U) {
        throw std::invalid_argument(fmt::format("{} expects {} to be positive, got 0.", op_name, heads_name));
    }
    if (width % num_heads != 0U) {
        throw std::invalid_argument(fmt::format(
            "{} expects the input width ({}) to be divisible by {} ({}).", op_name, width, heads_name, num_heads));
    }
    // nlp_create_qkv_heads derives its tile count as head_dim / TILE_WIDTH, so a sub-tile head_dim
    // silently writes nothing and returns a tile-padded tensor of garbage.
    const uint32_t head_dim = width / num_heads;
    if (head_dim % tt::constants::TILE_WIDTH != 0U) {
        throw std::invalid_argument(fmt::format(
            "{} expects head_dim (input width {} / {} {} = {}) to be a multiple of {}.",
            op_name,
            width,
            heads_name,
            num_heads,
            head_dim,
            tt::constants::TILE_WIDTH));
    }
    return head_dim;
}

// Splits `qs` (and `kvs`, when given) into heads with nlp_create_qkv_heads:
// q: (B, num_heads, S, head_dim); k, v: (B, num_kv_heads, S, head_dim).
// Without `kvs`, `qs` holds [Q | K | V]; with it, `qs` holds Q and `kvs` holds [K | V].
std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> create_heads(
    const autograd::TensorPtr& qs,
    const autograd::TensorPtr& kvs,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    std::string_view op_name) {
    const uint32_t qs_width = qs->get_value().logical_shape()[-1];
    if (kvs) {
        const uint32_t head_dim = validate_head_dim(qs_width, num_heads, op_name, "num_heads");
        const uint32_t kvs_width = kvs->get_value().logical_shape()[-1];
        if (kvs_width != 2U * num_kv_heads * head_dim) {
            throw std::invalid_argument(fmt::format(
                "{} expects the KV width ({}) to be 2 * {} KV heads * head_dim {}.",
                op_name,
                kvs_width,
                num_kv_heads,
                head_dim));
        }
    } else {
        if (num_heads == 0U) {
            throw std::invalid_argument(fmt::format("{} expects num_heads to be positive, got 0.", op_name));
        }
        validate_head_dim(qs_width, num_heads + 2U * num_kv_heads, op_name, "num_heads + 2 * num_groups");
    }

    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        qs->get_value(),
        kvs ? std::optional<ttnn::Tensor>(kvs->get_value()) : std::nullopt,
        /*num_q_heads=*/num_heads,
        /*num_kv_heads=*/num_kv_heads,
        /*transpose_k_heads=*/false,
        /*kv_tied=*/false,
        /*memory_config=*/std::nullopt,
        /*optional_output_tensors=*/std::nullopt);

    auto out_q = autograd::create_tensor(q);
    auto out_k = autograd::create_tensor(k);
    auto out_v = autograd::create_tensor(v);

    autograd::GradFunction grad_q = [out_q, out_k, out_v, qs, kvs]() {
        // (B, heads, S, head_dim) -> (B, 1, S, heads * head_dim)
        auto grad_q = ttnn::experimental::nlp_concat_heads(out_q->get_grad());
        auto grad_k = ttnn::experimental::nlp_concat_heads(out_k->get_grad());
        auto grad_v = ttnn::experimental::nlp_concat_heads(out_v->get_grad());
        if (kvs) {
            qs->add_grad(grad_q);
            kvs->add_grad(ttnn::concat(std::vector<ttnn::Tensor>({grad_k, grad_v}), /* dim */ 3));
        } else {
            qs->add_grad(ttnn::concat(std::vector<ttnn::Tensor>({grad_q, grad_k, grad_v}), /* dim */ 3));
        }
    };

    // grad_q function depends on gradients of q, k and v
    auto q_node = autograd::add_backward_node(std::move(grad_q), out_q, qs, kvs);
    out_q->set_node(q_node);
    // this needs to be added to make sure that gradients for k and v are computed before we run backward for q
    // Only add sync nodes if out_q has a node (i.e., gradients are needed)
    if (q_node.has_value()) {
        out_k->set_node(autograd::add_backward_node_always([]() {}, out_k, qs, kvs, out_q));
        out_v->set_node(autograd::add_backward_node_always([]() {}, out_v, qs, kvs, out_q));
    }
    return {out_q, out_k, out_v};
}

}  // namespace

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> heads_creation(
    const autograd::TensorPtr& qkv, uint32_t num_heads, std::optional<uint32_t> num_groups) {
    // qkv: (B, 1, S, (num_heads + 2 * num_groups) * head_dim), laid out [Q | K | V].
    // num_groups defaults to num_heads (standard MHA); num_groups < num_heads selects GQA.
    return create_heads(qkv, nullptr, num_heads, num_groups.value_or(num_heads), "heads_creation");
}

autograd::TensorPtr split_heads(const autograd::TensorPtr& x, uint32_t num_heads) {
    auto x_shape = x->get_value().logical_shape();
    if (x_shape[1] != 1U) {
        throw std::invalid_argument(
            fmt::format("split_heads expects an input of shape (B, 1, S, E), but dim 1 is {}.", x_shape[1]));
    }
    validate_head_dim(x_shape[3], num_heads, "split_heads", "num_heads");
    // nlp_create_qkv_heads picks its program factory from the input alone, and the sharded factory
    // divides by num_kv_heads, which is 0 here. Reject rather than take a SIGFPE inside the kernel.
    if (x->get_value().is_sharded()) {
        throw std::invalid_argument("split_heads does not support sharded inputs; provide an interleaved tensor.");
    }

    // (B, 1, S, E) -> (B, num_heads, S, E / num_heads)
    // num_kv_heads = 0 splits the whole input into query heads, leaving the k and v outputs empty.
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        x->get_value(),
        std::nullopt,
        /*num_q_heads=*/num_heads,
        /*num_kv_heads=*/0,
        /*transpose_k_heads=*/false,
        /*kv_tied=*/false,
        /*memory_config=*/std::nullopt,
        /*optional_output_tensors=*/std::nullopt);

    auto out = autograd::create_tensor(q);

    autograd::GradFunction grad = [out, x]() {
        // (B, num_heads, S, E / num_heads) -> (B, 1, S, E)
        x->add_grad(ttnn::experimental::nlp_concat_heads(out->get_grad()));
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, x));

    return out;
}

autograd::TensorPtr heads_fusion(const autograd::TensorPtr& x) {
    auto x_shape = x->get_value().logical_shape();

    uint32_t batch_size = x_shape[0];
    uint32_t num_heads = x_shape[1];
    uint32_t sequence_length = x_shape[2];
    uint32_t embedding_dim = x_shape[3];

    // (B, H, S, E/H) -> (B, 1, S, E)
    auto fused_heads = ttnn::experimental::nlp_concat_heads(x->get_value());
    auto out = autograd::create_tensor(fused_heads);

    autograd::GradFunction grad = [out, x, num_heads, batch_size, sequence_length, embedding_dim]() {
        auto grad_output = out->get_grad();
        // (B, 1, S, E) -> (B, 1, E, S)
        auto grad_result = ttnn::transpose(grad_output, -2, -1);
        // (B, 1, E, S) -> (B, H, E/H, S)
        grad_result = ttnn::reshape(grad_result, ttnn::Shape({batch_size, num_heads, embedding_dim, sequence_length}));
        // (B, H, E/H, S) -> (B, H, S, E/H)
        grad_result = ttnn::transpose(grad_result, -2, -1);
        x->add_grad(grad_result);
    };

    out->set_node(ttml::autograd::add_backward_node(std::move(grad), out, x));

    return out;
}

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> grouped_heads_creation(
    const autograd::TensorPtr& qs, const autograd::TensorPtr& kvs, uint32_t num_heads, uint32_t num_groups) {
    // qs: (B, 1, S, num_heads * head_dim); kvs: (B, 1, S, 2 * num_groups * head_dim), laid out [K | V].
    return create_heads(qs, kvs, num_heads, num_groups, "grouped_heads_creation");
}

}  // namespace ttml::ops
