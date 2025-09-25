// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor.hpp"

#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

// TODO: implement stack based topological sort
void topological_sort(
    size_t node_id,
    const std::vector<std::vector<size_t>>& edges,
    std::unordered_set<size_t>& visited,
    std::vector<size_t>& sorted_nodes) {
    if (visited.contains(node_id)) {
        return;
    }
    visited.insert(node_id);
    for (const auto& next_node : edges[node_id]) {
        topological_sort(next_node, edges, visited, sorted_nodes);
    }
    sorted_nodes.push_back(node_id);
}

}  // namespace

namespace ttml::autograd {

Tensor::Tensor(const tt::tt_metal::Tensor& value, bool requires_grad) : m_value(value), m_requires_grad(requires_grad) {
}

void Tensor::add_grad(const tt::tt_metal::Tensor& grad) {
    if (!is_grad_initialized()) {
        auto value_shape = m_value.get_tensor().logical_shape();
        if (grad.logical_shape() != value_shape) {
            throw std::logic_error(fmt::format(
                "Shapes of gradients are not equal. Expected: {}, got: {}", value_shape, grad.logical_shape()));
        }

        m_grad = grad;
        return;
    }

    const auto& grad_shape = grad.logical_shape();
    const auto& m_grad_shape = m_grad.logical_shape();
    if (grad_shape != m_grad_shape) {
        throw std::logic_error(
            fmt::format("Shapes of gradients are not equal. Expected: {}, got: {}", m_grad_shape, grad_shape));
    }

    // It is important to not use inline addition here
    // m_grad might share memory with other tensors
    constexpr ttsl::Span<const ttnn::operations::unary::UnaryWithParam> none{};
    m_grad = ttnn::add(m_grad, grad, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
}

void Tensor::backward(bool retain_graph) {
    if (!m_node_id.has_value()) {
        return;
    }
    std::vector<size_t> sorted_nodes;
    std::unordered_set<std::size_t> visited_nodes;
    auto& graph = m_node_id->get_graph();
    topological_sort(m_node_id->get_id(), graph.get_edges(), visited_nodes, sorted_nodes);

    auto& graph_nodes = graph.get_graph_nodes();
    std::ranges::reverse(sorted_nodes);
    try_init_grad(/* init_ones */ true);
    for (const auto& node_id : sorted_nodes) {
        graph_nodes[node_id].grad_function();
        if (!retain_graph) {
            graph_nodes[node_id].grad_function = [] {
                throw std::runtime_error(
                    "[Tensor::backward] This backward function should not be called! Memory from the node is released! "
                    "Please consider tweaking the retain_graph parameter if you need to call backward twice on the "
                    "same graph nodes.");
            };
        }
    }
}

bool Tensor::is_grad_initialized() const {
    return core::is_tensor_initialized(get_grad());
}

void Tensor::try_init_grad(bool init_ones) {
    if (is_grad_initialized()) {
        return;
    }

    const auto& value = get_value();
    this->set_grad(init_ones ? ttml::core::ones_like(value) : ttml::core::zeros_like(value));
}
void Tensor::set_node(const std::optional<NodeId>& node) {
    if (m_node_id.has_value()) {
        throw std::runtime_error("Graph node is already set for this tensor!");
    }
    m_node_id = node;
}

void print_tensor_stats(const autograd::TensorPtr& tensor, const std::string& name) {
    core::print_tensor_stats(tensor->get_value(), name);
}

void Tensor::set_value(const tt::tt_metal::Tensor& value) {
    m_value.set_tensor(value);
}

void Tensor::set_grad(const tt::tt_metal::Tensor& grad) {
    if (core::is_tensor_initialized(grad)) {
        auto grad_shape = grad.logical_shape();
        auto value_shape = m_value.get_tensor().logical_shape();
        if (grad_shape != value_shape) {
            throw std::logic_error(fmt::format(
                "Shapes of gradients are not equal. Expected: {}, got: {}",
                m_value.get_tensor().logical_shape(),
                grad.logical_shape()));
        }
    }
    m_grad = grad;
}

void Tensor::clean_node() {
    m_node_id = std::nullopt;
}

void Tensor::set_requires_grad(bool requires_grad) {
    m_requires_grad = requires_grad;
}

const tt::tt_metal::Tensor& Tensor::get_value(PreferredPrecision preferred_precision) const {
    return m_value.get_tensor(preferred_precision);
}

const tt::tt_metal::Tensor& Tensor::get_grad() const {
    return m_grad;
}

tt::tt_metal::Tensor& Tensor::get_grad() {
    return m_grad;
}

bool Tensor::get_requires_grad() const {
    return m_requires_grad;
}

const std::optional<NodeId>& Tensor::get_node() const {
    return m_node_id;
}

const ttnn::Shape& Tensor::get_shape() const {
    return get_value().logical_shape();
}

uint32_t Tensor::get_rank() const {
    return get_shape().rank();
}

}  // namespace ttml::autograd
