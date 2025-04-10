// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "graph.hpp"

#include <fmt/core.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <typeinfo>
#include <vector>

#include "core/debug.hpp"
#include "core/system_utils.hpp"

namespace ttml::autograd {

const std::vector<std::vector<size_t>>& Graph::get_edges() const {
    return m_links;
}

std::vector<GraphNode>& Graph::get_graph_nodes() {
    return m_graph_nodes;
}

NodeId Graph::add_node(GradFunction&& grad_function, std::span<NodeId> links) {
    size_t curr_id = m_graph_nodes.size();
    if (core::debug::Debug::enable_backward_performance_measurement()) {
        //  we are using this wrapper to measure the time taken by each node.
        GradFunction wrapper = [grad_function = std::move(grad_function), curr_id, this]() {
            const std::type_info& typeInfo = grad_function.target_type();
            auto demangled_name = core::demangle(typeInfo.name());
            auto time = std::chrono::high_resolution_clock::now();
            fmt::print("Node {} Demangled name {} start running...\n", curr_id, demangled_name);
            grad_function();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time);
            fmt::print(
                "Node {} took {} ms Demangled name {}\n", curr_id, (double)duration.count() / 1000., demangled_name);
        };
        m_graph_nodes.emplace_back(std::move(wrapper));
    } else {
        m_graph_nodes.emplace_back(std::move(grad_function));
    }

    auto& node_links = m_links.emplace_back();
    node_links.reserve(links.size());
    for (const auto& link : links) {
        node_links.push_back(link.get_id());
    }

    return {curr_id, this};
}

NodeId::NodeId(size_t node_id, Graph* graph) : m_node_id(node_id), m_graph(graph) {
}

size_t NodeId::get_id() const {
    return m_node_id;
}

Graph& NodeId::get_graph() const {
    return *m_graph;
}

void Graph::reset() {
    m_graph_nodes.clear();
    m_links.clear();
}
}  // namespace ttml::autograd
