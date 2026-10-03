// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "core/not_null.hpp"

namespace ttml::autograd {
class Graph;
class Tensor;
struct GraphNode;

using GradFunction = std::function<void()>;

struct GraphNode {
    GradFunction grad_function;
    std::vector<std::weak_ptr<Tensor>> outputs;
};

class NodeId {
public:
    NodeId(size_t node_id, Graph* graph);
    [[nodiscard]] size_t get_id() const;
    [[nodiscard]] Graph& get_graph() const;

private:
    size_t m_node_id = 0;
    core::not_null<Graph*> m_graph;
};

class Graph {
private:
    std::vector<GraphNode> m_graph_nodes;
    std::vector<std::vector<size_t>> m_links;

public:
    [[nodiscard]] const std::vector<std::vector<size_t>>& get_edges() const;
    [[nodiscard]] std::vector<GraphNode>& get_graph_nodes();
    NodeId add_node(
        GradFunction&& grad_function, std::span<NodeId> links, std::span<const std::shared_ptr<Tensor>> outputs = {});
    void set_node_output(size_t node_id, const std::shared_ptr<Tensor>& output);

    void reset();
};

}  // namespace ttml::autograd
