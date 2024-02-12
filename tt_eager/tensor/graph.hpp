// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt {

namespace tt_metal {

template <typename KeyType, typename ValueType>
using Map = std::unordered_map<KeyType, ValueType>;

template <typename NodeType, typename NodeAttributesType, typename EdgeAttributesType>
class Graph {
    struct EdgeMap {
        Map<NodeType, EdgeAttributesType> map;
        std::size_t size() const { return this->map.size(); }

        const auto begin() const { return std::begin(this->map); }
        const auto end() const { return std::end(this->map); }
    };

    Map<NodeType, NodeAttributesType> _node;
    Map<NodeType, EdgeMap> _succ;
    Map<NodeType, EdgeMap> _pred;

   public:
    bool contains(const NodeType& node) const { return this->_node.count(node) > 0; }

    std::size_t size() const { return this->_node.size(); }

    void add_node(const NodeType& node, const NodeAttributesType& attributes = {}) {
        this->_node[node] = attributes;
        this->_pred[node] = EdgeMap{};
        this->_succ[node] = EdgeMap{};
    }

    void add_edge(const NodeType& source, const NodeType& sink, const EdgeAttributesType& attributes = {}) {
        if (not this->contains(source)) {
            this->add_node(source);
        }
        if (not this->contains(sink)) {
            this->add_node(sink);
        }
        this->_succ[source].map.insert({sink, attributes});
        this->_pred[sink].map.insert({source, attributes});
    }

    Map<NodeType, std::size_t> in_degree() const {
        Map<NodeType, std::size_t> output;
        for (auto&& [node, _] : this->_node) {
            output[node] = this->_pred.at(node).size();
        }
        return output;
    }

    const auto& succ(const NodeType& node) const { return this->_succ.at(node); }
};

struct Node {
    std::string name;
};

bool operator==(const Node& lhs, const Node& rhs) { return lhs.name == rhs.name; }

template <typename NodeType, typename... Rest>
std::vector<NodeType> topological_sort(const Graph<NodeType, Rest...>& graph) {
    auto output = std::vector<NodeType>{};
    output.reserve(graph.size());

    auto indegree_map = graph.in_degree();
    auto zero_indegree = std::vector<NodeType>{};
    for (auto&& [node, in_degree] : indegree_map) {
        if (in_degree == 0) {
            zero_indegree.push_back(node);
        }
    }

    while (not zero_indegree.empty()) {
        auto new_zero_indegree = std::vector<NodeType>{};

        for (const auto& node : zero_indegree) {
            output.push_back(node);
            if (not graph.contains(node)) {
                throw std::runtime_error("Graph changed during iteration");
            }

            for (auto&& [child, _] : graph.succ(node)) {
                // indegree_map[child] -= len(G[node][child]) if multigraph else 1
                indegree_map[child] -= 1;
                if (indegree_map[child] == 0) {
                    new_zero_indegree.push_back(child);
                    indegree_map.erase(child);
                }
            }
        }
        std::swap(zero_indegree, new_zero_indegree);
    }

    return output;
}

struct EdgeAttributes {
    const std::size_t source_output_index;
    const std::size_t sink_input_index;
};

class NodeAttributes {};

struct GraphHook {
    Node node;
    std::size_t output_index;
};

}  // namespace tt_metal

}  // namespace tt

namespace std {
template <>
struct hash<tt::tt_metal::Node> {
    std::size_t operator()(const tt::tt_metal::Node& node) const { return std::hash<std::string>{}(node.name); }
};
}  // namespace std

namespace tt {
namespace tt_metal {
constexpr bool ENABLE_GRAPH = false;
inline Graph<Node, NodeAttributes, EdgeAttributes> GRAPH;
}  // namespace tt_metal
}  // namespace tt
