#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph_lib/common.hpp"
#include "graph_lib/edge.hpp"

// Jumping through some hoops to allow modifiable edge attributes
struct EdgeUniqueIdHash : public std::unary_function<tt::graphlib::EdgeUniqueId, std::size_t>
{
 std::size_t operator()(const tt::graphlib::EdgeUniqueId& edge) const
 {
        std::size_t seed = 0;
        tt::utils::hash_combine(seed, static_cast<std::size_t>(std::get<0>(edge)));
        tt::utils::hash_combine(seed, static_cast<std::size_t>(std::get<1>(edge)));
        tt::utils::hash_combine(seed, static_cast<std::size_t>(std::get<2>(edge)));
        tt::utils::hash_combine(seed, static_cast<std::size_t>(std::get<3>(edge)));
        tt::utils::hash_combine(seed, static_cast<std::size_t>(std::get<4>(edge)));
        return seed;
 }
};

#include <iostream>
namespace tt {

namespace graphlib {

class Node;
class EdgeAttributes;

enum class IRLevel {
    IR_PYBUDA,
    IR_BUDA,
    IR_CONSTEVAL,
};

class Graph {
   public:
    using NodeIdToEdgeSet = std::unordered_map<NodeId, std::unordered_set<Edge>>;
    using NodeIdToNodePtr = std::unordered_map<NodeId, Node *>;
    using NodeIdToNodeUniquePtr = std::unordered_map<NodeId, std::unique_ptr<Node>>;
    using NodeNameToNodeId = std::unordered_map<std::string, NodeId>;
    using EdgeToAttributes = std::unordered_map<EdgeUniqueId, std::shared_ptr<EdgeAttributes>, EdgeUniqueIdHash>;

    ~Graph() = default;
    Graph(IRLevel ir_level);
    Graph(IRLevel ir_level, std::string name, int unique_id = -1);
    Graph(const Graph &other) = delete;

    static NodeId generate_unique_node_id() { return ++last_node_id_assigned_; }
    static GraphId generate_unique_graph_id() { 
        while (Graph::assigned_graph_ids_.find(last_graph_id_assigned_) != Graph::assigned_graph_ids_.end()) {
            last_graph_id_assigned_++;
        }
        return last_graph_id_assigned_++; 
    }

    // instead of copy-constructor, prefer to explicitly provide a clone() method
    // for clearer semantics and avoid accidental copy-constructor usage.
    // Optionally pass your own graph ptr to clone into otherwise,
    // Returns a raw (buda-compatibility) pointer to heap-allocated deep copy of the input graph
    Graph *clone(Graph *cloned = nullptr) const;

    IRLevel get_ir_level() const { return ir_level_; }

    // Node-level queries
    Node *node_by_id(NodeId id) const { 
      // Some kind of memory corruption here when running from python... even though the element is in
      // the map, the "at" function throws an exception.
      // Working around for now, but need to run some valgrind or something to figure it out :(.
      try {
        return this->nodes_map_raw_.at(id); 
      } catch (...) {
        for(const auto& elem : this->nodes_map_raw_)
          if (elem.first == id) return elem.second;
      }
      throw std::runtime_error("Node not found");
    }
    Node *get_node_by_name(const std::string& name) const;

    bool has_node_with_id(NodeId id) const { return this->nodes_map_.find(id) != this->nodes_map_.end(); }
    bool has_node_with_name(const std::string& name) const { return this->node_name_to_node_id_.find(name) != this->node_name_to_node_id_.end(); }
    NodeId generate_unique_id() { return ++last_node_id_assigned_; }

    std::unordered_set<Edge> &operand_edges_set(NodeId node_id);
    const std::unordered_set<Edge> &operand_edges_set(const Node *node) const;

    std::vector<Edge> edges(EdgeType edge_type) const;
    std::vector<Edge> edges(
        const Node *node, std::function<bool(Edge)> edge_filter = [](Edge) { return true; }) const;
    std::vector<Edge> operand_edges(
        const Node *node, std::function<bool(Edge)> edge_filter = [](Edge) { return true; }) const;
    std::vector<Edge> user_edges(
        const Node *node, std::function<bool(Edge)> edge_filter = [](Edge) { return true; }) const;
    std::vector<Edge> operand_data_edges(
        const Node *node, std::function<bool(Edge)> edge_filter = [](Edge) { return true; }) const;
    std::vector<Edge> user_data_edges(
        const Node *node, std::function<bool(Edge)> edge_filter = [](Edge) { return true; }) const;

    std::unordered_set<Edge> &user_edges_set(NodeId node_id);
    const std::unordered_set<Edge> &user_edges_set(const Node *node) const;
    std::vector<Node *> operands(const Node *node) const;
    std::vector<Node *> data_operands(const Node *node) const;
    std::vector<Node *> users(const Node *node) const;
    std::vector<Node *> data_users(const Node *node) const;
    std::unordered_set<NodeId> node_ids();

    std::vector<Edge> get_edges(const Node *producer, const Node *consumer) const;
    std::pair<PortId, int> output_port_and_index_for_data_user_port(const Node *node, Edge user_edge) const;

    // Given an ordered list of edges between (producer, consumer)
    // return an ordered list of <producer_output_port, edge_index_for_given_producer_output_port>
    std::vector<std::pair<PortId, int>> output_port_and_index_for_consumer(const Node *producer, const Node *consumer) const;
    std::vector<Edge> user_data_edges_for_operand_port(const Node *node, PortId port_id) const;

    std::shared_ptr<EdgeAttributes> get_edge_attributes(const Edge &edge) const;
    void copy_edge_attributes(const Edge &src_edge, const Edge &dest_edge, const Graph *old_graph = nullptr);
    void copy_node_attributes(Node *src, Node *dst);

    // graph-level queries

    // templated to allow us to return a derived-class pointer back
    // to the user.
    template <typename NodeClassType>
    NodeClassType *add_node(std::unique_ptr<NodeClassType> node);

    // If edge_attributes not provided explicitly, we default construct them.
    void add_edge(const Edge& edge, std::shared_ptr<EdgeAttributes> edge_attributes = nullptr);

    void add_edge(
        const Node &producer,
        const Node &consumer,
        PortId producer_output_port_id,
        PortId consumer_input_port_id,
        EdgeType edge_type = EdgeType::kData);
    void add_edge(
        const Node *producer,
        const Node *consumer,
        PortId producer_output_port_id,
        PortId consumer_input_port_id,
        EdgeType edge_type = EdgeType::kData);

    void add_edge(const Node &producer, const Node &consumer, EdgeType edge_type = EdgeType::kData);

    void add_edge(const Node *producer, const Node *consumer, EdgeType edge_type = EdgeType::kData);

    void add_edge(
        Node *producer,
        Node *consumer,
        PortId producer_output_port_id,
        PortId consumer_input_port_id,
        EdgeType edge_type = EdgeType::kData);

    int num_nodes() const { return this->nodes_map_.size(); }
    int num_operands(NodeId node_id) const { return this->operands_map_.at(node_id).size(); }
    int num_users(NodeId node_id) const { return this->users_map().at(node_id).size(); }
    int num_users(const Node *node) const;

    // Note: swap these out for ttl::ordered map or ensure assigned ids are monotonically increasing
    // otherwise we may get non-deterministic iteration of key-value pairs
    // Note: prefer to explicitly type until we stabilize.. alias these types later
    const NodeIdToEdgeSet &operands_map() const { return operands_map_; }
    const NodeIdToEdgeSet &users_map() const { return users_map_; }

    NodeIdToEdgeSet &operands_map() { return operands_map_; }
    NodeIdToEdgeSet &users_map() { return users_map_; }

    const NodeIdToNodePtr &nodes_map() const;
    std::vector<Node*> nodes(std::function<bool(Node*)> node_filter) const;
    std::vector<Node*> nodes_by_type(NodeType type) const;
    const std::vector<Node*>& nodes() const;

    std::unique_ptr<Node> remove_node(const NodeId node_id);
    std::unique_ptr<Node> remove_node(const Node *node);

    void remove_edge(const Edge &edge);
    void set_id(int id) { this->unique_id_ = id; }
    int id() const { return this->unique_id_; }
    const std::string& name() const { return this->name_; }
    bool enable_training() const { return enable_training_; }
    void set_enable_training(bool enable_training) { enable_training_ = enable_training; }
    int get_microbatch() const { return microbatch_; }
    void set_microbatch(int microbatch) { microbatch_ = microbatch; }

    void update_node_name(Node *node, const std::string &new_name);

    void register_module_inputs(const std::vector<NodeId>& module_inputs);
    void register_module_outputs(const std::vector<NodeId>& module_outputs, std::vector<bool> requires_grad);
    void register_module_targets(const std::vector<NodeId>& module_targets);
    void copy_module_inputs(Graph *old_graph, const std::unordered_map<Node *, Node *> &old_to_new);
    void copy_module_outputs(Graph *old_graph, const std::unordered_map<Node *, Node *> &old_to_new);
    void copy_module_targets(Graph *old_graph, const std::unordered_map<Node *, Node *> &old_to_new);

    // Get tile broadcast dims for various types of inputs
    std::vector<int> get_tile_broadcast_dims_for_input(std::uint32_t input_index) const;
    std::vector<int> get_tile_broadcast_dims_for_bw_input(std::uint32_t output_index) const;
    std::vector<int> get_tile_broadcast_dims_for_target(std::uint32_t target_index) const;

    // Return inputs to the graph in order they were added
    std::vector<Node *> ordered_module_inputs() const;
    std::vector<Node *> ordered_module_outputs() const;
    std::vector<Node *> get_constant_nodes(bool recurse = false) const;
    std::vector<Node *> get_parameter_nodes() const;
    std::vector<std::string> get_constant_names() const;
    std::vector<std::string> get_ordered_input_names() const;
    std::vector<std::string> get_ordered_output_names() const;
    std::vector<std::string> get_ordered_input_gradient_names() const;
    std::vector<std::string> get_ordered_output_gradient_names() const;
    std::vector<std::string> get_ordered_target_names() const;
    std::vector<bool> get_ordered_input_requires_grad() const;
    std::vector<bool> get_ordered_output_requires_grad() const;
    std::vector<std::vector<std::uint32_t>> get_ordered_input_shapes() const;
    std::vector<std::vector<std::uint32_t>> get_ordered_target_shapes() const;
    std::vector<std::vector<std::uint32_t>> get_ordered_output_shapes() const;

    bool contains_nodes_of_epoch_type(NodeEpochType node_epoch_type) const;

    // Autograd mapping retrieval
    bool contains_bwd_nodes() const;
    bool contains_opt_nodes() const;
    bool contains_recompute_nodes() const;

    std::unordered_map<int, std::vector<Node *>> get_recompute_nodes(Node *fwd_node) const;
    std::unordered_map<int, std::vector<Node *>> get_bwd_nodes(Node *fwd_node) const;
    std::unordered_map<int, std::vector<Node *>> get_gradient_nodes(Node *fwd_node) const;
    std::unordered_map<int, std::vector<Node *>> get_opt_nodes(Node *fwd_node) const;

    void dump(std::string const &pass_name) const;

   private:
    // two attributes to accomodate user-assigned graph-ids
    static GraphId last_graph_id_assigned_;
    static std::unordered_set<GraphId> assigned_graph_ids_;
    static NodeId last_node_id_assigned_;

    std::string name_ = "";
    GraphId unique_id_;
    IRLevel ir_level_;
    bool enable_training_ = false;
    int microbatch_ = 0;

    std::vector<NodeId> ordered_module_input_node_ids_;
    std::vector<NodeId> ordered_module_output_node_ids_;
    std::vector<NodeId> ordered_module_target_node_ids_;

    // ordered by insertion order
    std::vector<Node *> nodes_;

    NodeNameToNodeId node_name_to_node_id_;
    NodeIdToNodePtr nodes_map_raw_;

    NodeIdToNodeUniquePtr nodes_map_;
    NodeIdToEdgeSet operands_map_;
    NodeIdToEdgeSet users_map_;

    EdgeToAttributes edge_to_attr_map_;
};

template <typename NodeClassType>
NodeClassType *Graph::add_node(std::unique_ptr<NodeClassType> node) {
    NodeId node_id = this->generate_unique_node_id();
    node->set_id(node_id);

    if (this->has_node_with_name(node->name())) {
        throw std::runtime_error("In graph " + std::to_string(this->id()) + ": trying to add a node with a name that already exists: " + node->name() + "\n");
    }

    node_name_to_node_id_[node->name()] = node_id;
    nodes_map_[node_id] = std::move(node);
    NodeClassType *result = (NodeClassType *)nodes_map_[node_id].get();
    nodes_.push_back(result);
    nodes_map_raw_[node_id] = result;
    operands_map_[node_id] = {};
    users_map_[node_id] = {};
    return result;
}

std::ostream &operator<<(std::ostream &out, const Edge &e);
std::ostream &operator<<(std::ostream &out, const Graph &g);

}  // namespace graphlib

}  // namespace tt
