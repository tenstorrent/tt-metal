#pragma once

#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <functional>

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// namespace py = pybind11;

#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/edge.hpp"

template <typename T>
constexpr auto type_name(const T&) noexcept {
    std::string_view name = __PRETTY_FUNCTION__;
    std::string_view prefix = "auto type_name(const T &) [T = ";
    std::string_view suffix = "]";
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}

namespace tt {

namespace graphlib {
struct OpType;

// pass through
bool default_node_filter(Node*);

// Returns a topological sort of all nodes in the graph
std::vector<Node*>
topological_sort(
    Graph const& graph,
    std::function<bool(Node*)> node_filter = default_node_filter,
    bool unroll_loops = false);

// Find the longest path from the graph. Optionally look for paths that don't start from ordered inputs.
std::vector<Node *> get_longest_path(const Graph *graph, bool from_inputs_only = true);

std::vector<Node*> get_nodes_with_indegree_zero(Graph* graph);
std::vector<Node*> get_nodes_with_outdegree_zero(Graph* graph);
std::vector<Node*> get_nodes_with_data_outdegree_zero(Graph* graph);

// Insert new node on the given edge. Node attributes will be picked up from consumer node.
// Returns new edges to and from the new node.
std::pair<Edge, Edge> insert_node_on_edge(Graph *graph, Edge &edge, Node *node, bool inherit_consumer_attrs = true);

// Bypass node, connecting its source to its destination(s). The node must only have one input operand.
// Optionally, user can provide callback on each of the newly created edges, and original edge.
std::unique_ptr<Node> bypass_node(
    Graph *graph, Node *node, bool remove_node, std::function<void(Edge, Edge)> callback = [](Edge, Edge) {});

// Replace node with a new one, removing the old one and reconnecting all edges as before.
// The new node must have the same number of operands, or skip_operands must be set.
void replace_node(Graph *graph, Node *original_node, Node *new_node, bool skip_operands);

// Replace implicit bcasts with explicit ones
void convert_implicit_to_explicit_bcasts(Graph *graph, Edge edge);

// Insert squeezes / unsqueezes to satisfy change in rank
// void handle_change_rank(graphlib::Graph *graph, graphlib::Edge edge);
// void handle_change_rank(graphlib::Graph *graph, graphlib::Node *node);

// This function clones the input producer node and creates a new edge connection replacing
// the old edge. user_edge must come from an input node, returns new edge.
graphlib::Edge clone_input_forking_edge(graphlib::Graph *graph, graphlib::Edge user_edge);

// graphlib::Shape post_tms_shape(
//     Graph const *graph,
//     graphlib::Edge edge,
//     std::function<bool(graphlib::OpType const &)> tm_filter = [](graphlib::OpType const &) { return true; });

// Calculate node shape from operand shapes, using python callback
// void calculate_and_set_node_shape(Graph *graph, Node *node);

std::vector<tt::graphlib::UBlockOrder> get_input_ublock_order(Graph const *graph, Node const *node);
tt::graphlib::UBlockOrder get_output_ublock_order(Graph const *graph, Node const *node);

// Return a vector of pairs of optimizer parameter input nodes and optimizer key names for a given model parameter node
class InputNode;
std::vector<std::pair<InputNode *, std::string>> get_optimizer_param_info(const Graph *graph, const Node *model_parameter);

bool is_recompute(const Graph *graph, const Node *node);
Node* get_fwd_from_recompute(const Graph *graph, const Node *node);

bool can_swap_operands(Graph *graph, Node *node);
void swap_operands(Graph *graph, Node *node);

//
// Consteval
//
bool is_consteval_capable_input_type(Node *node);
// Note: if allow_forks is true, caller must explicitly deal with splitting the fork, consteval
//       inputs have no way of naturally dealing with a fork. Only used by consteval pass.
// bool is_consteval_capable_op(Graph *graph, Node *node, bool allow_forks = false);
// Returns removed runtime node if successful consteval else nullptr
// std::unique_ptr<Node> try_consteval_op(Graph *graph, Node *node);

class ConstEvalGraph {
   public:
    explicit ConstEvalGraph(std::string const &name, Node *runtime_input, bool promote_input, int unique_id = -1);
    Graph *get_graph()
    {
        TT_ASSERT(not ran_autograd or not graph_updated_since_autograd);
        return &consteval_graph;
    }
    Node *get_output() { return consteval_output; }
    bool has_node_with_name(std::string const &n) const { return consteval_graph.has_node_with_name(n); }
    std::unique_ptr<Node> promote_node(std::unique_ptr<Node> &&consteval_node);
    std::unique_ptr<Node> promote_node(Graph *runtime_graph, Node *runtime_node);
    std::unique_ptr<ConstEvalGraph> clone(Node *runtime_input, std::string const &new_input_node_name = "");
    void pad_output_to_buda_dims(std::string const &name_prefix);
    void set_needs_autograd(bool new_needs_autograd) { needs_autograd = new_needs_autograd; }
    // void autograd();

   private:
    std::unique_ptr<Node> promote_node(
        Graph *runtime_graph, Node *runtime_node, std::unique_ptr<Node> &&consteval_node);
    Node *graft(Graph *other);

   private:
    Graph consteval_graph;
    Node *runtime_input = nullptr;
    Node *consteval_output = nullptr;
    std::unordered_map<NodeId, NodeId> runtime_to_consteval_map;
    bool needs_autograd = false;
    bool ran_autograd = false;
    bool graph_updated_since_autograd = false;
};

}  // namespace graphlib
}  // namespace tt
