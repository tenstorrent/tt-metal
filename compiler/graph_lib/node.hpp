#pragma once

#include "common/assert.hpp"

#include <memory>
#include <unordered_map>
#include <functional>
#include "graph_lib/common.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/shape.hpp"

namespace tt {

namespace graphlib {

class Graph;
class Node;

std::string node_epoch_type_to_string(const NodeEpochType& node_epoch_type);
std::string node_type_to_string(const NodeType& node_type);
std::ostream& operator<<(std::ostream& out, const NodeType& node_type);

// Node context is used to represent graph nodes in python, and is a small, copy-able summary
// of a node that can be passed around as subgraphs are defined in lightweight python code.
struct NodeContext {
  NodeId id; // node id
  std::uint32_t output_index;       // output index of the op, if it produces multiple outputs (rare)
  NodeType type;
  Shape shape;
  Shape unbroadcast_shape;
  tt::DataFormat output_df;

  NodeContext(tt::graphlib::Node *node, int output_index = 0);
};


// Base class for graph node. All node types sublass for this to implement specific
// node type behaviours.
class Node {
   private:
    std::string name_;
    NodeId unique_id_ = -1;
    NodeId pybuda_id_ = -1;

   protected:
    NodeType node_type_;
    Shape shape_;
    NodeEpochType epoch_type_ = NodeEpochType::Forward;
    tt::DataFormat output_df_ = tt::DataFormat::Float16_b;

   public:
    Node(std::string name, NodeType node_type) : name_(name), node_type_(node_type) {}
    virtual ~Node() = default;

    NodeId id() const;
    NodeId pybuda_id() const;
    void set_id(NodeId node_id);
    void set_pybuda_id(NodeId node_id);
    const std::string& name() const;
    void set_name(const std::string& name);

    NodeType node_type() const;
    void set_node_type(NodeType node_type);

    Shape shape() const;
    Shape shape_of_operand(const Graph* graph, const Node* operand, bool ignore_broadcasts = false) const;
    virtual void set_shape(const Shape& shape);

    tt::DataFormat output_df() const;
    void set_output_df(tt::DataFormat df);

    void set_backward();
    void set_optimizer();
    bool is_forward() const;
    bool is_backward() const;
    bool is_optimizer() const;

    virtual std::unique_ptr<Node> clone(std::string const& name = "");

    std::string get_type() const;
    void set_epoch_type(NodeEpochType epoch_type);
    NodeEpochType get_epoch_type() const;

    // Checked casting to sub-node type
    template <typename T> T* as();
    template <typename T> const T* as() const;

   protected:
    void clone(Node const* other, std::string const& name = "");
};

std::ostream& operator<<(std::ostream& out, const Node& node);
}  // namespace graphlib
}  // namespace tt

