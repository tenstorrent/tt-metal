
#include <memory>

#include "graph_lib/node.hpp"
#include "common/assert.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
namespace tt {

namespace graphlib {

NodeContext::NodeContext(Node* node, int output_index) :
    id(node->id()), output_index(output_index), type(node->node_type()), shape(node->shape()), output_df(node->output_df()) {}

NodeId Node::id() const {
    TT_ASSERT(unique_id_ >= 0);
    return unique_id_;
}
NodeId Node::pybuda_id() const {
    return pybuda_id_;
}

void Node::set_id(NodeId node_id) { unique_id_ = node_id; }
void Node::set_pybuda_id(NodeId node_id) { pybuda_id_ = node_id; }
const std::string& Node::name() const { return name_; }
void Node::set_name(const std::string& name) { name_ = name; }

// instruction-specific methods:
Shape Node::shape() const { return shape_; }
Shape Node::shape_of_operand(const Graph* graph, const Node* operand, bool ignore_broadcasts) const {
    // Takes into account TMs along the edge from operand to this Node
    bool found_operand = false;
    Shape operand_shape;
    for (graphlib::Edge &e : graph->operand_data_edges(this)) {
        if (e.producer_node_id != operand->id())
        {
            continue;
        }

        found_operand = true;

        operand_shape = graph->node_by_id(e.producer_node_id)->shape();
        std::vector<OpType> tms = graph->get_edge_attributes(e)->get_tms();
        for (OpType tm: tms)
        {
            if (ignore_broadcasts and tm.op == "broadcast")
                continue;
            std::vector<Shape> shapes = {operand_shape};
            // std::tuple<Shape, std::vector<DimBroadcast>> shape_data = get_op_shape(tm, shapes, graph->get_ir_level() == IRLevel::IR_BUDA);
            // operand_shape = std::get<0>(shape_data);
            // TT_ASSERT(std::get<1>(shape_data).size() == 0, "TMs should not cause broadcasts");
        }
    }

    if (!found_operand)
    {
        throw std::invalid_argument("Provided operand was not an operand of this node");
    }

    return operand_shape;
}
NodeType Node::node_type() const { return node_type_; }
void Node::set_node_type(NodeType node_type) { node_type_ = node_type; }

void Node::set_shape(const Shape& shape) {
    TT_ASSERT(shape.is_valid());
    shape_ = shape;
}

tt::DataFormat Node::output_df() const { return output_df_; }
void Node::set_output_df(tt::DataFormat df) { output_df_ = df; }

std::unique_ptr<Node> Node::clone(std::string const&)
{
    TT_ASSERT(false, "a derived type node does not have clone() defined");
    return nullptr;
}

void Node::clone(Node const* other, std::string const& name)
{
    name_ = name.empty() ? other->name() : name;
    shape_ = other->shape_;
    epoch_type_ = other->epoch_type_;
    output_df_ = other->output_df_;
}

std::string Node::get_type() const
{
    if (node_type_ == NodeType::kPyOp or node_type_ == NodeType::kBudaOp) {
        OpNode const* op = this->as<OpNode>();
        return node_type_to_string(node_type_) + "::" + op->op_name();
    } else {
        return node_type_to_string(node_type_);
    }
}

std::ostream& operator<<(std::ostream& out, const Node& node) {
    out << node.name() << "(id=" << node.id() << "): " << node.node_type();
    return out;
}

// TODO: move this to appropriate factory calls
void Node::set_backward() { epoch_type_ = NodeEpochType::Backward; }
void Node::set_optimizer() { epoch_type_ = NodeEpochType::Optimizer; }
bool Node::is_forward() const { return epoch_type_ == NodeEpochType::Forward; }
bool Node::is_backward() const { return epoch_type_ == NodeEpochType::Backward; }
bool Node::is_optimizer() const { return epoch_type_ == NodeEpochType::Optimizer; }
NodeEpochType Node::get_epoch_type() const { return epoch_type_; }
void Node::set_epoch_type(NodeEpochType epoch_type) { epoch_type_ = epoch_type; }

std::string node_type_to_string(const NodeType& node_type)
{
    switch (node_type) {
        case NodeType::kInput: return "Input";
        case NodeType::kOutput: return "Output";
        case NodeType::kQueue: return "Queue";
        case NodeType::kBudaOp: return "BudaOp";
        case NodeType::kBudaNaryTM: return "BudaNaryTM";
        case NodeType::kPyOp: return "PyBudaOp";
        case NodeType::kIntConstant: return "IntConstant";
        default: TT_ASSERT(false, "Invalid node type");
    }
    return "";
}


std::string node_epoch_type_to_string(const NodeEpochType& node_epoch_type)
{
    switch (node_epoch_type) {
        case NodeEpochType::Forward: return "Forward";
        case NodeEpochType::Backward: return "Backward";
        case NodeEpochType::Optimizer: return "Optimizer";
        default: TT_ASSERT(false, "Invalid node epoch type");
    }
    return "";
}

std::ostream& operator<<(std::ostream& out, const NodeType& node_type)
{
    return out << node_type_to_string(node_type);
}

}  // namespace graphlib
}  // namespace tt
