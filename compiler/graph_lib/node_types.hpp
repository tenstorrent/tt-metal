#pragma once

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <variant>

#include "graph_lib/shape.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/common.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/edge.hpp"

namespace tt {
class FusedOp;

namespace graphlib {

bool is_permute_xy_order(const std::vector<int>& order);
std::vector<int> create_permute_xy_order(const int rank);

// fwd declares
class Graph;
class ConstEvalGraph;

// Simple create(..) function under the hood just constructs a unique-ptr of the specified type with arguments forwarded and returns the object.
template <typename ClassType, typename... ClassArgs>
std::unique_ptr<ClassType> create_node(ClassArgs&&... args) {
    return std::make_unique<ClassType>(std::forward<ClassArgs>(args)...);
}

enum QueueNodeType {
    Input,
    Output,
    EpochToEpoch,
    GradAccumulator,
    Buffering,
};

enum MemoryAccessType {
    FIFO,
    RAM,
};

class QueueNode : public Node {
protected:
    QueueNodeType queue_type_;
    MemoryAccessType memory_access_type_;
    int entries_;

   public:
    QueueNode(const std::string &name, QueueNodeType queue_type, NodeType node_type = NodeType::kQueue)
        : Node(name, node_type), queue_type_(queue_type), memory_access_type_(MemoryAccessType::FIFO), entries_(0) { }
    std::string queue_type_string() const;

    virtual std::unique_ptr<Node> clone(std::string const& name = "") override;

    void set_num_entries(int entries) { entries_ = entries; }
    int get_num_entries() const { return entries_; }

    bool is_epoch_to_epoch() const { return queue_type_ == EpochToEpoch; }
    bool is_grad_accumulator() const { return queue_type_ == GradAccumulator; }
    bool is_input() const { return queue_type_ == Input; }
    bool is_output() const { return queue_type_ == Output; }
    bool is_buffering() const { return queue_type_ == Buffering; }

    QueueNodeType queue_type() const { return this->queue_type_; }
    void set_queue_type(QueueNodeType queue_type) { this->queue_type_ = queue_type; }

    MemoryAccessType memory_access_type() const { return this->memory_access_type_; }
    void set_memory_access_type(MemoryAccessType memory_access_type) { this->memory_access_type_ = memory_access_type; }
    std::string memory_access_type_string() const;
};

class EpochToEpochQueueNode : public QueueNode {
protected:
    bool cross_epoch_type_; // it's used between two epochs that are not of the same type (usually fwd->bwd)
public:
    EpochToEpochQueueNode(const std::string &name, bool cross_epoch_type) :
        QueueNode(name, QueueNodeType::EpochToEpoch), cross_epoch_type_(cross_epoch_type) {}
    virtual std::unique_ptr<Node> clone(std::string const& name = "") override;

    bool is_cross_epoch_type() const { return cross_epoch_type_; }
};


class BufferingQueueNode : public QueueNode {
public:
    BufferingQueueNode(const std::string &name, int num_entries) :
        QueueNode(name, QueueNodeType::Buffering)
    {
        this->set_num_entries(num_entries);
    }
    virtual std::unique_ptr<Node> clone(std::string const& name = "") override;
};

enum InputNodeType {
    Parameter,
    Constant,
    Accumulator,
    Activation,
    Loss,
    OptimizerParameter,
    Target,
};

class InputNode : public QueueNode {
private:
    InputNodeType input_type_;
    bool requires_grad_;
    std::vector<int> tile_broadcast_dims_;
    bool prologue_ = false;
    std::string fractured_parameter_mapping_;
    graphlib::Shape reinterpret_original_shape;
    graphlib::Shape reinterpret_shape;

   protected:
    std::unique_ptr<ConstEvalGraph> consteval_graph_;

public:
    InputNode(const std::string &name, InputNodeType input_type, bool requires_grad);
    virtual ~InputNode();

    InputNodeType input_type() const { return input_type_; }
    std::string input_type_string() const;
    bool requires_grad() const { return requires_grad_; }
    void clone_consteval_graph_from(Node* original);
    ConstEvalGraph *get_consteval_graph(bool create = false, bool promote_input = false);
    void clear_consteval_graph();
    virtual std::unique_ptr<Node> clone(std::string const& name = "") override;

    void set_tile_broadcast_dim(int dim) { tile_broadcast_dims_.push_back(dim); }
    std::vector<int> get_tile_broadcast_dims() const { return tile_broadcast_dims_; }

    void set_prologue(bool prologue) { prologue_ = prologue; }
    bool is_prologue() const { return prologue_; }
    void set_fractured_parameter_mapping(std::string name) { fractured_parameter_mapping_ = name; }
    std::string get_fractured_parameter_mapping() const { return fractured_parameter_mapping_; }

    void set_reinterpret_shape(graphlib::Shape original, graphlib::Shape reinterpret)
    {
        if (not reinterpret_original_shape.is_valid())
            reinterpret_original_shape = original;
        reinterpret_shape = reinterpret;
    }
    std::pair<graphlib::Shape, graphlib::Shape> get_reinterpret_shape() const
    {
        return std::make_pair(reinterpret_original_shape, reinterpret_shape);
    }

    bool is_constant() const { return input_type_ == Constant; }
    bool is_parameter() const { return input_type_ == Parameter; }
    bool is_loss() const { return input_type_ == Loss; }
    bool is_target() const { return input_type_ == Target; }
    bool is_accumulator() const { return input_type_ == Accumulator; }
    bool is_activation() const { return input_type_ == Activation; }
    bool is_optimizer_parameter() const { return input_type_ == OptimizerParameter; }
};

enum ConstantInputNodeType {
    SingleValue,
    SingleTile,
    Tensor,
};

class ConstantInputNode : public InputNode {
private:
    ConstantInputNodeType node_type_;
    float constant_value_;
    std::vector<float> tile_value_;
    std::shared_ptr<void> tensor_handle_;
    Shape tensor_shape_;

    int dim_r_;
    int dim_c_;
public:
    ConstantInputNode(const std::string &name, float constant_value, int dim_r = -1, int dim_c = -1)
        : InputNode(name, InputNodeType::Constant, NodeType::kInput), 
          node_type_(ConstantInputNodeType::SingleValue), constant_value_(constant_value), dim_r_(dim_r), dim_c_(dim_c) {}
    ConstantInputNode(const std::string &name, std::vector<float> const &tile_value)
        : InputNode(name, InputNodeType::Constant, NodeType::kInput), 
          node_type_(ConstantInputNodeType::SingleTile), tile_value_(tile_value), dim_r_(-1), dim_c_(-1) {}
    ConstantInputNode(const std::string &name, std::shared_ptr<void> tensor_handle, Shape const &tensor_shape) :
        InputNode(name, InputNodeType::Constant, NodeType::kInput),
        node_type_(ConstantInputNodeType::Tensor),
        tensor_handle_(tensor_handle),
        tensor_shape_(tensor_shape),
        dim_r_(-1),
        dim_c_(-1) {
            set_shape(tensor_shape);
        }

    virtual std::unique_ptr<Node> clone(std::string const& name = "") override;
    bool is_single_value() const { return this->node_type_ == ConstantInputNodeType::SingleValue; }
    bool is_single_tile() const { return this->node_type_ == ConstantInputNodeType::SingleTile; }
    bool is_tensor() const { return this->node_type_ == ConstantInputNodeType::Tensor; }
    float constant_value() const { TT_ASSERT(is_single_value()); return this->constant_value_; }
    std::pair<int, int> constant_dims() const { TT_ASSERT(is_single_value()); return std::make_pair(dim_r_, dim_c_); }
    const std::vector<float> &tile_value() const { TT_ASSERT(is_single_tile()); return tile_value_; }
    std::shared_ptr<void> tensor() const { TT_ASSERT(is_tensor()); return tensor_handle_; }
    void set_tensor_handle(std::shared_ptr<void> t_h) { TT_ASSERT(is_tensor()); this->tensor_handle_ = t_h; }
    const Shape &tensor_shape() const { TT_ASSERT(is_tensor()); return tensor_shape_; }

    bool equivalent(const ConstantInputNode *other) const;
};

class OutputNode : public QueueNode {
protected:
    bool requires_grad_;
    bool is_loss_output_;
    bool untilize_;
    graphlib::Shape reinterpret_original_shape;
    graphlib::Shape reinterpret_shape;

   public:
    OutputNode(std::string name) : QueueNode(name, QueueNodeType::Output, NodeType::kOutput), requires_grad_(false), is_loss_output_(false), untilize_(true) {}
    bool requires_grad() const { return requires_grad_; }
    bool is_loss_output() const { return is_loss_output_; }
    bool untilize() const { return untilize_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    void set_loss_output() { is_loss_output_ = true; }
    void set_untilize(bool should_untilize) { untilize_ = should_untilize; }
    virtual std::unique_ptr<Node> clone(std::string const& name = "") override;

    void set_reinterpret_shape(graphlib::Shape original, graphlib::Shape reinterpret)
    {
        reinterpret_original_shape = original;
        reinterpret_shape = reinterpret;
    }
    std::pair<graphlib::Shape, graphlib::Shape> get_reinterpret_shape() const
    {
        return std::make_pair(reinterpret_original_shape, reinterpret_shape);
    }
};

struct OpType {
    using Attr = BudaOpAttr;

    std::string op;
    std::vector<Attr> attr;
    BudaOpAttrs buda_attrs;
    std::vector<std::string> tags = {}; // arbitrary hints from one compile pass to another

    bool operator==(const char *name) const { return op == name; }
    bool operator==(const std::string &name) const { return op == name; }
    bool operator==(const OpType &other) const
    {
        return op == other.op and attr == other.attr and buda_attrs == other.buda_attrs and tags == other.tags;
    }
    bool operator!=(const OpType &other) const { return not(*this == other); }

    bool has_tag(const std::string &tag) const { return std::find(tags.begin(), tags.end(), tag) != tags.end(); }
    void tag(const std::string &tag) { if (!has_tag(tag)) tags.push_back(tag); }

    std::string as_string() const {
        std::string ret = op;
        if (attr.size() > 0) {
            ret += "(";
            for (unsigned int i = 0; i < attr.size(); i++) {
                if (std::holds_alternative<bool>(attr[i])) {
                    ret += std::to_string(std::get<bool>(attr[i])) + ",";
                } else if (std::holds_alternative<int>(attr[i])) {
                    ret += std::to_string(std::get<int>(attr[i])) + ",";
                } else if (std::holds_alternative<float>(attr[i])) {
                    ret += std::to_string(std::get<float>(attr[i])) + ",";
                } else if (std::holds_alternative<std::string>(attr[i])) {
                    ret += std::get<std::string>(attr[i]) + ",";
                } else {
                    TT_ASSERT(false, "Unknown alternative in Attr");
                }
            }
            // ret += "(" + std::to_string(attr[0]);
            // for (std::size_t i = 1; i < attr.size(); i++) ret += "," + std::to_string(attr[i]);
            ret += ")";
        }
        return ret;
    }

};

class OpNode : public Node {
private:
    OpType op_type_;
    bool gradient_op_; // accumulator op
    std::vector<OpType> golden_transforms;

    // fusing/graph changes have the output of this node be equal to a different golden node
    bool has_golden_id_ = false;
    std::uint32_t golden_id_; 

public:
    OpNode(const std::string &name, const std::string &op_type, NodeType node_type) : Node(name, node_type), op_type_({op_type, {}, {}}), gradient_op_(false) {}
    OpNode(const std::string &name, OpType op_type, NodeType node_type) : Node(name, node_type), op_type_(op_type), gradient_op_(false) {}
    void change_op_type(const std::string &new_op_type, std::vector<OpType::Attr> attrs = {}) {
        op_type_ = {new_op_type, attrs, {}};
    }
    void change_op_type(OpType const &new_op_type) { op_type_ = new_op_type; }
    OpType op_type() const { return op_type_; }
    const std::string &op_name() const { return op_type_.op; }
    const std::vector<OpType::Attr> &op_attrs() const { return op_type_.attr; }
    void overwrite_op_attrs(std::vector<OpType::Attr> op_attrs) { op_type_.attr = op_attrs; }
    const BudaOpAttrs &buda_attrs() const { return op_type_.buda_attrs; }
    void overwrite_buda_attrs(BudaOpAttrs buda_attrs) { op_type_.buda_attrs = buda_attrs; }
    void set_gradient_op(bool value = true) { gradient_op_ = value; }
    bool is_gradient_op() const { return gradient_op_; }
    bool is_matmul() const { return op_name().find("matmul") != std::string::npos; }
    bool is_sparse_matmul() const { return is_matmul() and (buda_attrs().find("identity") != buda_attrs().end()); }
    void set_output_df_from_operands(const Graph * graph);
    void add_golden_transform(OpType const& op_type) { golden_transforms.insert(golden_transforms.begin(), op_type); }
    void set_golden_transforms(std::vector<OpType> const &other) { golden_transforms = other; }
    std::vector<OpType> const& get_golden_transforms() const { return golden_transforms; }
    std::vector<OpType> &get_golden_transforms() { return golden_transforms; }

    void set_golden_id(std::uint32_t golden_id) { has_golden_id_ = true; golden_id_ = golden_id; }
    bool has_golden_id() const { return has_golden_id_; }
    std::uint32_t golden_id() const { TT_ASSERT(has_golden_id_); return golden_id_; }
};

class PyOpNode : public OpNode {

public:
    PyOpNode(const std::string &name, const std::string &op_type) : OpNode(name, op_type, NodeType::kPyOp) {}
    PyOpNode(const std::string &name, OpType op_type) : OpNode(name, op_type, NodeType::kPyOp) {}
    virtual std::unique_ptr<Node> clone(std::string const& name = "") override;

    void copy_parent_op_attributes(PyOpNode *node);
};

class BudaOpNode : public OpNode {

private:
    tt::DataFormat accumulate_df_  = tt::DataFormat::Float16_b;
    tt::DataFormat intermediate_df_  = tt::DataFormat::Float16_b;
    tt::MathFidelity math_fidelity_  = tt::MathFidelity::HiFi3;
    std::shared_ptr<FusedOp> fused_op_ = nullptr;
    std::unordered_map<std::uint32_t, std::uint32_t> min_input_buffer_multiplier_;
    bool buffering_op_ = false;

public:
    BudaOpNode(const std::string &name, const std::string &op_type) : OpNode(name, op_type, NodeType::kBudaOp) {}
    BudaOpNode(const std::string &name, OpType op_type) : OpNode(name, op_type, NodeType::kBudaOp) {}

    tt::DataFormat intermediate_df() const { return intermediate_df_; }
    void set_intermediate_df(tt::DataFormat df) { intermediate_df_ = df; }

    tt::DataFormat accumulate_df() const { return accumulate_df_; }
    void set_accumulate_df(tt::DataFormat df) { accumulate_df_ = df; }

    tt::MathFidelity math_fidelity() const { return math_fidelity_; }
    void set_math_fidelity(tt::MathFidelity mf) { math_fidelity_ = mf; }

    void copy_lowered_op_attributes(PyOpNode *node);
    void copy_parent_op_attributes(BudaOpNode *node);

    std::uint32_t min_input_buffer_multiplier(std::uint32_t operand, std::uint32_t default_value) const;
    void set_min_input_buffer_multiplier(std::uint32_t operand, std::uint32_t tiles); 

    virtual std::unique_ptr<Node> clone(std::string const& name = "") override;

    void set_fused_op(std::shared_ptr<FusedOp> fused_op) { fused_op_ = fused_op; }
    bool is_fused_op() const { return fused_op_ != nullptr; }
    std::shared_ptr<FusedOp> get_fused_op() const { TT_ASSERT(fused_op_ != nullptr); return fused_op_; }

    void set_buffering_op(bool buffering_op) { buffering_op_ = buffering_op; }
    bool is_buffering_op() const { return buffering_op_; }
};

class BudaNaryTMNode : public Node
{
   private:
    OpType op_type_;

   public:
    BudaNaryTMNode(const std::string &name, OpType const &op_type) :
        Node(name, NodeType::kBudaNaryTM), op_type_(op_type)
    {
    }
    OpType const &op_type() const { return op_type_; }
    void change_op_type(OpType const &new_op_type) { op_type_ = new_op_type; }
    const std::string &tm_name() const { return op_type_.op; }
    const std::vector<OpType::Attr> &tm_attrs() const { return op_type_.attr; }
    const BudaOpAttrs &buda_attrs() const { return op_type_.buda_attrs; }
    void copy_lowered_op_attributes(PyOpNode *node);
};

class IntConstantNode : public Node {

private:
    int value_;
public:
    IntConstantNode(const std::string &name, int value) : Node(name, NodeType::kIntConstant), value_(value) { this->set_shape(Shape::create({1, 1, 1, 1})); }
    int value() const { return value_; }
};

// Modifiable edge attributes outside of Edge itself because Edge is mostly immutable in current
// graph design
class EdgeAttributes {

private:
    EdgeType edge_type_;
    std::vector<OpType> tms;
    UBlockOrder ublock_order = UBlockOrder::R;

   public:
    EdgeAttributes(EdgeType edge_type) : edge_type_(edge_type) {}
    virtual ~EdgeAttributes() = default;

    bool has_broadcast_dims() const;
    void clear_broadcast_dims();
    void set_broadcast_dim(int dim, int size_or_factor, bool explicit_bcast = false)
    {
        tms.push_back({.op = "broadcast", .attr = {dim, size_or_factor, explicit_bcast}, .buda_attrs = {}});
    }
    inline UBlockOrder get_ublock_order() const { return ublock_order; }
    inline void set_ublock_order(UBlockOrder new_ublock_order) { ublock_order = new_ublock_order; }
    void append_tm(OpType type) { tms.push_back(type); }
    void set_tms(std::vector<OpType> new_tms) { tms = new_tms; }
    void append_tms(std::vector<OpType> new_tms) { tms.insert(tms.end(), new_tms.begin(), new_tms.end()); }
    void prepend_tm(std::string op_type, std::vector<OpType::Attr> attrs) {
        tms.insert(tms.begin(), {.op = op_type, .attr = attrs, .buda_attrs = {}});
    }
    void prepend_tm(OpType type) { tms.insert(tms.begin(), type); }

    const std::vector<OpType> &get_tms() const { return tms; }
    std::vector<OpType> &get_tms() { return tms; }

    // Copy values from another edge attributes
    void copy_from(const EdgeAttributes &other)
    {
        tms = other.tms;
        ublock_order = other.ublock_order;
    }

    EdgeType edge_type() const { return edge_type_; }
    bool has_tms() const { return not tms.empty(); }
    bool has_tm(std::string const &tm) const
    {
        return std::find_if(tms.begin(), tms.end(), [tm](OpType const &op) { return op.op == tm; }) != tms.end();
    }

    static std::shared_ptr<EdgeAttributes> create(EdgeType edge_type);

    // Checked casting to sub-node type
    template <typename T> static std::shared_ptr<T> as(std::shared_ptr<EdgeAttributes>& base);
    template <typename T> static const std::shared_ptr<T> as(const std::shared_ptr<EdgeAttributes>& base);
};

class LoopEdgeAttributes : public EdgeAttributes {
public:
    // TypeDefs
    using IterationParametersMap = std::unordered_map<std::string, std::vector<std::string>>;
    struct LoopEdgeAttributesInternal {
        int loop_iterations_;
        IterationParametersMap parameter_to_matched_parameters_;
        std::unordered_set<NodeId> nodes_processed_in_loop_;
    };

    explicit LoopEdgeAttributes(EdgeType edge_type) : EdgeAttributes(edge_type) {}
    LoopEdgeAttributes(EdgeType edge_type, const LoopEdgeAttributesInternal&& attributes)
        : EdgeAttributes(edge_type), attributes(std::move(attributes)) {}
    LoopEdgeAttributes(EdgeType edge_type, const LoopEdgeAttributesInternal& attributes)
        : EdgeAttributes(edge_type), attributes(attributes) {}

    int loop_iterations() const { return this->attributes.loop_iterations_; }
    bool is_processed_in_loop(NodeId node_id) const {
        return attributes.nodes_processed_in_loop_.find(node_id) != attributes.nodes_processed_in_loop_.end();
    }
    const std::vector<std::string> matched_parameters(const std::string& parameter) const {
        return attributes.parameter_to_matched_parameters_.at(parameter);
    }
    std::string matched_parameters(const std::string& parameter, int loop_iteration_idx) const {
        return attributes.parameter_to_matched_parameters_.at(parameter).at(loop_iteration_idx);
    }
    void set_loop_iterations(int loop_iterations) { this->attributes.loop_iterations_ = loop_iterations; }
    void set_iteration_parameters(const IterationParametersMap& parameter_to_matched_parameters) {
        this->attributes.parameter_to_matched_parameters_ = parameter_to_matched_parameters;
    }
    void set_nodes_processed_in_loop(const std::unordered_set<NodeId>& nodes_processed_in_loop){
        this->attributes.nodes_processed_in_loop_ = nodes_processed_in_loop;
    }
private:
    LoopEdgeAttributesInternal attributes;
};

bool op_type_is_accumulate(const std::string &type);

std::ostream& operator<<(std::ostream& out, const NodeType& opcode);
std::ostream &operator<<(std::ostream &out, const OpType &op_type);
std::ostream &operator<<(std::ostream &out, const UBlockOrder &ublock_order);

}  // namespace graphlib
}  // namespace tt
