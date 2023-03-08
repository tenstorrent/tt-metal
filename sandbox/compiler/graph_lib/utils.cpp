#include <functional>
#include <map>
#include <unordered_set>
#include <vector>
#include <queue>

#include "graph_lib/graph.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/utils.hpp"
#include "graph_lib/node_types.hpp"
// #include "autograd/binding.hpp"
// #include "common/logger.hpp"

namespace tt {

namespace graphlib {

bool default_node_filter(Node*) {
    return true;
}

static bool requires_visit(const std::unordered_map<NodeId, bool>& visited, NodeId node_id) {
    return visited.find(node_id) == visited.end() or visited.at(node_id) == false;
}

std::vector<Node*> topological_sort(const Graph& graph, std::function<bool(Node*)> node_filter, bool unroll_loops) {
    std::vector<Node*> result;
    std::unordered_map<NodeId, bool> visited{};
    std::unordered_map<Edge, int> control_loop_edge_to_iteration;

    std::vector<Node*> nodes = graph.nodes();
    std::queue<Node*> node_queue;
    for (Node* node : nodes) {
        node_queue.push(node);
    }

    std::function<void(Node*)> VisitNode = [&](Node* node) {
        visited[node->id()] = true;

        for (const Edge& operand_edge : graph.operand_edges(node)) {
            if (operand_edge.edge_type == EdgeType::kDataLoopback)
            {
                continue;
            }
            else if (operand_edge.edge_type == EdgeType::kControlLoop)
            {
                continue; // not unrolling loops, just terminate
            }


            NodeId predecessor_id = operand_edge.producer_node_id;
            Node* predecessor_node = graph.node_by_id(predecessor_id);
            if (requires_visit(visited, predecessor_id)) {
                VisitNode(predecessor_node);
            }
        }
        if (node_filter(node)) {
            result.push_back(node);
        }

        if (unroll_loops)
        {
            for (const Edge& user_edge : graph.user_edges(node)) {
                if (user_edge.edge_type == EdgeType::kControlLoop)
                {
                    auto loop_attributes = EdgeAttributes::as<LoopEdgeAttributes>(graph.get_edge_attributes(user_edge));
                    if (control_loop_edge_to_iteration.find(user_edge) == control_loop_edge_to_iteration.end())
                    {
                        control_loop_edge_to_iteration[user_edge] = 1; // initialize loop count
                    }
                    if (control_loop_edge_to_iteration[user_edge] < loop_attributes->loop_iterations())
                    {
                        // Re-enqueue nodes in the same order they were originally intended to be processed
                        for (Node* node : nodes) {
                            if (loop_attributes->is_processed_in_loop(node->id())) {
                                visited[node->id()] = false;
                                node_queue.push(node);

                            }
                        }
                    }
                    control_loop_edge_to_iteration[user_edge] += 1;
                }
            }
        }


    };

    while (not node_queue.empty()) {
        Node* node = node_queue.front();

        if (requires_visit(visited, node->id())) {
            VisitNode(node);
        }
        node_queue.pop();
    }
    return result;
}

// Find the longest path from the graph. Optionally look for paths that don't start from ordered inputs.
// TODO: write a few unit tests
std::vector<Node *> get_longest_path(const Graph *graph, bool from_inputs_only) {

    std::unordered_map<Node *, int> cost;
    std::unordered_map<Node *, Node *> parent_map;

    if (from_inputs_only) {
        // set big negative numbers on all other inputs
        for (Node *node : graph->nodes()) cost.emplace(std::make_pair(node, std::numeric_limits<int>::lowest()));
        for (Node *node : graph->ordered_module_inputs()) cost[node] = 0;
    }

    int max_distance = std::numeric_limits<int>::lowest();
    Node *max_path_output = NULL;
    for (Node *node: topological_sort(*graph))
    {
        for (Node *user : graph->data_users(node))
        {
            if (cost[user] < cost[node] + 1) {
                cost[user] = cost[node] + 1;
                parent_map[user] = node;
            }
            if (cost[node] > max_distance) {
                max_distance = cost[node];
                max_path_output = node;
            }
        }
    }

    std::vector<Node *> max_path = {max_path_output};
    while (parent_map.find(max_path_output) != parent_map.end())
    {
        max_path_output = parent_map.at(max_path_output);
        max_path.push_back(max_path_output);
    }

    std::reverse(max_path.begin(), max_path.end());

    return max_path;
}

std::vector<Node *> get_nodes_with_indegree_zero(Graph* graph) {
    std::vector<Node *> indegree_zero_nodes;
    for (Node *node : graph->nodes()) {
        int num_operands = 0;
        for (auto operand : graph->operands(node)) {
            if (operand->node_type() != NodeType::kInput) {
                num_operands++;
            }
        }
        if (num_operands == 0) {
            if (node->node_type() != NodeType::kInput) {
                indegree_zero_nodes.push_back(node);
            }
        }
    }
    return indegree_zero_nodes;
}

std::vector<Node *> get_nodes_with_outdegree_zero(Graph* graph) {
    std::vector<Node*> outdegree_zero_nodes;
    for (Node* node : graph->nodes()) {
        if (graph->users(node).size() == 0) {
            if (node->node_type() != NodeType::kInput) {
                outdegree_zero_nodes.push_back(node);
            }
        }
    }
    return outdegree_zero_nodes;
}

std::vector<Node *> get_nodes_with_data_outdegree_zero(Graph* graph) {
    std::vector<Node*> outdegree_zero_nodes;
    for (Node* node : graph->nodes()) {
        if (graph->user_data_edges(node).size() == 0) {
            if (node->node_type() != NodeType::kInput) {
                outdegree_zero_nodes.push_back(node);
            }
        }
    }
    return outdegree_zero_nodes;
}


// Insert new node on the given edge. Node attributes will be picked up from consumer node.
std::pair<Edge, Edge> insert_node_on_edge(Graph *graph, Edge &edge, Node *node, bool inherit_consumer_attrs)
{
    Node *consumer = graph->node_by_id(edge.consumer_node_id);
    Node *producer = graph->node_by_id(edge.producer_node_id);

    graph->copy_node_attributes(inherit_consumer_attrs ? consumer : producer, node);

    // Don't copy "gradient op" flag, since the last node is still the one accumulating
    if ( (node->node_type() == NodeType::kBudaOp) ||
         (node->node_type() == NodeType::kPyOp) )
        node->as<graphlib::OpNode>()->set_gradient_op(false);

    // Create new edges
    Edge new_edge0 = Edge(
            edge.producer_node_id,
            edge.producer_output_port_id,
            node->id(),
            0,
            edge.edge_type);

    Edge new_edge1 = Edge(
            node->id(),
            0,
            edge.consumer_node_id,
            edge.consumer_input_port_id,
            edge.edge_type);

    graph->add_edge(new_edge0);
    graph->add_edge(new_edge1);
    graph->copy_edge_attributes(edge, new_edge0);
    graph->get_edge_attributes(new_edge1)->set_ublock_order(graph->get_edge_attributes(edge)->get_ublock_order());

    bool edges_added = false;
    for (Edge &e : graph->operand_edges(consumer)) {
        // Adjust control & autograd edges
        if ( (e.edge_type != EdgeType::kData) && (e.edge_type != EdgeType::kAutogradOutputToLoss) &&
                (e.edge_type != EdgeType::kAutogradInputToGradientOut) &&
                (e.edge_type != EdgeType::kAutogradFwdToGradient)  &&
                (e.edge_type != EdgeType::kAutogradFwdToRecompute)

                ) {
            edges_added = true;
            graph->add_edge(
                    graph->node_by_id(e.producer_node_id),
                    node,
                    e.producer_output_port_id,
                    0,
                    e.edge_type);
        }
    }

    // If the producer was in backward (or optimizer) epoch, and there are fwd->bwd edges going to it,
    // the need to go to the new op, too
    if (not edges_added and producer->get_epoch_type() != graphlib::NodeEpochType::Forward)
    {
        for (Edge &e : graph->operand_edges(producer)) {
            // Adjust control & autograd edges
            if ( (e.edge_type == EdgeType::kAutogradFwdToBwd) ||
                (e.edge_type == EdgeType::kAutogradFwdToOptimizer) ||
                (e.edge_type == EdgeType::kAutogradFwdToGradient) )
            {
                graph->add_edge(
                        graph->node_by_id(e.producer_node_id),
                        node,
                        e.producer_output_port_id,
                        0,
                        e.edge_type);
            }
            // Move the kAutogradFwdToGradient edge, since we can only have one
            if (e.edge_type == EdgeType::kAutogradFwdToGradient) {
                graph->remove_edge(e);
            }
        }
    }
    // If the consumer of the edge we're trying to add a node on is a "recompute-node",
    // we need to also create replicated fwd->recompute edges on the newly added node.
    // this is to keep track of which nodes are considered to be "recompute".
    for (Edge &e : graph->operand_edges(consumer)) {
        if (e.edge_type == EdgeType::kAutogradFwdToRecompute)
        {
            Node* fwd_node_being_recompute = graph->node_by_id(e.producer_node_id);
            graph->add_edge(
                    fwd_node_being_recompute,
                    node,
                    e.producer_output_port_id,
                    0,
                    e.edge_type);
        }
    }


    graph->remove_edge(edge);

    return std::make_pair(new_edge0, new_edge1);
}

// Copy non-data edges from old dest to new
void copy_control_edges(Graph *graph, Node *old_dest, Node *new_dest)
{
    std::vector<Node*> data_operands = graph->data_operands(old_dest);
    Node *data_operand = data_operands.at(0);

    for (Edge &e : graph->operand_edges(old_dest)) {
        if (e.edge_type == EdgeType::kData) {
            continue;
        }
        Node *new_consumer = data_operand;

        if (new_consumer->node_type() != NodeType::kBudaOp) {
            // If `new_dest` is an OutputNode, we'll fetch it off of its data-operand since we still want to
            // copy this control edge over (consider kInputToGradient being connected to kOutput node)
            new_consumer = data_operand;
        }

        if (new_consumer->node_type() != NodeType::kBudaOp) {
            continue;
        }

        if ((e.edge_type == EdgeType::kAutogradFwdToBwd and new_consumer->get_epoch_type() != NodeEpochType::Backward)
            or (e.edge_type == EdgeType::kAutogradFwdToOptimizer and new_consumer->get_epoch_type() != NodeEpochType::Optimizer))
        {
            // There are cases where we're trying to connect kAutogradFwdToBwd on a Fwd consumer node which doesn't make sense.
            continue;
        }

        // Copy control & autograd edges
        graph->add_edge(
                graph->node_by_id(e.producer_node_id),
                new_consumer,
                e.producer_output_port_id,
                e.consumer_input_port_id,
                e.edge_type);
    }

    for (Edge &e : graph->user_edges(old_dest)) {
        if (e.edge_type == EdgeType::kData) {
            continue;
        }

        // Copy control & autograd edges
        if (e.edge_type == EdgeType::kControl) {
            graph->add_edge(new_dest, graph->node_by_id(e.consumer_node_id), 0, 0, e.edge_type);
        } else {
            // if it's an autograd-edge between <NODE_TO_DELETE> -> consumer, we'll reassign
            // the edge to the producer node since `new_dest` may be an output node
            graph->add_edge(data_operand, graph->node_by_id(e.consumer_node_id), 0, 0, e.edge_type);
        }
    }
}

// Copy non-data edges when removing a node
void handle_control_edges_when_removing_node(Graph *graph, Node *node_being_removed)
{
    std::vector<Edge> operand_data_edges = graph->operand_data_edges(node_being_removed);
    TT_ASSERT(
        operand_data_edges.size() == 1,
        "Tried to handle control edges, but node being removed has more than 1 operand!");

    Edge& producer_to_nbr_edge = operand_data_edges.front();
    Node* producer = graph->node_by_id(producer_to_nbr_edge.producer_node_id);

    auto is_not_data_edge = [](Edge e) {
        return (e.edge_type != EdgeType::kData);
    };
    std::vector<Edge> operand_edges = graph->operand_edges(node_being_removed, is_not_data_edge);
    std::vector<Edge> user_edges = graph->user_edges(node_being_removed, is_not_data_edge);

    // Handle operand edges
    for (Edge& o_e : operand_edges)
    {
        if (node_being_removed->is_forward())
        {
            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(o_e.edge_type));
        }
        else if (node_being_removed->is_backward())
        {
            if (o_e.edge_type == EdgeType::kAutogradFwdToBwd)
            {
                // We can just "delete" this edge, i.e. not copy it
                continue;
            }
            if (o_e.edge_type == EdgeType::kAutogradFwdToGradient)
            {
                graph->add_edge(graph->node_by_id(o_e.producer_node_id), producer, o_e.edge_type);
                continue;
            }
            if (o_e.edge_type == EdgeType::kAutogradFwdToRecompute)
            {
                // We can just "delete" this edge, i.e. not copy it
                continue;
            }

            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(o_e.edge_type));
        }
        else if (node_being_removed->is_optimizer())
        {
            if (o_e.edge_type == EdgeType::kAutogradFwdToOptimizer)
            {
                // We can just "delete" this edge, i.e. not copy it
                continue;
            }

            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(o_e.edge_type));
        }

        // TODO: Other control edges
    }

    // Handle user edges
    for (Edge& u_e : user_edges)
    {
        if (node_being_removed->is_forward())
        {
            if (u_e.edge_type == EdgeType::kAutogradFwdToBwd)
            {
                // Push the edge to parent of node being removed
                graph->add_edge(producer, graph->node_by_id(u_e.consumer_node_id), u_e.edge_type);
                continue;
            }
            if (u_e.edge_type == EdgeType::kAutogradFwdToOptimizer)
            {
                graph->add_edge(producer, graph->node_by_id(u_e.consumer_node_id), u_e.edge_type);
                continue;
            }
            if (u_e.edge_type == EdgeType::kAutogradFwdToGradient)
            {
                // Since there will be no fwd node anymore, we can just delete this edge
                continue;
            }
            if (u_e.edge_type == EdgeType::kAutogradFwdToRecompute)
            {
                // Moving this edge from nbr(fwd)->recompute(bwd) to nbr's_parent(fwd)->recompute(bwd)
                // Not sure this makes sense though, depends what the edge is used for later on
                graph->add_edge(producer, graph->node_by_id(u_e.consumer_node_id), u_e.edge_type);
                continue;
            }

            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(u_e.edge_type));
        }
        else if (node_being_removed->is_backward())
        {
            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(u_e.edge_type));
        }
        else if (node_being_removed->is_optimizer())
        {
            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(u_e.edge_type));
        }
        // TODO: Other control edges
    }
}

// Bypass node, connecting its source to its destination(s). The node must only have one input operand.
// Optionally, user can provide callback on each of the newly created edges, and original edge.
std::unique_ptr<Node> bypass_node(Graph *graph, Node *node, bool remove_node, std::function<void(Edge, Edge)> callback)
{
    std::vector<Edge> op_edges = graph->operand_data_edges(node);
    TT_ASSERT(op_edges.size() == 1, "bypass_node can only be called on nodes with one operand");

    Edge src_edge = op_edges[0];
    std::vector<graphlib::OpType> operand_tms = graph->get_edge_attributes(src_edge)->get_tms();


    for (Edge &user : graph->user_data_edges(node))
    {
        std::vector<graphlib::OpType> user_tms = graph->get_edge_attributes(user)->get_tms();

        Edge new_edge(
                src_edge.producer_node_id,
                src_edge.producer_output_port_id,
                user.consumer_node_id,
                user.consumer_input_port_id,
                user.edge_type);
        graph->add_edge(new_edge);

        std::vector<graphlib::OpType> new_edge_tms;
        new_edge_tms.insert(new_edge_tms.end(), operand_tms.begin(), operand_tms.end());
        new_edge_tms.insert(new_edge_tms.end(), user_tms.begin(), user_tms.end());

        auto new_edge_attributes = graph->get_edge_attributes(new_edge);
        new_edge_attributes->set_tms(new_edge_tms);

        callback(new_edge, user);
    }

    handle_control_edges_when_removing_node(graph, node);

    OpNode *op_node = dynamic_cast<OpNode *>(node);
    if (op_node and op_node->is_gradient_op()) {
        OpNode *producer_op_node = dynamic_cast<OpNode *>(graph->node_by_id(src_edge.producer_node_id));
        if (producer_op_node)
            producer_op_node->set_gradient_op();
    }

    return remove_node ? graph->remove_node(node) : nullptr;
}

// Replace node with a new one, removing the old one and reconnecting all edges as before.
// The new node must have the same number of operands, or skip_operands must be set.
void replace_node(Graph *graph, Node *original_node, Node *new_node, bool skip_operands)
{
    if (!skip_operands) {
        for (Edge &operand : graph->operand_data_edges(original_node)) {
            Edge new_edge = Edge(
                    operand.producer_node_id,
                    operand.producer_output_port_id,
                    new_node->id(),
                    operand.consumer_input_port_id,
                    operand.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(operand, new_edge);
        }
    }

    for (Edge &user : graph->user_edges(original_node)) {
        if (user.edge_type == graphlib::EdgeType::kData) {
            Edge new_edge = Edge(
                    new_node->id(),
                    (graphlib::PortId)0,
                    user.consumer_node_id,
                    user.consumer_input_port_id,
                    user.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(user, new_edge);
        }
    }

    copy_control_edges(graph, original_node, new_node);
    graph->copy_node_attributes(original_node, new_node);
    graph->remove_node(original_node);
}

void convert_implicit_to_explicit_bcasts(Graph *graph, Edge edge)
{
    auto edge_attr = graph->get_edge_attributes(edge);
    for (OpType &op_type : graph->get_edge_attributes(edge)->get_tms())
    {
        if (op_type.op == "broadcast")
        {
            constexpr bool explicit_bcast = true;
            std::get<bool>(op_type.attr[2]) = explicit_bcast;
        }
    }
}

// void handle_change_rank(graphlib::Graph *graph, graphlib::Edge edge)
// {
//     auto get_consumer_size = [](std::uint32_t producer_size, graphlib::Node *node)
//     {
//         std::uint32_t consumer_size = node->shape().size();
//         graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
//         if (not op)
//             return consumer_size;
//         if (op->op_name() == "reshape")
//             return producer_size;
//         if (op->op_name() == "squeeze")
//             return (consumer_size + 1);
//         if (op->op_name() == "unsqueeze")
//             return (consumer_size - 1);
//         return consumer_size;
//     };

//     auto producer_size = graph->node_by_id(edge.producer_node_id)->shape().size();
//     auto consumer_size = get_consumer_size(producer_size, graph->node_by_id(edge.consumer_node_id));

//     if (producer_size == consumer_size)
//         return;

//     // This is one of the few cases where we actually want to move tms downstream
//     auto tms = graph->get_edge_attributes(edge)->get_tms();
//     graph->get_edge_attributes(edge)->set_tms({});

//     auto insert = [graph](graphlib::Edge edge, std::string op, std::uint32_t rank) -> graphlib::Edge
//     {
//         graphlib::Node *producer = graph->node_by_id(edge.producer_node_id);
//         graphlib::Node *consumer = graph->node_by_id(edge.consumer_node_id);
//         graphlib::OpNode *inherit = dynamic_cast<graphlib::OpNode *>(consumer)
//                                         ? dynamic_cast<graphlib::OpNode *>(consumer)
//                                         : dynamic_cast<graphlib::OpNode *>(producer);
//         TT_ASSERT(inherit);
//         // If there are 2 edges from the same producer to the same consumer (eg. eltwise binary op),
//         // need edge_creation_id to differentiate naming.
//         std::string name = producer->name() + "_" + consumer->name() + "_" + op + std::to_string(rank) + "_" + std::to_string(edge.edge_creation_id);
//         graphlib::OpNode *change_rank = dynamic_cast<graphlib::OpNode *>(graph->add_node(inherit->clone(name)));
//         TT_ASSERT(change_rank);
//         auto attr = (op == "squeeze") ? std::vector<graphlib::OpType::Attr>{0}
//                                       : std::vector<graphlib::OpType::Attr>{0, ((int)rank - 1)};
//         change_rank->change_op_type(graphlib::OpType{.op = op, .attr = attr, .buda_attrs = {}});
//         change_rank->set_shape(producer->shape().as_rank(rank));
//         auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, edge, change_rank);
//         if (try_consteval_op(graph, change_rank))
//             return graph->operand_data_edges(consumer)[0];
//         return outgoing_edge;
//     };

//     int orig_producer_size = (int)producer_size;
//     while (producer_size < consumer_size)
//     {
//         producer_size++;
//         edge = insert(edge, "unsqueeze", producer_size);
//     }

//     while (producer_size > consumer_size)
//     {
//         producer_size--;
//         TT_ASSERT(producer_size > 0);
//         edge = insert(edge, "squeeze", producer_size);
//     }

//     int diff = (int)producer_size - orig_producer_size;
//     for (OpType &op_type : tms)
//     {
//         if (op_type.op == "broadcast")
//         {
//             if (std::get<int>(op_type.attr[0]) >= 0)
//                 std::get<int>(op_type.attr[0]) += diff;
//         }
//     }
//     graph->get_edge_attributes(edge)->set_tms(tms);
// }

// void handle_change_rank(graphlib::Graph *graph, graphlib::Node *node)
// {
//     for (graphlib::Edge e : graph->operand_data_edges(node)) handle_change_rank(graph, e);
//     for (graphlib::Edge e : graph->user_data_edges(node)) handle_change_rank(graph, e);
// }

graphlib::Edge clone_input_forking_edge(graphlib::Graph *graph, graphlib::Edge user_edge)
{
    Node *input = graph->node_by_id(user_edge.producer_node_id);
    TT_ASSERT(input->node_type() == NodeType::kInput);
    TT_ASSERT(graph->data_operands(input).empty(), "Cannot clone a loopback input");
    TT_ASSERT(graph->data_users(input).size() > 1, "Cannot clone input that doesn't fork");
    Node *clone =
        graph->add_node(input->clone(input->name() + "_fork_clone" + std::to_string(user_edge.consumer_node_id)));

    auto edge_attr = graph->get_edge_attributes(user_edge);
    graph->remove_edge(user_edge);
    graphlib::Edge new_edge(
        clone->id(),
        user_edge.producer_output_port_id,
        user_edge.consumer_node_id,
        user_edge.consumer_input_port_id,
        user_edge.edge_type);
    graph->add_edge(new_edge, edge_attr);
    return new_edge;
}

// graphlib::Shape post_tms_shape(
//     Graph const *graph, graphlib::Edge edge, std::function<bool(graphlib::OpType const &)> tm_filter)
// {
//     graphlib::Shape producer_shape = graph->node_by_id(edge.producer_node_id)->shape();
//     for (OpType const &tm : graph->get_edge_attributes(edge)->get_tms())
//     {
//         if (not tm_filter(tm))
//             continue;
//         std::vector<Shape> shapes = {producer_shape};
//         std::tuple<Shape, std::vector<DimBroadcast>> shape_data =
//             get_op_shape(tm, shapes, graph->get_ir_level() == IRLevel::IR_BUDA);
//         producer_shape = std::get<0>(shape_data);
//         TT_ASSERT(std::get<1>(shape_data).size() == 0, "TMs should not cause broadcasts");
//     }
//     return producer_shape;
// }

// Calculate node shape from operand shapes, using python callback
// void calculate_and_set_node_shape(Graph *graph, Node *node)
// {
//     log_trace(LogGraphCompiler, "Calculate and set node shape for: {} {}", node->name(), node->get_type());
//     // Apply TMs and get post-TM operand shapes
//     std::vector<Shape> operand_shapes;
//     for (graphlib::Edge &e : graph->operand_data_edges(node)) {
//         auto operand_shape = graph->node_by_id(e.producer_node_id)->shape();
//         std::vector<OpType> tms = graph->get_edge_attributes(e)->get_tms();
//         for (OpType tm: tms)
//         {
//             std::vector<Shape> shapes = {operand_shape};
//             std::tuple<Shape, std::vector<DimBroadcast>> shape_data = get_op_shape(tm, shapes, graph->get_ir_level() == IRLevel::IR_BUDA);
//             operand_shape = std::get<0>(shape_data);
//             TT_ASSERT(std::get<1>(shape_data).size() == 0, "TMs should not cause broadcasts");
//             log_trace(LogGraphCompiler, "    TM {} {}", tm.as_string(), operand_shape);
//         }
//         log_trace(
//             LogGraphCompiler,
//             "  Operand[{}] {} {}",
//             e.consumer_input_port_id,
//             operand_shape,
//             graph->node_by_id(e.producer_node_id)->name());
//         operand_shapes.push_back(operand_shape);
//     }

//     if ( (node->node_type() == graphlib::NodeType::kOutput) || (node->node_type() == graphlib::NodeType::kQueue) ) {
//         // Graph shape from first, and only, operand
//         TT_ASSERT(operand_shapes.size() == 1, "Node should have exactly one operand");
//         node->set_shape(operand_shapes[0]);
//         return;
//     }

//     if ((node->node_type() != NodeType::kPyOp) && (node->node_type() != NodeType::kBudaOp) &&
//         (node->node_type() != NodeType::kBudaNaryTM))
//         return;

//     graphlib::OpType op_type = node->node_type() == NodeType::kBudaNaryTM
//                                    ? dynamic_cast<graphlib::BudaNaryTMNode *>(node)->op_type()
//                                    : dynamic_cast<graphlib::OpNode *>(node)->op_type();

//     bool is_fused_op = (node->node_type() == graphlib::kBudaOp) && node->as<graphlib::BudaOpNode>()->is_fused_op();
//     std::tuple<Shape, std::vector<DimBroadcast>> shape_data; //=
//         is_fused_op ? get_fused_op_shape(node->as<graphlib::BudaOpNode>(), operand_shapes) :
//         get_op_shape(op_type, operand_shapes, graph->get_ir_level() == IRLevel::IR_BUDA);

//     log_trace(LogGraphCompiler, "  {}", std::get<0>(shape_data));
//     node->set_shape(std::get<0>(shape_data));

//     // Set broadcast attributes on edges
//     for (graphlib::Edge &e : graph->operand_data_edges(node)) {

//         for (DimBroadcast &b : std::get<1>(shape_data)) {
//             log_trace(LogGraphCompiler, "  brcst {} {} {}", std::get<0>(b), std::get<1>(b), std::get<2>(b));

//             int operand = std::get<0>(b);
//             if (operand == (int)e.consumer_input_port_id) {
//                 int dim = std::get<1>(b);
//                 int size = std::get<2>(b);
//                 bool const is_buda = graph->get_ir_level() == IRLevel::IR_BUDA;
//                 if (is_buda and dim >= 2)
//                 {
//                     size /= graphlib::Shape::BUDA_TILE_DIM;
//                 }
//                 graph->get_edge_attributes(e)->set_broadcast_dim(dim, size);
//             }
//         }
//     }
// }

std::vector<UBlockOrder> get_input_ublock_order(Graph const *graph, Node const *node)
{
    std::vector<UBlockOrder> ublock_order;

    std::vector<Edge> operands = graph->operand_data_edges(node);
    if (graphlib::OpNode const *op_node = dynamic_cast<graphlib::OpNode const *>(node))
    {
        if (op_node->is_matmul())
        {
            auto edge_attrs0 = graph->get_edge_attributes(operands[0]);
            auto edge_attrs1 = graph->get_edge_attributes(operands[1]);
            TT_ASSERT(edge_attrs0->get_ublock_order() == UBlockOrder::C, op_node->name());
            TT_ASSERT(edge_attrs1->get_ublock_order() == UBlockOrder::R, op_node->name());
            ublock_order.push_back(edge_attrs0->get_ublock_order());
            ublock_order.push_back(edge_attrs1->get_ublock_order());
            if (op_node->is_sparse_matmul())
            {
                ublock_order.push_back(UBlockOrder::R);
            }
        }
        else
        {
            auto edge_attrs0 = graph->get_edge_attributes(operands[0]);
            for (Edge edge : operands)
            {
                auto edge_attrs = graph->get_edge_attributes(edge);
                TT_ASSERT(edge_attrs->get_ublock_order() == edge_attrs0->get_ublock_order());
                ublock_order.push_back(edge_attrs->get_ublock_order());
            }
        }
    }
    else
    {
        // Is output or queue node
        TT_ASSERT(operands.size() == 1);
        ublock_order = {graph->get_edge_attributes(operands[0])->get_ublock_order()};
    }

    return ublock_order;
}

UBlockOrder get_output_ublock_order(Graph const *graph, Node const *node)
{
    if (node->node_type() == graphlib::NodeType::kInput)
        return UBlockOrder::R;

    graphlib::OpNode const *op_node = dynamic_cast<graphlib::OpNode const *>(node);
    if (op_node and op_node->op_name() == "reduce")
        return UBlockOrder::R;

    return get_input_ublock_order(graph, node).back();
}

// Return a vector of pairs of optimizer parameter input nodes and optimizer key names for a given model parameter node
std::vector<std::pair<InputNode *, std::string>> get_optimizer_param_info(const Graph *graph, const Node *model_parameter)
{
    // If autograd has run, there will be EdgeType::kAutogradFwdToOptimizer edges. We parse through this
    // list looking for inputs that require its tensors to be populated by the python-side optimizer obj
    std::vector<std::pair<InputNode *, std::string>> ret;
    for (graphlib::Edge edge : graph->user_edges(model_parameter)) {

        if (edge.edge_type != graphlib::EdgeType::kAutogradFwdToOptimizer) continue;
        if (graph->node_by_id(edge.consumer_node_id)->node_type() != NodeType::kInput) continue;

        graphlib::InputNode *input = graph->node_by_id(edge.consumer_node_id)->as<graphlib::InputNode>();
        if (not input->is_optimizer_parameter()) { continue; }

        // Parse out the optimizer-param suffix string and do a lookup to get the tensor
        std::string optimizer_input_name = input->name();
        std::string::size_type optimizer_param_idx = optimizer_input_name.rfind('.');
        TT_ASSERT(optimizer_param_idx != std::string::npos,
                "Expecting optimizer node to have a '.<optimizer-param>' suffix identifier");

        std::string optimizer_param_key = optimizer_input_name.substr(optimizer_param_idx+1);
        ret.push_back(std::make_pair(input, optimizer_param_key));
    }
    return ret;
}

bool is_recompute(const Graph *graph, const Node *node)
{
    for (const Edge& edge : graph->operand_edges(node)) {
        if (edge.edge_type == graphlib::EdgeType::kAutogradFwdToRecompute) {
            return true;
        }
    }
    return false;
}

Node* get_fwd_from_recompute(const Graph *graph, const Node *node)
{
    for (const Edge& edge : graph->operand_edges(node)) {
        if (edge.edge_type == graphlib::EdgeType::kAutogradFwdToRecompute) {
            return graph->node_by_id(edge.producer_node_id);
        }
    }
    return nullptr;
}

ConstEvalGraph::ConstEvalGraph(std::string const &name, Node *runtime_input, bool promote_input, int unique_id) :
    consteval_graph(IRLevel::IR_CONSTEVAL, name, unique_id == -1 ? Graph::generate_unique_graph_id() : unique_id),
    runtime_input(runtime_input)
{
    TT_ASSERT(runtime_input->node_type() == NodeType::kInput);
    if (promote_input)
        promote_node(nullptr, runtime_input, runtime_input->clone());
}

std::unique_ptr<Node> ConstEvalGraph::promote_node(std::unique_ptr<Node> &&consteval_node)
{
    return promote_node(nullptr, nullptr, std::forward<std::unique_ptr<Node>>(consteval_node));
}

std::unique_ptr<Node> ConstEvalGraph::promote_node(Graph *runtime_graph, Node *runtime_node)
{
    return promote_node(runtime_graph, runtime_node, runtime_node->clone());
}

std::unique_ptr<Node> ConstEvalGraph::promote_node(
    Graph *runtime_graph, Node *runtime_node, std::unique_ptr<Node> &&consteval_node_free)
{
    TT_ASSERT(not runtime_graph or runtime_node);
    TT_ASSERT(not runtime_graph or runtime_graph->get_ir_level() == IRLevel::IR_PYBUDA);

    graph_updated_since_autograd = true;

    Node *consteval_node = consteval_graph.add_node<Node>(std::move(consteval_node_free));

    if (consteval_output)
    {
        // Runtime input node needs to always map to the consteval graph output
        auto output_operands = consteval_graph.data_operands(consteval_output);
        TT_ASSERT(output_operands.size() == 1);
        runtime_to_consteval_map[runtime_input->id()] = output_operands[0]->id();
    }

    // Create mapping from runtime node id to consteval
    if (runtime_node)
    {
        runtime_to_consteval_map.insert({runtime_node->id(), consteval_node->id()});
    }

    // Create edges inherited from the runtime_graph
    if (runtime_graph)
    {
        for (Edge const &runtime_edge : runtime_graph->operand_data_edges(runtime_node))
        {
            auto runtime_attr = runtime_graph->get_edge_attributes(runtime_edge);

            if (runtime_to_consteval_map.find(runtime_edge.producer_node_id) == runtime_to_consteval_map.end())
            {
                InputNode *runtime_operand =
                    dynamic_cast<InputNode *>(runtime_graph->node_by_id(runtime_edge.producer_node_id));
                TT_ASSERT(runtime_operand, "All operands of promoted nodes must be graph inputs");
                Node *consteval_operand = nullptr;
                if (ConstEvalGraph *nested_consteval_graph = runtime_operand->get_consteval_graph())
                    consteval_operand = graft(nested_consteval_graph->get_graph());
                else
                    consteval_operand = consteval_graph.add_node<Node>(runtime_operand->clone());

                runtime_to_consteval_map.insert({runtime_operand->id(), consteval_operand->id()});
                runtime_graph->remove_edge(runtime_edge);
                auto users = runtime_graph->user_edges(runtime_operand);
                if (users.empty())
                    runtime_graph->remove_node(runtime_operand);
            }

            Edge consteval_edge = Edge(
                runtime_to_consteval_map.at(runtime_edge.producer_node_id),
                runtime_edge.producer_output_port_id,
                runtime_to_consteval_map.at(runtime_edge.consumer_node_id),
                runtime_edge.consumer_input_port_id,
                runtime_edge.edge_type);

            consteval_graph.add_edge(consteval_edge);
            consteval_graph.get_edge_attributes(consteval_edge)->copy_from(*runtime_attr);
            runtime_attr->get_tms().clear();  // remove all operand runtime tms, they are consumed by consteval
        }
    }
    else if (dynamic_cast<graphlib::OpNode *>(consteval_node))
    {
        TT_ASSERT(consteval_output);
        // If there is no runtime graph then new consteval nodes are simply appended as the new output node
        Edge output_edge = consteval_graph.operand_data_edges(consteval_output).at(0);
        Edge new_edge(
            output_edge.producer_node_id,
            output_edge.producer_output_port_id,
            consteval_node->id(),
            0,
            EdgeType::kData);
        consteval_graph.add_edge(new_edge);
    }

    // Connect to the graph output
    if (consteval_output)
    {
        consteval_graph.remove_edge(consteval_graph.operand_data_edges(consteval_output).at(0));
    }
    else
    {
        consteval_output =
            consteval_graph.add_node<Node>(std::make_unique<OutputNode>(consteval_graph.name() + ".output"));
    }

    Edge consteval_edge(consteval_node->id(), 0, consteval_output->id(), 0, EdgeType::kData);
    consteval_graph.add_edge(consteval_edge);

    runtime_input->set_shape(consteval_node->shape());
    runtime_input->set_output_df(consteval_node->output_df());
    consteval_output->set_shape(consteval_node->shape());
    consteval_output->set_output_df(consteval_node->output_df());

    return runtime_graph ? graphlib::bypass_node(runtime_graph, runtime_node, true /*remove_node*/) : nullptr;
}

Node *ConstEvalGraph::graft(Graph *other)
{
    NodeId other_output_op_id = -1;
    std::unordered_map<NodeId, NodeId> node_id_map;
    std::vector<Node *> nodes = other->nodes();
    std::vector<Edge> edges = other->edges(EdgeType::kData);

    // Copy all nodes except for the output node
    for (Node *node : nodes)
    {
        if (node->node_type() == NodeType::kOutput)
        {
            TT_ASSERT(other_output_op_id == -1, "Only one output is supported for consteval graphs");
            other_output_op_id = other->data_operands(node)[0]->id();
            continue;
        }

        // If the graph being graft is from a common ancenstor nodes can overlap
        if (consteval_graph.has_node_with_name(node->name()))
        {
            node_id_map.insert({node->id(), consteval_graph.get_node_by_name(node->name())->id()});
            continue;
        }

        Node *new_node = consteval_graph.add_node<Node>(node->clone());
        node_id_map.insert({node->id(), new_node->id()});
    }

    // Copy all edges except for the output edge
    for (Edge const &edge : edges)
    {
        if (edge.producer_node_id == other_output_op_id)
            continue;

        Edge new_edge(
            node_id_map.at(edge.producer_node_id),
            edge.producer_output_port_id,
            node_id_map.at(edge.consumer_node_id),
            edge.consumer_input_port_id,
            edge.edge_type);
        consteval_graph.add_edge(new_edge);
        consteval_graph.copy_edge_attributes(edge, new_edge, other);
    }

    TT_ASSERT(other_output_op_id != -1);
    TT_ASSERT(node_id_map.find(other_output_op_id) != node_id_map.end());
    Node *output = consteval_graph.node_by_id(node_id_map.at(other_output_op_id));
    return output;
}

std::unique_ptr<ConstEvalGraph> ConstEvalGraph::clone(Node *new_runtime_input, const std::string& new_input_node_name)
{
    TT_ASSERT(new_runtime_input);
    int unique_id = Graph::generate_unique_graph_id();
    std::unique_ptr<ConstEvalGraph> cloned = std::make_unique<ConstEvalGraph>(
        consteval_graph.name() + "." + std::to_string(unique_id), new_runtime_input, false, unique_id);

    consteval_graph.clone(&cloned->consteval_graph);
    cloned->needs_autograd = needs_autograd;
    cloned->ran_autograd = ran_autograd;
    cloned->graph_updated_since_autograd = graph_updated_since_autograd;

    if (consteval_output)
        cloned->consteval_output = cloned->consteval_graph.get_node_by_name(consteval_output->name());
    // Map the old ids to cloned ones
    for (auto [runtime_node_id, consteval_node_id] : runtime_to_consteval_map)
    {
        Node* consteval_node = consteval_graph.node_by_id(consteval_node_id);
        std::string node_name = consteval_node->name();

        if (consteval_node->node_type() == NodeType::kInput and new_input_node_name != "")
        {
            std::string const &old_node_name = consteval_node->name();
            cloned->consteval_graph.update_node_name(cloned->consteval_graph.get_node_by_name(old_node_name), new_input_node_name);
            node_name = new_input_node_name;
        }
        cloned->runtime_to_consteval_map[runtime_node_id] = cloned->consteval_graph.get_node_by_name(node_name)->id();
    }
    return cloned;
}

void ConstEvalGraph::pad_output_to_buda_dims(std::string const &name_prefix)
{
    auto align_up_tile = [](auto d)
    {
        d -= 1;
        return d - (d % graphlib::Shape::BUDA_TILE_DIM) + graphlib::Shape::BUDA_TILE_DIM;
    };

    graphlib::Node *output = get_output();
    graphlib::Shape shape = output->shape();

    for (int dim : {-1, -2})
    {
        if (shape[dim] % graphlib::Shape::BUDA_TILE_DIM != 0)
        {
            graphlib::OpType pad_tile = {.op = "pad_tile", .attr = {dim, (int)shape[dim]}, .buda_attrs = {}};
            auto consteval_pad_tile = graphlib::create_node<graphlib::PyOpNode>(
                name_prefix + "_pad_tile_" + ((dim == -1) ? "c_" : "r_") + output->name(), pad_tile);
            shape[dim] = align_up_tile(shape[dim]);
            consteval_pad_tile->set_output_df(output->output_df());
            consteval_pad_tile->set_epoch_type(output->get_epoch_type());
            consteval_pad_tile->set_shape(shape);
            promote_node(std::move(consteval_pad_tile));
        }
    }
}

// void ConstEvalGraph::autograd()
// {
//     if (not needs_autograd)
//         return;

//     if (ran_autograd)
//     {
//         // Remove BW graph and build it again from scratch
//         auto bw_nodes = consteval_graph.nodes([](Node *n) { return n->get_epoch_type() == NodeEpochType::Backward; });
//         for (Node *bw_node : bw_nodes)
//         {
//             consteval_graph.remove_node(bw_node);
//         }
//     }

//     autograd2::autograd2_engine consteval_autograd_engine(&consteval_graph, autograd2::autograd_config{});
//     consteval_autograd_engine.run();

//     ran_autograd = true;
//     graph_updated_since_autograd = false;
// }

bool is_consteval_capable_input_type(Node *node)
{
    graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);
    return input and (input->is_parameter() or input->is_constant());
}

// bool is_consteval_capable_op(Graph *graph, Node *node, bool allow_forks)
// {
//     graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
//     if (not op)
//         return false;

//     std::vector<graphlib::Node *> operands = graph->data_operands(op);

//     if (not std::all_of(operands.begin(), operands.end(), is_consteval_capable_input_type))
//         return false;

//     bool disable_forks = not allow_forks;

//     auto requires_grad = [graph](graphlib::Node *n)
//     { return graph->enable_training() and n->as<graphlib::InputNode>()->requires_grad(); };

//     auto fork = [graph, disable_forks, requires_grad](graphlib::Node *n)
//     { return (requires_grad(n) or disable_forks) and (graph->data_users(n).size() > 1); };

//     auto bcast = [graph, requires_grad](graphlib::Node *n)
//     {
//         bool any_bcast = false;
//         for (auto e : graph->user_data_edges(n))
//         {
//             auto edge_attr = graph->get_edge_attributes(e);
//             any_bcast |= edge_attr->has_broadcast_dims();
//         }
//         return requires_grad(n) and any_bcast;
//     };

//     if (std::any_of(operands.begin(), operands.end(), fork))
//         return false;

//     if (std::any_of(operands.begin(), operands.end(), bcast))
//         return false;

//     if (std::none_of(operands.begin(), operands.end(), requires_grad))
//         return true;

//     // requires_grad = true
//     //   - if grad is required then we limit consteval to tm ops only
//     py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
//     py::function is_tm = eval_module.attr("is_tm");
//     return is_tm(op->op_name()).cast<bool>();
// }

// std::unique_ptr<Node> try_consteval_op(Graph *graph, Node *node)
// {
//     if (not is_consteval_capable_op(graph, node))
//         return nullptr;

//     std::vector<graphlib::Node *> operands = graph->data_operands(node);
//     graphlib::InputNode *input = operands[0]->as<graphlib::InputNode>();
//     auto consteval_graph = input->get_consteval_graph(true, true);
//     return consteval_graph->promote_node(graph, node);
// }

bool can_swap_operands(Graph *graph, Node *node)
{
        if (graph->data_operands(node).size() != 2)
            return false;
    if (node->node_type() == kBudaOp)
    {
        auto op = node->as<BudaOpNode>()->op_type().op;
        return ( (op != "sub") && (op != "matmul") );
    }

    if (node->node_type() == kPyOp)
    {
        auto op = node->as<PyOpNode>()->op_type().op;
        return ( (op != "sub") && (op != "matmul") );
    }
    return false;
}
void swap_operands(Graph *graph, Node *node)
{
    TT_ASSERT(can_swap_operands(graph, node));

    auto operand_edges = graph->operand_edges(node);

    for (Edge operand_edge : operand_edges)
    {
        Edge new_edge(operand_edge);
        new_edge.consumer_input_port_id = 1 - new_edge.consumer_input_port_id;
        graph->add_edge(new_edge);
        graph->copy_edge_attributes(operand_edge, new_edge);
        graph->remove_edge(operand_edge);
    }
}

}  // namespace graphlib

}  // namespace tt
