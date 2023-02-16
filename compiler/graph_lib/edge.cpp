#include "graph_lib/edge.hpp"

std::uint64_t tt::graphlib::Edge::last_assigned_edge_creation_id = 0;

namespace tt {
namespace graphlib {

Edge::Edge(
    NodeId _producer_node_id,
    PortId _producer_output_port_id,
    NodeId _consumer_node_id,
    PortId _consumer_input_port_id,
    EdgeType _edge_type) :
    edge_creation_id(Edge::last_assigned_edge_creation_id++),
    producer_node_id(_producer_node_id),
    producer_output_port_id(_producer_output_port_id),
    consumer_node_id(_consumer_node_id),
    consumer_input_port_id(_consumer_input_port_id),
    edge_type(_edge_type) {}

Edge::Edge(
    NodeId _producer_node_id,
    PortId _producer_output_port_id,
    NodeId _consumer_node_id,
    PortId _consumer_input_port_id,
    EdgeType _edge_type,
    std::uint64_t _edge_creation_id) :
    edge_creation_id(_edge_creation_id),
    producer_node_id(_producer_node_id),
    producer_output_port_id(_producer_output_port_id),
    consumer_node_id(_consumer_node_id),
    consumer_input_port_id(_consumer_input_port_id),
    edge_type(_edge_type) {}
} // namespace graphlib
} // namespace tt
