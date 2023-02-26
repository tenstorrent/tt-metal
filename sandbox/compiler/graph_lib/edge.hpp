#pragma once
#include <cassert>
#include <functional>

#include "common/utils.hpp"
#include "defines.hpp"

namespace tt {

namespace graphlib {

enum class EdgeType {
    kData = 0,
    kControl = 1,  // deprecated?
    kDataLoopback = 2,  // data edge with write into queue (param update)
    kAutogradFwdToBwd = 3,  // symbolic, fwd node -> bwd nodes created from it
    kAutogradFwdToGradient = 4,  // symbolic, fwd node -> propagated error (bwd node) (at max 1 edge from any fwd node)
    kAutogradFwdToOptimizer = 5,  // symbolic, fwd node (param) -> opt nodes created for its update
    kAutogradFwdToRecompute = 6,
    kControlLoop = 7,  // probably from subgraph's output to its input
    kAutogradOutputToLoss = 8,
    kAutogradInputToGradientOut = 9,
};
using EdgeUniqueId = std::tuple<NodeId, PortId, NodeId, PortId, EdgeType>;
using EdgeCreationId = std::int64_t;

struct Edge {
    static std::uint64_t last_assigned_edge_creation_id;

    // the edge_creation_id is a property used to preserve insertion
    // ordering of the user edges
    std::uint64_t edge_creation_id;

    NodeId producer_node_id;
    PortId producer_output_port_id;

    NodeId consumer_node_id;
    PortId consumer_input_port_id;

    EdgeType edge_type = EdgeType::kData;

    EdgeUniqueId unique_id() const {
        // intentionally exclude edge_creation_id from the tuple hash since 
        // it's unrelated to Edge identity
        return std::make_tuple(
            static_cast<NodeId>(this->producer_node_id),
            static_cast<PortId>(this->producer_output_port_id),
            static_cast<NodeId>(this->consumer_node_id),
            static_cast<PortId>(this->consumer_input_port_id),
            static_cast<EdgeType>(this->edge_type));
    }
    Edge(
        NodeId producer_node_id,
        PortId producer_output_port_id,
        NodeId consumer_node_id,
        PortId consumer_input_port_id,
        EdgeType edge_type);
    Edge(
        NodeId producer_node_id,
        PortId producer_output_port_id,
        NodeId consumer_node_id,
        PortId consumer_input_port_id,
        EdgeType edge_type,
        std::uint64_t edge_creation_id);
};
inline bool operator==(const Edge& lhs, const Edge& rhs) { return lhs.unique_id() == rhs.unique_id(); }
inline bool operator!=(const Edge& lhs, const Edge& rhs) { return not(lhs == rhs); }
inline bool operator<(const Edge& lhs, const Edge& rhs) { return lhs.unique_id() < rhs.unique_id(); }

inline std::string edge_type_to_string(const EdgeType& edge_type) {
    std::string retstring;
    switch (edge_type) {
        case EdgeType::kData: retstring = "Data"; break;
        case EdgeType::kControl: retstring = "Control"; break;
        case EdgeType::kDataLoopback: retstring = "DataLoopback"; break;
        case EdgeType::kAutogradFwdToBwd: retstring = "AutogradFwdToBwd"; break;
        case EdgeType::kAutogradFwdToGradient: retstring = "AutogradFwdToGradient"; break;
        case EdgeType::kAutogradFwdToOptimizer: retstring = "AutogradFwdToOptimizer"; break;
        case EdgeType::kAutogradFwdToRecompute: retstring = "AutogradFwdToRecompute"; break;
        case EdgeType::kAutogradOutputToLoss: retstring = "AutogradOutputToLoss"; break;
        case EdgeType::kAutogradInputToGradientOut: retstring = "AutogradInputToGradientOut"; break;
        case EdgeType::kControlLoop: retstring = "kControlLoop"; break;
        default: assert(false && "Unimplemented edge_type ostream");
    }
    return retstring;
}

}  // namespace graphlib
}  // namespace tt

namespace std {
template <>
struct hash<tt::graphlib::Edge> {
    std::size_t operator()(const tt::graphlib::Edge& edge) const {
        std::size_t seed = 0;
        // intentionally exclude edge_creation_id from the hash
        tt::utils::hash_combine(seed, static_cast<std::size_t>(edge.producer_node_id));
        tt::utils::hash_combine(seed, static_cast<std::size_t>(edge.producer_output_port_id));
        tt::utils::hash_combine(seed, static_cast<std::size_t>(edge.consumer_node_id));
        tt::utils::hash_combine(seed, static_cast<std::size_t>(edge.consumer_input_port_id));
        tt::utils::hash_combine(seed, static_cast<std::size_t>(edge.edge_type));
        return seed;
    }
};



}  // namespace std
